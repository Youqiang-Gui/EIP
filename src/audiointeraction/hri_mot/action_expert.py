from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HriRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight.float()).to(dtype=dtype)


@dataclass
class HriActionExpertConfig:
    num_layers: int
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    head_dim: int = 128
    rope_n_elem: Optional[int] = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    dropout: float = 0.0
    parallel_residual: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rope_n_elem is None:
            self.rope_n_elem = self.head_dim
        if self.rope_n_elem > self.head_dim:
            raise ValueError("rope_n_elem cannot exceed head_dim")


class HriActionMLP(nn.Module):
    def __init__(self, config: HriActionExpertConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HriActionExpertLayer(nn.Module):
    def __init__(self, config: HriActionExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.query_size = config.num_attention_heads * config.head_dim
        self.key_value_size = config.num_key_value_heads * config.head_dim
        self.qkv_size = self.query_size + 2 * self.key_value_size

        self.norm_1 = HriRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, self.qkv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.query_size, config.hidden_size, bias=config.attention_bias)
        self.post_attention_norm = nn.Identity()
        self.norm_2 = HriRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HriActionMLP(config)
        self.post_mlp_norm = nn.Identity()

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_normed = self.norm_1(hidden_states)
        batch_size, seq_len, _ = x_normed.shape
        qkv = self.qkv(x_normed)
        if qkv.shape[-1] != self.qkv_size:
            raise ValueError(f"Expected fused qkv size {self.qkv_size}, got {qkv.shape[-1]}")
        q, k, v = qkv.split(
            (self.query_size, self.key_value_size, self.key_value_size), dim=-1
        )
        q = q.view(
            batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim
        )
        k = k.view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )
        v = v.view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )
        return x_normed, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def apply_attention_output(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        x_normed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_output = self.o_proj(attention_output)
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            mlp_input = self.norm_2(hidden_states)
            hidden_states = hidden_states + attention_output
        else:
            hidden_states = hidden_states + attention_output
            mlp_input = self.norm_2(hidden_states)

        return hidden_states + self.post_mlp_norm(self.mlp(mlp_input))

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_normed, q, k, v = self.project_qkv(hidden_states)
        if self.config.num_key_value_heads != self.config.num_attention_heads:
            repeat = self.config.num_attention_heads // self.config.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.config.head_dim),
            is_causal=attention_mask is None,
        )
        batch_size, _, seq_len, _ = attn_output.shape
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.config.num_attention_heads * self.config.head_dim
        )
        return self.apply_attention_output(hidden_states, attn_output, x_normed)


class HriActionExpertStack(nn.Module):
    def __init__(self, config: HriActionExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(HriActionExpertLayer(config) for _ in range(config.num_layers))

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class _StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_tokens: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.vector_proj = nn.Linear(input_dim, hidden_size * num_tokens)
        self.sequence_proj = nn.Linear(input_dim, hidden_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 2:
            batch_size = state.shape[0]
            return self.vector_proj(state).view(batch_size, self.num_tokens, self.hidden_size)
        if state.dim() == 3:
            return self.sequence_proj(state)
        raise ValueError(f"state must have shape [B, D] or [B, T, D], got {tuple(state.shape)}")


class PartnerStateEncoder(_StateEncoder):
    pass


class RobotStateEncoder(_StateEncoder):
    pass


class RobotStateAdapter(nn.Module):
    def __init__(
        self,
        robot_state_dim: int,
        hidden_size: int,
        adapter_hidden_size: int = 512,
        num_steps: int = 12,
        use_time_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.mlp = nn.Sequential(
            nn.Linear(robot_state_dim, adapter_hidden_size),
            nn.SiLU(),
            nn.Linear(adapter_hidden_size, hidden_size),
        )
        self.time_embedding = (
            nn.Parameter(torch.zeros(num_steps, hidden_size)) if use_time_embedding else None
        )

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        if robot_state.dim() != 3:
            raise ValueError(
                f"robot_state must have shape [B, {self.num_steps}, robot_state_dim], got {tuple(robot_state.shape)}"
            )
        if robot_state.shape[1] != self.num_steps:
            raise ValueError(
                f"robot_state must contain {self.num_steps} time steps, got {robot_state.shape[1]}"
            )
        if robot_state.shape[2] != self.robot_state_dim:
            raise ValueError(
                f"robot_state dim {robot_state.shape[2]} does not match configured dim {self.robot_state_dim}"
            )
        tokens = self.mlp(robot_state)
        if self.time_embedding is not None:
            tokens = tokens + self.time_embedding.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        return tokens


class ResponseTimeEncoder(nn.Module):
    """Encode chunks elapsed since response onset for the action-only suffix.

    ``elapsed_chunks`` uses ``-1`` outside target speech and ``0, 1, ...``
    from the first speaking interval onward.  Silent entries produce an exact
    zero vector so the existing binary speak embedding remains authoritative.
    """

    def __init__(self, hidden_size: int, *, max_period: float = 10_000.0) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than 1")
        self.hidden_size = int(hidden_size)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, elapsed_chunks: torch.Tensor) -> torch.Tensor:
        if elapsed_chunks.dim() != 2:
            raise ValueError(
                "elapsed_chunks must have shape [B, chunks], "
                f"got {tuple(elapsed_chunks.shape)}"
            )
        valid = elapsed_chunks.ge(0)
        elapsed = elapsed_chunks.clamp_min(0).to(dtype=torch.float32)
        half = self.hidden_size // 2
        if half > 0:
            exponent = torch.arange(
                half, device=elapsed.device, dtype=elapsed.dtype
            ) / float(max(1, half - 1))
            frequencies = torch.exp(-math.log(self.max_period) * exponent)
            phases = elapsed.unsqueeze(-1) * frequencies
            embedding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        else:
            embedding = elapsed.unsqueeze(-1).new_zeros(
                (*elapsed.shape, 0)
            )
        if embedding.shape[-1] < self.hidden_size:
            embedding = F.pad(embedding, (0, self.hidden_size - embedding.shape[-1]))
        # Build sinusoidal features in fp32 for numerical stability, then match
        # the projection weights.  The wrapper aligns trainable modules to the
        # frozen backbone dtype (normally bf16), while elapsed chunk ids are
        # intentionally expanded in fp32 above.
        projection_weight = self.mlp[0].weight
        embedding = embedding.to(
            device=projection_weight.device,
            dtype=projection_weight.dtype,
        )
        encoded = self.mlp(embedding)
        return encoded * valid.unsqueeze(-1).to(dtype=encoded.dtype)


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor,
    dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    timestep = timestep.view(-1).to(dtype=torch.float32)
    half = dim // 2
    if half <= 0:
        return timestep.new_zeros((timestep.shape[0], dim))
    periods = torch.exp(
        torch.linspace(
            math.log(min_period),
            math.log(max_period),
            half,
            device=timestep.device,
            dtype=timestep.dtype,
        )
    )
    phases = 2.0 * math.pi * timestep[:, None] / periods[None, :]
    emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class ActionTimeEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        *,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.min_period = min_period
        self.max_period = max_period
        self.action_in_proj = nn.Linear(action_dim, hidden_size)
        self.action_time_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if noisy_action.dim() != 3:
            raise ValueError(
                f"noisy_action must have shape [B, action_horizon, action_dim], got {tuple(noisy_action.shape)}"
            )
        action_emb = self.action_in_proj(noisy_action)
        batch_size, action_horizon, _ = action_emb.shape
        timestep = timestep.to(device=action_emb.device)
        if timestep.dim() == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError(
                    f"timestep must have shape [{batch_size}] or [{batch_size}, {action_horizon}], "
                    f"got {tuple(timestep.shape)}"
                )
            time_emb = sinusoidal_timestep_embedding(
                timestep,
                self.hidden_size,
                min_period=self.min_period,
                max_period=self.max_period,
            ).to(device=action_emb.device, dtype=action_emb.dtype)
            time_emb = time_emb[:, None, :].expand_as(action_emb)
        elif timestep.dim() == 2:
            if tuple(timestep.shape) != (batch_size, action_horizon):
                raise ValueError(
                    f"timestep must have shape [{batch_size}] or [{batch_size}, {action_horizon}], "
                    f"got {tuple(timestep.shape)}"
                )
            time_emb = sinusoidal_timestep_embedding(
                timestep.reshape(-1),
                self.hidden_size,
                min_period=self.min_period,
                max_period=self.max_period,
            ).to(device=action_emb.device, dtype=action_emb.dtype)
            time_emb = time_emb.reshape(batch_size, action_horizon, self.hidden_size)
        else:
            raise ValueError(
                f"timestep must have shape [{batch_size}] or [{batch_size}, {action_horizon}], "
                f"got {tuple(timestep.shape)}"
            )
        return self.action_time_mlp(torch.cat([action_emb, time_emb], dim=-1))


class NoisyActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_size)

    def forward(self, noisy_action: torch.Tensor) -> torch.Tensor:
        if noisy_action.dim() != 3:
            raise ValueError(
                f"noisy_action must have shape [B, action_horizon, action_dim], got {tuple(noisy_action.shape)}"
            )
        return self.proj(noisy_action)


class FlowTimestepEncoder(nn.Module):
    def __init__(self, hidden_size: int, max_period: int = 10000) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(tau, self.hidden_size, max_period=float(self.max_period))
        return self.mlp(emb.to(device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype))


class ActionVelocityHead(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = HriRMSNorm(hidden_size, eps=norm_eps)
        self.proj = nn.Linear(hidden_size, action_dim)

    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(action_hidden_states))


class SpeakPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = HriRMSNorm(hidden_size, eps=norm_eps)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, speak_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(speak_hidden_states)).squeeze(-1)
