import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audiointeraction.hri_mot.action_expert import (
    ActionTimeEncoder,
    ActionVelocityHead,
    HriActionExpertConfig,
    HriActionExpertLayer,
    HriActionExpertStack,
    ResponseTimeEncoder,
    RobotStateAdapter,
    SpeakPredictionHead,
)
from src.audiointeraction.hri_mot.emotion_conditioning import (
    NUM_EMOTION_CONDITIONS,
)
from src.audiointeraction.hri_mot.data_config import (
    CONTEXT_TOKEN_TARGET_CONTROL,
    CONTEXT_TOKEN_TARGET_TEXT,
)
from src.audiointeraction.hri_mot.decoder_discovery import (
    INPUT_NORM_NAMES,
    find_attention_module,
    get_module_by_path,
    select_decoder_candidate,
)


def _first_module(module: nn.Module, names: Iterable[str]) -> Optional[nn.Module]:
    for name in names:
        child = getattr(module, name, None)
        if isinstance(child, nn.Module):
            return child
    return None


def _module_attr(module: nn.Module, names: Iterable[str], default=None):
    config = getattr(module, "config", None)
    for owner in (module, config):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return value
    return default


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x
    dtype = x.dtype
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        cos = cos.view(cos.shape[0], *([1] * dims_diff), cos.shape[1], cos.shape[2])
        sin = sin.view(sin.shape[0], *([1] * dims_diff), sin.shape[1], sin.shape[2])
    return (x * cos + rotated * sin).to(dtype=dtype)


def _apply_rope_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    rope_n_elem: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None or rope_n_elem <= 0:
        return q, k
    rope_n_elem = min(rope_n_elem, q.shape[-1], k.shape[-1], cos.shape[-1], sin.shape[-1])
    q_rope = _apply_rope(q[..., :rope_n_elem], cos[..., :rope_n_elem], sin[..., :rope_n_elem])
    k_rope = _apply_rope(k[..., :rope_n_elem], cos[..., :rope_n_elem], sin[..., :rope_n_elem])
    return torch.cat((q_rope, q[..., rope_n_elem:]), dim=-1), torch.cat(
        (k_rope, k[..., rope_n_elem:]), dim=-1
    )


def _apply_rope_to_k(
    k: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    rope_n_elem: int,
) -> torch.Tensor:
    if cos is None or sin is None or rope_n_elem <= 0:
        return k
    rope_n_elem = min(rope_n_elem, k.shape[-1], cos.shape[-1], sin.shape[-1])
    k_rope = _apply_rope(k[..., :rope_n_elem], cos[..., :rope_n_elem], sin[..., :rope_n_elem])
    return torch.cat((k_rope, k[..., rope_n_elem:]), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, target_heads: int) -> torch.Tensor:
    current_heads = hidden_states.shape[1]
    if current_heads == target_heads:
        return hidden_states
    if target_heads % current_heads != 0:
        raise ValueError(f"Cannot repeat {current_heads} kv heads to {target_heads} attention heads")
    return hidden_states.repeat_interleave(target_heads // current_heads, dim=1)


def _batched_index_select(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    if index.dim() == 1:
        return torch.index_select(tensor, dim, index)

    *batch_shape, index_size = index.shape
    selected = torch.index_select(tensor, dim, index.reshape(-1))
    selected = selected.view(*tensor.shape[:dim], -1, index_size, *tensor.shape[dim + 1 :])
    if dim > 0:
        dims = [dim] + list(range(selected.dim()))
        del dims[dim + 1]
        selected = selected.permute(dims)
    return selected.view(*batch_shape, *selected.shape[1:])


def _select_position_rows(rows: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    if input_pos.dim() == 1:
        return torch.index_select(rows, 0, input_pos).unsqueeze(0)
    flat = torch.index_select(rows, 0, input_pos.reshape(-1))
    return flat.view(*input_pos.shape, rows.shape[-1])


_SUFFIX_STATE = 0
_SUFFIX_SPEAK = 1
_SUFFIX_ACTION = 2


def _gather_sequence(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() != 3 or indices.dim() != 2:
        raise ValueError("hidden_states/indices must have shapes [B, T, H] and [B, T]")
    return hidden_states.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
    )


def _interleaved_suffix_layout(
    *,
    batch_size: int,
    state_len: int,
    action_len: int,
    device: torch.device,
    state_chunk_ids: Optional[torch.Tensor],
    action_chunk_ids: Optional[torch.Tensor],
    speak_len: int = 0,
    speak_chunk_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return chunk-major [state_c, speak_c, action_c] suffix metadata.

    ``permutation`` maps source layout ``[all state, all speak, all action]``
    to the per-chunk layout. ``restore_indices`` maps an interleaved hidden
    sequence back to source layout so public action outputs stay chunk-major.
    """

    if state_chunk_ids is None:
        state_chunks = torch.zeros(batch_size, state_len, device=device, dtype=torch.long)
    else:
        state_chunks = state_chunk_ids.to(device=device, dtype=torch.long)
        if tuple(state_chunks.shape) != (batch_size, state_len):
            raise ValueError(
                f"state_chunk_ids must have shape {(batch_size, state_len)}, "
                f"got {tuple(state_chunks.shape)}"
            )

    if action_chunk_ids is None:
        action_chunks = torch.zeros(batch_size, action_len, device=device, dtype=torch.long)
    else:
        action_chunks = action_chunk_ids.to(device=device, dtype=torch.long)
        if tuple(action_chunks.shape) != (batch_size, action_len):
            raise ValueError(
                f"action_chunk_ids must have shape {(batch_size, action_len)}, "
                f"got {tuple(action_chunks.shape)}"
            )

    if speak_chunk_ids is None:
        speak_chunks = torch.zeros(batch_size, speak_len, device=device, dtype=torch.long)
    else:
        speak_chunks = speak_chunk_ids.to(device=device, dtype=torch.long)
        if tuple(speak_chunks.shape) != (batch_size, speak_len):
            raise ValueError(
                f"speak_chunk_ids must have shape {(batch_size, speak_len)}, "
                f"got {tuple(speak_chunks.shape)}"
            )

    source_chunk_ids = torch.cat([state_chunks, speak_chunks, action_chunks], dim=1)
    source_token_types = torch.cat(
        [
            torch.full_like(state_chunks, _SUFFIX_STATE),
            torch.full_like(speak_chunks, _SUFFIX_SPEAK),
            torch.full_like(action_chunks, _SUFFIX_ACTION),
        ],
        dim=1,
    )
    # Stable sorting preserves the original time-step order inside each state
    # or action block while producing [S_0, Speak_0, A_0, ...].
    sort_key = source_chunk_ids * 3 + source_token_types
    permutation = torch.argsort(sort_key, dim=1, stable=True)
    restore_indices = torch.argsort(permutation, dim=1)
    suffix_chunk_ids = source_chunk_ids.gather(1, permutation)
    suffix_token_types = source_token_types.gather(1, permutation)
    return suffix_chunk_ids, suffix_token_types, permutation, restore_indices


def _local_suffix_position_ids(
    context_allowed: torch.Tensor,
    suffix_chunk_ids: torch.Tensor,
) -> torch.Tensor:
    """Place each [state_c, speak_c, action_c] group after visible context c."""

    if context_allowed.dim() != 3:
        raise ValueError("context_allowed must have shape [B, T_suffix, T_context]")
    if tuple(context_allowed.shape[:2]) != tuple(suffix_chunk_ids.shape):
        raise ValueError(
            "context_allowed suffix dimensions must match suffix_chunk_ids: "
            f"{tuple(context_allowed.shape[:2])} vs {tuple(suffix_chunk_ids.shape)}"
        )

    batch_size, suffix_len, context_len = context_allowed.shape
    device = suffix_chunk_ids.device
    if context_len:
        context_positions = torch.arange(context_len, device=device).view(1, 1, context_len)
        last_visible = torch.where(
            context_allowed,
            context_positions,
            torch.full_like(context_positions, -1),
        ).amax(dim=2)
        group_anchor = last_visible + 1
    else:
        group_anchor = torch.zeros(batch_size, suffix_len, device=device, dtype=torch.long)

    if suffix_len == 0:
        return group_anchor
    token_positions = torch.arange(suffix_len, device=device).unsqueeze(0).expand(batch_size, -1)
    group_boundary = torch.ones(batch_size, suffix_len, device=device, dtype=torch.bool)
    if suffix_len > 1:
        group_boundary[:, 1:] = suffix_chunk_ids[:, 1:] != suffix_chunk_ids[:, :-1]
    group_starts = torch.where(group_boundary, token_positions, torch.zeros_like(token_positions))
    group_starts = torch.cummax(group_starts, dim=1).values
    within_group_offset = token_positions - group_starts
    return group_anchor + within_group_offset


class HriMoTWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        robot_state_dim: int = 20,
        action_dim: int = 20,
        decoder_path: Optional[str] = None,
        action_hidden_size: int = 768,
        action_intermediate_size: int = 3072,
        robot_state_steps: int = 12,
        robot_adapter_hidden_size: int = 512,
        use_robot_time_embedding: bool = True,
        speak_conditioning: str = "none",
        use_response_time_embedding: bool = False,
        use_emotion_conditioning: bool = False,
        emotion_film_layers: int = 4,
        action_config: Optional[HriActionExpertConfig] = None,
    ) -> None:
        super().__init__()

        if speak_conditioning not in ("none", "predicted", "teacher_forced"):
            raise ValueError(
                "speak_conditioning must be one of none/predicted/teacher_forced, "
                f"got {speak_conditioning!r}"
            )
        self.speak_conditioning = speak_conditioning
        self.use_response_time_embedding = bool(use_response_time_embedding)
        self.use_emotion_conditioning = bool(
            use_emotion_conditioning
        )
        self.emotion_film_layers = (
            int(emotion_film_layers)
            if self.use_emotion_conditioning
            else 0
        )
        if (
            self.use_emotion_conditioning
            and speak_conditioning != "teacher_forced"
        ):
            raise ValueError(
                "emotion conditioning requires teacher_forced speak conditioning"
            )
        if self.use_response_time_embedding and speak_conditioning != "teacher_forced":
            raise ValueError(
                "response time embedding currently requires teacher_forced "
                "speak conditioning"
            )

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        self.forward_module = getattr(base_model, "_forward_module", base_model)

        self.decoder_candidate = select_decoder_candidate(base_model, decoder_path=decoder_path)
        self.decoder_path = self.decoder_candidate.path
        self.decoder_layers = self.decoder_candidate.module

        self.text_hidden_size = self._infer_text_hidden_size()
        self.num_attention_heads = self._infer_num_attention_heads()
        self.num_key_value_heads = self._infer_num_key_value_heads()
        self.head_dim = self._infer_head_dim()
        self.rope_n_elem = self._infer_rope_n_elem()
        self.token_embedding = self._find_token_embedding()

        first_layer = next(iter(self.decoder_layers))
        first_attention = find_attention_module(first_layer)
        first_qkv = (
            getattr(first_attention, "qkv", None) or getattr(first_attention, "c_attn", None)
            if first_attention is not None
            else None
        )
        if action_config is None:
            action_config = HriActionExpertConfig(
                num_layers=len(self.decoder_layers),
                hidden_size=action_hidden_size,
                intermediate_size=action_intermediate_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope_n_elem=self.rope_n_elem,
                rms_norm_eps=float(_module_attr(first_layer, ("norm_eps",), 1e-6)),
                attention_bias=bool(first_qkv is not None and getattr(first_qkv, "bias", None) is not None),
                parallel_residual=bool(_module_attr(first_layer, ("parallel_residual",), False)),
            )
        if action_config.num_layers != len(self.decoder_layers):
            raise ValueError(
                f"Action expert depth {action_config.num_layers} must match decoder depth {len(self.decoder_layers)}"
            )
        if self.use_emotion_conditioning and not (
            1 <= self.emotion_film_layers <= action_config.num_layers
        ):
            raise ValueError(
                "emotion_film_layers must be between 1 and "
                f"{action_config.num_layers}"
            )

        self.robot_state_adapter = RobotStateAdapter(
            robot_state_dim=robot_state_dim,
            hidden_size=action_config.hidden_size,
            adapter_hidden_size=robot_adapter_hidden_size,
            num_steps=robot_state_steps,
            use_time_embedding=use_robot_time_embedding,
        )
        self.action_time_encoder = ActionTimeEncoder(action_dim, action_config.hidden_size)
        self.action_expert = HriActionExpertStack(action_config)
        self.action_velocity_head = ActionVelocityHead(
            action_config.hidden_size, action_dim, norm_eps=action_config.rms_norm_eps
        )
        if self.speak_conditioning == "predicted":
            self.speak_token_embedding: Optional[nn.Embedding] = nn.Embedding(
                1, action_config.hidden_size
            )
            self.speak_prediction_head: Optional[nn.Module] = SpeakPredictionHead(
                action_config.hidden_size, norm_eps=action_config.rms_norm_eps
            )
        elif self.speak_conditioning == "teacher_forced":
            self.speak_token_embedding = nn.Embedding(2, action_config.hidden_size)
            self.speak_prediction_head = None
        else:
            self.speak_token_embedding = None
            self.speak_prediction_head = None
        self.response_time_encoder: Optional[nn.Module] = (
            ResponseTimeEncoder(action_config.hidden_size)
            if self.use_response_time_embedding
            else None
        )
        self.emotion_embedding: Optional[nn.Embedding] = (
            nn.Embedding(
                NUM_EMOTION_CONDITIONS,
                action_config.hidden_size,
                padding_idx=0,
            )
            if self.use_emotion_conditioning
            else None
        )
        if self.emotion_embedding is not None:
            # Emotion fine-tuning starts from a checkpoint without
            # this branch.  A zero residual keeps step zero exactly equivalent
            # to that checkpoint; the six emotions then learn their own offsets.
            nn.init.zeros_(self.emotion_embedding.weight)
        film_start = action_config.num_layers - self.emotion_film_layers
        self.emotion_layer_film = nn.ModuleDict(
            {
                str(layer_idx): nn.Embedding(
                    NUM_EMOTION_CONDITIONS,
                    2 * action_config.hidden_size,
                    padding_idx=0,
                )
                for layer_idx in range(film_start, action_config.num_layers)
            }
            if self.use_emotion_conditioning
            else {}
        )
        for film in self.emotion_layer_film.values():
            # Each selected block has independent [gamma, beta] rows. Zero
            # initialization makes all six emotions exactly reproduce the
            # frozen source Action Expert before fine-tuning.
            nn.init.zeros_(film.weight)
        self._align_trainable_modules_to_base()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _first_decoder_layer(self) -> nn.Module:
        return next(iter(self.decoder_layers))

    def _find_token_embedding(self) -> nn.Module:
        transformer = getattr(self.forward_module, "transformer", None)
        if transformer is not None and hasattr(transformer, "wte"):
            return transformer.wte
        try:
            return get_module_by_path(self.base_model, "_forward_module.transformer.wte")
        except Exception as exc:
            raise ValueError("Could not find original token embedding module") from exc

    def _align_trainable_modules_to_base(self) -> None:
        try:
            base_param = next(self.base_model.parameters())
        except StopIteration:
            return
        modules: List[nn.Module] = [
            self.robot_state_adapter,
            self.action_time_encoder,
            self.action_expert,
            self.action_velocity_head,
        ]
        if self.speak_token_embedding is not None:
            modules.append(self.speak_token_embedding)
        if self.speak_prediction_head is not None:
            modules.append(self.speak_prediction_head)
        if self.response_time_encoder is not None:
            modules.append(self.response_time_encoder)
        for module in modules:
            module.to(device=base_param.device, dtype=base_param.dtype)
        # Keep the newly learned emotion tables and their Adam moments in FP32.
        # At the warm-start LR (1e-5), BF16 rounding otherwise swallows almost
        # every embedding update.  Its forward output is cast to the action
        # expert dtype below.
        if self.emotion_embedding is not None:
            self.emotion_embedding.to(
                device=base_param.device,
                dtype=torch.float32,
            )
        if self.emotion_layer_film:
            self.emotion_layer_film.to(
                device=base_param.device,
                dtype=torch.float32,
            )

    @staticmethod
    def _align_chunk_conditions_to_actions(
        condition_labels: torch.Tensor,
        condition_chunk_ids: torch.Tensor,
        action_chunk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast one chunk condition to every action token in that chunk."""

        if condition_labels.dim() != 2 or condition_chunk_ids.dim() != 2:
            raise ValueError("condition labels/chunk ids must have shape [B, C]")
        if action_chunk_ids.dim() != 2:
            raise ValueError("action_chunk_ids must have shape [B, A]")
        if condition_labels.shape != condition_chunk_ids.shape:
            raise ValueError("condition labels and chunk ids must have matching shapes")
        if condition_labels.shape[0] != action_chunk_ids.shape[0]:
            raise ValueError("condition/action chunk ids must have matching batches")

        matches = action_chunk_ids.unsqueeze(-1).eq(
            condition_chunk_ids.unsqueeze(1)
        )
        match_count = matches.sum(dim=-1)
        if not torch.all(match_count == 1):
            bad = int((match_count != 1).sum().item())
            raise ValueError(
                "Each action token must match exactly one emotion "
                f"chunk; found {bad} unmatched or ambiguous tokens"
            )
        condition_indices = matches.to(dtype=torch.long).argmax(dim=-1)
        return condition_labels.gather(1, condition_indices)

    def _apply_emotion_layer_film(
        self,
        h_suffix: torch.Tensor,
        *,
        layer_idx: int,
        state_len: int,
        speak_len: int,
        restore_indices: torch.Tensor,
        action_condition_ids: torch.Tensor,
    ) -> torch.Tensor:
        """FiLM-modulate action tokens after one selected Action Expert layer."""

        film_key = str(int(layer_idx))
        if film_key not in self.emotion_layer_film:
            return h_suffix
        action_len = int(action_condition_ids.shape[1])
        if int(h_suffix.shape[1]) != state_len + speak_len + action_len:
            raise ValueError(
                "suffix length does not match state/speak/action layout"
            )
        restored_suffix = _gather_sequence(h_suffix, restore_indices)
        action_offset = state_len + speak_len
        h_action = restored_suffix[:, action_offset:]
        if tuple(h_action.shape[:2]) != tuple(action_condition_ids.shape):
            raise ValueError(
                "action condition ids do not match restored action tokens"
            )
        film_module = self.emotion_layer_film[film_key]
        film = film_module(
            action_condition_ids.to(
                device=film_module.weight.device,
                dtype=torch.long,
            )
        ).to(dtype=h_action.dtype)
        gamma, beta = film.chunk(2, dim=-1)
        adapted_action = h_action * (1.0 + gamma) + beta
        restored_suffix = torch.cat(
            [restored_suffix[:, :action_offset], adapted_action],
            dim=1,
        )
        # restore_indices maps interleaved -> source, so its argsort maps the
        # adapted source layout back to the interleaved layer layout.
        permutation = torch.argsort(restore_indices, dim=1)
        return _gather_sequence(restored_suffix, permutation)

    def _infer_text_hidden_size(self) -> int:
        if self.decoder_candidate.hidden_size is not None:
            return int(self.decoder_candidate.hidden_size)
        first_layer = self._first_decoder_layer()
        norm = _first_module(first_layer, INPUT_NORM_NAMES)
        if norm is not None and hasattr(norm, "weight"):
            return int(norm.weight.numel())
        raise ValueError("Could not infer original decoder hidden size")

    def _infer_num_attention_heads(self) -> int:
        if self.decoder_candidate.num_attention_heads is not None:
            return int(self.decoder_candidate.num_attention_heads)
        value = _module_attr(self.base_model, ("n_head", "num_attention_heads"))
        if value is not None:
            return int(value)
        raise ValueError("Could not infer original decoder attention head count")

    def _infer_num_key_value_heads(self) -> int:
        if self.decoder_candidate.num_key_value_heads is not None:
            return int(self.decoder_candidate.num_key_value_heads)
        value = _module_attr(self.base_model, ("n_query_groups", "num_key_value_heads"))
        return int(value) if value is not None else self.num_attention_heads

    def _infer_head_dim(self) -> int:
        if self.decoder_candidate.head_dim is not None:
            return int(self.decoder_candidate.head_dim)
        if self.text_hidden_size % self.num_attention_heads != 0:
            raise ValueError("Could not infer head_dim from hidden size and attention heads")
        return self.text_hidden_size // self.num_attention_heads

    def _infer_rope_n_elem(self) -> int:
        value = _module_attr(self.forward_module, ("rope_n_elem",), None)
        if value is not None:
            return int(value)
        config = getattr(self.forward_module, "config", None)
        value = getattr(config, "rope_n_elem", None) if config is not None else None
        return int(value) if value is not None else self.head_dim

    def _rope_cache(
        self, seq_len: int, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if seq_len <= 0 or self.rope_n_elem <= 0:
            return None, None
        cos = getattr(self.forward_module, "cos", None)
        sin = getattr(self.forward_module, "sin", None)
        if cos is None or sin is None or cos.shape[0] < seq_len:
            rope_cache = getattr(self.forward_module, "rope_cache", None)
            if rope_cache is None:
                return None, None
            cos, sin = rope_cache(device=device)
        cos = cos[:seq_len, : self.rope_n_elem].to(device=device).unsqueeze(0)
        sin = sin[:seq_len, : self.rope_n_elem].to(device=device).unsqueeze(0)
        return cos, sin

    def _split_fused_qkv_like_backbone(
        self, projected: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        expected_qkv_size = query_size + 2 * key_value_size
        if projected.shape[-1] != expected_qkv_size:
            raise ValueError(f"Expected fused qkv size {expected_qkv_size}, got {projected.shape[-1]}")
        return projected.split((query_size, key_value_size, key_value_size), dim=-1)

    def _project_context_kv(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention = find_attention_module(layer)
        if attention is None:
            raise ValueError(f"Layer {layer.__class__.__name__} has no attention module")
        norm = _first_module(layer, INPUT_NORM_NAMES)
        if norm is None:
            raise ValueError(f"Layer {layer.__class__.__name__} has no input norm")
        qkv = getattr(attention, "qkv", None) or getattr(attention, "c_attn", None)
        if qkv is None:
            raise ValueError(f"Layer {layer.__class__.__name__} attention has no fused qkv projection")

        x_normed = norm(hidden_states)
        batch_size, seq_len, _ = x_normed.shape
        _, k, v = self._split_fused_qkv_like_backbone(qkv(x_normed))
        norm_k = getattr(attention, "norm_k", None) or getattr(attention, "k_norm", None)
        if norm_k is not None:
            k = norm_k(k)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        k = _apply_rope_to_k(k, cos, sin, self.rope_n_elem)
        return k, v

    def _project_context_qkv(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = find_attention_module(layer)
        if attention is None:
            raise ValueError(f"Layer {layer.__class__.__name__} has no attention module")
        norm = _first_module(layer, INPUT_NORM_NAMES)
        if norm is None:
            raise ValueError(f"Layer {layer.__class__.__name__} has no input norm")
        qkv = getattr(attention, "qkv", None) or getattr(attention, "c_attn", None)
        if qkv is None:
            raise ValueError(f"Layer {layer.__class__.__name__} attention has no fused qkv projection")

        x_normed = norm(hidden_states)
        batch_size, seq_len, _ = x_normed.shape
        q, k, v = self._split_fused_qkv_like_backbone(qkv(x_normed))
        norm_q = getattr(attention, "norm_q", None) or getattr(attention, "q_norm", None)
        norm_k = getattr(attention, "norm_k", None) or getattr(attention, "k_norm", None)
        if norm_q is not None:
            q = norm_q(q)
        if norm_k is not None:
            k = norm_k(k)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope_prefix(q, k, cos, sin, self.rope_n_elem)
        return x_normed, q, k, v

    def _attention_output_projection(self, layer: nn.Module) -> nn.Module:
        attention = find_attention_module(layer)
        if attention is None:
            raise ValueError(f"Layer {layer.__class__.__name__} has no attention module")
        for name in ("proj", "o_proj", "c_proj"):
            projection = getattr(attention, name, None)
            if isinstance(projection, nn.Module):
                return projection
        raise ValueError(f"Layer {layer.__class__.__name__} attention has no output projection")

    def _apply_context_output(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        x_normed: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self._attention_output_projection(layer)(attention_output)
        post_attention_norm = getattr(layer, "post_attention_norm", None)
        if isinstance(post_attention_norm, nn.Module):
            attention_output = post_attention_norm(attention_output)

        parallel_residual = bool(_module_attr(layer, ("parallel_residual",), False))
        shared_attention_norm = bool(_module_attr(layer, ("shared_attention_norm",), False))
        norm_2 = _first_module(layer, ("post_attention_layernorm", "norm_2", "ln_2", "final_layernorm"))
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
        if not isinstance(mlp, nn.Module):
            raise ValueError(f"Layer {layer.__class__.__name__} has no MLP module")
        post_mlp_norm = getattr(layer, "post_mlp_norm", None)

        if parallel_residual:
            mlp_input = x_normed if shared_attention_norm or norm_2 is None else norm_2(hidden_states)
            hidden_states = hidden_states + attention_output
        else:
            hidden_states = hidden_states + attention_output
            if norm_2 is None:
                raise ValueError(f"Layer {layer.__class__.__name__} has no post-attention norm")
            mlp_input = norm_2(hidden_states)

        mlp_output = mlp(mlp_input)
        if isinstance(post_mlp_norm, nn.Module):
            mlp_output = post_mlp_norm(mlp_output)
        return hidden_states + mlp_output

    def _attention_scale(self, layer: nn.Module) -> float:
        attention = find_attention_module(layer)
        scalar = getattr(getattr(attention, "config", None), "attention_scores_scalar", None)
        return 1.0 / math.sqrt(scalar or self.head_dim)

    def _scale_embeddings_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        config = getattr(self.forward_module, "config", None)
        if bool(getattr(config, "scale_embeddings", False)):
            return x * torch.tensor(self.text_hidden_size ** 0.5, dtype=x.dtype, device=x.device)
        return x

    def _inject_audio_features(
        self,
        embeddings: torch.Tensor,
        audio_features: Optional[torch.Tensor],
        audio_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if audio_features is None:
            return embeddings
        if audio_positions is None:
            if audio_features.dim() != 3:
                raise ValueError("audio_features without audio_positions must have shape [B, T_audio, D]")
            audio_start = 1
            audio_end = 1 + audio_features.shape[1]
            embeddings[:, audio_start:audio_end, :] = audio_features.to(dtype=embeddings.dtype)
            return embeddings

        if audio_features.dim() != 4:
            raise ValueError("audio_features with audio_positions must have shape [B, C, T_audio, D]")
        batch_size, chunks, tokens, hidden = audio_features.shape
        if hidden != embeddings.shape[-1]:
            raise ValueError(f"audio feature dim {hidden} does not match context dim {embeddings.shape[-1]}")
        for batch_idx in range(batch_size):
            for chunk_idx in range(chunks):
                start, end = audio_positions[batch_idx, chunk_idx].tolist()
                if start < 0:
                    continue
                if end - start != tokens:
                    raise ValueError(f"audio span length {end - start} does not match features {tokens}")
                if end <= embeddings.shape[1]:
                    embeddings[batch_idx, start:end, :] = audio_features[batch_idx, chunk_idx].to(
                        dtype=embeddings.dtype
                    )
        return embeddings

    @torch.no_grad()
    def encode_context_full(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        h_context: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if h_context is None:
            if input_ids is None:
                raise ValueError("Either input_ids or h_context must be provided")
            h_context = self.token_embedding(input_ids)
            h_context = self._inject_audio_features(h_context, audio_features, audio_positions)
            h_context = self._scale_embeddings_if_needed(h_context)
        else:
            h_context = h_context.to(device=next(self.base_model.parameters()).device)

        x = h_context.detach()
        seq_len = x.shape[1]
        cos, sin = self._rope_cache(seq_len, x.device)
        layer_inputs: List[torch.Tensor] = []
        self.base_model.eval()
        for block in self.decoder_layers:
            layer_inputs.append(x)
            x = block(x, cos, sin, None, None, None)
        final_norm = getattr(getattr(self.forward_module, "transformer", None), "ln_f", None)
        if isinstance(final_norm, nn.Module):
            x = final_norm(x)
        return x.detach(), [item.detach() for item in layer_inputs]

    def _build_suffix_mask(
        self,
        *,
        batch_size: int,
        context_len: int,
        state_len: int,
        action_len: int,
        device: torch.device,
        speak_len: int = 0,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        suffix_len = state_len + speak_len + action_len
        total_key_len = context_len + suffix_len
        mask = torch.zeros(batch_size, suffix_len, total_key_len, device=device, dtype=torch.bool)

        suffix_chunk_ids, suffix_token_types, _, _ = _interleaved_suffix_layout(
            batch_size=batch_size,
            state_len=state_len,
            action_len=action_len,
            device=device,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_len=speak_len,
            speak_chunk_ids=speak_chunk_ids,
        )

        if context_valid_mask is None:
            context_allowed = torch.ones(batch_size, suffix_len, context_len, device=device, dtype=torch.bool)
        else:
            context_key_mask = context_valid_mask.to(device=device, dtype=torch.bool)
            if tuple(context_key_mask.shape) != (batch_size, context_len):
                raise ValueError(
                    f"context_valid_mask must have shape {(batch_size, context_len)}, "
                    f"got {tuple(context_key_mask.shape)}"
                )
            context_allowed = context_key_mask.unsqueeze(1).expand(batch_size, suffix_len, context_len)

        if context_chunk_ids is not None:
            context_chunks = context_chunk_ids.to(device=device)
            if tuple(context_chunks.shape) != (batch_size, context_len):
                raise ValueError(
                    f"context_chunk_ids must have shape {(batch_size, context_len)}, "
                    f"got {tuple(context_chunks.shape)}"
                )
            context_cutoff = context_chunks.unsqueeze(1) <= suffix_chunk_ids.unsqueeze(2)
            target_teacher_forcing = None
            if context_token_types is not None:
                token_types = context_token_types.to(device=device)
                if tuple(token_types.shape) != (batch_size, context_len):
                    raise ValueError(
                        f"context_token_types must have shape {(batch_size, context_len)}, "
                        f"got {tuple(token_types.shape)}"
                    )
                target_teacher_forcing = (
                    (token_types == CONTEXT_TOKEN_TARGET_CONTROL)
                    | (token_types == CONTEXT_TOKEN_TARGET_TEXT)
                )
                # Target control/text chunk c is timestamped in the observed
                # physical interval I_c.  The suffix decision c predicts the
                # next physical action interval I_{c+1}, so same-chunk target
                # tokens are history at that decision boundary, not future
                # labels.  Keep them visible while still masking chunks > c.
                context_cutoff = torch.where(
                    target_teacher_forcing.unsqueeze(1),
                    context_chunks.unsqueeze(1) <= suffix_chunk_ids.unsqueeze(2),
                    context_cutoff,
                )
            context_allowed = context_allowed & context_cutoff
            if context_observed_mask is not None:
                observed = context_observed_mask.to(device=device, dtype=torch.bool).unsqueeze(1)
                if target_teacher_forcing is not None:
                    observed = observed & ~target_teacher_forcing.unsqueeze(1)
                history_or_observed = (
                    context_chunks.unsqueeze(1) < suffix_chunk_ids.unsqueeze(2)
                ) | observed
                context_allowed = context_allowed & history_or_observed

        mask[:, :, :context_len] = context_allowed

        same_chunk = suffix_chunk_ids.unsqueeze(2) == suffix_chunk_ids.unsqueeze(1)
        query_types = suffix_token_types.unsqueeze(2)
        key_types = suffix_token_types.unsqueeze(1)
        # Per chunk: State sees State; Speak sees State+Speak; Action sees all.
        # Tokens within the same type block (e.g. the noisy-action horizon)
        # remain bidirectional.
        suffix_allowed = same_chunk & (key_types <= query_types)
        mask[:, :, context_len:] = suffix_allowed

        return mask.unsqueeze(1)

    def _build_joint_mask(
        self,
        *,
        batch_size: int,
        context_len: int,
        state_len: int,
        action_len: int,
        device: torch.device,
        speak_len: int = 0,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        suffix_len = state_len + speak_len + action_len
        total_len = context_len + suffix_len
        mask = torch.zeros(batch_size, total_len, total_len, device=device, dtype=torch.bool)

        if context_valid_mask is None:
            context_valid = torch.ones(batch_size, context_len, device=device, dtype=torch.bool)
        else:
            context_valid = context_valid_mask.to(device=device, dtype=torch.bool)
            if tuple(context_valid.shape) != (batch_size, context_len):
                raise ValueError(
                    f"context_valid_mask must have shape {(batch_size, context_len)}, "
                    f"got {tuple(context_valid.shape)}"
                )

        pos = torch.arange(context_len, device=device)
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)
        context_to_context = causal.unsqueeze(0) & context_valid.unsqueeze(1)
        context_to_context = context_to_context & context_valid.unsqueeze(2)
        mask[:, :context_len, :context_len] = context_to_context

        suffix_mask = self._build_suffix_mask(
            batch_size=batch_size,
            context_len=context_len,
            state_len=state_len,
            action_len=action_len,
            device=device,
            speak_len=speak_len,
            context_valid_mask=context_valid,
            context_chunk_ids=context_chunk_ids,
            context_token_types=context_token_types,
            context_observed_mask=context_observed_mask,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_chunk_ids=speak_chunk_ids,
        ).squeeze(1)
        mask[:, context_len:, :] = suffix_mask
        return mask.unsqueeze(1)

    def _mot_joint_layer_forward(
        self,
        *,
        layer_idx: int,
        context_layer: nn.Module,
        action_layer: HriActionExpertLayer,
        h_context: torch.Tensor,
        h_suffix: torch.Tensor,
        state_len: int,
        speak_len: int,
        action_len: int,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del layer_idx
        context_len = h_context.shape[1]
        suffix_len = h_suffix.shape[1]
        total_len = context_len + suffix_len
        joint_mask = self._build_joint_mask(
            batch_size=h_context.shape[0],
            context_len=context_len,
            state_len=state_len,
            speak_len=speak_len,
            action_len=action_len,
            device=h_context.device,
            context_valid_mask=context_valid_mask,
            context_chunk_ids=context_chunk_ids,
            context_token_types=context_token_types,
            context_observed_mask=context_observed_mask,
            state_chunk_ids=state_chunk_ids,
            speak_chunk_ids=speak_chunk_ids,
            action_chunk_ids=action_chunk_ids,
        )
        suffix_chunk_ids, _, _, _ = _interleaved_suffix_layout(
            batch_size=h_context.shape[0],
            state_len=state_len,
            action_len=action_len,
            device=h_context.device,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_len=speak_len,
            speak_chunk_ids=speak_chunk_ids,
        )
        suffix_position_ids = _local_suffix_position_ids(
            joint_mask[:, 0, context_len:, :context_len],
            suffix_chunk_ids,
        )
        rope_cache_len = context_len
        if suffix_position_ids.numel():
            rope_cache_len = max(rope_cache_len, int(suffix_position_ids.max().item()) + 1)
        cos, sin = self._rope_cache(rope_cache_len, h_context.device)
        cos_context = cos[:, :context_len] if cos is not None and context_len else None
        sin_context = sin[:, :context_len] if sin is not None and context_len else None
        cos_suffix = (
            _select_position_rows(cos.squeeze(0), suffix_position_ids) if cos is not None else None
        )
        sin_suffix = (
            _select_position_rows(sin.squeeze(0), suffix_position_ids) if sin is not None else None
        )

        context_normed, q_context, k_context, v_context = self._project_context_qkv(
            context_layer, h_context, cos_context, sin_context
        )
        suffix_normed, q_suffix, k_suffix, v_suffix = action_layer.project_qkv(h_suffix)
        q_suffix, k_suffix = _apply_rope_prefix(
            q_suffix, k_suffix, cos_suffix, sin_suffix, action_layer.config.rope_n_elem
        )

        q_all = torch.cat([q_context, q_suffix], dim=2)
        k_all = torch.cat([k_context, k_suffix], dim=2)
        v_all = torch.cat([v_context, v_suffix], dim=2)
        k_all = _repeat_kv(k_all, self.num_attention_heads)
        v_all = _repeat_kv(v_all, self.num_attention_heads)
        attn_output = F.scaled_dot_product_attention(
            q_all,
            k_all,
            v_all,
            attn_mask=joint_mask,
            dropout_p=0.0,
            scale=self._attention_scale(context_layer),
            is_causal=False,
        )
        batch_size = h_context.shape[0]
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, total_len, self.num_attention_heads * self.head_dim
        )
        context_attn = attn_output[:, :context_len]
        suffix_attn = attn_output[:, context_len:]
        h_context = self._apply_context_output(context_layer, h_context, context_attn, context_normed)
        h_suffix = action_layer.apply_attention_output(h_suffix, suffix_attn, suffix_normed)
        return h_context, h_suffix

    def _mot_layer_forward(
        self,
        *,
        layer_idx: int,
        context_layer_input: Optional[torch.Tensor],
        context_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        action_layer: HriActionExpertLayer,
        h_suffix: torch.Tensor,
        context_len: int,
        state_len: int,
        speak_len: int,
        action_len: int,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        suffix_len = h_suffix.shape[1]
        block_mask = self._build_suffix_mask(
            batch_size=h_suffix.shape[0],
            context_len=context_len,
            state_len=state_len,
            speak_len=speak_len,
            action_len=action_len,
            device=h_suffix.device,
            context_valid_mask=context_valid_mask,
            context_chunk_ids=context_chunk_ids,
            context_token_types=context_token_types,
            context_observed_mask=context_observed_mask,
            state_chunk_ids=state_chunk_ids,
            speak_chunk_ids=speak_chunk_ids,
            action_chunk_ids=action_chunk_ids,
        )
        suffix_chunk_ids, _, _, _ = _interleaved_suffix_layout(
            batch_size=h_suffix.shape[0],
            state_len=state_len,
            action_len=action_len,
            device=h_suffix.device,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_len=speak_len,
            speak_chunk_ids=speak_chunk_ids,
        )
        suffix_position_ids = _local_suffix_position_ids(
            block_mask[:, 0, :, :context_len],
            suffix_chunk_ids,
        )
        rope_cache_len = context_len
        if suffix_position_ids.numel():
            rope_cache_len = max(rope_cache_len, int(suffix_position_ids.max().item()) + 1)
        cos, sin = self._rope_cache(rope_cache_len, h_suffix.device)
        cos_context = cos[:, :context_len] if cos is not None and context_len else None
        sin_context = sin[:, :context_len] if sin is not None and context_len else None
        cos_suffix = (
            _select_position_rows(cos.squeeze(0), suffix_position_ids) if cos is not None else None
        )
        sin_suffix = (
            _select_position_rows(sin.squeeze(0), suffix_position_ids) if sin is not None else None
        )

        if context_kv is None:
            if context_layer_input is None:
                raise ValueError("context_layer_input or context_kv must be provided")
            k_context, v_context = self._project_context_kv(
                self.decoder_layers[layer_idx], context_layer_input, cos_context, sin_context
            )
        else:
            k_context, v_context = context_kv

        suffix_normed, q_suffix, k_suffix, v_suffix = action_layer.project_qkv(h_suffix)
        q_suffix, k_suffix = _apply_rope_prefix(
            q_suffix, k_suffix, cos_suffix, sin_suffix, action_layer.config.rope_n_elem
        )
        k_all = torch.cat([k_context, k_suffix], dim=2)
        v_all = torch.cat([v_context, v_suffix], dim=2)
        k_all = _repeat_kv(k_all, self.num_attention_heads)
        v_all = _repeat_kv(v_all, self.num_attention_heads)
        attn_output = F.scaled_dot_product_attention(
            q_suffix,
            k_all,
            v_all,
            attn_mask=block_mask,
            dropout_p=0.0,
            scale=self._attention_scale(self.decoder_layers[layer_idx]),
            is_causal=False,
        )
        batch_size = h_suffix.shape[0]
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, suffix_len, self.num_attention_heads * self.head_dim
        )
        return action_layer.apply_attention_output(h_suffix, attn_output, suffix_normed)

    def _prepare_suffix_inputs(
        self,
        *,
        robot_state: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        state_chunk_ids: Optional[torch.Tensor],
        speak_labels: Optional[torch.Tensor],
        speak_chunk_ids: Optional[torch.Tensor],
        emotion_labels: Optional[torch.Tensor],
        emotion_chunk_ids: Optional[torch.Tensor],
        response_elapsed_chunks: Optional[torch.Tensor],
        action_chunk_ids: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        dtype = next(self.action_expert.parameters()).dtype
        device = next(self.action_expert.parameters()).device
        robot_state = robot_state.to(device=device, dtype=dtype)
        noisy_action = noisy_action.to(device=device, dtype=dtype)

        if robot_state.dim() == 4:
            batch_size, chunks, robot_steps, robot_dim = robot_state.shape
            robot_tokens = self.robot_state_adapter(
                robot_state.reshape(batch_size * chunks, robot_steps, robot_dim)
            ).reshape(batch_size, chunks * robot_steps, -1)
            if state_chunk_ids is None:
                state_chunk_ids = torch.arange(chunks, device=device).repeat_interleave(robot_steps)
                state_chunk_ids = state_chunk_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            robot_tokens = self.robot_state_adapter(robot_state)
            if state_chunk_ids is None:
                state_chunk_ids = torch.zeros(
                    robot_tokens.shape[:2], device=device, dtype=torch.long
                )

        if noisy_action.dim() == 4:
            batch_size, chunks, action_horizon, action_dim = noisy_action.shape
            noisy_action = noisy_action.reshape(batch_size, chunks * action_horizon, action_dim)
            if tau.dim() == 2:
                if tuple(tau.shape) != (batch_size, chunks):
                    raise ValueError(
                        f"tau must have shape {(batch_size, chunks)} for 4D noisy_action, "
                        f"got {tuple(tau.shape)}"
                    )
                tau = tau.repeat_interleave(action_horizon, dim=1)
            elif tau.dim() == 3:
                if tuple(tau.shape) != (batch_size, chunks, action_horizon):
                    raise ValueError(
                        f"tau must have shape {(batch_size, chunks, action_horizon)} for 4D noisy_action, "
                        f"got {tuple(tau.shape)}"
                    )
                tau = tau.reshape(batch_size, chunks * action_horizon)
            elif tau.dim() != 1:
                raise ValueError(
                    f"tau must have shape [{batch_size}], [{batch_size}, {chunks}], or "
                    f"[{batch_size}, {chunks}, {action_horizon}], got {tuple(tau.shape)}"
                )
            if action_chunk_ids is None:
                action_chunk_ids = torch.arange(chunks, device=device).repeat_interleave(action_horizon)
                action_chunk_ids = action_chunk_ids.unsqueeze(0).expand(batch_size, -1)

        action_tokens = self.action_time_encoder(noisy_action, tau.to(device=device))
        if action_chunk_ids is None:
            action_chunk_ids = torch.zeros(
                action_tokens.shape[:2], device=device, dtype=torch.long
            )

        batch_size = robot_tokens.shape[0]
        hidden_size = robot_tokens.shape[-1]
        action_emotion_ids = torch.zeros(
            action_tokens.shape[:2],
            device=device,
            dtype=torch.long,
        )
        if self.speak_conditioning == "none":
            speak_tokens = robot_tokens.new_empty((batch_size, 0, hidden_size))
            speak_chunk_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        else:
            if speak_chunk_ids is None:
                if robot_state.dim() == 4:
                    speak_chunk_ids = torch.arange(
                        robot_state.shape[1], device=device, dtype=torch.long
                    ).unsqueeze(0).expand(batch_size, -1)
                else:
                    speak_chunk_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
            else:
                speak_chunk_ids = speak_chunk_ids.to(device=device, dtype=torch.long)
            speak_len = speak_chunk_ids.shape[1]
            if self.speak_token_embedding is None:
                raise RuntimeError("Speak conditioning is enabled but its token embedding is missing")
            if self.speak_conditioning == "predicted":
                query_ids = torch.zeros((batch_size, speak_len), device=device, dtype=torch.long)
                speak_tokens = self.speak_token_embedding(query_ids)
            else:
                if speak_labels is None:
                    raise ValueError("teacher_forced speak conditioning requires speak_labels")
                speak_labels = speak_labels.to(device=device, dtype=torch.long)
                if tuple(speak_labels.shape) != (batch_size, speak_len):
                    raise ValueError(
                        f"speak_labels must have shape {(batch_size, speak_len)}, "
                        f"got {tuple(speak_labels.shape)}"
                    )
                if ((speak_labels < 0) | (speak_labels > 1)).any():
                    raise ValueError("speak_labels must contain only 0 (silent) or 1 (speaking)")
                speak_tokens = self.speak_token_embedding(speak_labels)

            if self.response_time_encoder is not None:
                if response_elapsed_chunks is None:
                    raise ValueError(
                        "response time embedding requires response_elapsed_chunks"
                    )
                response_elapsed_chunks = response_elapsed_chunks.to(
                    device=device, dtype=torch.long
                )
                if tuple(response_elapsed_chunks.shape) != (batch_size, speak_len):
                    raise ValueError(
                        "response_elapsed_chunks must have shape "
                        f"{(batch_size, speak_len)}, got "
                        f"{tuple(response_elapsed_chunks.shape)}"
                    )
                expected_active = speak_labels.bool()
                if not torch.equal(
                    response_elapsed_chunks.ge(0), expected_active
                ):
                    raise ValueError(
                        "response_elapsed_chunks must be >= 0 exactly where "
                        "teacher-forced speak_labels are 1"
                    )
                speak_tokens = speak_tokens + self.response_time_encoder(
                    response_elapsed_chunks
                ).to(dtype=speak_tokens.dtype)
            elif response_elapsed_chunks is not None:
                raise ValueError(
                    "response_elapsed_chunks were provided but response time "
                    "embedding is disabled"
                )

            if self.use_emotion_conditioning:
                if (
                    self.emotion_embedding is None
                    or emotion_labels is None
                ):
                    raise ValueError(
                        "emotion conditioning requires emotion_labels"
                    )
                emotion_labels = emotion_labels.to(
                    device=device, dtype=torch.long
                )
                if tuple(emotion_labels.shape) != (
                    batch_size,
                    speak_len,
                ):
                    raise ValueError(
                        "emotion_labels must have shape "
                        f"{(batch_size, speak_len)}, got "
                        f"{tuple(emotion_labels.shape)}"
                    )
                if (
                    (emotion_labels < 0)
                    | (
                        emotion_labels
                        >= NUM_EMOTION_CONDITIONS
                    )
                ).any():
                    raise ValueError(
                        "emotion_labels must contain ids in [0, 6]"
                    )
                speaking = speak_labels.bool()
                if not torch.equal(
                    emotion_labels.ne(0),
                    speaking,
                ):
                    raise ValueError(
                        "emotion must be none exactly when SPEAK is silent"
                    )
                if emotion_chunk_ids is not None:
                    emotion_chunk_ids = emotion_chunk_ids.to(
                        device=device, dtype=torch.long
                    )
                    if tuple(emotion_chunk_ids.shape) != (
                        batch_size,
                        speak_len,
                    ):
                        raise ValueError(
                            "emotion_chunk_ids must match "
                            "emotion_labels"
                        )
                    if not torch.equal(
                        emotion_chunk_ids,
                        speak_chunk_ids,
                    ):
                        raise ValueError(
                            "emotion_chunk_ids must equal speak_chunk_ids "
                            "because emotion is an additive speaking condition"
                        )
                emotion_tokens = self.emotion_embedding(
                    emotion_labels
                ).to(dtype=speak_tokens.dtype)
                # Keep suffix length and RoPE positions identical to the
                # baseline: emotion conditions the existing speaking token
                # instead of inserting another sequence token.
                speak_tokens = speak_tokens + emotion_tokens
                action_emotion_ids = (
                    self._align_chunk_conditions_to_actions(
                        emotion_labels,
                        speak_chunk_ids,
                        action_chunk_ids.to(device=device, dtype=torch.long),
                    )
                )
            elif (
                emotion_labels is not None
                or emotion_chunk_ids is not None
            ):
                raise ValueError(
                    "emotion tensors were provided but conditioning is disabled"
                )

        state_len = robot_tokens.shape[1]
        speak_len = speak_tokens.shape[1]
        action_len = action_tokens.shape[1]
        _, _, permutation, restore_indices = _interleaved_suffix_layout(
            batch_size=batch_size,
            state_len=state_len,
            action_len=action_len,
            device=device,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_len=speak_len,
            speak_chunk_ids=speak_chunk_ids,
        )
        source_suffix = torch.cat([robot_tokens, speak_tokens, action_tokens], dim=1)
        h_suffix = _gather_sequence(source_suffix, permutation)
        return (
            h_suffix,
            state_len,
            speak_len,
            action_len,
            state_chunk_ids.to(device=device, dtype=torch.long),
            speak_chunk_ids.to(device=device, dtype=torch.long),
            action_chunk_ids.to(device=device, dtype=torch.long),
            restore_indices,
            action_emotion_ids,
        )

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        h_context: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        robot_state: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        speak_labels: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        emotion_chunk_ids: Optional[torch.Tensor] = None,
        response_elapsed_chunks: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if h_context is None:
            if input_ids is None:
                raise ValueError("Either input_ids or h_context must be provided")
            h_context = self.token_embedding(input_ids)
            h_context = self._inject_audio_features(h_context, audio_features, audio_positions)
            h_context = self._scale_embeddings_if_needed(h_context)
        else:
            h_context = h_context.to(device=next(self.base_model.parameters()).device)
        (
            h_suffix,
            state_len,
            speak_len,
            action_len,
            state_chunk_ids,
            speak_chunk_ids,
            action_chunk_ids,
            restore_indices,
            action_emotion_ids,
        ) = self._prepare_suffix_inputs(
            robot_state=robot_state,
            noisy_action=noisy_action,
            tau=tau,
            state_chunk_ids=state_chunk_ids,
            speak_labels=speak_labels,
            speak_chunk_ids=speak_chunk_ids,
            emotion_labels=emotion_labels,
            emotion_chunk_ids=emotion_chunk_ids,
            response_elapsed_chunks=response_elapsed_chunks,
            action_chunk_ids=action_chunk_ids,
        )
        for layer_idx, (context_layer, action_layer) in enumerate(zip(self.decoder_layers, self.action_expert.layers)):
            h_context, h_suffix = self._mot_joint_layer_forward(
                layer_idx=layer_idx,
                context_layer=context_layer,
                action_layer=action_layer,
                h_context=h_context,
                h_suffix=h_suffix,
                state_len=state_len,
                speak_len=speak_len,
                action_len=action_len,
                context_valid_mask=context_valid_mask,
                context_chunk_ids=context_chunk_ids,
                context_token_types=context_token_types,
                context_observed_mask=context_observed_mask,
                state_chunk_ids=state_chunk_ids,
                speak_chunk_ids=speak_chunk_ids,
                action_chunk_ids=action_chunk_ids,
            )
            h_suffix = self._apply_emotion_layer_film(
                h_suffix,
                layer_idx=layer_idx,
                state_len=state_len,
                speak_len=speak_len,
                restore_indices=restore_indices,
                action_condition_ids=action_emotion_ids,
            )
        final_norm = getattr(getattr(self.forward_module, "transformer", None), "ln_f", None)
        context_hidden = final_norm(h_context) if isinstance(final_norm, nn.Module) else h_context
        restored_suffix = _gather_sequence(h_suffix, restore_indices)
        h_state = restored_suffix[:, :state_len]
        h_speak = restored_suffix[:, state_len:state_len + speak_len]
        h_action = restored_suffix[:, state_len + speak_len:]
        outputs: Dict[str, torch.Tensor] = {
            "action_velocity": self.action_velocity_head(h_action)
        }
        if self.speak_prediction_head is not None:
            outputs["speak_logits"] = self.speak_prediction_head(h_speak)
        if return_hidden_states:
            outputs.update(
                {
                    "context_hidden": context_hidden,
                    "h_state": h_state,
                    "h_speak": h_speak,
                    "h_action": h_action,
                    "state_chunk_ids": state_chunk_ids,
                    "speak_chunk_ids": speak_chunk_ids,
                    "action_chunk_ids": action_chunk_ids,
                }
            )
        return outputs

    def set_context_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.forward_module.set_kv_cache(
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            rope_cache_length=rope_cache_length,
            device=device,
            dtype=dtype,
        )

    def clear_context_kv_cache(self) -> None:
        self.forward_module.clear_kv_cache()

    @torch.no_grad()
    def append_context_kv_cache(
        self,
        *,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        input_pos_maxp1: Optional[int] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_pos.shape[-1] != input_ids.shape[1]:
            raise ValueError(
                f"input_pos length {input_pos.shape[-1]} must match input_ids length {input_ids.shape[1]}"
            )

        device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(device=device)
        input_pos = input_pos.to(device=device)
        embeddings = self.token_embedding(input_ids)
        embeddings = self._inject_audio_features(embeddings, audio_features, audio_positions)
        x = self._scale_embeddings_if_needed(embeddings).detach()

        cos_source, sin_source = self._rope_cache(int(input_pos.max().item()) + 1, device)
        if cos_source is None or sin_source is None:
            cos = sin = None
        else:
            cos = _select_position_rows(cos_source.squeeze(0), input_pos)
            sin = _select_position_rows(sin_source.squeeze(0), input_pos)

        mask_cache = getattr(self.forward_module, "mask_cache", None)
        if mask_cache is None:
            raise RuntimeError("Context KV cache mask is empty. Call set_context_kv_cache first.")
        mask = _batched_index_select(mask_cache, 2, input_pos)
        if mask.dim() > 4:
            mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
        if input_pos_maxp1 is not None:
            mask = mask[..., : int(input_pos_maxp1)]

        self.base_model.eval()
        for block in self.decoder_layers:
            x = block(x, cos, sin, mask, input_pos, input_pos_maxp1)
        final_norm = getattr(getattr(self.forward_module, "transformer", None), "ln_f", None)
        if isinstance(final_norm, nn.Module):
            x = final_norm(x)
        return x.detach()

    def context_kv_cache(
        self,
        *,
        batch_size: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.decoder_layers:
            attention = find_attention_module(layer)
            cache = getattr(attention, "kv_cache", None) if attention is not None else None
            if cache is None:
                raise RuntimeError("Base model KV cache is empty. Call set_context_kv_cache and run context tokens first.")
            k = cache.k
            v = cache.v
            if batch_size is not None:
                k = k[:batch_size]
                v = v[:batch_size]
            if context_length is not None:
                k = k[..., :context_length, :]
                v = v[..., :context_length, :]
            caches.append((k.detach(), v.detach()))
        return caches

    def forward_from_context_kv_cache(
        self,
        *,
        robot_state: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        context_length: int,
        context_valid_mask: Optional[torch.Tensor] = None,
        context_chunk_ids: Optional[torch.Tensor] = None,
        context_token_types: Optional[torch.Tensor] = None,
        context_observed_mask: Optional[torch.Tensor] = None,
        state_chunk_ids: Optional[torch.Tensor] = None,
        speak_labels: Optional[torch.Tensor] = None,
        speak_chunk_ids: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        emotion_chunk_ids: Optional[torch.Tensor] = None,
        response_elapsed_chunks: Optional[torch.Tensor] = None,
        action_chunk_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        (
            h_suffix,
            state_len,
            speak_len,
            action_len,
            state_chunk_ids,
            speak_chunk_ids,
            action_chunk_ids,
            restore_indices,
            action_emotion_ids,
        ) = self._prepare_suffix_inputs(
            robot_state=robot_state,
            noisy_action=noisy_action,
            tau=tau,
            state_chunk_ids=state_chunk_ids,
            speak_labels=speak_labels,
            speak_chunk_ids=speak_chunk_ids,
            emotion_labels=emotion_labels,
            emotion_chunk_ids=emotion_chunk_ids,
            response_elapsed_chunks=response_elapsed_chunks,
            action_chunk_ids=action_chunk_ids,
        )
        caches = self.context_kv_cache(batch_size=h_suffix.shape[0], context_length=context_length)
        for layer_idx, action_layer in enumerate(self.action_expert.layers):
            h_suffix = self._mot_layer_forward(
                layer_idx=layer_idx,
                context_layer_input=None,
                context_kv=caches[layer_idx],
                action_layer=action_layer,
                h_suffix=h_suffix,
                context_len=context_length,
                state_len=state_len,
                speak_len=speak_len,
                action_len=action_len,
                context_valid_mask=context_valid_mask,
                context_chunk_ids=context_chunk_ids,
                context_token_types=context_token_types,
                context_observed_mask=context_observed_mask,
                state_chunk_ids=state_chunk_ids,
                speak_chunk_ids=speak_chunk_ids,
                action_chunk_ids=action_chunk_ids,
            )
            h_suffix = self._apply_emotion_layer_film(
                h_suffix,
                layer_idx=layer_idx,
                state_len=state_len,
                speak_len=speak_len,
                restore_indices=restore_indices,
                action_condition_ids=action_emotion_ids,
            )
        restored_suffix = _gather_sequence(h_suffix, restore_indices)
        h_state = restored_suffix[:, :state_len]
        h_speak = restored_suffix[:, state_len:state_len + speak_len]
        h_action = restored_suffix[:, state_len + speak_len:]
        outputs: Dict[str, torch.Tensor] = {
            "action_velocity": self.action_velocity_head(h_action)
        }
        if self.speak_prediction_head is not None:
            outputs["speak_logits"] = self.speak_prediction_head(h_speak)
        if return_hidden_states:
            outputs.update({"h_state": h_state, "h_speak": h_speak, "h_action": h_action})
        return outputs

    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def action_expert_parameter_count(self) -> int:
        return self.trainable_parameter_count()

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def base_parameters_frozen(self) -> bool:
        return all(not param.requires_grad for param in self.base_model.parameters())
