import torch


DEFAULT_TAU_BETA_ALPHA = 1.5
DEFAULT_TAU_CUTOFF = 0.999


def sample_pi0_beta_tau(
    shape,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cutoff: float = DEFAULT_TAU_CUTOFF,
    alpha: float = DEFAULT_TAU_BETA_ALPHA,
) -> torch.Tensor:
    if cutoff <= 0.0 or cutoff > 1.0:
        raise ValueError(f"cutoff must be in (0, 1], got {cutoff}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    sample_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    beta_sample = torch.rand(shape, device=device, dtype=sample_dtype).pow(1.0 / alpha)
    tau = float(cutoff) * (1.0 - beta_sample)
    return tau.to(dtype=dtype)


def sample_flow_matching_pair(action_gt: torch.Tensor):
    if action_gt.dim() not in (3, 4):
        raise ValueError(
            "action_gt must have shape [B, action_horizon, action_dim] or "
            f"[B, chunks, action_horizon, action_dim], got {tuple(action_gt.shape)}"
        )
    batch_size = action_gt.shape[0]
    noise = torch.randn_like(action_gt)
    if action_gt.dim() == 4:
        chunks = action_gt.shape[1]
        tau = sample_pi0_beta_tau(
            (batch_size, chunks),
            device=action_gt.device,
            dtype=action_gt.dtype,
        )
        tau_view = tau.view(batch_size, chunks, 1, 1)
    else:
        tau = sample_pi0_beta_tau(
            (batch_size,),
            device=action_gt.device,
            dtype=action_gt.dtype,
        )
        tau_view = tau.view(batch_size, 1, 1)
    x_tau = (1.0 - tau_view) * noise + tau_view * action_gt
    target_velocity = action_gt - noise
    return x_tau, tau, target_velocity
