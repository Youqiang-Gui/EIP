"""Response-relative timing shared by HRI-MoT training and inference."""

from __future__ import annotations

import torch


def response_elapsed_chunks_from_labels(speak_labels: torch.Tensor) -> torch.Tensor:
    """Map binary speech labels to ``-1`` or response-relative ``0, 1, ...``.

    A new run starts from zero after every silent-to-speaking transition.  The
    last dimension is treated as the physical 400 ms timeline; any leading
    dimensions are independent sequences.
    """

    if speak_labels.dim() < 1:
        raise ValueError("speak_labels must have at least one dimension")
    labels = speak_labels.to(dtype=torch.long)
    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("speak_labels must contain only 0 or 1")

    timeline = labels.reshape(-1, labels.shape[-1])
    output = torch.full_like(timeline, -1)
    elapsed = torch.full(
        (timeline.shape[0],), -1, device=timeline.device, dtype=torch.long
    )
    previous_active = torch.zeros(
        timeline.shape[0], device=timeline.device, dtype=torch.bool
    )
    for chunk_idx in range(timeline.shape[1]):
        active = timeline[:, chunk_idx].bool()
        elapsed = torch.where(
            active,
            torch.where(previous_active, elapsed + 1, torch.zeros_like(elapsed)),
            torch.full_like(elapsed, -1),
        )
        output[:, chunk_idx] = elapsed
        previous_active = active
    return output.reshape_as(labels)


def shifted_response_elapsed_chunks(
    target_state_ids: torch.Tensor,
    *,
    decisions: int,
) -> torch.Tensor:
    """Return response time for decisions whose targets are physical chunks c+1."""

    decisions = int(decisions)
    if decisions < 0:
        raise ValueError(f"decisions must be non-negative, got {decisions}")
    if target_state_ids.dim() != 2:
        raise ValueError(
            "target_state_ids must have shape [B, physical_chunks], "
            f"got {tuple(target_state_ids.shape)}"
        )
    if decisions + 1 > target_state_ids.shape[1]:
        raise ValueError(
            f"Need {decisions + 1} physical labels for {decisions} decisions, "
            f"got {target_state_ids.shape[1]}"
        )
    elapsed = response_elapsed_chunks_from_labels(target_state_ids)
    return elapsed[:, 1 : decisions + 1]
