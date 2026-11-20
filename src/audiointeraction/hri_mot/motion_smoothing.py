"""Offline motion stitching and zero-phase smoothing for HRI-MoT actions."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional

import torch


def _validate_action_chunks(action: torch.Tensor) -> None:
    if action.dim() != 3:
        raise ValueError(f"action must have shape [chunks, horizon, dim], got {tuple(action.shape)}")
    if action.shape[1] <= 0 or action.shape[2] <= 0:
        raise ValueError(f"action horizon and dim must be positive, got {tuple(action.shape)}")


def _quintic_samples(
    *,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    end_position: torch.Tensor,
    end_velocity: torch.Tensor,
    end_acceleration: torch.Tensor,
    duration_frames: int,
) -> torch.Tensor:
    """Sample a quintic trajectory at t=1..T-1 with C2 endpoint constraints."""
    if duration_frames <= 1:
        return start_position.new_empty((0, start_position.shape[-1]))

    duration = float(duration_frames)
    c0 = start_position
    c1 = start_velocity
    c2 = 0.5 * start_acceleration
    position_residual = end_position - (c0 + c1 * duration + c2 * duration**2)
    velocity_residual = end_velocity - (c1 + 2.0 * c2 * duration)
    acceleration_residual = end_acceleration - 2.0 * c2

    c3 = (
        10.0 * position_residual / duration**3
        - 4.0 * velocity_residual / duration**2
        + 0.5 * acceleration_residual / duration
    )
    c4 = (
        -15.0 * position_residual / duration**4
        + 7.0 * velocity_residual / duration**3
        - acceleration_residual / duration**2
    )
    c5 = (
        6.0 * position_residual / duration**5
        - 3.0 * velocity_residual / duration**4
        + 0.5 * acceleration_residual / duration**3
    )

    time = torch.arange(
        1,
        duration_frames,
        device=start_position.device,
        dtype=start_position.dtype,
    ).unsqueeze(1)
    return c0 + c1 * time + c2 * time**2 + c3 * time**3 + c4 * time**4 + c5 * time**5


def stitch_action_chunk_c2(
    previous_action: torch.Tensor,
    current_action: torch.Tensor,
    *,
    bridge_frames: int,
) -> torch.Tensor:
    """Join one action chunk to the previous emitted chunk with a C2 bridge.

    The first ``bridge_frames`` samples of ``current_action`` are replaced by
    a quintic trajectory. The bridge starts at the previous chunk endpoint and
    meets an untouched future sample in the current chunk while matching
    position, velocity, and acceleration at both ends.
    """
    if previous_action.dim() != 2 or current_action.dim() != 2:
        raise ValueError(
            "previous_action and current_action must have shape [horizon, dim], "
            f"got {tuple(previous_action.shape)} and {tuple(current_action.shape)}"
        )
    if previous_action.shape[1] != current_action.shape[1]:
        raise ValueError(
            f"action dims must match, got {previous_action.shape[1]} and {current_action.shape[1]}"
        )
    if bridge_frames < 0:
        raise ValueError(f"bridge_frames must be >= 0, got {bridge_frames}")

    output = current_action.clone()
    # Two untouched future samples are needed to estimate the endpoint
    # velocity and acceleration without using samples replaced by the bridge.
    bridge = min(int(bridge_frames), max(int(current_action.shape[0]) - 3, 0))
    if bridge == 0:
        return output

    start_position = previous_action[-1]
    if previous_action.shape[0] >= 2:
        start_velocity = previous_action[-1] - previous_action[-2]
    else:
        start_velocity = torch.zeros_like(start_position)
    if previous_action.shape[0] >= 3:
        start_acceleration = previous_action[-1] - 2.0 * previous_action[-2] + previous_action[-3]
    else:
        start_acceleration = torch.zeros_like(start_position)

    end_position = current_action[bridge]
    end_velocity = current_action[bridge + 1] - current_action[bridge]
    end_acceleration = (
        current_action[bridge + 2]
        - 2.0 * current_action[bridge + 1]
        + current_action[bridge]
    )
    output[:bridge] = _quintic_samples(
        start_position=start_position,
        start_velocity=start_velocity,
        start_acceleration=start_acceleration,
        end_position=end_position,
        end_velocity=end_velocity,
        end_acceleration=end_acceleration,
        duration_frames=bridge + 1,
    )
    return output


def stitch_action_chunks_c2(action: torch.Tensor, *, bridge_frames: int) -> torch.Tensor:
    """C2-stitch all chunks sequentially so each bridge uses emitted history."""
    _validate_action_chunks(action)
    if bridge_frames < 0:
        raise ValueError(f"bridge_frames must be >= 0, got {bridge_frames}")
    output = action.clone()
    for chunk_idx in range(1, output.shape[0]):
        output[chunk_idx] = stitch_action_chunk_c2(
            output[chunk_idx - 1],
            action[chunk_idx],
            bridge_frames=bridge_frames,
        )
    return output


def _butterworth_biquad_coefficients(*, fps: float, cutoff_hz: float) -> tuple[float, ...]:
    if fps <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps}")
    nyquist = 0.5 * float(fps)
    if not (0.0 < float(cutoff_hz) < nyquist):
        raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({nyquist}), got {cutoff_hz}")

    tangent = math.tan(math.pi * float(cutoff_hz) / float(fps))
    norm = 1.0 / (1.0 + math.sqrt(2.0) * tangent + tangent * tangent)
    b0 = tangent * tangent * norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (tangent * tangent - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * tangent + tangent * tangent) * norm
    return b0, b1, b2, a1, a2


def _biquad_forward(sequence: torch.Tensor, coefficients: tuple[float, ...]) -> torch.Tensor:
    """Apply one causal biquad pass to a [frames, dim] sequence."""
    b0, b1, b2, a1, a2 = coefficients
    output = torch.empty_like(sequence)
    x1 = sequence[0]
    x2 = sequence[0]
    y1 = sequence[0]
    y2 = sequence[0]
    for frame_idx in range(sequence.shape[0]):
        x0 = sequence[frame_idx]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[frame_idx] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return output


class CausalButterworthActionFilter:
    """Stateful second-order low-pass filter for real-time action chunks.

    Unlike the offline forward/backward filter, this filter uses only the
    current and previous samples. Its biquad state is retained across chunk
    boundaries so a 400 ms boundary never resets the trajectory response.
    """

    def __init__(self, *, fps: float, cutoff_hz: float) -> None:
        self.fps = float(fps)
        self.cutoff_hz = float(cutoff_hz)
        self.coefficients = _butterworth_biquad_coefficients(
            fps=self.fps,
            cutoff_hz=self.cutoff_hz,
        )
        self._state: Optional[tuple[torch.Tensor, ...]] = None

    def reset(self, initial_value: Optional[torch.Tensor] = None) -> None:
        """Clear history or seed it with the robot's current joint position."""

        self._state = None
        if initial_value is None:
            return
        value = torch.as_tensor(initial_value).detach()
        if value.dim() != 1 or value.numel() == 0:
            raise ValueError(
                "initial_value must have shape [action_dim], "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("initial_value contains NaN or infinity")
        seed = value.clone()
        self._state = (seed.clone(), seed.clone(), seed.clone(), seed.clone())

    def process(self, sequence: torch.Tensor) -> torch.Tensor:
        """Filter one ``[frames, action_dim]`` chunk and retain its state."""

        if sequence.dim() != 2 or sequence.shape[0] < 1 or sequence.shape[1] < 1:
            raise ValueError(
                "sequence must have shape [frames, action_dim], "
                f"got {tuple(sequence.shape)}"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("sequence contains NaN or infinity")

        original_dtype = sequence.dtype
        working = (
            sequence.float()
            if sequence.dtype in (torch.float16, torch.bfloat16)
            else sequence
        )
        if self._state is None:
            seed = working[0].detach().clone()
            x1, x2, y1, y2 = (
                seed.clone(),
                seed.clone(),
                seed.clone(),
                seed.clone(),
            )
        else:
            if self._state[0].numel() != working.shape[1]:
                raise ValueError(
                    "action dimension changed from "
                    f"{self._state[0].numel()} to {working.shape[1]}"
                )
            x1, x2, y1, y2 = (
                value.to(device=working.device, dtype=working.dtype)
                for value in self._state
            )

        b0, b1, b2, a1, a2 = self.coefficients
        output = torch.empty_like(working)
        for frame_idx in range(working.shape[0]):
            x0 = working[frame_idx]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            output[frame_idx] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        self._state = tuple(
            value.detach().clone() for value in (x1, x2, y1, y2)
        )
        return output.to(dtype=original_dtype)


def zero_phase_butterworth_lowpass(
    sequence: torch.Tensor,
    *,
    fps: float,
    cutoff_hz: float,
) -> torch.Tensor:
    """Apply a second-order Butterworth forward/backward with no phase delay."""
    if sequence.dim() != 2:
        raise ValueError(f"sequence must have shape [frames, dim], got {tuple(sequence.shape)}")
    if sequence.shape[0] <= 1:
        return sequence.clone()

    coefficients = _butterworth_biquad_coefficients(fps=fps, cutoff_hz=cutoff_hz)
    original_dtype = sequence.dtype
    working = sequence.float() if sequence.dtype in (torch.float16, torch.bfloat16) else sequence
    pad_frames = min(9, int(working.shape[0]) - 1)
    if pad_frames > 0:
        left = working[1 : pad_frames + 1].flip(0)
        right = working[-pad_frames - 1 : -1].flip(0)
        working = torch.cat((left, working, right), dim=0)

    filtered = _biquad_forward(working, coefficients)
    filtered = _biquad_forward(filtered.flip(0), coefficients).flip(0)
    if pad_frames > 0:
        filtered = filtered[pad_frames:-pad_frames]
    return filtered.to(dtype=original_dtype)


def smooth_action_chunks_offline(
    action: torch.Tensor,
    *,
    bridge_frames: int,
    lowpass_cutoff_hz: float,
    fps: float,
) -> torch.Tensor:
    """C2-stitch chunks, then apply an offline zero-phase low-pass filter."""
    stitched = stitch_action_chunks_c2(action, bridge_frames=bridge_frames)
    cutoff = float(lowpass_cutoff_hz)
    if cutoff < 0.0:
        raise ValueError(f"lowpass_cutoff_hz must be >= 0, got {lowpass_cutoff_hz}")
    if cutoff == 0.0 or stitched.shape[0] * stitched.shape[1] <= 1:
        return stitched
    chunks, horizon, action_dim = stitched.shape
    flattened = stitched.reshape(chunks * horizon, action_dim)
    filtered = zero_phase_butterworth_lowpass(flattened, fps=fps, cutoff_hz=cutoff)
    return filtered.reshape_as(stitched)


class FixedLagZeroPhaseActionSmoother:
    """Approximate offline zero-phase smoothing with bounded future context.

    A genuinely offline forward/backward filter can use every future frame in
    the recording.  An unbounded microphone stream cannot.  This helper waits
    for ``lookahead_chunks`` future chunks, filters an overlapping window, and
    emits the oldest now-stable chunk.  A final C2 bridge keeps independently
    filtered windows continuous at their emitted boundaries.
    """

    def __init__(
        self,
        *,
        lookahead_chunks: int,
        history_chunks: int,
        bridge_frames: int,
        cutoff_hz: float,
        fps: float,
    ) -> None:
        if lookahead_chunks < 1:
            raise ValueError("lookahead_chunks must be at least 1")
        if history_chunks < 0:
            raise ValueError("history_chunks must be non-negative")
        if bridge_frames < 0:
            raise ValueError("bridge_frames must be non-negative")
        if cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.lookahead_chunks = int(lookahead_chunks)
        self.history_chunks = int(history_chunks)
        self.bridge_frames = int(bridge_frames)
        self.cutoff_hz = float(cutoff_hz)
        self.fps = float(fps)
        self._chunks: Deque[torch.Tensor] = deque()
        self._window_start_index = 0
        self._next_output_index = 0
        self._previous_output: Optional[torch.Tensor] = None
        self._shape: Optional[tuple[int, int]] = None

    @property
    def buffered_chunks(self) -> int:
        latest_index = self._window_start_index + len(self._chunks) - 1
        if latest_index < self._next_output_index:
            return 0
        return int(latest_index - self._next_output_index + 1)

    def reset(self) -> None:
        self._chunks.clear()
        self._window_start_index = 0
        self._next_output_index = 0
        self._previous_output = None
        self._shape = None

    def push(
        self,
        action: torch.Tensor,
    ) -> Optional[tuple[int, torch.Tensor]]:
        if action.dim() != 2:
            raise ValueError(
                "action must have shape [horizon, dim], "
                f"got {tuple(action.shape)}"
            )
        shape = (int(action.shape[0]), int(action.shape[1]))
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"action dimensions must be positive, got {shape}")
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError(
                f"action shape changed from {self._shape} to {shape}"
            )
        self._chunks.append(action.detach().clone())
        latest_index = self._window_start_index + len(self._chunks) - 1
        if latest_index - self._next_output_index < self.lookahead_chunks:
            return None

        target_index = self._next_output_index
        desired_start = max(0, target_index - self.history_chunks)
        local_start = max(0, desired_start - self._window_start_index)
        window_start = self._window_start_index + local_start
        window_chunks = list(self._chunks)[local_start:]
        window = torch.stack(window_chunks, dim=0)
        chunks, horizon, action_dim = window.shape
        filtered = zero_phase_butterworth_lowpass(
            window.reshape(chunks * horizon, action_dim),
            fps=self.fps,
            cutoff_hz=self.cutoff_hz,
        ).reshape_as(window)
        target_local = target_index - window_start
        output = filtered[target_local]
        if self._previous_output is not None:
            output = stitch_action_chunk_c2(
                self._previous_output,
                output,
                bridge_frames=self.bridge_frames,
            )
        self._previous_output = output.detach().clone()
        self._next_output_index += 1

        retain_from = max(0, self._next_output_index - self.history_chunks)
        while self._window_start_index < retain_from:
            self._chunks.popleft()
            self._window_start_index += 1
        return int(target_index), output


def _summary(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def action_motion_stats(action: torch.Tensor, *, boundary_window_frames: int = 3) -> Dict[str, object]:
    """Summarize velocity/acceleration/jerk, including boundary neighborhoods."""
    _validate_action_chunks(action)
    if boundary_window_frames < 0:
        raise ValueError(f"boundary_window_frames must be >= 0, got {boundary_window_frames}")
    chunks, horizon, action_dim = action.shape
    flat = action.reshape(chunks * horizon, action_dim).float()
    velocity = flat[1:] - flat[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    velocity_norm = velocity.norm(dim=-1)
    acceleration_norm = acceleration.norm(dim=-1)
    jerk_norm = jerk.norm(dim=-1)

    boundary_frames = torch.arange(horizon, flat.shape[0], horizon, device=flat.device)
    if boundary_frames.numel() > 0:
        exact_boundary_velocity = velocity_norm[boundary_frames - 1]
        acceleration_centers = torch.arange(1, flat.shape[0] - 1, device=flat.device)
        near_boundary = torch.zeros_like(acceleration_centers, dtype=torch.bool)
        for boundary in boundary_frames:
            near_boundary |= (acceleration_centers - boundary).abs() <= int(boundary_window_frames)
        boundary_acceleration = acceleration_norm[near_boundary]
    else:
        exact_boundary_velocity = velocity_norm.new_empty((0,))
        boundary_acceleration = acceleration_norm.new_empty((0,))

    phase_step_mean = []
    destination_frames = torch.arange(1, flat.shape[0], device=flat.device)
    for phase in range(horizon):
        phase_values = velocity_norm[destination_frames.remainder(horizon) == phase]
        phase_step_mean.append(float(phase_values.mean().item()) if phase_values.numel() else 0.0)

    return {
        "frames": int(flat.shape[0]),
        "action_dim": int(action_dim),
        "step_norm": _summary(velocity_norm),
        "acceleration_norm": _summary(acceleration_norm),
        "jerk_norm": _summary(jerk_norm),
        "exact_boundary_step_norm": _summary(exact_boundary_velocity),
        "boundary_window_acceleration_norm": _summary(boundary_acceleration),
        "chunk_phase_step_mean": phase_step_mean,
    }
