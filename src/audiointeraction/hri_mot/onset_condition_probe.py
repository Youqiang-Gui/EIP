"""Matched six-emotion rollouts at each observed speaking onset.

The normal inference supplies the language context once per physical chunk.
Action forwards only read that context's KV cache. The six probes share it,
but own their robot history and noise generators; neither is committed back
to the normal inference. No future text or audio is exposed to the probes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .emotion_conditioning import EMOTION_LABELS
from .motion_smoothing import stitch_action_chunk_c2


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class OnsetConditionProbe:
    def __init__(self, output_dir: Path, *, seconds: float, chunk_seconds: float,
                 fps: float, qpos_indices: list[int], bridge_frames: int,
                 base_yaw_deg: float, metadata: dict[str, Any]):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("onset probe duration must be finite and positive")
        chunks = seconds / chunk_seconds
        if not math.isclose(chunks, round(chunks), abs_tol=1e-6):
            raise ValueError("onset probe duration must be a multiple of the action chunk duration")
        self.max_chunks = int(round(chunks))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.bridge_frames = int(bridge_frames)
        self.active = None
        self.turn_count = 0
        self.manifest = dict(
            version="speaking_onset_six_emotions_v2", status="running",
            requested_seconds=float(seconds), chunk_seconds=float(chunk_seconds),
            fps=float(fps), qpos_indices=list(qpos_indices),
            bridge_frames=self.bridge_frames, base_yaw_deg=float(base_yaw_deg),
            context_policy="same_live_causal_language_context_per_chunk",
            noise_policy="independent_generators_cloned_at_onset_and_matched_each_chunk",
            history_policy="same_onset_state_then_each_branch_uses_its_own_c2_rollout",
            video_motion="c2_rollout_without_offline_zero_phase_filter",
            metadata=metadata, turns=[],
        )
        self._write_manifest()

    def _write_manifest(self):
        path = self.output_dir / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))
        temporary.replace(path)

    @torch.no_grad()
    def step(self, *, session, sample_action_fn, source_state, previous_action,
             action_rng_state, chunk_idx, speak_label, elapsed_chunk,
             emotion_token_id, sampling_steps, initial_noise, main_raw, main_c2,
             interrupted=False):
        if self.active is not None and (not speak_label or elapsed_chunk == 0):
            self._finish("interrupted" if interrupted else "speaking_ended_or_restarted")
        if not speak_label:
            return
        device = source_state.device
        if elapsed_chunk == 0:
            self.turn_count += 1
            branches = {}
            for label in EMOTION_LABELS:
                generator = torch.Generator(device=device)
                generator.set_state(action_rng_state.clone())
                branches[label] = dict(
                    generator=generator, state=source_state.detach().clone(),
                    previous=None if previous_action is None else previous_action.detach().clone(),
                    raw=[], c2=[],
                )
            self.active = dict(
                turn=self.turn_count, onset_chunk=int(chunk_idx),
                start_sec=(chunk_idx + 1) * self.manifest["chunk_seconds"],
                semantic_emotion_token_id=int(emotion_token_id),
                initial_state=source_state.detach().cpu().clone(),
                initial_rng=action_rng_state.cpu().clone(), branches=branches,
                chunks=[], auto_raw=[], auto_c2=[],
            )
        if self.active is None:
            return
        turn = self.active
        if elapsed_chunk != len(turn["chunks"]) or chunk_idx != turn["onset_chunk"] + elapsed_chunk:
            raise RuntimeError("Non-contiguous speaking chunks in onset probe")
        if emotion_token_id != turn["semantic_emotion_token_id"]:
            raise RuntimeError("Semantic emotion changed inside the probed speaking turn")
        branches = list(turn["branches"].values())
        reference = branches[0]
        if any(not torch.equal(reference["generator"].get_state(), branch["generator"].get_state()) for branch in branches[1:]):
            raise RuntimeError("Paired action noise generators diverged")
        if elapsed_chunk == 0 and any(not torch.equal(reference["state"], branch["state"]) for branch in branches[1:]):
            raise RuntimeError("Paired onset states differ")
        noise_digest = tensor_digest(reference["generator"].get_state())
        context = session.action_context_tensors()
        # Preserve even the global RNG used by legacy language sampling. The
        # actual action noise comes exclusively from each private generator.
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            for label, branch in turn["branches"].items():
                proxy = SimpleNamespace(
                    wrapper=session.wrapper, normalizer=session.normalizer,
                    emotion_override=label,
                    action_context_tensors=lambda: context,
                )
                raw = sample_action_fn(
                    session=proxy, robot_state=branch["state"], chunk_idx=chunk_idx,
                    action_horizon=main_raw.shape[0], action_dim=main_raw.shape[1],
                    sampling_steps=sampling_steps, solver="euler", initial_noise=initial_noise,
                    device=device, action_generator=branch["generator"],
                    teacher_forced_speak_label=1, response_elapsed_chunk=elapsed_chunk,
                    emotion_token_id=emotion_token_id,
                )
                if not torch.isfinite(raw).all():
                    raise RuntimeError(f"Non-finite {label} action in onset probe")
                c2 = raw.clone() if branch["previous"] is None else stitch_action_chunk_c2(
                    branch["previous"], raw, bridge_frames=self.bridge_frames)
                branch["raw"].append(raw.detach().cpu().clone())
                branch["c2"].append(c2.detach().cpu().clone())
                branch["state"] = c2.detach()
                branch["previous"] = c2.detach()
        if any(not torch.equal(reference["generator"].get_state(), branch["generator"].get_state()) for branch in branches[1:]):
            raise RuntimeError("Paired branches consumed different amounts of action noise")
        turn["auto_raw"].append(main_raw.detach().cpu().clone())
        turn["auto_c2"].append(main_c2.detach().cpu().clone())
        turn["chunks"].append(dict(
            chunk_idx=int(chunk_idx), elapsed_chunk=int(elapsed_chunk),
            context_length=int(context["context_length"]), noise_state_sha256=noise_digest,
            context_metadata_sha256={k: tensor_digest(v) for k, v in context.items() if torch.is_tensor(v)},
        ))
        if len(turn["chunks"]) >= self.max_chunks:
            self._finish("window_complete")

    def _finish(self, reason):
        turn = self.active
        if turn is None:
            return
        path = self.output_dir / f"turn_{turn['turn']:02d}.npz"
        arrays = dict(initial_state=turn["initial_state"].numpy(), initial_rng_state=turn["initial_rng"].numpy())
        for label in EMOTION_LABELS:
            for kind in ("raw", "c2"):
                arrays[f"{label}_{kind}"] = torch.stack(turn["branches"][label][kind]).float().numpy()
        for kind in ("raw", "c2"):
            arrays[f"auto_{kind}"] = torch.stack(turn[f"auto_{kind}"]).float().numpy()
        np.savez_compressed(path, **arrays)
        row = {k: turn[k] for k in ("turn", "onset_chunk", "start_sec", "semantic_emotion_token_id", "chunks")}
        row.update(data=path.name, duration_sec=len(turn["chunks"]) * self.manifest["chunk_seconds"],
                   finish_reason=reason, initial_state_sha256=tensor_digest(turn["initial_state"]),
                   initial_rng_sha256=tensor_digest(turn["initial_rng"]))
        self.manifest["turns"].append(row)
        self._write_manifest()
        self.active = None
        print(f"[onset-probe] turn={row['turn']} seconds={row['duration_sec']:.1f} reason={reason} data={path}", flush=True)

    def close(self):
        self._finish("inference_ended")
        self.manifest["status"] = "complete"
        self._write_manifest()
