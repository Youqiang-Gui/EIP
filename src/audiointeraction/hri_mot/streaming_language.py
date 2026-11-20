"""Language planning helpers for causal HRI action inference.

Full response generation is speculative: it may write temporary KV entries,
but session metadata is restored and those positions are overwritten by
playback-confirmed fragments.  The action-visible context therefore advances
only with the 400 ms playback timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.audiointeraction.dataset.TOKENS import (
    KEEP_SILENCE,
    TEXT_BEGIN,
    TEXT_END,
)
from src.audiointeraction.hri_mot.data_config import (
    CONTEXT_TOKEN_TARGET_CONTROL,
    CONTEXT_TOKEN_TARGET_TEXT,
)
from src.audiointeraction.hri_mot.streaming_tts import PlaybackFragment


@dataclass(frozen=True)
class PlannedResponse:
    generated_token_ids: List[int]
    spoken_text: str
    display_text: str
    emotion_token_id: Optional[int]
    completed: bool


@dataclass(frozen=True)
class CommittedTextFragment:
    token_ids: List[int]
    text: str
    onset: bool
    end: bool
    pcm_start_sample: int
    pcm_end_sample: int
    sample_rate: int


@dataclass(frozen=True)
class ReplyGateResult:
    control_token: int
    reply_pending: bool
    reply_allowed: bool
    reply_deferred: bool
    reply_pending_released: bool
    reply_probability_deferred: bool


def _clone_optional_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return value.clone() if value is not None else None


def replace_pending_control_token(session: Any, *, token_id: int) -> None:
    """Replace an uncommitted sampled control without advancing the KV cursor."""

    if session.pending_ids.numel() != 1:
        raise RuntimeError(
            "Pending control replacement requires exactly one sampled token, "
            f"got {session.pending_ids.numel()}"
        )
    if (
        len(session.pending_token_types) != 1
        or int(session.pending_token_types[0]) != CONTEXT_TOKEN_TARGET_CONTROL
    ):
        raise RuntimeError(
            "Pending control replacement requires one target-control token"
        )
    session.pending_ids = torch.tensor(
        [int(token_id)],
        dtype=torch.long,
        device=session.device,
    )


def apply_reply_silence_gate(
    session: Any,
    *,
    sampled_control_token: int,
    reply_pending: bool,
    reply_allowed: bool,
    text_begin_probability: Optional[float] = None,
    deferred_probability_threshold: float = 0.5,
    allow_probability_defer: bool = False,
) -> ReplyGateResult:
    """Defer an early TEXT_BEGIN and release it after the VAD silence gate.

    Audio and the previous KEEP_SILENCE decisions remain committed normally.
    Only the current not-yet-forwarded control token is replaced, matching the
    deferred-reply behavior in ``web/infer_wav.py``.
    """

    sampled_control_token = int(sampled_control_token)
    pending = bool(reply_pending)
    deferred = False
    released = False
    probability_deferred = False
    control_token = sampled_control_token

    if sampled_control_token == TEXT_BEGIN and not reply_allowed:
        pending = True
        deferred = True
        control_token = KEEP_SILENCE
        replace_pending_control_token(session, token_id=KEEP_SILENCE)
    elif reply_allowed and pending:
        pending = False
        released = True
        control_token = TEXT_BEGIN
        replace_pending_control_token(session, token_id=TEXT_BEGIN)
    elif sampled_control_token == TEXT_BEGIN:
        pending = False
    elif (
        allow_probability_defer
        and not reply_allowed
        and not pending
        and text_begin_probability is not None
        and float(text_begin_probability) >= float(deferred_probability_threshold)
    ):
        # Keep the sampled silence in the causal context, but preserve this
        # high-confidence reply opportunity until the silence gate opens.
        pending = True
        deferred = True
        probability_deferred = True

    return ReplyGateResult(
        control_token=int(control_token),
        reply_pending=bool(pending),
        reply_allowed=bool(reply_allowed),
        reply_deferred=bool(deferred),
        reply_pending_released=bool(released),
        reply_probability_deferred=bool(probability_deferred),
    )


def plan_response_without_committing(
    session: Any,
    *,
    chunk_idx: int,
    max_tokens: int,
    emotion_token_to_text: Dict[int, str],
) -> PlannedResponse:
    """Generate a complete response, then restore the action-visible cursor."""

    snapshot = {
        "pending_ids": session.pending_ids.clone(),
        "pending_pos": session.pending_pos.clone(),
        "pending_chunk_ids": list(session.pending_chunk_ids),
        "pending_token_types": list(session.pending_token_types),
        "context_chunk_ids": list(session.context_chunk_ids),
        "context_token_types": list(session.context_token_types),
        "pcm_history": list(session.pcm_history),
        "last_sample_debug": dict(session.last_sample_debug),
        "kv_context_length": int(session.kv_context_length),
        "input_pos_maxp1": _clone_optional_tensor(session.input_pos_maxp1),
    }
    if session.pending_ids.numel() != 1 or int(session.pending_ids[0].item()) != TEXT_BEGIN:
        raise RuntimeError(
            "Speculative response planning requires one pending TEXT_BEGIN token"
        )
    try:
        generated, _, completed = session.generate_response_text(
            chunk_idx=chunk_idx,
            max_tokens=max_tokens,
        )
    finally:
        session.pending_ids = snapshot["pending_ids"]
        session.pending_pos = snapshot["pending_pos"]
        session.pending_chunk_ids = snapshot["pending_chunk_ids"]
        session.pending_token_types = snapshot["pending_token_types"]
        session.context_chunk_ids = snapshot["context_chunk_ids"]
        session.context_token_types = snapshot["context_token_types"]
        session.pcm_history = snapshot["pcm_history"]
        session.last_sample_debug = snapshot["last_sample_debug"]
        session.kv_context_length = snapshot["kv_context_length"]
        session.input_pos_maxp1 = snapshot["input_pos_maxp1"]

    emotion_token_id: Optional[int] = None
    text_ids: List[int] = []
    for token_id in generated:
        token_id = int(token_id)
        if token_id in (TEXT_BEGIN, TEXT_END, KEEP_SILENCE):
            continue
        if (
            emotion_token_id is None
            and not text_ids
            and token_id in emotion_token_to_text
        ):
            emotion_token_id = token_id
            continue
        text_ids.append(token_id)
    spoken_text = (
        session.tokenizer.decode(torch.tensor(text_ids, dtype=torch.long))
        if text_ids
        else ""
    ).strip()
    emotion = (
        emotion_token_to_text.get(emotion_token_id, "")
        if emotion_token_id is not None
        else ""
    )
    display_text = f"{emotion} {spoken_text}".strip()
    return PlannedResponse(
        generated_token_ids=[TEXT_BEGIN] + [int(item) for item in generated],
        spoken_text=spoken_text,
        display_text=display_text,
        emotion_token_id=emotion_token_id,
        completed=bool(completed),
    )


def _replace_pending_and_forward(
    session: Any,
    *,
    token_ids: Sequence[int],
    chunk_idx: int,
    token_types: Sequence[int],
) -> None:
    if len(token_ids) != len(token_types):
        raise ValueError("token_ids and token_types must have equal length")
    if not token_ids:
        # Discard a sampled KEEP_SILENCE placeholder while an existing response
        # is speaking but has no text unit starting in this 400 ms interval.
        session.pending_ids = torch.empty(0, dtype=torch.long, device=session.device)
        session.pending_pos = torch.empty(0, dtype=torch.int64, device=session.device)
        session.pending_chunk_ids = []
        session.pending_token_types = []
        if session.input_pos_maxp1 is not None:
            session.input_pos_maxp1.fill_(int(session.kv_context_length))
        return

    session._ensure_capacity(len(token_ids))
    start_pos = int(session.kv_context_length)
    ids = torch.tensor(token_ids, dtype=torch.long, device=session.device)
    positions = torch.arange(
        start_pos,
        start_pos + len(token_ids),
        dtype=torch.int64,
        device=session.device,
    )
    input_pos_maxp1 = (
        torch.tensor(start_pos + len(token_ids), device=session.device)
        if session.input_pos_maxp1 is not None
        else None
    )
    session.base_model(
        ids.view(1, -1),
        None,
        1,
        None,
        positions,
        input_pos_maxp1=input_pos_maxp1,
        audio_tokens_per_chunk=10,
    )
    session.context_chunk_ids.extend([int(chunk_idx)] * len(token_ids))
    session.context_token_types.extend(int(item) for item in token_types)
    session.kv_context_length += len(token_ids)
    session.pending_ids = torch.empty(0, dtype=torch.long, device=session.device)
    session.pending_pos = torch.empty(0, dtype=torch.int64, device=session.device)
    session.pending_chunk_ids = []
    session.pending_token_types = []
    if session.input_pos_maxp1 is not None:
        session.input_pos_maxp1.fill_(int(session.kv_context_length))


def commit_keep_silence(session: Any, *, chunk_idx: int) -> None:
    """Commit the current physical interval as a silent target turn."""

    _replace_pending_and_forward(
        session,
        token_ids=[KEEP_SILENCE],
        chunk_idx=chunk_idx,
        token_types=[CONTEXT_TOKEN_TARGET_CONTROL],
    )


def commit_response_end(session: Any, *, chunk_idx: int) -> CommittedTextFragment:
    """Close an action-visible response at a 400 ms barge-in boundary."""

    _replace_pending_and_forward(
        session,
        token_ids=[TEXT_END],
        chunk_idx=chunk_idx,
        token_types=[CONTEXT_TOKEN_TARGET_TEXT],
    )
    return CommittedTextFragment(
        token_ids=[TEXT_END],
        text="",
        onset=False,
        end=True,
        pcm_start_sample=0,
        pcm_end_sample=0,
        sample_rate=0,
    )



def commit_playback_fragment(
    session: Any,
    *,
    chunk_idx: int,
    fragment: PlaybackFragment,
    emotion_token_id: Optional[int],
) -> CommittedTextFragment:
    """Commit exactly one playback interval to the action-visible text KV."""

    token_ids: List[int] = []
    token_types: List[int] = []
    if fragment.onset:
        token_ids.append(TEXT_BEGIN)
        token_types.append(CONTEXT_TOKEN_TARGET_CONTROL)
        if emotion_token_id is not None:
            token_ids.append(int(emotion_token_id))
            token_types.append(CONTEXT_TOKEN_TARGET_TEXT)
    if fragment.text:
        text_ids = session.tokenizer.encode(
            fragment.text, bos=False, eos=False
        ).tolist()
        token_ids.extend(int(item) for item in text_ids)
        token_types.extend([CONTEXT_TOKEN_TARGET_TEXT] * len(text_ids))
    if fragment.end:
        token_ids.append(TEXT_END)
        token_types.append(CONTEXT_TOKEN_TARGET_TEXT)
    _replace_pending_and_forward(
        session,
        token_ids=token_ids,
        chunk_idx=chunk_idx,
        token_types=token_types,
    )
    return CommittedTextFragment(
        token_ids=token_ids,
        text=fragment.text,
        onset=fragment.onset,
        end=fragment.end,
        pcm_start_sample=int(fragment.start_sample),
        pcm_end_sample=int(fragment.end_sample),
        sample_rate=int(fragment.sample_rate),
    )
