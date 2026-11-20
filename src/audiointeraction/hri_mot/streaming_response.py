"""Stateful response planning and 400 ms playback commitment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, List, Optional

from src.audiointeraction.dataset.TOKENS import KEEP_SILENCE, TEXT_BEGIN
from src.audiointeraction.hri_mot.streaming_language import (
    CommittedTextFragment,
    PlannedResponse,
    commit_keep_silence,
    commit_playback_fragment,
    commit_response_end,
    plan_response_without_committing,
)
from src.audiointeraction.hri_mot.streaming_tts import (
    ResponsePlaybackTimeline,
    SynthesizedUtterance,
    TTSClient,
)


@dataclass(frozen=True)
class ResponseStepResult:
    control: str
    planned_response: Optional[PlannedResponse]
    committed_fragment: Optional[CommittedTextFragment]
    utterance_index: Optional[int]
    playback_remaining_chunks: int
    interrupted: bool = False
    interrupted_after_onset: bool = False


class StreamingResponseController:
    """Keep speculative full text/TTS separate from action-visible text KV."""

    def __init__(
        self,
        *,
        tts_client: TTSClient,
        chunk_size_sec: float,
        language: str,
        speaker: str,
        instruct: str,
        emotion_token_to_text: Dict[int, str],
        commit_emotion_token_to_action_context: bool = True,
        measure_stage: Optional[
            Callable[[str], ContextManager[None]]
        ] = None,
    ) -> None:
        self.tts_client = tts_client
        self.chunk_size_sec = float(chunk_size_sec)
        self.language = str(language)
        self.speaker = str(speaker)
        self.instruct = str(instruct)
        self.emotion_token_to_text = dict(emotion_token_to_text)
        # The generated semantic emotion token is part of AIM's response
        # history. Callers normally keep this enabled so future AIM logits see
        # exactly the sequence AIM generated.
        self.commit_emotion_token_to_action_context = bool(
            commit_emotion_token_to_action_context
        )
        self.measure_stage = measure_stage or (lambda _name: nullcontext())
        self.timeline: Optional[ResponsePlaybackTimeline] = None
        self.current_plan: Optional[PlannedResponse] = None
        self.current_utterance_index: Optional[int] = None
        self.current_response_started = False
        self.utterances: List[SynthesizedUtterance] = []

    @property
    def active(self) -> bool:
        return self.timeline is not None and self.timeline.has_pending

    def action_speak_label(self) -> int:
        """Inference proxy for training target_response_state[c+1]."""

        return int(self.active)

    def action_response_elapsed_chunk(self) -> int:
        """Response-relative index for the next action interval, else -1."""

        if not self.active or self.timeline is None:
            return -1
        return self.timeline.next_index

    def action_emotion_token_id(self) -> Optional[int]:
        """Emotion planned for the next speaking action interval."""
        if not self.active or self.current_plan is None:
            return None
        if self.current_plan.emotion_token_id is None:
            raise RuntimeError("Planned response is missing its required emotion token")
        return int(self.current_plan.emotion_token_id)

    def advance_before_action(
        self,
        *,
        session,
        chunk_idx: int,
        control_token: int,
        response_was_active: bool,
        max_response_tokens: int,
        interrupt_response: bool = False,
    ) -> ResponseStepResult:
        """Commit text/control[c], then leave playback[c+1] available to SpeakQuery_c."""

        if interrupt_response:
            if not response_was_active or not self.active:
                raise AssertionError("Barge-in requested without an active response")
            utterance_index = self.current_utterance_index
            interrupted_after_onset = bool(self.current_response_started)
            if interrupted_after_onset:
                committed = commit_response_end(session, chunk_idx=chunk_idx)
                control = "response_interrupted"
            else:
                # Speculative full-text generation has been rolled back and no
                # TEXT_BEGIN is action-visible yet. Commit this partner-audio
                # interval as silent instead of introducing an unmatched END.
                commit_keep_silence(session, chunk_idx=chunk_idx)
                committed = None
                control = "response_cancelled_before_onset"
            self.timeline = None
            self.current_plan = None
            self.current_utterance_index = None
            self.current_response_started = False
            return ResponseStepResult(
                control=control,
                planned_response=None,
                committed_fragment=committed,
                utterance_index=utterance_index,
                playback_remaining_chunks=0,
                interrupted=True,
                interrupted_after_onset=interrupted_after_onset,
            )

        if response_was_active:
            if self.timeline is None or self.current_plan is None or not self.timeline.has_pending:
                raise AssertionError("Active response has no playback timeline or plan")
            playback_fragment = self.timeline.pop()
            committed = commit_playback_fragment(
                session,
                chunk_idx=chunk_idx,
                fragment=playback_fragment,
                emotion_token_id=(
                    self.current_plan.emotion_token_id
                    if playback_fragment.onset
                    and self.commit_emotion_token_to_action_context
                    else None
                ),
            )
            self.current_response_started = True
            remaining = self.timeline.remaining
            utterance_index = self.current_utterance_index
            control = "response" if playback_fragment.onset else "speaking"
            if not self.timeline.has_pending:
                self.timeline = None
                self.current_plan = None
                self.current_utterance_index = None
                self.current_response_started = False
            return ResponseStepResult(
                control=control,
                planned_response=None,
                committed_fragment=committed,
                utterance_index=utterance_index,
                playback_remaining_chunks=remaining,
            )

        planned: Optional[PlannedResponse] = None
        utterance_index: Optional[int] = None
        if int(control_token) == TEXT_BEGIN:
            with self.measure_stage("text_decode"):
                planned = plan_response_without_committing(
                    session,
                    chunk_idx=chunk_idx + 1,
                    max_tokens=max_response_tokens,
                    emotion_token_to_text=self.emotion_token_to_text,
                )
            if not planned.spoken_text:
                raise RuntimeError("TEXT_BEGIN produced no speakable response text for TTS")
            with self.measure_stage("tts_synthesis"):
                utterance = self.tts_client.synthesize(
                    planned.spoken_text,
                    language=self.language,
                    speaker=self.speaker,
                    instruct=self.instruct,
                )
            self.utterances.append(utterance)
            utterance_index = len(self.utterances) - 1
            self.current_utterance_index = utterance_index
            self.current_plan = planned
            self.current_response_started = False
            self.timeline = ResponsePlaybackTimeline(
                utterance,
                chunk_size_sec=self.chunk_size_sec,
            )
            control = "response_planned"
        elif int(control_token) == KEEP_SILENCE:
            control = "keep_silence"
        else:
            control = "unexpected"

        # No robot speech was playing in I_c.  Even when TEXT_BEGIN has just
        # planned speech for I_(c+1), the action-visible control for I_c is KEEP.
        commit_keep_silence(session, chunk_idx=chunk_idx)
        return ResponseStepResult(
            control=control,
            planned_response=planned,
            committed_fragment=None,
            utterance_index=utterance_index,
            playback_remaining_chunks=(self.timeline.remaining if self.timeline is not None else 0),
        )
