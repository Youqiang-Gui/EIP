"""TTS service types and a 400 ms text/PCM playback scheduler.

The action model never consumes TTS acoustic tokens.  TTS provides waveform
samples plus text-to-sample alignment; the interaction model commits only the
text whose corresponding waveform interval has reached the playback cursor.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np


_TEXT_UNIT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|"
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|"
    r"[^\w\s]",
    flags=re.UNICODE,
)
_STRONG_PUNCTUATION = set("。！？!?；;：:")
_WEAK_PUNCTUATION = set("，,、…—-")


@dataclass(frozen=True)
class TextAudioSpan:
    """One text unit aligned to a half-open PCM sample interval."""

    text_start: int
    text_end: int
    start_sample: int
    end_sample: int

    def validate(self, *, text_length: int, pcm_samples: int) -> None:
        if not 0 <= self.text_start < self.text_end <= text_length:
            raise ValueError(
                f"Invalid text span [{self.text_start}, {self.text_end}) "
                f"for text length {text_length}"
            )
        if not 0 <= self.start_sample <= self.end_sample <= pcm_samples:
            raise ValueError(
                f"Invalid audio span [{self.start_sample}, {self.end_sample}) "
                f"for PCM length {pcm_samples}"
            )

    def to_dict(self) -> Dict[str, int]:
        return {
            "text_start": int(self.text_start),
            "text_end": int(self.text_end),
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TextAudioSpan":
        return cls(
            text_start=int(value["text_start"]),
            text_end=int(value["text_end"]),
            start_sample=int(value["start_sample"]),
            end_sample=int(value["end_sample"]),
        )


@dataclass
class SynthesizedUtterance:
    text: str
    pcm: np.ndarray
    sample_rate: int
    spans: List[TextAudioSpan]
    backend: str
    alignment_method: str

    def __post_init__(self) -> None:
        self.text = str(self.text)
        self.pcm = np.asarray(self.pcm, dtype=np.float32).reshape(-1)
        self.sample_rate = int(self.sample_rate)
        self.backend = str(self.backend)
        self.alignment_method = str(self.alignment_method)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.pcm.size <= 0:
            raise ValueError("TTS returned an empty waveform")
        previous_text_end = 0
        previous_audio_end = 0
        for span in self.spans:
            span.validate(text_length=len(self.text), pcm_samples=int(self.pcm.size))
            if span.text_start < previous_text_end:
                raise ValueError("Text alignment spans overlap or are out of order")
            if span.start_sample < previous_audio_end:
                raise ValueError("Audio alignment spans overlap or are out of order")
            previous_text_end = span.text_end
            previous_audio_end = span.end_sample

    @property
    def duration_sec(self) -> float:
        return float(self.pcm.size / self.sample_rate)

    def to_transport_dict(self) -> Dict[str, Any]:
        pcm = np.ascontiguousarray(self.pcm.astype("<f4", copy=False))
        return {
            "version": 1,
            "text": self.text,
            "sample_rate": self.sample_rate,
            "pcm_f32le_base64": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "spans": [span.to_dict() for span in self.spans],
            "backend": self.backend,
            "alignment_method": self.alignment_method,
        }

    @classmethod
    def from_transport_dict(cls, value: Dict[str, Any]) -> "SynthesizedUtterance":
        raw = base64.b64decode(str(value["pcm_f32le_base64"]), validate=True)
        if len(raw) % np.dtype("<f4").itemsize:
            raise ValueError("TTS PCM payload is not valid float32 little-endian data")
        pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        return cls(
            text=str(value["text"]),
            pcm=pcm,
            sample_rate=int(value["sample_rate"]),
            spans=[TextAudioSpan.from_dict(item) for item in value.get("spans", [])],
            backend=str(value.get("backend", "unknown")),
            alignment_method=str(value.get("alignment_method", "unknown")),
        )


class TTSClient(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker: str,
        instruct: str = "",
    ) -> SynthesizedUtterance:
        ...


def _unit_weight(unit: str) -> float:
    if unit in _STRONG_PUNCTUATION:
        return 0.65
    if unit in _WEAK_PUNCTUATION:
        return 0.35
    if len(unit) == 1 and "\u3400" <= unit <= "\u9fff":
        return 1.0
    # A compact English syllable proxy. It is used only when a backend does
    # not provide forced-alignment timestamps.
    vowel_groups = re.findall(r"[aeiouy]+", unit.lower())
    return float(max(1, len(vowel_groups), math.ceil(len(unit) / 5)))


def estimate_text_audio_spans(text: str, pcm_samples: int) -> List[TextAudioSpan]:
    """Allocate text units over an existing waveform using linguistic weights.

    This is an explicit fallback, not a claimed forced alignment.  Backends may
    replace these spans with real character/word timestamps without changing
    the playback scheduler or action inference code.
    """

    text = str(text)
    pcm_samples = int(pcm_samples)
    matches = list(_TEXT_UNIT_RE.finditer(text))
    if not matches or pcm_samples <= 0:
        return []
    weights = np.asarray([_unit_weight(match.group(0)) for match in matches], dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    boundaries = np.rint(cumulative / cumulative[-1] * pcm_samples).astype(np.int64)
    boundaries[0] = 0
    boundaries[-1] = pcm_samples
    return [
        TextAudioSpan(
            text_start=int(match.start()),
            text_end=int(match.end()),
            start_sample=int(boundaries[index]),
            end_sample=int(boundaries[index + 1]),
        )
        for index, match in enumerate(matches)
    ]


class HttpTTSClient:
    """Dependency-free client for an HRI-compatible local TTS service."""

    def __init__(self, base_url: str, *, timeout_sec: float = 120.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0:
            raise ValueError("TTS timeout must be positive")
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))

    def health(self) -> Dict[str, Any]:
        with self._opener.open(
            f"{self.base_url}/health", timeout=self.timeout_sec
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker: str,
        instruct: str = "",
    ) -> SynthesizedUtterance:
        payload_value = {
            "text": str(text),
            "language": str(language),
            "instruct": str(instruct),
        }
        if str(speaker).strip():
            payload_value["speaker"] = str(speaker).strip()
        payload = json.dumps(payload_value, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}/v1/synthesize",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_sec) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS server returned HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach TTS server at {self.base_url}: {exc.reason}"
            ) from exc
        utterance = SynthesizedUtterance.from_transport_dict(value)
        if utterance.text != str(text):
            raise RuntimeError("TTS response text does not match the request text")
        return utterance


class EstimatedTTSClient:
    """Deterministic no-model backend for tests and explicit offline fallback."""

    def __init__(self, *, sample_rate: int = 24_000, units_per_second: float = 4.5) -> None:
        self.sample_rate = int(sample_rate)
        self.units_per_second = float(units_per_second)
        if self.sample_rate <= 0 or self.units_per_second <= 0:
            raise ValueError("Estimated TTS sample rate and units/second must be positive")

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker: str,
        instruct: str = "",
    ) -> SynthesizedUtterance:
        del language, speaker, instruct
        matches = list(_TEXT_UNIT_RE.finditer(str(text)))
        total_weight = sum(_unit_weight(match.group(0)) for match in matches)
        duration = max(0.12, total_weight / self.units_per_second)
        sample_count = max(1, int(round(duration * self.sample_rate)))
        pcm = np.zeros(sample_count, dtype=np.float32)
        return SynthesizedUtterance(
            text=str(text),
            pcm=pcm,
            sample_rate=self.sample_rate,
            spans=estimate_text_audio_spans(str(text), sample_count),
            backend="estimated/no_audio",
            alignment_method="weighted_text_units",
        )


@dataclass(frozen=True)
class PlaybackFragment:
    index: int
    text: str
    pcm: np.ndarray
    sample_rate: int
    start_sample: int
    end_sample: int
    onset: bool
    end: bool

    @property
    def speaking(self) -> bool:
        return self.end_sample > self.start_sample


class ResponsePlaybackTimeline:
    """Split one synthesized utterance into fixed-duration playback intervals."""

    def __init__(self, utterance: SynthesizedUtterance, *, chunk_size_sec: float) -> None:
        self.utterance = utterance
        self.chunk_size_sec = float(chunk_size_sec)
        if self.chunk_size_sec <= 0:
            raise ValueError("chunk_size_sec must be positive")
        self.samples_per_chunk = max(
            1, int(round(self.chunk_size_sec * utterance.sample_rate))
        )
        self._fragments = self._build_fragments()
        self._cursor = 0

    def _build_fragments(self) -> List[PlaybackFragment]:
        total_samples = int(self.utterance.pcm.size)
        chunk_count = max(1, int(math.ceil(total_samples / self.samples_per_chunk)))
        spans_by_chunk: List[List[TextAudioSpan]] = [[] for _ in range(chunk_count)]
        for span in self.utterance.spans:
            chunk_index = min(
                chunk_count - 1, max(0, int(span.start_sample // self.samples_per_chunk))
            )
            spans_by_chunk[chunk_index].append(span)

        fragments: List[PlaybackFragment] = []
        text_cursor = 0
        for index in range(chunk_count):
            start = index * self.samples_per_chunk
            end = min(total_samples, start + self.samples_per_chunk)
            pcm = np.zeros(self.samples_per_chunk, dtype=np.float32)
            pcm[: end - start] = self.utterance.pcm[start:end]
            bucket = spans_by_chunk[index]
            if bucket:
                text_end = max(span.text_end for span in bucket)
                if index == chunk_count - 1:
                    text_end = len(self.utterance.text)
                text = self.utterance.text[text_cursor:text_end]
                text_cursor = text_end
            elif index == chunk_count - 1 and text_cursor < len(self.utterance.text):
                text = self.utterance.text[text_cursor:]
                text_cursor = len(self.utterance.text)
            else:
                text = ""
            fragments.append(
                PlaybackFragment(
                    index=index,
                    text=text,
                    pcm=pcm,
                    sample_rate=self.utterance.sample_rate,
                    start_sample=start,
                    end_sample=end,
                    onset=index == 0,
                    end=index == chunk_count - 1,
                )
            )
        return fragments

    def __len__(self) -> int:
        return len(self._fragments)

    @property
    def remaining(self) -> int:
        return len(self._fragments) - self._cursor

    @property
    def next_index(self) -> int:
        """Zero-based response-relative index of the next playback interval."""

        return int(self._cursor)

    @property
    def has_pending(self) -> bool:
        return self._cursor < len(self._fragments)

    @property
    def next_interval_speaking(self) -> bool:
        return self.has_pending and self._fragments[self._cursor].speaking

    @property
    def full_pcm(self) -> np.ndarray:
        return self.utterance.pcm

    def pop(self) -> PlaybackFragment:
        if not self.has_pending:
            raise RuntimeError("Response playback timeline is exhausted")
        fragment = self._fragments[self._cursor]
        self._cursor += 1
        return fragment


def concatenate_utterance_pcm(
    utterances: Sequence[SynthesizedUtterance], *, gap_sec: float = 0.0
) -> tuple[np.ndarray, Optional[int]]:
    if not utterances:
        return np.zeros(0, dtype=np.float32), None
    sample_rate = int(utterances[0].sample_rate)
    if any(item.sample_rate != sample_rate for item in utterances):
        raise ValueError("Cannot concatenate TTS utterances with different sample rates")
    gap = np.zeros(max(0, int(round(gap_sec * sample_rate))), dtype=np.float32)
    pieces: List[np.ndarray] = []
    for index, item in enumerate(utterances):
        if index and gap.size:
            pieces.append(gap)
        pieces.append(item.pcm)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32), sample_rate
