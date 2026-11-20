from dataclasses import dataclass, field
from typing import List, Optional


TRAIN_INDEX_PATHS = [
    "data/episodes/train_index.jsonl",
]
TEST_INDEX_PATH = "data/episodes/test_index.jsonl"

PARTNER_SILENT = 0
PARTNER_SPEAKING = 1

PARTNER_STATE_TO_ID = {
    "non_speech": PARTNER_SILENT,
    "onset": PARTNER_SPEAKING,
    "speaking": PARTNER_SPEAKING,
}

PARTNER_STATE_ID_TO_TEXT = {
    PARTNER_SILENT: "partner_state: silent",
    PARTNER_SPEAKING: "partner_state: speaking",
}

RESPONSE_SILENT = PARTNER_SILENT
RESPONSE_SPEAKING = PARTNER_SPEAKING
RESPONSE_STATE_TO_ID = PARTNER_STATE_TO_ID
RESPONSE_STATE_ID_TO_TEXT = {
    RESPONSE_SILENT: "response_state: silent",
    RESPONSE_SPEAKING: "response_state: speaking",
}

CONTEXT_TOKEN_SYSTEM = 0
CONTEXT_TOKEN_PARTNER_AUDIO = 1
CONTEXT_TOKEN_TARGET_CONTROL = 2
CONTEXT_TOKEN_TARGET_TEXT = 3

TEXT_CHUNK_TIME_MODES = ("midpoint", "start")
DEFAULT_TEXT_CHUNK_TIME_MODE = "midpoint"
AUDIO_LOADERS = ("native", "whisper")

VALID_QPOS_DIMS = [
    # MotionData S1 qpos layout after dropping the CSV ``Frame`` column:
    #   0:3   chassis (fixed; not predicted)
    #   3:16  torso + head + left arm
    #   16:22 left gripper (fixed zeros; not predicted)
    #   22:29 right arm
    #   29:35 right gripper (fixed zeros; not predicted)
    *range(3, 16),
    *range(22, 29),
]


@dataclass
class HriMotDataConfig:
    train_index_paths: List[str] = field(default_factory=lambda: list(TRAIN_INDEX_PATHS))
    test_index_path: str = TEST_INDEX_PATH

    chunk_size_sec: float = 0.4
    episode_length_sec: float = 20.0
    num_chunks_per_episode: int = 50

    audio_sample_rate: int = 16000
    audio_mono: bool = True
    audio_loader: str = "native"

    robot_fps: int = 30
    robot_steps_per_chunk: int = 12
    robot_source_key: str = "qpos_list"
    robot_valid_qpos_indices: List[int] = field(default_factory=lambda: list(VALID_QPOS_DIMS))
    robot_state_dim: int = 20
    action_dim: int = 20
    action_horizon: int = 12

    text_begin_token: str = "<TEXT_BEGIN>"
    text_end_token: str = "<TEXT_END>"
    text_chunk_time_mode: str = DEFAULT_TEXT_CHUNK_TIME_MODE

    strict_missing_files: bool = False
    skip_malformed_episodes: bool = True
    strict_unknown_partner_state: bool = False
    include_partner_state: bool = True
    include_text_fields: bool = True

    # Useful for quick smoke tests on large JSONL indices. None means no cap.
    max_episodes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.audio_loader not in AUDIO_LOADERS:
            raise ValueError(
                f"audio_loader must be one of {AUDIO_LOADERS}, got {self.audio_loader!r}"
            )
        if self.text_chunk_time_mode not in TEXT_CHUNK_TIME_MODES:
            raise ValueError(
                f"text_chunk_time_mode must be one of {TEXT_CHUNK_TIME_MODES}, "
                f"got {self.text_chunk_time_mode!r}"
            )
