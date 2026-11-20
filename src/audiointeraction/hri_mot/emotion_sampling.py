"""Equal per-emotion quotas over all annotated episodes, with DDP sharding."""

from __future__ import annotations

from typing import Iterator, List

import torch
from torch.utils.data import Dataset, Sampler

from .emotion_conditioning import EMOTION_LABELS, emotion_label


class DistributedBalancedEmotionSampler(Sampler[int]):
    """Sample every semantic class equally without inspecting motion magnitude."""

    def __init__(self, dataset: Dataset, *, samples_per_class: int = 512,
                 num_replicas: int = 1, rank: int = 0, seed: int = 0) -> None:
        self.samples_per_class = int(samples_per_class)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.global_num_samples = len(EMOTION_LABELS) * self.samples_per_class
        if self.global_num_samples % self.num_replicas:
            raise ValueError("Six-class sample count must be divisible by num_replicas")
        self.num_samples = self.global_num_samples // self.num_replicas
        records = getattr(dataset, "records", None)
        if not isinstance(records, list) or len(records) != len(dataset):
            raise TypeError("Emotion sampling requires an aligned dataset.records list")
        self.class_indices = {label: [] for label in EMOTION_LABELS}
        for index, record in enumerate(records):
            label = emotion_label(record.get("target_emotion"), source=f"record {index}")
            self.class_indices[label].append(index)
        missing = [label for label, indices in self.class_indices.items() if not indices]
        if missing:
            raise ValueError(f"No annotated episodes for emotion categories {missing}")
        self.source_class_counts = {
            label: len(indices) for label, indices in self.class_indices.items()
        }
        self.eligible_class_counts = dict(self.source_class_counts)
        self.label_quotas = {label: self.samples_per_class for label in EMOTION_LABELS}
        self.effective_repetitions_per_epoch = {
            label: self.samples_per_class / len(indices)
            for label, indices in self.class_indices.items()
        }

    @staticmethod
    def _draw(population: List[int], quota: int, *, generator: torch.Generator) -> List[int]:
        sampled = []
        while len(sampled) < quota:
            offsets = torch.randperm(len(population), generator=generator).tolist()
            remaining = quota - len(sampled)
            sampled.extend(population[offset] for offset in offsets[:remaining])
        return sampled

    def global_indices_for_epoch(self) -> List[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sampled = []
        for label in EMOTION_LABELS:
            sampled.extend(self._draw(self.class_indices[label], self.samples_per_class,
                                      generator=generator))
        order = torch.randperm(len(sampled), generator=generator).tolist()
        return [sampled[offset] for offset in order]

    def __iter__(self) -> Iterator[int]:
        return iter(self.global_indices_for_epoch()[self.rank::self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
