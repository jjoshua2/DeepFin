"""Uniform physical-row replay for quiescent, finite learning generations.

The caller admits already authenticated, resolved-game chunks and changes the
window only between completed Trainer calls. This is a sampler, not a verifier
of outcome provenance, source hashes, or applied optimizer-update counts.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .buffer import ArrayReplayBuffer


@dataclass(frozen=True)
class GenerationChunk:
    source_sha256: str
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class DrawnRow:
    generation: int
    source_sha256: str
    row_index: int


class GenerationReplay:
    """Keep whole generations; draw uniformly with replacement over their rows.

    Source arrays must remain immutable for the lifetime of the window. Refills
    preserve the SAME RNG object, which Trainer also uses for augmentation.
    ``on_draw`` records sampled physical identities before augmentation. A
    prefetched or rejected batch is still a draw, not proof of an applied view.
    This direct finite-caller adapter is not a Worker/YAML replay backend.
    """

    def __init__(
        self, *, generations: int, rng: np.random.Generator,
        on_draw: Callable[[tuple[DrawnRow, ...]], None] | None = None,
    ) -> None:
        if type(generations) is not int or generations <= 0:
            raise ValueError("generations must be a positive integer")
        self.rng = rng
        self._limit = generations
        self._on_draw = on_draw
        self._window: tuple[tuple[int, tuple[GenerationChunk, ...]], ...] = ()
        self._seen_sources: set[str] = set()
        self._last_generation = 0
        self._buffer = ArrayReplayBuffer(0, rng=rng)
        self._ranges: list[tuple[int, GenerationChunk, int]] = []
        self._ends = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def membership(self) -> tuple[tuple[int, str, int], ...]:
        """Ordered generation/source/row-count roster of the entire window."""
        return tuple(
            (generation, chunk.source_sha256, int(chunk.arrays["x"].shape[0]))
            for generation, chunks in self._window for chunk in chunks
        )

    def admit_generation(self, generation: int, chunks: Sequence[GenerationChunk]) -> None:
        if type(generation) is not int or generation != self._last_generation + 1:
            raise ValueError("generation must be the next consecutive positive integer")
        chunks = tuple(chunks)
        if not chunks:
            raise ValueError("generation must contain nonempty admitted chunks")
        sources: set[str] = set()
        for chunk in chunks:
            if re.fullmatch(r"[0-9a-f]{64}", chunk.source_sha256) is None:
                raise ValueError("source_sha256 must be a lowercase SHA256 identity")
            if chunk.source_sha256 in sources or chunk.source_sha256 in self._seen_sources:
                raise ValueError("source chunk was already admitted")
            if chunk.arrays["x"].ndim == 0 or chunk.arrays["x"].shape[0] <= 0:
                raise ValueError("generation chunk must contain rows")
            sources.add(chunk.source_sha256)
        window = (*self._window, (generation, chunks))[-self._limit:]
        ranges: list[tuple[int, GenerationChunk, int]] = []
        ends: list[int] = []
        count = 0
        for number, generation_chunks in window:
            for chunk in generation_chunks:
                ranges.append((number, chunk, count))
                count += int(chunk.arrays["x"].shape[0])
                ends.append(count)
        # Build before publishing membership. An invalid chunk cannot clear a
        # previously usable window or consume its sampling/augmentation RNG.
        buffer = ArrayReplayBuffer(count, rng=self.rng)
        for _, chunk, _ in ranges:
            buffer.add_many_arrays(chunk.arrays)
        if len(buffer) != count:
            raise ValueError("generation refill lost rows")
        self._buffer = buffer
        self._window = window
        self._ranges = ranges
        self._ends = np.asarray(ends, dtype=np.int64)
        self._last_generation = generation
        self._seen_sources.update(sources)

    def sample_batch_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not len(self):
            raise ValueError("generation replay is empty")
        # Match ArrayReplayBuffer's uniform-with-replacement draw. Explicitly
        # bypass both surprise priorities and its default WDL-class quotas.
        indices = self.rng.choice(len(self), size=batch_size, replace=True)
        arrays = self._buffer._gather_rows(indices)
        if self._on_draw is not None:
            locations = np.searchsorted(self._ends, indices, side="right")
            rows = []
            for index, location in zip(indices, locations, strict=True):
                generation, chunk, start = self._ranges[int(location)]
                rows.append(DrawnRow(generation, chunk.source_sha256, int(index) - start))
            self._on_draw(tuple(rows))
        return arrays
