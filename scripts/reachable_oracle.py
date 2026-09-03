"""Exact hindsight allocation for nested stop/continue trajectories.

For a trajectory with ``T`` continuation stages, assigning a position a stop
depth ``d`` means selecting exactly its first ``d`` stages.  Nonincreasing
per-stage continuation counts therefore induce exact capacities for the
``T + 1`` possible stop depths.  This module solves that capacitated assignment
problem exactly, while keeping the implementation independent of SciPy,
NetworkX, or another optimization package.

The solver is a successive-shortest-augmenting-path min-cost flow specialized
to the small number of stop-depth bins.  Residual moves between two bins are
maintained in lazy heaps.  With ``n`` positions and ``K = T + 1`` bins, runtime
is ``O(n K^3 + n K^2 log(nK))`` and heap storage is ``O(n K^2)``.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real


@dataclass(frozen=True)
class ReachableOracleResult:
    """An exact nested allocation, with positions indexed as supplied.

    Selected indices and keys are ordered lexicographically by key within each
    stage.  ``assigned_depths[i]`` is the number of stages selected for input
    position ``i``.
    """

    objective: float
    assigned_depths: tuple[int, ...]
    selected_indices_by_stage: tuple[tuple[int, ...], ...]
    selected_keys_by_stage: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _AugmentPath:
    cost: float
    categories: tuple[int, ...]
    moved_items: tuple[int, ...]


def _validated_inputs(
    keys: Sequence[str],
    gains: Sequence[Sequence[object]],
    stage_counts: Sequence[int],
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...], tuple[int, ...]]:
    def require_key(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("position keys must be nonempty strings")
        return value

    raw_keys = tuple(keys)
    if not raw_keys:
        raise ValueError("reachable oracle requires at least one position")
    normalized_keys = tuple(require_key(key) for key in raw_keys)
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("position keys must be unique")

    counts_raw = tuple(stage_counts)
    if not counts_raw:
        raise ValueError("reachable oracle requires at least one stage")
    counts: list[int] = []
    for count in counts_raw:
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError("stage counts must be integers")
        normalized = int(count)
        if not 0 <= normalized <= len(normalized_keys):
            raise ValueError("stage counts must lie between zero and the position count")
        counts.append(normalized)
    if any(later > earlier for earlier, later in pairwise(counts)):
        raise ValueError("stage counts must be nonincreasing")

    gain_rows = tuple(gains)
    if len(gain_rows) != len(normalized_keys):
        raise ValueError("gains must contain exactly one row per position key")
    normalized_gains: list[tuple[float, ...]] = []
    for position, row in enumerate(gain_rows):
        try:
            raw_values = tuple(row)
        except TypeError as exc:
            raise ValueError(f"gain row {position} must be a sequence") from exc
        if len(raw_values) != len(counts):
            raise ValueError(
                f"gain row {position} must contain exactly {len(counts)} stages"
            )
        values: list[float] = []
        for stage, value in enumerate(raw_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"gain row {position}, stage {stage} must be a finite real number"
                )
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(
                    f"gain row {position}, stage {stage} must be a finite real number"
                )
            values.append(normalized)
        normalized_gains.append(tuple(values))
    return normalized_keys, tuple(normalized_gains), tuple(counts)


def _depth_values(gains: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    values: list[tuple[float, ...]] = []
    for position, row in enumerate(gains):
        cumulative = [0.0]
        for stage, gain in enumerate(row):
            total = cumulative[-1] + gain
            if not math.isfinite(total):
                raise ValueError(
                    f"cumulative gain overflows at position {position}, stage {stage}"
                )
            cumulative.append(total)
        for left in cumulative:
            if any(not math.isfinite(left - right) for right in cumulative):
                raise ValueError(
                    f"residual assignment cost overflows at position {position}"
                )
        values.append(tuple(cumulative))
    return tuple(values)


def _depth_capacities(position_count: int, counts: Sequence[int]) -> tuple[int, ...]:
    capacities = [position_count - counts[0]]
    capacities.extend(
        counts[stage - 1] - counts[stage] for stage in range(1, len(counts))
    )
    capacities.append(counts[-1])
    if any(capacity < 0 for capacity in capacities) or sum(capacities) != position_count:
        raise AssertionError("invalid stop-depth capacities")
    return tuple(capacities)


def _path_tie_key(path: _AugmentPath, keys: Sequence[str]) -> tuple[object, ...]:
    return (
        path.categories,
        tuple(keys[item] for item in path.moved_items),
    )


def solve_reachable_oracle(
    keys: Sequence[str],
    gains: Sequence[Sequence[object]],
    stage_counts: Sequence[int],
) -> ReachableOracleResult:
    """Maximize signed gain under exact, nested continuation counts.

    ``gains[i][t]`` is the signed marginal gain from continuing position ``i``
    through stage ``t``.  ``stage_counts[t]`` is the exact number of positions
    which must receive that stage.  Counts must be nonincreasing because a
    stopped position cannot re-enter at a later stage.

    Ties are resolved from position keys and stop-depth paths, so selected key
    sets do not depend on the caller's input ordering.
    """
    normalized_keys, normalized_gains, counts = _validated_inputs(
        keys, gains, stage_counts,
    )
    values = _depth_values(normalized_gains)
    capacities = _depth_capacities(len(normalized_keys), counts)
    category_count = len(capacities)
    owner = [-1] * len(normalized_keys)
    members: list[set[int]] = [set() for _ in capacities]
    switch_heaps: list[list[list[tuple[float, str, int]]]] = [
        [[] for _ in capacities] for _ in capacities
    ]

    def push_switches(item: int, source: int) -> None:
        for target in range(category_count):
            if target == source:
                continue
            loss = values[item][source] - values[item][target]
            heapq.heappush(
                switch_heaps[source][target],
                (loss, normalized_keys[item], item),
            )

    def enter(item: int, category: int) -> None:
        owner[item] = category
        members[category].add(item)
        push_switches(item, category)

    def best_switch(source: int, target: int) -> tuple[float, int] | None:
        heap = switch_heaps[source][target]
        while heap and owner[heap[0][2]] != source:
            heapq.heappop(heap)
        if not heap:
            return None
        loss, _, item = heap[0]
        return loss, item

    processing_order = sorted(
        range(len(normalized_keys)), key=lambda item: normalized_keys[item],
    )
    for item in processing_order:
        best = [
            _AugmentPath(-values[item][category], (category,), ())
            for category in range(category_count)
        ]
        # An improving residual cycle would contradict optimality of the
        # assignment maintained for the already-processed positions.  Thus a
        # shortest augmenting path can be simple and use at most K-1 switches.
        for _ in range(category_count - 1):
            previous = tuple(best)
            changed = False
            for source, prefix in enumerate(previous):
                for target in range(category_count):
                    if target == source or target in prefix.categories:
                        continue
                    switch = best_switch(source, target)
                    if switch is None:
                        continue
                    loss, moved_item = switch
                    candidate_cost = prefix.cost + loss
                    if not math.isfinite(candidate_cost):
                        raise ValueError("augmenting-path cost overflowed")
                    candidate = _AugmentPath(
                        candidate_cost,
                        (*prefix.categories, target),
                        (*prefix.moved_items, moved_item),
                    )
                    incumbent = best[target]
                    if (
                        candidate.cost < incumbent.cost
                        or (
                            candidate.cost == incumbent.cost
                            and _path_tie_key(candidate, normalized_keys)
                            < _path_tie_key(incumbent, normalized_keys)
                        )
                    ):
                        best[target] = candidate
                        changed = True
            if not changed:
                break

        available = [
            category for category, capacity in enumerate(capacities)
            if len(members[category]) < capacity
        ]
        if not available:
            raise AssertionError("no residual stop-depth capacity remains")
        end = min(
            available,
            key=lambda category: (
                best[category].cost,
                _path_tie_key(best[category], normalized_keys),
            ),
        )
        path = best[end]
        for step in range(len(path.moved_items) - 1, -1, -1):
            moved_item = path.moved_items[step]
            source = path.categories[step]
            target = path.categories[step + 1]
            if owner[moved_item] != source:
                raise AssertionError("augmenting path reuses a moved position")
            members[source].remove(moved_item)
            enter(moved_item, target)
        enter(item, path.categories[0])

    if any(depth < 0 for depth in owner):
        raise AssertionError("reachable oracle left a position unassigned")
    if tuple(len(group) for group in members) != capacities:
        raise AssertionError("reachable oracle did not fill exact stop-depth capacities")

    selected_indices: list[tuple[int, ...]] = []
    selected_keys: list[tuple[str, ...]] = []
    for stage, expected_count in enumerate(counts):
        indices = tuple(sorted(
            (index for index, depth in enumerate(owner) if depth > stage),
            key=lambda index: normalized_keys[index],
        ))
        if len(indices) != expected_count:
            raise AssertionError("reachable oracle violated a stage count")
        selected_indices.append(indices)
        selected_keys.append(tuple(normalized_keys[index] for index in indices))

    try:
        objective = math.fsum(
            values[index][depth] for index, depth in enumerate(owner)
        )
    except OverflowError as exc:
        raise ValueError("reachable-oracle objective overflowed") from exc
    if not math.isfinite(objective):
        raise ValueError("reachable-oracle objective overflowed")
    return ReachableOracleResult(
        objective=objective,
        assigned_depths=tuple(owner),
        selected_indices_by_stage=tuple(selected_indices),
        selected_keys_by_stage=tuple(selected_keys),
    )
