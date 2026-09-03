from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import pytest

from scripts.reachable_oracle import ReachableOracleResult, solve_reachable_oracle


def _depth_capacities(position_count: int, counts: Sequence[int]) -> list[int]:
    return [
        position_count - counts[0],
        *(counts[stage - 1] - counts[stage] for stage in range(1, len(counts))),
        counts[-1],
    ]


def _objective(gains: Sequence[Sequence[float]], depths: Sequence[int]) -> float:
    return sum(
        sum(gains[position][:depth])
        for position, depth in enumerate(depths)
    )


def _brute_force_objective(
    gains: Sequence[Sequence[float]], counts: Sequence[int],
) -> float:
    capacities = _depth_capacities(len(gains), counts)
    best = -float("inf")
    for depths in itertools.product(range(len(counts) + 1), repeat=len(gains)):
        if [depths.count(depth) for depth in range(len(counts) + 1)] != capacities:
            continue
        best = max(best, _objective(gains, depths))
    return best


def _assert_feasible(
    result: ReachableOracleResult,
    keys: Sequence[str],
    gains: Sequence[Sequence[float]],
    counts: Sequence[int],
) -> None:
    assert len(result.assigned_depths) == len(keys)
    assert result.objective == pytest.approx(_objective(gains, result.assigned_depths))
    for stage, expected_count in enumerate(counts):
        indices = result.selected_indices_by_stage[stage]
        assert len(indices) == expected_count
        assert result.selected_keys_by_stage[stage] == tuple(keys[index] for index in indices)
        assert set(indices) == {
            index for index, depth in enumerate(result.assigned_depths)
            if depth > stage
        }
        if stage:
            assert set(indices) <= set(result.selected_indices_by_stage[stage - 1])


def test_independent_stage_oracle_can_be_unreachable() -> None:
    keys = ["early", "late"]
    gains = [[100.0, 0.0], [0.0, 100.0]]

    result = solve_reachable_oracle(keys, gains, [1, 1])
    independent_stage_bound = max(row[0] for row in gains) + max(row[1] for row in gains)

    assert independent_stage_bound == 200.0
    assert result.objective == 100.0
    assert result.selected_keys_by_stage[1] == result.selected_keys_by_stage[0]


def test_oracle_looks_past_a_poisoned_first_stage_ranking() -> None:
    result = solve_reachable_oracle(
        ["tempting", "durable"],
        [[10.0, -100.0], [9.0, 100.0]],
        [1, 1],
    )

    assert result.objective == 109.0
    assert result.selected_keys_by_stage == (("durable",), ("durable",))


def test_oracle_handles_exact_stop_depths_and_signed_regressions() -> None:
    keys = ["a", "b", "c"]
    gains = [[10.0, -100.0], [9.0, 5.0], [-10.0, 100.0]]

    result = solve_reachable_oracle(keys, gains, [2, 1])

    assert result.objective == 100.0
    assert result.assigned_depths == (1, 0, 2)
    assert result.selected_keys_by_stage == (("a", "c"), ("c",))
    _assert_feasible(result, keys, gains, [2, 1])


def test_all_stop_and_all_continue_schedules() -> None:
    keys = ["a", "b", "c"]
    gains = [[2.0, -1.0], [-3.0, 7.0], [5.0, 4.0]]

    stopped = solve_reachable_oracle(keys, gains, [0, 0])
    continued = solve_reachable_oracle(keys, gains, [3, 3])

    assert stopped.objective == 0.0
    assert stopped.assigned_depths == (0, 0, 0)
    assert stopped.selected_indices_by_stage == ((), ())
    assert continued.objective == 14.0
    assert continued.assigned_depths == (2, 2, 2)
    assert continued.selected_keys_by_stage == (("a", "b", "c"), ("a", "b", "c"))


def test_selected_keys_are_deterministic_under_input_permutation_and_ties() -> None:
    keys = ["z", "a", "m", "b"]
    gains = [[0.0, 0.0] for _ in keys]
    first = solve_reachable_oracle(keys, gains, [3, 2])
    permutation = [2, 0, 3, 1]
    second = solve_reachable_oracle(
        [keys[index] for index in permutation],
        [gains[index] for index in permutation],
        [3, 2],
    )

    assert first.selected_keys_by_stage == second.selected_keys_by_stage
    assert first == solve_reachable_oracle(keys, gains, [3, 2])


def test_matches_exhaustive_assignment_on_random_small_instances() -> None:
    rng = random.Random(1729)
    for position_count in range(1, 7):
        for stage_count in range(1, 4):
            for case in range(20):
                keys = [f"p{index:02d}" for index in range(position_count)]
                gains = [
                    [float(rng.randint(-8, 8)) for _ in range(stage_count)]
                    for _ in keys
                ]
                witness_depths = [rng.randrange(stage_count + 1) for _ in keys]
                counts = [
                    sum(depth > stage for depth in witness_depths)
                    for stage in range(stage_count)
                ]

                result = solve_reachable_oracle(keys, gains, counts)

                assert result.objective == pytest.approx(
                    _brute_force_objective(gains, counts)
                ), (position_count, stage_count, case, gains, counts)
                _assert_feasible(result, keys, gains, counts)


@pytest.mark.parametrize(
    ("keys", "gains", "counts", "message"),
    [
        ([], [], [0], "at least one position"),
        (["a"], [[1.0]], [], "at least one stage"),
        (["a", "a"], [[1.0], [2.0]], [1], "must be unique"),
        ([""], [[1.0]], [1], "nonempty strings"),
        (["a"], [], [1], "one row per position"),
        (["a"], [[1.0, 2.0]], [1], "exactly 1 stages"),
        (["a"], [[float("nan")]], [1], "finite real number"),
        (["a"], [[True]], [1], "finite real number"),
        (["a"], [[1.0]], [2], "between zero"),
        (["a", "b"], [[1.0], [2.0]], [1.0], "must be integers"),
        (["a", "b"], [[1.0, 2.0], [3.0, 4.0]], [1, 2], "nonincreasing"),
    ],
)
def test_rejects_invalid_inputs(
    keys: Sequence[str],
    gains: Sequence[Sequence[float]],
    counts: Sequence[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_reachable_oracle(keys, gains, counts)
