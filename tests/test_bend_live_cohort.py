"""Inexpensive identity/accounting validation; native live execution is opt-in."""
from __future__ import annotations

import copy
import json

import pytest

from native.bend_engine.multi_root.verify_live import CONTROL, ROOT, WORK, tree_snapshots, validate


def root() -> dict[str, object]:
    return {"schema": "deepfin.live-root.v1", "slot": 1, "generation": 1,
            "completed_simulations": 1, "dispatched_real_rows": 2, "executed_real_rows": 2,
            "accepted_neural_rows": 1, "cancelled_rows": 1}


def work() -> dict[str, object]:
    return {"schema": "deepfin.live-cohort-work.v1", "batch_size": 4, "reported_generations": 1,
            "completed_simulations": 1, "forward_calls": 2, "dispatched_real_rows": 2,
            "executed_real_rows": 2, "accepted_neural_rows": 1, "cancelled_rows": 1,
            "executed_wasted_rows": 1, "physical_rows": 8, "padded_rows": 6,
            "unconfirmed_forward_rows": 0, "unresolved_rows": 0}


def transcript(r: object, w: object) -> list[str]:
    return [CONTROL + "admitted 1 1", ROOT + json.dumps(r), WORK + json.dumps(w)]


def test_valid_lifetime_never_refunds_retired_work() -> None:
    parsed = validate(transcript(root(), work()))
    assert parsed["work"]["executed_real_rows"] == 2
    assert parsed["work"]["accepted_neural_rows"] == 1


@pytest.mark.parametrize("key", ["slot", "generation", "completed_simulations", "dispatched_real_rows", "executed_real_rows", "accepted_neural_rows", "cancelled_rows"])
@pytest.mark.parametrize("bad", [True, None, "1", -1, 1.0])
def test_reject_malformed_root_integers(key: str, bad: object) -> None:
    r = root()
    r[key] = bad
    with pytest.raises(ValueError, match=r"invalid|stale"):
        validate(transcript(r, work()))


@pytest.mark.parametrize("key", ["reported_generations", "completed_simulations", "dispatched_real_rows", "executed_real_rows", "accepted_neural_rows", "cancelled_rows", "executed_wasted_rows", "physical_rows", "padded_rows", "unconfirmed_forward_rows", "unresolved_rows"])
@pytest.mark.parametrize("bad", [True, None, "1", -1])
def test_reject_malformed_lifetime_integers(key: str, bad: object) -> None:
    w = work()
    w[key] = bad
    with pytest.raises(ValueError, match=r"invalid|mismatch|summary|drain"):
        validate(transcript(root(), w))


@pytest.mark.parametrize("key", list(work())[1:])
def test_counter_contradiction(key: str) -> None:
    w = work()
    value = w[key]
    assert isinstance(value, int)
    w[key] = value + 1
    with pytest.raises(ValueError, match=r"invalid|mismatch|summary|drain"):
        validate(transcript(root(), w))


@pytest.mark.parametrize("r", [[], True, 2, None, {}])
def test_bad_root_object(r: object) -> None:
    with pytest.raises(ValueError, match=r"invalid|stale"):
        validate(transcript(r, work()))


def test_result_must_precede_slot_reuse_and_removal() -> None:
    valid = transcript(root(), work())
    for rows in ([valid[0], CONTROL + "admitted 1 2", *valid[1:]],
                 [valid[0], CONTROL + "removed 1 1", *valid[1:]],
                 [valid[0], valid[1], valid[1], valid[2]],
                 [valid[0], CONTROL + "admitted 1 1", *valid[1:]]):
        with pytest.raises(ValueError, match=r"unreported|retirement|duplicate|recycled|missing"):
            validate(rows)


def test_generation_ids_are_process_global_and_monotonic() -> None:
    valid = transcript(root(), work())
    for notice in ("admitted 2 1", "admitted 17 2", "admitted 0 2", "admitted 1 0"):
        with pytest.raises(ValueError, match=r"invalid|recycled"):
            validate([*valid[:2], CONTROL + notice, valid[2]])
    with pytest.raises(ValueError, match="non-monotonic"):
        validate([CONTROL + "admitted 1 3", CONTROL + "admitted 2 2"])


def test_one_lifetime_summary_required() -> None:
    valid = transcript(root(), work())
    for rows in (valid[:-1], [*valid, valid[-1]]):
        with pytest.raises(ValueError, match=r"unreported|retirement|duplicate|recycled|missing"):
            validate(rows)


def test_successful_empty_session() -> None:
    w = copy.deepcopy(work())
    for k in w:
        if k not in ("schema", "batch_size"):
            w[k] = 0
    assert validate([WORK + json.dumps(w)])["roots"] == {}


def test_lifecycle_cannot_follow_final_summary() -> None:
    valid = transcript(root(), work())
    for row in (ROOT + json.dumps(root()), CONTROL + "admitted 1 2", CONTROL + "removed 1 1"):
        with pytest.raises(ValueError, match="after final summary"):
            validate([*valid, row])


def test_accepted_rows_require_completed_simulations() -> None:
    r = root()
    r["completed_simulations"] = 0
    with pytest.raises(ValueError, match="accepted work exceeds"):
        validate(transcript(r, work()))


def test_tree_snapshot_matches_complete_generation() -> None:
    rows = ["info string live_result_begin 1 2", "info string cohort_node 1 0 " + json.dumps([0] * 29),
            ROOT + json.dumps({"slot": 1, "generation": 2, "used_nodes": 1})]
    assert tree_snapshots(rows) == {(1, 2): [[0] * 29]}
    for wrong, expected in ((rows[1:], "unscoped"), (rows[:-1], "unfinished"),
                            ([rows[0], rows[0]], "unfinished"),
                            ([rows[0], rows[1].replace("node 1 0", "node 1 1"), rows[2]], "reordered"),
                            ([rows[0], rows[1].replace("node 1 0", "node 2 0"), rows[2]], "unscoped"),
                            ([rows[0], rows[1], rows[2].replace('"generation": 2', '"generation": 3')], "incomplete")):
        with pytest.raises(ValueError, match=expected):
            tree_snapshots(wrong)
