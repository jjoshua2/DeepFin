"""Synthetic payload tests of actual engine/adapter; no historical data or model."""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.bounded_shuffle import BoundedShuffleReplay, ReplayLimits, digest
from chess_anti_engine.replay.bounded_shuffle_source import SequentialArraySource


def source_fixture(n: int = 23) -> tuple[SequentialArraySource, list[dict[str, Any]], list[int]]:
    arrays: list[dict[str, Any]] = []
    membership: list[tuple[int, str, int]] = []
    budgets: list[dict[str, int]] = []
    calls: list[int] = []
    start = 0
    while start < n:
        rows = min(5, n - start)
        slot = len(arrays)
        policy = np.zeros((rows, 1858), dtype=np.float16)
        policy[:, 0] = 1
        a = {"x": np.full((rows, 175, 8, 8), start, dtype=np.float16), "policy_target": policy,
             "legal_mask": np.ones((rows, 1858), dtype=np.uint8),
             "wdl_target": np.arange(rows, dtype=np.int8) % 3, "priority": np.ones(rows, dtype=np.float32),
             "has_policy": np.ones(rows, dtype=np.uint8), "has_legal_mask": np.ones(rows, dtype=np.uint8),
             "game_id": np.full(rows, slot, dtype=np.int64), "ply_index": np.arange(rows, dtype=np.int32),
             "has_game_id": np.ones(rows, dtype=np.uint8), "has_ply_index": np.ones(rows, dtype=np.uint8),
             "is_selfplay": np.ones(rows, dtype=np.uint8), "has_is_selfplay": np.ones(rows, dtype=np.uint8),
             "is_network_turn": np.ones(rows, dtype=np.uint8), "has_is_network_turn": np.ones(rows, dtype=np.uint8),
             "_input_history_encoding": np.asarray("lc0_root_legacy_meta"),
             "_policy_encoding": np.asarray("lc0_1858"), "_history_rep_fix": np.asarray("true")}
        arrays.append(a)
        membership.append((slot + 1, f"{slot + 1:064x}", rows))
        size = sum(v.nbytes for v in a.values())
        budgets.append({"rows": rows, "decoded_bytes": size + 65536,
                        "workspace_bytes": 8 * size + (1 << 20), "gather_row_bytes": (size + rows - 1) // rows + 4096})
        start += rows

    def decode(slot: int) -> dict[str, Any]:
        calls.append(slot)
        return arrays[slot]

    source = SequentialArraySource(identity={"schema": "SYNTHETIC_METADATA_ONLY", "rows": n,
                                            "membership": [list(s) for s in membership]},
                                   membership=tuple(membership), budgets=budgets,
                                   rng=np.random.default_rng(123), decode=decode)
    return source, arrays, calls


def limits(_source: SequentialArraySource, batch: int = 4) -> ReplayLimits:
    return ReplayLimits(total_bytes=128 << 20, source_bytes=16 << 20, buffer_bytes=8 << 20,
                        ready_bytes=4 << 20, consumer_bytes=96 << 20, scratch_bytes=4 << 20,
                        buffer_rows=max(8, batch), ready_batches=2)


def replay_fixture(n: int = 23, *, reserve: Any = None) -> tuple[BoundedShuffleReplay, list[Any], list[int]]:
    source, _, calls = source_fixture(n)
    records: list[Any] = []

    def saved(rows: Any, encoded: bytes) -> None:
        records.append((rows, json.loads(encoded)))

    return BoundedShuffleReplay(source, seed=71, batch_size=4, limits=limits(source),
                                lineage="synthetic-fresh-student", reserve=saved if reserve is None else reserve), records, calls


def consume(replay: BoundedShuffleReplay) -> tuple[list[Any], list[int], list[float]]:
    ids, sizes, mirror = [], [], []
    while replay.remaining:
        batch = replay.sample_batch_arrays(4)
        sizes.append(int(batch["x"].shape[0]))
        ids.extend(zip(batch["game_id"].tolist(), batch["ply_index"].tolist(), strict=True))
        mirror.append(float(replay.rng.random()))
        del batch
        replay.complete_batch()
    return ids, sizes, mirror


@pytest.mark.parametrize("n", [1, 4, 5, 23])
def test_exact_once_real_tail_and_exhaustion_with_bounded_prefetch(n: int) -> None:
    replay, records, calls = replay_fixture(n)
    with replay:
        replay.start_prefetch()
        ids, sizes, _ = consume(replay)
        assert len(ids) == len(set(ids)) == n
        assert sum(sizes) == n
        assert sizes[-1] == (n % 4 or 4)
        assert sum(len(rows) for rows, _ in records) == n
        assert calls == list(range((n + 4) // 5))  # one sequential source load, no replacement
        assert replay.memory_accounting["peak_bytes"] <= replay.limits.total_bytes
        assert replay.memory_accounting["ready_batches"] <= replay.limits.ready_batches
        with pytest.raises(ValueError, match="exhausted"):
            replay.sample_batch_arrays(4)
    assert replay._worker is None


def test_json_resume_nonempty_slots_and_ready_preserves_order_and_mirror_rng() -> None:
    whole, _, _ = replay_fixture()
    with whole:
        expected = consume(whole)
    first, first_records, _ = replay_fixture()
    with first:
        batch = first.sample_batch_arrays(4)
        first_ids = list(zip(batch["game_id"].tolist(), batch["ply_index"].tolist(), strict=True))
        first_mirror = float(first.rng.random())
        del batch
        first.complete_batch()
        first.start_prefetch()
        with first._condition:
            assert first._condition.wait_for(lambda: len(first._ready) == 2, timeout=5)
        state = json.loads(json.dumps(first.checkpoint()))
        assert state["slots"]
        assert len(state["ready"]) == 2
        assert len(first_records) == 1  # prefetched rows never hit presentation ledger
    resumed, records, _ = replay_fixture()
    with resumed:
        resumed.restore(state, journal_rows=4, journal_prefix=state["reservation_prefix"])
        resumed.start_prefetch()
        rest, sizes, mirrors = consume(resumed)
        assert first_ids + rest == expected[0]
        assert [4, *sizes] == expected[1]
        assert [first_mirror, *mirrors] == expected[2]
        assert sum(len(rows) for rows, _ in records) == 19


def test_queue_backpressure_never_spends_quota_or_aliases_source_or_other_rows() -> None:
    source, original, _ = source_fixture(23)
    records: list[Any] = []
    replay = BoundedShuffleReplay(source, seed=7, batch_size=4, limits=limits(source), lineage="copies",
                                  reserve=lambda rows, raw: records.append((rows, raw)))
    with replay:
        replay.start_prefetch()
        with replay._condition:
            assert replay._condition.wait_for(lambda: len(replay._ready) == 2, timeout=5)
        frozen = replay.checkpoint()
        assert records == []
        assert frozen["emitted"] == 0
        assert len(frozen["ready"]) == 2
        before = [a["x"].copy() for a in original]
        batch = replay.sample_batch_arrays(4)
        assert batch["x"].flags.writeable
        batch["x"][:] = -99  # mirrors/Trainer may mutate borrowed output
        for old, now in zip(before, original, strict=True):
            np.testing.assert_array_equal(now["x"], old)
        del batch
        replay.complete_batch()
        following = replay.sample_batch_arrays(4)
        assert np.all(following["x"] >= 0)
        del following
        replay.complete_batch()


@pytest.mark.parametrize("fault", ["duplicate", "cursor", "spent", "queue", "identity", "journal", "bool_limit", "rng", "mirror"])
def test_corrupt_rehashed_state_refuses_before_source_decode_and_taints(fault: str) -> None:
    replay, _, _ = replay_fixture()
    with replay:
        batch = replay.sample_batch_arrays(4)
        del batch
        replay.complete_batch()
        state = json.loads(json.dumps(replay.checkpoint()))
    journal_rows, prefix = state["emitted"], state["reservation_prefix"]
    if fault == "duplicate":
        state["slots"].append(state["slots"][0])
    elif fault == "cursor":
        state["upstream"] -= 1
    elif fault == "spent":
        state["spent_hex"] = "00" * (len(state["spent_hex"]) // 2)
    elif fault == "queue":
        state["ready"] = [[0]]
    elif fault == "identity":
        state["identity"]["rows"] = 999
    elif fault == "journal":
        journal_rows += 1
    elif fault == "rng":
        state["shuffle_state"][1][0] = True
    elif fault == "mirror":
        state["mirror_state"]["state"]["inc"] = 2
    else:
        state["limits"]["ready_batches"] = True
    state["sha256"] = digest({k: v for k, v in state.items() if k != "sha256"})
    fresh, _, calls = replay_fixture()
    with fresh:
        with pytest.raises((ValueError, TypeError)):
            fresh.restore(state, journal_rows=journal_rows, journal_prefix=prefix)
        assert calls == []
        with pytest.raises(ValueError, match="tainted"):
            fresh.checkpoint()


def test_partial_reservation_failure_and_forward_abort_cannot_checkpoint_or_redraw() -> None:
    def failed(rows: Any, raw: bytes) -> None:
        assert rows
        assert raw.endswith(b"\n")
        raise OSError("partial durable write")

    replay, _, _ = replay_fixture(reserve=failed)
    with replay:
        with pytest.raises(OSError, match="partial durable"):
            replay.sample_batch_arrays(4)
        with pytest.raises(ValueError, match="tainted"):
            replay.checkpoint()
        with pytest.raises(ValueError, match="tainted"):
            replay.sample_batch_arrays(4)
    forward, _, _ = replay_fixture()
    with forward:
        batch = forward.sample_batch_arrays(4)
        del batch
        forward.abort_consumption()
        with pytest.raises(ValueError, match="tainted"):
            forward.checkpoint()


def test_configuration_and_inflight_checkpoint_enforce_owned_byte_frontiers() -> None:
    source, _, _ = source_fixture()
    bad = limits(source)
    with pytest.raises(ValueError, match="category caps"):
        ReplayLimits(**{**bad.__dict__, "total_bytes": 1})
    with pytest.raises(ValueError, match="cannot hold"):
        BoundedShuffleReplay(source, seed=1, batch_size=4, limits=ReplayLimits(**{**bad.__dict__, "buffer_bytes": 1}),
                             lineage="too-small", reserve=lambda *_args: None)
    replay, _, _ = replay_fixture()
    with replay:
        batch = replay.sample_batch_arrays(4)
        with pytest.raises(ValueError, match="inflight"):
            replay.checkpoint()
        del batch
        replay.complete_batch()
        replay.checkpoint()


def test_symbolic_lineage_audit_checks_actual_records_and_unspent_queue_without_payload() -> None:
    from chess_anti_engine.replay.bounded_shuffle import audit_selection

    replay, records, _ = replay_fixture()
    with replay:
        batch = replay.sample_batch_arrays(4)
        del batch
        replay.complete_batch()
        replay.start_prefetch()
        with replay._condition:
            assert replay._condition.wait_for(lambda: len(replay._ready) == 2, timeout=5)
        state = json.loads(json.dumps(replay.checkpoint()))
    audit = audit_selection(state, (record for _, record in records))
    assert audit["payload_reads"] == 0
    assert audit["emitted"] == 4
    assert audit["queued_unspent"] == 8
    corrupt = copy.deepcopy(state)
    corrupt["ready"][0].reverse()
    corrupt["sha256"] = digest({k: v for k, v in corrupt.items() if k != "sha256"})
    with pytest.raises(ValueError, match="ready order"):
        audit_selection(corrupt, (record for _, record in records))
    changed_record = copy.deepcopy(records[0][1])
    changed_record["rows"].reverse()
    with pytest.raises(ValueError, match="reservation order"):
        audit_selection(state, iter([changed_record]))
