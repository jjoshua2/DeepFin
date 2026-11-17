"""Synthetic stored-row identity/quota and actual Trainer-batch integration."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.generation import GenerationReplay
from chess_anti_engine.replay.stored_row_once import QuotaLedger, StoredRowOnceReplay, canonical, verify_journals
from chess_anti_engine.train.trainer import Trainer
from tests.test_generation_replay import _chunk


def adapter(n: int, batch: int, callback: Any, seed: int = 7) -> StoredRowOnceReplay:
    replay = GenerationReplay(generations=4, rng=np.random.default_rng(45), first_generation=2)
    replay.admit_generation(2, [_chunk(1, n)])
    return StoredRowOnceReplay(replay, seed=seed, batch_size=batch, expected_rows=n, on_draw=callback)


def sealed_journal(path: Path, batches: list[Any], first: int = 0) -> dict[str, str]:
    path.write_text("".join(canonical({"draw_batch": first + i, "rows": [asdict(r) for r in rows]}) + "\n"
                            for i, rows in enumerate(batches)))
    path.chmod(0o444)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_17568_rows_are_549_full_updates_and_never_wrap() -> None:
    # Test identity/dose using a tiny gather object, not a 17568-board allocation.
    class Gather:
        def _gather_rows(self, indices: list[int]) -> dict[str, Any]:
            return {"x": np.asarray(indices).reshape(-1, 1)}
    class Population:
        rng = np.random.default_rng(3)
        membership = ((2, "a" * 64, 17568),)
        _buffer = Gather()
        def __len__(self) -> int:
            return 17568
    seen: list[Any] = []
    replay = StoredRowOnceReplay(Population(), seed=1673, batch_size=32, expected_rows=17568, on_draw=seen.extend)
    for _ in range(549):
        assert replay.sample_batch_arrays(32)["x"].shape[0] == 32
    assert len(seen) == len({(r.generation, r.source_sha256, r.row_index) for r in seen}) == 17568
    assert replay.state()["batches"] == 549
    with pytest.raises(ValueError, match="exhausted"):
        replay.sample_batch_arrays(32)


@pytest.mark.parametrize(("n", "batch"), [(1, 2), (5, 2), (17567, 32)])
def test_partial_population_refuses_without_drawing(n: int, batch: int) -> None:
    seen: list[Any] = []
    with pytest.raises(ValueError, match="partial batch"):
        adapter(n, batch, seen.append)
    assert not seen


def test_natural_identical_boards_keep_distinct_stored_row_identities(tmp_path: Path) -> None:
    draws: list[Any] = []
    replay = adapter(4, 2, draws.append)
    # Deliberately equal board arrays. They remain four stored identities.
    replay._replay._buffer._chunks[0]["x"][:] = 0
    trainer = Trainer(torch.nn.Linear(1, 1), device="cpu", lr=1e-3, use_amp=False,
                      log_dir=tmp_path, prefetch_batches=True)
    try:
        for _ in range(2):
            batch = trainer._sample_batch_host(cast(Any, replay), batch_size=2, mirror_prob=1.0)
            assert isinstance(batch, dict)
            assert batch["x"].shape[0] == 2
        assert sorted(r.row_index for rows in draws for r in rows) == list(range(4))
        assert replay.state()["cursor"] == 4
        assert replay.state()["mirror_shares_quota"] is True
    finally:
        trainer.writer.close()


def test_json_state_roundtrip_continues_exact_order_and_journal_chain(tmp_path: Path) -> None:
    first_draws: list[Any] = []
    first = adapter(8, 2, first_draws.append)
    first.sample_batch_arrays(2)
    first.sample_batch_arrays(2)
    state = json.loads(canonical(first.state()))
    journal = sealed_journal(tmp_path / "one.jsonl", first_draws)
    tail: list[Any] = []
    resumed = adapter(8, 2, tail.append)
    resumed.restore(state, [journal], max_file_bytes=1 << 20, max_total_bytes=1 << 20)
    while first.remaining:
        first.sample_batch_arrays(2)
        resumed.sample_batch_arrays(2)
    assert tail == first_draws[2:]
    assert resumed.state() == first.state()
    assert resumed.state()["cursor"] == 8


@pytest.mark.parametrize("fault", ["duplicate_order", "reorder", "cursor", "batches", "seed", "membership", "bool_index"])
def test_saved_state_tamper_refuses(tmp_path: Path, fault: str) -> None:
    draws: list[Any] = []
    replay = adapter(4, 2, draws.append)
    replay.sample_batch_arrays(2)
    state = copy.deepcopy(replay.state())
    if fault == "duplicate_order":
        state["permutation"][1] = state["permutation"][0]
    elif fault == "reorder":
        state["permutation"].reverse()
    elif fault == "cursor":
        state["cursor"] = 0
    elif fault == "batches":
        state["batches"] = 0
    elif fault == "seed":
        state["seed"] += 1
    elif fault == "membership":
        state["membership"][0][1] = "b" * 64
    else:
        state["permutation"][state["permutation"].index(0)] = False
    ref = sealed_journal(tmp_path / "journal.jsonl", draws)
    with pytest.raises(ValueError, match=r"stored-row|checkpoint/journal"):
        adapter(4, 2, lambda _: None).restore(state, [ref], max_file_bytes=1 << 20, max_total_bytes=1 << 20)


def test_newer_journal_tail_cannot_be_replayed_from_old_checkpoint(tmp_path: Path) -> None:
    draws: list[Any] = []
    replay = adapter(4, 2, draws.append)
    replay.sample_batch_arrays(2)
    old = replay.state()
    replay.sample_batch_arrays(2)
    ref = sealed_journal(tmp_path / "tail.jsonl", draws)
    with pytest.raises(ValueError, match="checkpoint/journal cursor mismatch"):
        adapter(4, 2, lambda _: None).restore(old, [ref], max_file_bytes=1 << 20, max_total_bytes=1 << 20)


@pytest.mark.parametrize("fault", ["hash", "duplicate_ref", "budget", "truncated", "bool_row"])
def test_journal_proof_refuses_invalid_actual_streams(tmp_path: Path, fault: str) -> None:
    draws: list[Any] = []
    replay = adapter(4, 2, draws.append)
    replay.sample_batch_arrays(2)
    ref = sealed_journal(tmp_path / "j.jsonl", draws)
    refs = [ref]
    budget = 1 << 20
    if fault == "hash":
        ref["sha256"] = "0" * 64
    elif fault == "duplicate_ref":
        refs.append(dict(ref))
    elif fault == "budget":
        budget = 1
    else:
        path = Path(ref["path"])
        path.chmod(0o644)
        value = path.read_bytes()
        if fault == "truncated":
            value = value[:-1]
        else:
            record = json.loads(value)
            record["rows"][0]["row_index"] = False
            value = (canonical(record) + "\n").encode()
        path.write_bytes(value)
        path.chmod(0o444)
        ref["sha256"] = hashlib.sha256(value).hexdigest()
    with pytest.raises(ValueError, match="stored-row"):
        verify_journals(replay, refs, max_file_bytes=budget, max_total_bytes=budget)


def test_callback_failure_spends_and_taints_without_repeat() -> None:
    called: list[Any] = []
    def failed(rows: Any) -> None:
        called.extend(rows)
        raise OSError("synthetic journal fsync failure")
    replay = adapter(4, 2, failed)
    with pytest.raises(OSError, match="fsync"):
        replay.sample_batch_arrays(2)
    assert len(called) == 2
    with pytest.raises(ValueError, match="tainted"):
        replay.sample_batch_arrays(2)
    with pytest.raises(ValueError, match="tainted"):
        replay.state()


def test_quota_marker_serializes_prefetch_and_fault_stops_before_second_draw(tmp_path: Path) -> None:
    seen: list[Any] = []
    replay = adapter(4, 2, seen.append)
    trainer = Trainer(torch.nn.Linear(1, 1), device="cpu", lr=1e-3, use_amp=False,
                      log_dir=tmp_path, prefetch_batches=True)
    iterator = trainer._iter_training_batches(cast(Any, replay), batch_size=2, mirror_prob=0.0, count=2)
    try:
        next(iterator)
        assert len(seen) == 1
    finally:
        close = getattr(iterator, "close", None)
        assert callable(close)
        close()
        trainer.writer.close()
    assert len(seen) == 1


@pytest.mark.parametrize("fault", ["tail", "changed_prefix", "concurrent", "restart_first"])
def test_persistent_ledger_blocks_fresh_output_rewind_and_concurrent_owner(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "quota.jsonl"
    draws: list[Any] = []
    replay = adapter(4, 2, draws.append)
    replay.sample_batch_arrays(2)
    ref = sealed_journal(tmp_path / "closed.jsonl", draws)
    line = Path(ref["path"]).read_bytes()
    kwargs = {"max_file_bytes": 1 << 20, "max_total_bytes": 1 << 20}
    with QuotaLedger(path, first=True, expected_journals=[], **kwargs) as ledger:
        ledger.reserve(line)
        assert ledger.checkpoint()["sha256"] == ref["sha256"]
        if fault == "concurrent":
            with (
                pytest.raises(BlockingIOError),
                QuotaLedger(path, first=False, expected_journals=[ref], **kwargs),
            ):
                pytest.fail("concurrent quota owner admitted")
            return
        if fault == "tail":
            ledger.reserve(line)
    if fault == "restart_first":
        with (
            pytest.raises(FileExistsError),
            QuotaLedger(path, first=True, expected_journals=[], **kwargs),
        ):
            pytest.fail("first ledger recreated")
        return
    if fault == "changed_prefix":
        data = path.read_bytes()
        path.write_bytes(data.replace(b"row_index", b"row_indeX", 1))
    with (
        pytest.raises(ValueError, match="quota ledger"),
        QuotaLedger(path, first=False, expected_journals=[ref], **kwargs),
    ):
        pytest.fail("stale checkpoint admitted")
