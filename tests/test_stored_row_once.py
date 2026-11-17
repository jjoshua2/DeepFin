"""Synthetic stored-row identity/quota and actual Trainer-batch integration."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import random
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
def test_partial_population_emits_every_row_once_without_padding(n: int, batch: int) -> None:
    seen: list[Any] = []
    replay = adapter(n, batch, seen.append)
    sizes: list[int] = []
    while replay.remaining:
        sizes.append(int(replay.sample_batch_arrays(batch)["x"].shape[0]))
    assert sizes == [batch] * (n // batch) + ([n % batch] if n % batch else [])
    assert len(sizes) == (n + batch - 1) // batch
    rows = [r for entries in seen for r in entries]
    assert len(rows) == len({(r.generation, r.source_sha256, r.row_index) for r in rows}) == n
    assert sorted(r.row_index for r in rows) == list(range(n))
    assert replay.state()["cursor"] == n
    assert replay.state()["last_batch_rows"] == sizes[-1]
    with pytest.raises(ValueError, match="exhausted"):
        replay.sample_batch_arrays(batch)


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


@pytest.mark.parametrize("fault", ["duplicate_order", "reorder", "cursor", "batches", "seed", "membership", "bool_index", "last_batch", "permutation_sha"])
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
    elif fault == "last_batch":
        state["last_batch_rows"] = 1
    elif fault == "permutation_sha":
        state["permutation_sha256"] = "0" * 64
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


def test_uniform_shuffle_is_independent_of_mirror_and_global_rng() -> None:
    global_state = random.getstate()
    draws: list[Any] = []
    replay = adapter(17, 4, draws.append, seed=91)
    augmentation_state = copy.deepcopy(replay.rng.bit_generator.state)
    expected = list(range(17))
    random.Random(91).shuffle(expected)
    assert replay.permutation == expected
    assert sorted(replay.permutation) == list(range(17))
    assert random.getstate() == global_state
    assert replay.rng.bit_generator.state == augmentation_state
    assert replay.state()["order"] == "python_random_mt19937_shuffle_v1"
    assert replay.state()["permutation_sha256"] == hashlib.sha256(canonical(expected).encode()).hexdigest()


def test_resume_before_and_after_final_partial_preserves_exact_cursor(tmp_path: Path) -> None:
    draws: list[Any] = []
    first = adapter(5, 2, draws.append)
    first.sample_batch_arrays(2)
    first.sample_batch_arrays(2)
    prefix_state = first.state()
    prefix = sealed_journal(tmp_path / "prefix.jsonl", draws)
    tail_draws: list[Any] = []
    resumed = adapter(5, 2, tail_draws.append)
    resumed.restore(prefix_state, [prefix], max_file_bytes=1 << 20, max_total_bytes=1 << 20)
    assert resumed.sample_batch_arrays(2)["x"].shape[0] == 1
    tail = sealed_journal(tmp_path / "tail.jsonl", tail_draws, first=2)
    final = resumed.state()
    assert final["cursor"] == 5
    assert final["batches"] == 3
    assert final["last_batch_rows"] == 1
    assert len({r.row_index for batch in [*draws, *tail_draws] for r in batch}) == 5
    reopened = adapter(5, 2, lambda _: pytest.fail("exhausted cursor redrew a row"))
    reopened.restore(final, [prefix, tail], max_file_bytes=1 << 20, max_total_bytes=1 << 20)
    assert reopened.state() == final
    with pytest.raises(ValueError, match="exhausted"):
        reopened.sample_batch_arrays(2)
    phantom = copy.deepcopy(final)
    phantom["batches"] = 4
    with pytest.raises(ValueError, match="batch count exceeds"):
        adapter(5, 2, lambda _: None).restore(phantom, [prefix, tail], max_file_bytes=1 << 20, max_total_bytes=1 << 20)


@pytest.mark.parametrize("fault", ["duplicate_padding", "missing_tail", "early_partial", "extra_tail"])
def test_partial_journal_cannot_pad_drop_or_move_remainder(tmp_path: Path, fault: str) -> None:
    seen: list[Any] = []
    replay = adapter(5, 2, seen.append)
    while replay.remaining:
        replay.sample_batch_arrays(2)
    state = replay.state()
    records: list[dict[str, Any]] = [{"draw_batch": i, "rows": [asdict(r) for r in rows]} for i, rows in enumerate(seen)]
    if fault == "duplicate_padding":
        records[-1]["rows"].append(dict(records[-1]["rows"][0]))
    elif fault == "missing_tail":
        records.pop()
    elif fault == "early_partial":
        records[0]["rows"].pop()
    else:
        records.append({"draw_batch": 3, "rows": [dict(records[-1]["rows"][0])]})
    path = tmp_path / "wrong.jsonl"
    path.write_text("".join(canonical(record) + "\n" for record in records))
    path.chmod(0o444)
    ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match=r"journal|checkpoint"):
        adapter(5, 2, lambda _: None).restore(state, [ref], max_file_bytes=1 << 20, max_total_bytes=1 << 20)


@pytest.mark.parametrize("anchor", [False, True])
def test_real_trainer_scales_whole_partial_objective_by_actual_over_nominal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anchor: bool,
) -> None:
    # Analytic objective injection isolates normalization; real finite-family
    # acceptance/anchor/checkpoint tests below exercise the actual loss pipeline.
    from chess_anti_engine.train import trainer as trainer_module
    from tests.test_grad_clip_scope import _make_trainer
    trainer = _make_trainer(tmp_path, optimizer="nadamw", grad_clip_mode="none")
    parameter = dict(trainer.model.named_parameters())["head.weight"]
    captured: list[torch.Tensor] = []
    calls: list[str] = []
    def losses(_out: Any, _batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["report_exact_masked_sums"] is False
        assert kwargs["exact_corpus_rows"] is None
        zero = torch.zeros(())
        return {**dict.fromkeys(trainer_module._LOSS_KEY_TO_METRIC_FIELD, zero), "total": parameter.sum()}
    class Penalty:
        def penalty(self, _out: Any, _batch: Any) -> torch.Tensor:
            calls.append("anchor")
            return parameter.sum() * 2
    def observe(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        gradient = parameter.grad
        assert gradient is not None
        captured.append(gradient.detach().clone())
    monkeypatch.setattr(trainer_module, "compute_loss", losses)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda *_args: {})
    if anchor:
        trainer._frozen_policy_anchor = cast(Any, Penalty())
    replay = adapter(1, 3, lambda _: None)
    hook = trainer.opt.register_step_pre_hook(observe)
    try:
        metrics = trainer.train_steps(cast(Any, replay), batch_size=3, steps=1)
        expected = (3.0 if anchor else 1.0) / 3.0
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], torch.full_like(parameter, expected))
        assert metrics.train_samples_seen == 1
        assert metrics.train_steps_done == 1
        assert calls == (["anchor"] if anchor else [])
    finally:
        hook.remove()
        trainer.writer.close()
