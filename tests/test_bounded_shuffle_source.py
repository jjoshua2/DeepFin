"""Actual source/ledger adapter tests, all fixtures synthetic and locally owned."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.bounded_shuffle import BoundedShuffleReplay
from chess_anti_engine.replay.bounded_shuffle_source import BoundedReservationWriter, SequentialArraySource, from_fixed_manifest
from chess_anti_engine.replay.stored_row_once import QuotaLedger
from tests.test_bounded_shuffle import consume, limits, replay_fixture, source_fixture
from tests.test_fixed_population import population_fixture
from tests.test_core_selfplay_union_manifest import ARCH


def test_actual_fixed_manifest_adapter_preserves_defaults_rows_and_natural_duplicate_boards(tmp_path: Path) -> None:
    manifest, ref = population_fixture(tmp_path)
    source = from_fixed_manifest(ARCH, manifest, ref, rng=np.random.default_rng(2))
    records: list[Any] = []
    with BoundedShuffleReplay(source, seed=11, batch_size=4, limits=limits(source), lineage="synthetic-manifest",
                              reserve=lambda rows, raw: records.extend(rows)) as replay:
        while replay.remaining:
            batch = replay.sample_batch_arrays(4)
            assert np.all(batch["x"] == 1)
            assert batch["policy_target"].dtype == np.float16
            assert batch["legal_mask"].dtype == np.uint8
            assert np.all(batch["has_policy"] == 1)
            del batch
            replay.complete_batch()
    assert len(records) == 24
    assert len({(r.generation, r.source_sha256, r.row_index) for r in records}) == 24


@pytest.mark.parametrize("prefetch", [False, True])
def test_source_views_charge_full_owning_allocation_before_returning_row(prefetch: bool) -> None:
    source, arrays, _ = source_fixture(5)
    small = source._budgets[0].copy()
    # A decoder returning a small logical x view still owns this MUCH larger root.
    backing = np.ones((20, 175, 8, 8), dtype=np.float16)
    viewed = dict(arrays[0])
    viewed["x"] = backing[:5]
    small["decoded_bytes"] = sum(v.nbytes for v in viewed.values()) + 1
    guarded = SequentialArraySource(identity=source.identity, membership=source.membership,
                                    budgets=[small], rng=source.rng, decode=lambda _slot: viewed)
    with BoundedShuffleReplay(guarded, seed=1, batch_size=4, limits=limits(guarded), lineage="view",
                              reserve=lambda *_args: None) as replay:
        if prefetch:
            replay.start_prefetch()
        with pytest.raises(ValueError, match=r"owning allocation|tainted|prefetch failed"):
            replay.sample_batch_arrays(4)
        with pytest.raises(ValueError, match="tainted"):
            replay.checkpoint()


def test_actual_quota_ledger_uses_new_order_records_and_refuses_ahead_tail(tmp_path: Path) -> None:
    path, journal_path = tmp_path / "quota.jsonl", tmp_path / "draws.jsonl"
    with QuotaLedger(path, first=True, expected_journals=[], max_file_bytes=1 << 20, max_total_bytes=1 << 20) as ledger, journal_path.open("xb", buffering=0) as journal:
        writer = BoundedReservationWriter(ledger, journal)
        replay, _, _ = replay_fixture(5, reserve=writer)
        with replay:
            consume(replay)
            state = replay.checkpoint()
        assert ledger.checkpoint()["bytes"] == journal_path.stat().st_size
    records = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
    assert [len(r["rows"]) for r in records] == [4, 1]
    assert all(r["order"] == "APPROXIMATE_BOUNDED_SHUFFLE_V1" for r in records)
    assert records[-1]["emitted"] == state["emitted"] == 5
    with pytest.raises(ValueError, match="ahead/behind"), QuotaLedger(path, first=False, expected_journals=[], max_file_bytes=1 << 20, max_total_bytes=1 << 20):
        pass


def test_prefetch_close_joins_worker_and_releases_owned_storage() -> None:
    replay, _, _ = replay_fixture()
    replay.start_prefetch()
    replay.close()
    assert replay._worker is None
    assert sum(replay.memory_accounting["used"].values()) == 0
    with pytest.raises(ValueError, match="closed"):
        replay.sample_batch_arrays(4)


def test_real_ledger_reservation_precedes_partial_journal_failure_and_stays_spent(tmp_path: Path) -> None:
    ledger_path, journal_path = tmp_path / "spent.jsonl", tmp_path / "partial.jsonl"

    class PartialJournal:
        def __init__(self, stream: Any):
            self.stream = stream

        def write(self, raw: bytes) -> int:
            return self.stream.write(raw[:-1])

    with QuotaLedger(ledger_path, first=True, expected_journals=[], max_file_bytes=1 << 20, max_total_bytes=1 << 20) as ledger, journal_path.open("xb", buffering=0) as stream:
        writer = BoundedReservationWriter(ledger, PartialJournal(stream))
        replay, _, _ = replay_fixture(5, reserve=writer)
        with replay:
            with pytest.raises(ValueError, match="partial bounded draw journal"):
                replay.sample_batch_arrays(4)
            assert ledger.checkpoint()["bytes"] > journal_path.stat().st_size
            with pytest.raises(ValueError, match="tainted"):
                replay.checkpoint()
            with pytest.raises(ValueError, match="tainted"):
                replay.sample_batch_arrays(4)
