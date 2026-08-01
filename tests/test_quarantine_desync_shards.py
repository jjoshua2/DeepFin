"""Pins for scripts/quarantine_desync_shards.py.

Every mechanism this tool relies on is one edit from being a control that cannot
fire, so each gets a test that FAILS when the mechanism is removed:

  * the zero-row marker exemption (without it the yardstick fails forever)
  * the index reservation, and --verify's dependence on it (without it the
    yardstick PASSES on the exact state the reservation exists to prevent)
  * the manifest's shard_dir binding, digest checking, and journal
  * restore resumability
  * agreement with the axis functions and thresholds `judge()` wraps
    (NOT with `scripts/value_optimism.py` -- see that test's docstring)
"""
from __future__ import annotations

import importlib.util
import sys
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.eval.value_optimism import (
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
    sf_label_attachment_corr,
    sf_multipv_missing_rate,
)
from chess_anti_engine.replay.shard import (
    SF_MULTIPV_RAW_MAX,
    save_local_shard_arrays,
    shard_positions,
)

_SPEC = importlib.util.spec_from_file_location(
    "quarantine_desync_shards",
    Path(__file__).resolve().parents[1] / "scripts" / "quarantine_desync_shards.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
qds = importlib.util.module_from_spec(_SPEC)
# Registered BEFORE exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for an unregistered spec.
sys.modules[_SPEC.name] = qds
_SPEC.loader.exec_module(qds)

POLICY = 1858


def _shard_arrays(n: int, *, miss_frac: float = 0.0, attached: bool = True) -> dict[str, Any]:
    """A minimal but schema-valid shard.

    ``miss_frac`` of the labelled rows get no MultiPV block (the desync
    fingerprint). ``attached=False`` decorrelates sf_wdl from material, which is
    what the attachment axis measures.
    """
    rng = np.random.default_rng(0)
    x = np.zeros((n, 175, 8, 8), dtype=np.float16)
    # Material: a varying number of own pawns on plane 0.
    counts = rng.integers(1, 9, size=n)
    for i, c in enumerate(counts):
        x[i, 0, 0, :c] = 1.0
    pol = np.full((n, POLICY), 1.0 / POLICY, dtype=np.float16)
    signal = (counts / 8.0) if attached else rng.random(n)
    sf_wdl = np.stack([signal, np.zeros(n), 1.0 - signal], axis=1).astype(np.float16)
    n_miss = round(n * miss_frac)
    has_raw = np.ones(n, dtype=np.uint8)
    has_raw[:n_miss] = 0
    raw = np.zeros((n, SF_MULTIPV_RAW_MAX, 5), dtype=np.int16)
    raw[:, :, 0] = -1
    raw[:, 0, 0] = 7
    return {
        "x": x,
        "policy_target": pol,
        "wdl_target": np.zeros(n, dtype=np.int8),
        "priority": np.ones(n, dtype=np.float32),
        "has_policy": np.ones(n, dtype=np.uint8),
        "sf_wdl": sf_wdl,
        "has_sf_wdl": np.ones(n, dtype=np.uint8),
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": has_raw,
    }


def _write(dirpath: Path, index: int, **kw: Any) -> Path:
    p = dirpath / f"shard_{index:06d}.zarr"
    save_local_shard_arrays(p, arrs=_shard_arrays(kw.pop("n", 200), **kw))
    return p


@pytest.fixture
def window(tmp_path: Path) -> Path:
    """Five clean shards and three dirty ones; the TOP id is dirty."""
    d = tmp_path / "replay_shards"
    d.mkdir()
    for i in (100, 101, 102, 103, 104):
        _write(d, i)
    for i in (105, 106, 107):
        _write(d, i, miss_frac=0.6, attached=False)
    return d


def _apply(window: Path, quar: Path) -> int:
    paths = qds.iter_shard_paths(window)
    verdicts = [qds.judge(p) for p in paths]
    return qds.do_apply(window, quar, verdicts, paths)


# --------------------------------------------------------------------------
# the gate itself


def test_judge_matches_the_shipped_reject_rule(window: Path) -> None:
    """`judge` must agree with the axis functions + thresholds it wraps.

    ⚑ SCOPE, stated exactly because an earlier version of this docstring
    overclaimed it: this recomputes the verdict from
    `chess_anti_engine.eval.value_optimism` and compares it to `judge()`. It
    therefore pins `judge()` against the AXIS FUNCTIONS AND THRESHOLDS ONLY.
    It does NOT import, execute or observe `scripts/value_optimism.py`, whose
    inline copy of the same reject rule is a third implementation — review
    demonstrated that breaking that copy (`> multipv_miss_max` -> `> 999.0`,
    which flips 118 of the 834 live shards) leaves this whole suite green.
    Closing that hole needs the shared predicate, not another assertion here.
    """
    import zarr
    for p in qds.iter_shard_paths(window):
        v = qds.judge(p)
        z = zarr.open_group(str(p), mode="r")
        lab = np.asarray(z["has_sf_wdl"][:]).astype(bool)
        att = sf_label_attachment_corr(
            np.asarray(z["x"][:, 0:12]), np.asarray(z["sf_wdl"][:]).astype(np.float64), lab,
        )
        miss = sf_multipv_missing_rate(np.asarray(z["has_sf_multipv_raw"][:]), lab)
        expect = (
            att.status != "ok" or att.value < SF_LABEL_ATTACHMENT_MIN
            or miss.status != "ok" or miss.value > SF_MULTIPV_MISS_MAX
        )
        assert v.reject is bool(expect), f"{p.name}: {v.reason}"


def test_a_shard_the_gate_cannot_evaluate_is_rejected(tmp_path: Path) -> None:
    """Non-ok status is a REJECT, not a pass."""
    d = tmp_path / "w"
    d.mkdir()
    _write(d, 1, n=5)  # below SF_AXIS_MIN_ROWS
    v = qds.judge(next(iter(qds.iter_shard_paths(d))))
    assert v.reject
    assert "too_few_rows" in v.reason


# --------------------------------------------------------------------------
# the zero-row marker exemption


def test_zero_row_reservation_is_not_data(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    res = window / "shard_000107.zarr"
    assert res.exists()
    assert shard_positions(res) == 0
    v = qds.judge(res)
    assert v.is_marker, "the marker must be recognised as a reservation"
    assert not v.reject, "the marker must not be re-quarantined"


def test_verify_passes_after_a_correct_apply(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    assert qds.do_verify(window, quar) == 0


# --------------------------------------------------------------------------
# S1: the reservation is a verify AXIS


def test_verify_FAILS_when_the_reservation_is_missing(window: Path, tmp_path: Path) -> None:
    """The state the mechanism exists to prevent must not read PASS.

    Simulates an --apply killed between the last move and the reservation
    install: shards gone, marker absent, every other axis clean.
    """
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    shutil.rmtree(window / "shard_000107.zarr")
    assert qds.do_verify(window, quar) == 1


def test_verify_FAILS_without_a_manifest(window: Path, tmp_path: Path) -> None:
    """An unchecked axis is not a pass."""
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    assert qds.do_verify(window, None) == 1


# --------------------------------------------------------------------------
# S5: the yardstick is two-sided


def test_verify_FAILS_on_over_removal(window: Path, tmp_path: Path) -> None:
    """Deleting a CLEAN shard must fail, though no dirt remains."""
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    assert qds.do_verify(window, quar) == 0
    shutil.rmtree(window / "shard_000100.zarr")
    assert qds.do_verify(window, quar) == 1


# --------------------------------------------------------------------------
# S2/S4: manifest binding and digest checking


def test_restore_refuses_a_foreign_shard_dir(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    other = tmp_path / "other_replay_shards"
    other.mkdir()
    assert qds.do_restore(other, quar) == 2
    assert not list(other.glob("shard_*.zarr")), "nothing may land in a foreign window"


def test_restore_refuses_a_tampered_shard(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    (quar / "shard_000105.zarr" / ".zgroup").write_text("GARBAGE")
    assert qds.do_restore(window, quar) == 2


def test_verify_FAILS_on_a_tampered_quarantined_shard(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    (quar / "shard_000106.zarr" / ".zgroup").write_text("GARBAGE")
    assert qds.do_verify(window, quar) == 1


def test_a_corrupt_manifest_is_an_error_not_a_traceback(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    (quar / qds.MANIFEST_NAME).write_text("{not json")
    assert qds.do_restore(window, quar) == 2
    assert qds.do_verify(window, quar) == 1


def test_missing_manifest_is_an_error_not_a_traceback(tmp_path: Path) -> None:
    d = tmp_path / "w"
    d.mkdir()
    q = tmp_path / "q"
    q.mkdir()
    assert qds.do_restore(d, q) == 2


# --------------------------------------------------------------------------
# S3/S6: restore is resumable, and the journal survives an interrupted apply


def test_restore_round_trips_and_is_idempotent(window: Path, tmp_path: Path) -> None:
    before = {p.name: qds.digest(p) for p in qds.iter_shard_paths(window)}
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    assert qds.do_restore(window, quar) == 0
    after = {p.name: qds.digest(p) for p in qds.iter_shard_paths(window)}
    assert before == after, "restore must be byte-identical"


def test_restore_resumes_after_a_partial_run(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    first = str(man["shards"][0]["name"])
    shutil.move(str(quar / first), str(window / first))  # simulate a half-done restore
    assert qds.do_restore(window, quar) == 0
    assert len(qds.iter_shard_paths(window)) == 8


def test_the_journal_lands_before_the_first_move(window: Path, tmp_path: Path) -> None:
    """A crash mid-move must leave a manifest, so --restore can clean up."""
    quar = tmp_path / "q"
    real_move = shutil.move
    calls: list[Any] = []

    def boom(src: Any, dst: Any) -> Any:
        calls.append(src)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated crash mid-move")
        return real_move(src, dst)

    shutil.move = boom
    try:
        with pytest.raises(KeyboardInterrupt):
            _apply(window, quar)
    finally:
        shutil.move = real_move

    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    assert man["complete"] is False, "an interrupted apply must not read complete"
    assert qds.do_restore(window, quar) == 0
    assert len(qds.iter_shard_paths(window)) == 8


def test_verify_FAILS_while_the_journal_is_incomplete(window: Path, tmp_path: Path) -> None:
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    man["complete"] = False
    (quar / qds.MANIFEST_NAME).write_text(json.dumps(man))
    assert qds.do_verify(window, quar) == 1


# --------------------------------------------------------------------------
# the reservation only needs the top id


def test_reserving_the_top_id_protects_every_lower_one(window: Path, tmp_path: Path) -> None:
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    buf = DiskReplayBuffer(
        capacity=10**9, shard_dir=window, rng=np.random.default_rng(0),
        read_only=True, shard_size=100,
    )
    assert buf._shard_index == 108, "the counter must not fall below the quarantined max"
    assert buf._next_free_shard_path().name == "shard_000108.zarr"
    assert buf._total_positions == 5 * 200, "the marker contributes no rows"
