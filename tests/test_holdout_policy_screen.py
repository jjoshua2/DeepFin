"""The offline policy screen's gates must be able to FAIL, and its training must happen.

Every test here was written against a mutant. The screen's central number is "how
much does this arm degrade held-out policy CE", and the ways that number can be
manufactured rather than measured are:

* the training loop silently not training -- every arm then reports exactly zero
  degradation and nothing else in the script notices (`run_training`);
* the eval set being contaminated or having no labels at all, behind a gate that
  cannot fire (`EvalSet.gate`);
* the train pool sharing games with the eval set, which no code used to check
  (`assert_pools_game_disjoint`);
* part of the pool never reaching the buffer while the manifest's count is
  printed anyway (`link_name` / `assert_pool_loaded`);
* the negative control permuting a column the loss never reads (`ShuffleTargets`);
* the "full" pass covering only some rows (`full_pass_rows`);
* the `base` mask being accepted and ignored (`summarize`);
* the per-row CE being paired with the WRONG game ids, which leaves every arm
  mean untouched and only narrows the CI (`EvalSet._buffer_game_ids`);
* `--verify-production-metric` reporting a rig that has stopped tracking
  production's metric, and banking the arms anyway
  (`assert_production_metric_matches`).
"""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    POLICY_SIZE,
    SF_MULTIPV_RAW_MAX,
    iter_shard_paths,
    samples_to_arrays,
    save_local_shard_arrays,
)

_SPEC = importlib.util.spec_from_file_location(
    "holdout_policy_screen",
    Path(__file__).resolve().parents[1] / "scripts" / "holdout_policy_screen.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_PLANES = 8


def _write_shard(path: Path, *, n: int, game_ids: list[int],
                 labelled: bool = True, desync: bool = False) -> Path:
    """A tiny real shard. `labelled` sets sf_wdl; `desync` withholds sf_multipv_raw."""
    rng = np.random.default_rng(len(str(path)))
    samples = []
    for i in range(n):
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[i % POLICY_SIZE] = 1.0
        samples.append(ReplaySample(
            x=rng.random((_PLANES, 8, 8)).astype(np.float32),
            policy_target=pol,
            wdl_target=i % 3,
            game_id=game_ids[i % len(game_ids)],
            sf_wdl=np.asarray([0.4, 0.3, 0.3], np.float32) if labelled else None,
            sf_multipv_raw=(None if (desync or not labelled)
                            else np.zeros((SF_MULTIPV_RAW_MAX, 5), np.int16)),
        ))
    return save_local_shard_arrays(path, arrs=samples_to_arrays(samples))


# --------------------------------------------------------------------------
# F1: shard indices are reused across salvage bundles, so the basename is not a key.
# --------------------------------------------------------------------------

def test_link_dir_keeps_two_bundles_reusing_one_shard_index_apart(tmp_path: Path) -> None:
    """The bug that dropped 707 of A1_BIG's 5,244 shards (13.6% of its rows).

    Both sources are named `shard_000042.zarr`; linking by basename links the
    first and silently skips the second.
    """
    a = _write_shard(tmp_path / "bundleA" / "shard_000042.zarr", n=3, game_ids=[1])
    b = _write_shard(tmp_path / "bundleB" / "shard_000042.zarr", n=5, game_ids=[2])
    assert a.name == b.name

    dst, linked = _MOD.link_dir([str(a), str(b)], tmp_path / "links")

    assert len(linked) == 2, "a reused shard index must not collapse two distinct sources"
    assert {p.resolve() for p in linked.values()} == {a.resolve(), b.resolve()}
    # The buffer only ever sees what its own glob finds, so the names must survive it.
    found = iter_shard_paths(dst)
    assert len(found) == 2
    assert {Path(p).resolve() for p in found} == {a.resolve(), b.resolve()}


def test_link_names_sort_in_shard_index_order(tmp_path: Path) -> None:
    """`iter_shard_paths` sorts lexicographically and that order is the pool's recency."""
    srcs = [_write_shard(tmp_path / f"b{i}" / f"shard_{idx:06d}.zarr", n=1, game_ids=[i])
            for i, idx in enumerate([900001, 33118, 7])]
    dst, _ = _MOD.link_dir([str(p) for p in srcs], tmp_path / "links")
    from chess_anti_engine.replay.shard import shard_index
    got = [shard_index(p) for p in iter_shard_paths(dst)]
    assert got == sorted(got) == [7, 33118, 900001]


def test_link_dir_aborts_if_two_sources_ever_claim_one_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest makes this unreachable in practice; the guard must still bite.

    Forcing every source onto one name is the only way to observe it, and a
    guard that cannot be observed to fire is the thing this whole file is about.
    """
    a = _write_shard(tmp_path / "bundleA" / "shard_000042.zarr", n=3, game_ids=[1])
    b = _write_shard(tmp_path / "bundleB" / "shard_000043.zarr", n=5, game_ids=[2])
    monkeypatch.setattr(_MOD, "link_name", lambda _src: "shard_000000000_x.zarr")
    with pytest.raises(SystemExit, match="claimed by both"):
        _MOD.link_dir([str(a), str(b)], tmp_path / "links")


def test_link_dir_collapses_a_path_listed_twice(tmp_path: Path) -> None:
    a = _write_shard(tmp_path / "bundleA" / "shard_000001.zarr", n=2, game_ids=[1])
    _dst, linked = _MOD.link_dir([str(a), str(a)], tmp_path / "links")
    assert len(linked) == 1


def test_assert_pool_loaded_is_loud_when_rows_go_missing(tmp_path: Path) -> None:
    src_a, src_b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    linked = {"shard_000000000_aa.zarr": src_a, "shard_000000001_bb.zarr": src_b}
    rows = {src_a: 100, src_b: 55}
    found = [tmp_path / n for n in linked]

    assert _MOD.assert_pool_loaded(linked, rows, found, 155) == 155
    with pytest.raises(SystemExit, match="never entered the pool"):
        _MOD.assert_pool_loaded(linked, rows, found, 100)
    with pytest.raises(SystemExit, match="glob found"):
        _MOD.assert_pool_loaded(linked, rows, found[:1], 155)


def test_pool_row_counts_match_the_buffer_end_to_end(tmp_path: Path) -> None:
    """The reconciliation the screen actually runs, against a real DiskReplayBuffer."""
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    a = _write_shard(tmp_path / "bundleA" / "shard_000042.zarr", n=7, game_ids=[1])
    b = _write_shard(tmp_path / "bundleB" / "shard_000042.zarr", n=11, game_ids=[2])
    paths = [str(a), str(b)]

    gid, rows_by_src = _MOD.read_pool_game_ids(paths)
    assert gid.size == 18
    dst, linked = _MOD.link_dir(paths, tmp_path / "links")
    buf = DiskReplayBuffer(10**9, shard_dir=dst, rng=np.random.default_rng(0),
                           read_only=True, input_planes=_PLANES)
    try:
        assert _MOD.assert_pool_loaded(linked, rows_by_src, iter_shard_paths(dst), len(buf)) == 18
        assert len(buf) == 18, "both same-index shards must be in the pool, not just the first"
    finally:
        buf.close()


# --------------------------------------------------------------------------
# F5 / desync gate.
# --------------------------------------------------------------------------

def _eval_set(paths: list[Path], name: str = "ev") -> Any:
    return _MOD.EvalSet(name, [str(p) for p in paths], _PLANES)


def test_clean_eval_set_is_accepted(tmp_path: Path) -> None:
    p = _write_shard(tmp_path / "shard_000001.zarr", n=6, game_ids=[10, 11])
    ev = _eval_set([p])
    assert ev.labelled == 6
    assert ev.desync_rows == 0
    assert ev.desync_frac == 0.0


def test_desync_gate_fires_on_a_contaminated_eval_set(tmp_path: Path) -> None:
    p = _write_shard(tmp_path / "shard_000001.zarr", n=6, game_ids=[10], desync=True)
    with pytest.raises(SystemExit, match="not exactly 0"):
        _eval_set([p])


def test_eval_set_with_zero_labelled_rows_aborts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`des / lab if lab else 0.0` let an unlabelled set PASS the contamination gate.

    The printed reading must say `nan`, not `0.000000`: the banked number is what
    a human reads, and a 0.0 there is indistinguishable from a clean set.
    """
    p = _write_shard(tmp_path / "shard_000001.zarr", n=6, game_ids=[10], labelled=False)
    with pytest.raises(SystemExit, match="no denominator"):
        _eval_set([p])
    assert "DESYNC_FRAC=nan" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The CE <-> game_id pairing, which is what the paired CI clusters on.
# --------------------------------------------------------------------------

def test_eval_game_ids_are_the_buffers_own_column_in_its_own_order(tmp_path: Path) -> None:
    """The positional pairing `full_pass_rows` relies on, proved element-wise.

    `EvalSet.game_id` is a SEPARATE read of the shard column; CE comes out in
    the buffer's row order. Nothing else in the screen ever compares them.
    """
    a = _write_shard(tmp_path / "a" / "shard_000001.zarr", n=5, game_ids=[11, 12, 13])
    b = _write_shard(tmp_path / "b" / "shard_000002.zarr", n=7, game_ids=[21, 22])
    ev = _eval_set([a, b])

    from chess_anti_engine.replay.buffer import ArrayReplayBuffer
    buf = cast(ArrayReplayBuffer, ev.buf)
    order = np.concatenate([np.asarray(buf.rows_slice_arrays(lo, hi)["game_id"])
                            for lo, hi in buf.batch_row_bounds(4)])
    assert order.size == 12
    assert np.array_equal(order, ev.game_id)
    # ...and the shards' own concatenation order, oldest shard first
    assert np.array_equal(ev.game_id[:5], np.asarray([11, 12, 13, 11, 12]))


def test_eval_set_aborts_when_the_buffer_order_diverges_from_the_shard_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count check this replaced could not see this, and it is the harmful one.

    Same rows, same count, permuted pairing: every arm mean is unchanged and the
    game clusters drift toward independence, so `holdout_paired_ci.py` returns a
    NARROWER interval -- the direction that makes every verdict look stronger.
    Restoring `if len(buf) != self.game_id.size` in place of the identity check
    makes this test pass silently, which is why it is written as a permutation
    of a FIXED-SIZE array rather than a truncation.
    """
    p = _write_shard(tmp_path / "shard_000001.zarr", n=6, game_ids=[10, 11, 12])
    monkeypatch.setattr(
        _MOD.EvalSet, "_buffer_game_ids",
        staticmethod(lambda buf: np.asarray([12, 10, 11, 12, 10, 11], np.int64)))
    with pytest.raises(SystemExit, match="not in the buffer's row order"):
        _eval_set([p])


def test_eval_set_aborts_when_the_buffer_holds_fewer_rows_than_were_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capacity that stops covering the set silently drops the OLDEST rows.

    Nothing else notices: `full_pass_rows` covers `len(ev.buf)`, so its own
    coverage assertion still passes, and every CE lands against a game id
    belonging to a row that was evicted.
    """
    from chess_anti_engine.replay.buffer import ArrayReplayBuffer

    p = _write_shard(tmp_path / "shard_000001.zarr", n=6, game_ids=[10, 11])
    monkeypatch.setattr(
        _MOD, "ArrayReplayBuffer",
        lambda _cap, rng: ArrayReplayBuffer(4, rng=rng))
    with pytest.raises(SystemExit, match="not in the buffer's row order"):
        _eval_set([p])


def test_eval_set_refuses_a_shard_without_game_ids(tmp_path: Path) -> None:
    """`arrs["game_id"]` raised a bare KeyError -- the column CAN be absent.

    `load_shard_arrays` materializes only the fields the zarr group holds, so a
    shard written from samples that carry no game id has no column at all --
    exactly like the `has_sf_multipv_raw` case handled four lines above it.
    """
    rng = np.random.default_rng(0)
    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
    pol[0] = 1.0
    arrs = samples_to_arrays([ReplaySample(
        x=rng.random((_PLANES, 8, 8)).astype(np.float32), policy_target=pol, wdl_target=0,
        sf_wdl=np.asarray([0.4, 0.3, 0.3], np.float32),
        sf_multipv_raw=np.zeros((SF_MULTIPV_RAW_MAX, 5), np.int16))])
    p = save_local_shard_arrays(tmp_path / "shard_000001.zarr", arrs=arrs)
    # the column really is absent from the written shard, not merely all-zero
    assert "game_id" not in _MOD.load_shard_arrays(str(p), lazy=False)[0]
    with pytest.raises(SystemExit, match="no game_id column"):
        _eval_set([p])


def test_eval_set_aborts_when_the_buffer_drops_the_game_id_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`prune_storage_arrays` drops an optional value whose has-flag is all-False.

    The shard read would then succeed while the buffer holds no game ids at all,
    so there is nothing to pair the CE against.
    """
    p = _write_shard(tmp_path / "shard_000001.zarr", n=4, game_ids=[10, 11])
    real = _MOD.load_shard_arrays

    def _blank_flag(path: Any, **kw: Any) -> Any:
        arrs, meta = real(path, **kw)
        arrs["has_game_id"] = np.zeros_like(np.asarray(arrs["has_game_id"]))
        return arrs, meta

    monkeypatch.setattr(_MOD, "load_shard_arrays", _blank_flag)
    with pytest.raises(SystemExit, match="dropped its game_id column"):
        _eval_set([p])


# --------------------------------------------------------------------------
# F3: game disjointness, which nothing in the shipped script used to check.
# --------------------------------------------------------------------------

def test_game_leak_gate_fires_when_a_game_is_shared(tmp_path: Path) -> None:
    ev = _eval_set([_write_shard(tmp_path / "shard_000001.zarr", n=4, game_ids=[7, 8])])
    with pytest.raises(SystemExit, match="leaking"):
        _MOD.assert_pools_game_disjoint(np.asarray([5, 6, 7], np.int64), [ev])


def test_game_leak_gate_passes_a_disjoint_pool(tmp_path: Path) -> None:
    ev = _eval_set([_write_shard(tmp_path / "shard_000001.zarr", n=4, game_ids=[7, 8])])
    _MOD.assert_pools_game_disjoint(np.asarray([5, 6, 9], np.int64), [ev])


def test_prepare_train_pool_aborts_before_linking_a_leaking_pool(tmp_path: Path) -> None:
    """The gate is inside the only function that can produce a link dir.

    Deleting it from `main` instead is not an option: `main` has nowhere else to
    get `tdir` from. And nothing is linked on the failing path, so an aborted run
    leaves no half-built directory behind to be picked up by the next one.
    """
    shared = _write_shard(tmp_path / "ev" / "shard_000001.zarr", n=4, game_ids=[7, 8])
    ev = _eval_set([shared])
    train = _write_shard(tmp_path / "tr" / "shard_000002.zarr", n=4, game_ids=[8, 9])
    dst = tmp_path / "links"
    with pytest.raises(SystemExit, match="leaking"):
        _MOD.prepare_train_pool([str(train)], [ev], dst)
    assert not dst.exists() or not list(dst.iterdir())


def test_prepare_train_pool_links_a_disjoint_pool(tmp_path: Path) -> None:
    ev = _eval_set([_write_shard(tmp_path / "ev" / "shard_000001.zarr", n=4, game_ids=[7, 8])])
    train = _write_shard(tmp_path / "tr" / "shard_000002.zarr", n=6, game_ids=[9, 10])
    tdir, linked, rows = _MOD.prepare_train_pool([str(train)], [ev], tmp_path / "links")
    assert len(linked) == 1
    assert rows[train.resolve()] == 6
    assert len(iter_shard_paths(tdir)) == 1


def test_read_pool_game_ids_refuses_a_shard_without_game_ids(tmp_path: Path) -> None:
    """No game_id column means the leak gate cannot be enforced -- that must abort."""
    rng = np.random.default_rng(0)
    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
    pol[0] = 1.0
    arrs = samples_to_arrays([ReplaySample(
        x=rng.random((_PLANES, 8, 8)).astype(np.float32), policy_target=pol, wdl_target=0)])
    p = save_local_shard_arrays(tmp_path / "shard_000001.zarr", arrs=arrs)
    with pytest.raises(SystemExit, match="leak gate cannot be enforced"):
        _MOD.read_pool_game_ids([str(p)])


# --------------------------------------------------------------------------
# The negative control has to reach the column the loss reads.
# --------------------------------------------------------------------------

class _FakeSampler:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.calls: list[tuple[int, bool]] = []

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict:
        n = int(batch_size)
        self.calls.append((n, bool(wdl_balance)))
        return {
            "policy_target": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
            "priority": np.arange(n, dtype=np.float32),
            "wdl_target": np.zeros(n, np.int8),
        }


def test_shuffle_targets_permutes_policy_target_and_nothing_else() -> None:
    inner = _FakeSampler()
    sh = _MOD.ShuffleTargets(inner, np.random.default_rng(3))
    ref = inner.sample_batch_arrays(8)
    out = sh.sample_batch_arrays(8)

    pt_in, pt_out = ref["policy_target"], out["policy_target"]
    assert pt_out.shape == pt_in.shape
    # a permutation of the same rows...
    assert sorted(pt_out[:, 0].tolist()) == sorted(pt_in[:, 0].tolist())
    # ...with no row left in place, which is what makes it a real control
    assert not np.any(pt_out[:, 0] == pt_in[:, 0])
    assert np.array_equal(out["priority"], ref["priority"]), "only policy_target may move"
    # the wrapper must not quietly change how the inner buffer is drawn from
    assert inner.calls == [(8, True), (8, True)]


# --------------------------------------------------------------------------
# The training must actually happen.
# --------------------------------------------------------------------------

class _FakeTrainer:
    """Records train_steps calls; `advance` is how many steps it really takes."""

    def __init__(self, advance_per_call: int | None = None) -> None:
        self.step = 1000
        self.calls: list[tuple[Any, int, int]] = []
        self._advance = advance_per_call

    def train_steps(self, buf: Any, *, batch_size: int, steps: int) -> str:
        self.calls.append((buf, int(batch_size), int(steps)))
        self.step += int(steps) if self._advance is None else self._advance
        return f"metrics@{steps}"


def test_run_training_chunks_and_returns_the_last_metrics() -> None:
    tr = _FakeTrainer()
    sampler = object()
    out = _MOD.run_training(cast(Any, tr), sampler, steps=200, chunk_steps=88,
                            batch_size=512, label="arm")
    assert [c[2] for c in tr.calls] == [88, 88, 24]
    assert sum(c[2] for c in tr.calls) == 200
    # the SAMPLER reaches the trainer -- with --shuffle-targets that is the
    # wrapper, and a screen that trained off the raw buffer instead would have
    # no negative control at all
    assert all(c[0] is sampler and c[1] == 512 for c in tr.calls)
    assert tr.step == 1200
    assert out == "metrics@24"


def test_run_training_aborts_when_no_optimizer_step_is_taken() -> None:
    """M8: the mutation that makes every arm report ZERO degradation."""
    tr = _FakeTrainer(advance_per_call=0)
    with pytest.raises(SystemExit, match="training did not happen"):
        _MOD.run_training(cast(Any, tr), None, steps=100, chunk_steps=88,
                          batch_size=512, label="arm")


def test_run_training_aborts_on_a_short_count() -> None:
    tr = _FakeTrainer(advance_per_call=1)
    with pytest.raises(SystemExit, match="advanced 2, expected 100"):
        _MOD.run_training(cast(Any, tr), None, steps=100, chunk_steps=88,
                          batch_size=512, label="arm")


# --------------------------------------------------------------------------
# The full pass must cover every row, and must honour the `base` mask.
# --------------------------------------------------------------------------

class _IdentityPolicy(torch.nn.Module):
    """forward(x) -> {"policy": x}: the batch carries the logits directly."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"policy": x}


class _FakeFullPassTrainer:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.model = _IdentityPolicy()
        self._batches = batches
        self.seen: list[tuple[Any, int]] = []

    def _iter_full_pass_batches(self, buf: Any, *, batch_size: int):
        self.seen.append((buf, int(batch_size)))
        yield from self._batches


class _FakeEvalSet:
    def __init__(self, rows: int, game_id: np.ndarray) -> None:
        self.buf = [0] * rows
        self.game_id = game_id


def _one_hot(idx: list[int], width: int) -> torch.Tensor:
    t = torch.zeros(len(idx), width)
    t[torch.arange(len(idx)), torch.as_tensor(idx)] = 1.0
    return t


def test_full_pass_row_values_and_base_mask() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    tgt = _one_hot([0, 2, 2], 3)
    batch = {"x": logits, "policy_t": tgt,
             "is_network_turn": torch.tensor([1.0, 1.0, 0.0]),
             "has_policy": torch.tensor([1.0, 1.0, 1.0])}
    tr = _FakeFullPassTrainer([batch])
    ev = _FakeEvalSet(3, np.asarray([1, 1, 2], np.int64))

    r = _MOD.full_pass_rows(cast(Any, tr), cast(Any, ev), 3)

    # the eval buffer and the eval batch size are what production's own
    # fixed-order iterator gets -- not a resampled lookalike
    assert tr.seen == [(ev.buf, 3)]
    expect_ce = (-torch.log_softmax(logits, -1)[torch.arange(3), torch.as_tensor([0, 2, 2])])
    assert np.allclose(r["ce"], expect_ce.numpy(), atol=1e-6)
    assert np.array_equal(r["base"], np.asarray([1.0, 1.0, 0.0], np.float32))
    assert np.allclose(r["floor"], 0.0, atol=1e-6)          # one-hot targets
    assert np.array_equal(r["agree"], np.asarray([1.0, 0.0, 1.0], np.float32))

    # T=1.0 in the free temperature scan must reproduce the base-weighted mean CE.
    t1 = float(r["temp_ce"][list(r["temps"]).index(1.0)])
    assert t1 == pytest.approx(float(expect_ce[:2].mean()), abs=1e-5)


def test_full_pass_refuses_to_cover_only_some_rows() -> None:
    """M7: a 'full' pass that quietly stops being one."""
    batch = {"x": torch.zeros(2, 3), "policy_t": _one_hot([0, 1], 3)}
    tr = _FakeFullPassTrainer([batch])
    ev = _FakeEvalSet(5, np.zeros(5, np.int64))
    with pytest.raises(SystemExit, match="full pass covered 2 of 5"):
        _MOD.full_pass_rows(cast(Any, tr), cast(Any, ev), 2)


def test_verify_production_metric_is_a_gate_and_not_a_print() -> None:
    """It computed the deltas, wrote verify.json, and proceeded regardless.

    The numbers are the measured ones (banked iter-478 `trainer.pt`): the bf16
    probe tracks production to within a few 1e-5 at worst, and the defect the
    gate exists to catch -- the eval forward leaving `_amp_context()` -- shows up
    as the ~2.1e-4 fp32 residual. Both bounds of the tolerance are asserted, so
    widening it to swallow the defect or narrowing it onto the noise both fail.
    """
    prod = 1.1004884

    # accepted: the largest benign reading ever observed (evalB 3 shards @512,
    # production's first eval_full_pass in a fresh process)
    _MOD.assert_production_metric_matches(prod, prod + 5.409e-05, prod - 2.311e-04)
    # accepted: steady state
    _MOD.assert_production_metric_matches(prod, prod + 6.697e-08, prod - 2.732e-04)

    # rejected: the rig scoring fp32 while production scores bf16 -- the
    # SMALLEST such residual measured on any real eval set
    with pytest.raises(SystemExit, match="no longer measures production's metric"):
        _MOD.assert_production_metric_matches(prod, prod - 2.146e-04, prod - 2.146e-04)
    # rejected: production stopped reporting policy_loss at all
    with pytest.raises(SystemExit):
        _MOD.assert_production_metric_matches(float("nan"), prod, prod)

    assert 5.409e-05 < _MOD._PRODUCTION_METRIC_TOL < 2.146e-04


class _ScaledPolicy(torch.nn.Module):
    """forward(x) -> {"policy": x * scale}. `scale` is the fixture's "precision".

    A fake `_amp_context` that only counts its entries leaves the fp32 and bf16
    probes numerically IDENTICAL, and then no test can tell which one the call
    site hands to the gate -- swapping the two arguments passes the whole suite.
    Perturbing the forward is what makes the wiring observable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"policy": x * self.scale}


# Chosen so the fixture's bf16-minus-fp32 residual is +2.73e-4 -- the real one
# measured on the full evalB (rig fp32 1.1002153 vs rig bf16 1.1004884).
_AMP_SCALE = 0.999358


class _FakeVerifyTrainer(_FakeFullPassTrainer):
    """`_FakeFullPassTrainer` plus the two members the verification calls."""

    def __init__(self, batches: list[dict[str, torch.Tensor]], prod: float) -> None:
        super().__init__(batches)
        self.model = _ScaledPolicy()
        self._prod = prod
        self.amp_entered = 0
        self.prod_calls: list[tuple[Any, int]] = []

    def eval_full_pass(self, buf: Any, *, batch_size: int) -> Any:
        self.prod_calls.append((buf, int(batch_size)))
        return SimpleNamespace(policy_loss=self._prod)

    @contextmanager
    def _amp_context(self) -> Generator[None]:
        self.amp_entered += 1
        self.model.scale = _AMP_SCALE
        try:
            yield
        finally:
            self.model.scale = 1.0


def _fixture_ce(scale: float) -> float:
    lg = _VERIFY_LOGITS * scale
    return float((-torch.log_softmax(lg, -1)[torch.arange(2), torch.as_tensor([0, 1])]).mean())


_VERIFY_LOGITS = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _verify_fixture(prod_offset: float) -> tuple[_FakeVerifyTrainer, _FakeEvalSet]:
    """A rig whose bf16 probe sits `prod_offset` from production's metric.

    The fp32 probe sits a further +2.73e-4 away, as it does in reality, so the
    two probes are NOT interchangeable and the call site's argument order is
    under test rather than merely its interior.
    """
    batch = {"x": _VERIFY_LOGITS, "policy_t": _one_hot([0, 1], 3)}
    tr = _FakeVerifyTrainer([batch], _fixture_ce(_AMP_SCALE) + prod_offset)
    return tr, _FakeEvalSet(2, np.asarray([1, 2], np.int64))


def test_the_verify_fixture_reproduces_the_real_precision_residual() -> None:
    """Without this the two probes are one number and the wiring is untestable."""
    gap = _fixture_ce(_AMP_SCALE) - _fixture_ce(1.0)
    assert gap == pytest.approx(2.73e-04, rel=0.02)
    assert gap > _MOD._PRODUCTION_METRIC_TOL, "the fp32 probe must be OUTSIDE the tolerance"


def test_verify_production_metric_dumps_then_aborts(tmp_path: Path) -> None:
    """The abort lives inside the only function that produces the verification.

    Measuring, printing and gating are one call, so deleting the gate from
    `main` is not an option -- `main` has nowhere else to get the numbers from.
    And `verify.json` must be on disk BEFORE the abort: a run that fails this
    gate is exactly the run whose numbers someone will want to look at.
    """
    tr, ev = _verify_fixture(2.146e-04)
    with pytest.raises(SystemExit, match="no longer measures production's metric"):
        _MOD.verify_production_metric(cast(Any, tr), cast(Any, ev), 2, tmp_path)
    rec = json.loads((tmp_path / "verify.json").read_text())
    assert rec["abort_bound"] == _MOD._PRODUCTION_METRIC_TOL
    assert rec["delta_vs_bf16"] == pytest.approx(2.146e-04, rel=1e-3)
    # the two probes are distinct, and each is labelled with the one it is
    assert rec["probe_ce_bf16"] == pytest.approx(_fixture_ce(_AMP_SCALE), abs=1e-9)
    assert rec["probe_ce_fp32"] == pytest.approx(_fixture_ce(1.0), abs=1e-9)
    # the bf16 probe really was taken under the amp context, not relabelled
    assert tr.amp_entered == 1
    # and production's metric was read off the SAME buffer at the SAME shape
    assert tr.prod_calls == [(ev.buf, 2)]


def test_verify_production_metric_passes_a_matching_rig(tmp_path: Path) -> None:
    """N7: the call site must hand the gate the BF16 probe, not the fp32 one.

    This is the discriminating case for the wiring. The rig is healthy -- its
    bf16 probe is 5.4e-5 from production, inside the bound -- but its fp32 probe
    is a further 2.73e-4 away, well outside it. So swapping the two arguments at
    the call site turns this passing run into an abort. Every other case in the
    file aborts either way, and the interior of `assert_production_metric_matches`
    cannot see its own caller.
    """
    tr, ev = _verify_fixture(5.409e-05)
    rec = _MOD.verify_production_metric(cast(Any, tr), cast(Any, ev), 2, tmp_path)
    assert rec["delta_vs_bf16"] == pytest.approx(5.409e-05, rel=1e-3)
    assert abs(rec["delta_vs_fp32"]) > _MOD._PRODUCTION_METRIC_TOL, (
        "the fp32 probe must be outside the bound, or the argument order is untested")
    assert json.loads((tmp_path / "verify.json").read_text())["probe_ce_bf16"] == rec["probe_ce_bf16"]


def test_summarize_excludes_rows_the_base_mask_drops() -> None:
    """M9: `base` read and then ignored."""
    r = {"ce": np.asarray([1.0, 2.0, 100.0], np.float32),
         "floor": np.asarray([0.5, 0.5, 50.0], np.float32),
         "agree": np.asarray([1.0, 0.0, 1.0], np.float32),
         "base": np.asarray([1.0, 1.0, 0.0], np.float32)}
    s = _MOD.summarize(r)
    assert s["rows"] == 2
    assert s["ce"] == pytest.approx(1.5)
    assert s["excess"] == pytest.approx(1.0)
    assert s["agree"] == pytest.approx(0.5)
