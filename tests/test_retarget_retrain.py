"""Guards for the offline retarget/retrain driver and the strict-load path.

The script's whole value is A/B integrity: the model must be the exact
trained net (not a partial random init) and every arm must differ only in
the knob under test. These tests pin the guardrails that enforce that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.retarget_retrain as rr
from chess_anti_engine.model import load_state_dict_tolerant
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from scripts.retarget_retrain import _parse_variant


def _replay_sample() -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy, wdl_target=0, priority=1.0, has_policy=True,
    )


def _patch_first_shard(monkeypatch, planes: int) -> None:
    monkeypatch.setattr(rr, "iter_shard_paths", lambda _d: [Path("fake.zarr")])
    monkeypatch.setattr(
        rr, "load_shard_arrays",
        lambda _p, **_kw: ({"x": np.zeros((1, planes, 8, 8), np.float32)}, {}),
    )


def _tiny_net() -> ChessNet:
    cfg = TransformerConfig(
        in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False,
    )
    return ChessNet(cfg).eval()


def test_parse_variant_rejects_rebuild_sf_targets_override() -> None:
    # Per-variant control would let two arms train on different targets while
    # claiming to A/B a single knob; only the global CLI flag may set it.
    with pytest.raises(SystemExit, match="rebuild_sf_targets"):
        _parse_variant("sneaky:rebuild_sf_targets=false")


def test_parse_variant_coerces_types() -> None:
    name, overrides = _parse_variant(
        "arm:replay_sf_gap_priority_weight=5,replay_upgrade_v1_planes=true,note=abc"
    )
    assert name == "arm"
    assert overrides == {
        "replay_sf_gap_priority_weight": 5.0,
        "replay_upgrade_v1_planes": True,
        "note": "abc",
    }


def test_require_complete_accepts_exact_state_dict() -> None:
    src = _tiny_net()
    dst = _tiny_net()
    load_state_dict_tolerant(
        dst, src.state_dict(), label="test", require_complete=True,
    )
    for k, v in dst.state_dict().items():
        assert torch.equal(v, src.state_dict()[k])


def test_require_complete_raises_on_missing_key() -> None:
    src = _tiny_net()
    dst = _tiny_net()
    state = dict(src.state_dict())
    dropped = next(iter(state))
    del state[dropped]

    # Default tolerant mode: single missing key passes (logged only).
    load_state_dict_tolerant(dst, dict(state), label="test")

    with pytest.raises(RuntimeError, match="require_complete"):
        load_state_dict_tolerant(
            dst, dict(state), label="test", require_complete=True,
        )


def test_plane_guard_passes_on_exact_match(monkeypatch) -> None:
    _patch_first_shard(monkeypatch, 175)
    rr._assert_replay_planes_match(Path("x"), 175, upgrade_v1=False)  # no raise


def test_plane_guard_fires_on_silent_zeropad(monkeypatch) -> None:
    # stored (146) < target (175): the buffer would silently zero-pad the 29
    # threat planes and report success — the guard must abort loudly instead.
    _patch_first_shard(monkeypatch, 146)
    with pytest.raises(SystemExit, match="ZERO-PADDED"):
        rr._assert_replay_planes_match(Path("x"), 175, upgrade_v1=False)


def test_plane_guard_allows_v1_upgrade(monkeypatch) -> None:
    # v1 (146) shards with replay_upgrade_v1_planes on are recomputed to the 29
    # threat planes — an intended path, not a silent zero-pad.
    _patch_first_shard(monkeypatch, 146)
    rr._assert_replay_planes_match(Path("x"), 175, upgrade_v1=True)  # no raise


def test_plane_guard_fires_on_wider_shards(monkeypatch) -> None:
    # stored (175) > target (146): wider shards would be rejected -> empty pool.
    _patch_first_shard(monkeypatch, 175)
    with pytest.raises(SystemExit, match="wider than the arch"):
        rr._assert_replay_planes_match(Path("x"), 146, upgrade_v1=False)


def test_plane_guard_quiet_on_empty_dir(monkeypatch) -> None:
    # No shards: defer to the empty-buffer guard, don't raise here.
    monkeypatch.setattr(rr, "iter_shard_paths", lambda _d: [])
    rr._assert_replay_planes_match(Path("x"), 175, upgrade_v1=False)  # no raise


def test_the_buffer_is_built_with_a_deterministic_refresh(monkeypatch, tmp_path) -> None:
    """``deterministic_refresh=True`` must reach the CALL SITE, not just exist.

    The flag is the entire reason this ruler is paired. The default refresh
    picks between an async and a synchronous shuffle-pool refresh by who won a
    race and only the synchronous one advances ``self.rng``, so one lost race
    permanently desynchronises an arm's draw sequence from every other arm's --
    measured on the sibling probe as 3 distinct sequences from 15 identical
    invocations, 0.0056 nats of held-out CE apart. Every delta this script
    reports rests on "each variant is seeded identically, so the buffer's random
    draws match across arms".

    ``tests/test_replay_deterministic_refresh.py`` pins what the flag DOES;
    nothing pinned that this script passes it, and deleting the line left all 13
    tests here green -- a knob that never reaches the worker, one level up.
    """
    seen: dict = {}

    class _Reached(Exception):
        """Stop before the Trainer: the call site is the whole assertion."""

    def _spy(*_args, **kwargs):
        seen.update(kwargs)
        raise _Reached

    class _Cfg:
        input_extra_features = "v2_threats"

    monkeypatch.setattr(rr, "DiskReplayBuffer", _spy)
    monkeypatch.setattr(rr, "model_config_from_arch", lambda _a: _Cfg())
    monkeypatch.setattr(rr, "build_model", lambda _cfg: _tiny_net())
    monkeypatch.setattr(rr, "load_state_dict_tolerant", lambda *a, **k: None)
    monkeypatch.setattr(rr, "trainer_kwargs_from_config", lambda _c: {})
    monkeypatch.setattr(rr, "Trainer", lambda *a, **k: object())
    monkeypatch.setattr(rr, "_assert_replay_planes_match", lambda *a, **k: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}, "arch": {}})

    with pytest.raises(_Reached):
        rr._run_variant(
            name="base", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=1, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=None,
        )
    assert seen.get("deterministic_refresh") is True, (
        "retarget_retrain builds its replay buffer WITHOUT "
        "deterministic_refresh=True; the arms no longer share a draw sequence "
        f"and every variant delta this script prints is unpaired. kwargs: {sorted(seen)}"
    )
    # The paired claim is also made in prose at the top of the module; a reader
    # who trusts it is trusting the line above.
    assert "deterministic_refresh=True" in (rr.__doc__ or "")


def _shards(*names: str) -> list[Path]:
    return [Path("/pool") / n for n in names]


def test_shard_guard_passes_on_an_unchanged_pool() -> None:
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    rr._assert_shards_unchanged(list(snap), snap, name="base")  # no raise


def test_shard_guard_fires_when_a_shard_is_ADDED_mid_sweep() -> None:
    # The live case: --replay-dir is the running window (or a pool being topped
    # up), variants run sequentially, each is a full retrain, so arm 2 scans a
    # pool arm 1 never saw. Measured at --steps 800, same seed, flag on:
    # +1 shard moved 617/800 sampled rows (77.1%).
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    grown = snap + _shards("shard_000003.zarr")
    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._assert_shards_unchanged(grown, snap, name="sharp")


def test_shard_guard_fires_when_a_shard_is_REMOVED_mid_sweep() -> None:
    # Eviction/quarantine is the other direction: -1 shard moved 343/800 (42.9%).
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._assert_shards_unchanged(snap[:1], snap, name="smooth")


def test_shard_guard_fires_on_reordering_alone() -> None:
    # Same SET, different scan order: the buffer's draw sequence indexes the
    # ordered pool, so order alone de-pairs the arms.
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    with pytest.raises(SystemExit, match="different scan order"):
        rr._assert_shards_unchanged(list(reversed(snap)), snap, name="base")


def test_shard_guard_names_the_shards_that_moved() -> None:
    snap = _shards("shard_000001.zarr")
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(_shards("shard_000009.zarr"), snap, name="x")
    msg = str(ei.value)
    assert "shard_000009.zarr" in msg
    assert "shard_000001.zarr" in msg


def test_shard_guard_argument_ORDER_is_pinned() -> None:
    """X1: the two lists are same-typed, so a swap is silent unless pinned.

    Every other test here passes ``(observed, snapshot)`` positionally and only
    requires SOME abort -- which a swapped call still produces, with added and
    removed inverted and the two counts the wrong way round. That sends the
    operator hunting for a shard that was DELETED when one was in fact ADDED.
    Assert the direction, not merely the raise.
    """
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    grown = snap + _shards("shard_000003.zarr")
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(grown, snap, name="sharp")
    msg = str(ei.value)
    assert "1 added (shard_000003.zarr)" in msg, (
        f"the grown pool was not reported as ADDED -- arguments swapped? {msg}"
    )
    assert "0 removed ()" in msg, msg
    assert "(2 shards at start, 3 now)" in msg, (
        f"start/now counts are inverted -- arguments swapped? {msg}"
    )


def test_shard_guard_fires_against_an_EMPTY_reference() -> None:
    """X9: an empty reference is a reference, not a licence to skip the gate.

    ``_run_variant`` signals "this arm defines the reference" with ``None``.
    ``[]`` means arm 1 genuinely scanned nothing, and an arm that then finds
    shards is de-paired from it. A falsy-snapshot early return -- the natural
    shape of an accepted-and-ignored value -- would turn the gate off for the
    whole rest of the sweep, silently.
    """
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(_shards("shard_000001.zarr"), [], name="sharp")
    msg = str(ei.value)
    assert "DE-PAIRED" in msg, msg
    assert "1 added (shard_000001.zarr)" in msg, msg
    assert "(0 shards at start, 1 now)" in msg, msg


def test_the_shard_guard_runs_on_the_real_call_path(monkeypatch, tmp_path) -> None:
    """The guard must fire from ``_run_variant``, off the BUFFER's own list.

    A guard that exists but is not reached is this repo's signature defect, and
    a guard that re-globs the directory itself would compare two independent
    reads rather than "what this arm will draw from" vs "what arm 1 drew from".
    So this drives the whole variant path with a buffer that reports a grown
    pool and requires the abort, before any training happens.
    """
    class _FakeBuf:
        def __init__(self, *_a, **_k) -> None:
            self.closed = False

        def _snapshot_shards(self) -> list[Path]:
            return _shards("shard_000001.zarr", "shard_000002.zarr")

        def __len__(self) -> int:
            return 10_000

        def close(self) -> None:
            self.closed = True

    class _Cfg:
        input_extra_features = "v2_threats"

    def _no_training(*_a, **_k):
        raise AssertionError(
            "train_steps ran despite a changed shard pool: the arms are "
            "de-paired and the guard did not stop the sweep"
        )

    monkeypatch.setattr(rr, "DiskReplayBuffer", _FakeBuf)
    monkeypatch.setattr(rr, "model_config_from_arch", lambda _a: _Cfg())
    monkeypatch.setattr(rr, "build_model", lambda _cfg: _tiny_net())
    monkeypatch.setattr(rr, "load_state_dict_tolerant", lambda *a, **k: None)
    monkeypatch.setattr(rr, "trainer_kwargs_from_config", lambda _c: {})
    monkeypatch.setattr(
        rr, "Trainer", lambda *a, **k: type("_T", (), {"train_steps": _no_training})(),
    )
    monkeypatch.setattr(rr, "_assert_replay_planes_match", lambda *a, **k: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}, "arch": {}})

    with pytest.raises(SystemExit) as ei:
        rr._run_variant(
            name="arm2", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=800, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=_shards("shard_000001.zarr"),
        )
    msg = str(ei.value)
    assert "DE-PAIRED" in msg, msg
    # X1 at the CALL SITE: `_assert_shards_unchanged(this_arm, reference)`. A
    # swapped call still aborts, so only the direction of the report catches it.
    assert "1 added (shard_000002.zarr)" in msg, (
        f"_run_variant passed the reference as `observed` -- arguments swapped? {msg}"
    )
    assert "(1 shards at start, 2 now)" in msg, msg


def test_an_EMPTY_reference_still_gates_the_real_call_path(monkeypatch, tmp_path) -> None:
    """X9 on the production path: ``[]`` must not read as "no reference yet".

    ``_run_variant`` distinguishes "arm 1, adopt what you see" (``None``) from
    "arm 1 scanned an empty pool" (``[]``). Conflating them -- ``if not
    shard_snapshot`` instead of ``is None`` -- silently disables the gate for
    every later arm of any sweep whose first arm found no shards.
    """
    class _OneShardBuf:
        def __init__(self, *_a, **_k) -> None:
            self.closed = False

        def _snapshot_shards(self) -> list[Path]:
            return _shards("shard_000001.zarr")

        def __len__(self) -> int:
            return 10_000

        def close(self) -> None:
            self.closed = True

    _variant_harness(monkeypatch, _OneShardBuf)

    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._run_variant(
            name="arm2", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=800, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=[],
        )


def test_arm1_adopts_its_OWN_scan_and_does_not_check_it_against_a_glob(
    monkeypatch, tmp_path,
) -> None:
    """The fix: arm 1's reference is its own buffer scan, never a pre-glob.

    ``main()`` used to glob the replay directory and hand that list to arm 1 as
    well. Arm 1's own scan happens tens of seconds later -- ``torch.load`` +
    ``build_model`` + ``Trainer()`` + CUDA init all run in between -- so a shard
    landing inside that window aborted a sweep in which every arm would have
    drawn from the same pool. Here the on-disk glob DISAGREES with what the
    buffer scans, and arm 1 must train anyway and report its own list.
    """
    pool = _shards("shard_000001.zarr", "shard_000002.zarr")

    class _Buf:
        def __init__(self, *_a, **_k) -> None:
            self.closed = False

        def _snapshot_shards(self) -> list[Path]:
            return list(pool)

        def __len__(self) -> int:
            return 10_000

        def close(self) -> None:
            self.closed = True

    _variant_harness(monkeypatch, _Buf, train=True)
    # A shard landed between the (now removed) pre-startup glob and arm 1's scan.
    monkeypatch.setattr(rr, "iter_shard_paths", lambda _d: _shards("shard_000001.zarr"))

    summary = rr._run_variant(
        name="base", overrides={}, base_config={"seed": 0},
        checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
        steps=1, batch_size=2, device="cpu", out_dir=tmp_path,
        shard_snapshot=None,
    )
    assert summary["shard_pool"] == [str(p) for p in pool], (
        "arm 1 did not report the pool its own buffer scanned; arms 2..N would "
        f"be paired against the wrong reference: {summary.get('shard_pool')}"
    )


def test_main_makes_arm1s_pool_the_reference_and_never_pre_globs(
    monkeypatch, tmp_path,
) -> None:
    """Every LATER arm must be compared against arm 1's own list.

    Two failure modes at once. Re-globbing per arm would hand each arm a
    snapshot taken microseconds before its own buffer scan -- guard present,
    reached, and unable to fire. Globbing ONCE before arm 1 starts (what #307
    shipped) makes arm 1 fail on shards that landed during startup, which
    de-pairs nothing. So: arm 1 receives ``None``, arms 2 and 3 receive exactly
    arm 1's reported pool, and ``main()`` does not glob at all.
    """
    scans: list[int] = []

    def _growing(_d) -> list[Path]:
        scans.append(1)
        return _shards(*(f"shard_{i:06d}.zarr" for i in range(len(scans))))

    got: list[list[Path] | None] = []
    arm1_pool = _shards("shard_000042.zarr", "shard_000043.zarr")

    def _fake_variant(**kw) -> dict:
        snap = kw["shard_snapshot"]
        got.append(None if snap is None else list(snap))
        return {"variant": kw["name"], "shard_pool": [str(p) for p in arm1_pool]}

    monkeypatch.setattr(rr, "iter_shard_paths", _growing)
    monkeypatch.setattr(rr, "_run_variant", _fake_variant)
    monkeypatch.setattr(rr, "flatten_run_config_defaults", lambda _c: {"batch_size": 8})
    monkeypatch.setattr(rr, "load_yaml_file", lambda _p: {})
    monkeypatch.setattr(sys, "argv", [
        "retarget_retrain.py", "--config", "c.yaml", "--checkpoint", "ck.pt",
        "--replay-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
        "--variant", "base:", "--variant", "sharp:sf_policy_temp=0.006",
        "--variant", "smooth:sf_policy_temp=0.05",
    ])

    rr.main()

    assert len(got) == 3
    assert got[0] is None, (
        "arm 1 was handed a reference it could only fail against: a pre-startup "
        f"glob aborts on shards that landed before any arm drew a row. got={got[0]}"
    )
    assert got[1] == arm1_pool, (
        "arm 2 was not paired against arm 1's OWN scan; the guard is comparing "
        f"against something else: {got[1]}"
    )
    assert got[2] == arm1_pool, (
        f"arm 3 drifted off arm 1's reference: {got[2]}"
    )
    assert scans == [], (
        f"main() globbed the replay dir {len(scans)}x; the reference must come "
        "from arm 1's buffer, and a pre-startup glob re-introduces the false abort"
    )


def test_the_reference_source_matches_what_the_buffer_scans(tmp_path) -> None:
    """Arm 1's list and a directory glob must agree on an unchanged pool.

    The reference is ``DiskReplayBuffer._snapshot_shards()``, and every later
    arm is compared against it with the same call. If the buffer's list ever
    stops agreeing with the directory on an UNCHANGED pool the guard becomes
    either a permanent false abort or -- worse -- pins nothing. Held to a REAL
    buffer over real shards, not to two calls of the same helper.
    """
    shard_dir = tmp_path / "replay"
    writer = DiskReplayBuffer(
        10_000, shard_dir=shard_dir, rng=np.random.default_rng(0),
        read_only=False, shuffle_cap=64, shard_size=8,
    )
    try:
        writer.add_many([_replay_sample() for _ in range(24)])
        writer.flush()
    finally:
        writer.close()
    assert len(rr.iter_shard_paths(shard_dir)) >= 3, "fixture wrote no shards"

    arm1 = DiskReplayBuffer(
        10**9, shard_dir=shard_dir, rng=np.random.default_rng(0), read_only=True,
        shuffle_cap=64, shard_size=8, deterministic_refresh=True,
    )
    try:
        snapshot = arm1._snapshot_shards()  # the reference arm 1 hands onward
        assert snapshot == rr.iter_shard_paths(shard_dir), (
            "the buffer's scan and the directory disagree on an unchanged pool"
        )
        rr._assert_shards_unchanged(arm1._snapshot_shards(), snapshot, name="base")
    finally:
        arm1.close()

    # Now the live-window case, for real: the pool gains a shard while arm 1 is
    # training, and arm 2's buffer scans the grown pool.
    topup = DiskReplayBuffer(
        10_000, shard_dir=shard_dir, rng=np.random.default_rng(1),
        read_only=False, shuffle_cap=64, shard_size=8,
    )
    try:
        topup.add_many([_replay_sample() for _ in range(8)])
        topup.flush()
    finally:
        topup.close()
    assert len(rr.iter_shard_paths(shard_dir)) > len(snapshot), "top-up wrote no shard"

    arm2 = DiskReplayBuffer(
        10**9, shard_dir=shard_dir, rng=np.random.default_rng(0), read_only=True,
        shuffle_cap=64, shard_size=8, deterministic_refresh=True,
    )
    try:
        with pytest.raises(SystemExit, match="DE-PAIRED"):
            rr._assert_shards_unchanged(arm2._snapshot_shards(), snapshot, name="sharp")
    finally:
        arm2.close()


def test_require_complete_raises_on_shape_mismatch() -> None:
    src = _tiny_net()
    dst = _tiny_net()
    state = dict(src.state_dict())
    key = next(k for k, v in state.items() if v.ndim >= 1)
    state[key] = torch.zeros(tuple(d + 1 for d in state[key].shape))

    # Default tolerant mode: the mismatched tensor is silently skipped.
    load_state_dict_tolerant(dst, dict(state), label="test")

    with pytest.raises(RuntimeError, match="require_complete"):
        load_state_dict_tolerant(
            dst, dict(state), label="test", require_complete=True,
        )


def _variant_harness(monkeypatch, buf_cls, *, train: bool = False) -> None:
    """Wire ``_run_variant``'s collaborators so only ``buf_cls`` varies.

    ``train=False`` (the default) makes ``train_steps`` an assertion failure, so
    a guard that lets a de-paired sweep through is reported as training-ran
    rather than as a missing exception. ``train=True`` lets the arm complete,
    for the cases that assert what a PASSING arm returns.
    """
    class _Cfg:
        input_extra_features = "v2_threats"

    def _no_training(*_a, **_k):
        raise AssertionError(
            "train_steps ran despite a changed shard pool: the arms are "
            "de-paired and the guard did not stop the sweep"
        )

    def _train_steps(*_a, **_k):
        return type("_M", (), {"loss": 1.0})()

    trainer_ns = {"train_steps": _train_steps if train else _no_training,
                  "save": lambda *_a, **_k: None}

    monkeypatch.setattr(rr, "DiskReplayBuffer", buf_cls)
    monkeypatch.setattr(rr, "model_config_from_arch", lambda _a: _Cfg())
    monkeypatch.setattr(rr, "build_model", lambda _cfg: _tiny_net())
    monkeypatch.setattr(rr, "load_state_dict_tolerant", lambda *a, **k: None)
    monkeypatch.setattr(rr, "trainer_kwargs_from_config", lambda _c: {})
    monkeypatch.setattr(rr, "Trainer", lambda *a, **k: type("_T", (), trainer_ns)())
    monkeypatch.setattr(rr, "_assert_replay_planes_match", lambda *a, **k: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}, "arch": {}})


def test_the_guard_reads_the_BUFFER_list_not_a_re_glob(monkeypatch, tmp_path) -> None:
    """MUTATION: compare ``iter_shard_paths(replay_dir)`` instead of
    ``buf._snapshot_shards()``.

    That mutation survived an independent review's whole suite, because every
    other test makes the two agree. Here the on-disk glob MATCHES the snapshot
    while the buffer reports a different pool -- so a re-globbing guard sees no
    change and trains, and only a guard reading what THIS ARM WILL ACTUALLY
    DRAW FROM aborts.

    The module comment claims the buffer's list is authoritative because a
    re-glob "would agree even if the buffer had filtered". On today's path the
    buffer cannot filter (`read_only=True` makes `_enforce_window` return at
    once), so that justification describes a case which cannot arise -- but the
    choice is still right, and this is the test that makes it a pinned decision
    rather than an unverifiable preference.
    """
    snap = _shards("shard_000001.zarr")

    class _DisagreeingBuf:
        def __init__(self, *_a, **_k) -> None:
            self.closed = False

        def _snapshot_shards(self) -> list[Path]:
            # What this arm will draw from -- NOT what a directory glob says.
            return _shards("shard_000001.zarr", "shard_000002.zarr")

        def __len__(self) -> int:
            return 10_000

        def close(self) -> None:
            self.closed = True

    _variant_harness(monkeypatch, _DisagreeingBuf)
    # The on-disk view agrees with the snapshot, so a re-glob cannot detect it.
    monkeypatch.setattr(rr, "iter_shard_paths", lambda _d: list(snap))

    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._run_variant(
            name="arm2", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=800, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=list(snap),
        )


def test_a_depaired_EMPTY_pool_reports_de_pairing_not_wrong_replay_dir(
    monkeypatch, tmp_path,
) -> None:
    """MUTATION: move the guard BELOW the ``len(buf) == 0`` check.

    That ordering is argued for in the module comment and in the ledger, and it
    survived an independent review's suite -- because every other fixture's
    fake buffer reports ``__len__ == 10_000``, so the empty branch is never
    reachable and the ordering is never exercised.

    A pool emptied mid-sweep is a PAIRING failure. Diagnosing it as "wrong
    --replay-dir, or plane-count mismatch" sends the operator to re-check a
    path that was right, and loses the fact that the arms diverged.
    """
    class _EmptyDisagreeingBuf:
        def __init__(self, *_a, **_k) -> None:
            self.closed = False

        def _snapshot_shards(self) -> list[Path]:
            return []

        def __len__(self) -> int:
            return 0

        def close(self) -> None:
            self.closed = True

    _variant_harness(monkeypatch, _EmptyDisagreeingBuf)

    with pytest.raises(SystemExit) as ei:
        rr._run_variant(
            name="arm2", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=800, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=_shards("shard_000001.zarr", "shard_000002.zarr"),
        )
    msg = str(ei.value)
    assert "DE-PAIRED" in msg, (
        "an emptied pool was reported as an empty buffer, not as a pairing "
        f"failure -- the guard now runs AFTER the empty-pool check: {msg}"
    )
    assert "wrong --replay-dir" not in msg, msg
