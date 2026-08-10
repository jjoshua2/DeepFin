"""Guards for the offline retarget/retrain driver and the strict-load path.

The script's whole value is A/B integrity: the model must be the exact
trained net (not a partial random init) and every arm must differ only in
the knob under test. These tests pin the guardrails that enforce that.
"""
from __future__ import annotations

import inspect
import json
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
    rr._assert_shards_unchanged(observed=list(snap), snapshot=snap, name="base")  # no raise


def test_shard_guard_fires_when_a_shard_is_ADDED_mid_sweep() -> None:
    # The live case: --replay-dir is the running window (or a pool being topped
    # up), variants run sequentially, each is a full retrain, so arm 2 scans a
    # pool arm 1 never saw. Measured at --steps 800, same seed, flag on:
    # +1 shard moved 617/800 sampled rows (77.1%).
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    grown = snap + _shards("shard_000003.zarr")
    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._assert_shards_unchanged(observed=grown, snapshot=snap, name="sharp")


def test_shard_guard_fires_when_a_shard_is_REMOVED_mid_sweep() -> None:
    # Eviction/quarantine is the other direction: -1 shard moved 343/800 (42.9%).
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    with pytest.raises(SystemExit, match="DE-PAIRED"):
        rr._assert_shards_unchanged(observed=snap[:1], snapshot=snap, name="smooth")


def test_shard_guard_fires_on_reordering_alone() -> None:
    # Same SET, different scan order: the buffer's draw sequence indexes the
    # ordered pool, so order alone de-pairs the arms.
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    with pytest.raises(SystemExit, match="different scan order"):
        rr._assert_shards_unchanged(
            observed=list(reversed(snap)), snapshot=snap, name="base",
        )


def test_shard_guard_names_the_shards_that_moved() -> None:
    snap = _shards("shard_000001.zarr")
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(
            observed=_shards("shard_000009.zarr"), snapshot=snap, name="x",
        )
    msg = str(ei.value)
    assert "shard_000009.zarr" in msg
    assert "shard_000001.zarr" in msg


def test_shard_guard_argument_ORDER_is_pinned() -> None:
    """X1: the two lists are same-typed, so a swap must not be expressible.

    The primary defence is the signature -- both lists are keyword-only, so the
    positional swap that #307 allowed is now a ``TypeError`` rather than a
    silent inversion. This test is the second line: it pins the DIRECTION of the
    report, so renaming the two parameters (the one swap keyword-only cannot
    stop) still fails here rather than quietly reporting a shard as DELETED when
    one was in fact ADDED.
    """
    snap = _shards("shard_000001.zarr", "shard_000002.zarr")
    grown = snap + _shards("shard_000003.zarr")
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(observed=grown, snapshot=snap, name="sharp")
    msg = str(ei.value)
    assert "1 added (shard_000003.zarr)" in msg, (
        f"the grown pool was not reported as ADDED -- arguments swapped? {msg}"
    )
    assert "0 removed ()" in msg, msg
    assert "(2 shards at start, 3 now)" in msg, (
        f"start/now counts are inverted -- arguments swapped? {msg}"
    )


def test_the_two_shard_lists_are_KEYWORD_ONLY() -> None:
    """A positional swap must be unexpressible, not merely detected.

    ``observed`` and ``snapshot`` are both ``list[Path]``, so passed
    positionally a swap type-checks, still aborts, and inverts the report. #307
    shipped them positional and every one of its tests survived the swap. The
    signature is the fix; ``..._argument_ORDER_is_pinned`` is the backstop for a
    parameter RENAME, which keyword-only cannot prevent.
    """
    with pytest.raises(TypeError):
        rr._assert_shards_unchanged(
            _shards("shard_000001.zarr"),  # pyright: ignore[reportCallIssue]
            _shards("shard_000002.zarr"),
            name="x",
        )


def test_run_variant_requires_an_EXPLICIT_shard_snapshot() -> None:
    """``shard_snapshot`` must have NO default.

    A default of ``None`` reads harmless and is the exact shape of this repo's
    signature defect: a future call site that forgets the kwarg silently gets
    arm-1 semantics -- "adopt whatever you scan" -- so the gate is off for that
    arm with nothing on screen to say so. Requiring it forces every caller to
    state whether it is defining the reference or being checked against one.
    """
    param = inspect.signature(rr._run_variant).parameters["shard_snapshot"]
    assert param.default is inspect.Parameter.empty, (
        "shard_snapshot acquired a default; a caller that omits it now silently "
        f"gets arm-1 semantics and its pool is never checked (default={param.default!r})"
    )


def test_main_demands_the_shard_pool_field_LOUDLY(monkeypatch, tmp_path) -> None:
    """A summary without ``shard_pool`` must raise, not yield an empty reference.

    ``summary["shard_pool"]`` is deliberately a subscript. With
    ``.get("shard_pool", [])`` a summary that lost the field would hand arm 2 an
    empty reference, and the sweep would die on a *false* `(0 shards at start, N
    now)` de-pairing report instead of naming the real fault.
    """
    def _no_pool(**kw) -> dict:
        return {"variant": kw["name"]}

    monkeypatch.setattr(rr, "_run_variant", _no_pool)
    monkeypatch.setattr(rr, "flatten_run_config_defaults", lambda _c: {"batch_size": 8})
    monkeypatch.setattr(rr, "load_yaml_file", lambda _p: {})
    monkeypatch.setattr(sys, "argv", [
        "retarget_retrain.py", "--config", "c.yaml", "--checkpoint", "ck.pt",
        "--replay-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
        "--variant", "base:", "--variant", "sharp:sf_policy_temp=0.006",
    ])

    with pytest.raises(KeyError, match="shard_pool"):
        rr.main()


def test_shard_guard_fires_against_an_EMPTY_reference() -> None:
    """X9: an empty reference is a reference, not a licence to skip the gate.

    ``_run_variant`` signals "this arm defines the reference" with ``None``, so
    ``[]`` means arm 1 genuinely scanned nothing and an arm that then finds
    shards is de-paired from it.

    ⚠ On today's path ``main()`` cannot actually hold ``[]``: an arm-1 pool of
    ``[]`` means ``len(buf) == 0``, which ``SystemExit``s before a summary is
    returned, so the sweep dies at arm 1 either way. This pins the helper's
    behaviour as a decision -- no falsy-snapshot early return -- rather than a
    reachable bug, and keeps the ``None``/``[]`` distinction honest if the
    empty-pool guard's ordering ever changes.
    """
    with pytest.raises(SystemExit) as ei:
        rr._assert_shards_unchanged(
            observed=_shards("shard_000001.zarr"), snapshot=[], name="sharp",
        )
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
    got_draws: list[dict | None] = []
    arm1_pool = _shards("shard_000042.zarr", "shard_000043.zarr")
    arm1_draws = {"batches_drawn": 1600.0, "transient_cuda_retry_batches": 0.0}

    def _fake_variant(**kw) -> dict:
        snap = kw["shard_snapshot"]
        got.append(None if snap is None else list(snap))
        d = kw["draw_snapshot"]
        got_draws.append(None if d is None else dict(d))
        return {
            "variant": kw["name"],
            "shard_pool": [str(p) for p in arm1_pool],
            "draws": dict(arm1_draws),
        }

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
    # Same discipline for the DRAW reference (F4): arm 1 defines it, later arms
    # are measured against arm 1's counts, and main() invents nothing.
    assert got_draws[0] is None, (
        f"arm 1 was handed a draw reference it did not produce: {got_draws[0]}"
    )
    assert got_draws[1] == arm1_draws, got_draws
    assert got_draws[2] == arm1_draws, got_draws


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
        # Deliberately NOT re-asserting `_assert_shards_unchanged(snapshot,
        # snapshot)` here: arm 1 is the reference, so comparing it to itself is
        # a tautology that would pass against any implementation. The real
        # assertions are the line above and the arm-2 block below.
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
            rr._assert_shards_unchanged(
                observed=arm2._snapshot_shards(), snapshot=snapshot, name="sharp",
            )
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
  # Carries the draw-provenance pair because `TrainMetrics` does. The stub
  # omitted them and passed only because `_run_variant` read them through
  # `getattr(..., 0.0)`; that default is gone, so a stub that does not model
  # the real object now fails here instead of silently feeding the de-pairing
  # guard two zeros.
        return type("_M", (), {
            "loss": 1.0, "batches_drawn": 1.0, "transient_cuda_retry_batches": 0.0,
        })()

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


# ── F3: an override key that reaches nothing must fail the sweep ────────────
#
# ⚑ THE 2026-07-06 DOSE SCREEN SHIPPED A NO-OP AS A RESULT. Its arms were
# launched with `sf_gap_priority_signed=...`; `TrialConfig.from_dict` drops
# unknown keys in silence and `retarget_report.json` recorded the override as
# applied, so four arms of GPU time measured the control twice. An independent
# review of PR #373 reproduced the same shape with one doubled letter
# (`policy_target_tempp=1.5`), inside the instrument a ledger entry designates
# as deciding. These pin the guard AND its exemptions -- a guard that refused
# real knobs would be worse than none, because operators would delete it.


def _prod_flat() -> dict:
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
    return flatten_run_config_defaults(load_yaml_file(Path("configs/pbt2_small.yaml")))


@pytest.mark.parametrize(
    "key",
    [
        "policy_target_tempp",     # the review's doubled letter
        "policy_target_temp_",     # trailing underscore
        "policy_temp",             # plausible wrong name
        "sf_gap_priority_signed",  # the ACTUAL 2026-07-06 key
        "definitely_not_a_key",
    ],
)
def test_a_dead_variant_key_ABORTS_instead_of_training_the_control_twice(
    key: str,
) -> None:
    with pytest.raises(SystemExit) as ei:
        rr._assert_overrides_reach_the_trainer(
            name="arm", overrides={key: 1.5}, base_config=_prod_flat(),
        )
    msg = str(ei.value)
    assert key in msg, msg
    assert "reach NOTHING" in msg, msg


# ── codex P1: a SWEEP-LEVEL key is not variant-overridable ──────────────────
#
# ⚑ The reachability guard above asks the wrong question for one class of key.
# `batch_size` IS a TrialConfig field, so it passes the probe honestly -- and
# `main()` still resolves it ONCE off `base_config`/argv and hands the same
# value to every arm, so `--variant arm:batch_size=1024` trains identically to
# the control while `retarget_report.json` records the override as applied.
# Reachability is satisfied; SHADOWING is the defect.


@pytest.mark.parametrize(
    "key", ["batch_size", "steps", "device", "gpu_mem_fraction",
            "checkpoint", "replay_dir", "rebuild_sf_targets"],
)
def test_a_SWEEP_LEVEL_key_is_refused_as_a_variant_override(key: str) -> None:
    with pytest.raises(SystemExit) as ei:
        rr._parse_variant(f"arm:{key}=1024")
    assert key in str(ei.value), str(ei.value)


def test_the_shadowed_key_set_is_DERIVED_from_the_signature_not_hand_listed() -> None:
    """A hand-kept list is covered the day someone remembers it. This one is
    covered the day a parameter is added -- so pin the derivation, not the
    membership: every name in the set must really be a `_run_variant` parameter,
    and the real collisions with production config keys must be in it."""
    import inspect
    shadowed = rr._sweep_level_keys()
    params = set(inspect.signature(rr._run_variant).parameters)
    assert shadowed <= params, sorted(shadowed - params)
    flat = _prod_flat()
    assert {"batch_size", "steps", "device"} <= shadowed
    # only keys that COLLIDE with a real config key can bite; the rest are inert
    assert shadowed & set(flat), "the guard covers no real config key at all"


def test_the_reachability_probe_would_have_PASSED_the_shadowed_key() -> None:
    """The reason this needed its own guard rather than a wider reachability
    probe. If this ever starts failing, the two guards have converged and one
    of them is redundant -- decide which, do not leave both."""
    assert rr._override_key_reaches_the_trainer("batch_size", _prod_flat())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("policy_target_temp", 1.5),           # read off the flat dict by the Trainer
        ("sf_policy_temp", 0.006),             # documented target knob
        ("replay_sf_gap_priority_weight", 30), # documented sampling knob
        ("lr", 1e-4),                          # plain config key
        ("seed", 1),                           # the null-arm knob the ledger needs
    ],
)
def test_the_guard_does_NOT_refuse_a_key_that_really_reaches_an_arm(
    key: str, value: float,
) -> None:
    """The exemption half. `policy_target_temp` is the interesting one: it is
    NOT a TrialConfig field and is absent from the shipped yaml, so only the
    behavioural probe (does it move `trainer_kwargs_from_config`?) admits it."""
    rr._assert_overrides_reach_the_trainer(
        name="arm", overrides={key: value}, base_config=_prod_flat(),
    )


def test_policy_target_temp_is_admitted_by_the_BEHAVIOURAL_probe_only() -> None:
    """Pin WHY it is admitted, so a future refactor that drops the probe and
    keeps only the allowlist halves turns this red instead of silently
    refusing the very knob PR #373 exists for."""
    from chess_anti_engine.tune.trial_config import TrialConfig
    flat = _prod_flat()
    assert "policy_target_temp" not in flat
    assert not hasattr(TrialConfig.from_dict({}), "policy_target_temp")
    assert rr._override_key_reaches_the_trainer("policy_target_temp", flat)


def test_the_dead_key_guard_runs_BEFORE_the_checkpoint_is_loaded(
    monkeypatch, tmp_path,
) -> None:
    """A guard that fires after `torch.load` + CUDA init costs the operator the
    startup every time; worse, on the real path it would sit behind work that
    can fail for unrelated reasons. Drive `_run_variant` and require the abort
    with `torch.load` never called."""
    def _no_load(*_a, **_k):
        raise AssertionError("torch.load ran before the dead-key guard aborted")

    monkeypatch.setattr(torch, "load", _no_load)
    with pytest.raises(SystemExit) as ei:
        rr._run_variant(
            name="arm", overrides={"policy_target_tempp": 1.5},
            base_config=_prod_flat(),
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=1, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=None,
        )
    assert "reach NOTHING" in str(ei.value)


# ── F4: a transient-CUDA retry de-pairs the arms, silently ──────────────────
#
# `_assert_shards_unchanged` pins the POOL. `Trainer.train_steps` catches a
# transient CUDA error and calls `add_retry_batches(...)`, which pulls
# replacement batches and advances the buffer's private RNG -- so one retry in
# ONE arm desynchronises its rows from every other arm's permanently, with the
# pool untouched. It was reported only as a `logging.warning`.


def test_draw_guard_passes_when_both_arms_drew_the_same_batches() -> None:
    same = {"batches_drawn": 1600.0, "transient_cuda_retry_batches": 0.0}
    rr._assert_draws_unchanged(observed=dict(same), snapshot=dict(same), name="arm2")


def test_draw_guard_fires_when_a_LATER_arm_retried() -> None:
    with pytest.raises(SystemExit) as ei:
        rr._assert_draws_unchanged(
            observed={"batches_drawn": 1603.0, "transient_cuda_retry_batches": 3.0},
            snapshot={"batches_drawn": 1600.0, "transient_cuda_retry_batches": 0.0},
            name="arm2",
        )
    msg = str(ei.value)
    assert "de-paired" in msg
    assert "1600" in msg, msg
    assert "1603" in msg, msg


def test_draw_guard_fires_when_ARM_1_ITSELF_retried() -> None:
    """⚑ The case a plain equality check misses. Arm 1 defines the reference, so
    its counts always equal themselves -- but if arm 1 retried, the REFERENCE
    sequence is already the perturbed one and no later arm can disagree with
    it. A non-zero retry count is fatal on its own."""
    retried = {"batches_drawn": 1603.0, "transient_cuda_retry_batches": 3.0}
    with pytest.raises(SystemExit) as ei:
        rr._assert_draws_unchanged(
            observed=dict(retried), snapshot=dict(retried), name="ctrl",
        )
    assert "de-paired" in str(ei.value)


def test_draw_guard_fires_on_a_count_mismatch_with_NO_retry_recorded() -> None:
    """Belt and braces: any future path that changes the draw count without
    going through `add_retry_batches` must still abort."""
    with pytest.raises(SystemExit):
        rr._assert_draws_unchanged(
            observed={"batches_drawn": 1599.0, "transient_cuda_retry_batches": 0.0},
            snapshot={"batches_drawn": 1600.0, "transient_cuda_retry_batches": 0.0},
            name="arm2",
        )


def test_the_draw_guard_runs_on_the_real_call_path_and_lands_in_the_report(
    monkeypatch, tmp_path,
) -> None:
    """Both halves at once: `_run_variant` must ABORT on a retried arm, and a
    clean arm must record its draw counts in the summary that becomes
    `retarget_report.json` -- a number nobody can read is not a guard."""
    class _FakeBuf:
        def __init__(self, *_a, **_k) -> None: ...
        def _snapshot_shards(self) -> list[Path]:
            return _shards("shard_000001.zarr")
        def __len__(self) -> int:
            return 10_000
        def close(self) -> None: ...

    class _Cfg:
        input_extra_features = "v2_threats"

    retries = {"n": 0.0}

    class _Metrics:
        loss = 1.0
        batches_drawn = 1600.0
        @property
        def transient_cuda_retry_batches(self) -> float:
            return retries["n"]

    class _T:
        def train_steps(self, *_a, **_k):
            return _Metrics()
        def save(self, path) -> None:
            Path(path).write_text("x")

    monkeypatch.setattr(rr, "DiskReplayBuffer", _FakeBuf)
    monkeypatch.setattr(rr, "model_config_from_arch", lambda _a: _Cfg())
    monkeypatch.setattr(rr, "build_model", lambda _cfg: _tiny_net())
    monkeypatch.setattr(rr, "load_state_dict_tolerant", lambda *a, **k: None)
    monkeypatch.setattr(rr, "trainer_kwargs_from_config", lambda _c: {})
    monkeypatch.setattr(rr, "Trainer", lambda *a, **k: _T())
    monkeypatch.setattr(rr, "_assert_replay_planes_match", lambda *a, **k: None)
    monkeypatch.setattr(rr, "TrialConfig", type("_TC", (), {"from_dict": staticmethod(lambda _c: _FakeTC())}))
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}, "arch": {}})

    def _run():
        return rr._run_variant(
            name="ctrl", overrides={}, base_config={"seed": 0},
            checkpoint=tmp_path / "trainer.pt", replay_dir=tmp_path,
            steps=1, batch_size=2, device="cpu", out_dir=tmp_path,
            shard_snapshot=None,
        )

    summary = _run()
    assert summary["draws"] == {
        "batches_drawn": 1600.0, "transient_cuda_retry_batches": 0.0,
    }, "the draw counts never reached retarget_report.json"

    retries["n"] = 4.0
    with pytest.raises(SystemExit) as ei:
        _run()
    assert "de-paired" in str(ei.value)


class _FakeTC:
    """Minimal stand-in for TrialConfig on the mocked `_run_variant` path."""
    replay_upgrade_v1_planes = False
    shuffle_buffer_size = 1
    shard_size = 1
    shuffle_refresh_interval = 1
    shuffle_refresh_shards = 1
    replay_shard_recency_exponent = 1.0
    shuffle_draw_cap_frac = 1.0
    shuffle_wl_max_ratio = 1.0
    replay_sf_gap_priority_weight = 0.0
    replay_sf_gap_priority_signed = False
    replay_fast_low_surprise_priority = 0.0
    diff_focus_pol_scale = 0.0
    diff_focus_q_weight = 0.0


# ── codex P3: the abort path must BANK what the completed arms drew ─────────
#
# ⚑ The de-pairing guards raise inside `_run_variant`, and the report was
# written only after every arm returned. So the one path whose whole purpose is
# provenance ("the sweep is void, here is what the arms drew") wrote nothing --
# and in a REUSED --out-dir the previous run's report survived the abort and
# read as a clean sweep of the run that had just failed. `bank the dump`.


def _drive_main(monkeypatch, tmp_path, variants: list[str], run_variant) -> None:
    monkeypatch.setattr(rr, "_run_variant", run_variant)
    monkeypatch.setattr(
        sys, "argv",
        ["retarget_retrain.py", "--config", "configs/pbt2_small.yaml",
         "--checkpoint", str(tmp_path / "ckpt.pt"),
         "--replay-dir", str(tmp_path / "shards"),
         "--out-dir", str(tmp_path / "out"), "--steps", "1",
         *[a for v in variants for a in ("--variant", v)]],
    )
    rr.main()


def _summary(name: str) -> dict:
    return {"variant": name, "shard_pool": ["a.zarr"],
            "draws": {"batches_drawn": 8.0, "transient_cuda_retry_batches": 0.0}}


def test_an_ABORTED_sweep_still_writes_the_completed_arms_provenance(
    monkeypatch, tmp_path,
) -> None:
    calls: list[str] = []

    def _run(*, name: str, **_k: object) -> dict:
        calls.append(name)
        if name == "arm":
            raise SystemExit("replay DRAW SEQUENCE de-paired at variant 'arm'")
        return _summary(name)

    with pytest.raises(SystemExit):
        _drive_main(monkeypatch, tmp_path, ["ctrl:", "arm:lr=1e-4"], _run)

    report = tmp_path / "out" / "retarget_report.json"
    assert report.exists(), "the abort path wrote no report at all"
    payload = json.loads(report.read_text())
    assert "de-paired" in payload["aborted"], payload["aborted"]
    assert [s["variant"] for s in payload["completed_variants"]] == ["ctrl"]
    assert payload["completed_variants"][0]["draws"]["batches_drawn"] == 8.0
    assert calls == ["ctrl", "arm"]


def test_a_STALE_report_is_gone_BEFORE_the_first_arm_runs(
    monkeypatch, tmp_path,
) -> None:
    """⚑ MUTATION NOTE, and the reason this asserts mid-sweep rather than after.

    Deleting `report.unlink(...)` and checking the file AFTERWARDS is an
    EQUIVALENT MUTANT: the `finally:` rewrites the report on every exit path, so
    a post-hoc read cannot tell the unlink apart from the overwrite. The unlink
    earns its place only for the exits `finally` does not get -- SIGKILL, the
    OOM killer, a box reboot part-way through a 21-hour sweep -- after which a
    reused `--out-dir` still holds the PREVIOUS run's clean two-arm report and
    an operator reads it as this run's.

    That window is observable without killing a process: while the sweep is
    running, the report on disk must never be a previous run's. Assert it from
    inside the first arm.
    """
    out = tmp_path / "out"
    out.mkdir(parents=True)
    stale = out / "retarget_report.json"
    stale.write_text(json.dumps([_summary("ctrl"), _summary("arm")]))
    seen: list[bool] = []

    def _run(**_k: object) -> dict:
        seen.append(stale.exists())
        raise SystemExit("boom on the first arm")

    with pytest.raises(SystemExit):
        _drive_main(monkeypatch, tmp_path, ["ctrl:"], _run)

    assert seen == [False], (
        "a previous run's retarget_report.json was still on disk while this "
        "sweep was running; a hard kill here leaves it there to be misread"
    )
    payload = json.loads(stale.read_text())
    assert isinstance(payload, dict), "the stale two-arm list survived the abort"
    assert payload["completed_variants"] == []
    assert "boom" in payload["aborted"]


def test_a_CLEAN_sweep_still_writes_a_plain_list(monkeypatch, tmp_path) -> None:
    """The null half: banking on abort must not change the shape every existing
    reader (and the ledger's `jq '.[].overrides'` step) parses."""
    _drive_main(monkeypatch, tmp_path, ["ctrl:", "arm:lr=1e-4"],
                lambda *, name, **_k: _summary(name))
    payload = json.loads((tmp_path / "out" / "retarget_report.json").read_text())
    assert isinstance(payload, list)
    assert [s["variant"] for s in payload] == ["ctrl", "arm"]
