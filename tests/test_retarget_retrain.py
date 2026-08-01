"""Guards for the offline retarget/retrain driver and the strict-load path.

The script's whole value is A/B integrity: the model must be the exact
trained net (not a partial random init) and every arm must differ only in
the knob under test. These tests pin the guardrails that enforce that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.retarget_retrain as rr
from chess_anti_engine.model import load_state_dict_tolerant
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from scripts.retarget_retrain import _parse_variant


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
        )
    assert seen.get("deterministic_refresh") is True, (
        "retarget_retrain builds its replay buffer WITHOUT "
        "deterministic_refresh=True; the arms no longer share a draw sequence "
        f"and every variant delta this script prints is unpaired. kwargs: {sorted(seen)}"
    )
    # The paired claim is also made in prose at the top of the module; a reader
    # who trusts it is trusting the line above.
    assert "deterministic_refresh=True" in (rr.__doc__ or "")


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
