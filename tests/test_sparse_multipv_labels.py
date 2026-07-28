"""Sparse MultiPV label storage + train-time target rebuild.

Parity contract: with params equal to the capture-time config, targets
rebuilt from sf_multipv_raw/sf_label_meta match the stored ones to 1e-5.
Default behavior (rebuild_sf_targets=False) is bitwise-unchanged.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    SF_CP_SENTINEL,
    SF_MULTIPV_RAW_MAX,
    arrays_to_samples,
    samples_to_arrays,
    validate_array_declarations,
)
from chess_anti_engine.train.target_builder import (
    SfTargetParams,
    mask_cross_ply_sf_targets,
    rebuild_sf_policy_target,
    rebuild_sf_policy_targets_batch,
    rebuild_sf_targets_in_arrays,
    rebuild_sf_wdl,
    rebuild_sf_wdl_batch,
)

from tests.stockfish_binary import find_stockfish

SF_PATH = find_stockfish()


def test_shard_roundtrip_preserves_exact_ints():
    pol = np.zeros(4672, np.float32)
    pol[0] = 1.0
    raw = np.full((SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, 1] = SF_CP_SENTINEL
    raw[:, 2] = 0
    raw[0] = (1857, 31999, 0, 870, 100)   # large cp must survive exactly
    raw[1] = (12, SF_CP_SENTINEL, -5, -1, -1)
    meta = np.array([50000, 24, -31999, 0, 870, 100], np.int32)
    samples = [
        ReplaySample(x=np.zeros((146, 8, 8), np.float32), policy_target=pol,
                     wdl_target=0, sf_multipv_raw=raw, sf_label_meta=meta),
        ReplaySample(x=np.zeros((146, 8, 8), np.float32), policy_target=pol,
                     wdl_target=1),   # old-style sample: fields absent
    ]
    arrs = samples_to_arrays(samples)
    validate_array_declarations(arrs)
    back = arrays_to_samples(arrs)
    assert back[0].sf_multipv_raw is not None
    np.testing.assert_array_equal(back[0].sf_multipv_raw, raw)
    assert back[0].sf_label_meta is not None
    np.testing.assert_array_equal(back[0].sf_label_meta, meta)
    assert back[1].sf_multipv_raw is None
    assert back[1].sf_label_meta is None


def _synthetic_rows() -> np.ndarray:
    rows = np.full((SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    rows[:, 1] = SF_CP_SENTINEL
    rows[:, 2] = 0
    rows[0] = (100, 50, 0, 600, 300)
    rows[1] = (200, -20, 0, 450, 320)
    rows[2] = (300, SF_CP_SENTINEL, 4, 990, 10)   # mate-in-4 line
    return rows


@pytest.mark.parametrize("use_logistic", [False, True])
def test_rebuild_matches_live_construction_synthetic(use_logistic):
    """Synthetic rows through both score paths must equal the live builder."""
    from chess_anti_engine.selfplay.stockfish_turn import (
        _build_sf_policy_target,
        _pv_wdl_score,
    )
    from chess_anti_engine.stockfish.uci import StockfishPV

    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=use_logistic,
        sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
    )
    rows = _synthetic_rows()
    legal = np.array([100, 200, 300, 400], dtype=np.int64)

    # live path: build pvs and run the production functions
    cand_idxs, cand_scores = [], []
    for move_idx, cp, mate, w, d in rows[rows[:, 0] >= 0].tolist():
        pv = StockfishPV(
            move_uci="0000",
            wdl=None if w < 0 else np.array([w, d, 1000 - w - d], np.float32) / 1000.0,
            cp=None if cp == SF_CP_SENTINEL else int(cp),
            mate=None if mate == 0 else int(mate),
        )
        score = _pv_wdl_score(
            pv, sf_wdl_use_cp_logistic=use_logistic,
            sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
        )
        assert score is not None
        cand_idxs.append(int(move_idx))
        cand_scores.append(score)
    live = _build_sf_policy_target(
        cand_idxs, cand_scores, legal_indices=legal,
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
    )

    rebuilt = rebuild_sf_policy_target(
        rows, legal_indices=legal, policy_size=4672, params=params,
    )
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt, live, atol=1e-5)


@pytest.mark.parametrize("use_logistic", [False, True])
def test_rebuild_parity_compact_lc0_1858(use_logistic):
    """Production policy encoding: rebuild in compact shard space must equal
    the live full-space target converted exactly like finalize converts it
    (policy_vector_to_encoding for the dense target, policy_index_for_encoding
    for sparse rows + legal mask)."""
    from chess_anti_engine.moves.encode import (
        policy_index_for_encoding,
        policy_size_for_encoding,
        policy_vector_to_encoding,
    )
    from chess_anti_engine.selfplay.stockfish_turn import (
        _build_sf_policy_target,
        _pv_wdl_score,
    )
    from chess_anti_engine.stockfish.uci import StockfishPV

    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY

    enc = "lc0_1858"
    compact_size = policy_size_for_encoding(enc)
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=use_logistic,
        sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
    )
    # Full-space indices that exist in the compact mapping (arbitrary picks).
    legal_full = np.asarray(COMPACT_TO_FULL_POLICY, dtype=np.int64)[[100, 200, 300, 400]]
    rows_full = np.full((SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    rows_full[:, 1] = SF_CP_SENTINEL
    rows_full[:, 2] = 0
    rows_full[0] = (legal_full[0], 50, 0, 600, 300)
    rows_full[1] = (legal_full[1], -20, 0, 450, 320)
    rows_full[2] = (legal_full[2], SF_CP_SENTINEL, 4, 990, 10)   # mate-in-4 line

    cand_idxs, cand_scores = [], []
    for move_idx, cp, mate, w, d in rows_full[rows_full[:, 0] >= 0].tolist():
        pv = StockfishPV(
            move_uci="0000",
            wdl=None if w < 0 else np.array([w, d, 1000 - w - d], np.float32) / 1000.0,
            cp=None if cp == SF_CP_SENTINEL else int(cp),
            mate=None if mate == 0 else int(mate),
        )
        score = _pv_wdl_score(
            pv, sf_wdl_use_cp_logistic=use_logistic,
            sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
        )
        assert score is not None
        cand_idxs.append(int(move_idx))
        cand_scores.append(score)
    live_full = _build_sf_policy_target(
        cand_idxs, cand_scores, legal_indices=legal_full,
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
    )
    live_compact = policy_vector_to_encoding(live_full, policy_encoding=enc)

    rows_compact = rows_full.copy()
    for j in range(rows_compact.shape[0]):
        if rows_compact[j, 0] >= 0:
            rows_compact[j, 0] = policy_index_for_encoding(
                int(rows_compact[j, 0]), policy_encoding=enc,
            )
    legal_compact = np.array(
        [policy_index_for_encoding(int(i), policy_encoding=enc) for i in legal_full],
        dtype=np.int64,
    )
    assert (legal_compact >= 0).all()

    rebuilt = rebuild_sf_policy_target(
        rows_compact, legal_indices=legal_compact,
        policy_size=int(compact_size), params=params,
    )
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt, live_compact, atol=1e-5)


def test_rebuild_sf_wdl_matches_live():
    from chess_anti_engine.selfplay.stockfish_turn import _sf_result_wdl_for_record
    from chess_anti_engine.stockfish.uci import StockfishResult

    for use_logistic in (False, True):
        params = SfTargetParams(sf_wdl_use_cp_logistic=use_logistic)
        res = StockfishResult(
            bestmove_uci="e2e4",
            wdl=np.array([700, 200, 100], np.float32) / 1000.0,
            pvs=[], cp=35, mate=None,
        )
        live = _sf_result_wdl_for_record(
            res, sf_wdl_use_cp_logistic=use_logistic,
            sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
        )
        meta = np.array([1000, 10, 35, 0, 700, 200], np.int32)
        rebuilt = rebuild_sf_wdl(meta, params)
        assert live is not None
        assert rebuilt is not None
        np.testing.assert_allclose(rebuilt, live, atol=1e-5)


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
def test_parity_through_real_stockfish_turn_path():
    """Generate samples through the real selfplay+SF path (tiny nodes) and
    assert stored targets == rebuilt targets with the capture params."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.stockfish import StockfishUCI

    torch.manual_seed(0)
    model = build_model(ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
    )).eval()
    game = GameConfig(
        max_plies=24, sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
        categorical_bins=32, hlgauss_sigma=0.04,
        selfplay_fraction=0.0,   # curriculum games => SF labels on every sample
    )
    assert SF_PATH is not None
    sf = StockfishUCI(SF_PATH, nodes=100, multipv=4)
    try:
        samples, _stats = play_batch(
            model, device="cpu", rng=np.random.default_rng(0), stockfish=sf, games=2,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=4, mcts_type="gumbel"),
            diff_focus=DiffFocusConfig(min_keep=1.0),
            game=game,
        )
    finally:
        sf.close()

    params = SfTargetParams(
        sf_policy_temp=game.sf_policy_temp,
        sf_policy_label_smooth=game.sf_policy_label_smooth,
        sf_wdl_use_cp_logistic=game.sf_wdl_use_cp_logistic,
        sf_wdl_cp_slope=game.sf_wdl_cp_slope,
        sf_wdl_cp_draw_width=game.sf_wdl_cp_draw_width,
    )
    checked_policy = 0
    checked_wdl = 0
    for s in samples:
        if s.sf_multipv_raw is not None and s.sf_policy_target is not None:
            assert s.sf_legal_mask is not None
            rebuilt = rebuild_sf_policy_target(
                s.sf_multipv_raw,
                legal_indices=np.flatnonzero(s.sf_legal_mask),
                policy_size=int(s.sf_policy_target.shape[0]),
                params=params,
            )
            assert rebuilt is not None
            np.testing.assert_allclose(rebuilt, s.sf_policy_target, atol=1e-5)
            checked_policy += 1
        if s.sf_label_meta is not None and s.sf_wdl is not None:
            rebuilt_wdl = rebuild_sf_wdl(s.sf_label_meta, params)
            assert rebuilt_wdl is not None
            np.testing.assert_allclose(rebuilt_wdl, s.sf_wdl, atol=1e-5)
            checked_wdl += 1
    assert checked_policy >= 3, f"too few SF-labeled samples ({checked_policy})"
    assert checked_wdl >= 3


def test_arrays_rebuild_only_touches_sparse_rows():
    pol = np.zeros((2, 4672), np.float16)
    # Row 0 is fully covered (both legal moves are candidates), so the rebuild
    # applies NO label smoothing -> a sharp-temp ~one-hot at the better move
    # (100). Seed the init at the WORSE move (200) so "rebuilt != init" stays a
    # meaningful check now that smoothing no longer perturbs a fully-covered row.
    pol[0, 200] = 1.0
    pol[1, 7] = 1.0
    legal = np.zeros((2, 4672), np.uint8)
    legal[0, [100, 200]] = 1
    raw = np.full((2, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (100, 30, 0, 600, 300)
    raw[0, 1] = (200, -10, 0, 500, 300)
    arrs = {
        "sf_policy_target": pol,
        "sf_legal_mask": legal,
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": np.array([1, 0], np.uint8),
        "sf_wdl": np.zeros((2, 3), np.float16),
        "sf_label_meta": np.zeros((2, 6), np.int32),
        "has_sf_label_meta": np.array([0, 0], np.uint8),
    }
    before_row1 = arrs["sf_policy_target"][1].copy()
    out = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams(sf_policy_temp=0.012))
    assert float(out["sf_policy_target"][0].astype(np.float32).sum()) == pytest.approx(1.0, abs=1e-3)
    assert not np.array_equal(out["sf_policy_target"][0], pol[0])  # rebuilt
    np.testing.assert_array_equal(out["sf_policy_target"][1], before_row1)  # untouched


def test_trainer_default_is_bitwise_unchanged(monkeypatch):
    """rebuild_sf_targets=False must never call the rebuilder, and the
    sampled batch must be byte-identical to the pre-flag pipeline."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import trainer as trainer_mod

    calls = {"n": 0}

    def _spy(arrs, *, params):
        del params
        calls["n"] += 1
        return arrs

    monkeypatch.setattr(trainer_mod, "rebuild_sf_targets_in_arrays", _spy)

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    t = trainer_mod.Trainer(model, device="cpu", lr=1e-3)
    assert t.rebuild_sf_targets is False

    class _Buf:
        rng = np.random.default_rng(0)

        def sample_batch_arrays(self, batch_size, *, wdl_balance=True):
            del wdl_balance
            pol = np.zeros((batch_size, 4672), np.float16)
            pol[:, 0] = 1.0
            return {
                "x": np.zeros((batch_size, 146, 8, 8), np.float16),
                "policy_target": pol,
                "wdl_target": np.zeros((batch_size,), np.int8),
                "priority": np.ones((batch_size,), np.float32),
                "has_policy": np.ones((batch_size,), np.uint8),
            }

    buf = cast(Any, _Buf())
    out_default = t._sample_batch_host(buf, batch_size=2, mirror_prob=0.0)
    assert calls["n"] == 0   # never invoked at the default

    t.rebuild_sf_targets = True
    t._sample_batch_host(buf, batch_size=2, mirror_prob=0.0)
    assert calls["n"] == 1   # invoked exactly when enabled

    t.rebuild_sf_targets = False
    out_again = t._sample_batch_host(buf, batch_size=2, mirror_prob=0.0)
    assert isinstance(out_default, dict)
    assert isinstance(out_again, dict)
    for k in out_default:
        np.testing.assert_array_equal(out_default[k], out_again[k])


# ---------------------------------------------------------------------------
# Sparse CE over gathered log-probs (train/sparse_sf_ce.py)
# ---------------------------------------------------------------------------


def _sparse_ce_batch(
    *, use_logistic: bool, smooth: float,
) -> tuple[torch.Tensor, dict, SfTargetParams, np.ndarray]:
    """Random logits + a 3-row batch: scored rows / fallback row / no labels.

    Shard space is full 4672 (the only width capture writes for az_4672
    shards; the compact-logits projection is exercised separately). Returns
    (masked_logits, batch, params, dense_targets) where dense_targets holds
    the live-equivalent dense vectors for rows 0..1. Legal indices are picked
    from COMPACT_TO_FULL_POLICY so the same batch works against compact
    logits.
    """
    from chess_anti_engine.selfplay.stockfish_turn import _build_sf_policy_target
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY

    shard_width = 4672
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=smooth,
        sf_wdl_use_cp_logistic=use_logistic,
        sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
    )
    legal_full = np.asarray(COMPACT_TO_FULL_POLICY, dtype=np.int64)[[100, 200, 300, 400]]

    raw = np.full((3, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (legal_full[0], 50, 0, 600, 300)
    raw[0, 1] = (legal_full[1], -20, 0, 450, 320)
    raw[0, 2] = (legal_full[2], SF_CP_SENTINEL, 4, 990, 10)
    # row 1: candidates present but none scoreable -> live one-hot fallback
    raw[1, 0] = (legal_full[1], SF_CP_SENTINEL, 0, -1, -1)
    # row 2: no sparse labels at all

    legal = np.zeros((3, shard_width), np.float32)
    legal[:, legal_full] = 1.0

    # Dense reference targets, built exactly like the live path.
    dense = np.zeros((3, shard_width), np.float32)
    rebuilt = rebuild_sf_policy_target(
        raw[0], legal_indices=legal_full, policy_size=shard_width, params=params,
    )
    assert rebuilt is not None
    dense[0] = rebuilt
    dense[1] = _build_sf_policy_target(
        [int(legal_full[1])], [0.0], legal_indices=legal_full,
        sf_policy_temp=params.sf_policy_temp,
        sf_policy_label_smooth=params.sf_policy_label_smooth,
    )
    dense[2, legal_full[0]] = 1.0  # arbitrary stored target for the no-label row

    torch.manual_seed(0)
    logits = torch.randn(3, shard_width)
    batch = {
        "sf_multipv_raw": torch.from_numpy(raw.astype(np.int32)),
        "has_sf_multipv_raw": torch.tensor([1.0, 1.0, 0.0]),
        "sf_legal_mask": torch.from_numpy(legal),
        "has_sf_legal_mask": torch.ones(3),
        "sf_move_index": torch.tensor([int(legal_full[0]), int(legal_full[1]), -1]),
        "has_sf_move": torch.tensor([1.0, 1.0, 0.0]),
    }
    return logits, batch, params, dense


@pytest.mark.parametrize("use_logistic", [False, True])
@pytest.mark.parametrize("smooth", [0.0, 0.01])
def test_sparse_ce_matches_dense_soft_ce(use_logistic, smooth):
    """Same-width case: sparse CE must equal soft CE against the dense target."""
    from chess_anti_engine.train.losses import soft_cross_entropy
    from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

    logits, batch, params, dense = _sparse_ce_batch(
        use_logistic=use_logistic, smooth=smooth,
    )
    dense_ce = soft_cross_entropy(logits, torch.from_numpy(dense))
    sparse_ce, ok = sparse_sf_policy_ce(
        logits, batch, params=params, legal_aligned=batch["sf_legal_mask"],
    )
    assert ok.tolist() == [1.0, 1.0, 0.0]
    torch.testing.assert_close(sparse_ce[:2], dense_ce[:2], atol=1e-5, rtol=1e-5)
    assert float(sparse_ce[2]) == 0.0


def test_sparse_ce_compact_logits_over_full_shard():
    """Production projection: full-4672 shard rows against compact-1858 logits
    must equal the dense path's align_policy_target projection."""
    from chess_anti_engine.moves import COMPACT_POLICY_SIZE, FULL_TO_COMPACT_POLICY
    from chess_anti_engine.train.losses import (
        align_policy_mask,
        align_policy_target,
        soft_cross_entropy,
    )
    from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

    _, batch, params, dense = _sparse_ce_batch(use_logistic=True, smooth=0.01)
    torch.manual_seed(1)
    logits = torch.randn(3, COMPACT_POLICY_SIZE)
    dense_aligned = align_policy_target(torch.from_numpy(dense), COMPACT_POLICY_SIZE)
    dense_ce = soft_cross_entropy(logits, dense_aligned)
    sparse_ce, ok = sparse_sf_policy_ce(
        logits, batch, params=params,
        legal_aligned=align_policy_mask(batch["sf_legal_mask"], COMPACT_POLICY_SIZE),
    )
    assert ok.tolist() == [1.0, 1.0, 0.0]
    torch.testing.assert_close(sparse_ce[:2], dense_ce[:2], atol=1e-5, rtol=1e-5)
    assert int(FULL_TO_COMPACT_POLICY[int(batch["sf_move_index"][0])]) >= 0


def test_compute_loss_sparse_flag_only_touches_sf_move_ce():
    from chess_anti_engine.train.losses import compute_loss

    logits, sparse_batch, params, dense = _sparse_ce_batch(
        use_logistic=False, smooth=0.01,
    )
    n = 3
    torch.manual_seed(2)
    outputs = {
        "policy_own": torch.randn(n, 4672),
        "policy_sf": logits,
        "wdl": torch.randn(n, 3),
    }
    pol = torch.zeros(n, 4672)
    pol[:, 5] = 1.0
    batch = {
        "x": torch.zeros(n, 1),
        "policy_t": pol,
        "wdl_t": torch.tensor([0, 1, 2]),
        "sf_policy_t": torch.from_numpy(dense),
        "has_sf_policy": torch.tensor([1.0, 1.0, 1.0]),
        **sparse_batch,
    }
    base = compute_loss(outputs, batch)
    flagged = compute_loss(outputs, batch, sf_sparse_params=params)
    # Sparse CE equals the dense CE on these rows, so the aggregate must agree
    # to fp tolerance; everything else is bitwise identical.
    torch.testing.assert_close(flagged["sf_move_ce"], base["sf_move_ce"], atol=1e-5, rtol=1e-5)
    for k in base:
        if k in ("sf_move_ce", "total"):
            continue
        assert torch.equal(base[k], flagged[k]), f"{k} changed under sparse CE"


def test_mirror_batch_arrays_mirrors_sparse_rows():
    from chess_anti_engine.moves.encode import MIRROR_POLICY_MAP
    from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays

    raw = np.full((2, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (100, 30, 0, 600, 300)
    raw[1, 0] = (200, -10, 2, 500, 300)
    arrs = {
        "x": np.zeros((2, 146, 8, 8), np.float16),
        "policy_target": np.zeros((2, 4672), np.float16),
        "sf_multipv_raw": raw.copy(),
    }
    rng = np.random.default_rng(0)
    out = maybe_mirror_batch_arrays(arrs, rng=rng, prob=1.0)
    got = np.asarray(out["sf_multipv_raw"])
    assert int(got[0, 0, 0]) == int(MIRROR_POLICY_MAP[100])
    assert int(got[1, 0, 0]) == int(MIRROR_POLICY_MAP[200])
    # non-index columns and pad rows untouched
    np.testing.assert_array_equal(got[0, 0, 1:], raw[0, 0, 1:])
    np.testing.assert_array_equal(got[:, 1:], raw[:, 1:])


def test_mirror_sample_carries_sparse_fields():
    from chess_anti_engine.moves.encode import MIRROR_POLICY_MAP
    from chess_anti_engine.replay.augment import mirror_sample

    raw = np.full((SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, 1] = SF_CP_SENTINEL
    raw[0] = (100, 30, 0, 600, 300)
    meta = np.array([1000, 10, 35, 0, 700, 200], np.int32)
    pol = np.zeros(4672, np.float32)
    pol[0] = 1.0
    s = ReplaySample(
        x=np.zeros((146, 8, 8), np.float32), policy_target=pol, wdl_target=0,
        sf_multipv_raw=raw, sf_label_meta=meta,
    )
    m = mirror_sample(s)
    assert m.sf_multipv_raw is not None
    assert int(m.sf_multipv_raw[0, 0]) == int(MIRROR_POLICY_MAP[100])
    assert m.sf_label_meta is not None
    np.testing.assert_array_equal(m.sf_label_meta, meta)


def test_trainer_drops_sparse_fields_unless_enabled():
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import trainer as trainer_mod

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    t = trainer_mod.Trainer(model, device="cpu", lr=1e-3)
    assert t.sf_policy_sparse_ce is False

    class _Buf:
        rng = np.random.default_rng(0)

        def sample_batch_arrays(self, batch_size, *, wdl_balance=True):
            del wdl_balance
            pol = np.zeros((batch_size, 4672), np.float16)
            pol[:, 0] = 1.0
            return {
                "x": np.zeros((batch_size, 146, 8, 8), np.float16),
                "policy_target": pol,
                "wdl_target": np.zeros((batch_size,), np.int8),
                "priority": np.ones((batch_size,), np.float32),
                "has_policy": np.ones((batch_size,), np.uint8),
                "sf_multipv_raw": np.full((batch_size, SF_MULTIPV_RAW_MAX, 5), -1, np.int16),
                "has_sf_multipv_raw": np.zeros((batch_size,), np.uint8),
            }

    buf = cast(Any, _Buf())
    out = t._sample_batch_host(buf, batch_size=2, mirror_prob=0.0)
    assert isinstance(out, dict)
    assert "sf_multipv_raw" not in out

    t.sf_policy_sparse_ce = True
    out2 = t._sample_batch_host(buf, batch_size=2, mirror_prob=0.0)
    assert isinstance(out2, dict)
    assert "sf_multipv_raw" in out2


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
def test_record_dense_sf_policy_false_trains_via_sparse_ce():
    """End-to-end: dense target writing off -> samples carry only sparse
    labels, and compute_loss(sf_sparse_params=...) still supervises policy_sf."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.replay.dataset import collate_arrays
    from chess_anti_engine.replay.shard import samples_to_arrays
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.stockfish import StockfishUCI
    from chess_anti_engine.train.losses import compute_loss

    torch.manual_seed(0)
    model = build_model(ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
    )).eval()
    game = GameConfig(
        max_plies=24, sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
        categorical_bins=32, hlgauss_sigma=0.04,
        selfplay_fraction=0.0,
        record_dense_sf_policy=False,
    )
    assert SF_PATH is not None
    sf = StockfishUCI(SF_PATH, nodes=100, multipv=4)
    try:
        samples, _stats = play_batch(
            model, device="cpu", rng=np.random.default_rng(0), stockfish=sf, games=2,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=4, mcts_type="gumbel"),
            diff_focus=DiffFocusConfig(min_keep=1.0),
            game=game,
        )
    finally:
        sf.close()

    labeled = [s for s in samples if s.sf_multipv_raw is not None]
    assert labeled, "no SF-labeled samples produced"
    assert all(s.sf_policy_target is None for s in samples)

    arrs = samples_to_arrays(samples)
    assert "sf_policy_target" not in arrs or not np.asarray(arrs.get("has_sf_policy", ())).any()
    batch = collate_arrays(
        {k: np.asarray(v) for k, v in arrs.items()}, device="cpu",
    )
    n = batch["x"].shape[0]
    outputs = {
        "policy_own": torch.randn(n, 4672),
        "policy_sf": torch.randn(n, 4672),
        "wdl": torch.randn(n, 3),
    }
    params = SfTargetParams(sf_policy_temp=0.012, sf_policy_label_smooth=0.01)
    losses = compute_loss(outputs, batch, sf_sparse_params=params)
    assert torch.isfinite(losses["sf_move_ce"]).all()
    assert float(losses["sf_move_ce"]) > 0.0


def test_config_rejects_dense_off_without_sparse_ce():
    """record_dense_sf_policy: false without sf_policy_sparse_ce: true would
    silently drop policy_sf supervision — must fail loudly at config load."""
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    bad = {"selfplay": {"record_dense_sf_policy": False}}
    with pytest.raises(ValueError, match="sf_policy_sparse_ce"):
        flatten_run_config_defaults(bad)

    ok = {
        "selfplay": {"record_dense_sf_policy": False},
        "train": {"sf_policy_sparse_ce": True},
    }
    flat = flatten_run_config_defaults(ok)
    assert flat["record_dense_sf_policy"] is False
    assert flat["sf_policy_sparse_ce"] is True


def test_sf_policy_label_smoothing_only_when_uncovered() -> None:
    """Label smoothing is applied only when SF's candidates don't cover every
    legal move. Fully covered -> pure softmax (no flattening); uncovered -> the
    uncovered legal moves get a floor strictly below every covered move."""
    from chess_anti_engine.selfplay.stockfish_turn import _build_sf_policy_target

    legal = np.array([0, 1, 2], dtype=np.int64)
    # Fully covered: all 3 legal moves are candidates -> no smoothing.
    p_full = _build_sf_policy_target(
        [0, 1, 2], [3.0, 1.0, 0.0], legal_indices=legal,
        sf_policy_temp=1.0, sf_policy_label_smooth=0.1,
    )
    raw = np.exp(np.array([3.0, 1.0, 0.0]))
    raw /= raw.sum()
    assert np.allclose([p_full[0], p_full[1], p_full[2]], raw, atol=1e-5)

    # Uncovered: only 2 of 3 legal moves scored -> smoothing on; idx2 gets a floor.
    p_unc = _build_sf_policy_target(
        [0, 1], [3.0, 1.0], legal_indices=legal,
        sf_policy_temp=1.0, sf_policy_label_smooth=0.1,
    )
    assert p_unc[2] > 0.0
    assert p_unc[0] > p_unc[2]
    assert p_unc[1] > p_unc[2]


def test_native_wdl_stored_as_permille_from_fraction_scale():
    """_parse_wdl normalizes UCI permille to FRACTIONS; the collectors must
    rescale to permille per the shard schema. Regression for the dormant bug
    where int(round(0.87)) stored 0/1 junk in the native-WDL columns."""
    from types import SimpleNamespace

    from chess_anti_engine.selfplay.stockfish_turn import (
        _collect_sf_label_meta,
        _collect_sparse_pv_rows,
    )

    res = SimpleNamespace(
        nodes=1000, depth=20, cp=150, mate=None,
        wdl=np.array([0.87, 0.09, 0.04], np.float32),
        pvs=[SimpleNamespace(
            move_uci="e2e4", cp=150, mate=None,
            wdl=np.array([0.87, 0.09, 0.04], np.float32),
        )],
    )
    meta = _collect_sf_label_meta(res)
    assert (meta[4], meta[5]) == (870, 90)

    from chess_anti_engine.moves.encode import uci_to_policy_index
    a = uci_to_policy_index("e2e4", True)
    rows = _collect_sparse_pv_rows(res, turn=True, legal_set={a})
    assert rows is not None
    assert (rows[0][3], rows[0][4]) == (870, 90)


def test_sparse_ce_native_wdl_fallback_matches_dense_at_fraction_scale():
    """Codex P2 (PR #94): candidates with NO cp/mate fall back to native-WDL
    scoring even with the logistic enabled; the sparse torch path must rescale
    permille to fractions like target_builder._row_score, or the softmax
    temperature sees 1000x-scale scores."""
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY
    from chess_anti_engine.train.losses import soft_cross_entropy
    from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

    shard_width = 4672
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=True,
        sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0,
    )
    legal_full = np.asarray(COMPACT_TO_FULL_POLICY, dtype=np.int64)[[100, 200, 300]]
    raw = np.full((1, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    # native-only candidates: no cp, no mate, permille wdl present
    raw[0, 0] = (legal_full[0], SF_CP_SENTINEL, 0, 870, 90)
    raw[0, 1] = (legal_full[1], SF_CP_SENTINEL, 0, 400, 350)

    legal = np.zeros((1, shard_width), np.float32)
    legal[:, legal_full] = 1.0
    dense = rebuild_sf_policy_target(
        raw[0], legal_indices=legal_full, policy_size=shard_width, params=params,
    )
    assert dense is not None

    torch.manual_seed(3)
    logits = torch.randn(1, shard_width)
    batch = {
        "sf_multipv_raw": torch.from_numpy(raw.astype(np.int64)),
        "has_sf_multipv_raw": torch.ones(1),
        "sf_legal_mask": torch.from_numpy(legal),
        "has_sf_legal_mask": torch.ones(1),
        "sf_move_index": torch.tensor([int(legal_full[0])]),
        "has_sf_move": torch.ones(1),
    }
    dense_ce = soft_cross_entropy(logits, torch.from_numpy(dense[None]))
    sparse_ce, ok = sparse_sf_policy_ce(
        logits, batch, params=params, legal_aligned=batch["sf_legal_mask"],
    )
    assert ok.tolist() == [1.0]
    torch.testing.assert_close(sparse_ce, dense_ce, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# Vectorized batch rebuild: bitwise parity with the per-row reference, and the
# cross-ply masking that keeps sf_p0 from going stale under it.
# --------------------------------------------------------------------------

def _adversarial_batch(
    rng: np.random.Generator, *, n: int, width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(raw, legal_dense) covering every branch the scalar rebuild can take.

    Mixes: variable candidate counts (including 0 and the full 48), cp-only /
    mate-only / native-wdl-only / entirely unscorable rows, legal sets that are
    fully covered and partially covered, and empty legal masks.
    """
    raw = np.full((n, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[:, :, 2] = 0
    legal = np.zeros((n, width), np.uint8)
    for i in range(n):
        k = int(rng.integers(0, SF_MULTIPV_RAW_MAX + 1))
        moves = rng.choice(width, size=max(k, 1), replace=False)[:k]
        for j, mv in enumerate(moves):
            kind = int(rng.integers(0, 4))
            if kind == 0:                                  # cp only
                raw[i, j] = (mv, int(rng.integers(-3000, 3000)), 0, -1, -1)
            elif kind == 1:                                # mate only
                raw[i, j] = (mv, SF_CP_SENTINEL, int(rng.integers(-60, 61)) or 3, -1, -1)
            elif kind == 2:                                # native permille wdl
                w = int(rng.integers(0, 1001))
                raw[i, j] = (mv, SF_CP_SENTINEL, 0, w, int(rng.integers(0, 1001 - w)))
            else:                                          # unscorable
                raw[i, j] = (mv, SF_CP_SENTINEL, 0, -1, -1)
        mode = int(rng.integers(0, 3))
        if mode == 0 and k:                    # legal set == candidates (covered)
            legal[i, moves] = 1
        elif mode == 1:                        # candidates + extra legal moves
            extra = rng.choice(width, size=int(rng.integers(1, 12)), replace=False)
            legal[i, moves.astype(np.int64)] = 1
            legal[i, extra] = 1
        # mode == 2: empty legal mask
    return raw, legal


def _scalar_reference(
    raw: np.ndarray, legal: np.ndarray, width: int, params: SfTargetParams,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((raw.shape[0], width), np.float32)
    ok = np.zeros((raw.shape[0],), bool)
    for i in range(raw.shape[0]):
        r = rebuild_sf_policy_target(
            raw[i], legal_indices=np.flatnonzero(legal[i]),
            policy_size=width, params=params,
        )
        if r is not None:
            out[i], ok[i] = r, True
    return out, ok


@pytest.mark.parametrize("use_logistic", [False, True])
@pytest.mark.parametrize("smooth", [0.0, 0.01, 0.05])
def test_batch_rebuild_is_bitwise_equal_to_scalar(use_logistic: bool, smooth: float):
    """The vectorized path must be BITWISE equal, not merely close: it is the
    live target, and a last-ulp drift would make every parity assertion
    elsewhere in this file a tolerance question instead of an identity."""
    width = 1858
    rng = np.random.default_rng(7)
    raw, legal = _adversarial_batch(rng, n=64, width=width)
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=smooth,
        sf_wdl_use_cp_logistic=use_logistic,
        sf_wdl_cp_slope=0.0060, sf_wdl_cp_draw_width=120.0,
    )
    want, want_ok = _scalar_reference(raw, legal, width, params)
    got, got_ok = rebuild_sf_policy_targets_batch(
        raw, legal_dense=legal, policy_size=width, params=params,
    )
    np.testing.assert_array_equal(got_ok, want_ok)
    np.testing.assert_array_equal(got[want_ok], want[want_ok])


def test_batch_rebuild_handles_no_legal_mask_and_empty_batch():
    width = 64
    rng = np.random.default_rng(3)
    raw, legal = _adversarial_batch(rng, n=16, width=width)
    params = SfTargetParams(sf_policy_temp=0.05, sf_policy_label_smooth=0.05)
    # legal_dense=None must behave exactly like an all-zero legal mask (the
    # scalar path's empty legal_indices: smoothing is skipped).
    got_none, ok_none = rebuild_sf_policy_targets_batch(
        raw, legal_dense=None, policy_size=width, params=params,
    )
    want, want_ok = _scalar_reference(raw, np.zeros_like(legal), width, params)
    np.testing.assert_array_equal(ok_none, want_ok)
    np.testing.assert_array_equal(got_none[want_ok], want[want_ok])

    empty, empty_ok = rebuild_sf_policy_targets_batch(
        raw[:0], legal_dense=legal[:0], policy_size=width, params=params,
    )
    assert empty.shape == (0, width)
    assert empty_ok.shape == (0,)


def test_batch_rebuild_all_rows_unscorable_returns_not_ok():
    width = 32
    raw = np.full((3, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    got, ok = rebuild_sf_policy_targets_batch(
        raw, legal_dense=np.ones((3, width), np.uint8), policy_size=width,
        params=SfTargetParams(sf_policy_temp=0.012),
    )
    assert not ok.any()
    assert not got.any()


def test_batch_rebuild_matches_scalar_with_duplicate_move_indices():
    """The scatter is an ACCUMULATION in the scalar path. Real MultiPV rows
    never repeat a move, but nothing in the storage format forbids it, so pin
    the two paths together on a batch that does repeat one."""
    width = 32
    raw = np.full((2, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (5, 40, 0, 700, 200)
    raw[0, 1] = (5, 10, 0, 600, 250)      # same move index twice
    raw[0, 2] = (9, -30, 0, 300, 350)
    raw[1, 0] = (1, 0, 0, 500, 400)
    legal = np.zeros((2, width), np.uint8)
    legal[:, [1, 5, 9, 11]] = 1
    params = SfTargetParams(
        sf_policy_temp=0.05, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.0060,
        sf_wdl_cp_draw_width=120.0,
    )
    want, want_ok = _scalar_reference(raw, legal, width, params)
    got, got_ok = rebuild_sf_policy_targets_batch(
        raw, legal_dense=legal, policy_size=width, params=params,
    )
    np.testing.assert_array_equal(got_ok, want_ok)
    np.testing.assert_array_equal(got[want_ok], want[want_ok])


@pytest.mark.parametrize("use_logistic", [False, True])
def test_batch_sf_wdl_is_bitwise_equal_to_scalar(use_logistic: bool):
    rng = np.random.default_rng(11)
    meta = np.zeros((32, 6), np.int32)
    meta[:, 2] = rng.integers(-2000, 2000, size=32)
    meta[:, 3] = rng.integers(-5, 6, size=32)
    meta[:, 4] = rng.integers(-1, 1001, size=32)
    meta[:, 5] = rng.integers(-1, 400, size=32)
    meta[0, 2] = SF_CP_SENTINEL           # no cp, no mate, no native -> None
    meta[0, 3] = 0
    meta[0, 4] = meta[0, 5] = -1
    params = SfTargetParams(
        sf_wdl_use_cp_logistic=use_logistic,
        sf_wdl_cp_slope=0.0060, sf_wdl_cp_draw_width=120.0,
    )
    want = np.zeros((32, 3), np.float32)
    want_ok = np.zeros((32,), bool)
    for i in range(32):
        r = rebuild_sf_wdl(meta[i], params)
        if r is not None:
            want[i], want_ok[i] = r, True
    got, got_ok = rebuild_sf_wdl_batch(meta, params)
    np.testing.assert_array_equal(got_ok, want_ok)
    np.testing.assert_array_equal(got[want_ok], want[want_ok])


def _cross_ply_arrs(n: int = 4, width: int = 64) -> dict[str, np.ndarray]:
    raw = np.full((n, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[:, 0] = (3, 40, 0, 700, 200)
    raw[:, 1] = (5, -20, 0, 400, 300)
    legal = np.zeros((n, width), np.uint8)
    legal[:, [3, 5, 9]] = 1
    return {
        "sf_policy_target": np.zeros((n, width), np.float16),
        "sf_legal_mask": legal,
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": np.ones((n,), np.uint8),
        "sf_wdl": np.zeros((n, 3), np.float16),
        "sf_label_meta": np.zeros((n, 6), np.int32),
        "has_sf_label_meta": np.ones((n,), np.uint8),
        "sf_p0_policy_target": np.zeros((n, width), np.float16),
        "has_sf_p0": np.ones((n,), np.uint8),
        "sf_p0_regret": np.zeros((n, width), np.float16),
        "has_sf_p0_regret": np.ones((n,), np.uint8),
        "sf_volatility_target": np.zeros((n, 3), np.float16),
        "has_sf_volatility": np.ones((n,), np.uint8),
    }


def test_rebuild_masks_cross_ply_targets_and_spares_p0_regret():
    """sf_p0_policy_target is ply t-1's sf_policy_target and
    sf_volatility_target is |sf_wdl[t+6] - sf_wdl[t]|: both sources live on
    OTHER shard rows, so a sampled batch cannot move them with their source.
    They must be masked, not left on capture-time values. sf_p0_regret carries
    no SfTargetParams dependence at all, so it must survive."""
    arrs = _cross_ply_arrs()
    out = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams(sf_policy_temp=0.012))
    assert not out["has_sf_p0"].any()
    assert not out["has_sf_volatility"].any()
    assert out["has_sf_p0_regret"].all()


def test_masking_is_unconditional_so_control_and_treatment_pair():
    """Masking must NOT be conditional on the params having moved: a control
    run (flag on, capture-identical params) and a treatment run have to mask
    the same rows or the paired comparison is confounded by the own-move
    teacher switching on and off."""
    sharp = rebuild_sf_targets_in_arrays(
        _cross_ply_arrs(), params=SfTargetParams(sf_policy_temp=0.012),
    )
    soft = rebuild_sf_targets_in_arrays(
        _cross_ply_arrs(), params=SfTargetParams(sf_policy_temp=0.5),
    )
    np.testing.assert_array_equal(sharp["has_sf_p0"], soft["has_sf_p0"])
    np.testing.assert_array_equal(sharp["has_sf_volatility"], soft["has_sf_volatility"])
    # ...and the thing under test actually moved.
    assert not np.array_equal(sharp["sf_policy_target"], soft["sf_policy_target"])


def test_mask_cross_ply_is_a_noop_without_those_fields():
    arrs = {"sf_policy_target": np.zeros((2, 8), np.float16)}
    assert mask_cross_ply_sf_targets(arrs) == 0
    assert set(arrs) == {"sf_policy_target"}


def test_disabled_rebuild_leaves_cross_ply_flags_alone():
    """The masking is a consequence of the rebuild, so it must not happen when
    the flag is off — otherwise the default pipeline changes."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import trainer as trainer_mod

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    t = trainer_mod.Trainer(model, device="cpu", lr=1e-3)
    assert t.rebuild_sf_targets is False
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    out = t._prepare_host_arrays(arrs, rng=np.random.default_rng(0), mirror_prob=0.0)
    assert out["has_sf_p0"].all()
    assert out["has_sf_volatility"].all()

    t.set_sf_target_rebuild(enabled=True, params=t.sf_target_params)
    arrs2 = _cross_ply_arrs()
    arrs2["x"] = np.zeros((4, 146, 8, 8), np.float16)
    out2 = t._prepare_host_arrays(arrs2, rng=np.random.default_rng(0), mirror_prob=0.0)
    assert not out2["has_sf_p0"].any()
    assert not out2["has_sf_volatility"].any()


def test_set_sf_target_rebuild_is_a_live_surface():
    """rebuild_sf_targets + every SfTargetParams knob are construction-time on
    the Trainer; without this setter a live yaml edit is a silent no-op until
    the next restart."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import trainer as trainer_mod

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    t = trainer_mod.Trainer(model, device="cpu", lr=1e-3)
    base = t.sf_target_params

    assert t.set_sf_target_rebuild(enabled=True, params=base) is True
    assert t.rebuild_sf_targets is True
    assert t.set_sf_target_rebuild(enabled=True, params=base) is False  # idempotent
    moved = SfTargetParams(sf_policy_temp=base.sf_policy_temp + 0.1)
    assert t.set_sf_target_rebuild(enabled=True, params=moved) is True
    assert t.sf_target_params == moved
    assert t.set_sf_target_rebuild(enabled=False, params=moved) is True
    assert t.rebuild_sf_targets is False


def test_resolve_sf_target_params_is_shared_by_construction_and_live_push():
    """One reader for the yaml keys, so the constructor and the per-iteration
    push cannot disagree about what the config says."""
    from chess_anti_engine.train.trainer import (
        resolve_sf_target_params,
        trainer_kwargs_from_config,
    )

    cfg = {
        "sf_policy_temp": 0.012, "sf_policy_label_smooth": 0.01,
        "sf_wdl_use_cp_logistic": True, "sf_wdl_cp_slope": 0.0060,
        "sf_wdl_cp_draw_width": 120.0,
    }
    assert trainer_kwargs_from_config(cfg)["sf_target_params"] == resolve_sf_target_params(cfg)
    assert resolve_sf_target_params({}) == SfTargetParams()


def test_rebuild_sf_targets_is_a_live_yaml_key_and_defaults_off():
    from chess_anti_engine.utils import flatten_run_config_defaults

    flat = flatten_run_config_defaults({"train": {"rebuild_sf_targets": True}})
    assert flat["rebuild_sf_targets"] is True
    assert flatten_run_config_defaults({}).get("rebuild_sf_targets", False) is False
