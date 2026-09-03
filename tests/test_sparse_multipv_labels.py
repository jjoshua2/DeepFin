"""Sparse MultiPV label storage + train-time target rebuild.

Parity contract: with params equal to the capture-time config, targets
rebuilt from sf_multipv_raw/sf_label_meta match the stored ones to 1e-5.
Default behavior (rebuild_sf_targets=False) is bitwise-unchanged.
"""
from __future__ import annotations

import dataclasses
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
    CrossPlyMaskCounts,
    SfRebuildCoverage,
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
    out, _cov = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams(sf_policy_temp=0.012))
    assert float(out["sf_policy_target"][0].astype(np.float32).sum()) == pytest.approx(1.0, abs=1e-3)
    assert not np.array_equal(out["sf_policy_target"][0], pol[0])  # rebuilt
    np.testing.assert_array_equal(out["sf_policy_target"][1], before_row1)  # untouched


def test_trainer_default_is_bitwise_unchanged(monkeypatch):
    """rebuild_sf_targets=False must never call the rebuilder, and the
    sampled batch must be byte-identical to the pre-flag pipeline."""
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import target_builder as target_builder_mod
    from chess_anti_engine.train import trainer as trainer_mod

    calls = {"n": 0}

    def _spy(arrs, *, params):
        del params
        calls["n"] += 1
        return arrs, target_builder_mod.SfRebuildCoverage()

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
    from chess_anti_engine.train.trainer import _EXACT_MASKED_METRIC_FIELDS

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

    # Sparse-only rows widen the final sf_move mask inside compute_loss.  The
    # exact pooling denominator must use that realized mask, not the original
    # all-zero dense-target flag; rows 0 and 1 are valid, row 2 is not.
    sparse_only_batch = {**batch, "has_sf_policy": torch.zeros(n)}
    sparse_only = compute_loss(
        outputs,
        sparse_only_batch,
        sf_sparse_params=params,
        w_policy=0.0,
        w_soft=0.0,
        w_future=0.0,
        w_wdl=0.0,
        w_sf_move=1.0,
        w_sf_eval=0.0,
        w_categorical=0.0,
        w_volatility=0.0,
        w_sf_volatility=0.0,
        w_moves_left=0.0,
        report_exact_masked_sums=True,
    )
    _, exact_weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_move_loss"]
    assert float(sparse_only[exact_weight_key]) == 2.0
    assert float(sparse_only["total"]) == pytest.approx(
        float(sparse_only["sf_move_ce"]) * 2.0 / 3.0,
    )


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
    mate-only / native-wdl-only / entirely unscorable rows, rows carrying
    cp AND mate AND native wdl SIMULTANEOUSLY (mate precedence in the logistic
    path, native score otherwise), legal sets that are fully covered and
    partially covered, empty legal masks, and legal sets that EXCLUDE some
    candidates (live MultiPV rows are always legal moves, but the storage does
    not forbid otherwise and the smoothing gate counts candidates-IN-the-legal-
    set, so the exclusion branch must stay exercised).
    """
    raw = np.full((n, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[:, :, 2] = 0
    legal = np.zeros((n, width), np.uint8)
    for i in range(n):
        k = int(rng.integers(0, SF_MULTIPV_RAW_MAX + 1))
        moves = rng.choice(width, size=max(k, 1), replace=False)[:k]
        for j, mv in enumerate(moves):
            kind = int(rng.integers(0, 5))
            if kind == 0:                                  # cp only
                raw[i, j] = (mv, int(rng.integers(-3000, 3000)), 0, -1, -1)
            elif kind == 1:                                # mate only
                raw[i, j] = (mv, SF_CP_SENTINEL, int(rng.integers(-60, 61)) or 3, -1, -1)
            elif kind == 2:                                # native permille wdl
                w = int(rng.integers(0, 1001))
                raw[i, j] = (mv, SF_CP_SENTINEL, 0, w, int(rng.integers(0, 1001 - w)))
            elif kind == 3:                                # unscorable
                raw[i, j] = (mv, SF_CP_SENTINEL, 0, -1, -1)
            else:                                          # cp + mate + native, all at once
                w = int(rng.integers(0, 1001))
                raw[i, j] = (
                    mv, int(rng.integers(-3000, 3000)),
                    int(rng.integers(-60, 61)) or 3,
                    w, int(rng.integers(0, 1001 - w)),
                )
        mode = int(rng.integers(0, 4))
        if mode == 0 and k:                    # legal set == candidates (covered)
            legal[i, moves] = 1
        elif mode == 1:                        # candidates + extra legal moves
            extra = rng.choice(width, size=int(rng.integers(1, 12)), replace=False)
            legal[i, moves.astype(np.int64)] = 1
            legal[i, extra] = 1
        elif mode == 3:                        # only HALF the candidates are legal
            extra = rng.choice(width, size=int(rng.integers(1, 12)), replace=False)
            legal[i, moves.astype(np.int64)[: max(1, k // 2)]] = 1
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
    """`np.add.at` is an ACCUMULATION, so anything folded in before the scatter
    is not order-neutral when a move index repeats: scaling first computes
    p1*s + p2*s where the scalar path computes (p1+p2)*s. Real MultiPV rows
    never repeat a move, but nothing in the storage format forbids it and a
    single hand-picked pair passes by coincidence (both orderings round the
    same way). Randomize hard enough that only genuine bitwise equality
    survives -- an earlier revision of this file folded the scale in and 25 of
    30 seeds diverged at ~6e-08."""
    width = 24
    params = SfTargetParams(
        sf_policy_temp=0.05, sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.0060,
        sf_wdl_cp_draw_width=120.0,
    )
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n = 6
        raw = np.full((n, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
        raw[:, :, 1] = SF_CP_SENTINEL
        for i in range(n):
            k = int(rng.integers(4, 12))
            # Draw WITH replacement from a small index pool: guarantees repeats.
            moves = rng.integers(0, 6, size=k)
            for j, mv in enumerate(moves):
                raw[i, j] = (
                    int(mv), int(rng.integers(-800, 800)), 0,
                    int(rng.integers(0, 1001)), int(rng.integers(0, 300)),
                )
        legal = np.zeros((n, width), np.uint8)
        legal[:, :9] = 1                     # 9 legal > 6 covered -> smoothing fires
        want, want_ok = _scalar_reference(raw, legal, width, params)
        got, got_ok = rebuild_sf_policy_targets_batch(
            raw, legal_dense=legal, policy_size=width, params=params,
        )
        np.testing.assert_array_equal(got_ok, want_ok)
        np.testing.assert_array_equal(
            got[want_ok], want[want_ok],
            err_msg=f"seed {seed}: batch != scalar with duplicate move indices",
        )


def test_batch_rebuild_raises_on_out_of_range_move_index():
    """A move index >= policy_size must raise, exactly as the scalar path does.

    Scattering on a flattened `rows * width + cols` would instead write row i's
    probability mass into row i+1 -- silent cross-row target contamination from
    a policy-encoding mismatch (legacy 4672-space raw against an 1858-wide
    target, or a future width change). No live row triggers it; the point is
    that the failure mode is a crash and not wrong targets."""
    width = 32
    raw = np.full((3, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (5, 40, 0, 700, 200)
    raw[1, 0] = (39, 40, 0, 700, 200)      # out of range for width=32
    raw[2, 0] = (7, 10, 0, 600, 250)
    legal = np.zeros((3, width), np.uint8)
    legal[:, [5, 7, 9]] = 1
    params = SfTargetParams(sf_policy_temp=0.012, sf_policy_label_smooth=0.01)

    with pytest.raises(IndexError):
        rebuild_sf_policy_target(
            raw[1], legal_indices=np.flatnonzero(legal[1]),
            policy_size=width, params=params,
        )
    with pytest.raises(IndexError):
        rebuild_sf_policy_targets_batch(
            raw, legal_dense=legal, policy_size=width, params=params,
        )


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
    out, cov = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams(sf_policy_temp=0.012))
    assert cov.cross_ply_masked == 4      # ROWS masked, not flags cleared (8)
    assert cov.policy_rebuilt == 4
    # Per-flag PRE-mask decomposition: every fixture row carried both flags.
    assert cov.p0_masked == 4
    assert cov.volatility_masked == 4
    assert not out["has_sf_p0"].any()
    assert not out["has_sf_volatility"].any()
    assert out["has_sf_p0_regret"].all()


def test_masking_is_unconditional_so_control_and_treatment_pair():
    """Masking must NOT be conditional on the params having moved: a control
    run (flag on, capture-identical params) and a treatment run have to mask
    the same rows or the paired comparison is confounded by the own-move
    teacher switching on and off."""
    sharp, _ = rebuild_sf_targets_in_arrays(
        _cross_ply_arrs(), params=SfTargetParams(sf_policy_temp=0.012),
    )
    soft, _ = rebuild_sf_targets_in_arrays(
        _cross_ply_arrs(), params=SfTargetParams(sf_policy_temp=0.5),
    )
    np.testing.assert_array_equal(sharp["has_sf_p0"], soft["has_sf_p0"])
    np.testing.assert_array_equal(sharp["has_sf_volatility"], soft["has_sf_volatility"])
    # ...and the thing under test actually moved.
    assert not np.array_equal(sharp["sf_policy_target"], soft["sf_policy_target"])


def test_mask_cross_ply_is_a_noop_without_those_fields():
    arrs = {"sf_policy_target": np.zeros((2, 8), np.float16)}
    assert mask_cross_ply_sf_targets(arrs) == CrossPlyMaskCounts()
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
    out2 = t._prepare_host_arrays(
        arrs2, rng=np.random.default_rng(0), mirror_prob=0.0,
        rebuild_sf_targets=True,   # the training path's explicit opt-in
    )
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


# --------------------------------------------------------------------------
# The frozen holdout ruler must not move when the TRAINING target moves.
# --------------------------------------------------------------------------

def _tiny_trainer():
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.train import trainer as trainer_mod

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    return trainer_mod.Trainer(model, device="cpu", lr=1e-3)


class _SliceBuf:
    """Minimal ReplayBuffer stand-in for the deterministic full-pass path."""

    rng = np.random.default_rng(0)

    def __init__(self, arrs: dict[str, np.ndarray], n: int) -> None:
        self._arrs = arrs
        self._n = n

    def __len__(self) -> int:
        return self._n

    def batch_row_bounds(self, bs: int) -> list[tuple[int, int]]:
        return [(i, min(i + bs, self._n)) for i in range(0, self._n, bs)]

    def rows_slice_arrays(self, start: int, stop: int) -> dict[str, np.ndarray]:
        return {k: np.array(v[start:stop], copy=True) for k, v in self._arrs.items()}


def test_full_pass_ruler_is_not_rebuilt_even_when_the_flag_is_on():
    """`_full_pass_host_batch` is the frozen holdout ruler. With the rebuild on
    it would (a) score against rebuilt targets and (b) lose w_sf_own and
    w_sf_volatility from `total` via the cross-ply masking — both change what
    test_loss MEANS with no holdout_generation bump, and (b) moves it DOWN, so
    it reads as improvement. Pinned off, exactly like mirror_prob."""
    t = _tiny_trainer()
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    buf = cast(Any, _SliceBuf(arrs, 4))

    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.5))
    assert t.rebuild_sf_targets is True

    ruler = t._full_pass_host_batch(buf, start=0, stop=4)
    # The eval keeps every loss term...
    assert ruler["has_sf_p0"].all()
    assert ruler["has_sf_volatility"].all()
    # ...and the stored targets, untouched.
    np.testing.assert_array_equal(ruler["sf_policy_target"], arrs["sf_policy_target"][:4])
    np.testing.assert_array_equal(ruler["sf_wdl"], arrs["sf_wdl"][:4])

    # ...while the TRAINING path with the same trainer does rebuild.
    trained = t._prepare_host_arrays(
        {k: np.array(v, copy=True) for k, v in arrs.items()},
        rng=np.random.default_rng(0), mirror_prob=0.0,
        rebuild_sf_targets=True,
    )
    assert not trained["has_sf_p0"].any()
    assert not np.array_equal(trained["sf_policy_target"], arrs["sf_policy_target"])


def test_full_pass_ruler_is_byte_identical_across_a_flag_flip():
    """The consequence that matters: flipping rebuild_sf_targets must not move
    a single byte of the holdout batch, so pre-flip and post-flip test_loss
    stay on one instrument and `_update_best_model` cannot promote across a
    definitional step."""
    t = _tiny_trainer()
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    buf = cast(Any, _SliceBuf(arrs, 4))

    off = t._full_pass_host_batch(buf, start=0, stop=4)
    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.5))
    on = t._full_pass_host_batch(buf, start=0, stop=4)

    assert set(off) == set(on)
    for k in off:
        np.testing.assert_array_equal(off[k], on[k], err_msg=f"holdout batch moved: {k}")


def test_rebuild_coverage_is_reported_and_is_not_total():
    """A rebuild whose coverage cannot be observed is unfalsifiable. The
    counters must (a) read 0 with the flag off, (b) go non-zero with it on, and
    (c) report BELOW 1.0 when some SF-labelled rows carry no sf_multipv_raw —
    those keep capture-time targets, which is the gap the PR body has to own."""
    t = _tiny_trainer()
    arrs = _cross_ply_arrs(n=4)
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    # Row 3 is SF-labelled but has NO raw rows: unreachable by the rebuild.
    arrs["has_sf_multipv_raw"] = np.array([1, 1, 1, 0], np.uint8)

    assert t._sf_rebuild_coverage.drain() == {
        "sf_rebuild_policy_frac": 0.0,
        "sf_rebuild_wdl_frac": 0.0,
        "sf_rebuild_masked_frac": 0.0,
        "sf_rebuild_masked_p0_frac": 0.0,
        "sf_rebuild_masked_volatility_frac": 0.0,
    }

    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.012))
    t._prepare_host_arrays(
        {k: np.array(v, copy=True) for k, v in arrs.items()},
        rng=np.random.default_rng(0), mirror_prob=0.0,
        rebuild_sf_targets=True,
    )
    got = t._sf_rebuild_coverage.drain()
    assert got["sf_rebuild_policy_frac"] == pytest.approx(3 / 4)   # NOT 1.0
    assert got["sf_rebuild_wdl_frac"] == pytest.approx(1.0)
    # ROWS that lost a cross-ply target / rows — a real fraction, never > 1.0
    # even though each row here carried BOTH has_sf_p0 and has_sf_volatility.
    assert got["sf_rebuild_masked_frac"] == pytest.approx(1.0)
    # ...and the per-flag PRE-mask decomposition of that row count.
    assert got["sf_rebuild_masked_p0_frac"] == pytest.approx(1.0)
    assert got["sf_rebuild_masked_volatility_frac"] == pytest.approx(1.0)
    # Drained, not accumulated forever.
    assert t._sf_rebuild_coverage.drain()["sf_rebuild_policy_frac"] == 0.0


def test_full_pass_contributes_nothing_to_the_coverage_metric():
    """sf_rebuild_* on the `eval_full_pass` row must stay 0.0 by construction,
    so a non-zero value there is itself the alarm that the ruler moved."""
    t = _tiny_trainer()
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.012))
    t._full_pass_host_batch(cast(Any, _SliceBuf(arrs, 4)), start=0, stop=4)
    assert t._sf_rebuild_coverage.drain()["sf_rebuild_policy_frac"] == 0.0


def test_an_eval_does_not_drain_the_training_coverage_counters(monkeypatch):
    """`drain()` RESETS, so a shared accumulator is not merely imprecise.

    The async holdout eval calls `_compute_metrics` on the SAME Trainer from
    its own thread while the next iteration trains
    (`distributed_async_test_eval: true`), so a shared sink would have it
    publish the TRAINING path's counts on the `eval` row and leave the `train`
    row short by an unknowable amount -- breaking both the proof-of-effect and
    the "non-zero on the ruler's row means the ruler rebuilt" rule.

    Reasoning about which paths ACCUMULATE does not catch this: the full pass
    accumulates nothing and still drains.
    """
    t = _tiny_trainer()
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)
    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.012))

    # Training accumulates into the trainer-wide sink.
    t._prepare_host_arrays(
        {k: np.array(v, copy=True) for k, v in arrs.items()},
        rng=np.random.default_rng(0), mirror_prob=0.0,
        rebuild_sf_targets=True,
    )

    # A full-pass eval runs on the SAME trainer. The batch iterator is stubbed
    # to yield nothing (the tiny fixture rows cannot be collated) but to record
    # the sink it was handed and add its own counts to it -- so this pins the
    # THREADING and the drain site, not the forward pass.
    seen: list[Any] = []

    def _fake_iter(_buf, *, coverage=None, **_kw):
        seen.append(coverage)
        if coverage is not None:
            coverage.add(SfRebuildCoverage(rows=8, policy_rebuilt=2))
        return iter(())

    published: dict[str, Any] = {}

    def _fake_build(*_a, **kw):
        published.update(kw)
        return "metrics"

    monkeypatch.setattr(t, "_iter_full_pass_batches", _fake_iter)
    monkeypatch.setattr(t, "_build_metrics", _fake_build)
    monkeypatch.setattr(t, "_log_metrics", lambda *_a, **_k: None)
    t._compute_metrics(buf=cast(Any, _SliceBuf(arrs, 4)), batch_size=4,
                       steps=0, tag="eval", full_pass=True)

    # The eval got a sink of its own, and publishes ITS counts -- so "non-zero
    # on the ruler's row" stays a statement that can fail.
    assert len(seen) == 1
    assert seen[0] is not t._sf_rebuild_coverage
    assert published["sf_rebuild_policy_frac"] == pytest.approx(2 / 8)

    # ...and the training counts it must not have consumed are still there.
    assert t._sf_rebuild_coverage.drain()["sf_rebuild_policy_frac"] == pytest.approx(1.0)


def test_the_ruler_alarm_reaches_progress_csv_not_only_tensorboard():
    """The eval-row coverage is the alarm that the frozen ruler rebuilt its own
    targets, and `_full_pass_host_batch` forwards its `coverage` sink purely to
    keep that alarm fail-able. An alarm has to be readable where the doc sends
    the operator: `_TEST_METRIC_KEYS` / `_test_and_drift_dict` are an
    ENUMERATED whitelist, so a key absent from them reaches only TensorBoard --
    whose event files rotate per Ray session, which is why the grad-norm
    metrics were promoted to TrainMetrics in the first place.
    """
    from chess_anti_engine.train.trainer import TrainMetrics
    from chess_anti_engine.tune.trainable_report import (
        _TEST_METRIC_KEYS,
        _test_and_drift_dict,
    )
    from chess_anti_engine.tune.trial_config import DriftMetrics, TrainingResult

    want = (
        "test_sf_rebuild_policy_frac",
        "test_sf_rebuild_wdl_frac",
        "test_sf_rebuild_masked_frac",
        "test_sf_rebuild_masked_p0_frac",
        "test_sf_rebuild_masked_volatility_frac",
    )
    for key in want:
        assert key in _TEST_METRIC_KEYS, f"{key} would never reach progress.csv"

    # Present (as NaN) even when no eval ran, so Ray locks the column on row 1.
    empty = _test_and_drift_dict(
        tr=TrainingResult(), drift=DriftMetrics(),
        holdout_frozen=False, holdout_generation=0,
    )
    for key in want:
        assert key in empty

    # ...and carries the eval's OWN value when one did.
    tr = TrainingResult()
    tr.test_metrics = TrainMetrics(
        loss=0.5, policy_loss=0.5, soft_policy_loss=0.5, future_policy_loss=0.5,
        wdl_loss=0.5, sf_move_loss=0.5, sf_move_acc=0.5, sf_eval_loss=0.5,
        categorical_loss=0.5, volatility_loss=0.5, sf_volatility_loss=0.5,
        moves_left_loss=0.5, eval_rows=8,
        sf_rebuild_policy_frac=0.25,
    )
    tr.test_metrics_source_iter = 41
    row = _test_and_drift_dict(
        tr=tr, drift=DriftMetrics(),
        holdout_frozen=False, holdout_generation=0,
    )
    # A non-zero value here IS the alarm -- it must survive to the row.
    assert row["test_sf_rebuild_policy_frac"] == pytest.approx(0.25)
    assert row["test_sf_rebuild_wdl_frac"] == 0.0


def test_sf_target_params_are_not_written_when_no_consumer_reads_them():
    """`sf_target_params` also feeds `sf_sparse_params` in `_loss_kwargs`. An
    unconditional live write would mean that the day `sf_policy_sparse_ce` is
    switched on, a live sf_policy_temp edit silently retargets the sparse-CE
    loss. Inert-because-a-yaml-key-is-absent is not a guarantee."""
    t = _tiny_trainer()
    base = t.sf_target_params
    moved = SfTargetParams(sf_policy_temp=base.sf_policy_temp + 0.1)

    assert t.rebuild_sf_targets is False
    assert t.sf_policy_sparse_ce is False
    assert t.set_sf_target_rebuild(enabled=False, params=moved) is False
    assert t.sf_target_params == base            # untouched, and nothing reads it

    t.sf_policy_sparse_ce = True                 # second consumer now live
    assert t.set_sf_target_rebuild(enabled=False, params=moved) is True
    assert t.sf_target_params == moved           # written, and LOGGED by the caller
    assert t._loss_kwargs["sf_sparse_params"] == moved


# --------------------------------------------------------------------------
# PR #283 review follow-up: observability, fail-closed gating, default
# unification, and the parity gates for the vectorized-path efficiency edits.
# --------------------------------------------------------------------------

def test_masked_frac_decomposition_is_the_rebuild_mode_outage_detector():
    """The cross-ply mask zeroes `has_sf_p0` indistinguishably from "never
    recorded", so `has_sf_p0_frac` — documented in trainable_report.py as the
    sf_p0 OUTAGE detector — is pinned at 0.0 for the whole of any rebuild
    experiment. The per-flag PRE-mask fractions are the detector that keeps
    working: a real p0 outage reads masked_p0 -> 0 while volatility stays up,
    which the row-level `sf_rebuild_masked_frac` alone cannot distinguish."""
    arrs = _cross_ply_arrs()
    arrs["has_sf_p0"] = np.array([1, 1, 0, 0], np.uint8)
    arrs["has_sf_volatility"] = np.array([1, 0, 1, 0], np.uint8)
    _, cov = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams())
    assert cov.p0_masked == 2
    assert cov.volatility_masked == 2
    assert cov.cross_ply_masked == 3          # rows 0, 1, 2 lost >= 1 flag
    kw = cov.metric_kwargs()
    assert kw["sf_rebuild_masked_p0_frac"] == pytest.approx(2 / 4)
    assert kw["sf_rebuild_masked_volatility_frac"] == pytest.approx(2 / 4)
    assert kw["sf_rebuild_masked_frac"] == pytest.approx(3 / 4)

    # The outage itself: workers stop recording p0. Post-mask flags read 0
    # either way; only the pre-mask column can see it go to exactly 0.0.
    outage = _cross_ply_arrs()
    outage["has_sf_p0"] = np.zeros(4, np.uint8)
    _, cov2 = rebuild_sf_targets_in_arrays(outage, params=SfTargetParams())
    kw2 = cov2.metric_kwargs()
    assert kw2["sf_rebuild_masked_p0_frac"] == 0.0
    assert kw2["sf_rebuild_masked_volatility_frac"] == pytest.approx(1.0)


def test_rows_probe_falls_back_to_cross_ply_flags_so_frac_stays_a_fraction():
    """A batch lacking all four row-count probe keys but carrying `has_sf_p0`
    used to get rows=0 while the mask still counted rows, turning
    `sf_rebuild_masked_frac` into a raw COUNT (> 1.0). Unreachable from the
    live schema today; the fallback keeps the invariant structural."""
    arrs = {
        "has_sf_p0": np.ones(6, np.uint8),
        "has_sf_volatility": np.ones(6, np.uint8),
    }
    _, cov = rebuild_sf_targets_in_arrays(arrs, params=SfTargetParams())
    assert cov.rows == 6
    kw = cov.metric_kwargs()
    assert kw["sf_rebuild_masked_frac"] == pytest.approx(1.0)
    for key, value in kw.items():
        assert 0.0 <= value <= 1.0, f"{key} is not a fraction: {value}"


class _SampleBuf:
    """Minimal ReplayBuffer stand-in for the SAMPLED (training) path."""

    rng = np.random.default_rng(0)

    def __init__(self, arrs: dict[str, np.ndarray]) -> None:
        self._arrs = arrs

    def sample_batch_arrays(self, _bs: int) -> dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in self._arrs.items()}


def test_prepare_host_arrays_defaults_to_stored_targets():
    """The rebuild gate FAILS CLOSED: a NEW producer calling
    `_prepare_host_arrays` without taking a position gets STORED targets even
    when the trainer flag is on. The default used to be True with the ruler
    protected only by a per-callsite False pin — under that arrangement this
    very call would have silently rebuilt. The sampled training path is the
    one explicit opt-in."""
    t = _tiny_trainer()
    t.set_sf_target_rebuild(enabled=True, params=SfTargetParams(sf_policy_temp=0.5))
    arrs = _cross_ply_arrs()
    arrs["x"] = np.zeros((4, 146, 8, 8), np.float16)

    # Hypothetical future producer: no kwarg -> stored bytes, flags intact,
    # nothing added to the coverage counters.
    out = t._prepare_host_arrays(
        {k: np.array(v, copy=True) for k, v in arrs.items()},
        rng=np.random.default_rng(0), mirror_prob=0.0,
    )
    np.testing.assert_array_equal(out["sf_policy_target"], arrs["sf_policy_target"])
    np.testing.assert_array_equal(out["sf_wdl"], arrs["sf_wdl"])
    assert out["has_sf_p0"].all()
    assert out["has_sf_volatility"].all()
    assert t._sf_rebuild_coverage.drain()["sf_rebuild_policy_frac"] == 0.0

    # The training path opts in explicitly and does rebuild.
    trained = t._sample_batch_host(
        cast(Any, _SampleBuf(arrs)), batch_size=4, mirror_prob=0.0,
    )
    assert isinstance(trained, dict)
    assert not np.array_equal(trained["sf_policy_target"], arrs["sf_policy_target"])
    assert not trained["has_sf_p0"].any()
    assert t._sf_rebuild_coverage.drain()["sf_rebuild_policy_frac"] == pytest.approx(1.0)


def test_metric_splat_sources_cannot_collide_in_build_metrics():
    """`train_steps` splats `coverage.drain()` AND `_loss_sums_to_metric_kwargs`
    (plus grad-clip and explicit extras) into `_build_metrics` ->
    `TrainMetrics(**...)`. A duplicate keyword raises TypeError only AT
    RUNTIME, in production, on the first iteration after someone names a
    compute_loss scalar like a coverage column. Pin the disjointness here so
    that collision fails in CI instead. The loss side is DRIVEN (a real
    compute_loss), then widened by the static key maps."""
    from chess_anti_engine.train.losses import compute_loss
    from chess_anti_engine.train import trainer as trainer_mod

    logits, sparse_batch, _params, dense = _sparse_ce_batch(
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
    losses = compute_loss(outputs, batch)
    sums = trainer_mod.Trainer._extract_loss_scalars(losses)
    loss_kwargs = trainer_mod._loss_sums_to_metric_kwargs(dict(sums), 1.0)
    loss_surface = (
        set(loss_kwargs)
        | set(trainer_mod._LOSS_KEY_TO_METRIC_FIELD.values())
        | set(trainer_mod._RATIO_METRIC_FIELDS)
    )

    coverage_kwargs = SfRebuildCoverage(
        rows=8, policy_rebuilt=4, wdl_rebuilt=2,
        cross_ply_masked=1, p0_masked=1, volatility_masked=1,
    ).metric_kwargs()
    grad_kwargs = trainer_mod._grad_clip_metric_kwargs(
        [1.0, 2.0],
        {"clipped": 1, "adaptive_clip": 0, "hard_clip": 1, "nonfinite_grad": 0},
        [0.5],
    )
    explicit = {
        "train_time_s": 1.0, "opt_step_time_s": 0.5, "train_steps_done": 2,
        "train_samples_seen": 4, "opt_lr_mean": 1e-4, "opt_lr_max": 2e-4,
    }
    extras = {**coverage_kwargs, **grad_kwargs, **explicit}
    # The extras must be disjoint among THEMSELVES too (a dict merge silently
    # keeps the last writer)...
    assert len(extras) == len(coverage_kwargs) + len(grad_kwargs) + len(explicit)
    # ...and none of them reachable from the loss path's splat.
    overlap = loss_surface & set(extras)
    assert not overlap, f"loss-path keys collide with _build_metrics extras: {overlap}"
    # The union constructs TrainMetrics — the exact call `_build_metrics`
    # makes, including its named accuracy kwargs (which are also keywords a
    # loss-path or extras key could collide with).
    acc_kwargs = {
        "sf_move_acc": 0.5, "sf_move_acc_top5": 0.5,
        "policy_own_acc_top1": 0.5, "policy_own_acc_top5": 0.5,
        "policy_future_acc_top1": 0.5, "policy_future_acc_top5": 0.5,
    }
    assert not set(acc_kwargs) & (loss_surface | set(extras))
    # cast: the runtime construction is the assertion; pyright cannot type a
    # heterogeneous kwargs splat against the dataclass's int/float fields.
    trainer_mod.TrainMetrics(
        **cast("dict[str, Any]", {**loss_kwargs, **acc_kwargs, **extras})
    )


# Derived, not hardcoded: a new SfTargetParams field (e.g. the 2026-08-05
# sf_policy_score_mode/sf_policy_cp_temp pair) is pinned automatically —
# the literal form silently exempted new keys from the one-home invariant.
_SF_TARGET_FIELDS = tuple(f.name for f in dataclasses.fields(SfTargetParams))


def test_sf_target_param_defaults_have_one_home():
    """The five SF target-construction defaults live on `SfTargetParams`.
    Every other reader either DERIVES from it (`resolve_sf_target_params`,
    `tune/distributed_runtime.build_recommended_worker`,
    `TrialConfig.from_dict`) or is PINNED equal here (the `GameConfig`
    / `TrialConfig` dataclass defaults), so a drifted copy fails CI instead
    of silently splitting capture-time and rebuilt targets on any config that
    omits the key.

    worker.py is the FIFTH reader and is covered by
    `test_worker_reco_defaults_derive_from_sf_target_params`, which drives it
    rather than asserting about it — this test used to merely NAME worker.py
    in its docstring while never executing a line of it, so hardcoding a
    literal at `worker.py:3384` survived as a mutation.
    """
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.selfplay.config import GameConfig
    from chess_anti_engine.train.trainer import resolve_sf_target_params
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
    from chess_anti_engine.tune.trial_config import TrialConfig

    d = SfTargetParams()
    fields = _SF_TARGET_FIELDS
    assert resolve_sf_target_params({}) == d
    for owner in (GameConfig, TrialConfig):
        for f in fields:
            assert getattr(owner, f) == getattr(d, f), (owner.__name__, f)
    reco = build_recommended_worker(
        config={}, model_cfg=ModelConfig(), sf_nodes=5000, mcts_simulations=8,
    )
    for f in fields:
        assert reco[f] == getattr(d, f), f


def test_worker_reco_defaults_derive_from_sf_target_params():
    """worker.py's reco reads must FALL BACK to `SfTargetParams`, not to a
    literal typed at the call site.

    A manifest that omits these keys (an older server, or any config that
    never set them) makes the worker stamp its own fallback onto the captured
    record while `resolve_sf_target_params` stamps `SfTargetParams` onto the
    rebuild — the exact capture-vs-rebuild split this family of tests exists
    to prevent, and the one a docstring naming worker.py could not catch.

    Driven through the real `_build_selfplay_configs` with an EMPTY reco, so
    replacing any default at `worker.py:3384+` with its current literal value
    fails here.
    """
    from chess_anti_engine.worker import WorkerSession

    from tests.test_reco_coverage import _bare_session

    d = SfTargetParams()
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        _bare_session(), {"sf_nodes": 5000},
    )
    game = cfgs["game"]
    for f in _SF_TARGET_FIELDS:
        assert getattr(game, f) == getattr(d, f), f

    # Negative control: the assertions above pass trivially if the worker
    # ignores the reco entirely (GameConfig's own dataclass defaults equal
    # SfTargetParams by the pin above), so prove EVERY field is reco-driven.
    # Dynamic over _SF_TARGET_FIELDS: deleting any single GameConfig kwarg in
    # the worker build must fail here — PR #355 review F2 found exactly that
    # deletion surviving while only two fields were bumped.
    def _bumped_value(default):
        if isinstance(default, bool):
            return not default
        if isinstance(default, str):
            return "cp" if default == "wdl" else "wdl"
        return default + 0.25

    bumped: dict[str, Any] = {"sf_nodes": 5000}
    for f in _SF_TARGET_FIELDS:
        bumped[f] = _bumped_value(getattr(d, f))
    cfgs2, _ = WorkerSession._build_selfplay_configs(_bare_session(), bumped)
    for f in _SF_TARGET_FIELDS:
        assert getattr(cfgs2["game"], f) == bumped[f], (
            f"worker GameConfig build dropped reco key {f}"
        )


def test_batch_sf_wdl_logistic_has_no_sentinel_overflow():
    """The WDL logistic used to run over ALL rows and select afterwards; rows
    without a cp carry the -32768 sentinel, which overflows exp() for
    sf_wdl_cp_slope >= ~0.022 — exactly the knob this rebuild exists to
    sweep. Compressed to the use_log rows (the policy twin's convention)
    there is nothing to overflow, and the result stays bitwise equal to the
    scalar path at the same slope."""
    import warnings

    meta = np.zeros((8, 6), np.int32)
    meta[:, 2] = SF_CP_SENTINEL      # no cp, no mate -> native rows
    meta[:, 3] = 0
    meta[:, 4] = 500
    meta[:, 5] = 300
    meta[0, 2] = 120                 # one logistic row so the branch runs
    params = SfTargetParams(
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.025,
        sf_wdl_cp_draw_width=60.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any overflow RuntimeWarning fails
        got, ok = rebuild_sf_wdl_batch(meta, params)
    want = np.zeros((8, 3), np.float32)
    want_ok = np.zeros(8, bool)
    for i in range(8):
        r = rebuild_sf_wdl(meta[i], params)
        if r is not None:
            want[i], want_ok[i] = r, True
    np.testing.assert_array_equal(ok, want_ok)
    np.testing.assert_array_equal(got[ok], want[ok])


def test_batch_rebuild_bitwise_across_randomized_params():
    """Bitwise equality against the scalar reference across RANDOMIZED param
    configs — including slopes past the old exp-overflow threshold — over the
    adversarial generator (which mixes cp+mate+native-simultaneous rows).
    This is the standing parity gate any future efficiency edit to the
    vectorized path must pass; the fixture params elsewhere in this file are
    points, this is the sweep."""
    width = 512
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        params = SfTargetParams(
            sf_policy_temp=float(rng.choice([0.006, 0.012, 0.05, 0.25, 0.5])),
            sf_policy_label_smooth=float(rng.choice([0.0, 0.01, 0.05])),
            sf_wdl_use_cp_logistic=bool(seed % 2),
            sf_wdl_cp_slope=float(rng.choice([0.006, 0.010, 0.025])),
            sf_wdl_cp_draw_width=float(rng.choice([60.0, 120.0])),
        )
        raw, legal = _adversarial_batch(rng, n=32, width=width)
        want, want_ok = _scalar_reference(raw, legal, width, params)
        got, got_ok = rebuild_sf_policy_targets_batch(
            raw, legal_dense=legal, policy_size=width, params=params,
        )
        np.testing.assert_array_equal(got_ok, want_ok, err_msg=f"seed {seed}")
        np.testing.assert_array_equal(
            got[want_ok], want[want_ok], err_msg=f"seed {seed}",
        )


@pytest.mark.parametrize("use_logistic", [False, True])
def test_torch_sparse_ce_row_scores_match_numpy_batch_row_scores(use_logistic):
    """`_batch_row_scores` (numpy, dense rebuild) and `sparse_sf_ce._row_scores`
    (torch, sparse CE) implement the same MultiPV row-score semantics from the
    same params object with NO cross-pin until now — a drift would silently
    split the dense and sparse trainings of the same loss. Pin them.

    Scoreable masks must agree EXACTLY. Scores agree within fp32 assembly
    noise: the torch leg computes in float32 by design (it feeds an fp32 CE;
    see the sparse_sf_ce module docstring), the numpy leg in float64. Measured
    max |Δ| over 20 seeds of this generator: 3.0e-8 native, 7.5e-8 logistic;
    atol=1e-6 keeps >10x margin above that noise while any semantic drift —
    swapped slope/draw_width, dropped mate precedence, permille-vs-fraction
    scale — moves scores by >= 1e-3."""
    from chess_anti_engine.train.sparse_sf_ce import _row_scores
    from chess_anti_engine.train.target_builder import _batch_row_scores

    for seed in range(6):
        rng = np.random.default_rng(300 + seed)
        raw, _legal = _adversarial_batch(rng, n=48, width=512)
        params = SfTargetParams(
            sf_wdl_use_cp_logistic=use_logistic,
            sf_wdl_cp_slope=float(rng.choice([0.006, 0.010, 0.025])),
            sf_wdl_cp_draw_width=float(rng.choice([60.0, 120.0])),
        )
        np_scores, np_ok = _batch_row_scores(np.asarray(raw), params)
        t_scores, t_ok = _row_scores(
            torch.from_numpy(raw.astype(np.int64)), params=params,
        )
        np.testing.assert_array_equal(t_ok.numpy(), np_ok, err_msg=f"seed {seed}")
        np.testing.assert_allclose(
            t_scores.numpy().astype(np.float64)[np_ok], np_scores[np_ok],
            atol=1e-6, rtol=0.0, err_msg=f"seed {seed}",
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_rebuild_in_arrays_writeback_matches_astype(dtype):
    """The writeback relies on fancy-index assignment casting f32 to the
    stored dtype bit-identically to an explicit astype (which is what the
    code used to do, at the cost of a full-width temporary). Pin the
    equivalence in both stored dtypes and on BOTH writeback branches
    (every-row-ok fast path and the partial `[ok]` gather)."""
    rng = np.random.default_rng(5)
    width = 96
    n = 16
    params = SfTargetParams(sf_policy_temp=0.012, sf_policy_label_smooth=0.05)

    raw_partial, legal = _adversarial_batch(rng, n=n, width=width)
    raw_all = np.array(raw_partial, copy=True)
    raw_all[:, 0] = (3, 40, 0, 700, 200)     # force >=1 scoreable row everywhere
    for raw in (raw_partial, raw_all):
        arrs = {
            "sf_policy_target": rng.random((n, width)).astype(dtype),
            "sf_legal_mask": legal,
            "sf_multipv_raw": raw,
            "has_sf_multipv_raw": np.ones(n, np.uint8),
        }
        stored = np.array(arrs["sf_policy_target"], copy=True)
        out, _cov = rebuild_sf_targets_in_arrays(arrs, params=params)
        want, want_ok = _scalar_reference(raw, legal, width, params)
        expect = np.array(stored, copy=True)
        expect[want_ok] = want[want_ok].astype(dtype)   # the explicit-astype form
        assert out["sf_policy_target"].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(out["sf_policy_target"], expect)
