"""Additive provenance must survive real storage transforms without blessing corruption."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import POLICY_SIZE, uci_to_policy_index, policy_vector_to_encoding
from chess_anti_engine.replay.augment import mirror_sample, maybe_mirror_batch_arrays
from chess_anti_engine.replay.shard import arrays_to_samples, samples_to_arrays, sf_eval_pv_orphan_flags
from chess_anti_engine.replay.sf_eval_provenance import (
    SF_EVAL_WITNESS_LEN, make_sf_eval_witness, remap_sf_eval_witness,
    valid_sf_eval_witness, sf_eval_witness_mismatch,
)
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.selfplay.finalize import _build_replay_samples
from chess_anti_engine.selfplay.resume import (
    _game_payload, _record_fields, _rebuild_record,
    _pending_original_payload, _pending_original_result, ResumeStateError,
)
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    _stamp_sparse_sf_labels, _sf_result_wdl_for_record, _sf_eval_pv_orphaned,
    _collect_sf_pv_candidates, _build_sf_policy_target,
)
from chess_anti_engine.stockfish.uci import _SearchInfoAccumulator
from chess_anti_engine.train.target_builder import SfTargetParams, rebuild_sf_policy_target
from tests.test_stockfish_multipv_cast_compat import _sample


def _result():
    acc = _SearchInfoAccumulator()
    lines: list[str] = [
        "info multipv 1 score cp 40 pv e2e4", "info multipv 2 score cp 10 pv d2d4",
        "info multipv 1 score cp 200 lowerbound pv d2d4", "info multipv 2 score cp 35 pv e2e4",
    ]
    for line in lines:
        acc.consume(line.split())
    return acc.result("d2d4")


def _record_state():
    board = chess.Board()
    cb = CBoard.from_board(board)
    legal = np.asarray(cb.legal_move_indices(), dtype=np.int64)
    probs = np.zeros(POLICY_SIZE, dtype=np.float32)
    probs[legal] = 1.0 / len(legal)
    mask = (probs > 0).astype(np.uint8)
    game = GameConfig(input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    rec = _NetRecord(
        x=encode_position(board, input_history_encoding=game.input_history_encoding, input_extra_features="v1"),
        policy_probs=probs, net_wdl_est=np.array([.2, .5, .3], dtype=np.float32),
        search_wdl_est=np.array([.3, .4, .3], dtype=np.float32), pov_color=True,
        ply_index=0, has_policy=True, priority=1.0, sample_weight=1.0,
        keep_prob=1.0, legal_mask=mask, move_offset=0, pos_hash=int(cb.zobrist_hash),
    )
    res = _result()
    _stamp_sparse_sf_labels(rec, res, turn=True, legal_set=set(legal.tolist()))
    rec.sf_wdl = _sf_result_wdl_for_record(res, sf_wdl_use_cp_logistic=True,
                                         sf_wdl_cp_slope=.010, sf_wdl_cp_draw_width=60.)
    state = SimpleNamespace(
        samples_per_game=[[rec]], starting_boards=[board], cboards=[cb],
        move_idx_history=[[]], net_color_arr=[True], selfplay_arr=[True],
        opening_source_arr=["startpos"], sf_refute_opp_plies_left=[0], tb_adj_roll_arr=[0],
        consecutive_low_winrate=[0], last_net_full=[True], force_full_next=[False],
        sf_label_escalations=[0], has_c_ply=True, pending_sf_labels=[],
        game=game, rng=np.random.default_rng(0),
    )
    return rec, cast(Any, state), cb


@pytest.mark.parametrize(("cp", "mate"), [(40, None), (None, 3), (None, 0), (None, -2)])
def test_capture_and_pending_result_roundtrip(cp: int | None, mate: int | None):
    acc = _SearchInfoAccumulator()
    score = f"cp {cp}" if cp is not None else f"mate {mate}"
    acc.consume(f"info multipv 1 score {score} pv e2e4".split())
    res = acc.result("e2e4")
    assert res.eval_witness is not None
    restored = _pending_original_result(_pending_original_payload(res))
    assert restored.eval_witness == res.eval_witness
    assert (restored.cp, restored.mate) == (cp, mate)
    raw = np.array([[877, -32768 if cp is None else cp, mate or 0, -1, -1]], dtype=np.int16)
    witness = make_sf_eval_witness(877, cp, mate, raw)
    assert valid_sf_eval_witness(witness, raw)


def test_scored_no_pv_invalidates_and_pv_only_cannot_restore():
    acc = _SearchInfoAccumulator()
    acc.consume(["info", "score", "cp", "40", "pv", "e2e4"])
    acc.consume(["info", "score", "cp", "50"])
    assert acc.result("e2e4").eval_witness is None
    acc.consume(["info", "pv", "d2d4"])
    assert acc.result("d2d4").eval_witness is None
    acc.consume(["info", "score", "mate", "2", "pv", "d2d4"])
    assert acc.result("d2d4").eval_witness is not None
    assert _SearchInfoAccumulator().result("d2d4").eval_witness is None


def test_pending_legacy_and_corrupt_witness():
    payload = _pending_original_payload(_result())
    legacy = dict(payload)
    legacy.pop("eval_witness")
    assert _pending_original_result(legacy).eval_witness is None
    payload["eval_witness"]["cp"] += 1
    with pytest.raises(ResumeStateError, match="disagrees"):
        _pending_original_result(payload)
    payload = _pending_original_payload(_result())
    payload["eval_witness"]["move_uci"] = "g1f3"
    with pytest.raises(ResumeStateError, match="integrity"):
        _pending_original_result(payload)


@pytest.mark.parametrize("field", ["cp", "move_uci", "mate", "order"])
def test_pending_policy_corruption_is_rejected_before_attachment(field: str):
    payload = _pending_original_payload(_result())
    if field == "cp":
        payload["pvs"][0]["cp"] += 1
    elif field == "move_uci":
        payload["pvs"][0][field] = "g1f3"
    elif field == "mate":
        payload["pvs"][0][field] = 2
    else:
        payload["pvs"].reverse()
    with pytest.raises(ResumeStateError, match="integrity"):
        _pending_original_result(payload)


@pytest.mark.parametrize("bad_flag", [2, 256, .5])
def test_sample_presence_flag_cannot_be_normalized(bad_flag: float):
    arrays = samples_to_arrays([_sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)])
    arrays["has_sf_eval_witness"] = np.array([bad_flag])
    with pytest.raises(ValueError, match="zero/one"):
        arrays_to_samples(arrays)


def test_sample_present_witness_cannot_lose_missing_flag():
    arrays = samples_to_arrays([_sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)])
    arrays.pop("has_sf_eval_witness")
    with pytest.raises(ValueError, match="presence flag"):
        arrays_to_samples(arrays)


@pytest.mark.parametrize("kind", ["fractional", "overflow"])
def test_sample_scalar_corruption_cannot_be_narrowed_healthy(kind: str):
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    assert sample.sf_label_meta is not None
    sample.sf_label_meta = sample.sf_label_meta.astype(np.float64 if kind == "fractional" else np.int64)
    sample.sf_label_meta[2] += .25 if kind == "fractional" else 2**32
    assert sf_eval_pv_orphan_flags(samples_to_arrays([sample]))[0][0] == 1


@pytest.mark.parametrize("bad_flag", [2, 256, .5])
def test_record_presence_flag_cannot_be_normalized(bad_flag: float):
    _, state, _ = _record_state()
    arrays = _game_payload(state, 0)["arrays"]
    arrays["sf_eval_witness_present"] = np.array([bad_flag])
    with pytest.raises(ResumeStateError, match="presence"):
        _record_fields(arrays, 1)


@pytest.mark.parametrize(("field", "bits"), [("sf_multipv_raw_values", 16), ("sf_label_meta", 32)])
@pytest.mark.parametrize("kind", ["fractional", "overflow"])
def test_record_integrity_inputs_validate_before_narrowing(field: str, bits: int, kind: str):
    _, state, _ = _record_state()
    arrays = _game_payload(state, 0)["arrays"]
    arrays[field] = arrays[field].astype(np.float64 if kind == "fractional" else np.int64)
    arrays[field][0, 1 if bits == 16 else 2] += .25 if kind == "fractional" else 2**bits
    with pytest.raises(ResumeStateError, match="witness input"):
        _record_fields(arrays, 1)


@pytest.mark.parametrize("field", range(SF_EVAL_WITNESS_LEN))
def test_every_witness_field_corruption_rejects_before_and_after_remap(field: int):
    raw = np.array([[877, 35, 0, -1, -1]], dtype=np.int16)
    witness = make_sf_eval_witness(877, 40, None, raw)
    witness[field] += 1
    meta = np.array([100, 6, 40, 0, -1, -1], dtype=np.int32)
    assert sf_eval_witness_mismatch(witness, meta, raw)
    remapped = remap_sf_eval_witness(witness, lambda action: action + 1, raw, raw)
    assert remapped is not None
    assert not valid_sf_eval_witness(remapped, raw)


@pytest.mark.parametrize("column", range(5))
def test_policy_row_corruption_rejects_even_with_valid_root_tuple(column: int):
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    arrs = samples_to_arrays([sample])
    arrs["sf_multipv_raw"][0, 0, column] += 1
    flags, checked = sf_eval_pv_orphan_flags(arrs)
    assert (flags[0], checked[0]) == (1, 1)


@pytest.mark.parametrize("kind", ["float", "overflow", "shape", "negative_action", "version"])
def test_malformed_evidence_is_not_repaired_by_sample_serialization(kind: str):
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    assert sample.sf_eval_witness is not None
    witness = sample.sf_eval_witness.copy()
    if kind == "float":
        witness = witness.astype(np.float64)
        witness[2] += .25
    elif kind == "overflow":
        witness = witness.astype(np.int64)
        witness[2] += 2**32
    elif kind == "shape":
        witness = witness[:-1]
    elif kind == "negative_action":
        witness[1] = -1
    else:
        witness[0] = 99
    sample.sf_eval_witness = witness
    arrs = samples_to_arrays([sample])
    assert sf_eval_pv_orphan_flags(arrs)[0][0] == 1


def test_legacy_mixed_rows_and_absent_flag_keep_old_predicate():
    a = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    b = _sample(chess.Board(), 1, bound="lowerbound", bad_identity=False)
    b.sf_eval_witness = None
    arrs = samples_to_arrays([a, b])
    np.testing.assert_array_equal(sf_eval_pv_orphan_flags(arrs)[0], [0, 1])
    back = arrays_to_samples(arrs)
    assert back[0].sf_eval_witness is not None
    assert back[1].sf_eval_witness is None
    arrs.pop("sf_eval_witness")
    arrs["has_sf_eval_witness"][:] = 0
    np.testing.assert_array_equal(sf_eval_pv_orphan_flags(arrs)[0], [1, 1])


@pytest.mark.parametrize("damage", ["flag", "shape", "missing_value", "float_meta", "encoding_width"])
def test_present_malformed_provenance_does_not_use_legacy_fallback(damage: str):
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    arrs = samples_to_arrays([sample])
    if damage == "flag":
        arrs["has_sf_eval_witness"][0] = 2
    elif damage == "shape":
        arrs["sf_eval_witness"] = arrs["sf_eval_witness"][:, :-1]
    elif damage == "missing_value":
        arrs.pop("sf_eval_witness")
    elif damage == "float_meta":
        arrs["sf_label_meta"] = arrs["sf_label_meta"].astype(np.float64)
        arrs["sf_label_meta"][0, 2] += .25
    else:
        arrs["policy_target"] = np.zeros((1, 4672), dtype=np.float32)
    assert sf_eval_pv_orphan_flags(arrs)[0][0] == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_mirror_roundtrips_keep_validity_and_do_not_bless_raw_corruption(corrupt: bool):
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    assert sample.sf_multipv_raw is not None
    if corrupt:
        sample.sf_multipv_raw[0, 1] += 1
    mirrored = mirror_sample(sample)
    assert sf_eval_pv_orphan_flags(samples_to_arrays([mirrored]))[0][0] == int(corrupt)
    arrs = samples_to_arrays([sample])
    out = maybe_mirror_batch_arrays(arrs, rng=np.random.default_rng(0), prob=1.)
    assert sf_eval_pv_orphan_flags(out)[0][0] == int(corrupt)
    if not corrupt:
        twice = mirror_sample(mirrored)
        np.testing.assert_array_equal(twice.sf_eval_witness, sample.sf_eval_witness)


def test_full_policy_space_mirror_and_check():
    rec, _, _ = _record_state()
    assert rec.sf_multipv_raw is not None
    assert rec.sf_eval_witness is not None
    assert rec.sf_label_meta is not None
    arrs = {"x": rec.x[None], "policy_target": rec.policy_probs[None],
            "sf_multipv_raw": rec.sf_multipv_raw[None], "sf_label_meta": rec.sf_label_meta[None],
            "sf_eval_witness": rec.sf_eval_witness[None],
            "has_sf_wdl": np.ones(1), "has_sf_multipv_raw": np.ones(1),
            "has_sf_label_meta": np.ones(1), "has_sf_eval_witness": np.ones(1)}
    assert sf_eval_pv_orphan_flags(arrs)[0][0] == 0
    out = maybe_mirror_batch_arrays(arrs, rng=np.random.default_rng(0), prob=1.)
    assert sf_eval_pv_orphan_flags(out)[0][0] == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_record_npz_resume_and_actual_native_finalization(tmp_path: Path, legacy: bool):
    rec, state, cb = _record_state()
    payload = _game_payload(state, 0)
    arrays = payload["arrays"]
    if legacy:
        arrays.pop("sf_eval_witness")
        arrays.pop("sf_eval_witness_present")
    path = tmp_path / "synthetic-resume.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as stored:
        fields = _record_fields(stored, 1)[0]
    restored = _rebuild_record(state, cb, fields, want_lc0_root_alt=False, want_relations=False)
    if legacy:
        assert restored.sf_eval_witness is None
    else:
        np.testing.assert_array_equal(restored.sf_eval_witness, rec.sf_eval_witness)
    samples = _build_replay_samples(state, 0, [restored], result="1/2-1/2",
        tb_policy_overrides={}, vol_targets=[None], sf_vol_targets=[None],
        total_plies_played=1, ply_to_index={0: 0})
    assert len(samples) == 1
    assert sf_eval_pv_orphan_flags(samples_to_arrays(samples))[0][0] == int(legacy)
    assert _sf_eval_pv_orphaned(restored) == legacy


def test_corruption_before_native_finalization_remains_invalid():
    rec, state, _ = _record_state()
    assert rec.sf_eval_witness is not None
    rec.sf_eval_witness = rec.sf_eval_witness.astype(np.float64)
    rec.sf_eval_witness[2] += .25
    samples = _build_replay_samples(state, 0, [rec], result="1/2-1/2",
        tb_policy_overrides={}, vol_targets=[None], sf_vol_targets=[None],
        total_plies_played=1, ply_to_index={0: 0})
    assert sf_eval_pv_orphan_flags(samples_to_arrays(samples))[0][0] == 1


def test_latest_policy_and_replay_rebuild_match_with_separate_root_witness():
    rec, _, _ = _record_state()
    sample = _sample(chess.Board(), 0, bound="lowerbound", bad_identity=False)
    res = _result()
    legal = np.array([uci_to_policy_index(move.uci(), True) for move in chess.Board().legal_moves])
    ids, scores = _collect_sf_pv_candidates(res, _turn=True, legal_set=set(legal.tolist()),
                                           sf_policy_score_mode="cp")
    assert scores == [10., 35.]
    live = _build_sf_policy_target(ids, scores, legal_indices=legal,
                                   sf_policy_temp=100., sf_policy_label_smooth=0.)
    params = SfTargetParams(sf_policy_score_mode="cp", sf_policy_cp_temp=100., sf_policy_label_smooth=0.)
    from chess_anti_engine.moves.encode import compact_policy_index
    assert sample.sf_multipv_raw is not None
    rebuilt = rebuild_sf_policy_target(sample.sf_multipv_raw,
        legal_indices=np.array([compact_policy_index(int(a)) for a in legal]), policy_size=1858, params=params)
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt, policy_vector_to_encoding(live), atol=1e-7)
    assert rec.sf_label_meta is not None
    assert rec.sf_label_meta[2] == 40
