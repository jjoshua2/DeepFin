"""Synthetic UCI evidence survives real shard I/O and CAST selection.

No engine, training data, or live corpus is used. The actual serializer,
Zarr writer/reader, orphan predicate, and CAST collector are exercised.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import compact_policy_index, uci_to_policy_index
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
    sf_eval_pv_orphan_flags,
)
from chess_anti_engine.selfplay.stockfish_turn import (
    _collect_sf_eval_witness,
    _collect_sf_label_meta,
    _collect_sparse_pv_rows,
    _sf_result_wdl_for_record,
)
from chess_anti_engine.selfplay.finalize import (
    _prepare_sf_eval_witness_for_finalize,
    _prepare_sf_multipv_for_finalize,
)
from chess_anti_engine.stockfish.uci import _SearchInfoAccumulator
from tests.script_loading import load_script_module


CAST = load_script_module("cast_probe.py")
ENCODING = "lc0_root_legacy_meta"


def _sample(board: chess.Board, ply: int, *, bound: str, bad_identity: bool) -> ReplaySample:
    a, b = ("e2e4", "d2d4") if board.turn else ("e7e5", "d7d5")
    if bad_identity:
        a = "a1a8" if board.turn else "a8a1"
    acc = _SearchInfoAccumulator()
    lines: tuple[str, ...] = (
        f"info depth 5 multipv 1 score cp 40 pv {a}",
        f"info depth 5 multipv 2 score cp 10 pv {b}",
        f"info depth 6 multipv 1 score cp 200 {bound} pv {b}",
        f"info depth 6 multipv 2 score cp 35 pv {a}",
    )
    for text in lines:
        parts: list[str] = list(text.split())
        acc.consume(parts)
    result = acc.result(b)
    assert result.cp == 40
    legal = {uci_to_policy_index(move.uci(), bool(board.turn)) for move in board.legal_moves}
    raw = _collect_sparse_pv_rows(result, turn=bool(board.turn), legal_set=legal)
    assert raw is not None
    witness = _collect_sf_eval_witness(result, turn=bool(board.turn), legal_set=legal, raw=raw)
    padded, _ = _prepare_sf_multipv_for_finalize(raw, policy_encoding="lc0_1858", want_regret=False)
    assert padded is not None
    record = SimpleNamespace(sf_eval_witness=witness, sf_multipv_raw=raw)
    stored_witness = _prepare_sf_eval_witness_for_finalize(record, padded, policy_encoding="lc0_1858")
    if not bad_identity:
        assert [(pv.move_uci, pv.cp) for pv in result.pvs] == [(b, 10), (a, 35)]
        np.testing.assert_array_equal(padded[:2, 1], [10, 35])
    wdl = _sf_result_wdl_for_record(result, sf_wdl_use_cp_logistic=True,
                                    sf_wdl_cp_slope=0.010, sf_wdl_cp_draw_width=60.0)
    assert wdl is not None
    policy = np.zeros(1858, dtype=np.float32)
    for action in legal:
        policy[compact_policy_index(action)] = 1.0 / len(legal)
    return ReplaySample(
        x=encode_position(board, input_history_encoding=ENCODING, input_extra_features="v1"),
        policy_target=policy, wdl_target=1, sf_wdl=wdl,
        sf_multipv_raw=padded, sf_label_meta=_collect_sf_label_meta(result),
        sf_eval_witness=stored_witness,
        game_id=17, ply_index=ply, input_history_encoding=ENCODING,
    )


def _stored_pair(tmp_path: Path, *, bound: str, corruption: str):
    parent = chess.Board()
    child = parent.copy()
    child.push_uci("e2e4")
    samples = [_sample(board, i, bound=bound, bad_identity=corruption == "identity" and i == 1)
               for i, board in enumerate((parent, child))]
    arrs = samples_to_arrays(samples)
    if corruption == "value":
        arrs["sf_label_meta"][1, 2] += 1
    elif corruption == "provenance":
        arrs["sf_eval_witness"][1, 1] += 1
    elif corruption == "raw_policy":
        arrs["sf_multipv_raw"][1, 0, 1] += 1
    path = save_local_shard_arrays(tmp_path / "synthetic.zarr", arrs=arrs)
    loaded, _ = load_shard_arrays(path)
    flags, checked = sf_eval_pv_orphan_flags(loaded)
    scan = CAST.new_scan("synthetic-serialization", [path])
    CAST.collect([path], scan, np.random.default_rng(0))
    return loaded, flags, checked, scan


@pytest.mark.parametrize("bound", ["lowerbound", "upperbound"])
def test_benign_rerank_survives_serialization_and_cast(tmp_path: Path, bound: str):
    loaded, flags, checked, scan = _stored_pair(tmp_path, bound=bound, corruption="none")
    np.testing.assert_array_equal(loaded["sf_label_meta"][:, 2], [40, 40])
    np.testing.assert_array_equal(checked, [1, 1])
    np.testing.assert_array_equal(flags, [0, 0])
    assert scan["cast_pairs"] == 1
    assert scan["desync_rows_rejected"] == 0


@pytest.mark.parametrize("corruption", ["value", "identity", "provenance", "raw_policy"])
def test_corrupt_value_or_illegal_root_still_rejects_cast_pair(tmp_path: Path, corruption: str):
    _, flags, checked, scan = _stored_pair(tmp_path, bound="lowerbound", corruption=corruption)
    assert checked[1] == 1
    assert flags[1] == 1
    assert scan["cast_pairs"] == 0
    assert scan["desync_rows_rejected"] == 1
