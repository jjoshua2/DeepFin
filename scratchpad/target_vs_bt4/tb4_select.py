"""tb4_select.py -- row selection + board reconstruction + alignment controls.

Stage 1 of the target-vs-BT4 probe. NO neural net runs here. Read-only, CPU-only.

Emits `tb4_rows.npz` holding, per selected row:
  fen (reconstructed), target argmax move, SF best move, listedness, top-1 mass,
  n_legal, sf cp regret of the target argmax, plus the foreign/shuffled control
  move and a random legal control move.

Controls implemented here:
  A1  decoded-board legal set == stored `legal_mask` (exact, per row)
  A2  re-encode round trip on planes 0..11 and 104..111
  A3  cross-row legal-mask control (target mass under a permuted mask)
"""
from __future__ import annotations

import json
import os

import chess
import numpy as np
import zarr

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.moves import (
    legal_move_mask_for_encoding,
    move_to_index_for_encoding,
)

REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"
ERA_E = "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"
N_SHARDS = 16
N_TARGET = 2000
CAP = 1000.0  # SF_OWN_REGRET_CAP_CP
SEED = 20260815
OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"

_INV: dict[bool, dict[tuple[int, int], int]] = {}


def inverse_coord_map(turn: chess.Color) -> dict[tuple[int, int], int]:
    """(plane_row, plane_col) -> real square, derived empirically per side."""
    if turn in _INV:
        return _INV[turn]
    inv: dict[tuple[int, int], int] = {}
    for sq in chess.SQUARES:
        b = chess.Board(None)
        b.turn = turn
        b.set_piece_at(sq, chess.Piece(chess.PAWN, turn))
        x = encode_position(
            b, add_features=False,
            input_history_encoding=LC0_HISTORY_ROOT_LEGACY_META,
        )
        coords = np.argwhere(x[0] > 0.5)
        assert coords.shape == (1, 2), (sq, coords)
        inv[(int(coords[0, 0]), int(coords[0, 1]))] = sq
    _INV[turn] = inv
    return inv


def decode_board(x: np.ndarray) -> chess.Board:
    """Rebuild a chess.Board from one stored `lc0_root_legacy_meta` row.

    Pieces from planes 0..11 (us 0-5, them 6-11, PAWN..KING), side from 108,
    castling from 104..107 in `_write_metadata_planes_root` order
    (us-Q, us-K, them-Q, them-K), rule50 from 109 (legacy scale: /100),
    EP FILE from 110 (the `legacy_meta` extra plane).
    """
    x = np.asarray(x, dtype=np.float32)
    turn = chess.BLACK if float(x[108, 0, 0]) > 0.5 else chess.WHITE
    inv = inverse_coord_map(turn)
    board = chess.Board(None)
    board.turn = turn
    for base, color in ((0, turn), (6, not turn)):
        for off, piece_type in enumerate(chess.PIECE_TYPES):
            for r, c in np.argwhere(x[base + off] > 0.5):
                board.set_piece_at(inv[(int(r), int(c))], chess.Piece(piece_type, color))

    rights = 0
    us, them = turn, not turn
    flags = [float(x[i, 0, 0]) > 0.5 for i in range(104, 108)]  # usQ usK themQ themK
    for color, q_side, k_side in ((us, flags[0], flags[1]), (them, flags[2], flags[3])):
        if q_side:
            rights |= chess.BB_A1 if color == chess.WHITE else chess.BB_A8
        if k_side:
            rights |= chess.BB_H1 if color == chess.WHITE else chess.BB_H8
    board.castling_rights = rights

    # EP: plane 110 marks the FILE; the rank is fixed in the oriented frame
    # (index 5 = the square behind "their" just-double-pushed pawn).
    cols = np.flatnonzero(x[110].max(axis=0) > 0.5)
    ep = None
    if cols.size == 1:
        ep = chess.square(int(cols[0]), 5 if turn == chess.WHITE else 2)
    board.ep_square = ep
    board.halfmove_clock = int(round(float(x[109, 0, 0]) * 100.0))
    board.fullmove_number = 1
    return board


def main() -> None:
    sd = os.path.join(REPLAY, ERA_E, "replay_shards")
    names = sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[-N_SHARDS:]
    acc: dict[str, list[np.ndarray]] = {}
    attrs = None
    for nm in names:
        z = zarr.open(os.path.join(sd, nm), mode="r")
        if attrs is None:
            attrs = dict(z.attrs)
        for k in ("x", "policy_target", "legal_mask", "has_policy",
                  "sf_p0_regret", "has_sf_p0_regret", "game_id", "ply_index"):
            acc.setdefault(k, []).append(np.asarray(z[k][:]))
    D = {k: np.concatenate(v, axis=0) for k, v in acc.items()}

    rep: dict[str, object] = {
        "shards": names,
        "input_history_encoding": (attrs or {}).get("input_history_encoding"),
        "policy_encoding": (attrs or {}).get("policy_encoding"),
        "n_rows_read": int(D["policy_target"].shape[0]),
    }

    hp = D["has_policy"].astype(bool)
    hs = D["has_sf_p0_regret"].astype(bool) & hp
    idx_all = np.flatnonzero(hs)
    rep["n_has_policy"] = int(hp.sum())
    rep["n_sf_labelled"] = int(idx_all.size)

    rng = np.random.default_rng(SEED)
    n = min(N_TARGET, idx_all.size)
    sel = np.sort(rng.choice(idx_all, size=n, replace=False))
    rep["n_sampled"] = int(n)

    pol = D["policy_target"][sel].astype(np.float64)
    lm = D["legal_mask"][sel].astype(bool)
    R = D["sf_p0_regret"][sel].astype(np.float64)
    X = D["x"][sel]

    P = pol * lm
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    fill = np.array([R[i][~lm[i]][0] if (~lm[i]).any() else np.nan
                     for i in range(R.shape[0])])
    ok_fill = np.isfinite(fill)
    covered = lm & (R != fill[:, None])
    sf_best_i = np.argmin(np.where(lm, R, np.inf), axis=1)
    tgt_best_i = np.argmax(P, axis=1)
    top1 = P[np.arange(n), tgt_best_i]
    n_legal = lm.sum(axis=1)

    # --- A3: cross-row legal-mask control ---------------------------------
    perm_mask = rng.permutation(n)
    rep["A3_target_mass_under_own_mask"] = float((pol * lm).sum(axis=1).mean())
    rep["A3_target_mass_under_permuted_mask"] = float(
        (pol * lm[perm_mask]).sum(axis=1).mean())

    # --- shuffle control move: argmax of a FOREIGN row's target, restricted
    #     to THIS row's legal set -------------------------------------------
    perm_tgt = rng.permutation(n)
    foreign_i = np.argmax(np.where(lm, pol[perm_tgt], -1.0), axis=1)

    # --- decode + A1 / A2 --------------------------------------------------
    fens: list[str] = []
    keep: list[int] = []
    a1_fail: list[dict] = []
    a2_piece_ok = a2_meta_ok = a2_rep_ok = 0
    rand_i = np.empty(n, dtype=np.int64)
    for i in range(n):
        board = decode_board(X[i])
        try:
            mask = legal_move_mask_for_encoding(board, policy_encoding="lc0_1858")
        except Exception as exc:  # noqa: BLE001
            a1_fail.append({"row": i, "err": repr(exc)})
            continue
        mask = np.asarray(mask).astype(bool).reshape(-1)
        if not np.array_equal(mask, lm[i]):
            a1_fail.append({
                "row": i, "fen": board.fen(),
                "n_decoded": int(mask.sum()), "n_stored": int(lm[i].sum()),
                "extra": int((mask & ~lm[i]).sum()), "missing": int((~mask & lm[i]).sum()),
            })
            continue
        # A2: round-trip planes
        re_x = encode_position(board, add_features=False,
                               input_history_encoding=LC0_HISTORY_ROOT_LEGACY_META)
        stored = np.asarray(X[i], dtype=np.float32)
        a2_piece_ok += int(np.array_equal(re_x[0:12] > 0.5, stored[0:12] > 0.5))
        a2_meta_ok += int(np.allclose(re_x[104:112], stored[104:112], atol=2e-3))
        a2_rep_ok += int(np.array_equal(re_x[12] > 0.5, stored[12] > 0.5))
        legal = list(board.legal_moves)
        rand_i[i] = move_to_index_for_encoding(
            legal[int(rng.integers(len(legal)))], board, policy_encoding="lc0_1858")
        fens.append(board.fen())
        keep.append(i)

    keep_a = np.array(keep, dtype=np.int64)
    rep["A1_n_checked"] = int(n)
    rep["A1_n_exact"] = int(keep_a.size)
    rep["A1_frac_exact"] = float(keep_a.size / n)
    rep["A1_failures_sample"] = a1_fail[:20]
    rep["A1_n_failures"] = len(a1_fail)
    rep["A2_frac_piece_planes_exact"] = float(a2_piece_ok / n)
    rep["A2_frac_meta_planes_exact"] = float(a2_meta_ok / n)
    rep["A2_frac_slot0_repetition_plane_match"] = float(a2_rep_ok / n)

    k = keep_a
    agree = (sf_best_i == tgt_best_i)
    rep["replication_agree_target_argmax_eq_sf_best"] = float(agree[k].mean())
    rep["replication_mean_top1_mass"] = float(top1[k].mean())
    rep["replication_mean_n_legal"] = float(n_legal[k].mean())
    tb_cov = covered[np.arange(n), tgt_best_i]
    rep["replication_frac_target_argmax_NOT_listed"] = float((~tb_cov[k]).mean())
    rep["S1_foreign_argmax_eq_sf_best"] = float((foreign_i == sf_best_i)[k].mean())
    rep["S1_chance_mean_inv_nlegal"] = float((1.0 / n_legal[k]).mean())
    rep["fill_value_is_constant_frac"] = float(ok_fill[k].mean())
    rep["fill_value_mean_cp"] = float(np.nanmean(fill[k]) * CAP)

    np.savez_compressed(
        os.path.join(OUT, "tb4_rows.npz"),
        row_index=sel[k],
        fens=np.array(fens, dtype=object),
        tgt_idx=tgt_best_i[k], sf_idx=sf_best_i[k],
        foreign_idx=foreign_i[k], rand_idx=rand_i[k],
        top1_mass=top1[k], n_legal=n_legal[k],
        tgt_listed=tb_cov[k],
        sf_cp_regret_tgt=(R[np.arange(n), tgt_best_i] * CAP)[k],
        agree=agree[k],
        allow_pickle=True,
    )
    with open(os.path.join(OUT, "tb4_select_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print(json.dumps({kk: vv for kk, vv in rep.items()
                      if kk not in ("A1_failures_sample", "shards")}, indent=2))


if __name__ == "__main__":
    main()
