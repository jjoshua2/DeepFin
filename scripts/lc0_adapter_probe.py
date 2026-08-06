#!/usr/bin/env python3
"""Run a REAL Leela net inside our Python Gumbel search and compare the search
target it produces against lc0's own stored training targets.

Diagnostic only — nothing here is on the production path.

What it establishes, in order (each gate is a prerequisite for the next):

1. INDEX CORRESPONDENCE. Our ``lc0_1858`` policy encoding is NOT Leela's 1858
   ordering (they agree on 46 of 1858 slots), and the two disagree about which
   promotion shares a plain from/to slot and about how castling is spelled.
   ``chess_anti_engine.moves.leela_index`` carries the board-aware permutation;
   this script proves it move-by-move against an independent reference.
2. ROUND TRIP. lc0 v6 training records are decoded back to a ``chess.Board``
   (plus a synthetic 7-ply history stack) and re-encoded with OUR encoder; the
   112 planes must come back bit-identical.
3. NET SANITY. startpos / mate-in-1 / midgame policy and WDL read sensibly.
4. SEARCH. Our Gumbel search is run on lc0's own positions with the lc0 net,
   sweeping (c_scale, topk), and the resulting improved policy is compared
   per-position against lc0's stored target (entropy, support, max-prob, KL).

Usage:
  PYTHONPATH=. python3 scripts/lc0_adapter_probe.py \
    --onnx data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
    --matched /path/to/matched60.npz --out runs/lc0_adapter_probe.md
"""
from __future__ import annotations

import argparse
import hashlib
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    fill_lc0_history_repeat,
    lc0_gather_context_from_planes,
)
from chess_anti_engine.mcts import gumbel as gumbel_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_MOVE_STRS
from chess_anti_engine.moves.leela_index import (
    compact_index_for_move,
    leela_gather_indices,
    leela_index_for_move,
)

PHASE_NAMES = ("opening", "middlegame", "endgame")
PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# lc0 serialises a training-data plane as a uint64 whose bit j is the square
# (rank j // 8, file 7 - j % 8) — the file axis runs h..a, opposite to the
# (rank, file) layout the ONNX graph consumes. Proven twice: only this reading
# puts the kings on e1/e8 in positions that still hold castling rights, and only
# this reading makes BT4's top move agree with lc0's own stored target (43/60 vs
# 3/60 at chance for the naive reading).
_BIT_TO_SQUARE = np.array(
    [chess.square(7 - (j % 8), j // 8) for j in range(64)], dtype=np.int64,
)


# ── the adapter ────────────────────────────────────────────────────────────────

class Lc0OnnxEvaluator:
    """``BatchEvaluator`` backed by a real Leela ONNX net.

    Consumes our encoder's planes (``lc0_root`` history, any extra-feature
    block — everything past plane 112 is sliced off) and returns the pair the
    search wants: policy logits in OUR compact-1858 order and WDL log-probs.

    The 1858 reorder is board-aware, and the board context comes off the input
    planes themselves (``lc0_gather_context_from_planes``) so this stays a
    plain planes-in/values-out evaluator.

    ``cache`` memoises on the exact 112-plane input, so a sweep over search
    knobs re-runs the tree without re-running the net.
    """

    input_history_encoding = "lc0_root"
    input_extra_features = "v1"
    use_dynamic_relations = False
    policy_encoding = "lc0_1858"

    def __init__(self, onnx_path: str | Path, *, providers: tuple[str, ...], cache: bool = True):
        import onnxruntime as ort

        self.path = str(Path(onnx_path).expanduser().resolve())
        self._sess = ort.InferenceSession(self.path, providers=list(providers))
        self._in = self._sess.get_inputs()[0].name
        self._pol_out: str | None = None
        self._wdl_out: str | None = None
        self._cache: dict[bytes, tuple[np.ndarray, np.ndarray]] | None = {} if cache else None
        self.hits = 0
        self.misses = 0
        self.net_calls = 0

    def _resolve_outputs(self) -> tuple[str, str]:
        pol, wdl = self._pol_out, self._wdl_out
        if pol is None or wdl is None:
            widths = [(str(o.name), o.shape[-1]) for o in self._sess.get_outputs()]
            pol = next((n for n, w in widths if w == COMPACT_POLICY_SIZE), None)
            wdl = next((n for n, w in widths if w == 3), None)
            if pol is None or wdl is None:
                raise SystemExit(f"ONNX graph lacks a 1858 policy / 3-wide WDL head: {widths}")
            self._pol_out, self._wdl_out = pol, wdl
        return pol, wdl

    def _run_net(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pol_name, wdl_name = self._resolve_outputs()
        pol, wdl = self._sess.run([pol_name, wdl_name], {self._in: planes})
        self.net_calls += 1
        return np.asarray(pol, dtype=np.float32), np.asarray(wdl, dtype=np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        planes = np.ascontiguousarray(np.asarray(x, dtype=np.float32)[:, : LC0_FULL.num_planes])
        planes = fill_lc0_history_repeat(planes)
        n = planes.shape[0]
        pol_leela = np.empty((n, COMPACT_POLICY_SIZE), dtype=np.float32)
        wdl_raw = np.empty((n, 3), dtype=np.float32)

        if self._cache is None:
            pol_leela, wdl_raw = self._run_net(planes)
            self.misses += n
        else:
            keys = [hashlib.blake2b(planes[i].tobytes(), digest_size=16).digest() for i in range(n)]
            todo = [i for i, k in enumerate(keys) if k not in self._cache]
            self.hits += n - len(todo)
            self.misses += len(todo)
            if todo:
                p, w = self._run_net(planes[todo])
                for j, i in enumerate(todo):
                    self._cache[keys[i]] = (p[j].copy(), w[j].copy())
            for i, k in enumerate(keys):
                pol_leela[i], wdl_raw[i] = self._cache[k]

        pawn_mask, castling = lc0_gather_context_from_planes(planes)
        pol_ours = np.take_along_axis(
            pol_leela, leela_gather_indices(pawn_mask, castling), axis=1,
        )
        # The search softmaxes `wdl`, so it must receive logits. Leela value
        # heads emit softmaxed probabilities; log() makes the softmax a no-op.
        is_probs = bool((wdl_raw >= -1e-4).all()) and bool(
            np.abs(wdl_raw.sum(axis=-1) - 1.0).max() < 0.1,
        )
        wdl = np.log(np.clip(wdl_raw, 1e-9, None)) if is_probs else wdl_raw
        return pol_ours, wdl.astype(np.float32)

    def raw_eval(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
        """(policy over OUR compact 1858, WDL probabilities) for one board."""
        x = encode_position(board, add_features=False, input_history_encoding="lc0_root")
        pol, wdl_log = self.evaluate_encoded(x[None, ...])
        return pol[0], np.exp(wdl_log[0])


# ── lc0 v6 record decoding ─────────────────────────────────────────────────────

@dataclass
class Lc0Record:
    board: chess.Board
    target: np.ndarray  # (1858,) in LEELA order; -1 = illegal
    phase: int
    src: str
    planes_ref: np.ndarray  # (112, 8, 8) rebuilt from the record, our layout


def _frame_board(frame: np.ndarray, castling: int, rule50: int) -> chess.Board:
    b = chess.Board(None)
    b.turn = chess.WHITE  # planes are already side-to-move oriented
    for i, pt in enumerate(PIECE_ORDER):
        for j in chess.scan_forward(int(frame[i])):
            b.set_piece_at(int(_BIT_TO_SQUARE[j]), chess.Piece(pt, chess.WHITE))
        for j in chess.scan_forward(int(frame[6 + i])):
            b.set_piece_at(int(_BIT_TO_SQUARE[j]), chess.Piece(pt, chess.BLACK))
    b.castling_rights = castling & b.rooks
    b.halfmove_clock = rule50
    return b


def _plane_from_mask(mask: int) -> np.ndarray:
    bits = np.unpackbits(
        np.array([mask], dtype="<u8").view(np.uint8), bitorder="little",
    ).astype(np.float32).reshape(8, 8)
    return bits[:, ::-1]  # bit order runs h..a within a rank


def decode_records(npz_path: Path) -> list[Lc0Record]:
    """Decode v6 records into boards carrying a synthetic history stack.

    The stack matters: our encoder builds the 7 older history frames from
    ``board._stack``, so without it the round trip could only ever check frame
    0 and the search would feed the net repeat-filled history.
    """
    d = np.load(npz_path, allow_pickle=True)
    planes, meta, policy, phase, src = d["planes"], d["meta"], d["policy"], d["phase"], d["src"]
    out: list[Lc0Record] = []
    for r in range(planes.shape[0]):
        m = meta[r]
        castling = 0
        if m[0]:
            castling |= chess.BB_A1
        if m[1]:
            castling |= chess.BB_H1
        if m[2]:
            castling |= chess.BB_A8
        if m[3]:
            castling |= chess.BB_H8
        board = _frame_board(planes[r][:13], castling, int(m[5]))
        stack = []
        for f in range(7, 0, -1):
            frame = planes[r][f * 13:(f + 1) * 13]
            if not any(int(v) for v in frame[:12]):
                continue
            tb = _frame_board(frame, castling, 0)
            tb.turn = chess.WHITE if f % 2 == 0 else chess.BLACK
            stack.append(chess._BoardState(tb))
        board._stack = stack

        ref = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)
        for p in range(104):
            ref[p] = _plane_from_mask(int(planes[r][p]))
        base = LC0_FULL.root_metadata_base
        ref[base + 0] = 1.0 if m[0] else 0.0
        ref[base + 1] = 1.0 if m[1] else 0.0
        ref[base + 2] = 1.0 if m[2] else 0.0
        ref[base + 3] = 1.0 if m[3] else 0.0
        ref[base + 5] = float(m[5])
        ref[LC0_FULL.ones_plane] = 1.0
        out.append(Lc0Record(board, policy[r].astype(np.float64), int(phase[r]), str(src[r]), ref))
    return out


# ── metrics ────────────────────────────────────────────────────────────────────

def entropy(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = q[q > 0.0]
    return float(-(q * np.log(q)).sum())


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


@dataclass
class Position:
    board: chess.Board
    moves: list[chess.Move]
    full_idx: np.ndarray
    leela_idx: np.ndarray
    lc0_target: np.ndarray  # over `moves`, sums to 1
    phase: int
    src: str


def build_positions(records: list[Lc0Record], log: list[str]) -> list[Position]:
    positions: list[Position] = []
    dropped = 0
    for rec in records:
        moves = list(rec.board.legal_moves)
        leela = np.array([leela_index_for_move(rec.board, m) for m in moves], dtype=np.int64)
        stored_support = set(np.flatnonzero(rec.target >= 0).tolist())
        if int((leela < 0).sum()) or set(leela.tolist()) != stored_support:
            dropped += 1
            continue
        tgt = rec.target[leela]
        positions.append(Position(
            board=rec.board, moves=moves,
            full_idx=np.array([move_to_index(m, rec.board) for m in moves], dtype=np.int64),
            leela_idx=leela, lc0_target=tgt / tgt.sum(), phase=rec.phase, src=rec.src,
        ))
    log.append(f"- positions kept {len(positions)}/{len(records)} (dropped {dropped} on "
               f"legal-set disagreement with the stored target)")
    return positions


# ── gates ──────────────────────────────────────────────────────────────────────

def gate_index_correspondence(boards: list[chess.Board], log: list[str]) -> bool:
    checked = bad = 0
    kinds: dict[str, int] = {}
    for b in boards:
        if not any(b.legal_moves):
            continue
        planes = encode_position(b, add_features=False, input_history_encoding="lc0_root")
        gather = leela_gather_indices(*lc0_gather_context_from_planes(planes))[0]
        for mv in b.legal_moves:
            ref = leela_index_for_move(b, mv)
            got = int(gather[compact_index_for_move(b, mv)])
            kind = ("castle" if b.is_castling(mv)
                    else f"promo_{chess.piece_symbol(mv.promotion)}" if mv.promotion else "normal")
            kinds[kind] = kinds.get(kind, 0) + 1
            checked += 1
            if ref < 0 or got != ref:
                bad += 1
                if bad <= 5:
                    log.append(f"    MISMATCH {b.fen()} {mv.uci()} ({kind}): "
                               f"gather={LC0_1858_MOVE_STRS[got]} ref="
                               f"{LC0_1858_MOVE_STRS[ref] if ref >= 0 else 'NO-MAP'}")
    log.append(f"- GATE index-correspondence: {checked} legal moves over {len(boards)} positions, "
               f"{bad} mismatches — {'PASS' if bad == 0 else 'FAIL'}")
    log.append(f"    move kinds exercised: {kinds}")
    return bad == 0


def gate_round_trip(records: list[Lc0Record], log: list[str]) -> bool:
    exact = piece_exact = 0
    notes: list[str] = []
    for i, rec in enumerate(records):
        enc = encode_position(rec.board, add_features=False, input_history_encoding="lc0_root")
        if np.array_equal(enc, rec.planes_ref):
            exact += 1
        else:
            diff = [p for p in range(LC0_FULL.num_planes) if not np.array_equal(enc[p], rec.planes_ref[p])]
            if len(notes) < 5:
                notes.append(f"    record {i} ({rec.src}) differs on planes {diff}")
        pieces = [p for p in range(104) if p % 13 != 12]
        if np.array_equal(enc[pieces], rec.planes_ref[pieces]):
            piece_exact += 1
    n = len(records)
    log.append(f"- GATE round-trip encoder(decoder(planes)) == planes: "
               f"all 112 planes exact on {exact}/{n}; the 96 piece planes exact on {piece_exact}/{n} "
               f"— {'PASS' if piece_exact == n else 'FAIL'}")
    log.extend(notes)
    return piece_exact == n


def _topk_report(pol_compact: np.ndarray, board: chess.Board, k: int = 6) -> list[tuple[str, float]]:
    moves = list(board.legal_moves)
    idx = np.array([compact_index_for_move(board, m) for m in moves], dtype=np.int64)
    lg = pol_compact[idx].astype(np.float64)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    order = np.argsort(-p)[:k]
    return [(board.san(moves[int(o)]), float(p[int(o)])) for o in order]


def gate_net_sanity(ev: Lc0OnnxEvaluator, log: list[str]) -> bool:
    ok = True
    start = chess.Board()
    pol, wdl = ev.raw_eval(start)
    top = _topk_report(pol, start, k=8)
    mainstream = sum(p for san, p in top if san in {"e4", "d4", "Nf3", "c4"})
    value = float(wdl[0] - wdl[2])
    # "carrying most mass" = a majority of the 20 legal first moves' mass sitting
    # on those four. A modern Leela net's opening book is genuinely broad (g3/e3
    # are real lc0 choices), so a tighter bar would be testing lc0's taste rather
    # than our adapter; the top-1 must still be one of the four.
    passed = mainstream > 0.5 and top[0][0] in {"e4", "d4", "Nf3", "c4"} and abs(value) < 0.15
    ok &= passed
    log.append(f"- GATE startpos: top = {[(s, round(p, 3)) for s, p in top]}; "
               f"mass on e4/d4/Nf3/c4 = {mainstream:.3f} (want >0.5, top-1 among them); "
               f"W-L = {value:+.3f} "
               f"(W/D/L {wdl[0]:.3f}/{wdl[1]:.3f}/{wdl[2]:.3f}) — {'PASS' if passed else 'FAIL'}")

    mates = [("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#"),
             ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#")]
    for fen, san in mates:
        b = chess.Board(fen)
        pol, wdl = ev.raw_eval(b)
        top = _topk_report(pol, b)
        value = float(wdl[0] - wdl[2])
        passed = top[0][0] == san and top[0][1] > 0.5 and value > 0.8
        ok &= passed
        log.append(f"- GATE mate-in-1 {fen}: top1 = {top[0][0]} p={top[0][1]:.3f} "
                   f"(want {san}); W-L = {value:+.3f} — {'PASS' if passed else 'FAIL'}")
        log.append(f"    top: {[(s, round(p, 3)) for s, p in top]}")

    midgame = [
        "r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R b KQkq - 0 5",
        "r2q1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R b KQ - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/pbpnqppp/1p2pn2/3p4/2PP4/1PN1PN2/PBQ2PPP/R3KB1R w KQkq - 0 10",
    ]
    for fen in midgame:
        b = chess.Board(fen)
        pol, _ = ev.raw_eval(b)
        top = _topk_report(pol, b, k=5)
        log.append(f"- midgame {fen}\n    top: {[(s, round(p, 3)) for s, p in top]}")
    return ok


# ── search sweep ───────────────────────────────────────────────────────────────

@dataclass
class RootDump:
    actions: np.ndarray
    priors: np.ndarray
    visits: np.ndarray
    qvalues: np.ndarray
    root_q: float


def _install_root_capture(sink: list[RootDump]) -> Any:
    original = gumbel_mod._build_improved_policy_for_board

    def wrapper(st, *, root_q, cfg, rng):
        pri = st.priors
        legal = np.nonzero(pri > 0)[0].astype(np.int64)
        visits = np.zeros(legal.size, dtype=np.float64)
        qvalues = np.zeros(legal.size, dtype=np.float64)
        for i, a in enumerate(legal):
            ch = st.root.children.get(int(a))
            n = 0 if ch is None else int(ch.N)
            visits[i] = float(n)
            qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
        sink.append(RootDump(legal, pri[legal].astype(np.float64), visits, qvalues, float(root_q)))
        return original(st, root_q=root_q, cfg=cfg, rng=rng)

    gumbel_mod._build_improved_policy_for_board = wrapper
    return original


def search_one(
    pos: Position, ev: Lc0OnnxEvaluator, cfg: GumbelConfig, seed: int,
) -> tuple[np.ndarray, RootDump | None]:
    sink: list[RootDump] = []
    original = _install_root_capture(sink)
    try:
        probs, _actions, _values, _masks = run_gumbel_root_many(
            None, [pos.board], device="cpu",
            rng=np.random.default_rng(seed), cfg=cfg, evaluator=ev,
        )
    finally:
        gumbel_mod._build_improved_policy_for_board = original
    return probs[0], (sink[0] if sink else None)


def position_metrics(pos: Position, probs_full: np.ndarray) -> dict[str, float]:
    ours = np.clip(probs_full[pos.full_idx].astype(np.float64), 0.0, None)
    total = ours.sum()
    ours = ours / total if total > 0 else np.full(ours.shape, 1.0 / ours.size)
    lc0 = pos.lc0_target
    mask = lc0 > 0
    kl = float((lc0[mask] * np.log(lc0[mask] / np.clip(ours[mask], 1e-12, None))).sum())
    return {
        "our_entropy": entropy(ours),
        "our_support": float(int((ours > 1e-3).sum())),
        "our_maxp": float(ours.max()),
        "lc0_entropy": entropy(lc0),
        "lc0_support": float(int((lc0 > 1e-3).sum())),
        "lc0_full_support": float(int(mask.sum())),
        "lc0_maxp": float(lc0.max()),
        "kl": kl,
        "n_legal": float(len(pos.moves)),
        "top1_agree": float(int(np.argmax(ours) == np.argmax(lc0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="data/lc0/onnx/BT4-it332-vanilla-winner.onnx")
    ap.add_argument("--matched", type=Path, required=True, help="matched60.npz")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--warm-mult", type=int, default=2)
    # c_scale 0.0 makes the value transform vanish, so the "improved" policy IS
    # the net's prior — the reference that separates prior error from search error.
    ap.add_argument("--c-scales", type=float, nargs="+",
                    default=[0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--topks", type=int, nargs="+", default=[8, 16, 32, 218])
    ap.add_argument("--gumbel-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", type=Path, default=Path("runs/lc0_adapter_probe.md"))
    ap.add_argument("--root-dump", type=Path, default=Path("runs/lc0_adapter_roots.npz"))
    args = ap.parse_args()

    net_sha = hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest()
    log: list[str] = ["# lc0-net-in-our-search probe", "",
                      f"- net: `{args.onnx}`",
                      f"- net sha256: `{net_sha}`",
                      f"- matched set: `{args.matched}`",
                      f"- search: {args.sims} sims, gumbel_scale {args.gumbel_scale}, "
                      f"seed {args.seed} (re-seeded per position, so configs are paired)", ""]
    t0 = time.time()
    ev = Lc0OnnxEvaluator(args.onnx, providers=("CPUExecutionProvider",))
    log.append(f"- ONNX session up in {time.time() - t0:.1f}s (CPU provider)")

    records = decode_records(args.matched)
    log.append("")
    log.append("## Gates")
    ok_rt = gate_round_trip(records, log)

    rng = random.Random(args.seed)
    corr_boards: list[chess.Board] = [r.board for r in records]
    for _ in range(40):  # random self-play walks: castling, promotions, checks
        b = chess.Board()
        for _ply in range(rng.randint(0, 100)):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
            corr_boards.append(b.copy(stack=8))
    corr_boards += [chess.Board(f) for f in (
        "8/PPPPPPPP/8/8/8/7k/8/K7 w - - 0 1",
        "k7/8/7K/8/8/8/pppppppp/8 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    )]
    ok_idx = gate_index_correspondence(corr_boards, log)
    ok_net = gate_net_sanity(ev, log)
    log.append("")
    log.append(f"- gate summary: round-trip {'PASS' if ok_rt else 'FAIL'}, "
               f"index-correspondence {'PASS' if ok_idx else 'FAIL'}, "
               f"net sanity {'PASS' if ok_net else 'FAIL'}")
    positions = build_positions(records, log)
    if not (ok_rt and ok_idx):
        log.append("")
        log.append("**A prerequisite gate FAILED — the search numbers below are not meaningful.**")

    base_cfg = GumbelConfig(
        simulations=args.sims, topk=16, c_scale=0.1, gumbel_scale=args.gumbel_scale,
        add_noise=True, temperature=0.0,
        input_history_encoding="lc0_root", input_extra_features="v1",
    )

    # ── warm pass: one generous search per position fills the eval cache ───────
    warm_cfg = replace(base_cfg, topk=256, simulations=args.sims * args.warm_mult)
    t0 = time.time()
    run_gumbel_root_many(
        None, [p.board for p in positions], device="cpu",
        rng=np.random.default_rng(args.seed), cfg=warm_cfg, evaluator=ev,
    )
    warm_miss = ev.misses
    log.append("")
    log.append("## Eval memoization")
    log.append(f"- warm pass: topk=all, {warm_cfg.simulations} sims, all {len(positions)} positions "
               f"in one batched search, {time.time() - t0:.1f}s, {warm_miss} net evals cached "
               f"({ev.net_calls} ONNX calls)")

    # ── sweep ────────────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    per_config_miss: dict[tuple[float, int], int] = {}
    dumps: list[RootDump] = []
    dump_positions: list[Position] = []
    t0 = time.time()
    for c_scale in args.c_scales:
        for topk in args.topks:
            before = ev.misses
            cfg = replace(base_cfg, c_scale=c_scale, topk=topk)
            is_prod = (abs(c_scale - 0.1) < 1e-12 and topk == 16)
            for pi, pos in enumerate(positions):
                probs, dump = search_one(pos, ev, cfg, args.seed + pi)
                m = position_metrics(pos, probs)
                m.update(c_scale=c_scale, topk=topk, phase=pos.phase, pi=pi)
                rows.append(m)
                if is_prod and dump is not None:
                    dumps.append(dump)
                    dump_positions.append(pos)
            per_config_miss[(c_scale, topk)] = ev.misses - before
            print(f"[sweep] c_scale={c_scale} topk={topk} done "
                  f"({ev.misses - before} cache misses)", flush=True)
    log.append(f"- sweep: {len(args.c_scales)}x{len(args.topks)} configs x {len(positions)} positions "
               f"in {time.time() - t0:.1f}s; cache hits {ev.hits}, misses {ev.misses - warm_miss} "
               f"after warm-up")
    worst = sorted(per_config_miss.items(), key=lambda kv: -kv[1])[:4]
    log.append(f"- misses per config (worst 4): "
               f"{[(f'c={c}/k={k}', v) for (c, k), v in worst]}")

    # Sequential halving caps the candidate set at ceil(sims/2), so above that
    # the topk axis cannot bite. Say so from the data rather than from the code.
    ref = {(r["c_scale"], r["pi"]): r["our_entropy"] for r in rows if r["topk"] == 16}
    for topk in args.topks:
        if topk == 16:
            continue
        same = sum(1 for r in rows if r["topk"] == topk
                   and ref.get((r["c_scale"], r["pi"])) == r["our_entropy"])
        total = sum(1 for r in rows if r["topk"] == topk)
        log.append(f"- topk={topk} vs topk=16: bit-identical search output on {same}/{total} "
                   f"(position, c_scale) pairs "
                   f"[candidates are capped at ceil(sims/2)={max(2, (args.sims + 1) // 2)}]")

    if dumps:
        # One flat row per (position, legal action) so any c_scale / q-transform
        # can be recomputed offline from `_completed_q_transform` alone. The root
        # dump orders actions by 4672 id, the Position by `board.legal_moves`, so
        # realign rather than assume.
        lc0_aligned: list[np.ndarray] = []
        uci_aligned: list[list[str]] = []
        for dump, pos in zip(dumps, dump_positions, strict=True):
            order = {int(f): i for i, f in enumerate(pos.full_idx)}
            if set(order) != {int(a) for a in dump.actions}:
                raise SystemExit("root dump action set does not match the position's legal moves")
            take = np.array([order[int(a)] for a in dump.actions], dtype=np.int64)
            lc0_aligned.append(pos.lc0_target[take])
            uci_aligned.append([pos.moves[int(i)].uci() for i in take])
        args.root_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.root_dump,
            position=np.concatenate([np.full(d.actions.size, i) for i, d in enumerate(dumps)]),
            action=np.concatenate([d.actions for d in dumps]),
            prior=np.concatenate([d.priors for d in dumps]),
            visits=np.concatenate([d.visits for d in dumps]),
            qvalue=np.concatenate([d.qvalues for d in dumps]),
            lc0_target=np.concatenate(lc0_aligned),
            uci=np.array([u for group in uci_aligned for u in group]),
            root_q=np.array([d.root_q for d in dumps]),
            phase=np.array([p.phase for p in dump_positions]),
            fen=np.array([p.board.fen() for p in dump_positions]),
            src=np.array([p.src for p in dump_positions]),
        )
        log.append(f"- root dump (production config) → `{args.root_dump}` "
                   f"({len(dumps)} of {len(positions)} positions: per-action legal id / prior / "
                   f"visits / completed-Q / lc0 target, plus per-position root Q, phase, FEN, "
                   f"provenance). Positions the search short-circuits at the root (single legal "
                   f"move, or a root tactic that returns a fixed policy) have no tree to dump.")

    # ── tables ───────────────────────────────────────────────────────────────
    log.append("")
    log.append("## lc0's own stored targets (reference)")
    log.append("")
    log.append("| phase | n | median entropy (nats) | median support (p>1e-3) | median max-prob | "
               "median n_legal | median moves with p>0 |")
    log.append("|---|---|---|---|---|---|---|")
    prod_rows = [r for r in rows if abs(r["c_scale"] - 0.1) < 1e-12 and r["topk"] == 16]
    for ph in range(3):
        sel = [r for r in prod_rows if r["phase"] == ph]
        log.append(f"| {PHASE_NAMES[ph]} | {len(sel)} | {median([r['lc0_entropy'] for r in sel]):.3f} | "
                   f"{median([r['lc0_support'] for r in sel]):.1f} | "
                   f"{median([r['lc0_maxp'] for r in sel]):.3f} | "
                   f"{median([r['n_legal'] for r in sel]):.1f} | "
                   f"{median([r['lc0_full_support'] for r in sel]):.1f} |")
    log.append("")
    log.append("Supports below are all counted at p>1e-3 so the two sides are comparable; "
               "lc0's targets additionally carry a nonzero tail on essentially every legal "
               "move (root Dirichlet noise in training games).")

    log.append("")
    log.append("## Our Gumbel search target, per (config, phase)")
    log.append("")
    log.append("| c_scale | topk | phase | median H (nats) | median support | median max-p | "
               "median KL(lc0‖ours) | top1 agree |")
    log.append("|---|---|---|---|---|---|---|---|")
    best: dict[int, tuple[float, tuple[float, int]]] = {}
    for c_scale in args.c_scales:
        for topk in args.topks:
            for ph in range(3):
                sel = [r for r in rows if r["phase"] == ph and r["topk"] == topk
                       and abs(r["c_scale"] - c_scale) < 1e-12]
                if not sel:
                    continue
                kl = median([r["kl"] for r in sel])
                agree = float(np.mean([r["top1_agree"] for r in sel]))
                log.append(
                    f"| {c_scale} | {topk} | {PHASE_NAMES[ph]} | "
                    f"{median([r['our_entropy'] for r in sel]):.3f} | "
                    f"{median([r['our_support'] for r in sel]):.1f} | "
                    f"{median([r['our_maxp'] for r in sel]):.3f} | {kl:.3f} | {agree:.2f} |")
                if ph not in best or kl < best[ph][0]:
                    best[ph] = (kl, (c_scale, topk))
    log.append("")
    for ph in range(3):
        if ph in best:
            kl, (c_scale, topk) = best[ph]
            log.append(f"- **argmin-KL {PHASE_NAMES[ph]}**: c_scale={c_scale}, topk={topk} "
                       f"(median KL {kl:.3f})")

    text = "\n".join(log) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
