"""Sibling-ordering value target: materialised child pairs and the training term.

**The target.** At a searched root, two explored children A and B carry completed
Q values ``q_A`` and ``q_B`` (``root_child_q``: the ROOT's point of view, Q of the
child when visited). The term trains the value head's DIFFERENCE between the two
children toward the searched difference ``dq = q_A - q_B``. A difference cancels
any shift shared by both children, and children are off-policy positions that
root-only training never presents. Scoping evidence:
``docs/experiments/2026-11-21-rl-loop-evidence-log.md`` (sibling-target scoping row).

**Materialiser** (``materialise``; CLI ``scripts/simple_rl_sibling_pairs.py``).
From a sealed collect it writes one sidecar directory per shard. For every row
with root statistics, it samples K pairs among the children with at least
``min_visits`` visits that the search did not score as terminal by its own rule
(``search_terminal``: game over, or the LC0-style 2-fold repetition the search
scores as an exact draw without evaluating the net). Pairs are drawn without
replacement with probability proportional to ``min(v_A, v_B)``, from an RNG keyed
by ``(seed, game_id, ply)``, so the choice is reproducible and independent of
shard and row order. The sidecar stores:

- both child inputs (float16, encoded by the shared encoder exactly as the collector
  encodes a root);
- ``dq`` in the PARENT's point of view, and both visit counts;
- both action ids (compact lc0_1858), and the row key (game id, ply, offset in the
  shard);
- a hold-out flag from the learner's ``is_holdout_game``.

**Rebuilding a row's board.** The start is the game's opening line: the
``starts.txt`` line whose ``line_id`` is the game's ``opening_id``, or the
manifest's train-openings file when it still has the sha256 the manifest recorded.
After that come the game's moves:

- **Collects with a per-game move list (#1444 and later):** the stored ``uci`` list,
  with no hashing.
- **Older collects:** each row's ``played_move_index``. A ply with no row, or a row
  with no played move, is filled ONLY when exactly one move sequence (at most
  ``MAX_BRIDGE_PLIES`` plies) encodes to the next row's stored ``x_hash``. This
  rule was proved on the 8,000-game pilot: 1,053 / 1,053 played children and 1,080 /
  1,080 roots byte-equal.

Every row's rebuilt root encoding must hash to its stored ``x_hash``, in BOTH
modes and for the whole collect (not a sample). A collect that fails anywhere is
refused (``CollectNotRebuildable``) before anything is written.

**rep_fix.** The C encoder's repetition planes depend on the process-global
``history_rep_fix``. The materialiser applies the collect's regime before its
first board. It stamps the declared regimes (collect arch and every shard) and
the realised one (``rep_fix.current()``), and refuses any disagreement.

**Training term** (``SiblingTerm``). Loss =

    w * mean over pairs [ (min(v_A, v_B) / 64) * Huber_0.2(dv - dq) ]

where ``dv = v(A) - v(B)`` comes from the WDL head of the SAME net in the same
step.

**Loss form** (``loss``). ``diff`` (the registered form): the Huber regression of
dv onto dq above, which enforces the size of the difference as well as its sign.
``order`` (form version 2): per pair ``(min_visits / 64) * min(|dq|, 0.62) *
relu(min(margin, |dq|) - sign(dq) * dv)`` for pairs with ``|dq| >= min_dq`` (else 0):
zero once the net orders the pair as the search does by at least the smaller of
``margin`` and the search's own gap (the cap is the form's definition; there is no
uncapped path), and a mis-ordered large gap costs in proportion to the gap, bounded.
Version 1 (unweighted) was never registered.

**Unit** (``unit``): ``pairs`` reads the version-2 pair sidecar (K sampled pairs per
row) and averages over the step's pairs. ``roots`` (``SiblingListTerm``, order form
only) reads a root-list sidecar (every eligible child of each root with >= 2), scores
EVERY pair among a root's children with ``|dq| >= min_dq``, takes each root's mean and
combines roots with weight n_eff = n / (1 + (n - 1) * 0.08) (``ROOT_PAIR_ICC``, fixed). Same pairs, same child
forwards, same chunking. An experiment arm, not an identity twin.

**Chunked, with a backward per chunk** (the only path). The step's pairs are
forwarded ``chunk_pairs`` at a time and each chunk runs its own
``(scale * w * chunk_weighted.sum() / n_pairs).backward()`` inside the penalty
hook, which then returns a 0-valued leaf (no parameter behind it). The chunk terms sum to ``w * mean``, so the
gradient is the single-forward term's up to float summation order, while only
ONE chunk's child graph is alive at a time: peak memory = the root batch's graph
+ one chunk (a 2 x 255-child forward at 128 rows exhausted a 32 GB card on top of
a 512-row batch, 2026-11-22). ``scale`` is the Trainer's own ``rows /
batch_size`` for the exact-once sampler (it multiplies whatever the hook returns,
and a gradient already taken inside the hook must carry it too). The hook runs
after the Trainer's ``zero_grad`` and before its ``backward``; it refuses to run
on gradients that were not cleared for this update.

POV algebra: a child is the position AFTER the parent's move, so the child's
side to move is the parent's OPPONENT. The head outputs ``(W, D, L)`` for the
child's side to move, and the parent's value of the child is
``W_parent - L_parent = L_child - W_child``. ``root_child_q`` is in the parent's
POV, so ``v(child) = softmax(wdl)[2] - softmax(wdl)[0]``.
"""
from __future__ import annotations

import json
import math
import resource
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.simple_rl.collect import GAMES, generation_shards, shard_row_meta, x_hash
from chess_anti_engine.simple_rl.common import (
    is_holdout_game, read_manifest, resolve_collect_dir, seed_for, sha256_file,
)

SIDECAR_VERSION = 2  # 2: children excluded by the search's terminal rule (incl. 2-fold)
SIDECAR_MANIFEST = "sibling_manifest.json"
GAME_MOVES = "game_moves.jsonl"
MAX_BRIDGE_PLIES = 3          # a pilot gap needed at most 3 plies with the first move known
MAX_BRIDGE_UNKNOWN_FIRST = 2  # ...and at most 2 when the row stored no played move
VISIT_SCALE = 64.0            # the loss weight is min(v_A, v_B) / 64 (the collection's sims)
HUBER_DELTA = 0.2
# The ORDER form's gap weight min(|dq|, ORDER_GAP_CAP): the cost of preferring the wrong child is the value given up,
# |dq|, bounded at the v2 pilot sidecar's min-visits-weighted p90 of |dq| (0.619) so decided / near-mate gaps (|dq| up
# to 2) do not dominate. Part of the form's definition (no setting).
ORDER_GAP_CAP = 0.62
# Version of the ORDER form. 1 = unweighted (never registered); 2 = gap-weighted, pair and root-list units, roots
# weighted with rho 0.13; 3 = the same with rho 0.08 (ROOT_PAIR_ICC below; the pair unit's loss is unchanged). Carried in
# the term's snapshot state and the phase resume key, so a snapshot of another version never resumes (it is refused).
ORDER_FORM_VERSION = 3
# Root-list unit: a root's pair losses are averaged, and roots are combined with weight n_eff = n / (1 + (n - 1) rho)
# (n = the root's eligible pairs): the inverse-variance weight of a root's mean under a within-root correlation rho.
# rho = 0.08 is FIXED: the pre-registered full-list measurement (operations bank rl-claude-20261119/
# sibling-fulllist-rho: the SL base on every eligible child of 66,334 roots, rho 0.0782 [0.0681, 0.0924] by game
# bootstrap; the earlier 0.13, from K = 2 sampled pairs per row, lies outside it). One rho for every n by that
# registration, although it varies with n (0.19 at 2-5 pairs, 0.06 at 16-30); never refitted.
ROOT_PAIR_ICC = 0.08
PAIR_KIND = "simple_rl_sibling_pairs"
LIST_KIND = "simple_rl_sibling_lists"
LIST_SIDECAR_VERSION = 1  # 1: every non-terminal child with >= min_visits visits of every root with >= 2 of them
PAIR_FIELDS: dict[str, Any] = {
    "x_a": np.float16, "x_b": np.float16, "dq": np.float32, "q_a": np.float32, "q_b": np.float32,
    "visits_a": np.float32, "visits_b": np.float32, "min_visits": np.float32,
    "action_a": np.int32, "action_b": np.int32, "game_id": np.int64, "ply_index": np.int32,
    "row": np.int32, "holdout": np.bool_,
}
LIST_FIELDS: dict[str, Any] = {
    "x": np.float16, "action": np.int32, "q": np.float32, "visits": np.float32, "game_id": np.int64,
    "ply_index": np.int32, "row": np.int32, "holdout": np.bool_,
}
_ROW_KEYS = ("has_root_child_q", "root_child_q", "root_child_visits", "played_move_index", "has_played_move")


class CollectNotRebuildable(ValueError):
    """A row's board cannot be rebuilt byte-exactly from what the collect stored."""


# ---------------------------------------------------------------------------
# The collect: openings, games, encoder regime
# ---------------------------------------------------------------------------


@dataclass
class CollectView:
    collect_dir: Path          # the directory the rows live in (a reference resolved)
    manifest_sha256: str
    arch: dict[str, Any]
    enc: dict[str, str]
    history_rep_fix: bool
    games: dict[int, dict[str, Any]]
    lines: dict[str, str]
    lines_source: str
    shards: list[Path] = field(default_factory=list)


def open_collect(collect_dir: Path) -> CollectView:
    """Read a SEALED collect's identity, encoder regime, games and opening lines."""
    from chess_anti_engine.simple_rl.openings import line_id

    collect_dir = Path(collect_dir)
    if read_manifest(collect_dir) is None:
        raise ValueError(f"{collect_dir} is not a sealed collect (no manifest.json)")
    root = resolve_collect_dir(collect_dir)
    manifest = read_manifest(root)
    if manifest is None or manifest.get("stage") != "collect":
        raise ValueError(f"{root} has no sealed collect manifest")
    arch = dict(manifest.get("config", {}).get("arch") or {})
    missing = [k for k in ("input_history_encoding", "input_extra_features", "history_rep_fix") if k not in arch]
    if missing:
        raise ValueError(f"{root}/manifest.json arch does not declare {missing}; the child encoding would be a guess")
    enc = {"input_history_encoding": str(arch["input_history_encoding"]),
           "input_extra_features": str(arch["input_extra_features"])}
    games = {}
    for raw in (root / GAMES).read_text().splitlines():
        if raw.strip():
            g = json.loads(raw)
            games[int(g["game_id"])] = g
    lines: dict[str, str] = {}
    source = "starts.txt line_id"
    starts = root / "starts.txt"
    if starts.is_file():
        for ln in starts.read_text().splitlines():
            if ln.strip():
                lines[line_id(ln.rstrip())] = ln.rstrip()
    ref = (manifest.get("inputs") or {}).get("train_openings") or {}
    path = Path(ref.get("path", "")) if ref.get("path") else None
    if path is not None and path.is_file() and sha256_file(path) == ref.get("sha256"):
        for raw in path.read_text().splitlines():
            if raw.strip():
                row = json.loads(raw)
                lines.setdefault(str(row["id"]), str(row["line"]).rstrip())
        source = "starts.txt line_id + train_openings (sha256 verified)"
    unresolved = sorted({str(g["opening_id"]) for g in games.values()} - set(lines))
    if unresolved:
        raise CollectNotRebuildable(f"{len(unresolved)} opening id(s) resolve to no line (first {unresolved[:3]}); "
                                    "the starts.txt line ids do not match and the train-openings file is missing "
                                    "or changed")
    return CollectView(collect_dir=root, manifest_sha256=sha256_file(root / "manifest.json"), arch=arch, enc=enc,
                       history_rep_fix=bool(arch["history_rep_fix"]), games=games, lines=lines,
                       lines_source=source, shards=generation_shards(root))


def apply_collect_regime(view: CollectView) -> dict[str, Any]:
    """Put this process in the collect's repetition-plane regime BEFORE any board
    exists, check every shard declares the same regime and encoding, and return
    the stamp (declared by the collect, declared by the shards, realised)."""
    from chess_anti_engine.encoding import rep_fix
    from chess_anti_engine.replay.shard import (
        INPUT_HISTORY_ENCODING_ARRAY_KEY, history_rep_fix_from_arrays, open_shard_arrays,
    )

    shard_regimes, shard_encodings = set(), set()
    for s in view.shards:
        with open_shard_arrays(s, lazy=True, validate=True) as (arrs, _meta):
            shard_regimes.add(bool(history_rep_fix_from_arrays(arrs)))
            raw = np.asarray(arrs.get(INPUT_HISTORY_ENCODING_ARRAY_KEY, np.asarray("")))
            shard_encodings.add(str(raw.reshape(-1)[0]) if raw.size else "")
    if shard_regimes != {view.history_rep_fix}:
        raise ValueError(f"shards declare history_rep_fix {sorted(shard_regimes)} but the collect arch declares "
                         f"{view.history_rep_fix}")
    if shard_encodings != {view.enc["input_history_encoding"]}:
        raise ValueError(f"shards declare input_history_encoding {sorted(shard_encodings)} but the collect arch "
                         f"declares {view.enc['input_history_encoding']!r}")
    # boards_discarded: every CBoard this module builds is created after this call
    # and dies inside encode() (python-chess boards hold no C state), so no board
    # of the previous regime is ever encoded here. (build_model certifies the same.)
    rep_fix.apply(view.history_rep_fix, boards_discarded=True)
    if rep_fix.current() is not view.history_rep_fix:
        raise RuntimeError(f"history_rep_fix realised {rep_fix.current()!r}, collect declares {view.history_rep_fix}")
    return {"collect_declared": view.history_rep_fix, "shards_declared": view.history_rep_fix,
            "process_realised": bool(rep_fix.current())}


def search_terminal(board: chess.Board) -> str | None:
    """Whether the SEARCH treats this position as terminal, by its own rule
    (``mcts/_search_terminal.h::cboard_search_terminal``, through the same C
    board): ``"game_over"`` (checkmate, stalemate, 3-fold, 50-move, insufficient
    material), ``"repetition"`` (LC0-style 2-fold, scored as an exact draw, q 0,
    the net never evaluated), else None. Such a child's q is a rule, not a
    search value, so it is never paired. python-chess's
    ``is_game_over(claim_draw=False)`` misses the 2-fold and the 3-fold / 50-move
    claims the C rule counts."""
    from chess_anti_engine.encoding.cboard_encode import CBoard

    cb = CBoard.from_board(board)
    if cb.is_game_over():
        return "game_over"
    if cb.is_repetition():
        return "repetition"
    return None


def encode(board: chess.Board, enc: dict[str, str]) -> np.ndarray:
    """The shared encoder, in the collector's float16 storage dtype."""
    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard

    return encode_cboard(CBoard.from_board(board), **enc).astype(np.float16)


# ---------------------------------------------------------------------------
# Rebuilding a game's row boards
# ---------------------------------------------------------------------------


def _start_board(view: CollectView, game: dict[str, Any]) -> chess.Board:
    line = view.lines[str(game["opening_id"])]
    fen, _, moves = line.partition(" | ")
    board = chess.Board(fen.strip())
    book = moves.split()
    for u in book:
        board.push_uci(u)
    if len(book) != int(game["start_ply"]):
        raise CollectNotRebuildable(f"game {game['game_id']}: book has {len(book)} plies, start_ply is "
                                    f"{game['start_ply']}")
    return board


def _bridges(board: chess.Board, depth: int, want: int, enc: dict[str, str]) -> list[list[chess.Move]]:
    if depth == 0:
        return [[]] if int(x_hash(encode(board, enc))) == want else []
    out: list[list[chess.Move]] = []
    for m in list(board.legal_moves):
        board.push(m)
        out += [[m, *rest] for rest in _bridges(board, depth - 1, want, enc)]
        board.pop()
    return out


def rebuild_game(view: CollectView, game_id: int, plies: np.ndarray, hashes: np.ndarray,
                 played: np.ndarray, has_played: np.ndarray) -> Iterator[tuple[int, chess.Board]]:
    """Yield ``(i, board)`` for the game's rows in ply order (``plies`` sorted).

    ``board`` is the live board at row i's position. A consumer may push and pop
    on it but must leave it as it found it. Every row's root encoding is checked
    against its stored ``x_hash`` before it is yielded. Raises
    ``CollectNotRebuildable`` naming the game and ply.
    """
    from chess_anti_engine.moves.encode import index_to_move_for_encoding

    game = view.games.get(int(game_id))
    if game is None:
        raise CollectNotRebuildable(f"game {game_id} has rows but no games.jsonl record")
    board = _start_board(view, game)
    start = int(game["start_ply"])
    if len(plies) and int(plies[0]) < start:
        raise CollectNotRebuildable(f"game {game_id}: a row at ply {int(plies[0])} precedes start_ply {start}")
    uci = game.get("uci")
    if uci is not None and len(uci) != int(game["total_plies"]) - start:
        raise CollectNotRebuildable(f"game {game_id}: {len(uci)} stored moves for {game['total_plies']} - {start} plies")
    ply = start
    for i in range(len(plies)):
        target = int(plies[i])
        if uci is not None:
            while ply < target:
                try:
                    board.push_uci(uci[ply - start])
                except ValueError as exc:  # IllegalMoveError / InvalidMoveError
                    raise CollectNotRebuildable(f"game {game_id} ply {ply}: stored move list does not replay "
                                                f"({exc})") from exc
                ply += 1
        elif ply != target:
            raise CollectNotRebuildable(f"game {game_id}: reached ply {ply}, next row is at {target}")
        if int(x_hash(encode(board, view.enc))) != int(hashes[i]):
            raise CollectNotRebuildable(f"game {game_id} ply {target}: rebuilt root does not hash to the stored x_hash")
        yield i, board
        if uci is not None or i + 1 == len(plies):
            continue
        gap = int(plies[i + 1]) - target
        want = int(hashes[i + 1])
        if bool(has_played[i]):
            first = index_to_move_for_encoding(int(played[i]), board)
            if first not in board.legal_moves:
                raise CollectNotRebuildable(f"game {game_id} ply {target}: stored played move {first} is illegal")
            if gap > MAX_BRIDGE_PLIES:
                raise CollectNotRebuildable(f"game {game_id} ply {target}: {gap}-ply gap exceeds {MAX_BRIDGE_PLIES}")
            board.push(first)
            seqs = [[first, *rest] for rest in _bridges(board, gap - 1, want, view.enc)]
            board.pop()
        else:
            if gap > MAX_BRIDGE_UNKNOWN_FIRST:
                raise CollectNotRebuildable(f"game {game_id} ply {target}: no played move and a {gap}-ply gap")
            seqs = _bridges(board, gap, want, view.enc)
        if len(seqs) != 1:
            raise CollectNotRebuildable(f"game {game_id} ply {target}: {len(seqs)} move sequences reach the next "
                                        "row's x_hash (exactly one is required)")
        for m in seqs[0]:
            board.push(m)
        ply += gap


def _game_rows(meta: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Row offsets of each game in the shard, in ply order."""
    out: dict[int, list[int]] = defaultdict(list)
    for off, g in enumerate(meta["game_id"]):
        out[int(g)].append(off)
    plies = np.asarray(meta["ply_index"])
    return {g: np.asarray(sorted(rows, key=lambda r: int(plies[r])), dtype=np.int64) for g, rows in out.items()}


def _check_games_within_shards(view: CollectView) -> list[dict[str, np.ndarray]]:
    seen: dict[int, int] = {}
    metas = []
    for si, s in enumerate(view.shards):
        meta = shard_row_meta(s)
        metas.append(meta)
        for g in np.unique(meta["game_id"]):
            if int(g) in seen:
                raise CollectNotRebuildable(f"game {int(g)} has rows in shards {seen[int(g)]} and {si}; a game is "
                                            "rebuilt within one shard")
            seen[int(g)] = si
    return metas


# ---------------------------------------------------------------------------
# Pair choice
# ---------------------------------------------------------------------------


def pair_rng(seed: int, game_id: int, ply: int) -> np.random.Generator:
    return np.random.default_rng(seed_for(int(seed), "sibling_pairs", int(game_id), int(ply)))


def choose_pairs(children: np.ndarray, visits: np.ndarray, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to ``k`` distinct pairs of ``children`` (compact ids), without replacement,
    with probability proportional to ``min(v_A, v_B)``. In each pair A is the
    more-visited child (ties: the lower id), so ``dq`` reads "search's preference
    minus the alternative"."""
    order = sorted((int(c) for c in children), key=lambda c: (-float(visits[c]), c))
    if len(order) < 2 or k < 1:
        return []
    ia, ib = np.triu_indices(len(order), 1)
    w = np.minimum(visits[np.asarray(order)[ia]], visits[np.asarray(order)[ib]]).astype(np.float64)
    take = rng.choice(len(w), size=min(int(k), len(w)), replace=False, p=w / w.sum())
    return [(order[int(ia[t])], order[int(ib[t])]) for t in sorted(take.tolist())]


# ---------------------------------------------------------------------------
# Materialiser
# ---------------------------------------------------------------------------


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _shard_rows(shard: Path) -> dict[str, np.ndarray]:
    from chess_anti_engine.replay.shard import open_shard_arrays

    with open_shard_arrays(shard, lazy=True, validate=True) as (arrs, _meta):
        out = {}
        for k in _ROW_KEYS:
            if k not in arrs:
                raise ValueError(f"{shard}: no {k!r}; the collect did not record root statistics")
            out[k] = np.asarray(arrs[k][:])
    return out


_PLAN_COLUMNS: dict[str, Any] = {
    "row": np.int32, "game_id": np.int64, "ply_index": np.int32, "action_a": np.int32, "action_b": np.int32,
    "q_a": np.float32, "q_b": np.float32, "visits_a": np.float32, "visits_b": np.float32, "holdout": np.bool_,
}


_LIST_PLAN_COLUMNS: dict[str, Any] = {
    "row": np.int32, "game_id": np.int64, "ply_index": np.int32, "action": np.int32, "q": np.float32,
    "visits": np.float32, "holdout": np.bool_,
}


def _plan_shard(view: CollectView, shard_index: int, meta: dict[str, np.ndarray], *, k: int, min_visits: float,
                seed: int, holdout_frac: float, unit: str = "pairs",
                ) -> tuple[dict[str, np.ndarray], dict[int, list[str]], dict[str, int]]:
    """Pass 1: rebuild every game of the shard (every root hash checked), choose
    each row's pairs, and record each game's moves as UCI for pass 2. The plan is
    in (game, ply) order -- the order pass 2 writes -- as compact columns."""
    from chess_anti_engine.moves.encode import index_to_move_for_encoding
    rows = _shard_rows(view.shards[shard_index])
    plan: list[tuple[Any, ...]] = []
    moves_of: dict[int, list[str]] = {}
    counts = {"rows": len(meta["game_id"]), "rows_with_stats": 0, "rows_with_pairs": 0,
              "children" if unit == "roots" else "pairs": 0,
              "terminal_children_excluded": 0, "repetition_children_excluded": 0}
    if unit == "roots":
        counts["rows_with_one_child"] = 0
    for gid, offs in _game_rows(meta).items():
        plies = np.asarray(meta["ply_index"])[offs]
        it = rebuild_game(view, gid, plies, np.asarray(meta["x_hash"])[offs], rows["played_move_index"][offs],
                          rows["has_played_move"][offs])
        held = is_holdout_game(gid, holdout_frac)
        board = None
        for i, board in it:
            off = int(offs[i])
            if not bool(rows["has_root_child_q"][off]):
                continue
            counts["rows_with_stats"] += 1
            visits = rows["root_child_visits"][off]
            cand = []
            for c in np.flatnonzero(visits >= float(min_visits)):
                m = index_to_move_for_encoding(int(c), board)
                board.push(m)
                kind = search_terminal(board)
                board.pop()
                if kind == "game_over":
                    counts["terminal_children_excluded"] += 1
                elif kind == "repetition":
                    counts["repetition_children_excluded"] += 1
                else:
                    cand.append(int(c))
            q = rows["root_child_q"][off]
            if unit == "roots":   # every eligible child of a root with at least two of them
                counts["rows_with_one_child"] += int(len(cand) == 1)
                if len(cand) >= 2:
                    counts["rows_with_pairs"] += 1
                    plan.extend((off, gid, int(plies[i]), c, float(q[c]), float(visits[c]), held) for c in sorted(cand))
                continue
            pairs = choose_pairs(np.asarray(cand), visits, k, pair_rng(seed, gid, int(plies[i])))
            if pairs:
                counts["rows_with_pairs"] += 1
            for a, b in pairs:
                plan.append((off, gid, int(plies[i]), a, b, float(q[a]), float(q[b]), float(visits[a]),
                             float(visits[b]), held))
        if board is not None:
            start = _start_board(view, view.games[gid])
            moves_of[gid] = [m.uci() for m in board.move_stack[len(start.move_stack):]]
    columns = _LIST_PLAN_COLUMNS if unit == "roots" else _PLAN_COLUMNS
    counts["children" if unit == "roots" else "pairs"] = len(plan)
    cols = list(zip(*plan, strict=True)) if plan else [() for _ in columns]
    out = {name: np.asarray(col, dtype=dt) for (name, dt), col in zip(columns.items(), cols, strict=True)}
    if unit == "pairs":
        out["dq"] = (out["q_a"] - out["q_b"]).astype(np.float32)
        out["min_visits"] = np.minimum(out["visits_a"], out["visits_b"]).astype(np.float32)
    return out, moves_of, counts


def _npy_writer(path: Path, n: int) -> Any:
    """An open .npy of shape (n, 175, 8, 8) float16, header written, for
    sequential row writes (through the page cache, not a growing mapping)."""
    fh = path.open("wb")
    np.lib.format.write_array_header_1_0(fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                                              "fortran_order": False, "shape": (n, 175, 8, 8)})
    return fh


def _write_shard(view: CollectView, out: Path, meta: dict[str, np.ndarray], plan: dict[str, np.ndarray],
                 moves_of: dict[int, list[str]], unit: str = "pairs") -> dict[str, str]:
    """Pass 2: replay each game's recorded moves and encode the chosen children,
    written sequentially in plan order (bounded RAM)."""
    from chess_anti_engine.moves.encode import index_to_move_for_encoding

    out.mkdir(parents=True, exist_ok=False)
    inputs = (("action", "x"),) if unit == "roots" else (("action_a", "x_a"), ("action_b", "x_b"))
    n = int(plan["row"].size)
    writers = [(col, _npy_writer(out / f"{name}.npy", n)) for col, name in inputs]
    plies = np.asarray(meta["ply_index"])
    board: chess.Board | None = None
    current = None
    start = 0
    try:
        for j in range(n):
            gid, ply, off = int(plan["game_id"][j]), int(plan["ply_index"][j]), int(plan["row"][j])
            if gid != current:
                game = view.games[gid]
                board, current, start = _start_board(view, game), gid, int(game["start_ply"])
            assert board is not None
            assert int(plies[off]) == ply
            moves = moves_of[gid]
            if len(board.move_stack) - start > ply - start:
                raise RuntimeError(f"plan out of order at game {gid} ply {ply}")
            while len(board.move_stack) - start < ply - start:
                board.push_uci(moves[len(board.move_stack) - start])
            for col, fh in writers:
                board.push(index_to_move_for_encoding(int(plan[col][j]), board))
                fh.write(np.ascontiguousarray(encode(board, view.enc)).tobytes())
                board.pop()
    finally:
        for _col, fh in writers:
            fh.close()
    fields = LIST_FIELDS if unit == "roots" else PAIR_FIELDS
    for name in fields:
        if not name.startswith("x"):
            np.save(out / f"{name}.npy", plan[name])
    with (out / GAME_MOVES).open("w") as fh:
        for gid in sorted(moves_of):
            fh.write(json.dumps({"game_id": gid, "uci": moves_of[gid]}) + "\n")
    return {p.name: sha256_file(p) for p in sorted(out.iterdir())}


def materialise(collect_dir: Path, out_dir: Path, *, k: int = 2, min_visits: float = 2.0, seed: int = 0,
                holdout_frac: float = 0.05, shards: list[int] | None = None, log: Any = print,
                unit: str = "pairs") -> dict[str, Any]:
    """Write the sibling sidecar of a sealed collect into ``out_dir`` (must not exist).

    ``unit`` "pairs" (the registered sidecar, version 2): ``k`` sampled pairs per
    row. "roots" (list sidecar, version 1): every eligible child (>= ``min_visits``
    visits, not search-terminal) of every root that has at least two, one input
    each, with its completed Q (parent POV) and visits; ``k`` and ``seed`` unused.

    Every game of every shard is rebuilt (each root hash checked) BEFORE anything
    is written, so a collect that cannot be rebuilt leaves no sidecar.
    ``shards`` restricts the run to those shard indices (recorded; a partial
    sidecar is refused by the trainer).
    """
    t0 = time.perf_counter()
    if unit not in ("pairs", "roots"):
        raise ValueError(f"unit must be 'pairs' or 'roots', got {unit!r}")
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not float(min_visits) >= 1.0:
        raise ValueError(f"min_visits must be >= 1 (an unvisited child's q is the root's net value), got {min_visits}")
    if not 0.0 <= float(holdout_frac) < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a sidecar is written once")
    view = open_collect(collect_dir)
    stamp = apply_collect_regime(view)
    metas = _check_games_within_shards(view)
    chosen = list(range(len(view.shards))) if shards is None else sorted({int(s) for s in shards})
    plans = {}
    for si in chosen:
        plans[si] = _plan_shard(view, si, metas[si], k=int(k), min_visits=float(min_visits), seed=int(seed),
                                holdout_frac=float(holdout_frac), unit=unit)
        log(f"planned shard {si}: {plans[si][2]} ({time.perf_counter() - t0:.0f}s, peak RSS {_peak_rss_mib():.0f} MiB)")
    tmp = out_dir.with_name(f".{out_dir.name}.tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    records = {}
    for si in chosen:
        plan, moves_of, counts = plans.pop(si)
        files = _write_shard(view, tmp / view.shards[si].name.replace(".zarr", ""), metas[si], plan, moves_of, unit)
        records[view.shards[si].name] = {"index": si, "rows": len(metas[si]["game_id"]), "counts": counts,
                                         "files": files}
        what = "children" if unit == "roots" else "pairs"
        log(f"wrote shard {si}: {counts[what]} {what} "
            f"({time.perf_counter() - t0:.0f}s, "
            f"peak RSS {_peak_rss_mib():.0f} MiB)")
    totals = {key: int(sum(r["counts"][key] for r in records.values()))
              for key in ("rows", "rows_with_stats", "rows_with_pairs", "children" if unit == "roots" else "pairs",
                          "terminal_children_excluded", "repetition_children_excluded")}
    unit_cfg: dict[str, Any] = ({"k": None, "seed": None, "unit": "roots", "q_pov": "parent"} if unit == "roots" else
                                {"k": int(k), "seed": int(seed), "weight": "min(v_A, v_B)",
                                 "orientation": "A = more-visited child", "dq_pov": "parent"})
    manifest = {
        "version": LIST_SIDECAR_VERSION if unit == "roots" else SIDECAR_VERSION,
        "kind": LIST_KIND if unit == "roots" else PAIR_KIND,
        "collect_dir": str(view.collect_dir), "collect_manifest_sha256": view.manifest_sha256,
        "config": {**unit_cfg, "min_visits": float(min_visits), "holdout_frac": float(holdout_frac),
                   "max_bridge_plies": MAX_BRIDGE_PLIES, "max_bridge_unknown_first": MAX_BRIDGE_UNKNOWN_FIRST,
                   "excluded_children": "search terminal: CBoard is_game_over() or is_repetition() (2-fold)"},
        "encoding": view.enc, "history_rep_fix": stamp, "lines_source": view.lines_source,
        "move_source": "games.jsonl uci" if all("uci" in g for g in view.games.values())
        else "played_move_index + unique-x_hash bridging",
        "shards_total": len(view.shards), "shards_written": chosen, "complete": chosen == list(range(len(view.shards))),
        "shards": records, "totals": totals,
        "peak_rss_mib": _peak_rss_mib(), "wall_seconds": time.perf_counter() - t0,
    }
    (tmp / SIDECAR_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(out_dir)
    return manifest


def self_check(collect_dir: Path, *, games: int, seed: int = 0) -> dict[str, Any]:
    """Rebuild ``games`` games (seeded choice) and compare, byte for byte, every
    rebuilt root with its stored ``x`` and every PLAYED child with the next row's
    stored ``x`` (where the next row is the next ply). Returns the counts; the
    caller exits non-zero on any mismatch."""
    from chess_anti_engine.replay.shard import open_shard_arrays

    view = open_collect(collect_dir)
    stamp = apply_collect_regime(view)
    metas = _check_games_within_shards(view)
    owner = {int(g): si for si, m in enumerate(metas) for g in np.unique(m["game_id"])}
    ids = sorted(owner)
    pick = np.random.default_rng(int(seed)).choice(len(ids), size=min(int(games), len(ids)), replace=False)
    res: dict[str, Any] = {"games": 0, "roots": 0, "roots_byte_equal": 0, "children": 0, "children_byte_equal": 0,
           "mismatches": [], "history_rep_fix": stamp}
    for gi in sorted(pick.tolist()):
        gid = ids[gi]
        si = owner[gid]
        meta = metas[si]
        offs = _game_rows(meta)[gid]
        rows = _shard_rows(view.shards[si])
        with open_shard_arrays(view.shards[si], lazy=True, validate=True) as (arrs, _m):
            lo, hi = int(offs.min()), int(offs.max()) + 1
            xs = np.asarray(arrs["x"][lo:hi])[offs - lo]
        plies = np.asarray(meta["ply_index"])[offs]
        res["games"] += 1
        try:
            for i, board in rebuild_game(view, gid, plies, np.asarray(meta["x_hash"])[offs],
                                         rows["played_move_index"][offs], rows["has_played_move"][offs]):
                res["roots"] += 1
                if encode(board, view.enc).tobytes() == np.ascontiguousarray(xs[i], dtype=np.float16).tobytes():
                    res["roots_byte_equal"] += 1
                else:
                    res["mismatches"].append({"game": gid, "ply": int(plies[i]), "what": "root"})
                if i + 1 < len(plies) and int(plies[i + 1]) == int(plies[i]) + 1 and bool(rows["has_played_move"][offs[i]]):
                    from chess_anti_engine.moves.encode import index_to_move_for_encoding

                    m = index_to_move_for_encoding(int(rows["played_move_index"][offs[i]]), board)
                    board.push(m)
                    child = encode(board, view.enc)
                    board.pop()
                    res["children"] += 1
                    if child.tobytes() == np.ascontiguousarray(xs[i + 1], dtype=np.float16).tobytes():
                        res["children_byte_equal"] += 1
                    else:
                        res["mismatches"].append({"game": gid, "ply": int(plies[i]), "what": "played child"})
        except CollectNotRebuildable as exc:
            res["mismatches"].append({"game": gid, "what": "not rebuildable", "detail": str(exc)})
    res["ok"] = not res["mismatches"] and res["roots"] > 0
    return res


# ---------------------------------------------------------------------------
# Training term
# ---------------------------------------------------------------------------


@dataclass
class SidecarPairs:
    """The window's pairs: memmapped arrays per sidecar shard plus a key index."""
    parts: list[dict[str, np.ndarray]]
    part_of: np.ndarray      # pair -> part index
    local: np.ndarray        # pair -> index within its part
    dq: np.ndarray
    min_visits: np.ndarray
    holdout: np.ndarray
    gen: np.ndarray
    row_start: np.ndarray    # train row -> first pair (into the sorted table)
    row_count: np.ndarray    # train row -> number of pairs
    probe: np.ndarray        # pair ids of the fixed hold-out probe
    record: dict[str, Any]

    def child_inputs(self, pair_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xa = np.empty((len(pair_ids), 175, 8, 8), dtype=np.float16)
        xb = np.empty_like(xa)
        for p in np.unique(self.part_of[pair_ids]):
            sel = np.flatnonzero(self.part_of[pair_ids] == p)
            idx = self.local[pair_ids[sel]]
            xa[sel] = self.parts[int(p)]["x_a"][idx]
            xb[sel] = self.parts[int(p)]["x_b"][idx]
        return xa, xb


@dataclass
class SidecarLists:
    """The window's root lists: one entry per eligible child, contiguous per root
    (row), sorted by the learner's row key; memmapped child inputs per part."""
    parts: list[dict[str, np.ndarray]]
    part_of: np.ndarray      # child -> part index
    local: np.ndarray        # child -> index within its part
    q: np.ndarray            # completed Q, parent POV
    visits: np.ndarray
    holdout: np.ndarray
    gen: np.ndarray
    row_start: np.ndarray    # train row -> first child
    row_count: np.ndarray    # train row -> number of children (0 or >= 2)
    probe_roots: list[tuple[int, int]]   # (first child, count) of the fixed hold-out probe roots
    record: dict[str, Any]

    def child_inputs(self, ids: np.ndarray) -> np.ndarray:
        x = np.empty((len(ids), 175, 8, 8), dtype=np.float16)
        for p in np.unique(self.part_of[ids]):
            sel = np.flatnonzero(self.part_of[ids] == p)
            x[sel] = self.parts[int(p)]["x"][self.local[ids[sel]]]
        return x


def load_sidecars(run_dir: Path, gens: list[int], row_ids: np.ndarray, sidecar_root: Path, *,
                  holdout_frac: float, newest: int, probe_pairs: int = 1024, unit: str = "pairs",
                  probe_roots: int = 128) -> Any:
    """Open ``sidecar_root/gen_<g>/`` for each window generation and index its
    pairs (``unit`` "pairs": ``SidecarPairs``) or root lists ("roots":
    ``SidecarLists``) by the learner's row ids. Refuses a sidecar of the other unit,
    of another collect, a partial or changed one, another hold-out split, or an
    entry of a hold-out game on a training row."""
    if unit not in ("pairs", "roots"):
        raise ValueError(f"unit must be 'pairs' or 'roots', got {unit!r}")
    roots = unit == "roots"
    want_kind, want_version = (LIST_KIND, LIST_SIDECAR_VERSION) if roots else (PAIR_KIND, SIDECAR_VERSION)
    fields = LIST_FIELDS if roots else PAIR_FIELDS
    parts: list[dict[str, np.ndarray]] = []
    part_of, local, keys, dq, mv, held, gen_of = [], [], [], [], [], [], []
    record: dict[str, Any] = {"generations": {}}
    for g in gens:
        sdir = Path(sidecar_root) / f"gen_{g:04d}"
        mpath = sdir / SIDECAR_MANIFEST
        if not mpath.is_file():
            raise ValueError(f"--sibling-dir {sidecar_root}: no sidecar for generation {g} ({mpath} missing)")
        man = json.loads(mpath.read_text())
        if man.get("version") != want_version or man.get("kind") != want_kind:
            raise ValueError(f"{mpath}: not a version-{want_version} {want_kind} sidecar (--sibling-unit {unit}); it is "
                             f"{man.get('kind')} version {man.get('version')}")
        collect = resolve_collect_dir(Path(run_dir) / f"gen_{g:04d}" / "collect")
        if man["collect_manifest_sha256"] != sha256_file(collect / "manifest.json"):
            raise ValueError(f"{mpath}: made from another collect than generation {g}'s ({collect})")
        if not man.get("complete"):
            raise ValueError(f"{mpath}: partial sidecar (shards {man.get('shards_written')}); training needs all")
        if float(man["config"]["holdout_frac"]) != float(holdout_frac):
            raise ValueError(f"{mpath}: hold-out split {man['config']['holdout_frac']} != the learner's {holdout_frac}")
        stamp = man.get("history_rep_fix") or {}
        if len({stamp.get("collect_declared"), stamp.get("shards_declared"), stamp.get("process_realised")}) != 1:
            raise ValueError(f"{mpath}: history_rep_fix stamp disagrees ({stamp})")
        base = 0
        files_sha: dict[str, dict[str, str]] = {}
        for s in generation_shards(collect):
            n_rows = len(shard_row_meta(s)["game_id"])
            rec = man["shards"].get(s.name)
            if rec is None or int(rec["rows"]) != n_rows:
                raise ValueError(f"{mpath}: shard {s.name} missing or has another row count")
            pdir = sdir / s.name.replace(".zarr", "")
            for name, want in rec["files"].items():
                if sha256_file(pdir / name) != want:
                    raise ValueError(f"{pdir / name}: sha256 differs from the sidecar manifest")
            files_sha[s.name] = dict(rec["files"])
            # Only the child inputs are memory-mapped (two descriptors per shard); the
            # small per-pair arrays are read into RAM.
            part = {name: np.load(pdir / f"{name}.npy", mmap_mode="r" if name.startswith("x") else None)
                    for name in fields}
            gids = np.asarray(part["game_id"])
            flags = np.asarray(part["holdout"]).astype(bool)
            recomputed = np.asarray([is_holdout_game(int(x), float(holdout_frac)) for x in np.unique(gids)], bool)
            if not np.array_equal(recomputed, np.asarray([bool(flags[gids == x][0]) for x in np.unique(gids)], bool)):
                raise ValueError(f"{pdir}: hold-out flags differ from is_holdout_game at {holdout_frac}")
            pi = len(parts)
            parts.append(part)
            n = len(gids)
            part_of.append(np.full(n, pi, np.int32))
            local.append(np.arange(n, dtype=np.int64))
            keys.append((np.int64(g) << 32) | (base + np.asarray(part["row"], np.int64)))
            dq.append(np.asarray(part["q" if roots else "dq"], np.float32))
            mv.append(np.asarray(part["visits" if roots else "min_visits"], np.float32))
            held.append(flags)
            gen_of.append(np.full(n, g, np.int32))
            base += n_rows
        record["generations"][str(g)] = {"sidecar_dir": str(sdir), "manifest_sha256": sha256_file(mpath),
                                         "config": man["config"], "totals": man["totals"], "files": files_sha,
                                         "history_rep_fix": stamp}
    def cat(xs: list[np.ndarray], dt: Any) -> np.ndarray:
        return np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)

    keys_all = cat(keys, np.int64)
    order = np.argsort(keys_all, kind="stable")
    keys_sorted = keys_all[order]
    rk = (np.asarray(row_ids[:, 0], np.int64) << 32) | np.asarray(row_ids[:, 1], np.int64)
    lo = np.searchsorted(keys_sorted, rk, side="left")
    hi = np.searchsorted(keys_sorted, rk, side="right")
    held_sorted = cat(held, np.bool_)[order]
    trained = np.zeros(len(order), bool)
    for a, b in zip(lo[hi > lo], hi[hi > lo], strict=True):
        trained[a:b] = True
    if bool((trained & held_sorted).any()):
        raise ValueError("a training row carries a pair flagged hold-out; the sidecar and the learner disagree")
    gen_sorted = cat(gen_of, np.int32)[order]
    if roots:
        if bool(((hi - lo) == 1).any()):
            raise ValueError("a training row carries a single list child; a list sidecar has 0 or >= 2 per root")
        cand = np.flatnonzero(held_sorted & (gen_sorted == int(newest)))
        probe_list: list[tuple[int, int]] = []
        if cand.size:
            ck = keys_sorted[cand]
            starts = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
            ends = np.r_[starts[1:], ck.size]
            probe_list = [(int(cand[a]), int(b - a)) for a, b in zip(starts, ends, strict=True)][: int(probe_roots)]
        return SidecarLists(parts=parts, part_of=cat(part_of, np.int32)[order], local=cat(local, np.int64)[order],
                            q=cat(dq, np.float32)[order], visits=cat(mv, np.float32)[order], holdout=held_sorted,
                            gen=gen_sorted, row_start=lo.astype(np.int64), row_count=(hi - lo).astype(np.int64),
                            probe_roots=probe_list, record=record)
    probe = np.flatnonzero(held_sorted & (gen_sorted == int(newest)))[: int(probe_pairs)]
    return SidecarPairs(parts=parts, part_of=cat(part_of, np.int32)[order], local=cat(local, np.int64)[order],
                        dq=cat(dq, np.float32)[order], min_visits=cat(mv, np.float32)[order], holdout=held_sorted,
                        gen=gen_sorted, row_start=lo.astype(np.int64), row_count=(hi - lo).astype(np.int64),
                        probe=probe, record=record)


def child_values(wdl_logits: Any) -> Any:
    """Parent-POV value of a child from the child's own WDL logits: L - W."""
    import torch

    p = torch.softmax(wdl_logits.float(), dim=-1)
    return p[:, 2] - p[:, 0]


def sibling_loss(model: Any, xa: Any, xb: Any, dq: Any, min_visits: Any) -> tuple[Any, Any]:
    """(per-pair weighted Huber, per-pair residual dv - dq). One forward over A and B."""
    import torch

    out = model(torch.cat([xa, xb], dim=0))
    v = child_values(out["wdl"])
    n = int(xa.shape[0])
    resid = (v[:n] - v[n:]) - dq
    huber = torch.nn.functional.huber_loss(resid, torch.zeros_like(resid), delta=HUBER_DELTA, reduction="none")
    return (min_visits / VISIT_SCALE) * huber, resid


def order_pair_loss(dv: Any, dq: Any, visit_weight: Any, *, margin: float, min_dq: float) -> Any:
    """The ORDER form per pair (version 2), given the net's dv = v_A - v_B:

    ``visit_weight * 1[|dq| >= min_dq] * min(|dq|, ORDER_GAP_CAP) * relu(m_i - sign(dq) * dv)``,
    ``m_i = min(margin, |dq|)``. Zero, with zero gradient, once the net orders the
    pair as the search does by ``m_i``; a mis-ordered large gap costs more than a
    small one, in proportion to the value given up (bounded)."""
    import torch

    gap = dq.abs().to(dv.dtype)
    clear = (gap >= min_dq).to(dv.dtype)
    m_i = torch.clamp(gap, max=margin)   # min(margin, |dq|): never ask for more than the search
    rank = torch.relu(m_i - torch.sign(dq).to(dv.dtype) * dv)
    return visit_weight * clear * torch.clamp(gap, max=ORDER_GAP_CAP) * rank


def sibling_order_loss(model: Any, xa: Any, xb: Any, dq: Any, min_visits: Any, *, margin: float,
                       min_dq: float) -> tuple[Any, Any]:
    """The ORDER form on pairs: (per-pair weighted loss, per-pair residual dv - dq).

    Per pair ``(min_visits / 64) * min(|dq|, 0.62) * relu(m_i - sign(dq) * dv)`` with
    ``m_i = min(margin, |dq|)``, if ``|dq| >= min_dq``, else 0 (``order_pair_loss``): zero (and zero
    gradient) once the net orders the pair as the search does by at least the smaller
    of ``margin`` and the search's own gap. The cap is part of the form: the loss never
    asks the net to separate a pair by more than the search separates it (no
    sharpening pressure without a target behind it). Only the ORDER of the search's
    values is a target, not the size of their difference beyond ``m_i``. Same pairs
    and the same single forward over A and B as ``sibling_loss``; the residual is
    returned for the same diagnostics."""
    import torch

    out = model(torch.cat([xa, xb], dim=0))
    v = child_values(out["wdl"])
    n = int(xa.shape[0])
    dv = v[:n] - v[n:]
    return order_pair_loss(dv, dq, min_visits / VISIT_SCALE, margin=margin, min_dq=min_dq), dv - dq


def root_pairs(q: np.ndarray, min_dq: float) -> tuple[np.ndarray, np.ndarray]:
    """Local index pairs (i, j), i < j, among one root's children with |q_i - q_j| >= min_dq."""
    ia, ib = np.triu_indices(len(q), 1)
    keep = np.abs(q[ia] - q[ib]) >= float(min_dq)
    return ia[keep], ib[keep]


def root_weight(n_pairs: int) -> float:
    """n_eff of a root's mean over its n eligible pairs at the fixed within-root correlation."""
    return 0.0 if n_pairs <= 0 else n_pairs / (1.0 + (n_pairs - 1) * ROOT_PAIR_ICC)


SIBLING_LOSS_FORMS = ("diff", "order")
SIBLING_COUNTERS = ("steps", "pairs", "loss_sum", "abs_resid_sum", "per_step", "last_cursor")


class SiblingTerm:
    """The sibling penalty on the Trainer's penalty hook.

    Each call reads the rows the ScheduledSampler served for THIS batch
    (``served[-1]``). It refuses a sampler that ran ahead: the hook must see the
    cursor advance by exactly one per call. It takes the first ``rows`` of them
    that carry pairs, in the batch's shuffled order, and forwards both children of
    every pair through the trainer's model in its current (train) mode. Train mode
    changes no output here: the model has no BatchNorm and no dropout, and the
    trainer's channel-dropout / balance knobs are 0 in simple_rl. Mirroring
    applies to the root batch only; children are presented unmirrored.
    """

    def __init__(self, model: Any, sampler: Any, pairs: SidecarPairs, *, w: float, rows: int, chunk_pairs: int,
                 batch_size: int, device: str, loss: str = "diff", order_margin: float | None = None,
                 order_min_dq: float | None = None) -> None:
        if loss not in SIBLING_LOSS_FORMS:
            raise ValueError(f"sibling loss must be one of {SIBLING_LOSS_FORMS}, got {loss!r}")
        if loss == "order":
            if order_margin is None or not math.isfinite(float(order_margin)) or float(order_margin) < 0:
                raise ValueError(f"sibling loss 'order' needs a finite order_margin >= 0, got {order_margin}")
            if order_min_dq is None or not math.isfinite(float(order_min_dq)) or float(order_min_dq) <= 0:
                raise ValueError(f"sibling loss 'order' needs a finite order_min_dq > 0, got {order_min_dq}")
        elif order_margin is not None or order_min_dq is not None:
            raise ValueError("order_margin / order_min_dq are read only with sibling loss 'order'")
        if int(chunk_pairs) < 1:
            raise ValueError(f"sibling chunk_pairs must be >= 1, got {chunk_pairs}")
        if int(batch_size) < 1:
            raise ValueError(f"sibling batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.sampler = sampler
        self.pairs = pairs
        self.w = float(w)
        self.rows = int(rows)
        self.chunk_pairs = int(chunk_pairs)
        self.chunk_segments: list[list[int]] = [[0, self.chunk_pairs]]   # [first step, chunk_pairs] per segment
        self.batch_size = int(batch_size)
        self.device = device
        self.loss = loss
        self.order_margin = None if order_margin is None else float(order_margin)
        self.order_min_dq = None if order_min_dq is None else float(order_min_dq)
        self.active = 0   # pairs with a non-zero loss (the order form's pairs still to be ordered)
        self.unit = "pairs"
        self.child_positions = 0   # child inputs forwarded with grad (the equal-budget check across units)
        self.child_positions_from_step = 0   # the term step at which child_positions began counting
        self.steps = 0
        self.pairs_seen = 0
        self.loss_sum = 0.0
        self.abs_resid_sum = 0.0
        self.per_step: list[list[float]] = []
        self.last_cursor = int(getattr(sampler, "cursor", 0))

    def _batch_pairs(self) -> np.ndarray:
        cursor = int(self.sampler.cursor)
        if cursor != self.last_cursor + 1:
            raise RuntimeError(f"sibling term: sampler cursor moved {self.last_cursor} -> {cursor} between penalty calls; "
                               "the served rows are not this batch's (prefetch or a second consumer)")
        self.last_cursor = cursor
        own = np.asarray(self.sampler.served[-1], dtype=np.int64)
        have = own[self.pairs.row_count[own] > 0][: self.rows]
        if not have.size:
            return np.zeros(0, np.int64)
        return np.concatenate([np.arange(self.pairs.row_start[r], self.pairs.row_start[r] + self.pairs.row_count[r])
                               for r in have]).astype(np.int64)

    def _tensors(self, ids: np.ndarray) -> tuple[Any, Any, Any, Any]:
        import torch

        xa, xb = self.pairs.child_inputs(ids)
        dev = self.device
        return (torch.from_numpy(xa).to(dev).float(), torch.from_numpy(xb).to(dev).float(),
                torch.from_numpy(self.pairs.dq[ids]).to(dev), torch.from_numpy(self.pairs.min_visits[ids]).to(dev))

    def _row_scale(self, batch: dict[str, Any]) -> float:
        """The Trainer's factor on what this hook returns, applied to the gradients
        taken inside it: ``rows / batch_size`` for a sampler with ``stored_row_quota``
        (the exact-once ScheduledSampler, whose last batch is short), else refused."""
        if not getattr(self.sampler, "stored_row_quota", False) or getattr(self.sampler, "exact_without_replacement",
                                                                           False):
            raise RuntimeError("sibling term: the Trainer's per-batch loss scale is only reproduced for the "
                               "exact-once ScheduledSampler (stored_row_quota)")
        rows = int(batch["x"].shape[0])
        if not 1 <= rows <= self.batch_size:
            raise RuntimeError(f"sibling term: the batch holds {rows} rows, outside [1, batch_size {self.batch_size}]")
        return rows / self.batch_size

    def _pair_loss(self, xa: Any, xb: Any, dq: Any, min_visits: Any) -> tuple[Any, Any]:
        if self.loss == "order":
            assert self.order_margin is not None
            assert self.order_min_dq is not None
            return sibling_order_loss(self.model, xa, xb, dq, min_visits, margin=self.order_margin,
                                      min_dq=self.order_min_dq)
        return sibling_loss(self.model, xa, xb, dq, min_visits)

    def _require_cleared_grads(self) -> None:
        """The chunk backwards add into ``.grad`` before the Trainer's own backward;
        gradients left over from anything else would be summed into this update."""
        stale = [n for n, p in self.model.named_parameters() if p.grad is not None]
        if stale:
            raise RuntimeError(f"sibling term: {len(stale)} parameter gradients were not cleared before the term's "
                               f"backward (first: {stale[0]}); the Trainer must zero_grad before the hook")

    def _zero_leaf(self) -> Any:
        """A 0-valued leaf with no parameter behind it: the Trainer adds it to its root loss and calls
        backward, which stays defined when the term is the only loss (every other weight 0)."""
        import torch

        return torch.zeros((), device=self.device, requires_grad=True)

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        del outputs
        ids = self._batch_pairs()
        self.steps += 1
        if not ids.size:   # no pair this step: still a 0-valued leaf, so a sibling-only step stays well-defined
            self.per_step.append([0, 0.0])
            return self._zero_leaf()
        scale = self._row_scale(batch)
        self._require_cleared_grads()
        n = int(ids.size)
        coef = scale * self.w / n
        weighted_sum = abs_resid_sum = active_sum = None
        for start in range(0, n, self.chunk_pairs):
            weighted, resid = self._pair_loss(*self._tensors(ids[start:start + self.chunk_pairs]))
            (weighted.sum() * coef).backward()
            w_part, r_part = weighted.detach().sum(), resid.detach().abs().sum()
            active_part = (weighted.detach() > 0).sum()
            active_sum = active_part if active_sum is None else active_sum + active_part
            weighted_sum = w_part if weighted_sum is None else weighted_sum + w_part
            abs_resid_sum = r_part if abs_resid_sum is None else abs_resid_sum + r_part
            del weighted, resid, w_part, r_part, active_part   # this chunk's graph is gone before the next forward
        assert weighted_sum is not None
        assert abs_resid_sum is not None
        assert active_sum is not None
        total = float(weighted_sum)
        self.pairs_seen += n
        self.child_positions += 2 * n
        self.loss_sum += total
        self.abs_resid_sum += float(abs_resid_sum)
        self.active += int(active_sum)
        self.per_step.append([n, total / n])
        # Its gradient is already in .grad. A 0-valued leaf (no parameter behind it) keeps the Trainer's
        # ``backward`` well-defined when this term is the only loss (every other weight 0).
        import torch

        return torch.zeros((), device=weighted_sum.device, requires_grad=True)

    def probe_abs_resid(self) -> dict[str, Any]:
        """mean |dv - dq| on the fixed hold-out probe, model in eval mode, no grad."""
        import torch

        ids = self.pairs.probe
        if not ids.size:
            return {"pairs": 0, "mean_abs_resid": None, "sign_agree": None, "sign_pairs": 0}
        was = self.model.training
        self.model.eval()
        tot = 0.0
        agree = 0
        n_sign = 0
        t = self.order_min_dq or 0.0
        with torch.no_grad():
            for s in range(0, ids.size, 256):
                xa, xb, dq, mv = self._tensors(ids[s:s + 256])
                _, resid = sibling_loss(self.model, xa, xb, dq, mv)
                tot += float(resid.abs().sum())
                dv = resid + dq
                sel = (dq.abs() >= t) & (dq != 0)
                agree += int((torch.sign(dv[sel]) == torch.sign(dq[sel])).sum())
                n_sign += int(sel.sum())
        self.model.train(was)
        return {"pairs": int(ids.size), "mean_abs_resid": tot / ids.size,
                "sign_agree": agree / n_sign if n_sign else None, "sign_pairs": n_sign}

    def state(self) -> dict[str, Any]:
        return {"steps": self.steps, "pairs": self.pairs_seen, "loss_sum": self.loss_sum,
                "abs_resid_sum": self.abs_resid_sum, "per_step": list(self.per_step), "last_cursor": self.last_cursor,
                "chunk_pairs": self.chunk_pairs, "chunk_segments": [list(x) for x in self.chunk_segments],
                "active": self.active, "loss": self.loss,
                "order_margin": self.order_margin, "order_min_dq": self.order_min_dq, "unit": self.unit,
                "child_positions": self.child_positions, "child_positions_from_step": self.child_positions_from_step,
                "order_form_version": ORDER_FORM_VERSION if self.loss == "order" else None}

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("unit", "pairs") != self.unit:
            raise RuntimeError(f"sibling term: the snapshot was taken at unit {state.get('unit', 'pairs')!r}, this "
                               f"phase runs {self.unit!r}")
        if self.loss == "order" and state.get("order_form_version") != ORDER_FORM_VERSION:
            raise RuntimeError(f"sibling term: the snapshot's order form is version {state.get('order_form_version')}, "
                               f"this code trains version {ORDER_FORM_VERSION}")
        for key, mine in (("loss", self.loss), ("order_margin", self.order_margin), ("order_min_dq", self.order_min_dq)):
            if key in state and state[key] != mine:
                raise RuntimeError(f"sibling term: the snapshot was taken at {key} {state[key]!r}, this phase runs "
                                   f"{mine!r}")
        self.steps = int(state["steps"])
        self.pairs_seen = int(state["pairs"])
        self.loss_sum = float(state["loss_sum"])
        self.abs_resid_sum = float(state["abs_resid_sum"])
        self.per_step = [list(x) for x in state["per_step"]]
        self.last_cursor = int(state["last_cursor"])
        self.active = int(state.get("active", 0))
        # chunk_pairs changes memory, not the update: a phase may resume at another value (OOM recovery). Each
        # value is recorded with the step it started at.
        segs = [list(x) for x in state.get("chunk_segments", [[0, int(state.get("chunk_pairs", self.chunk_pairs))]])]
        if segs[-1][1] != self.chunk_pairs:
            segs.append([self.steps, self.chunk_pairs])
        self.chunk_segments = segs
        # A snapshot from before the field counted nothing for its steps: count from here on, never divide the new
        # positions by the old steps.
        if "child_positions" in state:
            self.child_positions = int(state["child_positions"])
            self.child_positions_from_step = int(state.get("child_positions_from_step", 0))
        else:
            self.child_positions, self.child_positions_from_step = 0, self.steps

    def summary(self) -> dict[str, Any]:
        n = self.pairs_seen
        term_steps = sum(1 for r in self.per_step[self.child_positions_from_step:] if r[0])
        return {"w": self.w, "sibling_rows": self.rows, "chunk_pairs": self.chunk_pairs,
                "chunk_segments": [list(x) for x in self.chunk_segments], "loss": self.loss,
                "order_margin": self.order_margin, "order_min_dq": self.order_min_dq, "active_pairs": self.active,
                "unit": self.unit, "order_form_version": ORDER_FORM_VERSION if self.loss == "order" else None,
                "child_positions": self.child_positions, "child_positions_from_step": self.child_positions_from_step,
                "child_positions_per_step": self.child_positions / term_steps if term_steps else None,
                "steps": self.steps, "pairs": n,
                "mean_weighted_huber": self.loss_sum / n if n else None,   # kept for diff-era readers
                "mean_weighted_loss": self.loss_sum / n if n else None,    # form-neutral: Huber (diff) or ranking (order)
                "mean_abs_resid_train": self.abs_resid_sum / n if n else None,
                "per_step": self.per_step}


class SiblingListTerm(SiblingTerm):
    """The ORDER form on ROOT LISTS (``--sibling-unit roots``).

    Each call takes the first ``rows`` roots of THIS batch that carry a list (every
    eligible child: >= the sidecar's min visits, not search-terminal), forwards every
    child once, and applies the ORDER form (``order_pair_loss``) to EVERY pair
    among a root's children with ``|dq| >= order_min_dq``. A root's loss is the mean
    over its eligible pairs; roots are combined with weight n_eff (``root_weight``,
    rho fixed) normalised over the step's roots. Chunks hold whole roots (pairs never
    cross roots), at most ``2 * chunk_pairs`` child positions unless one root is
    larger, with a backward per chunk, as in pair mode."""

    def __init__(self, model: Any, sampler: Any, lists: SidecarLists, *, w: float, rows: int, chunk_pairs: int,
                 batch_size: int, device: str, loss: str = "order", order_margin: float | None = None,
                 order_min_dq: float | None = None) -> None:
        if loss != "order":
            raise ValueError("the root-list unit trains the order form only (--sibling-loss order)")
        as_pairs: Any = lists   # the base reads only the shared row index (row_start / row_count)
        super().__init__(model, sampler, as_pairs, w=w, rows=rows, chunk_pairs=chunk_pairs, batch_size=batch_size,
                         device=device, loss=loss, order_margin=order_margin, order_min_dq=order_min_dq)
        self.lists = lists
        self.unit = "roots"
        self.roots_seen = 0

    def _batch_roots(self) -> np.ndarray:
        cursor = int(self.sampler.cursor)
        if cursor != self.last_cursor + 1:
            raise RuntimeError(f"sibling term: sampler cursor moved {self.last_cursor} -> {cursor} between penalty calls; "
                               "the served rows are not this batch's (prefetch or a second consumer)")
        self.last_cursor = cursor
        own = np.asarray(self.sampler.served[-1], dtype=np.int64)
        return own[self.lists.row_count[own] > 0][: self.rows]

    def _root_plan(self, rows: np.ndarray) -> list[dict[str, Any]]:
        return self._root_plan_spans([(int(self.lists.row_start[r]), int(self.lists.row_count[r])) for r in rows])

    def _chunk_loss(self, roots: list[dict[str, Any]], z: float) -> tuple[Any, Any, Any]:
        """(sum over the chunk's pairs of a_r / (Z n_r) * pair loss, per-pair loss, per-pair dv - dq)."""
        import torch

        assert self.order_margin is not None
        assert self.order_min_dq is not None
        ids = np.concatenate([np.arange(r["start"], r["start"] + r["count"]) for r in roots]).astype(np.int64)
        x = torch.from_numpy(self.lists.child_inputs(ids)).to(self.device).float()
        v = child_values(self.model(x)["wdl"])
        offs = np.cumsum([0] + [r["count"] for r in roots[:-1]])
        pa = torch.from_numpy(np.concatenate([o + r["ia"] for o, r in zip(offs, roots, strict=True)])).to(self.device)
        pb = torch.from_numpy(np.concatenate([o + r["ib"] for o, r in zip(offs, roots, strict=True)])).to(self.device)
        dq = torch.from_numpy(np.concatenate([r["dq"] for r in roots])).to(self.device)
        wv = torch.from_numpy(np.concatenate([r["wv"] for r in roots])).to(self.device)
        coef = torch.from_numpy(np.concatenate([np.full(r["ia"].size, r["a"] / (z * r["ia"].size), np.float32)
                                                for r in roots])).to(self.device)
        dv = v[pa] - v[pb]
        pl = order_pair_loss(dv, dq, wv, margin=self.order_margin, min_dq=self.order_min_dq)
        return (coef * pl).sum(), pl, dv - dq

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        del outputs
        plan = self._root_plan(self._batch_roots())
        self.steps += 1
        z = float(sum(r["a"] for r in plan))
        if not plan or z <= 0:   # no eligible root this step: the same 0-valued leaf as the pair unit
            self.per_step.append([0, 0.0])
            return self._zero_leaf()
        scale = self._row_scale(batch)
        self._require_cleared_grads()
        budget = 2 * self.chunk_pairs
        chunks: list[list[dict[str, Any]]] = [[]]
        used = 0
        for r in plan:
            if chunks[-1] and used + r["count"] > budget:
                chunks.append([])
                used = 0
            chunks[-1].append(r)
            used += r["count"]
        step_value = abs_resid_sum = active_sum = None
        for ch in chunks:
            total, pl, resid = self._chunk_loss(ch, z)
            (total * (scale * self.w)).backward()
            parts = (total.detach(), resid.detach().abs().sum(), (pl.detach() > 0).sum())
            step_value = parts[0] if step_value is None else step_value + parts[0]
            abs_resid_sum = parts[1] if abs_resid_sum is None else abs_resid_sum + parts[1]
            active_sum = parts[2] if active_sum is None else active_sum + parts[2]
            del total, pl, resid, parts   # this chunk's graph is gone before the next forward
        assert step_value is not None
        assert abs_resid_sum is not None
        assert active_sum is not None
        n_pairs = int(sum(r["ia"].size for r in plan))
        value = float(step_value)
        self.pairs_seen += n_pairs
        self.roots_seen += len(plan)
        self.child_positions += int(sum(r["count"] for r in plan))
        self.loss_sum += value
        self.abs_resid_sum += float(abs_resid_sum)
        self.active += int(active_sum)
        self.per_step.append([n_pairs, value])
        import torch

        return torch.zeros((), device=step_value.device, requires_grad=True)

    def probe_abs_resid(self) -> dict[str, Any]:
        """On the fixed hold-out probe roots (model in eval mode, no grad), over every
        pair with |dq| >= order_min_dq: mean |dv - dq| and the rate at which the
        net orders the pair as the search does."""
        import torch

        if not self.lists.probe_roots:
            return {"roots": 0, "pairs": 0, "mean_abs_resid": None, "sign_agree": None}
        was = self.model.training
        self.model.eval()
        tot = 0.0
        agree = 0
        n = 0
        roots = self._root_plan_spans(self.lists.probe_roots)
        with torch.no_grad():
            for s in range(0, len(roots), 32):
                ch = roots[s:s + 32]
                if not ch:
                    continue
                _, _pl, resid = self._chunk_loss(ch, 1.0)
                dq = torch.from_numpy(np.concatenate([r["dq"] for r in ch]))
                dv = resid.cpu() + dq
                tot += float(resid.abs().sum())
                agree += int((torch.sign(dv) == torch.sign(dq)).sum())
                n += int(dq.numel())
        self.model.train(was)
        return {"roots": len(roots), "pairs": n, "mean_abs_resid": tot / n if n else None,
                "sign_agree": agree / n if n else None}

    def _root_plan_spans(self, spans: list[tuple[int, int]]) -> list[dict[str, Any]]:
        assert self.order_min_dq is not None
        out = []
        for st, c in spans:
            q, vis = self.lists.q[st:st + c], self.lists.visits[st:st + c]
            ia, ib = root_pairs(q, self.order_min_dq)
            if ia.size:
                out.append({"start": st, "count": c, "ia": ia, "ib": ib, "dq": (q[ia] - q[ib]).astype(np.float32),
                            "wv": (np.minimum(vis[ia], vis[ib]) / VISIT_SCALE).astype(np.float32),
                            "a": root_weight(int(ia.size))})
        return out

    def state(self) -> dict[str, Any]:
        return {**super().state(), "roots": self.roots_seen}

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.roots_seen = int(state.get("roots", 0))

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        term_steps = sum(1 for r in self.per_step if r[0])
        mean_step = self.loss_sum / term_steps if term_steps else None
        out.update({"roots": self.roots_seen, "mean_step_loss": mean_step, "mean_weighted_loss": mean_step,
                    "mean_weighted_huber": None, "root_weight": f"n / (1 + (n - 1) * {ROOT_PAIR_ICC})",
                    "loss_note": "roots: mean_weighted_loss is the mean per-step term value (per-root means, "
                                 "n_eff-weighted over roots)"})
        return out


class PenaltyChain:
    """Several penalties on the Trainer's single penalty hook, summed."""

    def __init__(self, parts: list[Any]) -> None:
        self.parts = parts

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        total = None
        for p in self.parts:
            v = p.penalty(outputs, batch)
            if v is not None:
                total = v if total is None else total + v
        return total

