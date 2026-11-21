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
``min_visits`` visits whose move does not end the game. Pairs are drawn without
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

POV algebra: a child is the position AFTER the parent's move, so the child's
side to move is the parent's OPPONENT. The head outputs ``(W, D, L)`` for the
child's side to move, and the parent's value of the child is
``W_parent - L_parent = L_child - W_child``. ``root_child_q`` is in the parent's
POV, so ``v(child) = softmax(wdl)[2] - softmax(wdl)[0]``.
"""
from __future__ import annotations

import json
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

SIDECAR_VERSION = 1
SIDECAR_MANIFEST = "sibling_manifest.json"
GAME_MOVES = "game_moves.jsonl"
MAX_BRIDGE_PLIES = 3          # a pilot gap needed at most 3 plies with the first move known
MAX_BRIDGE_UNKNOWN_FIRST = 2  # ...and at most 2 when the row stored no played move
VISIT_SCALE = 64.0            # the loss weight is min(v_A, v_B) / 64 (the collection's sims)
HUBER_DELTA = 0.2
PAIR_FIELDS: dict[str, Any] = {
    "x_a": np.float16, "x_b": np.float16, "dq": np.float32, "q_a": np.float32, "q_b": np.float32,
    "visits_a": np.float32, "visits_b": np.float32, "min_visits": np.float32,
    "action_a": np.int32, "action_b": np.int32, "game_id": np.int64, "ply_index": np.int32,
    "row": np.int32, "holdout": np.bool_,
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


def _plan_shard(view: CollectView, shard_index: int, meta: dict[str, np.ndarray], *, k: int, min_visits: float,
                seed: int, holdout_frac: float) -> tuple[dict[str, np.ndarray], dict[int, list[str]], dict[str, int]]:
    """Pass 1: rebuild every game of the shard (every root hash checked), choose
    each row's pairs, and record each game's moves as UCI for pass 2. The plan is
    in (game, ply) order -- the order pass 2 writes -- as compact columns."""
    from chess_anti_engine.moves.encode import index_to_move_for_encoding
    rows = _shard_rows(view.shards[shard_index])
    plan: list[tuple[Any, ...]] = []
    moves_of: dict[int, list[str]] = {}
    counts = {"rows": len(meta["game_id"]), "rows_with_stats": 0, "rows_with_pairs": 0, "pairs": 0,
              "terminal_children_excluded": 0}
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
                over = board.is_game_over(claim_draw=False)
                board.pop()
                if over:
                    counts["terminal_children_excluded"] += 1
                else:
                    cand.append(int(c))
            pairs = choose_pairs(np.asarray(cand), visits, k, pair_rng(seed, gid, int(plies[i])))
            if pairs:
                counts["rows_with_pairs"] += 1
            q = rows["root_child_q"][off]
            for a, b in pairs:
                plan.append((off, gid, int(plies[i]), a, b, float(q[a]), float(q[b]), float(visits[a]),
                             float(visits[b]), held))
        if board is not None:
            start = _start_board(view, view.games[gid])
            moves_of[gid] = [m.uci() for m in board.move_stack[len(start.move_stack):]]
    counts["pairs"] = len(plan)
    cols = list(zip(*plan, strict=True)) if plan else [() for _ in _PLAN_COLUMNS]
    out = {name: np.asarray(col, dtype=dt) for (name, dt), col in zip(_PLAN_COLUMNS.items(), cols, strict=True)}
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
                 moves_of: dict[int, list[str]]) -> dict[str, str]:
    """Pass 2: replay each game's recorded moves and encode the chosen children,
    written sequentially in plan order (bounded RAM)."""
    from chess_anti_engine.moves.encode import index_to_move_for_encoding

    out.mkdir(parents=True, exist_ok=False)
    n = int(plan["dq"].size)
    fa, fb = _npy_writer(out / "x_a.npy", n), _npy_writer(out / "x_b.npy", n)
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
            for c, fh in ((plan["action_a"][j], fa), (plan["action_b"][j], fb)):
                board.push(index_to_move_for_encoding(int(c), board))
                fh.write(np.ascontiguousarray(encode(board, view.enc)).tobytes())
                board.pop()
    finally:
        fa.close()
        fb.close()
    for name in ("row", "game_id", "ply_index", "action_a", "action_b", "q_a", "q_b", "visits_a", "visits_b",
                 "holdout", "dq", "min_visits"):
        np.save(out / f"{name}.npy", plan[name])
    with (out / GAME_MOVES).open("w") as fh:
        for gid in sorted(moves_of):
            fh.write(json.dumps({"game_id": gid, "uci": moves_of[gid]}) + "\n")
    return {p.name: sha256_file(p) for p in sorted(out.iterdir())}


def materialise(collect_dir: Path, out_dir: Path, *, k: int = 2, min_visits: float = 2.0, seed: int = 0,
                holdout_frac: float = 0.05, shards: list[int] | None = None, log: Any = print) -> dict[str, Any]:
    """Write the sibling sidecar of a sealed collect into ``out_dir`` (must not exist).

    Every game of every shard is rebuilt (each root hash checked) BEFORE anything
    is written, so a collect that cannot be rebuilt leaves no sidecar.
    ``shards`` restricts the run to those shard indices (recorded; a partial
    sidecar is refused by the trainer).
    """
    t0 = time.perf_counter()
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
                                holdout_frac=float(holdout_frac))
        log(f"planned shard {si}: {plans[si][2]} ({time.perf_counter() - t0:.0f}s, peak RSS {_peak_rss_mib():.0f} MiB)")
    tmp = out_dir.with_name(f".{out_dir.name}.tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    records = {}
    for si in chosen:
        plan, moves_of, counts = plans.pop(si)
        files = _write_shard(view, tmp / view.shards[si].name.replace(".zarr", ""), metas[si], plan, moves_of)
        records[view.shards[si].name] = {"index": si, "rows": len(metas[si]["game_id"]), "counts": counts,
                                         "files": files}
        log(f"wrote shard {si}: {counts['pairs']} pairs ({time.perf_counter() - t0:.0f}s, "
            f"peak RSS {_peak_rss_mib():.0f} MiB)")
    totals = {key: int(sum(r["counts"][key] for r in records.values()))
              for key in ("rows", "rows_with_stats", "rows_with_pairs", "pairs", "terminal_children_excluded")}
    manifest = {
        "version": SIDECAR_VERSION, "kind": "simple_rl_sibling_pairs",
        "collect_dir": str(view.collect_dir), "collect_manifest_sha256": view.manifest_sha256,
        "config": {"k": int(k), "min_visits": float(min_visits), "seed": int(seed), "holdout_frac": float(holdout_frac),
                   "weight": "min(v_A, v_B)", "orientation": "A = more-visited child", "dq_pov": "parent",
                   "max_bridge_plies": MAX_BRIDGE_PLIES, "max_bridge_unknown_first": MAX_BRIDGE_UNKNOWN_FIRST},
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


def load_sidecars(run_dir: Path, gens: list[int], row_ids: np.ndarray, sidecar_root: Path, *,
                  holdout_frac: float, newest: int, probe_pairs: int = 1024) -> SidecarPairs:
    """Open ``sidecar_root/gen_<g>/`` for each window generation and index its
    pairs by the learner's row ids. Refuses a sidecar of another collect, a partial
    or changed one, another hold-out split, or a pair of a hold-out game on a
    training row."""
    parts: list[dict[str, np.ndarray]] = []
    part_of, local, keys, dq, mv, held, gen_of = [], [], [], [], [], [], []
    record: dict[str, Any] = {"generations": {}}
    for g in gens:
        sdir = Path(sidecar_root) / f"gen_{g:04d}"
        mpath = sdir / SIDECAR_MANIFEST
        if not mpath.is_file():
            raise ValueError(f"--sibling-dir {sidecar_root}: no sidecar for generation {g} ({mpath} missing)")
        man = json.loads(mpath.read_text())
        if man.get("version") != SIDECAR_VERSION or man.get("kind") != "simple_rl_sibling_pairs":
            raise ValueError(f"{mpath}: not a version-{SIDECAR_VERSION} sibling sidecar")
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
            part = {name: np.load(pdir / f"{name}.npy", mmap_mode="r" if name in ("x_a", "x_b") else None)
                    for name in PAIR_FIELDS}
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
            dq.append(np.asarray(part["dq"], np.float32))
            mv.append(np.asarray(part["min_visits"], np.float32))
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

    def __init__(self, model: Any, sampler: Any, pairs: SidecarPairs, *, w: float, rows: int, device: str) -> None:
        self.model = model
        self.sampler = sampler
        self.pairs = pairs
        self.w = float(w)
        self.rows = int(rows)
        self.device = device
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

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        del outputs, batch
        ids = self._batch_pairs()
        self.steps += 1
        if not ids.size:
            self.per_step.append([0, 0.0])
            return None
        weighted, resid = sibling_loss(self.model, *self._tensors(ids))
        loss = weighted.mean()
        self.pairs_seen += int(ids.size)
        self.loss_sum += float(weighted.detach().sum())
        self.abs_resid_sum += float(resid.detach().abs().sum())
        self.per_step.append([int(ids.size), float(loss.detach())])
        return self.w * loss

    def probe_abs_resid(self) -> dict[str, Any]:
        """mean |dv - dq| on the fixed hold-out probe, model in eval mode, no grad."""
        import torch

        ids = self.pairs.probe
        if not ids.size:
            return {"pairs": 0, "mean_abs_resid": None}
        was = self.model.training
        self.model.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, ids.size, 256):
                _, resid = sibling_loss(self.model, *self._tensors(ids[s:s + 256]))
                tot += float(resid.abs().sum())
        self.model.train(was)
        return {"pairs": int(ids.size), "mean_abs_resid": tot / ids.size}

    def state(self) -> dict[str, Any]:
        return {"steps": self.steps, "pairs": self.pairs_seen, "loss_sum": self.loss_sum,
                "abs_resid_sum": self.abs_resid_sum, "per_step": list(self.per_step), "last_cursor": self.last_cursor}

    def restore(self, state: dict[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.pairs_seen = int(state["pairs"])
        self.loss_sum = float(state["loss_sum"])
        self.abs_resid_sum = float(state["abs_resid_sum"])
        self.per_step = [list(x) for x in state["per_step"]]
        self.last_cursor = int(state["last_cursor"])

    def summary(self) -> dict[str, Any]:
        n = self.pairs_seen
        return {"w": self.w, "sibling_rows": self.rows, "steps": self.steps, "pairs": n,
                "mean_weighted_huber": self.loss_sum / n if n else None,
                "mean_abs_resid_train": self.abs_resid_sum / n if n else None,
                "per_step": self.per_step}


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

