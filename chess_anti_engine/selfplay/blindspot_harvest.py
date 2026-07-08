"""Inline blind-spot harvesting from finished selfplay games.

A value-blind position is one the net's OWN search values as fine
(``net_q = search_wdl[W]-[L] > net_ok``) while the in-loop Stockfish eval says
lost (``sf_q = sf_wdl[W]-[L] < sf_lost``). Those are the positions worth
re-seeding: the net faced the decision and got the value wrong. This module
turns a finished game's per-ply records into seed lines carrying real LC0
history (``<start_fen> | <uci moves>``, the selfplay/opening.py format), tagged
``severe`` for the high-confidence band that can be auto-fed vs the broader band
that is only collected for later threshold tuning.

Detection is a pure function (``harvest_from_records``); ``run_harvest`` is the
fail-safe finalize-side wrapper that reconstructs the pre-move boards and
appends records to a file. It never raises into finalize — a harvest bug must
not cost a game's training data.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarvestConfig:
    net_ok: float = 0.2       # capture band: net thinks fine above this
    sf_lost: float = -0.5     # capture band: SF says lost below this
    # Auto-feed (severe) band: among captured value-blind rows (already sf_q <
    # sf_lost), the ones where the net was CONFIDENT it was winning. Defined by
    # net-confidence alone — sf is already constrained by the capture gate.
    severe_net_ok: float = 0.5
    history_plies: int = 8    # preceding moves stored (LC0 uses 8 history steps)
    # sf_wdl labels the position AFTER the net's played move; move-selection
    # temperature can resample a move the SEARCH didn't favor, so a safe search
    # value + a sampled losing move would look value-blind. Require the played
    # move to carry at least this much improved-policy weight, so we keep only
    # blind spots where the net played (near-)its best and was still lost —
    # not exploration blunders (Codex review).
    min_played_prob: float = 0.1
    # Auto-feed at most ONE seed per game: the maximum value-discrepancy
    # (net_q - sf_q) severe row. A single lost game otherwise emits a cluster of
    # correlated near-duplicate plies as it marches into the loss; one worst-
    # mismatch per game keeps the auto-feed pool diverse and lets us lower
    # severe_net_ok to grab every game's worst moment without clustering. The
    # broad (collect) file still keeps all rows for later threshold tuning.
    auto_feed_one_per_game: bool = True


@dataclass(frozen=True)
class HarvestedSeed:
    line: str        # '<start_fen> | <uci ...>' or a bare FEN (seed_board_from_line grammar)
    net_q: float
    sf_q: float
    severe: bool
    ply_index: int = -1   # game ply the blind spot was faced at (-1 = unknown)


def _q(wdl: np.ndarray) -> float:
    return float(wdl[0] - wdl[2])


def pre_move_boards(
    starting_board: chess.Board,
    final_move_stack: list[chess.Move],
    record_ply_indices: list[int],
    *,
    opening_len: int,
    wanted: set[int] | None = None,
) -> tuple[list[chess.Board | None], list[chess.Move | None]]:
    """Reconstruct each record's PRE-move board (position the net faced, with
    real history) and the move the net then PLAYED from it.

    Mirrors the syzygy-rescore walk in finalize.py: ``record_ply_index`` equals
    ``len(board_before.move_stack)`` (network_turn.py), so walking the final
    move stack and matching that count aligns each record to the position just
    before its move. ``opening_len`` skips the opening plies already present in
    ``starting_board`` on the Python play path (0 on the C-ply path).

    ``wanted`` (record indices) restricts the expensive per-ply Board.copy() to
    just those records — the caller applies the cheap WDL gate first so the hot
    path doesn't copy a board for every ply of every game."""
    at_ply: dict[int, int] = {}
    for t, ply in enumerate(record_ply_indices):
        at_ply.setdefault(int(ply), t)  # first record at a ply wins (forced-ply dups)
    boards: list[chess.Board | None] = [None] * len(record_ply_indices)
    played: list[chess.Move | None] = [None] * len(record_ply_indices)
    rb = starting_board.copy()
    for mv in final_move_stack[opening_len:]:
        t = at_ply.get(len(rb.move_stack))
        if t is not None and (wanted is None or t in wanted):
            boards[t] = rb.copy()
            played[t] = mv
        rb.push(mv)
    return boards, played


def seed_line_from_board(board: chess.Board, history_plies: int) -> str:
    """'<start_fen> | <last N uci>' so the terminal is this position with real
    history; a bare FEN when no history is available (near the game start)."""
    stack = list(board.move_stack)
    n = min(int(history_plies), len(stack))
    if n <= 0:
        return board.fen()
    moves = " ".join(m.uci() for m in stack[-n:])
    tmp = board.copy()
    for _ in range(n):
        tmp.pop()
    return f"{tmp.fen()} | {moves}"


def _probs_encoding(policy_probs: np.ndarray) -> str | None:
    """The move encoding that indexes ``policy_probs`` — chosen by its length,
    NOT the config's output encoding. rec.policy_probs is the internal az_4672
    vector (POLICY_SIZE); indexing it with a compact lc0_1858 index reads the
    wrong move (Codex/review). Returns None for an unknown size (skip)."""
    from chess_anti_engine.moves.encode import policy_size_for_encoding

    for enc in ("az_4672", "lc0_1858"):
        if len(policy_probs) == policy_size_for_encoding(enc):
            return enc
    return None


def _played_was_favored(
    board: chess.Board, played: chess.Move | None, policy_probs: np.ndarray | None,
    min_prob: float,
) -> bool:
    """True if the net's PLAYED move carried >= ``min_prob`` improved-policy
    weight (so sf_wdl reflects a near-best move, not a temperature blunder).
    Missing data / an unencodable move -> conservatively False (skip)."""
    if played is None or policy_probs is None:
        return False
    enc = _probs_encoding(policy_probs)
    if enc is None:
        return False
    try:
        from chess_anti_engine.moves.encode import move_to_index_for_encoding

        idx = int(move_to_index_for_encoding(played, board, policy_encoding=enc))
        return 0 <= idx < len(policy_probs) and float(policy_probs[idx]) >= min_prob
    except (ValueError, KeyError, IndexError):
        return False


def value_blind_candidates(
    evals: Sequence[tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    *,
    cfg: HarvestConfig,
) -> list[tuple[int, float, float]]:
    """(record index, net_q, sf_q) for records passing the CHEAP WDL value-blind
    gate (no board needed): full ply with both evals, net says fine, SF says
    lost. The caller reconstructs boards only for these."""
    out: list[tuple[int, float, float]] = []
    for t, (has_policy, search_wdl, sf_wdl, _pp) in enumerate(evals):
        if not has_policy or search_wdl is None or sf_wdl is None:
            continue
        net_q, sf_q = _q(search_wdl), _q(sf_wdl)
        if net_q > cfg.net_ok and sf_q < cfg.sf_lost:
            out.append((t, net_q, sf_q))
    return out


def harvest_from_records(
    evals: Sequence[tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    boards: Sequence[chess.Board | None],
    played_moves: Sequence[chess.Move | None],
    *,
    cfg: HarvestConfig,
    ply_indices: Sequence[int] | None = None,
) -> list[HarvestedSeed]:
    """Emit a HarvestedSeed for each value-blind full ply whose pre-move
    ``board`` was reconstructed and whose played move the search favored.
    Unreconstructed / temperature-exploration records are skipped. ``ply_indices``
    (the game ply per record) is stamped onto the seed so downstream analysis can
    locate it exactly; absent it, the pre-move board's stack length is used (the
    two are equal — see pre_move_boards)."""
    out: list[HarvestedSeed] = []
    for t, net_q, sf_q in value_blind_candidates(evals, cfg=cfg):
        board, played, policy_probs = boards[t], played_moves[t], evals[t][3]
        if board is None:
            continue  # not reconstructed (unaligned or not in `wanted`)
        if not _played_was_favored(board, played, policy_probs, cfg.min_played_prob):
            continue  # temperature-explored a move the search didn't favor
        ply = int(ply_indices[t]) if ply_indices is not None else len(board.move_stack)
        out.append(HarvestedSeed(
            line=seed_line_from_board(board, cfg.history_plies),
            net_q=net_q, sf_q=sf_q, severe=net_q >= cfg.severe_net_ok, ply_index=ply,
        ))
    return out


def format_record(seed: HarvestedSeed, *, game_id: str) -> str:
    """One seed-file line: the seed grammar + an inline provenance comment
    (stripped by the loader). ``sev=1`` marks the auto-feed band. ``ply=`` is
    appended last so parsers keyed on ``game=`` are unaffected."""
    return (f"{seed.line}  # nq={seed.net_q:.2f} sq={seed.sf_q:.2f} "
            f"sev={int(seed.severe)} game={game_id} ply={seed.ply_index}")


def severe_path_for(out_path: str) -> str:
    """Sibling file holding ONLY the severe (auto-feed) rows.

    The production FEN-list loader strips the inline '# ... sev=..' comment, so
    pointing opening_fen_list_path at the mixed collect file would feed the
    broad (sev=0) band too. The severe file is the safe auto-feed target.
    Splits the BASENAME only, so a dotted directory ('data/run.1/harvest')
    doesn't misplace the file."""
    root, ext = os.path.splitext(out_path)   # ext splits on the basename's last dot
    return f"{root}.severe{ext}" if ext else f"{out_path}.severe"


def _worker_path(out_path: str) -> str:
    """Per-process file so many workers sharing one configured path don't
    interleave buffered appends into garbled (dropped) lines."""
    root, ext = os.path.splitext(out_path)
    return f"{root}.p{os.getpid()}{ext}" if ext else f"{out_path}.p{os.getpid()}"


def games_path_for(out_path: str) -> str:
    """Sibling JSONL holding the FULL game for every harvested game — the moves
    (self-contained replay) plus the per-ply net/SF eval trajectory. Only games
    that produced a seed are saved (a tiny subset), so disk stays negligible,
    unlike persisting every selfplay game. Having the whole game means the
    continuation analysis needs no replay-window join (which ages out) and an
    opponent MISPLAY can be told from a real SF error post-hoc."""
    root, _ext = os.path.splitext(out_path)
    return f"{root}.games.jsonl"


def game_record_json(
    *,
    game_id: str,
    root_fen: str,
    moves: list[str],
    result: str,
    is_selfplay: bool,
    records: list,
    ply_indices: Sequence[int],
    seeds: list[HarvestedSeed],
) -> str:
    """Compact one-line JSON for a harvested game: replay root + full move list,
    the per-ply (net_q, sf_q, has_policy) trajectory, and which plies were
    harvested. ``ply_indices`` is the already-validated int ply per record."""
    plies = []
    for rec, ply in zip(records, ply_indices):
        sw = getattr(rec, "search_wdl_est", None)
        sf = getattr(rec, "sf_wdl", None)
        plies.append({
            "ply": int(ply),
            "hp": bool(rec.has_policy),
            "nq": None if sw is None else round(_q(sw), 4),
            "sq": None if sf is None else round(_q(sf), 4),
        })
    seed_plies = [
        {"ply": s.ply_index, "sev": int(s.severe),
         "nq": round(s.net_q, 3), "sq": round(s.sf_q, 3)}
        for s in seeds
    ]
    return json.dumps({
        "game_id": game_id, "root_fen": root_fen, "result": result,
        "selfplay": bool(is_selfplay), "moves": moves,
        "seed_plies": seed_plies, "plies": plies,
    }, separators=(",", ":"))


def _save_game(
    final_board: chess.Board, records: list, ply_indices: Sequence[int],
    seeds: list[HarvestedSeed], *, game_id: str, out_path: str,
    result: str, is_selfplay: bool,
) -> None:
    """Append the full game record for a harvested game. Own try/except: a
    game-save failure must not lose the seeds already written or the return count."""
    try:
        root = final_board.copy()
        while root.move_stack:
            root.pop()
        line = game_record_json(
            game_id=game_id, root_fen=root.fen(),
            moves=[m.uci() for m in final_board.move_stack],
            result=result, is_selfplay=is_selfplay,
            records=records, ply_indices=ply_indices, seeds=seeds,
        )
        with open(_worker_path(games_path_for(out_path)), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        _log.exception("blind-spot game-save failed for game %s", game_id)


def run_harvest(
    starting_board: chess.Board | None,
    final_board: chess.Board,
    records: list,
    *,
    has_c_ply: bool,
    game_id: str,
    out_path: str,
    cfg: HarvestConfig,
    result: str = "",
    is_selfplay: bool = False,
) -> int:
    """Finalize-side wrapper: gate value-blind plies (cheap), reconstruct only
    those boards, and append the collect band + the severe (auto-feed) band to
    per-process files (``<out>.pPID`` and ``<out>.severe.pPID``). Each harvested
    game's full record (moves + eval trajectory) is also saved to
    ``<out>.games.pPID.jsonl``. Returns the seed count written. NEVER raises into
    finalize — logs and returns 0 on any error."""
    try:
        if not out_path or starting_board is None:
            return 0
        evals = [
            (bool(rec.has_policy), getattr(rec, "search_wdl_est", None),
             getattr(rec, "sf_wdl", None), getattr(rec, "policy_probs", None))
            for rec in records
        ]
        # Cheap WDL gate FIRST (no board), then reconstruct only the value-blind
        # plies — a game with no blind spots pays no per-ply Board.copy().
        candidates = value_blind_candidates(evals, cfg=cfg)
        if not candidates:
            return 0
        ply_indices = [int(rec.ply_index) for rec in records]
        wanted = {t for t, _nq, _sq in candidates}
        opening_len = 0 if has_c_ply else len(starting_board.move_stack)
        boards, played = pre_move_boards(
            starting_board, list(final_board.move_stack), ply_indices,
            opening_len=opening_len, wanted=wanted,
        )
        unaligned = sum(1 for t, _nq, _sq in candidates if boards[t] is None)
        if unaligned:
            _log.debug("harvest: %d value-blind record(s) unaligned in game %s (skipped)",
                       unaligned, game_id)
        seeds = harvest_from_records(evals, boards, played, cfg=cfg, ply_indices=ply_indices)
        if not seeds:
            return 0
        collect_path = _worker_path(out_path)
        with open(collect_path, "a", encoding="utf-8") as fh:
            for s in seeds:
                fh.write(format_record(s, game_id=game_id) + "\n")
        severe = [s for s in seeds if s.severe]
        if severe and cfg.auto_feed_one_per_game:
            # one worst-mismatch (max net_q - sf_q) row per game — see HarvestConfig
            severe = [max(severe, key=lambda s: s.net_q - s.sf_q)]
        if severe:
            with open(_worker_path(severe_path_for(out_path)), "a", encoding="utf-8") as fh:
                for s in severe:
                    fh.write(format_record(s, game_id=game_id) + "\n")
        _save_game(final_board, records, ply_indices, seeds, game_id=game_id,
                   out_path=out_path, result=result, is_selfplay=is_selfplay)
        return len(seeds)
    except Exception:
        # Harvesting is a side output — it must never break finalize or cost a
        # game's training data; log and swallow.
        _log.exception("blind-spot harvest failed for game %s", game_id)
        return 0
