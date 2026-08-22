#!/usr/bin/env python3
"""IDENTITY HARNESS — one checkpoint vs HANDICAPPED Stockfish at a FROZEN difficulty.

WHAT THIS MEASURES, and why it is not an arena. ``scripts/arena_standard.py``
scores a checkpoint against ANOTHER CHECKPOINT, so it can only ever report a
difference between two of our nets. The trained specialty — exploiting Stockfish
at a handicap — has no net-vs-net expression at all: a ladder of arms can move
together against each other while every one of them gets worse at the thing the
loop is training. This harness is the missing instrument. It plays our
checkpoint against the SAME curriculum opponent production selfplay plays,
except that the difficulty is NAILED DOWN instead of being steered.

⚑ THE FREEZE IS THE POINT. In production a PID controller moves ``wdl_regret``
every iteration to hold curriculum winrate at ``sf_pid_target_winrate`` (0.5), so
the live winrate is a CONTROLLER SETPOINT and carries no information about the
net — a stronger net is spent on a harder opponent, not on a higher score (see
``docs/rl_loop_audit.md`` and the ``wdl_regret`` notes in CLAUDE.md). Here there
is no controller: ``--regret`` and ``--sf-nodes`` are constants for the whole
run, so the score is free to move and MEANS something. That also means a score
is comparable ONLY against another run with an identical fingerprint, which is
why the fingerprint is stamped into the output and printed at both ends of the
run.

⚑ THE HANDICAP RULE IS IMPORTED, NEVER RESTATED. Stockfish searches at
``--sf-nodes`` with production's MultiPV width, and the move it plays is drawn
uniformly from the qualifying set computed by
``selfplay.stockfish_turn.within_regret_candidates`` — the same function the
production curriculum opponent calls, over candidate scores built by the same
``_collect_sf_pv_candidates`` with the same arguments the production MOVE path
passes (native SF win fraction ``w + 0.5*d``, NOT the cp-logistic used for
LABELS). A reimplementation that drifted by one comparison would measure a
different opponent while still reporting a number named "score vs handicapped
SF".

WIRING PROOF, in the artifact — and read ``admitted_by_regret_mean``, NOT
``qualifying_set_size_mean``. Every SF move records its qualifying set size AND
the size of the regret-0 set (the candidates merely TIED with SF's best), because
the raw set size does not isolate the handicap: SF's native WDL saturates, so in
a dead-drawn or dead-lost position all six MultiPV lines score identically and the
whole list qualifies at ANY band including zero. This is not hypothetical — the
first 4-game smoke at ``--regret 0.0`` read a mean qualifying set of **4.39**
while ``played_regret`` was exactly 0.0 on every one of its 245 SF moves (the
early plies read 1; the 6s were a 300-ply dead draw). Their DIFFERENCE,
``admitted_by_regret_mean``, is 0 at ``--regret 0`` by construction and must rise
with the band. ``saturated_frac`` says how much of the run the handicap could not
act on at all, and ``regret_bound_respected`` is the exact invariant: no played
move may give up more win fraction than the band allows.

⚑ COLD TRANSPOSITION TABLE PER SEARCH, and it is a deliberate departure from
production. ``StockfishPool`` hands each request to whichever pooled engine is
free, and an engine keeps its TT across searches — so with a warm TT the
opponent's move depends on which engine served the position and on what that
engine happened to search before, i.e. on thread scheduling. MEASURED: two
byte-identical 4-game invocations returned scores of 0.75 and 0.875 and diverged
on all four games. With ``fresh=True`` (``ucinewgame`` before every search, the
default here) the same pair of invocations is bit-identical down to every
qualifying-set size. An opponent whose strength depends on TT history is not a
frozen opponent, and worse, that history differs between the checkpoints being
compared. ``--no-sf-fresh-tt`` restores production's warm behaviour; it is in the
fingerprint, so the two regimes can never be differenced by accident.

Usage::

    PYTHONPATH=. python3 scripts/match_vs_handicapped_sf.py \\
        --checkpoint data/salvage/<label>/trainer.pt \\
        --regret 0.05 --sf-nodes 75000 --games 400 --sims 100 --seed 42 \\
        --out scratchpad/identity/<label>.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.eval.arena_pgn import (
    ArenaGame,
    ArenaPgnWriter,
    engine_name_from_checkpoint,
)
from chess_anti_engine.moves.encode import index_to_move, legal_move_indices
from chess_anti_engine.selfplay.stockfish_turn import (
    _choose_curriculum_opponent_move,
    _collect_sf_pv_candidates,
    _sf_played_move_diagnostics,
    within_regret_candidates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CONFIG = REPO_ROOT / "configs" / "pbt2_small.yaml"

# The band ``--regret`` lives in. Scores are SF's native win fraction
# w + 0.5*d, so a regret of 1.0 already admits every candidate SF returned (a
# move that loses a full point of win fraction against the best) and anything
# above it is the same opponent with a more alarming number attached. Negative
# is not "no handicap", it is a set that can only hold ties and would silently
# become the fallback [best] on rounding.
REGRET_MIN = 0.0
REGRET_MAX = 1.0

# Belt-and-braces cap on the number of SF processes. Each one is a real
# subprocess with its own hash table; the production fleet runs 8.
DEFAULT_SF_WORKERS = 8


# ---------------------------------------------------------------------------
# Per-move handicap application
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SfMoveOutcome:
    """What the handicap did on ONE Stockfish move.

    ``qualifying`` is the size of the set the played move was drawn from, and it
    is the realized-handicap reading: 0 marks the degenerate branch where NOT ONE
    of SF's MultiPV moves was legal in the queried position, so the move came
    from the uniform-legal fallback and no handicap was applied at all. Those are
    counted separately rather than averaged in as a set of size zero — an
    unmeasured move must never dilute the statistic that proves the measurement.
    """

    qualifying: int
    candidates: int
    # The size of the regret-0 set: how many candidates TIE with SF's best.
    # ⚑ THE CONFOUND THIS FIELD EXISTS FOR, measured on the first smoke run.
    # `qualifying` alone does NOT isolate the handicap. SF's native WDL
    # SATURATES: in a dead-drawn or dead-lost position every MultiPV line reads
    # the same (W, D, L), so w + 0.5*d is identical across all of them and the
    # regret-0 set is already the FULL candidate list. A 4-game smoke at
    # --regret 0.0 read a mean qualifying set of 4.39, which looks exactly like
    # a handicap that leaked — while `played_regret` was exactly 0.0 on every
    # move, i.e. no handicap had been applied at all. The early plies of every
    # game read 1; the 6s were a 300-ply dead draw.
    #
    # `qualifying - tied_at_best` is therefore the un-confounded reading: the
    # moves the REGRET BAND admitted beyond the ties. It is 0 at --regret 0 by
    # construction and rises with the band.
    tied_at_best: int
    rank: int | None
    regret: float | None
    nodes: int | None
    depth: int | None

    @property
    def admitted_by_regret(self) -> int:
        """Candidates the regret band admitted that a zero band would not."""
        return max(0, self.qualifying - self.tied_at_best)

    @property
    def saturated(self) -> bool:
        """Every candidate scores identically, so NO band can discriminate."""
        return self.candidates >= 2 and self.tied_at_best >= self.candidates


def sf_move_rng(*, seed: int, game_index: int, ply: int) -> np.random.Generator:
    """The RNG for ONE handicapped SF draw, keyed by (seed, game, ply).

    Deliberately a FRESH generator per move rather than one stream advanced in
    play order. The games run interleaved in a batched loop and Stockfish futures
    complete out of order, so a shared stream would make the move sequence depend
    on thread scheduling — reproducible only by luck. Keyed this way, replaying a
    single game, or the whole run at a different concurrency, draws the same
    moves.

    ``_choose_curriculum_opponent_move`` consumes exactly one ``rng.integers``
    call per move on every branch it can reach here, so one generator per move is
    not merely sufficient, it is the whole stream.
    """
    return np.random.default_rng([int(seed), int(game_index), int(ply)])


def handicapped_sf_move(
    res: Any,
    board: chess.Board,
    *,
    regret: float,
    rng: np.random.Generator,
) -> tuple[chess.Move, SfMoveOutcome]:
    """The move handicapped Stockfish plays in *board*, plus the handicap reading.

    Every step below is production's, called rather than copied:
    ``_collect_sf_pv_candidates`` with the argument list the production MOVE path
    uses (``_process_sf_results``: no wdl kwargs, so scores are SF's native
    ``w + 0.5*d`` win fraction — the units ``wdl_regret`` is defined in),
    ``_choose_curriculum_opponent_move`` for the draw, and
    ``_sf_played_move_diagnostics`` for the realized rank/regret.

    ``within_regret_candidates`` is called a SECOND time only to read the set's
    size. It is a pure function of the same three inputs, so it cannot disagree
    with the set the draw came from.
    """
    legal_idx = legal_move_indices(board)
    if legal_idx.size == 0:
        raise ValueError(f"no legal moves in {board.fen()!r}")
    legal_set = {int(x) for x in legal_idx}
    turn = bool(board.turn)

    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res, _turn=turn, legal_set=legal_set,
    )
    move_idx = _choose_curriculum_opponent_move(
        rng=rng,
        legal_indices=legal_idx,
        cand_indices=cand_idxs,
        cand_scores=cand_scores,
        regret_limit=float(regret),
    )
    qualifying = within_regret_candidates(
        cand_indices=cand_idxs, cand_scores=cand_scores, regret_limit=float(regret),
    )
    # The SAME rule at a zero band: the candidates that merely TIE with the best.
    # Read through the production function rather than recomputed, so the two
    # numbers whose difference is the reported handicap cannot drift apart.
    tied_at_best = within_regret_candidates(
        cand_indices=cand_idxs, cand_scores=cand_scores, regret_limit=0.0,
    )
    rank, played_regret = _sf_played_move_diagnostics(move_idx, cand_idxs, cand_scores)
    move = index_to_move(int(move_idx), board)
    if move not in board.legal_moves:
        raise ValueError(
            f"handicapped SF produced an illegal move {move.uci()} in "
            f"{board.fen()!r} (action id {move_idx})"
        )
    return move, SfMoveOutcome(
        qualifying=len(qualifying),
        candidates=len(cand_idxs),
        tied_at_best=len(tied_at_best),
        rank=rank,
        regret=played_regret,
        nodes=getattr(res, "nodes", None),
        depth=getattr(res, "depth", None),
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str | None:
    """Content hash of a file, or None when it cannot be read.

    Used on the Stockfish BINARY, which is part of the opponent's identity in a
    way its path is not: two runs pointing at the same path across a Stockfish
    upgrade are not the same experiment, and nothing else in the fingerprint
    would notice.
    """
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def build_fingerprint(payload: dict[str, Any]) -> str:
    """12 hex chars over the sorted settings dict. Equal ⇒ comparable scores."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class GameOutcome:
    """One finished game, from OUR side's point of view."""

    game_index: int
    pair_id: int
    half: int
    net_is_white: bool
    result: str
    score: float
    plies: int
    termination: str
    start_fen: str
    moves: tuple[chess.Move, ...]
    sf_moves: list[SfMoveOutcome] = field(default_factory=list)
    duration_s: float = 0.0

    def qualifying_sizes(self) -> list[int]:
        """Sizes of the MEASURED qualifying sets (the no-PV fallbacks excluded)."""
        return [m.qualifying for m in self.sf_moves if m.qualifying > 0]

    def no_pv_moves(self) -> int:
        return sum(1 for m in self.sf_moves if m.qualifying == 0)


def net_score_from_result(result: str, *, net_is_white: bool) -> float:
    """Our side's score in [0, 1] from a PGN result string."""
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if net_is_white else 0.0
    if result == "0-1":
        return 0.0 if net_is_white else 1.0
    raise ValueError(f"unscoreable result {result!r}")


# ---------------------------------------------------------------------------
# Play loop
# ---------------------------------------------------------------------------


def _resolve_sf_futures(futures: dict[int, Future[Any]]) -> dict[int, Any]:
    return {i: fut.result() for i, fut in futures.items()}


def play_chunk(
    model: Any,
    openings: Sequence[chess.Board],
    *,
    pair_ids: Sequence[int],
    pool: Any,
    device: str,
    regret: float,
    sf_nodes: int,
    sf_syzygy_path: str | None,
    sf_fresh_tt: bool,
    sims: int,
    seed: int,
    temperature: float,
    gumbel_add_noise: bool,
    side: Any,
    max_plies: int,
    evaluator: Any,
    tablebase: Any | None,
    tb_max_pieces: int,
    net_rng: np.random.Generator,
) -> list[GameOutcome]:
    """Play each opening twice (colors swapped) against handicapped SF.

    Our side runs the SAME batched Gumbel entry point every arena uses
    (``selfplay.match.pick_moves_for_boards``) with the shape resolved by
    ``arena_standard.resolve_search_shape``, and — like every arena — a COLD tree
    per move: no ``tree``/``root_node_ids`` are threaded through, so each search
    starts from an empty root. Production selfplay carries the tree; this does
    not, and the difference is recorded in the fingerprint as ``tree_reuse``
    rather than papered over.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        split_active_by_side_to_move,
    )
    from scripts.arena_standard import overrides_with_volatility
    from scripts.match_vs_uci import _tb_adjudicate_result

    boards: list[chess.Board] = []
    net_white: list[bool] = []
    start_fens: list[str] = []
    start_offsets: list[int] = []
    game_ids: list[int] = []
    for opening, pid in zip(openings, pair_ids, strict=True):
        for half, white in enumerate((True, False)):
            boards.append(opening.copy())
            net_white.append(white)
            start_fens.append(opening.fen())
            start_offsets.append(len(opening.move_stack))
            game_ids.append(2 * int(pid) + half)

    g = len(boards)
    done = [False] * g
    adjudicated: list[str | None] = [None] * g
    termination = ["rules"] * g
    sf_log: list[list[SfMoveOutcome]] = [[] for _ in range(g)]
    t0 = time.time()

    for _ply in range(int(max_plies)):
        for i in range(g):
            if done[i]:
                continue
            if boards[i].is_game_over(claim_draw=True):
                done[i] = True
            elif tablebase is not None:
                tb = _tb_adjudicate_result(
                    boards[i], tablebase, max_pieces=tb_max_pieces,
                )
                if tb is not None:
                    adjudicated[i] = tb
                    done[i] = True
                    termination[i] = "syzygy"
        active = [i for i in range(g) if not done[i]]
        if not active:
            break

        net_idxs, sf_idxs = split_active_by_side_to_move(active, boards, net_white)

        # SF first, and SUBMITTED before the net search runs: the engine
        # subprocesses then think on their own cores while the GPU does our
        # side's forward passes, instead of the two serializing per ply.
        futures = {
            i: pool.submit(
                boards[i].fen(), nodes=int(sf_nodes), syzygy_path=sf_syzygy_path,
                fresh=bool(sf_fresh_tt),
            )
            for i in sf_idxs
        }

        if net_idxs:
            actions = pick_moves_for_boards(
                model,
                [boards[i] for i in net_idxs],
                device=device,
                rng=net_rng,
                mcts_type="gumbel",
                mcts_simulations=int(sims),
                temperature=float(temperature),
                c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=overrides_with_volatility(side, None),
                gumbel_vloss_weight=side.vloss_weight,
                gumbel_target_batch=side.target_batch,
                evaluator=evaluator,
            )
            apply_actions_to_boards(boards, net_idxs, actions)

        for i, res in _resolve_sf_futures(futures).items():
            # The RNG key is the board's OWN ply count, so it identifies the
            # position within the game regardless of how the loop interleaved
            # them or how many games ran concurrently.
            move, outcome = handicapped_sf_move(
                res,
                boards[i],
                regret=regret,
                rng=sf_move_rng(
                    seed=seed, game_index=game_ids[i], ply=len(boards[i].move_stack),
                ),
            )
            sf_log[i].append(outcome)
            boards[i].push(move)

    outcomes: list[GameOutcome] = []
    for i in range(g):
        res_str = adjudicated[i] or boards[i].result(claim_draw=True)
        term = termination[i]
        if res_str == "*":
            # Unfinished at --max-plies. SCORED as a draw and WRITTEN as a draw:
            # emitting "*" while scoring 0.5 would make the PGN disagree with the
            # summary, which is the cross-check the artifact exists to support.
            res_str = "1/2-1/2"
            term = "max_plies"
        outcomes.append(
            GameOutcome(
                game_index=game_ids[i],
                pair_id=game_ids[i] // 2,
                half=game_ids[i] % 2,
                net_is_white=bool(net_white[i]),
                result=res_str,
                score=net_score_from_result(res_str, net_is_white=bool(net_white[i])),
                plies=len(boards[i].move_stack) - start_offsets[i],
                termination=term,
                start_fen=start_fens[i],
                moves=tuple(boards[i].move_stack[start_offsets[i]:]),
                sf_moves=sf_log[i],
                duration_s=time.time() - t0,
            ),
        )
    return outcomes


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(outcomes: Sequence[GameOutcome], *, regret: float) -> dict[str, Any]:
    """Score + CI + the realized-handicap statistics, from our side's POV.

    The CI is PENTANOMIAL wherever both colorings of a pair finished — the pair
    is the sampling unit and the two halves share an opening, so a per-game
    binomial CI would be overconfident. Pairs are reused from
    ``arena_standard`` rather than restated. A pair with only one half present
    is dropped from the pentanomial (never imputed) and the per-game CI over ALL
    games is reported alongside, clearly labelled, so a truncated run still says
    something honest.
    """
    from scripts.arena_standard import pentanomial_counts, summarize_pentanomial

    by_game = {o.game_index: o for o in outcomes}
    pair_scores: list[float] = []
    for pid in sorted({o.pair_id for o in outcomes}):
        a, b = by_game.get(2 * pid), by_game.get(2 * pid + 1)
        if a is not None and b is not None:
            pair_scores.append(a.score + b.score)

    out: dict[str, Any] = {
        "games": len(outcomes),
        "pairs_complete": len(pair_scores),
        "pairs_dropped": len({o.pair_id for o in outcomes}) - len(pair_scores),
    }

    game_scores = [o.score for o in outcomes]
    n = len(game_scores)
    mean_g = sum(game_scores) / n if n else 0.0
    se_g = (
        (statistics.pstdev(game_scores) * (n / (n - 1)) ** 0.5) / n**0.5
        if n > 1 else 0.0
    )
    out["per_game"] = {
        "score": mean_g,
        "score_se": se_g,
        "score_ci95": [mean_g - 1.96 * se_g, mean_g + 1.96 * se_g],
        "note": "IGNORES within-pair correlation; use the pentanomial block",
    }

    if pair_scores:
        summary = summarize_pentanomial(pentanomial_counts(pair_scores))
        lo = summary.score - 1.96 * summary.score_se
        hi = summary.score + 1.96 * summary.score_se
        out["ci_unit"] = "pair (pentanomial)"
        out["score"] = summary.score
        out["score_se"] = summary.score_se
        out["score_ci95"] = [lo, hi]
        out["pentanomial"] = {
            "WW": summary.counts[0], "WD_DW": summary.counts[1],
            "DD_WL": summary.counts[2], "LD_DL": summary.counts[3],
            "LL": summary.counts[4],
        }
        out["elo"] = summary.elo
        out["elo_ci95"] = list(summary.elo_ci95)
        out["elo_note"] = (
            "Elo of our side against THIS frozen handicapped SF. It is anchored "
            "to the (regret, sf_nodes, multipv) opponent in the fingerprint and "
            "is NOT on the same scale as a net-vs-net arena Elo."
        )
    else:
        out["ci_unit"] = "game (no complete pair)"
        out["score"] = mean_g
        out["score_se"] = se_g
        out["score_ci95"] = [mean_g - 1.96 * se_g, mean_g + 1.96 * se_g]
        out["elo"] = None
        out["elo_ci95"] = [None, None]

    # THE WIRING PROOF. Mean over MEASURED sets only; the no-PV fallbacks are
    # reported next to it because they are moves the handicap never touched.
    sizes = [s for o in outcomes for s in o.qualifying_sizes()]
    regrets = [
        m.regret for o in outcomes for m in o.sf_moves if m.regret is not None
    ]
    hist: dict[str, int] = {}
    for s in sizes:
        key = str(s) if s < 10 else "10+"
        hist[key] = hist.get(key, 0) + 1

    measured = [m for o in outcomes for m in o.sf_moves if m.qualifying > 0]
    ties = [m.tied_at_best for m in measured]
    admitted = [m.admitted_by_regret for m in measured]
    unsaturated = [m for m in measured if not m.saturated]
    max_regret = max(regrets) if regrets else None
    out["handicap"] = {
        "sf_moves": sum(len(o.sf_moves) for o in outcomes),
        "measured_moves": len(sizes),
        "no_pv_moves": sum(o.no_pv_moves() for o in outcomes),
        "qualifying_set_size_mean": (sum(sizes) / len(sizes)) if sizes else None,
        "qualifying_set_size_median": statistics.median(sizes) if sizes else None,
        "qualifying_set_size_max": max(sizes) if sizes else None,
        "qualifying_set_size_hist": hist,
        # ⚑ THE UN-CONFOUNDED READING. See SfMoveOutcome.tied_at_best.
        "tied_at_best_mean": (sum(ties) / len(ties)) if ties else None,
        "admitted_by_regret_mean": (
            (sum(admitted) / len(admitted)) if admitted else None
        ),
        "saturated_moves": sum(1 for m in measured if m.saturated),
        "saturated_frac": (
            sum(1 for m in measured if m.saturated) / len(measured)
        ) if measured else None,
        "qualifying_set_size_mean_unsaturated": (
            sum(m.qualifying for m in unsaturated) / len(unsaturated)
        ) if unsaturated else None,
        "played_regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
        "played_regret_max": max_regret,
        "note": (
            "READ admitted_by_regret_mean, NOT qualifying_set_size_mean. The "
            "qualifying set is (moves TIED with SF's best) + (moves the regret "
            "band admitted), and SF's native WDL SATURATES — in a dead-drawn or "
            "dead-lost position every MultiPV line scores identically, so the "
            "whole candidate list ties and qualifies at ANY band including zero. "
            "A measured --regret 0.0 smoke read qualifying_set_size_mean 4.39 "
            "with played_regret exactly 0.0 on every move, i.e. no handicap at "
            "all. admitted_by_regret_mean separates the two: it is 0 at "
            "--regret 0 by construction and must rise with the band. "
            "saturated_frac says how much of the run the handicap could not act "
            "on at all. Everything here is CAPPED BY --sf-multipv: no band can "
            "admit more moves than Stockfish was asked to report. no_pv_moves "
            "counts moves where none of SF's MultiPV lines was legal in the "
            "queried position — those took a uniform random legal move and are "
            "excluded from every mean above."
        ),
    }
    # An exact, cheap INVARIANT rather than a statistic: the move actually
    # played can never give up more win fraction than the band allows. A False
    # here means the rule the harness ran is not the rule its fingerprint
    # claims, and no score from the run is usable.
    out["handicap"]["regret_bound_respected"] = (
        None if max_regret is None else bool(max_regret <= regret + 1e-9)
    )
    out["termination"] = {
        t: sum(1 for o in outcomes if o.termination == t)
        for t in sorted({o.termination for o in outcomes})
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _config_defaults(config_path: Path) -> dict[str, Any]:
    """SF/opening/syzygy defaults read from *config_path*.

    ⚑ Searched with ``arena_standard._find_nested`` rather than off a fixed
    section. These keys do NOT all live under ``selfplay:`` — ``stockfish_path``,
    ``sf_multipv`` and ``sf_hash_mb`` are under a separate top-level
    ``stockfish:`` block while the syzygy pair is under ``selfplay:`` — and a
    reader that assumed one section returned None for the binary and would have
    made ``--stockfish`` a mandatory flag that the config was supposed to
    supply. The recursive read also survives the next reorganization of the
    file.

    ``stockfish_syzygy_path`` wins over ``syzygy_path`` because it is the one
    the production selfplay path hands the ENGINE (``_sf_syzygy_path_for_slot``);
    they are the same pair on today's config, and preferring the engine's key
    means they can diverge without this silently handing SF the other one.
    """
    import yaml

    from scripts.arena_standard import _find_nested

    def _int(key: str, default: int) -> int:
        """``_find_nested`` is untyped by construction (it walks a raw yaml), so
        the narrowing happens once, here, rather than at each use site."""
        found = _find_nested(cfg, key)
        return default if found is None else int(str(found))

    cfg = yaml.safe_load(Path(config_path).read_text())
    return {
        "stockfish_path": _find_nested(cfg, "stockfish_path"),
        "sf_multipv": _int("sf_multipv", 6),
        "sf_hash_mb": _int("sf_hash_mb", 17),
        "syzygy_path": _find_nested(cfg, "stockfish_syzygy_path")
        or _find_nested(cfg, "syzygy_path"),
    }


def build_parser(
    defaults: dict[str, Any], *, config_path: Path,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="our side: trainer.pt or checkpoint dir")
    p.add_argument("--config", type=Path, default=config_path,
                   help="config the SF/opening/syzygy defaults are READ FROM "
                        f"(default: {PRODUCTION_CONFIG.name}). It is consumed by "
                        "a pre-parse before these defaults are built, so the "
                        "values below really do come from the file named here; "
                        "the resolved values are printed at startup from the "
                        "variables the Stockfish pool is constructed with.")
    p.add_argument("--regret", type=float, required=True,
                   help="FROZEN wdl_regret for the whole run, in SF win-fraction "
                        f"units [{REGRET_MIN}, {REGRET_MAX}]. Higher = weaker SF. "
                        "There is no PID here: this value never moves.")
    p.add_argument("--sf-nodes", type=int, default=75000,
                   help="frozen SF node budget per move (default: 75000, the "
                        "production sf_nodes)")
    p.add_argument("--sf-multipv", type=int, default=int(defaults["sf_multipv"]),
                   help="MultiPV width; CAPS the qualifying set "
                        f"(default: production {defaults['sf_multipv']})")
    p.add_argument("--sf-hash-mb", type=int, default=int(defaults["sf_hash_mb"]),
                   help=f"per-engine hash (default: production {defaults['sf_hash_mb']})")
    p.add_argument("--stockfish", default=defaults["stockfish_path"],
                   help="Stockfish binary (default: the production one)")
    p.add_argument("--no-sf-fresh-tt", dest="sf_fresh_tt", action="store_false",
                   help="let Stockfish keep its transposition table between "
                        "searches (production's behaviour). ⚑ This makes the "
                        "opponent depend on which pooled engine served the "
                        "position and on what it searched before, i.e. on "
                        "thread scheduling — MEASURED: two identical 4-game "
                        "invocations then returned different results. Default "
                        "is a COLD TT per search, which is what makes a frozen "
                        "opponent a function of the position alone.")
    p.add_argument("--sf-workers", type=int, default=DEFAULT_SF_WORKERS,
                   help=f"concurrent Stockfish processes (default: {DEFAULT_SF_WORKERS})")
    p.add_argument("--games", type=int, default=400,
                   help="total games; must be even — games/2 paired openings "
                        "(default: 400)")
    p.add_argument("--sims", type=int, default=100,
                   help="our side's MCTS simulations (default: 100)")
    p.add_argument("--search-shape", choices=("play", "training"), default="play",
                   help="our side's search shape, resolved by arena_standard "
                        "(default: play — the same shape the ledger's arenas use)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="our side's move-selection temperature (default: 0.1, "
                        "arena_standard's)")
    p.add_argument("--no-gumbel-noise", action="store_true",
                   help="disable root Gumbel noise on our side")
    p.add_argument("--openings", type=Path, default=None,
                   help="PGN(.zip)/Polyglot book (default: the production "
                        "opening_book_path_2)")
    p.add_argument("--opening-plies", type=int, default=16,
                   help="book plies per opening (default: 16 = 8 moves)")
    p.add_argument("--max-plies", type=int, default=300,
                   help="score as a draw after this many plies (default: 300)")
    p.add_argument("--syzygy", default=defaults["syzygy_path"],
                   help="colon-separated Syzygy pair for BOTH sides "
                        "(default: the production pair)")
    p.add_argument("--tb-max-pieces", type=int, default=6,
                   help="adjudicate positions at or below this piece count")
    p.add_argument("--no-syzygy-adjudication", action="store_true",
                   help="play endgames out instead of adjudicating from WDL")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="cap this process's share of the GPU (use when training "
                        "owns the card)")
    p.add_argument("--eval-max-batch", type=int, default=4096,
                   help="cap on the hoisted evaluator's batch; below the search's "
                        "uncapped leaf size it SHRINKS THE SEARCH (default: 4096)")
    p.add_argument("--max-concurrent-games", type=int, default=64,
                   help="games in flight at once (default: 64)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile our side (off by default; recorded in the "
                        "fingerprint either way)")
    p.add_argument("--out", type=Path, required=True,
                   help="result JSON (one object, overwritten)")
    p.add_argument("--pgn-out", type=Path, default=None)
    p.add_argument("--label", default=None,
                   help="free-text run label; NOT part of the fingerprint")
    p.add_argument("--report-every", type=int, default=32,
                   help="progress line every N finished games (default: 32)")
    return p


def _validate(args: argparse.Namespace) -> None:
    if not REGRET_MIN <= float(args.regret) <= REGRET_MAX:
        raise SystemExit(
            f"--regret {args.regret} is outside [{REGRET_MIN}, {REGRET_MAX}]. "
            "It is a win-fraction band (SF's w + 0.5*d), not centipawns: "
            "production's PID holds it near 0.03-0.06. Negative admits nothing "
            "but ties and would silently fall back to SF's best move; above 1.0 "
            "every candidate already qualifies, so the number would be a "
            "different opponent than it reads as."
        )
    if args.games < 2 or args.games % 2 != 0:
        raise SystemExit("--games must be even and >= 2 (paired openings)")
    if args.sf_nodes < 1:
        raise SystemExit("--sf-nodes must be >= 1")
    if args.sims < 1:
        raise SystemExit("--sims must be >= 1")
    if args.sf_multipv < 1:
        raise SystemExit("--sf-multipv must be >= 1")
    if args.sf_workers < 1:
        raise SystemExit("--sf-workers must be >= 1")
    if not args.stockfish:
        raise SystemExit(
            "no Stockfish binary: pass --stockfish (the config has no "
            "selfplay.stockfish_path)"
        )
    if not Path(args.stockfish).exists():
        raise SystemExit(f"Stockfish binary not found: {args.stockfish}")
    if float(args.regret) == 0.0:
        print(
            "[idharness] NOTE: --regret 0 is the FULL-STRENGTH opponent (only "
            "moves tied with SF's best qualify). The run is still valid; it is "
            "also the calibration point at which the reported mean qualifying "
            "set size must come out ~1.0.",
            flush=True,
        )


def main() -> None:
    # ⚑ TWO-STAGE PARSE, and it is not cosmetic. ``--config`` has to be read
    # BEFORE the defaults it feeds are computed. Building the parser against a
    # hard-coded PRODUCTION_CONFIG and then accepting a ``--config`` flag would
    # make that flag a value the program takes and silently ignores — this
    # repo's signature defect, and here it would silently measure the production
    # opponent while the operator believed they had pointed it elsewhere.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=PRODUCTION_CONFIG)
    pre_args, _rest = pre.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.exists():
        raise SystemExit(f"--config not found: {config_path}")
    defaults = _config_defaults(config_path)
    args = build_parser(defaults, config_path=config_path).parse_args()
    _validate(args)

    from scripts.arena_standard import (
        arena_uncapped_leaf_rows,
        build_arena_evaluator,
        default_compile_cache_dir,
        default_openings_path,
        load_paired_openings,
        resolve_search_shape,
    )

    if args.compile:
        from scripts.arena_standard import _configure_shared_compile_cache

        _configure_shared_compile_cache(default_compile_cache_dir())

    import torch

    from chess_anti_engine.stockfish.pool import StockfishPool
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_mem_fraction), torch.device(args.device).index or 0,
        )

    side = resolve_search_shape(str(args.search_shape))
    openings_path = args.openings or default_openings_path()
    n_pairs = args.games // 2
    chunk_pairs = max(1, int(args.max_concurrent_games) // 2)

    # Refused BEFORE the checkpoint load, exactly as arena_standard does: the
    # C search mins its leaf buffer against the evaluator's cap and ABSORBS the
    # overflow leaves as pseudo-terminals carrying the root's Q, so a cap below
    # the uncapped size changes the moves instead of just the memory.
    uncapped_leaf_rows = arena_uncapped_leaf_rows(
        max_concurrent_games=2 * chunk_pairs, sides=(side,),
    )
    if str(args.device).startswith("cuda") and args.eval_max_batch > 0:
        if args.eval_max_batch < 2 * chunk_pairs:
            raise SystemExit(
                f"--eval-max-batch {args.eval_max_batch} is below the "
                f"{2 * chunk_pairs} boards one root submit can carry; the "
                f"hoisted evaluator would refuse it mid-run. Use "
                f"--eval-max-batch {uncapped_leaf_rows} (the size at which the "
                f"search runs unchanged) or 0 for per-call evaluators."
            )
        if args.eval_max_batch < uncapped_leaf_rows:
            print(
                f"[idharness] WARNING: --eval-max-batch {args.eval_max_batch} < "
                f"{uncapped_leaf_rows} uncapped leaf rows: this SHRINKS THE "
                f"SEARCH, it does not just cap memory. The value is in the "
                f"fingerprint, so such a run is only comparable to another run "
                f"with the same cap.",
                flush=True,
            )

    syzygy = str(args.syzygy or "") or None
    sf_binary_sha = file_sha256(args.stockfish)

    # ⚑ EVERY field here changes the OPPONENT or the ruler. Two runs are
    # comparable iff this dict is equal — which is why the label, the output
    # paths, the device, the worker count and the concurrency are NOT in it:
    # they change the wall clock, not the measurement.
    fingerprint_payload = {
        "regret": float(args.regret),
        "sf_nodes": int(args.sf_nodes),
        "sf_multipv": int(args.sf_multipv),
        "sf_hash_mb": int(args.sf_hash_mb),
        "sf_fresh_tt": bool(args.sf_fresh_tt),
        "sf_binary_sha256": sf_binary_sha,
        "sims": int(args.sims),
        "search_shape": side.shape,
        "search_gumbel": {k: float(v) for k, v in side.realized_gumbel().items()},
        "search_vloss_weight": int(side.vloss_weight),
        "search_target_batch": int(side.target_batch),
        "tree_reuse": side.tree_reuse,
        "temperature": float(args.temperature),
        "gumbel_add_noise": not args.no_gumbel_noise,
        "seed": int(args.seed),
        "games": int(args.games),
        "book": str(openings_path),
        "opening_plies": int(args.opening_plies),
        "max_plies": int(args.max_plies),
        "syzygy": syzygy,
        "tb_adjudication": (not args.no_syzygy_adjudication),
        "tb_max_pieces": int(args.tb_max_pieces),
        "eval_max_batch": int(args.eval_max_batch),
        "compile": "on" if args.compile else "off",
    }
    fingerprint = build_fingerprint(fingerprint_payload)

    banner = (
        f"[idharness] FINGERPRINT {fingerprint} — regret={args.regret} "
        f"sf_nodes={args.sf_nodes} multipv={args.sf_multipv} sims={args.sims} "
        f"shape={side.shape} seed={args.seed} games={args.games}\n"
        f"[idharness] ⚑ A SCORE FROM THIS RUN IS COMPARABLE ONLY TO ANOTHER RUN "
        f"WITH FINGERPRINT {fingerprint}. Change the regret, the node budget, "
        f"the MultiPV width, the sims, the search shape, the book or the "
        f"Stockfish binary and you have measured a DIFFERENT opponent — the "
        f"scores are then not on one scale and must not be differenced."
    )
    print(banner, flush=True)
    print(f"[idharness] our side: {side.describe()}", flush=True)

    rng = np.random.default_rng(int(args.seed))
    print(f"[idharness] sampling {n_pairs} openings from {openings_path}", flush=True)
    openings = load_paired_openings(
        openings_path, n_pairs=n_pairs, max_plies=int(args.opening_plies), rng=rng,
    )

    tablebase = None
    if syzygy and not args.no_syzygy_adjudication:
        from scripts.match_vs_uci import (
            _open_syzygy_tablebase,
            _validate_syzygy_adjudicator,
        )

        tablebase = _open_syzygy_tablebase(syzygy)
        _validate_syzygy_adjudicator(
            syzygy, tablebase, max_pieces=int(args.tb_max_pieces),
        )

    print(f"[idharness] loading {args.checkpoint}", flush=True)
    t_load = time.time()
    model = load_model_from_checkpoint(str(args.checkpoint), device=str(args.device))
    print(f"[idharness] loaded in {time.time() - t_load:.0f}s", flush=True)
    if args.compile:
        model = torch.compile(model)
        print("[idharness] torch.compile ON", flush=True)

    evaluator = None
    if str(args.device).startswith("cuda") and args.eval_max_batch > 0:
        evaluator = build_arena_evaluator(
            model, device=str(args.device), max_batch=int(args.eval_max_batch),
        )

    pgn_writer = None
    if args.pgn_out is not None:
        pgn_writer = ArenaPgnWriter(
            args.pgn_out,
            event="identity_harness",
            base_tags={
                "Fingerprint": fingerprint,
                "Regret": str(args.regret),
                "SfNodes": str(args.sf_nodes),
            },
        )
    net_name = engine_name_from_checkpoint(str(args.checkpoint), fallback="net")
    sf_name = f"SF_handicap_r{args.regret}_n{args.sf_nodes}"

    # Bound to locals FIRST, then printed FROM THOSE LOCALS and passed from them.
    # The point is that the line below reports what the pool was built with,
    # not what the yaml or the argparse namespace said — a resolved-value report
    # taken off the producer is exactly how a config key that never reaches its
    # consumer keeps looking wired.
    sf_path = str(args.stockfish)
    sf_nodes = int(args.sf_nodes)
    sf_multipv = int(args.sf_multipv)
    sf_hash_mb = int(args.sf_hash_mb)
    sf_workers = int(args.sf_workers)
    print(
        f"[idharness] SF pool <- config {config_path}: path={sf_path} "
        f"nodes={sf_nodes} multipv={sf_multipv} hash_mb={sf_hash_mb} "
        f"workers={sf_workers} fresh_tt={bool(args.sf_fresh_tt)} syzygy={syzygy} "
        f"sha256={sf_binary_sha}",
        flush=True,
    )
    pool = StockfishPool(
        path=sf_path,
        nodes=sf_nodes,
        num_workers=sf_workers,
        multipv=sf_multipv,
        hash_mb=sf_hash_mb,
        syzygy_path=syzygy,
    )

    outcomes: list[GameOutcome] = []
    t_start = time.time()
    try:
        for start in range(0, n_pairs, chunk_pairs):
            block = list(range(start, min(start + chunk_pairs, n_pairs)))
            chunk = play_chunk(
                model,
                [openings[k] for k in block],
                pair_ids=block,
                pool=pool,
                device=str(args.device),
                regret=float(args.regret),
                sf_nodes=int(args.sf_nodes),
                sf_syzygy_path=syzygy,
                sf_fresh_tt=bool(args.sf_fresh_tt),
                sims=int(args.sims),
                seed=int(args.seed),
                temperature=float(args.temperature),
                gumbel_add_noise=not args.no_gumbel_noise,
                side=side,
                max_plies=int(args.max_plies),
                evaluator=evaluator,
                tablebase=tablebase,
                tb_max_pieces=int(args.tb_max_pieces),
                net_rng=rng,
            )
            outcomes.extend(chunk)
            if pgn_writer is not None:
                for o in chunk:
                    pgn_writer.write_game(
                        ArenaGame(
                            white=net_name if o.net_is_white else sf_name,
                            black=sf_name if o.net_is_white else net_name,
                            result=o.result,
                            moves=o.moves,
                            start_fen=o.start_fen,
                            pair_id=o.pair_id,
                            pair_half=o.half,
                            extra={"Termination": o.termination},
                        ),
                    )
            if len(outcomes) % max(1, int(args.report_every)) < len(chunk):
                partial = summarize(outcomes, regret=float(args.regret))
                print(
                    f"[idharness] {len(outcomes)}/{args.games} games "
                    f"({time.time() - t_start:.0f}s) score={partial['score']:.4f} "
                    f"admitted_by_regret="
                    f"{partial['handicap']['admitted_by_regret_mean']}",
                    flush=True,
                )
    finally:
        pool.close()
        if pgn_writer is not None:
            pgn_writer.close()
        if tablebase is not None:
            tablebase.close()

    record: dict[str, Any] = {
        "instrument": "identity_harness_vs_handicapped_sf",
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "config": str(config_path),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "comparability_warning": (
            "Scores are comparable ONLY across runs with an IDENTICAL "
            f"fingerprint ({fingerprint}). Differencing two runs whose "
            "fingerprints differ compares two different opponents."
        ),
        "duration_s": round(time.time() - t_start, 1),
        "sf_workers": int(args.sf_workers),
        "max_concurrent_games": int(args.max_concurrent_games),
        "device": str(args.device),
        "sf_engine_replacements": int(getattr(pool, "replacements", 0)),
        "argv": [sys.argv[0], *sys.argv[1:]],
        "command": " ".join(shlex.quote(a) for a in sys.argv),
    }
    record.update(summarize(outcomes, regret=float(args.regret)))
    record["per_game_log"] = [
        {
            "game": o.game_index,
            "pair": o.pair_id,
            "half": o.half,
            "net_is_white": o.net_is_white,
            "result": o.result,
            "score": o.score,
            "plies": o.plies,
            "termination": o.termination,
            "start_fen": o.start_fen,
            "sf_moves": len(o.sf_moves),
            "no_pv_moves": o.no_pv_moves(),
            "qualifying_set_size_mean": (
                sum(o.qualifying_sizes()) / len(o.qualifying_sizes())
                if o.qualifying_sizes() else None
            ),
            "admitted_by_regret_mean": (
                sum(m.admitted_by_regret for m in o.sf_moves if m.qualifying > 0)
                / len(o.qualifying_sizes())
                if o.qualifying_sizes() else None
            ),
            "saturated_moves": sum(1 for m in o.sf_moves if m.saturated),
            "qualifying_set_sizes": [m.qualifying for m in o.sf_moves],
            "tied_at_best_sizes": [m.tied_at_best for m in o.sf_moves],
            "played_regrets": [m.regret for m in o.sf_moves],
        }
        for o in sorted(outcomes, key=lambda o: o.game_index)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    h = record["handicap"]
    print(
        f"[idharness] score {record['score']:.4f} "
        f"[{record['score_ci95'][0]:.4f}, {record['score_ci95'][1]:.4f}] "
        f"({record['ci_unit']}), elo {record['elo']} {record['elo_ci95']}\n"
        f"[idharness] realized handicap over {h['measured_moves']} SF moves: "
        f"admitted_by_regret {h['admitted_by_regret_mean']} "
        f"(THE reading) | qualifying {h['qualifying_set_size_mean']} = "
        f"tied_at_best {h['tied_at_best_mean']} + admitted | "
        f"saturated {h['saturated_frac']} | no-PV {h['no_pv_moves']} | "
        f"played regret mean {h['played_regret_mean']} max "
        f"{h['played_regret_max']} | bound respected "
        f"{h['regret_bound_respected']}\n"
        f"[idharness] wrote {args.out}\n{banner}",
        flush=True,
    )


if __name__ == "__main__":
    main()
