#!/usr/bin/env python3
"""Standardized paired-opening arena for strength measurement.

This is THE script every architecture / training-target candidate is judged
with (see docs/eval_protocol.md). It plays a candidate checkpoint against a
reference checkpoint over paired openings — each opening is played twice with
colors swapped, and the PAIR (not the game) is the unit of analysis. Results
are summarized as pentanomial counts with an Elo estimate and 95% CI computed
from the pentanomial variance (paired games are correlated, so the trinomial
W/D/L variance would understate the CI), then appended as one JSON line to
``runs/arena_results.jsonl``.

Two budget modes:

- ``matched_sims``: in-process batched MCTS via the same gumbel path as
  selfplay matches (``chess_anti_engine/selfplay/match.py``). Isolates policy
  /value quality at a fixed search budget — model latency does not matter.
- ``matched_time``: each side runs as a real UCI engine subprocess
  (``python -m chess_anti_engine.uci``), i.e. the same batched-inference path
  the engine ships with, at a fixed per-move wall clock. Model latency
  differences ARE part of the result.

Usage::

    PYTHONPATH=. python3 scripts/arena_standard.py \\
        --candidate runs/candidate/trainer.pt --reference runs/ref/trainer.pt \\
        --games 1000 --mode matched_sims --sims 64

    PYTHONPATH=. python3 scripts/arena_standard.py \\
        --candidate ... --reference ... --mode matched_time --ms-per-move 100
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CONFIG = REPO_ROOT / "configs" / "pbt2_small.yaml"
DEFAULT_RESULTS_PATH = REPO_ROOT / "runs" / "arena_results.jsonl"

# matched_sims --compile=auto threshold: compile only when the run does enough
# inference work to amortize the ~2-4 min torch.compile cost. games*sims is a
# proxy for total forward passes; 12800 ~= 50 games at 256 sims. Below this the
# compile overhead dwarfs the per-call speedup, so eager is faster wall-clock.
AUTO_COMPILE_WORK_THRESHOLD = 12800

# Pentanomial bins from the candidate's point of view, by pair score
# (candidate points over the two games of one opening pair).
PAIR_SCORES = (2.0, 1.5, 1.0, 0.5, 0.0)
PAIR_LABELS = ("WW", "WD_DW", "DD_WL", "LD_DL", "LL")


# ---------------------------------------------------------------------------
# Pentanomial bookkeeping + Elo math
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PentanomialSummary:
    """Pentanomial match summary from the candidate's point of view."""

    counts: tuple[int, int, int, int, int]  # WW, WD/DW, DD+WL, LD/DL, LL
    pairs: int
    games: int
    score: float           # mean per-game score in [0, 1]
    score_se: float        # standard error of the mean per-game score
    elo: float | None
    elo_ci95: tuple[float | None, float | None]


def game_scores_to_pair_scores(game_scores: list[float]) -> list[float]:
    """Collapse per-game candidate scores (1/0.5/0) into per-pair scores.

    Games must be ordered so that ``game_scores[2*i]`` and
    ``game_scores[2*i + 1]`` are the two colorings of opening pair ``i``.
    """
    if len(game_scores) % 2 != 0:
        raise ValueError(f"need an even number of games, got {len(game_scores)}")
    for s in game_scores:
        if s not in (0.0, 0.5, 1.0):
            raise ValueError(f"per-game score must be 0, 0.5 or 1, got {s}")
    return [
        game_scores[i] + game_scores[i + 1]
        for i in range(0, len(game_scores), 2)
    ]


def pentanomial_counts(pair_scores: list[float]) -> tuple[int, int, int, int, int]:
    """Bin pair scores into (WW, WD/DW, DD+WL, LD/DL, LL) counts."""
    counts = [0, 0, 0, 0, 0]
    for s in pair_scores:
        try:
            counts[PAIR_SCORES.index(s)] += 1
        except ValueError:
            raise ValueError(f"pair score must be one of {PAIR_SCORES}, got {s}") from None
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def _elo_from_score(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0) + 0.0  # +0.0 normalizes -0.0


def summarize_pentanomial(
    counts: tuple[int, int, int, int, int], *, z: float = 1.96,
) -> PentanomialSummary:
    """Elo point estimate + CI from pentanomial pair counts.

    The pair is the sampling unit: per-pair normalized scores are
    x in {1, 0.75, 0.5, 0.25, 0} and the CI uses the empirical variance of x
    across pairs. This correctly accounts for the within-pair correlation
    that a trinomial (per-game W/D/L) variance would miss.
    """
    n = sum(counts)
    if n <= 0:
        raise ValueError("no pairs")
    xs = tuple(s / 2.0 for s in PAIR_SCORES)
    mu = sum(c * x for c, x in zip(counts, xs)) / n
    var = sum(c * (x - mu) ** 2 for c, x in zip(counts, xs)) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    lo = mu - z * se
    hi = mu + z * se
    return PentanomialSummary(
        counts=counts,
        pairs=n,
        games=2 * n,
        score=mu,
        score_se=se,
        elo=_elo_from_score(mu),
        elo_ci95=(_elo_from_score(lo), _elo_from_score(hi)),
    )


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def default_openings_path() -> Path:
    """The 8-move UHO book from the production config (opening_book_path_2)."""
    import yaml

    cfg = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    selfplay = cfg.get("selfplay", {}) if isinstance(cfg, dict) else {}
    book = selfplay.get("opening_book_path_2") or selfplay.get("opening_book_path")
    if not book:
        raise SystemExit(
            f"no opening_book_path(_2) in {PRODUCTION_CONFIG}; pass --openings"
        )
    return Path(str(book))


def _find_nested(cfg: object, key: str) -> object | None:
    """Return the first value of ``key`` anywhere in a nested dict, else None."""
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            found = _find_nested(v, key)
            if found is not None:
                return found
    return None


def default_compile_cache_dir() -> Path:
    """The production worker's shared torch.compile/triton cache root.

    Mirrors ``tune.distributed_runtime._resolve_shared_cache_root``: the
    explicit ``distributed_worker_shared_cache_dir`` from the production config
    wins (that is exactly the dir the live workers populate), otherwise we fall
    back to ``<work_dir>/server/worker_cache``. Pointing the arena's compile at
    this reuses the autotuned kernels + FX graphs that training already baked,
    so an arena run skips most of its cold-compile cost.

    Falls back to the documented literal ``runs/pbt2_small/server/worker_cache``
    (the production worker cache) if the config can't be read; an absent dir is
    harmless (just no reuse that run). Always returns an absolute path.
    """
    fallback = REPO_ROOT / "runs" / "pbt2_small" / "server" / "worker_cache"
    try:
        import yaml

        cfg = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    except (OSError, ValueError):
        return fallback
    explicit = _find_nested(cfg, "distributed_worker_shared_cache_dir")
    if explicit and str(explicit).strip():
        return Path(str(explicit).strip()).expanduser().resolve()
    work_dir = _find_nested(cfg, "work_dir")
    if work_dir and str(work_dir).strip():
        root = Path(str(work_dir).strip()).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        return (root / "server" / "worker_cache").resolve()
    return fallback


def _configure_shared_compile_cache(cache_dir: Path) -> None:
    """Point TorchInductor/Triton at ``cache_dir`` for cross-run kernel reuse.

    Mirrors ``worker._configure_shared_compile_cache`` so the arena hits the
    same on-disk cache layout the training workers populate. Uses
    ``os.environ.setdefault`` so an explicitly-exported env var still wins, and
    must be called BEFORE any ``torch.compile`` (i.e. before run_arena loads
    torch). Creating the dirs is harmless when they don't exist yet.
    """
    import os

    compile_cache_root = cache_dir / "compile_cache"
    inductor_dir = compile_cache_root / "torchinductor"
    triton_dir = compile_cache_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    print(
        f"[arena] compile cache: {inductor_dir} "
        f"(reusing training kernels if present)",
        flush=True,
    )


def load_paired_openings(
    path: Path, *, n_pairs: int, max_plies: int, rng: np.random.Generator,
) -> list[chess.Board]:
    """Sample ``n_pairs`` opening boards from a PGN(.zip)/Polyglot book.

    Sampling is weighted by book frequency (same as selfplay) and seeded, so
    a fixed --seed reproduces the exact opening set. Duplicate positions are
    rejected until the book runs out of variety.
    """
    from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board

    cfg = OpeningConfig(
        opening_book_path=str(path),
        opening_book_max_plies=int(max_plies),
        opening_book_prob=1.0,
    )
    boards: list[chess.Board] = []
    seen: set[str] = set()
    attempts = 0
    duplicates = 0
    max_attempts = max(1000, 50 * n_pairs)
    while len(boards) < n_pairs:
        board = sample_starting_board(rng=rng, cfg=cfg).board
        attempts += 1
        key = board.epd()
        if key in seen:
            if attempts < max_attempts:
                continue
            duplicates += 1
        seen.add(key)
        boards.append(board)
    if duplicates:
        print(
            f"[arena] WARNING: opening book ran out of unique openings; "
            f"{duplicates}/{n_pairs} pairs reuse an opening. Duplicate pairs "
            f"are correlated samples, so the reported CI is overconfident."
        )
    return boards


# ---------------------------------------------------------------------------
# Game play — matched sims (in-process batched MCTS)
# ---------------------------------------------------------------------------

def play_paired_games_matched_sims(
    model_candidate,
    model_reference,
    openings: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    sims_candidate: int,
    sims_reference: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    volatility_candidate: dict[str, float] | None = None,
    syzygy_tablebase: object | None = None,
    tb_max_pieces: int = 6,
    gumbel_candidate: dict[str, float] | None = None,
    gumbel_reference: dict[str, float] | None = None,
) -> list[float]:
    """Play each opening twice (colors swapped) and return per-pair scores.

    ``volatility_candidate`` (volatility_q_scale / volatility_fpu /
    volatility_anchor) applies volatility-aware Gumbel search to the
    CANDIDATE side only — the reference keeps today's search, which is the
    A/B the experiment needs. Non-zero flags force the Python search path
    (mcts/gumbel.py), so matched_sims is the honest mode.

    Reuses the selfplay match helpers so search behavior (gumbel MCTS, the
    model's own input_history_encoding / policy_encoding, including the
    compact lc0_1858 path) is identical to ``play_match_batch``; this loop
    only differs in pinning the starting boards and keeping per-game results.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        result_from_a_pov,
        split_active_by_side_to_move,
    )
    from scripts.match_vs_uci import _tb_adjudicate_result

    boards: list[chess.Board] = []
    a_plays_white: list[bool] = []
    for opening in openings:
        boards.append(opening.copy())
        a_plays_white.append(True)
        boards.append(opening.copy())
        a_plays_white.append(False)

    g = len(boards)
    done = [False] * g
    adjudicated: list[str | None] = [None] * g  # Syzygy-adjudicated result per game
    t0 = time.time()
    for ply in range(int(max_plies)):
        for i in range(g):
            if done[i]:
                continue
            if boards[i].is_game_over(claim_draw=True):
                done[i] = True
            elif syzygy_tablebase is not None:
                # Adjudicate the instant a game reaches a covered (<=N-man) position
                # — kills long endgame tails. Reuses match_vs_uci's WDL probe.
                _tb = _tb_adjudicate_result(boards[i], syzygy_tablebase, max_pieces=tb_max_pieces)
                if _tb is not None:
                    adjudicated[i] = _tb
                    done[i] = True
        active = [i for i in range(g) if not done[i]]
        if not active:
            break
        if ply and ply % 20 == 0:
            print(
                f"[arena] ply {ply}: {g - len(active)}/{g} games finished "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        a_to_move, b_to_move = split_active_by_side_to_move(
            active, boards, a_plays_white,
        )
        vol_kwargs = dict(volatility_candidate or {})
        for model, idxs, sims, extra, gov in (
            (model_candidate, a_to_move, sims_candidate, vol_kwargs, gumbel_candidate),
            (model_reference, b_to_move, sims_reference, {}, gumbel_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=gov,
                **extra,
            )
            apply_actions_to_boards(boards, idxs, actions)

    def _game_score(i: int) -> float:
        res = adjudicated[i] or boards[i].result(claim_draw=True)
        if res == "*":  # unfinished at max_plies, not TB-covered: adjudicate as draw
            return 0.5
        return {1: 1.0, 0: 0.5, -1: 0.0}[
            result_from_a_pov(res, a_is_white=bool(a_plays_white[i]))
        ]

    game_scores = [_game_score(i) for i in range(g)]
    return game_scores_to_pair_scores(game_scores)


def play_paired_games_matched_sims_rolling(
    model_candidate,
    model_reference,
    openings: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    sims_candidate: int,
    sims_reference: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    volatility_candidate: dict[str, float] | None = None,
    syzygy_tablebase: object | None = None,
    tb_max_pieces: int = 6,
    pool_size: int = 256,
    gumbel_candidate: dict[str, float] | None = None,
    gumbel_reference: dict[str, float] | None = None,
) -> list[float]:
    """Rolling-pool variant: keep ``pool_size`` games active at all times, starting
    a fresh game the instant one finishes (like production selfplay), instead of
    playing fixed chunks to completion.

    Two wins over the chunked path: (1) the GPU never drains until the very end
    (no per-chunk tail), and (2) the active-game count — hence the batched-inference
    shape — stays FIXED at ``pool_size`` while the queue lasts, so torch.compile
    compiles once and reuses it (no per-shape recompile thrash). Only the final
    drain (last ~pool_size games) has shrinking shapes.

    Each opening is still played twice (colors swapped) and scored as a pair;
    game_id ``2k``/``2k+1`` are the white/black halves of opening ``k``, so the
    flat ``game_scores`` reassemble into the same pairs as the lockstep path.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        result_from_a_pov,
        split_active_by_side_to_move,
    )
    from scripts.match_vs_uci import _tb_adjudicate_result

    queue: list[tuple[int, chess.Board, bool]] = []
    for k, opening in enumerate(openings):
        queue.append((2 * k, opening, True))
        queue.append((2 * k + 1, opening, False))
    n_games = len(queue)
    queue.reverse()  # pop() from the end
    game_scores: list[float | None] = [None] * n_games

    boards: list[chess.Board] = []
    gids: list[int] = []
    awhite: list[bool] = []
    gplies: list[int] = []

    def _refill() -> None:
        while len(boards) < pool_size and queue:
            gid, opening, aw = queue.pop()
            boards.append(opening.copy())
            gids.append(gid)
            awhite.append(aw)
            gplies.append(0)

    def _record(j: int, res: str) -> None:
        if res == "*":
            game_scores[gids[j]] = 0.5
        else:
            game_scores[gids[j]] = {1: 1.0, 0: 0.5, -1: 0.0}[
                result_from_a_pov(res, a_is_white=bool(awhite[j]))
            ]

    t0 = time.time()
    done = 0
    last_report = 0
    while queue or boards:
        _refill()
        # Reap finished / adjudicated / over-cap games, compacting the pool.
        kb: list[chess.Board] = []
        kg: list[int] = []
        ka: list[bool] = []
        kp: list[int] = []
        for j in range(len(boards)):
            b = boards[j]
            res: str | None = None
            if b.is_game_over(claim_draw=True):
                res = b.result(claim_draw=True)
            elif syzygy_tablebase is not None:
                res = _tb_adjudicate_result(b, syzygy_tablebase, max_pieces=tb_max_pieces)
            if res is None and gplies[j] >= int(max_plies):
                res = "*"  # not naturally decided and not TB-covered: adjudicate draw
            if res is not None:
                _record(j, res)
                done += 1
            else:
                kb.append(b)
                kg.append(gids[j])
                ka.append(awhite[j])
                kp.append(gplies[j])
        boards[:], gids[:], awhite[:], gplies[:] = kb, kg, ka, kp
        _refill()  # backfill the slots the reaped games freed — keep the pool full
        if not boards:
            break
        if done - last_report >= 64:
            print(
                f"[arena] rolling: {done}/{n_games} games done, "
                f"{len(boards)} active ({time.time() - t0:.0f}s)",
                flush=True,
            )
            # Running Elo over the pairs that have BOTH colorings finished so far,
            # so the standings stream in instead of only printing at the end.
            ready: list[float] = []
            for k in range(n_games // 2):
                w, blk = game_scores[2 * k], game_scores[2 * k + 1]
                if w is not None and blk is not None:
                    ready.append(w + blk)
            if ready:
                print(f"[arena] RUNNING Elo after {len(ready)} complete pairs:", flush=True)
                print_summary(summarize_pentanomial(pentanomial_counts(ready)))
            last_report = done
        active = list(range(len(boards)))
        a_to_move, b_to_move = split_active_by_side_to_move(active, boards, awhite)
        for model, idxs, sims, extra, gov in (
            (model_candidate, a_to_move, sims_candidate, dict(volatility_candidate or {}), gumbel_candidate),
            (model_reference, b_to_move, sims_reference, {}, gumbel_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=gov,
                **extra,
            )
            apply_actions_to_boards(boards, idxs, actions)
        for i in active:
            gplies[i] += 1

    return game_scores_to_pair_scores([s if s is not None else 0.5 for s in game_scores])


# ---------------------------------------------------------------------------
# Game play — matched time (real UCI engine subprocesses)
# ---------------------------------------------------------------------------

def play_paired_games_matched_time(
    candidate_ckpt: str,
    reference_ckpt: str,
    openings: list[chess.Board],
    *,
    device: str,
    ms_per_move: int,
    max_plies: int,
    uci_args: str,
) -> list[float]:
    """Pair-by-pair UCI match using the production engine inference path."""
    import chess.engine

    from scripts.match_vs_uci import _open_engine, _score_for_a, play_one_game

    limit = chess.engine.Limit(time=float(ms_per_move) / 1000.0)

    def engine_cmd(ckpt: str) -> str:
  # _open_engine shlex-splits the command, so quote anything path-like.
        cmd = (
            f"{shlex.quote(sys.executable)} -m chess_anti_engine.uci "
            f"--checkpoint {shlex.quote(ckpt)} --device {shlex.quote(device)}"
        )
        if uci_args:
            cmd = f"{cmd} {uci_args}"
        return cmd

    eng_a = eng_b = None
    pair_scores: list[float] = []
    try:
        print(f"[arena] starting candidate engine: {engine_cmd(candidate_ckpt)}")
        eng_a = _open_engine(engine_cmd(candidate_ckpt), cwd=str(REPO_ROOT))
        print(f"[arena] starting reference engine: {engine_cmd(reference_ckpt)}")
        eng_b = _open_engine(engine_cmd(reference_ckpt), cwd=str(REPO_ROOT))
        for pair_idx, opening in enumerate(openings):
            scores: list[float] = []
            for a_is_white in (True, False):
                eng_w, eng_b_side = (eng_a, eng_b) if a_is_white else (eng_b, eng_a)
                record = play_one_game(
                    eng_w, eng_b_side,
                    limit_w=limit, limit_b=limit,
                    enforce_nodes_w=False, enforce_nodes_b=False,
                    max_plies=int(max_plies),
                    start_board=opening,
                    game=(pair_idx, a_is_white),
                )
                scores.append(_score_for_a(record.result, a_is_white=a_is_white))
            pair_scores.append(scores[0] + scores[1])
            print(
                f"[arena] pair {pair_idx + 1}/{len(openings)}: "
                f"pair_score={pair_scores[-1]:.1f} "
                f"running_score={sum(pair_scores) / (2 * len(pair_scores)):.3f}",
                flush=True,
            )
    finally:
        for eng in (eng_a, eng_b):
            if eng is not None:
                try:
                    eng.quit()
                except chess.engine.EngineError:
                    pass  # already dead (e.g. crashed mid-match); keep closing the other
    return pair_scores


# ---------------------------------------------------------------------------
# Result record + JSONL log
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def production_config_hash() -> str:
    try:
        return hashlib.sha256(PRODUCTION_CONFIG.read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def build_result_record(
    summary: PentanomialSummary,
    *,
    mode: str,
    candidate: str,
    reference: str,
    openings_path: str,
    opening_plies: int,
    sims_candidate: int | None,
    sims_reference: int | None,
    ms_per_move: int | None,
    temperature: float,
    gumbel_add_noise: bool,
    max_plies: int,
    seed: int,
    device: str,
    duration_s: float,
    label: str | None = None,
    volatility_candidate: dict[str, float] | None = None,
) -> dict:
    elo_lo, elo_hi = summary.elo_ci95
    return {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config_hash": production_config_hash(),
        "mode": mode,
        "label": label,
        "volatility_candidate": volatility_candidate,
        "candidate": candidate,
        "reference": reference,
        "games": summary.games,
        "pairs": summary.pairs,
        "openings": openings_path,
        "opening_plies": opening_plies,
        "sims_candidate": sims_candidate,
        "sims_reference": sims_reference,
        "ms_per_move": ms_per_move,
        "temperature": temperature,
        "gumbel_add_noise": gumbel_add_noise,
        "max_plies": max_plies,
        "seed": seed,
        "device": device,
        "pentanomial": dict(zip(PAIR_LABELS, summary.counts)),
        "score": round(summary.score, 5),
        "score_se": round(summary.score_se, 5),
        "elo": None if summary.elo is None else round(summary.elo, 2),
        "elo_ci95": [
            None if elo_lo is None else round(elo_lo, 2),
            None if elo_hi is None else round(elo_hi, 2),
        ],
        "duration_s": round(duration_s, 1),
        "argv": sys.argv,
    }


def append_result(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def print_summary(summary: PentanomialSummary) -> None:
    counts = dict(zip(PAIR_LABELS, summary.counts))
    elo_lo, elo_hi = summary.elo_ci95

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:+.1f}"

    print()
    print(f"[arena] {summary.games} games ({summary.pairs} opening pairs)")
    print(f"[arena] pentanomial (candidate POV): {counts}")
    print(f"[arena] score: {summary.score:.4f} +/- {summary.score_se:.4f} (SE)")
    print(f"[arena] Elo: {fmt(summary.elo)}  95% CI: [{fmt(elo_lo)}, {fmt(elo_hi)}]")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_arena(
    *,
    candidate: str,
    reference: str,
    games: int,
    openings_path: Path,
    opening_plies: int,
    mode: str,
    sims_candidate: int,
    sims_reference: int,
    ms_per_move: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    device: str,
    seed: int,
    out_path: Path | None,
    uci_args: str = "",
    label: str | None = None,
    volatility_candidate: dict[str, float] | None = None,
    max_concurrent_games: int = 128,
    syzygy_path: str | None = None,
    tb_max_pieces: int = 6,
    compile_models: bool = True,
    rolling: bool = True,
    gumbel_candidate: dict[str, float] | None = None,
    gumbel_reference: dict[str, float] | None = None,
) -> dict:
    """Run one standardized arena and return (and optionally log) the record."""
    if games < 2 or games % 2 != 0:
        raise SystemExit("--games must be even and >= 2 (paired openings)")
    n_pairs = games // 2
    rng = np.random.default_rng(seed)

    print(f"[arena] sampling {n_pairs} openings from {openings_path} (seed={seed})")
    openings = load_paired_openings(
        openings_path, n_pairs=n_pairs, max_plies=opening_plies, rng=rng,
    )

    t0 = time.time()
    if mode == "matched_sims":
        from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

        print(f"[arena] loading candidate: {candidate}")
        model_candidate = load_model_from_checkpoint(candidate, device=device)
        print(f"[arena] loading reference: {reference}")
        model_reference = load_model_from_checkpoint(reference, device=device)
        if compile_models:
            import torch
            # Plain inductor compile (NOT reduce-overhead/cudagraphs, which recompile
            # per batch shape and OOM'd us). Auto-dynamic batch: a couple of warmup
            # recompiles in chunk 1, then cached + reused across the shrinking batch
            # sizes and into chunk 2.
            model_candidate = torch.compile(model_candidate)
            model_reference = torch.compile(model_reference)
            print("[arena] torch.compile ON (inductor, auto-dynamic batch)", flush=True)
        print(
            f"[arena] matched_sims: candidate={sims_candidate} sims/move, "
            f"reference={sims_reference} sims/move, temp={temperature}, "
            f"noise={gumbel_add_noise}"
        )
        # Syzygy adjudication: end each game the instant it reaches a covered
        # (<=N-man) position, so long endgame tails don't dominate the wall clock
        # (reuses match_vs_uci's WDL probe). Opened once, shared across chunks.
        syzygy_tb = None
        if syzygy_path:
            from scripts.match_vs_uci import _open_syzygy_tablebase
            try:
                syzygy_tb = _open_syzygy_tablebase(syzygy_path)
            except Exception as exc:
                syzygy_tb = None
                print(f"[arena] WARNING: syzygy open failed ({exc})", flush=True)
            print(
                f"[arena] syzygy adjudication {'ON' if syzygy_tb is not None else 'OFF'} "
                f"(<={tb_max_pieces}-man, {syzygy_path})",
                flush=True,
            )
        if rolling:
            # Rolling pool: fixed active-game count => fixed batch shape => compile
            # reuses one graph (no per-shape thrash), and the GPU never drains until
            # the very end (no per-chunk tail).
            print(
                f"[arena] ROLLING pool: keep {max_concurrent_games} games active, "
                f"start a fresh one as each finishes",
                flush=True,
            )
            pair_scores = play_paired_games_matched_sims_rolling(
                model_candidate, model_reference, openings,
                device=device, rng=rng,
                sims_candidate=sims_candidate, sims_reference=sims_reference,
                max_plies=max_plies, temperature=temperature,
                gumbel_add_noise=gumbel_add_noise,
                volatility_candidate=volatility_candidate,
                syzygy_tablebase=syzygy_tb, tb_max_pieces=tb_max_pieces,
                pool_size=int(max_concurrent_games),
                gumbel_candidate=gumbel_candidate, gumbel_reference=gumbel_reference,
            )
        else:
            # Chunked: plays each chunk of `max_concurrent_games` to completion
            # (drains per chunk). Numerically identical (pair scores concatenate).
            chunk_pairs = max(1, int(max_concurrent_games) // 2)
            n_chunks = (len(openings) + chunk_pairs - 1) // chunk_pairs
            pair_scores = []
            for ci in range(0, len(openings), chunk_pairs):
                sub = openings[ci:ci + chunk_pairs]
                print(
                    f"[arena] matched_sims chunk {ci // chunk_pairs + 1}/{n_chunks}: "
                    f"{len(sub)} pairs ({2 * len(sub)} games)",
                    flush=True,
                )
                pair_scores.extend(play_paired_games_matched_sims(
                    model_candidate, model_reference, sub,
                    device=device, rng=rng,
                    sims_candidate=sims_candidate, sims_reference=sims_reference,
                    max_plies=max_plies, temperature=temperature,
                    gumbel_add_noise=gumbel_add_noise,
                    volatility_candidate=volatility_candidate,
                    syzygy_tablebase=syzygy_tb, tb_max_pieces=tb_max_pieces,
                    gumbel_candidate=gumbel_candidate, gumbel_reference=gumbel_reference,
                ))
                print(f"[arena] RUNNING Elo after {2 * len(pair_scores)} games:", flush=True)
                print_summary(summarize_pentanomial(pentanomial_counts(pair_scores)))
        if syzygy_tb is not None:
            syzygy_tb.close()
    elif mode == "matched_time":
        print(f"[arena] matched_time: {ms_per_move}ms/move per side")
        pair_scores = play_paired_games_matched_time(
            candidate, reference, openings,
            device=device, ms_per_move=ms_per_move, max_plies=max_plies,
            uci_args=uci_args,
        )
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    duration_s = time.time() - t0

    summary = summarize_pentanomial(pentanomial_counts(pair_scores))
    print_summary(summary)

    record = build_result_record(
        summary,
        mode=mode,
        candidate=candidate,
        reference=reference,
        openings_path=str(openings_path),
        opening_plies=opening_plies,
        sims_candidate=sims_candidate if mode == "matched_sims" else None,
        sims_reference=sims_reference if mode == "matched_sims" else None,
        ms_per_move=ms_per_move if mode == "matched_time" else None,
        temperature=temperature,
        gumbel_add_noise=gumbel_add_noise,
        max_plies=max_plies,
        seed=seed,
        device=device,
        duration_s=duration_s,
        label=label,
        volatility_candidate=volatility_candidate,
    )
    if out_path is not None:
        append_result(record, out_path)
        print(f"[arena] result appended to {out_path}")
    return record


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arena knobs shared with scripts/elo_vs_sims.py."""
    p.add_argument("--games", type=int, default=1000,
                   help="total games; must be even — games/2 opening pairs (default: 1000)")
    p.add_argument("--openings", type=Path, default=None,
                   help="PGN(.zip)/Polyglot opening book "
                   "(default: the 8-move UHO book from configs/pbt2_small.yaml)")
    p.add_argument("--opening-plies", type=int, default=16,
                   help="book plies to apply per opening (default: 16 = 8 moves)")
    p.add_argument("--max-plies", type=int, default=300,
                   help="adjudicate as draw after this many plies (default: 300)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="move-selection temperature for matched_sims (default: 0.1)")
    p.add_argument("--no-gumbel-noise", action="store_true",
                   help="disable root Gumbel noise in matched_sims (fully deterministic; "
                   "self-play pairs then carry zero information)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_PATH,
                   help=f"JSONL results log (default: {DEFAULT_RESULTS_PATH})")


def _volatility_kwargs_from_args(args) -> dict[str, float] | None:
    """CANDIDATE-side volatility search kwargs, or None when all flags are off."""
    if float(args.volatility_q_scale) == 0.0 and float(args.volatility_fpu) == 0.0:
        return None
    if args.mode != "matched_sims":
        raise SystemExit(
            "--volatility-* flags require --mode matched_sims (the Python "
            "search path is slower, so matched_time would under-credit it)"
        )
    out = {
        "volatility_q_scale": float(args.volatility_q_scale),
        "volatility_fpu": float(args.volatility_fpu),
    }
    if args.volatility_anchor is not None:
        out["volatility_anchor"] = float(args.volatility_anchor)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidate", required=True, help="candidate checkpoint (trainer.pt or dir)")
    p.add_argument("--reference", required=True, help="reference checkpoint (trainer.pt or dir)")
    p.add_argument("--mode", choices=["matched_sims", "matched_time"],
                   default="matched_sims")
    p.add_argument("--sims", type=int, default=64,
                   help="MCTS sims/move for both sides in matched_sims (default: 64)")
    p.add_argument("--sims-candidate", type=int, default=None,
                   help="override candidate sims/move (defaults to --sims)")
    p.add_argument("--sims-reference", type=int, default=None,
                   help="override reference sims/move (defaults to --sims)")
    p.add_argument("--max-concurrent-games", type=int, default=128,
                   help="matched_sims: cap simultaneous games per batch to bound "
                        "GPU memory; total --games still played in chunks "
                        "(default: 128). Lower if you OOM on a small card.")
    p.add_argument("--syzygy", default=None,
                   help="matched_sims: colon-separated Syzygy dir(s) to adjudicate "
                        "games the instant they reach a covered position (kills "
                        "long endgame tails). e.g. data/syzygy_3-4-5")
    p.add_argument("--syzygy-max-pieces", type=int, default=6,
                   help="adjudicate positions with <= this many men (default: 6)")
    p.add_argument("--compile", choices=["auto", "on", "off"], default="auto",
                   help="matched_sims torch.compile policy (default: auto). "
                        "on=always compile; off=eager; auto=compile only when "
                        f"games*sims >= {AUTO_COMPILE_WORK_THRESHOLD} (else the "
                        "~2-4 min compile cost isn't worth it for a quick run).")
    p.add_argument("--no-compile", action="store_true",
                   help="matched_sims: back-compat alias for --compile off "
                        "(disables torch.compile regardless of --compile).")
    p.add_argument("--compile-cache-dir", type=Path, default=None,
                   help="matched_sims: torch.compile/triton cache root to reuse "
                        "(default: the production worker cache, so arena reuses "
                        "the kernels training already baked). Overridable; an "
                        "absent dir just means no reuse that run.")
    p.add_argument("--no-rolling", action="store_true",
                   help="matched_sims: disable the rolling pool and use fixed chunks "
                        "instead. Rolling (on by default) keeps a fixed pool of "
                        "--max-concurrent-games active, starting a fresh game as each "
                        "finishes => fixed batch shape so compile reuses one graph + "
                        "no per-chunk drain tail.")
    p.add_argument("--ms-per-move", type=int, default=100,
                   help="per-move wall clock for matched_time (default: 100)")
    p.add_argument("--uci-args", default="",
                   help="extra args appended to both UCI engine commands in matched_time "
                   "(e.g. '--no-compile')")
    p.add_argument("--label", default=None, help="free-form tag stored in the JSONL record")
    p.add_argument("--volatility-q-scale", type=float, default=0.0,
                   help="CANDIDATE-side volatility-aware sigma(q) exponent "
                        "(matched_sims only; forces the Python search path)")
    p.add_argument("--volatility-fpu", type=float, default=0.0,
                   help="CANDIDATE-side pessimistic FPU coefficient (matched_sims only)")
    p.add_argument("--volatility-anchor", type=float, default=None,
                   help="dataset-mean volatility anchor override (see exp_volatility_search.yaml)")
    p.add_argument("--cand-gumbel", default=None,
                   help="candidate gumbel knob overrides as k=v,k=v "
                        "(c_scale,c_visit,c_visit_root,topk,c_puct,fpu_reduction,halving_div). "
                        "Use the SAME checkpoint for --candidate/--reference + differing "
                        "gumbel here = a pure search-config Swiss (matched_sims).")
    p.add_argument("--ref-gumbel", default=None,
                   help="reference gumbel knob overrides (same k=v,k=v format as --cand-gumbel)")
    add_common_args(p)
    args = p.parse_args()

    openings_path = args.openings if args.openings is not None else default_openings_path()

    # Resolve the matched_sims torch.compile decision (matched_time is a UCI
    # subprocess and unaffected). --no-compile is the back-compat "off" alias and
    # wins over --compile. auto compiles only when the run does enough inference
    # work to amortize the ~2-4 min compile cost.
    sims_cand = int(args.sims if args.sims_candidate is None else args.sims_candidate)
    sims_ref = int(args.sims if args.sims_reference is None else args.sims_reference)
    effective_sims = max(sims_cand, sims_ref)
    compile_mode = "off" if args.no_compile else args.compile
    if compile_mode == "off":
        compile_models = False
        if args.mode == "matched_sims":
            print("[arena] compile=off -> EAGER", flush=True)
    elif compile_mode == "on":
        compile_models = True
        if args.mode == "matched_sims":
            print("[arena] compile=on -> COMPILE", flush=True)
    else:  # auto
        work = int(args.games) * int(effective_sims)
        compile_models = work >= AUTO_COMPILE_WORK_THRESHOLD
        if args.mode == "matched_sims":
            if compile_models:
                print(
                    f"[arena] compile=auto -> COMPILE "
                    f"(games*sims={work} >= {AUTO_COMPILE_WORK_THRESHOLD})",
                    flush=True,
                )
            else:
                print(
                    f"[arena] compile=auto -> EAGER (games*sims={work} < "
                    f"{AUTO_COMPILE_WORK_THRESHOLD}; compile overhead not worth it)",
                    flush=True,
                )

    # Point inductor/triton at the shared training cache BEFORE torch is imported
    # in run_arena, so the compile (when on) reuses already-baked kernels.
    if compile_models and args.mode == "matched_sims":
        cache_dir = (
            args.compile_cache_dir.expanduser().resolve()
            if args.compile_cache_dir is not None
            else default_compile_cache_dir()
        )
        _configure_shared_compile_cache(cache_dir)

    run_arena(
        candidate=args.candidate,
        reference=args.reference,
        games=args.games,
        max_concurrent_games=args.max_concurrent_games,
        syzygy_path=args.syzygy,
        tb_max_pieces=args.syzygy_max_pieces,
        compile_models=compile_models,
        rolling=not args.no_rolling,
        openings_path=openings_path,
        opening_plies=args.opening_plies,
        mode=args.mode,
        sims_candidate=sims_cand,
        sims_reference=sims_ref,
        ms_per_move=args.ms_per_move,
        max_plies=args.max_plies,
        temperature=args.temperature,
        gumbel_add_noise=not args.no_gumbel_noise,
        device=args.device,
        seed=args.seed,
        out_path=args.out,
        uci_args=args.uci_args,
        label=args.label,
        volatility_candidate=_volatility_kwargs_from_args(args),
        gumbel_candidate=_parse_gumbel_overrides(args.cand_gumbel),
        gumbel_reference=_parse_gumbel_overrides(args.ref_gumbel),
    )


_GUMBEL_INT_KEYS = {"topk", "halving_div", "simulations"}


def _parse_gumbel_overrides(spec: str | None) -> dict[str, float]:
    """Parse 'c_scale=0.025,c_visit=50,c_visit_root=900,topk=32' -> dict.

    Starts from the production PLAY_SEARCH_DEFAULTS optimum so that omitting a
    flag (or the whole spec) still plays the tuned settings instead of the stale
    GumbelConfig defaults; any parsed user keys override on top. int-coerces
    topk/halving_div/simulations; everything else is a float. Keys must be
    GumbelConfig fields (applied via dataclasses.replace downstream)."""
    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS

    out: dict[str, float] = {
        k: (int(v) if k in _GUMBEL_INT_KEYS else float(v))
        for k, v in PLAY_SEARCH_DEFAULTS.items()
    }
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--*-gumbel: expected k=v pairs, got {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        out[k] = int(v) if k in _GUMBEL_INT_KEYS else float(v)
    return out


if __name__ == "__main__":
    main()
