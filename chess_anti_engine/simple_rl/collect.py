"""One generation of pure self-play through the real finite ``play_batch``.

Both seats are the same net; ``selfplay_fraction`` is 1, so no Stockfish move
is ever played. ``play_batch`` still requires a Stockfish handle: every
self-play network turn submits a P1 label query (``manager._run_step``) and
the strict rule50 tablebase route refuses a handle without the explicit TB
contract (``strict_tb.validate_stockfish_contract``). There is no supported
"no labels" switch, so the collector runs the LIGHTEST valid engine -- a
one-worker pool at ``--sf-nodes`` (default 1) with MultiPV 1 -- and nothing
downstream reads what it writes: the trainer here runs with every SF loss
weight and the SF value fraction at 0.

Rows: every search-backed ply (``policy_row_retention='all_eligible'``,
playout cap off, diff-focus off; single-legal-move plies are skipped by the
search path as everywhere else), with ``GameConfig.record_root_stats`` on so
each row also carries the log prior, completed Q, visits, root value,
halving survivor and played move its target came from.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.simple_rl.common import (
    append_jsonl,
    input_ref,
    resolve_collect_dir,
    seed_for,
    sha256_file,
    write_json_atomic,
    write_manifest,
)
from chess_anti_engine.simple_rl.dup_sketch import sketch_path
from chess_anti_engine.simple_rl.openings import load_lines

ROW_META = "row_meta.npz"
GAMES = "games.jsonl"
STATS = "stats.json"


@dataclass(frozen=True)
class CollectSettings:
    games: int
    concurrency: int = 32
    sims: int = 64
    # Playout cap randomisation (KataGo): with probability full_prob a ply
    # searches `sims` and stores a policy row; otherwise it searches
    # `fast_sims`, is played, and stores no policy target (a value-only row
    # when fast_value_rows, else no row). full_prob 1.0 = off.
    full_prob: float = 1.0
    fast_sims: int = 16
    fast_value_rows: bool = False
    tree_reuse: bool = True
    topk: int = 16
    c_scale: float = 0.1
    c_visit: float = 50.0
    target_q_rescale: bool = True
    policy_temp: float = 1.0
    target_untempered_prior: bool = False
    vloss_weight: int = 1
    temp_start: float = 0.5
    temp_floor: float = 0.1
    temp_decay_plies: int = 10
    root_noise_scale: float = 1.0
    root_noise_after: float = 0.5
    root_noise_decay_start_move: int = 12
    root_noise_decay_moves: int = 3
    max_plies: int = 450
    syzygy: str | None = None
    sf_path: str = ""
    sf_nodes: int = 1
    sf_hash_mb: int = 16
    sf_timeout_s: float = 30.0
    book_mix: tuple[float, ...] = (0.95, 0.05)
    # "cycle" (default): one seeded permutation per book per pass, walked by a
    # cursor carried from generation to generation, so no TRAIN line is reused
    # before its book is used up. "independent": a fresh draw every generation
    # (without replacement only WITHIN the generation).
    opening_sampling: str = "cycle"
    # Cross-generation duplicate rate: constant-memory threshold-sampling
    # estimate (dup_sketch.py) with this many stored hashes; the exact set
    # computation over every earlier row is opt-in for small runs.
    dup_sketch_budget: int = 4_000_000
    exact_repeat_vs_previous: bool = False
    rows_per_shard: int = 20000
    device: str = "cpu"
    # Forward-batch cap of the CUDA evaluator. None (default) = the search's own
    # leaf-buffer size at this concurrency and topk (required_leaf_rows). It is a
    # SEARCH-SHAPE knob, not a memory one: gumbel_c caps its leaf buffer at the
    # evaluator's max batch. Before ABI 11 a leaf that found the buffer full was
    # NOT evaluated -- it was backed up with the ROOT's Q; since ABI 11 the batch
    # is flushed and resumed (counted, and a collect with flushes is refused
    # unsealed). At 64 sims / topk 16 / 128 games and the old
    # fixed 512 that hit most first-round children and corrupted the reply
    # side's targets (docs/simple_rl_loop.md, "Evaluator batch cap"). An explicit
    # value below required_leaf_rows is refused.
    eval_max_batch: int | None = None
    # Collect processes sharing the device (collect_generation). PINNED: each
    # process plays its own disjoint opening slice with its own seed, so K sets
    # the RNG streams and the batch composition. The driver defaults it to 3 on
    # CUDA (measured fast setting) and 1 on CPU; 1 is the same code path.
    collect_procs: int = 1
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _arch(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    arch = payload.get("arch") if isinstance(payload, dict) else None
    if not isinstance(arch, dict):
        raise ValueError(f"{checkpoint} has no embedded arch")
    return dict(arch)


def build_configs(settings: CollectSettings, arch: dict[str, Any], starts_path: Path) -> dict[str, Any]:
    """The exact six dataclasses handed to ``play_batch`` (kwargs by group)."""
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        OpponentConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.selfplay.opening import OpeningConfig

    strict = settings.syzygy is not None
    game = GameConfig(
        max_plies=int(settings.max_plies),
        selfplay_fraction=1.0,
        policy_row_retention="all_eligible",
        record_fast_ply_value=bool(settings.fast_value_rows),
        record_dense_sf_policy=False,
        record_prior_top1=True,
        record_root_stats=True,
        # Strict route: a game reaching the cap is UNRESOLVED and keeps no rows
        # (finalize), so Stockfish never decides an outcome. Without tablebases
        # the cap is scored a draw: no SF probability can exceed 1.0.
        timeout_adjudication_threshold=0.99 if strict else 1.0,
        policy_encoding=str(arch.get("policy_encoding", "lc0_1858")),
        input_history_encoding=str(arch["input_history_encoding"]),
        input_extra_features=str(arch["input_extra_features"]),
        history_rep_fix=bool(arch.get("history_rep_fix", False)),
        # The F12 collection's tablebase route: strict rule50 TB in search,
        # adjudication of every TB-eligible game, TB-correct policy rescoring.
        syzygy_outcome_mode="rule50_match_v1" if strict else "theoretical_wdl",
        syzygy_path=settings.syzygy,
        stockfish_syzygy_path=settings.syzygy,
        syzygy_rescore_policy=strict,
        syzygy_adjudicate=strict,
        syzygy_adjudicate_fraction=1.0,
        syzygy_in_search=strict,
    )
    search = SearchConfig(
        simulations=int(settings.sims),
        fast_simulations=int(settings.fast_sims) if float(settings.full_prob) < 1.0 else int(settings.sims),
        playout_cap_fraction=float(settings.full_prob),
        gumbel_tree_reuse=bool(settings.tree_reuse),
        mcts_type="gumbel",
        gumbel_topk=int(settings.topk),
        gumbel_c_scale=float(settings.c_scale),
        gumbel_c_visit=float(settings.c_visit),
        gumbel_target_q_rescale=bool(settings.target_q_rescale),
        gumbel_policy_temp=float(settings.policy_temp),
        gumbel_target_untempered_prior=bool(settings.target_untempered_prior),
        gumbel_vloss_weight=int(settings.vloss_weight),
        gumbel_scale=float(settings.root_noise_scale),
        gumbel_scale_after=float(settings.root_noise_after),
        gumbel_scale_decay_start_move=int(settings.root_noise_decay_start_move),
        gumbel_scale_decay_moves=int(settings.root_noise_decay_moves),
    )
    temp = TemperatureConfig(
        temperature=float(settings.temp_start), selfplay_temperature=float(settings.temp_start),
        decay_start_move=0, selfplay_decay_start_move=0,
        decay_moves=int(settings.temp_decay_plies), selfplay_decay_moves=int(settings.temp_decay_plies),
        endgame=float(settings.temp_floor), selfplay_endgame=float(settings.temp_floor),
        after=float(settings.temp_floor), schedule_plies_from_start=True,
    )
    opening = OpeningConfig(
        opening_book_path=None, opening_book_prob=0.0, random_start_plies=0,
        opening_fen_list_path=str(starts_path), opening_fen_prob=0.0,
        opening_fen_dole_per_iter=1, opening_fen_selfplay_only=True,
    )
    return {"game": game, "search": search, "temp": temp, "opening": opening,
            "opponent": OpponentConfig(), "diff_focus": DiffFocusConfig(enabled=False)}


def sample_openings(train_jsonl: Path, *, games: int, book_mix: tuple[float, ...], seed: int,
                    exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """``games`` TRAIN lines, without replacement within the generation.

    Each game first draws its book by ``book_mix`` and then an unused line of
    that book; a book that runs dry falls back to the others.
    """
    rows = load_lines(train_jsonl)
    by_book: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if exclude_ids and row["id"] in exclude_ids:
            continue
        by_book.setdefault(int(row["book"]), []).append(row)
    rng = np.random.default_rng(seed)
    for book in by_book:
        order = rng.permutation(len(by_book[book]))
        by_book[book] = [by_book[book][int(i)] for i in order]
    books = sorted(by_book)
    weights = np.asarray([book_mix[b - 1] if b - 1 < len(book_mix) else 0.0 for b in books], dtype=np.float64)
    if games > sum(len(v) for v in by_book.values()):
        raise ValueError(f"{games} games requested but only {sum(len(v) for v in by_book.values())} TRAIN lines")
    picked: list[dict[str, Any]] = []
    cursor = dict.fromkeys(books, 0)
    for _ in range(games):
        avail = np.asarray([cursor[b] < len(by_book[b]) for b in books])
        w = np.where(avail, weights, 0.0)
        if w.sum() <= 0:
            w = avail.astype(np.float64)
        book = books[int(rng.choice(len(books), p=w / w.sum()))]
        picked.append(by_book[book][cursor[book]])
        cursor[book] += 1
    return picked


OPENING_SAMPLING = ("cycle", "independent")


def _books(train_jsonl: Path) -> dict[int, list[dict[str, Any]]]:
    by_book: dict[int, list[dict[str, Any]]] = {}
    for row in load_lines(train_jsonl):
        by_book.setdefault(int(row["book"]), []).append(row)
    return by_book


def _cycle_permutation(base_seed: int, book: int, n_pass: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed_for(base_seed, "opening-cycle", book, n_pass)).permutation(n)


def cycle_openings(train_jsonl: Path, *, games: int, book_mix: tuple[float, ...], base_seed: int, generation: int,
                   start: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """``games`` TRAIN lines from a cursor that walks the whole book before
    reusing any line.

    Each book has one seeded permutation per PASS (``seed_for(base, book,
    pass)``); the cursor ``{book: {pass, pos}}`` is carried from the previous
    generation's collect manifest (``start``; None = a fresh run at pass 1).
    Each game draws its book by ``book_mix`` and takes that book's next line;
    an exhausted book starts its next pass (logged). Returns (lines, cursor
    after this generation, log lines).
    """
    by_book = _books(train_jsonl)
    books = sorted(by_book)
    sha = sha256_file(train_jsonl)
    if start is not None and start.get("train_sha256") != sha:
        raise ValueError("the TRAIN openings changed since the previous generation's cursor "
                         f"({start.get('train_sha256')} -> {sha}); a cursor over another book is meaningless")
    cursor = {b: dict((start or {}).get("books", {}).get(str(b), {"pass": 1, "pos": 0})) for b in books}
    for b in books:
        if int(cursor[b].get("lines", len(by_book[b]))) != len(by_book[b]):
            raise ValueError(f"book {b} line count changed under the cursor")
    perms = {b: _cycle_permutation(base_seed, b, int(cursor[b]["pass"]), len(by_book[b])) for b in books}
    weights = np.asarray([book_mix[b - 1] if b - 1 < len(book_mix) else 0.0 for b in books], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("book_mix gives every book weight 0")
    rng = np.random.default_rng(seed_for(base_seed, "openings", generation))
    picked: list[dict[str, Any]] = []
    log: list[str] = []
    for _ in range(int(games)):
        b = books[int(rng.choice(len(books), p=weights / weights.sum()))]
        c = cursor[b]
        if int(c["pos"]) >= len(by_book[b]):
            c["pass"], c["pos"] = int(c["pass"]) + 1, 0
            perms[b] = _cycle_permutation(base_seed, b, int(c["pass"]), len(by_book[b]))
            log.append(f"generation {generation}: book {b} exhausted; starting pass {c['pass']}")
        picked.append(by_book[b][int(perms[b][int(c["pos"])])])
        c["pos"] = int(c["pos"]) + 1
    after = {"train_sha256": sha,
             "books": {str(b): {"pass": int(cursor[b]["pass"]), "pos": int(cursor[b]["pos"]),
                                "lines": len(by_book[b])} for b in books}}
    return picked, after, log


def previous_cursor(gen_dir: Path, generation: int) -> dict[str, Any] | None:
    """The opening cursor the previous generation's (sealed) collect left."""
    if generation <= 1:
        return None
    from chess_anti_engine.simple_rl.common import read_manifest

    prev = read_manifest(Path(gen_dir).parent.parent / f"gen_{generation - 1:04d}" / "collect")
    if prev is None:
        raise ValueError(f"generation {generation - 1} collect is not sealed; its opening cursor is unknown")
    return prev.get("opening_cursor")  # None when it used independent sampling


def x_hash(x: np.ndarray) -> np.uint64:
    """Identity of an encoded input (position AND its history planes)."""
    digest = hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float16).tobytes(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _open_stockfish(settings: CollectSettings) -> Any:
    from chess_anti_engine.selfplay.strict_tb import validate_stockfish_contract
    from chess_anti_engine.stockfish.pool import StockfishPool

    if not settings.sf_path:
        raise ValueError("--sf-path is required: play_batch cannot run without a Stockfish handle "
                         "(see chess_anti_engine/simple_rl/collect.py)")
    strict = settings.syzygy is not None
    pool = StockfishPool(
        path=settings.sf_path, nodes=int(settings.sf_nodes), num_workers=1, multipv=1,
        hash_mb=int(settings.sf_hash_mb), syzygy_path=settings.syzygy,
        syzygy_50_move_rule=True if strict else None, syzygy_probe_limit=6 if strict else None,
        read_timeout_s=float(settings.sf_timeout_s),
    )
    if strict:
        assert settings.syzygy is not None
        validate_stockfish_contract(pool, settings.syzygy)
    return pool


def required_leaf_rows(settings: CollectSettings) -> int:
    """The leaf buffer one collect search call asks for, before any evaluator cap.

    Every network turn hands ``run_gumbel_root_many_c`` up to ``concurrency``
    boards together with the persistent game tree, and a supplied tree turns
    the 2-group pipeline off at every board count, so the single-buffer
    expression applies at the full concurrency.
    """
    from chess_anti_engine.mcts.gumbel_c import leaf_buffer_rows

    boards = max(1, min(int(settings.concurrency), int(settings.games)))
    return int(leaf_buffer_rows(boards, topk=int(settings.topk), pipelined=False))


# The fixed DirectGPUEvaluator cap every CUDA collect used before
# --eval-max-batch existed (2026-11-20); old manifests do not record it.
LEGACY_EVAL_MAX_BATCH = 512


def capped_collect_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Non-None when a SEALED collect ran at an evaluator cap that could absorb leaves.

    ``_mcts_tree.c`` (before the buffer-full fix) backs a leaf that finds the
    buffer full with the root's Q instead of evaluating it -- wrong-signed at
    every odd depth and fabricated at every even one. A search call asks for at
    most boards x topk leaves in one rep (the accumulate guard keeps later reps
    inside the buffer), so a cap below that could bind. This is the PRECISE
    could-bind bound, unlike the collect-time refusal, which conservatively
    compares against the 2x buffer. Read from the manifest's own config, so it
    works on old-format collects that predate the stats fields; the realised cap
    comes from stats when recorded, else the setting, else the legacy fixed 512.
    """
    settings = dict(manifest.get("config", {}).get("settings", {}))
    if not str(settings.get("device", "cpu")).startswith("cuda"):
        return None
    stats = manifest.get("stats") or {}
    realised = stats.get("eval_max_batch_realised")
    source = "stats"
    if realised is None:
        realised, source = settings.get("eval_max_batch"), "settings"
    if realised is None:
        realised, source = LEGACY_EVAL_MAX_BATCH, "legacy_fixed_512"
    boards = max(1, min(int(settings.get("concurrency", 32)), int(settings.get("games", 1))))
    request = boards * int(settings.get("topk", 16))
    if int(realised) >= request:
        return None
    return {"eval_max_batch": int(realised), "eval_max_batch_source": source, "boards": boards,
            "topk": int(settings.get("topk", 16)), "leaf_request_per_rep": request}


def realised_eval_max_batch(settings: CollectSettings) -> int:
    """``eval_max_batch``, defaulted to ``required_leaf_rows``; a smaller explicit
    value is refused, because it would silently drop leaf evaluations."""
    need = required_leaf_rows(settings)
    if settings.eval_max_batch is None:
        return need
    if int(settings.eval_max_batch) < need:
        # Conservative: `need` is the 2x leaf BUFFER, while one rep requests at
        # most boards x topk leaves, so a value between the two would not
        # actually bind (e.g. 512 at 17-32 games, topk 16) and is refused anyway.
        raise ValueError(
            f"--eval-max-batch {settings.eval_max_batch} is below the {need} leaf rows one search call asks for "
            f"at --concurrency {settings.concurrency} / --topk {settings.topk}: the C search would have to "
            "re-batch (flush) its leaf buffer, a different search than the pinned settings describe (before "
            "ABI 11 it backed those leaves up with the root's Q). Omit --eval-max-batch (auto) or pass at least "
            f"{need}, or lower --concurrency")
    return int(settings.eval_max_batch)


def validate_settings(settings: CollectSettings) -> None:
    """Every refusal a collect would raise from its settings alone, raised up front.

    The loop driver calls this BEFORE pinning run_config.json, so a refused
    setting is never pinned (dropping it again would need --allow-config-change).
    """
    if settings.opening_sampling not in OPENING_SAMPLING:
        raise ValueError(f"opening_sampling must be one of {OPENING_SAMPLING}")
    if not 1 <= int(settings.collect_procs) <= int(settings.games):
        raise ValueError(f"--collect-procs {settings.collect_procs} must be between 1 and --games {settings.games}")
    if str(settings.device).startswith("cuda"):
        realised_eval_max_batch(settings)  # a leaf-dropping evaluator cap


def _evaluator(model: Any, settings: CollectSettings) -> Any:
    """The collect's evaluator. On CUDA it always replays CUDA graphs of its own
    eager dense forward (the one fast path; outputs measured bitwise identical
    to the eager forward, PR #1414). Off CUDA graphs cannot run: play_batch
    builds an eager LocalModelEvaluator. Either way the realised mode is
    recorded (``evaluator_graph_record``)."""
    if not str(settings.device).startswith("cuda"):
        return None  # play_batch builds an eager LocalModelEvaluator on CPU
    from chess_anti_engine.inference import DirectGPUEvaluator

    return DirectGPUEvaluator(model, device=settings.device, max_batch=realised_eval_max_batch(settings),
                              use_amp=True, amp_dtype="bfloat16", n_slots=1, input_bf16=False,
                              legal_bf16=False, pad_to_compiled_buckets=False,
                              cuda_graphs=True)


def evaluator_graph_record(evaluator: Any) -> dict[str, Any]:
    """What the evaluator REALISED: graphs on/off and its replay/fallback counters."""
    max_batch = getattr(evaluator, "_max_batch", None)
    graphs = bool(getattr(evaluator, "cuda_graphs", False))
    return {"cuda_graphs": graphs,
            # realised evaluator mode: graphs on CUDA; eager only where graphs cannot run (CPU)
            "evaluator_mode": "cuda_graphs" if graphs else ("eager_cpu" if evaluator is None else "eager"),
            "cuda_graph_stats": getattr(evaluator, "cuda_graph_stats", None),
            # read off the evaluator the search consumed (None on CPU: no cap)
            "eval_max_batch_realised": None if max_batch is None else int(max_batch)}


PLAN = "plan.json"
PROCS_DIR = "procs"
PROC_DONE = "done.json"
PAUSED_ACK = "PAUSED"
RESUMES = "resumes.json"
# Peak VRAM per collect process, fitted (least squares) to the 2026-11-20 RTX 5090
# measurements (SL base, nvidia-smi MiB, total / K): 16 sims x 64 / 96 / 128 games
# = 2858 / 4321 / 4806, 32 sims x 128 = 6553, 64 sims x 128 / 256 = 7368 / 12085.
# Per process ~ 48 MiB per sim + 1.11 MiB per leaf row (the evaluator cap the
# search asks for: 2048 / 3072 / 4096 / 8192 rows at 64 / 96 / 128 / 256 games,
# topk 16), within -8% / +9% of every one of those points; x 1.10 safety.
# Above ~8k leaf rows in one process the peak grows faster than that line: 64 sims
# x 512 games in ONE process (16384 rows) peaked at 28,515 MiB, which the line puts
# at 21,260 (-25%). The steep segment through 1 x 256 and 1 x 512 (64 sims) is
# ~2.006 MiB per row - 4348 MiB; the model takes the larger of the two.
_VRAM_MIB_PER_SIM = 48.0
_VRAM_MIB_PER_LEAF_ROW = 1.11
_VRAM_STEEP_MIB_PER_LEAF_ROW = 2.006
_VRAM_STEEP_MIB_OFFSET = -4348.0
_VRAM_SAFETY = 1.10
# Usable fraction of the card's memory: 8 x 96 at 16 sims peaked at 31,048 MiB
# (95% of the 5090) and ran 22% slower than 6 x 96 at 25,926 MiB (80%), which ran
# at full speed. Refuse above 83% of total (27.0 GiB on the 5090), never at 100%.
VRAM_USABLE_FRACTION = 0.83

# Default (collect processes, games per process) on CUDA when neither flag is
# given, by --sims: the first row whose max_sims >= sims. Measured 2026-11-20
# (docs/simple_rl_loop.md, "Collect processes"): at 16 sims 8 x 64 forwards
# 9,689 rows/s (0.88 of the ~11k forward ceiling, 1.85M training rows/h, 22.9 GB)
# against 7,113 (0.65) at 3 x 128; at 64 sims 3 x 128 reaches 9,487 (0.86, 22.1
# GB). 32 sims has no K >= 6 measurement yet, so it keeps 3 x 128 (8,679, 0.79).
# Both values are pinned in run_config like explicit flags.
CUDA_COLLECT_SHAPES: tuple[tuple[int | None, int, int], ...] = (
    (16, 8, 64),
    (None, 3, 128),
)
CPU_COLLECT_SHAPE = (1, 64)  # one process, the long-standing --concurrency default


class PauseGate:
    """Cooperative pause of one collect process: ``play_batch``'s ``pause_fn``.

    True while ``<run>/PAUSE`` or any ``--pause-marker`` exists. ``play_batch``
    checks it once per step (one network turn of every in-flight game) and,
    while it holds, schedules no search, forward or Stockfish work and keeps
    every in-flight game in memory; it continues exactly where it stopped once
    the marker is gone. Nothing on that path has a wall-clock deadline (the
    engine read timeout applies only inside a search, and searches already
    submitted finish on the pool's own threads), so a pause may last any time.
    While holding, ``<proc_dir>/PAUSED`` exists (the acknowledgement), and every
    interval is recorded so throughput can exclude it. ``STOP`` is not a pause:
    the driver honours it at the next stage boundary.
    """

    def __init__(self, run_dir: Path, markers: tuple[str, ...], ack: Path) -> None:
        from chess_anti_engine.simple_rl.common import PAUSE_FILE

        self._paths = [Path(run_dir) / PAUSE_FILE, *(Path(m).expanduser() for m in markers)]
        self._ack = ack
        self._since: float | None = None
        self.intervals: list[tuple[float, float]] = []

    def __call__(self) -> bool:
        held = any(p.exists() or p.is_symlink() for p in self._paths)
        now = time.time()
        if held and self._since is None:
            self._since = now
            self._ack.write_text(f"{now}\n")
        elif not held and self._since is not None:
            self.intervals.append((self._since, now))
            self._since = None
            self._ack.unlink(missing_ok=True)
        return held

    def wait(self) -> None:
        """Hold before any GPU work starts (model load, graph capture)."""
        while self():
            time.sleep(0.25)

    @property
    def paused_seconds(self) -> float:
        return float(sum(b - a for a, b in self.intervals))


def default_collect_shape(device: str, sims: int) -> tuple[int, int]:
    """(collect processes, games per process) when the flags are not given."""
    if not str(device).startswith("cuda"):
        return CPU_COLLECT_SHAPE
    for max_sims, procs, games in CUDA_COLLECT_SHAPES:
        if max_sims is None or int(sims) <= max_sims:
            return procs, games
    raise AssertionError("CUDA_COLLECT_SHAPES must end with a catch-all row")


def proc_seed(settings: CollectSettings, generation: int, proc: int) -> int:
    """Seed of collect process ``proc``. K = 1 keeps the one-process seed every
    earlier collect used, so a K = 1 collect is byte-identical to one made before
    collect processes existed."""
    if int(settings.collect_procs) == 1:
        return seed_for(settings.seed, "collect", generation)
    return seed_for(settings.seed, "collect", generation, "proc", int(proc), "of", int(settings.collect_procs))


def proc_openings(openings: list[dict[str, Any]], proc: int, procs: int) -> list[dict[str, Any]]:
    """Process ``proc``'s disjoint share of the generation's opening list."""
    return openings[int(proc)::int(procs)]


def vram_need_gib(settings: CollectSettings) -> float:
    """Conservative peak VRAM of ``collect_procs`` processes at these settings (GiB)."""
    rows = realised_eval_max_batch(settings)
    per_mib = max(_VRAM_MIB_PER_SIM * int(settings.sims) + _VRAM_MIB_PER_LEAF_ROW * rows,
                  _VRAM_STEEP_MIB_PER_LEAF_ROW * rows + _VRAM_STEEP_MIB_OFFSET)
    return _VRAM_SAFETY * per_mib * int(settings.collect_procs) / 1024.0


def _nvidia_smi_target(index: int) -> str | None:
    """The nvidia-smi selector (an index or a ``GPU-`` UUID) for torch's ``cuda:index``,
    or None when it cannot be named unambiguously.

    torch numbers the devices ``CUDA_VISIBLE_DEVICES`` leaves visible, in
    ``CUDA_DEVICE_ORDER`` (default FASTEST_FIRST); nvidia-smi numbers every card
    in PCI bus order. A visible-devices entry is mapped through; a MIG or
    otherwise unrecognised entry is not."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if index >= len(entries):
        return None
    entry = entries[index]
    return entry if entry.isdigit() or entry.startswith("GPU-") else None


def _gpu_memory(settings: CollectSettings) -> tuple[int, int, str]:
    """(free, total) bytes of the collect's device, and which instrument read them.

    With K > 1 the driver plays nothing on the GPU, so it asks nvidia-smi rather
    than ``torch.cuda.mem_get_info``: that call would create a CUDA context in
    the DRIVER that lives for the whole collect, outside the VRAM model. K = 1
    plays in the driver, which needs the context anyway. nvidia-smi is used only
    when it can name torch's device unambiguously (a UUID, a single card, or
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``); otherwise, and when it fails, torch reads
    it and the source says so."""
    import os

    import torch

    device = torch.device(settings.device)
    if int(settings.collect_procs) > 1:
        import subprocess

        target = _nvidia_smi_target(device.index or 0)  # "cuda" alone means the current (first) device
        try:
            if target is None:
                raise ValueError("device not nameable for nvidia-smi")
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 check=True, timeout=30).stdout
            cards = [[c.strip() for c in ln.split(",")] for ln in out.strip().splitlines() if ln.strip()]
            pci_order = os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            if target.startswith("GPU-"):
                (card,) = [c for c in cards if c[1] == target]
            elif len(cards) == 1 or pci_order:
                (card,) = [c for c in cards if c[0] == target]
            else:
                raise ValueError("several cards and torch's order is not PCI bus order")
            return int(float(card[2]) * 2**20), int(float(card[3]) * 2**20), "nvidia-smi"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass  # no usable nvidia-smi reading: fall back to torch (and its context)
    free, total = torch.cuda.mem_get_info(device)
    return int(free), int(total), "torch"


def check_vram(settings: CollectSettings) -> dict[str, Any] | None:
    """Refuse before any work when K processes would not fit in usable GPU memory.

    Usable = min(free, VRAM_USABLE_FRACTION x total): past ~83% of the card the
    collect slows down instead of failing (measured). K is pinned (it sets every
    process's RNG stream), so it is never stepped down silently: the refusal names
    the value that would fit."""
    if not str(settings.device).startswith("cuda"):
        return None
    free, total, source = _gpu_memory(settings)
    usable = min(free, VRAM_USABLE_FRACTION * total) / 2**30
    need = vram_need_gib(settings)
    record = {"free_gib": free / 2**30, "total_gib": total / 2**30, "usable_gib": usable, "need_gib": need,
              "collect_procs": int(settings.collect_procs), "source": source}
    if need > usable:
        per = need / int(settings.collect_procs)
        fit = int(usable // per)
        advice = (f"pass --collect-procs {fit} explicitly (a pinned run also needs --allow-config-change: K "
                  "sets the RNG streams)" if fit >= 1 else
                  f"no process count fits at --concurrency {settings.concurrency}: lower --concurrency")
        raise ValueError(
            f"--collect-procs {settings.collect_procs} x --concurrency {settings.concurrency} at --sims "
            f"{settings.sims} needs about {need:.1f} GiB of GPU memory ({per:.1f} GiB per process, fitted peak x "
            f"{_VRAM_SAFETY}), but only {usable:.1f} GiB is usable: {free / 2**30:.1f} of {total / 2**30:.1f} GiB "
            f"free, and the measured usable limit is ~{VRAM_USABLE_FRACTION:.0%} of the card (31.0 GB of 32 ran "
            f"22% slower than 25.9 GB). Free the GPU, lower --concurrency, or {advice}")
    return record


@dataclass(frozen=True)
class ProcSpec:
    """Everything one collect process needs; picklable for a spawned child."""

    proc: int
    gen_dir: str
    generation: int
    checkpoint: str
    starts_path: str
    settings: CollectSettings
    openings: tuple[dict[str, Any], ...]
    seed: int
    sf_factory: Any
    torch_threads: int
    run_dir: str
    pause_markers: tuple[str, ...]
    parent_pid: int

    @property
    def proc_dir(self) -> Path:
        return Path(self.gen_dir) / PROCS_DIR / f"{self.proc:02d}"


def _plan(settings: CollectSettings, generation: int, checkpoint: Path, openings: list[dict[str, Any]],
          cursor_before: dict[str, Any] | None) -> dict[str, Any]:
    """What an interrupted collect must match for its finished processes to be kept."""
    k = int(settings.collect_procs)
    return {"generation": int(generation), "collect_procs": k, "checkpoint_sha256": sha256_file(checkpoint),
            "settings": json.loads(json.dumps(asdict(settings), default=str)),
            "opening_cursor_before": cursor_before,
            "procs": [{"proc": p, "seed": proc_seed(settings, generation, p),
                       "opening_ids": [row["id"] for row in proc_openings(openings, p, k)]} for p in range(k)]}


def _proc_done(spec: ProcSpec) -> dict[str, Any] | None:
    """The finished record of a process whose output is still all on disk."""
    path = spec.proc_dir / PROC_DONE
    if not path.is_file():
        return None
    done = json.loads(path.read_text())
    shards = sorted((spec.proc_dir / "shards").glob("shard_*.zarr"))
    if len(shards) != int(done["shards"]) or sha256_file(spec.proc_dir / GAMES) != done["games_sha256"]:
        return None
    return done


def _commit_path(shards: Path, index: int) -> Path:
    return shards / f"shard_{index:05d}.commit.json"


def _shard_meta_path(shards: Path, index: int) -> Path:
    return shards / f"shard_{index:05d}.meta.npz"


def _write_shard_meta(path: Path, arrays: dict[str, Any]) -> None:
    np.savez(path, **arrays)  # not atomic: a torn file has no commit record and is discarded on resume


def _committed_prefix(proc_dir: Path) -> tuple[int, int, list[str]]:
    """The shards and games.jsonl lines a killed process may keep.

    A shard is committed once its ``.commit.json`` (written atomically, LAST,
    after the zarr store and the meta file) exists and still agrees with the
    disk: the meta file loads with the recorded row count and the first
    ``games_lines`` lines of games.jsonl hash to the recorded digest. Shards
    hold whole games and games.jsonl lines are written before the flush that
    commits their rows, so the committed lines are exactly the games whose rows
    are all in committed shards. Committed shards are a contiguous prefix; the
    first that fails ends it. Returns (shards kept, lines kept, the kept lines)."""
    shards = proc_dir / "shards"
    games = proc_dir / GAMES
    raw = games.read_bytes() if games.is_file() else b""
    complete = raw[: raw.rfind(b"\n") + 1]  # a torn last line is never committed
    lines = complete.decode().splitlines(keepends=True)
    kept_shards, kept_lines = 0, 0
    while (commit := _commit_path(shards, kept_shards)).is_file():
        try:
            rec = json.loads(commit.read_text())
            n = int(rec["games_lines"])
            with np.load(_shard_meta_path(shards, kept_shards)) as meta:
                rows = len(meta["game_id"])
        except (OSError, ValueError, KeyError, EOFError):
            break
        prefix = "".join(lines[:n]).encode()
        if (rows != int(rec["rows"]) or n > len(lines) or n < kept_lines
                or hashlib.sha256(prefix).hexdigest() != rec["games_sha256"]
                or not (shards / f"shard_{kept_shards:05d}.zarr").is_dir()):
            break
        kept_shards, kept_lines = kept_shards + 1, n
    return kept_shards, kept_lines, lines[:kept_lines]


def _repair_proc_dir(spec: ProcSpec) -> dict[str, Any] | None:
    """Make a killed process's directory resumable at game granularity, ON DISK,
    before anything is appended to it.

    Keeps the committed prefix (``_committed_prefix``) and deletes everything
    else: uncommitted or torn shards, temporary zarr stores, orphaned meta or
    commit files, a stale done.json, and games.jsonl lines whose rows were never
    flushed (rewritten atomically). Returns the resume record (kept vs replayed
    openings and the re-seed), or None for a fresh start. Idempotent: a kill
    during or after the repair repairs to the same state next time."""
    proc_dir = spec.proc_dir
    shards = proc_dir / "shards"
    if not proc_dir.exists():
        return None
    (proc_dir / PROC_DONE).unlink(missing_ok=True)  # _proc_done already refused it
    (proc_dir / PAUSED_ACK).unlink(missing_ok=True)
    kept_shards, _, lines = _committed_prefix(proc_dir)
    shards.mkdir(parents=True, exist_ok=True)
    keep = {f"shard_{i:05d}{ext}" for i in range(kept_shards) for ext in (".zarr", ".meta.npz", ".commit.json")}
    discarded = sorted(p.name for p in shards.iterdir() if p.name not in keep)
    for p in shards.iterdir():
        if p.name not in keep:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    tmp = proc_dir / f".{GAMES}.repair"
    tmp.write_text("".join(lines))
    tmp.replace(proc_dir / GAMES)
    previous = json.loads((proc_dir / RESUMES).read_text()) if (proc_dir / RESUMES).is_file() else []
    kept = [json.loads(ln) for ln in lines]
    remaining = Counter(row["id"] for row in spec.openings)
    for g in kept:
        if remaining[g["opening_id"]] <= 0:
            raise ValueError(f"collect process {spec.proc}: kept game {g['game_id']} played opening "
                             f"{g['opening_id']!r}, which is not (or no longer) doled to it")
        remaining[g["opening_id"]] -= 1
    replay = []
    for row in spec.openings:
        if remaining[row["id"]] > 0:
            replay.append(row["id"])
            remaining[row["id"]] -= 1
    attempt = len(previous) + 1
    # Nothing kept: a plain restart on the process's own seed, so its rows are
    # those of an uninterrupted run. Anything kept: a fresh seed per attempt.
    seed = seed_for(spec.seed, "resume", attempt) if kept else spec.seed
    record = {"attempt": attempt, "kept_games": len(kept), "kept_shards": kept_shards,
              "kept_rows": int(sum(int(g["rows"]) for g in kept)),
              "kept_opening_ids": [g["opening_id"] for g in kept], "replayed_opening_ids": replay,
              "discarded": discarded, "seed": seed}
    write_json_atomic(proc_dir / RESUMES, [*previous, record])
    return record


def _play_proc(spec: ProcSpec) -> dict[str, Any]:
    """Play one process's share of the generation into its own directory.

    The same code for every K: K = 1 runs it in the driver's process, K > 1 in
    spawned children. Writes ``shards/``, ``games.jsonl`` and, last,
    ``done.json`` (its counts, evaluator record and flush count)."""
    import torch

    from chess_anti_engine.replay.shard import ShardMeta, samples_to_arrays, save_local_shard_arrays
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    settings = spec.settings
    proc_dir = spec.proc_dir
    # A killed process keeps its committed games and replays only the rest of
    # its openings, with a fresh seed per attempt: its rows are therefore NOT
    # those an unkilled process would have made (no digest identity).
    resume = _repair_proc_dir(spec)
    (proc_dir / "shards").mkdir(parents=True, exist_ok=True)
    resumes = json.loads((proc_dir / RESUMES).read_text()) if resume is not None else []
    seed = int(resume["seed"]) if resume is not None else spec.seed
    openings = list(spec.openings)
    if resume is not None:  # the replayed openings, in doled order
        openings, left = [], Counter(resume["replayed_opening_ids"])
        for row in spec.openings:
            if left[row["id"]] > 0:
                openings.append(row)
                left[row["id"]] -= 1
    gate = PauseGate(Path(spec.run_dir), spec.pause_markers, proc_dir / PAUSED_ACK)
    # The process clock starts BEFORE the pre-load hold, so wall_seconds contains
    # every pause interval it records and active = wall - paused stays >= 0.
    t0, started_at = time.perf_counter(), time.time()
    gate.wait()
    checkpoint = Path(spec.checkpoint)
    arch = _arch(checkpoint)
    cfg = build_configs(settings, arch, Path(spec.starts_path))
    by_line = {row["line"]: row for row in spec.openings}
    generation = spec.generation

    torch.manual_seed(seed & 0xFFFFFFFF)
    model = load_model_from_checkpoint(checkpoint, device=settings.device).eval()
    evaluator = _evaluator(model, settings)

    pending: dict[int, list[Any]] = {}
    buffered: list[Any] = []
    buffered_meta: list[tuple[int, int, str, bool, np.uint64]] = []
    shard_index = int(resume["kept_shards"]) if resume is not None else 0
    games_kept = int(resume["kept_games"]) if resume is not None else 0
    rows_kept = int(resume["kept_rows"]) if resume is not None else 0
    games_done, rows_done = games_kept, rows_kept
    from chess_anti_engine.mcts.gumbel_c import buffer_full_flushes

    flushes_before = buffer_full_flushes()
    model_sha = sha256_file(checkpoint)

    def flush() -> None:
        nonlocal buffered, buffered_meta, shard_index
        if not buffered:
            return
        arrays = samples_to_arrays(buffered)
        meta = ShardMeta(run_id=f"simple_rl_gen{generation:04d}", generated_at_unix=int(time.time()),
                         model_sha256=model_sha, games=None, positions=len(buffered),
                         input_history_encoding=cfg["game"].input_history_encoding,
                         history_rep_fix=bool(cfg["game"].history_rep_fix),
                         policy_encoding=cfg["game"].policy_encoding, sf_nodes=int(settings.sf_nodes))
        shards = proc_dir / "shards"
        save_local_shard_arrays(shards / f"shard_{shard_index:05d}.zarr", arrs=arrays, meta=meta)
        _write_shard_meta(_shard_meta_path(shards, shard_index), {
            "game_id": np.asarray([m[0] for m in buffered_meta], dtype=np.int64),
            "ply_index": np.asarray([m[1] for m in buffered_meta], dtype=np.int32),
            "opening_id": np.asarray([m[2] for m in buffered_meta]),
            "tb_policy_override": np.asarray([m[3] for m in buffered_meta], dtype=np.bool_),
            "x_hash": np.asarray([m[4] for m in buffered_meta], dtype=np.uint64)})
        # Last: the commit record that lets a resume keep this shard and the
        # games.jsonl lines written so far (every one of their rows is now on disk).
        games_bytes = (proc_dir / GAMES).read_bytes()
        write_json_atomic(_commit_path(shards, shard_index), {
            "rows": len(buffered), "games_lines": games_bytes.count(b"\n"),
            "games_sha256": hashlib.sha256(games_bytes).hexdigest()})
        shard_index += 1
        buffered, buffered_meta = [], []

    def on_samples(batch: Any) -> None:
        if not batch.samples:
            return
        pending[int(batch.samples[0].game_id)] = list(batch.samples)

    def on_game(audit: Any) -> None:
        nonlocal games_done, rows_done
        samples = pending.pop(int(audit.game_id), [])
        if len(samples) != int(audit.retained_samples):
            raise ValueError("sample callback and provenance disagree on retained rows")
        line = audit.start.root_fen + " | " + " ".join(audit.start.opening_moves)
        row = by_line.get(line.rstrip())
        if row is None:
            raise ValueError(f"game started from an opening that was not doled to process {spec.proc}: {line!r}")
        tb_plies = {int(p) for p in audit.tablebase_policy_applied_plies}
        final = chess.Board(audit.final_fen)
        for s in samples:
            buffered.append(s)
            buffered_meta.append((int(audit.game_id), int(s.ply_index), row["id"],
                                  int(s.ply_index) in tb_plies, x_hash(s.x)))
        append_jsonl(proc_dir / GAMES, {
            "game_id": int(audit.game_id), "opening_id": row["id"], "book": int(row["book"]),
            "result": audit.result, "resolution": audit.resolution,
            "total_plies": int(audit.total_plies), "start_ply": int(audit.start.start_ply),
            "rows": len(samples), "mate": bool(final.is_checkmate()),
            "tb_policy_rows": sum(1 for s in samples if int(s.ply_index) in tb_plies),
            "cap_reached": bool(audit.cap_reached),
        })
        games_done += 1
        rows_done += len(samples)
        if len(buffered) >= int(settings.rows_per_shard):
            flush()

    queue = [row["line"] for row in openings]
    n_games = len(spec.openings)
    played_games = 0
    if openings:
        pool = spec.sf_factory(settings)
        try:
            _, batch_stats = play_batch(
                model, device=settings.device, rng=np.random.default_rng(seed), stockfish=pool,
                evaluator=evaluator, games=min(int(settings.concurrency), len(openings)),
                target_games=len(openings), on_game_complete=on_samples, on_game_provenance=on_game,
                fen_dole_queue=queue, pause_fn=gate, **cfg,
            )
        finally:
            pool.close()
        played_games = int(batch_stats.games)
    flush()
    wall = time.perf_counter() - t0
    if games_done != n_games or queue or pending:
        raise ValueError(f"incomplete collect process {spec.proc}: {games_done}/{n_games} games, "
                         f"{len(queue)} undoled openings, {len(pending)} pending")
    played_ids = Counter(json.loads(r)["opening_id"] for r in (proc_dir / GAMES).read_text().splitlines() if r)
    if played_ids != Counter(row["id"] for row in spec.openings):
        raise ValueError(f"collect process {spec.proc}: the games played (kept + replayed) are not exactly "
                         "its doled openings, each once")
    if not (proc_dir / GAMES).exists():
        (proc_dir / GAMES).write_text("")
    active = max(wall - gate.paused_seconds, 1e-9)
    done = {"proc": spec.proc, "seed": spec.seed, "games": games_done, "rows": rows_done, "shards": shard_index,
            "play_batch_games": played_games + games_kept, "wall_seconds": wall,
            "started_at": started_at, "paused_seconds": gate.paused_seconds, "pause_intervals": gate.intervals,
            # Game-granularity resumes of this process (kept vs replayed openings,
            # the re-seed); empty when it ran uninterrupted.
            "resumes": resumes, "games_this_attempt": games_done - games_kept,
            "rows_this_attempt": rows_done - rows_kept,
            # Throughput over the time the process was not paused, games of THIS attempt.
            "games_per_hour": 3600.0 * (games_done - games_kept) / active,
            "games_sha256": sha256_file(proc_dir / GAMES),
            # Observed in THIS process (gumbel_c's counter is per process).
            "buffer_full_flushes": int(buffer_full_flushes() - flushes_before),
            "leaf_rows_required": required_leaf_rows(settings), **evaluator_graph_record(evaluator)}
    write_json_atomic(proc_dir / PROC_DONE, done)
    return done


def _proc_main(spec: ProcSpec) -> None:
    """Entry point of a spawned collect process."""
    from chess_anti_engine.stockfish.uci import _child_reap_guard

    # Die with the driver however it exits (SIGKILL included); the engines this
    # process starts carry the same guard against THIS process.
    _child_reap_guard(spec.parent_pid)
    os.environ["SIMPLE_RL_COLLECT_PROC"] = str(spec.proc)  # findable, and readable by an engine factory
    import torch

    torch.set_num_threads(max(1, int(spec.torch_threads)))
    _play_proc(spec)


def _stop_children(children: list[Any], grace_s: float = 10.0) -> None:
    children = [c for c in children if c.pid is not None]  # never started: nothing to stop
    for c in children:
        if c.is_alive():
            c.terminate()
    deadline = time.monotonic() + grace_s
    for c in children:
        c.join(max(0.0, deadline - time.monotonic()))
        if c.is_alive():
            c.kill()
            c.join()


def _run_procs(specs: list[ProcSpec]) -> list[dict[str, Any]]:
    """Run the unfinished processes: in this process for one, spawned children for more.

    Children die with the driver: a ``finally`` stops them on any exception or
    Ctrl-C, and each child's parent-death signal covers a driver SIGKILL/SIGTERM."""
    if len(specs) == 1 and int(specs[0].settings.collect_procs) == 1:
        return [_play_proc(specs[0])]
    import multiprocessing as mp
    from multiprocessing.connection import wait

    ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA or engine threads
    children = [ctx.Process(target=_proc_main, args=(s,), name=f"simple_rl_collect_{s.proc:02d}") for s in specs]
    try:
        for c in children:
            c.start()
        while True:
            codes = [c.exitcode for c in children]
            bad = [(s.proc, code) for s, code in zip(specs, codes, strict=True) if code not in (None, 0)]
            if bad:
                _stop_children(children)
                raise RuntimeError(f"collect process(es) failed (proc, exit code): {bad}; the finished ones are "
                                   "kept and the rest restart on the next attempt")
            if all(code == 0 for code in codes):
                break
            wait([c.sentinel for c in children if c.exitcode is None], timeout=5.0)
    finally:
        _stop_children(children)
    return [json.loads((s.proc_dir / PROC_DONE).read_text()) for s in specs]


def _merge_procs(gen_dir: Path, procs: int) -> int:
    """Hard-link every process's shards into ``gen_dir/shards`` (renumbered, in
    process order) and concatenate their games.jsonl. Idempotent: the process
    directories stay intact until the manifest seals the collect, so a kill here
    is redone from them."""
    import os

    out = gen_dir / "shards"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    index = 0
    with (gen_dir / f".{GAMES}.tmp").open("w") as games:
        for p in range(procs):
            pdir = gen_dir / PROCS_DIR / f"{p:02d}"
            for shard in sorted((pdir / "shards").glob("shard_*.zarr")):
                shutil.copytree(shard, out / f"shard_{index:05d}.zarr", copy_function=os.link)
                os.link(shard.with_name(shard.name.replace(".zarr", ".meta.npz")), out / f"shard_{index:05d}.meta.npz")
                index += 1
            games.write((pdir / GAMES).read_text())
    (gen_dir / f".{GAMES}.tmp").replace(gen_dir / GAMES)
    return index


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def _sum_graph_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = [r.get("cuda_graph_stats") for r in records]
    stats: list[dict[str, Any]] = [s for s in raw if s is not None]
    if len(stats) != len(raw) or not stats:
        return None
    out: dict[str, Any] = {}
    for key in stats[0]:
        values = [s.get(key) for s in stats]
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if key in ("max_rows", "graphs"):  # per-process settings / captured-shape counts: never summed
            out[key] = values[0] if len({json.dumps(v) for v in values}) == 1 else values
        elif len(numbers) == len(values):
            out[key] = sum(numbers)
        else:
            out[key] = values
    return out


def _combined_evaluator_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The collect-level evaluator fields the gates read, from every process's record.

    A field the processes disagree on is None ("mixed" for the mode), which fails
    its gate rather than reporting one process's value for all of them."""
    def common(key: str) -> Any:
        values = {json.dumps(r.get(key)) for r in records}
        return records[0].get(key) if len(values) == 1 else None

    modes = {r.get("evaluator_mode") for r in records}
    return {"cuda_graphs": all(bool(r.get("cuda_graphs")) for r in records),
            "evaluator_mode": records[0].get("evaluator_mode") if len(modes) == 1 else "mixed",
            "cuda_graph_stats": _sum_graph_stats(records),
            "eval_max_batch_realised": common("eval_max_batch_realised")}


def collect_generation(*, gen_dir: Path, generation: int, checkpoint: Path, train_jsonl: Path,
                       settings: CollectSettings, pause_markers: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Play one generation into ``gen_dir`` with ``settings.collect_procs`` processes.

    The opening list is drawn exactly as for one process (the cursor advances by
    ``games``); process k plays the disjoint slice ``openings[k::K]`` with its own
    seed (``proc_seed``) into ``gen_dir/procs/k``. The merge links their shards
    into ``gen_dir/shards`` in process order and concatenates games.jsonl, and ONE
    manifest seals the collect with every process's seed and evaluator record.
    An interrupted collect whose plan (K, seeds, openings, checkpoint, settings)
    is unchanged keeps its finished processes and restarts the rest; any other
    unsealed attempt is redone whole."""
    validate_settings(settings)  # refuse before any work (incl. a leaf-dropping cap)
    vram = check_vram(settings)
    import torch

    gen_dir = Path(gen_dir)
    run_dir = gen_dir.parent.parent
    k_procs = int(settings.collect_procs)
    cursor_before = previous_cursor(gen_dir, generation) if settings.opening_sampling == "cycle" else None
    cursor_after: dict[str, Any] | None = None
    if settings.opening_sampling == "cycle":
        openings, cursor_after, cycle_log = cycle_openings(
            train_jsonl, games=settings.games, book_mix=settings.book_mix, base_seed=settings.seed,
            generation=generation, start=cursor_before)
        for line in cycle_log:
            print(f"[simple_rl] {line}", flush=True)
    else:
        openings = sample_openings(train_jsonl, games=settings.games, book_mix=settings.book_mix,
                                   seed=seed_for(settings.seed, "openings", generation))
    plan = _plan(settings, generation, checkpoint, openings, cursor_before)
    if gen_dir.exists():
        old = json.loads((gen_dir / PLAN).read_text()) if (gen_dir / PLAN).is_file() else None
        if old != plan:
            if old is not None:
                print(f"[simple_rl] generation {generation}: the interrupted collect's plan differs "
                      "(processes, seeds, openings, checkpoint or settings); redoing it whole", flush=True)
            shutil.rmtree(gen_dir)  # an incomplete attempt of another plan is redone whole
    gen_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(gen_dir / PLAN, plan)
    starts_path = gen_dir / "starts.txt"
    starts_path.write_text("".join(row["line"] + "\n" for row in openings))
    arch = _arch(checkpoint)
    cfg = build_configs(settings, arch, starts_path)

    specs = [ProcSpec(proc=p, gen_dir=str(gen_dir), generation=int(generation), checkpoint=str(checkpoint),
                      starts_path=str(starts_path), settings=settings,
                      openings=tuple(proc_openings(openings, p, k_procs)),
                      seed=proc_seed(settings, generation, p), sf_factory=_open_stockfish,
                      torch_threads=int(torch.get_num_threads()), run_dir=str(run_dir),
                      pause_markers=tuple(str(m) for m in pause_markers), parent_pid=os.getpid())
             for p in range(k_procs)]
    done = {s.proc: d for s in specs if (d := _proc_done(s)) is not None}
    if done:
        print(f"[simple_rl] generation {generation}: keeping finished collect process(es) {sorted(done)}", flush=True)
    t0 = time.perf_counter()
    todo = [s for s in specs if s.proc not in done]
    if todo:
        for rec in _run_procs(todo):
            done[int(rec["proc"])] = rec
    wall = time.perf_counter() - t0
    records = [done[p] for p in range(k_procs)]
    shard_count = _merge_procs(gen_dir, k_procs)
    games_done = sum(int(r["games"]) for r in records)
    played = Counter(json.loads(r)["opening_id"] for r in (gen_dir / GAMES).read_text().splitlines() if r.strip())
    if games_done != settings.games or played != Counter(row["id"] for row in openings):
        raise ValueError(f"incomplete generation: {games_done}/{settings.games} games, or the openings played "
                         "differ from the ones drawn")
    stats = generation_stats(gen_dir, previous=[p for p in sorted(run_dir.glob("gen_*/collect"))
                                                if p != gen_dir and _gen_number(p) < generation],
                             generation=generation, settings=settings)
    stats["opening_sampling"] = settings.opening_sampling
    if cursor_after is not None:
        start_books = (cursor_before or {}).get("books", {})
        stats["book_pass"] = {b: {"start": int(start_books.get(b, {"pass": 1})["pass"]), "end": int(c["pass"])}
                              for b, c in cursor_after["books"].items()}
    # Paused time = the union of this attempt's processes' pause intervals (they
    # pause together); throughput is over the rest. A paused collect's timing is
    # not an independent measurement and says so.
    # Rates count only the games and rows THIS attempt's processes played: a
    # process kept from an interrupted attempt adds games but none of this wall.
    # None when no process ran (a resume that only merged and sealed).
    fresh = {s.proc for s in todo}
    paused = _union_seconds([tuple(i) for r in records if int(r["proc"]) in fresh
                             for i in r.get("pause_intervals", [])])
    active = max(wall - paused, 1e-9)
    resumed_games = {int(r["proc"]): k for r in records
                     if (k := max((int(x["kept_games"]) for x in r.get("resumes", [])), default=0)) > 0}
    fresh_records = [r for r in records if int(r["proc"]) in fresh]
    games_now = sum(int(r["games_this_attempt"]) for r in fresh_records)
    rows_now = sum(int(r["rows_this_attempt"]) for r in fresh_records)
    stats.update(wall_seconds=wall, paused_seconds=paused, active_seconds=active,
                 timing_independent=paused == 0.0 and not (set(done) - fresh) and not resumed_games,
                 games_per_hour=3600.0 * games_now / active if fresh else None,
                 # Processes resumed at game granularity: their kept games came from an
                 # earlier attempt and the rest were replayed with a fresh seed, so this
                 # collect is NOT digest-identical to an uninterrupted one.
                 procs_game_resumed=resumed_games,
                 rows_per_second=rows_now / active if fresh else None, sf_nodes=int(settings.sf_nodes),
                 leaf_rows_required=required_leaf_rows(settings), collect_procs=k_procs,
                 procs_resumed=sorted(p for p in done if p not in fresh),
                 **_combined_evaluator_record(records))
    stats["procs"] = records
    stats["gpu_memory_check"] = vram
    stats["duplicate_game_ids"] = int(sum(n - 1 for n in Counter(
        json.loads(r)["game_id"] for r in (gen_dir / GAMES).read_text().splitlines() if r.strip()).values() if n > 1))
    # Observed, not predicted: buffer-full flushes the generation's searches
    # took (gumbel_c's per-process counter, read around each process's play and
    # summed). > 0 means the evaluator cap re-batched the searches, i.e. the rows
    # were made by a different search than the pinned settings describe; written
    # to stats.json, and the generation is refused unsealed (redone whole).
    stats["buffer_full_flushes"] = int(sum(int(r["buffer_full_flushes"]) for r in records))
    write_json_atomic(gen_dir / STATS, stats)
    if stats["buffer_full_flushes"] > 0:
        shutil.rmtree(gen_dir / PROCS_DIR, ignore_errors=True)  # nothing of this attempt may be kept
        raise RuntimeError(
            f"collect generation {generation}: {stats['buffer_full_flushes']} buffer-full flush(es) -- the "
            "evaluator's max batch is below the leaves the search asks for, so its searches were re-batched "
            "(see gumbel_c.leaf_buffer_rows). stats.json records the count; the generation is not sealed. "
            "Raise the evaluator cap or lower --concurrency and redo it")
    config_record = {"settings": asdict(settings),
                     "dataclasses": {k: asdict(v) for k, v in cfg.items()}, "arch": arch}
    write_manifest(
        gen_dir, stage="collect", config=config_record,
        inputs={"checkpoint": input_ref(checkpoint), "train_openings": input_ref(train_jsonl)},
        outputs={"shards": input_ref(gen_dir / "shards"), "games": input_ref(gen_dir / GAMES),
                 "starts": input_ref(starts_path)},
        counts={"games": games_done, "rows": stats["rows"], "shards": shard_count,
                "play_batch_games": sum(int(r["play_batch_games"]) for r in records)},
        seeds={"collect": proc_seed(settings, generation, 0) if k_procs == 1 else None,
               "collect_procs": [int(r["seed"]) for r in records],
               "openings": seed_for(settings.seed, "openings", generation), "base": settings.seed},
        extra={"generation": generation, "stats": stats, "opening_cursor": cursor_after,
               "opening_cursor_before": cursor_before, "collect_procs": k_procs,
               # Every game-granularity resume per process (kept vs replayed
               # openings, discarded torn files, the re-seed). A collect with a
               # resume that kept games is not digest-identical to an unkilled one.
               "game_resumes": {str(r["proc"]): r.get("resumes", []) for r in records if r.get("resumes")},
               # Derived and rebuildable from the sealed collects; pruned after
               # two generations. A later collect uses one only if it matches.
               "diagnostics_files": {
                   "dup_sketch": input_ref(sketch_path(run_dir, generation)),
                   "openings_seen": input_ref(_seen_path(run_dir, generation))}},
    )
    # Sealed: the process directories only duplicate (hard links to) the merged shards.
    shutil.rmtree(gen_dir / PROCS_DIR, ignore_errors=True)
    (gen_dir / PLAN).unlink(missing_ok=True)
    return stats


def _gen_number(collect_dir: Path) -> int:
    return int(Path(collect_dir).parent.name.split("_")[1])


def load_generation_rows(collect_dir: Path, *, keys: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All shards of one generation, concatenated, plus the row-meta sidecar."""
    from chess_anti_engine.replay.shard import load_shard_arrays

    shard_dir = resolve_collect_dir(collect_dir) / "shards"
    parts: list[dict[str, np.ndarray]] = []
    metas: list[dict[str, np.ndarray]] = []
    for shard in sorted(shard_dir.glob("shard_*.zarr")):
        arrays, _ = load_shard_arrays(shard, lazy=False, validate=True)
        if keys is not None:
            arrays = {k: v for k, v in arrays.items() if k in keys or str(k).startswith("_")}
        parts.append(arrays)
        with np.load(shard.with_name(shard.name.replace(".zarr", ".meta.npz"))) as z:
            metas.append({k: z[k] for k in z.files})
    if not parts:
        raise ValueError(f"no shards in {shard_dir}")
    return concat_arrays(parts), {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}


def concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Row-concatenate shard arrays; a field absent from a part is zero-filled
    with the schema's own default (flag 0), metadata keys must agree."""
    from chess_anti_engine.replay.shard import zeros_for_storage_field

    keys: list[str] = []
    for part in parts:
        for k in part:
            if k not in keys:
                keys.append(k)
    out: dict[str, np.ndarray] = {}
    x_planes = int(parts[0]["x"].shape[1])
    policy_size = int(parts[0]["policy_target"].shape[1])
    for k in keys:
        if str(k).startswith("_"):
            values = {str(np.asarray(p[k]).tolist()) for p in parts if k in p}
            if len(values) != 1 or any(k not in p for p in parts):
                raise ValueError(f"metadata {k} differs across shards: {sorted(values)}")
            out[k] = np.asarray(parts[0][k])
            continue
        cols = []
        for p in parts:
            n = int(p["x"].shape[0])
            cols.append(np.asarray(p[k]) if k in p else zeros_for_storage_field(
                k, n=n, x_planes=x_planes, policy_size=policy_size))
        out[k] = np.concatenate(cols, axis=0)
    return out


def _entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=-1)


def policy_stats(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Target/prior entropy, KL(target||prior) and top-1 agreement, from root stats."""
    target = np.asarray(arrays["policy_target"], dtype=np.float64)
    target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-30)
    policy_rows = np.asarray(arrays.get("has_policy", np.ones(len(target))), dtype=bool)
    out: dict[str, float] = {
        "target_entropy": float(_entropy(target[policy_rows]).mean()) if policy_rows.any() else math.nan,
        "policy_rows": int(policy_rows.sum()),
    }
    has = np.asarray(arrays.get("has_root_log_prior", np.zeros(len(target))), dtype=bool)
    out["root_stats_rows"] = int(has.sum())
    # Exact root shortcuts (solved / tablebase / single-move / #1410's pruned-draw
    # fallback) carry no statistics; their value target falls back (learn.py).
    out["rows_without_root_stats"] = int((~has).sum())
    # Value diagnostics over every searched row; policy diagnostics (below)
    # over policy rows only -- a fast ply's stored target is masked.
    if not has.any():
        return out
    for name in ("root_net_value", "root_survivor_q", "root_node_q"):
        if name in arrays:
            vals = np.asarray(arrays[name], dtype=np.float64)[has]
            out[f"mean_{name}"] = float(np.nanmean(vals))
    if "root_survivor_q" in arrays and "root_node_q" in arrays:
        gap = np.asarray(arrays["root_survivor_q"], np.float64)[has] - np.asarray(arrays["root_node_q"], np.float64)[has]
        out["mean_survivor_minus_node_q"] = float(np.nanmean(gap))
    if "root_start_visits" in arrays:
        sv = np.asarray(arrays["root_start_visits"], dtype=np.float64)[has]
        out.update(mean_root_start_visits=float(sv.mean()), reused_root_share=float((sv > 0).mean()))
    has &= policy_rows
    if not has.any():
        return out
    support = np.asarray(arrays["root_child_visits"])[has] >= 0
    logp = np.where(support, np.asarray(arrays["root_log_prior"], dtype=np.float64)[has], -np.inf)
    logp = logp - logp.max(axis=1, keepdims=True)
    prior = np.where(support, np.exp(logp), 0.0)
    prior /= prior.sum(axis=1, keepdims=True)
    t = target[has]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(t > 0, t * (np.log(np.where(t > 0, t, 1.0)) - np.log(np.where(prior > 0, prior, 1e-300))), 0.0).sum(axis=1)
    out.update(prior_entropy=float(_entropy(prior).mean()), kl_target_prior=float(kl.mean()),
               top1_agreement=float((t.argmax(axis=1) == prior.argmax(axis=1)).mean()))
    return out


OPENINGS_SEEN_DIR = "openings_seen"


def _diagnostic_ok(run_dir: Path, generation: int, name: str, path: Path) -> bool:
    """A run-level diagnostics file (duplicate sketch, openings seen) is used
    only if the sealed collect of ``generation`` recorded exactly this file;
    otherwise (a superseded or interrupted collect wrote it) it is rebuilt
    from the sealed collects."""
    from chess_anti_engine.simple_rl.common import read_manifest

    manifest = read_manifest(Path(run_dir) / f"gen_{generation:04d}" / "collect")
    ref = ((manifest or {}).get("diagnostics_files") or {}).get(name)
    return bool(ref) and sha256_file(path) == ref["sha256"]


def _seen_path(run_dir: Path, generation: int) -> Path:
    return Path(run_dir) / OPENINGS_SEEN_DIR / f"through_gen_{generation:04d}.npy"


def openings_seen_before(run_dir: Path, generation: int, previous: list[Path]) -> np.ndarray:
    """Sorted unique opening ids played in generations < g (bounded by the
    book size). Read from the file generation g-1 wrote, else rebuilt."""
    path = _seen_path(run_dir, generation - 1)
    if generation > 1 and path.is_file() and _diagnostic_ok(run_dir, generation - 1, "openings_seen", path):
        return np.load(path, allow_pickle=False)
    ids: set[str] = set()
    for p in previous:
        for r in (resolve_collect_dir(p) / GAMES).read_text().splitlines():
            if r.strip():
                ids.add(json.loads(r)["opening_id"])
    return np.asarray(sorted(ids), dtype=str)


def dup_sketch_before(run_dir: Path, generation: int, previous: list[Path], budget: int) -> Any:
    from chess_anti_engine.simple_rl.dup_sketch import DupSketch

    path = sketch_path(run_dir, generation - 1)
    if generation > 1 and path.is_file() and _diagnostic_ok(run_dir, generation - 1, "dup_sketch", path):
        return DupSketch.load(path, budget=budget)
    sketch = DupSketch(budget=budget)
    for p in sorted(previous, key=_gen_number):
        for m in sorted((resolve_collect_dir(p) / "shards").glob("shard_*.meta.npz")):
            with np.load(m) as z:
                sketch.add(z["x_hash"])
    return sketch


def generation_stats(collect_dir: Path, *, previous: list[Path], generation: int | None = None,
                     settings: CollectSettings | None = None) -> dict[str, Any]:
    collect_dir = Path(collect_dir)
    games = [json.loads(r) for r in (collect_dir / GAMES).read_text().splitlines() if r.strip()]
    keys = ("x", "policy_target", "wdl_target", "has_policy", "root_log_prior", "has_root_log_prior",
            "root_child_visits", "has_root_child_visits", "root_net_value", "root_survivor_q", "root_node_q",
            "root_start_visits", "search_budget", "has_search_budget")
    arrays, meta = load_generation_rows(collect_dir, keys=keys)
    n = int(arrays["x"].shape[0])
    results = [g["result"] for g in games]
    w, d, l_ = results.count("1-0"), results.count("1/2-1/2"), results.count("0-1")
    hashes = np.asarray(meta["x_hash"], dtype=np.uint64)
    from chess_anti_engine.simple_rl.dup_sketch import prune

    generation = _gen_number(collect_dir) if generation is None else int(generation)
    run_dir = collect_dir.parent.parent
    budget = int(settings.dup_sketch_budget) if settings is not None else 4_000_000
    start_ply = {int(g["game_id"]): int(g["start_ply"]) for g in games}
    ply_after_book = np.asarray([int(p) - start_ply[int(gid)] for gid, p in zip(meta["game_id"], meta["ply_index"],
                                                                                 strict=True)], dtype=np.int64)
    sketch = dup_sketch_before(run_dir, generation, previous, budget)
    dup = sketch.estimate(hashes, ply_after_book)
    sketch.add(hashes)
    sketch.save(sketch_path(run_dir, generation))
    prune(run_dir, keep_from=generation - 1)
    seen = openings_seen_before(run_dir, generation, previous)
    ids = np.asarray([g["opening_id"] for g in games], dtype=str)
    earlier = np.isin(ids, seen)
    seen_after = np.union1d(seen, ids)
    _seen_path(run_dir, generation).parent.mkdir(parents=True, exist_ok=True)
    np.save(_seen_path(run_dir, generation), seen_after, allow_pickle=False)
    for old in (run_dir / OPENINGS_SEEN_DIR).glob("through_gen_*.npy"):
        if int(old.stem.rsplit("_", 1)[1]) < generation - 1:
            old.unlink(missing_ok=True)
    exact = bool(settings is not None and settings.exact_repeat_vs_previous)
    exact_rate = None
    if exact:
        prev: set[int] = set()
        for p in previous:
            for m in sorted((resolve_collect_dir(p) / "shards").glob("shard_*.meta.npz")):
                with np.load(m) as z:
                    prev.update(int(h) for h in z["x_hash"])
        exact_rate = (sum(1 for h in hashes if int(h) in prev) / max(1, n)) if previous else 0.0
    stats: dict[str, Any] = {
        "games": len(games), "rows": n,
        "white_wins": w, "draws": d, "black_wins": l_,
        "decisive_rate": (w + l_) / max(1, len(games)),
        "draw_rate": d / max(1, len(games)),
        "mean_game_plies": float(np.mean([g["total_plies"] for g in games])) if games else math.nan,
        "mean_plies_after_book": float(np.mean([g["total_plies"] - g["start_ply"] for g in games])) if games else math.nan,
        "mate_count": sum(1 for g in games if g["mate"]),
        "tb_adjudicated_share": sum(1 for g in games if g["resolution"] == "tablebase_adjudication") / max(1, len(games)),
        "cap_reached_games": sum(1 for g in games if g["cap_reached"]),
        "unresolved_games": sum(1 for g in games if g["resolution"] == "unresolved"),
        "tb_policy_override_rows": int(np.asarray(meta["tb_policy_override"]).sum()),
        "repeat_rate_within": 1.0 - len({int(h) for h in hashes}) / max(1, n),
        # Constant-memory estimate (dup_sketch.py) unless the exact set is asked for.
        "repeat_rate_vs_previous": (exact_rate if exact_rate is not None
                                    else (dup["rate"] if previous else 0.0)),
        "repeat_rate_vs_previous_method": "exact" if exact else "threshold_sample",
        "repeat_rate_vs_previous_estimate": dup,
        "repeat_rate_vs_previous_exact": exact_rate,
        "openings_played_in_earlier_generation_share": float(earlier.mean()) if ids.size else math.nan,
        "distinct_openings_played_so_far": int(seen_after.size),
        "previous_generations_compared": len(previous),
        "unique_openings": len({g["opening_id"] for g in games}),
    }
    stats.update(policy_stats(arrays))
    if "search_budget" in arrays:
        has_b = np.asarray(arrays["has_search_budget"], dtype=bool)
        budgets = np.asarray(arrays["search_budget"])[has_b]
        stats["rows_by_search_budget"] = {str(int(b)): int((budgets == b).sum()) for b in np.unique(budgets)}
    return stats
