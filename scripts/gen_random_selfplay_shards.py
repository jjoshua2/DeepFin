#!/usr/bin/env python3
"""CPU generation-zero selfplay shard generator (uniform prior, no teacher).

Warms a replay window with real game data for the AZ-purity loop experiment,
before any net exists. At random initialisation a net's policy and value heads
are noise, so a Gumbel search driven by a UNIFORM prior over the legal moves and
a cheap position-independent value is the same search the real net would run --
and it needs no GPU. Running it on the CPU keeps the accelerator free while the
first iterations get real games instead of inference against a meaningless net.

What it emits
-------------
Production-shaped replay shards (``shard_NNNNNN.zarr``) through the repo's own
writer, ``replay.shard.save_local_shard_arrays`` -- so the schema is not
reimplemented here and cannot drift from the loader:

* ``x``              (175, 8, 8) float32 ``v2_threats`` planes, encoded by the
                     same ``encoding.cboard_encode.encode_cboard`` selfplay uses.
* ``policy_target``  the search's improved policy, projected into compact
                     ``lc0_1858`` train space by ``policy_vector_to_encoding``
                     exactly as ``selfplay/finalize.py`` does.
* ``wdl_target``     the game outcome through ``selfplay.game._result_to_wdl``,
                     POV = the side to move AT THAT PLY (so it alternates down a
                     selfplay game). The convention is INHERITED from that
                     function, never restated here.
* ``legal_mask``, ``moves_left``, ``game_id``, ``ply_index``,
  ``opening_source_code``, ``is_selfplay``, ``is_network_turn``, ``has_policy``.

⚑ NO Stockfish fields and NO ``search_wdl``. Every ``has_sf_*`` flag is 0, which
is how the loader spells "absent" (``prune_storage_arrays`` drops an all-zero
optional pair entirely). Writing the stub's value into ``search_wdl`` was
considered and REJECTED: it would launder a constant -- or a hash of the piece
placement -- into a field every reader treats as a search's value estimate,
which is this codebase's signature defect verbatim.

⚑ CONSEQUENCE FOR THE TRAINING CONFIG, and it is silent. ``train/losses.py``
falls the SF component of the value blend back to the RAW ONE-HOT OUTCOME when
``has_sf_wdl`` is 0, with no error and no log line, and the ``search_wdl``
component has the same shape of fallback. On these rows the WHOLE value target
is therefore the game outcome no matter what the yaml says. For the AZ-purity
arm that is the intent -- but the config must SAY so: set ``sf_wdl_frac: 0.0``,
``sf_wdl_frac_floor: 0.0`` and ``search_wdl_frac: 0.0`` rather than leaving
production's values in place and relying on the fallback. A refusal inside the
trainer is the real end state; that is a training-affecting change and belongs
with the AZ-purity ledger entry, not smuggled into an offline data tool.

The evaluator stub
------------------
``UniformPriorEvaluator`` implements the ``inference.BatchEvaluator`` protocol,
so the production search calls it exactly where it would call the net.

* POLICY: full-width (4672) ZERO logits. ``_policy_logits_to_full`` dispatches on
  SHAPE, so a full-width vector passes through untouched and every legal move
  reaches ``_masked_priors`` with the same logit -- a provably uniform prior over
  the legal moves. (Compact 1858 logits would be scattered with ``fill_value =
  -1e9``, which is right for a real compact net and would silently zero the prior
  of any legal move outside the 1858 vocabulary.)
* VALUE: ``--value-source``, DEFAULT ``zero``.
    ``zero``     q = 0 everywhere. Pure: no handcrafted chess knowledge at all.
    ``material`` q = tanh(cp / 400) where cp is the standard piece-value balance
                 P=100 N=320 B=330 R=500 Q=900, us minus them, read from the
                 ENCODED planes (``plane_decode.decode_step0_bitboards``, which
                 returns the step-0 bitboards in the stored side-to-move frame),
                 so the value is already side-to-move POV by construction.
    ``random``   q = a fixed pseudo-random function of the position: blake2b over
                 those same bitboards plus ``--seed``, mapped to [-1, 1]. A fixed
                 function of the position, which is what a randomly-initialised
                 net actually is -- not fresh noise per visit.

  q reaches the search as WDL LOGITS, because that is what the protocol returns.
  The mapping is exact, not approximate: ``q_to_wdl_logits`` emits
  ``log(p + 1e-9)`` for ``p = (max(q, 0), 1 - |q|, max(-q, 0))`` -- the minimum-
  information triple whose W-L margin is q -- and both search paths turn WDL
  logits into a value with ``softmax([w, d, l])`` then ``p_w - p_l``
  (``mcts/puct.py::_value_scalar_from_wdl_logits``; ``mcts/_mcts_tree.c:42``), so
  the value the search sees is q back to a relative 3e-9.

Search
------
The C tree (``mcts.gumbel_c.run_gumbel_root_many_c``) -- production's own search,
and it accepts a ``BatchEvaluator`` directly, so wiring the stub to it was one
argument. Measured on this box at ``--sims 32``: 1190 plies/s against 49 plies/s
for the Python path, for a bit-identical game at the same seed. Search SHAPE
defaults come from the repo (``SELFPLAY_GUMBEL_C_SCALE``, ``gumbel_topk`` 16,
``gumbel_scale`` 0.75, final-action ``temperature`` 0.0); only ``--sims``
deliberately departs from production's 256, for CPU throughput.

⚑ ``gumbel_scale`` is held at its pre-decay value for the whole game. Production
decays it to 0 after move 12 because by then it trusts a TRAINED prior; at
generation zero the prior is uniform, so decaying the noise would only make the
games less diverse. Stated as a deviation rather than absorbed.

Terminal handling mirrors production: the game state is a ``CBoard``, ends on
``CBoard.is_game_over()`` (50-move, threefold, insufficient material, no legal
moves) and is labelled from ``CBoard.result()``. A game truncated at
``--max-plies`` returns ``"*"``, which ``_result_to_wdl`` labels a draw --
production would adjudicate that with Stockfish, and there is no Stockfish here.

Usage
-----
  PYTHONPATH=. python3 scripts/gen_random_selfplay_shards.py \\
      --out-dir data/gen0_shards --games 2000 --workers 8

  PYTHONPATH=. python3 scripts/gen_random_selfplay_shards.py \\
      --out-dir data/gen0_shards --games 2000 --workers 8 \\
      --value-source material --openings data/opening_books/book.pgn.zip

Re-running against a populated ``--out-dir`` APPENDS: shard numbering starts
above the highest index already there, and an occupied index is skipped rather
than overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V2_THREATS
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.encoding.plane_decode import decode_step0_bitboards
from chess_anti_engine.mcts.gumbel import (
    SELFPLAY_GUMBEL_C_SCALE,
    GumbelConfig,
    assert_c_path_can_run,
    validate_gumbel_config,
)
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    index_to_move_strict,
    policy_mask_to_encoding,
    policy_vector_to_encoding,
)
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    iter_shard_paths,
    load_shard_arrays,
    local_shard_path,
    samples_to_arrays,
    save_local_shard_arrays,
    shard_index,
)
from chess_anti_engine.selfplay.finalize import _stable_game_id
from chess_anti_engine.selfplay.game import _result_to_wdl
from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board
from chess_anti_engine.selfplay.seed_manifest import opening_source_code, position_key

_LOG = logging.getLogger("gen0")

# ── defaults, every one traceable to the repo ────────────────────────────────
# `--sims` is the ONE deliberate departure: production selfplay runs 256
# (configs/pbt2_small.yaml selfplay.mcts_simulations) and this is CPU tooling.
DEFAULT_SIMS = 32
DEFAULT_TOPK = 16              # selfplay.gumbel_topk
DEFAULT_GUMBEL_SCALE = 0.75    # selfplay.gumbel_scale (pre-decay; see module doc)
DEFAULT_POLICY_TEMP = 1.0      # selfplay.gumbel_policy_temp
DEFAULT_TEMPERATURE = 0.0      # selfplay.selfplay_temperature
DEFAULT_MAX_PLIES = 450        # selfplay.max_plies
DEFAULT_SHARD_SIZE = 2000      # distributed.shard_size
DEFAULT_OPENING_PLIES = 16     # selfplay.opening_book_max_plies_2
DEFAULT_OPENING_MAX_GAMES = 200_000  # selfplay.opening_book_max_games_2
DEFAULT_NICE = 10
DEFAULT_RUN_ID = "gen0_random_selfplay"

VALUE_SOURCE_ZERO = "zero"
VALUE_SOURCE_RANDOM = "random"
VALUE_SOURCE_MATERIAL = "material"
VALUE_SOURCES: tuple[str, ...] = (
    VALUE_SOURCE_ZERO, VALUE_SOURCE_RANDOM, VALUE_SOURCE_MATERIAL,
)

# Standard piece values in centipawns, indexed the way decode_step0_bitboards
# returns them: columns 0-5 are "us" P/N/B/R/Q/K, 6-11 are "them". The king
# entry is 0 because both sides always have exactly one.
PIECE_VALUES_CP: tuple[float, ...] = (100.0, 320.0, 330.0, 500.0, 900.0, 0.0)
MATERIAL_CP_SCALE = 400.0
# Probability floor under the log so a certain-result triple stays finite. At
# 1e-9 the value the search reads back is q / (1 + 3e-9).
WDL_PROB_FLOOR = 1e-9

_POPCOUNT_LUT: np.ndarray = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1,
).sum(axis=1).astype(np.int64)


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Set-bit count of a uint64 array, without numpy-version-gated builtins."""
    arr = np.ascontiguousarray(values, dtype=np.uint64)
    as_bytes = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT_LUT[as_bytes].sum(axis=-1)


def q_to_wdl_logits(q: np.ndarray) -> np.ndarray:
    """(N,) value in [-1, 1] → (N, 3) WDL logits the search reads back as q.

    ``p = (max(q, 0), 1 - |q|, max(-q, 0))`` is the triple with the least
    win/loss mass whose ``p_w - p_l`` is q; the logits are ``log(p + floor)``,
    and both search paths recover the value as ``softmax(logits)[0] -
    softmax(logits)[2]``, i.e. ``q / (1 + 3*floor)``.
    """
    clipped = np.clip(np.asarray(q, dtype=np.float64), -1.0, 1.0)
    probs = np.stack(
        [
            np.maximum(clipped, 0.0),
            1.0 - np.abs(clipped),
            np.maximum(-clipped, 0.0),
        ],
        axis=1,
    )
    return np.log(probs + WDL_PROB_FLOOR).astype(np.float32)


def material_q(x: np.ndarray) -> np.ndarray:
    """Side-to-move material balance of encoded positions, as tanh(cp / 400)."""
    counts = _popcount64(decode_step0_bitboards(x)).astype(np.float64)
    values = np.asarray(PIECE_VALUES_CP, dtype=np.float64)
    cp = counts[:, :6] @ values - counts[:, 6:] @ values
    return np.tanh(cp / MATERIAL_CP_SCALE)


def random_q(x: np.ndarray, *, salt: int) -> np.ndarray:
    """A fixed pseudo-random value per POSITION (not per visit), seeded by salt."""
    bitboards = decode_step0_bitboards(x)
    salt_bytes = int(salt).to_bytes(8, "little", signed=False)
    out = np.empty((bitboards.shape[0],), dtype=np.float64)
    for i in range(bitboards.shape[0]):
        payload = np.ascontiguousarray(bitboards[i]).tobytes() + salt_bytes
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / float(1 << 64) * 2.0 - 1.0
    return out


class UniformPriorEvaluator:
    """``BatchEvaluator`` stub: uniform prior over legal moves + a cheap value.

    ``eval_calls`` / ``eval_rows`` are not decoration: they are how a test
    proves the search actually consumed ``--sims`` rather than merely storing it
    (a bigger budget must ask this object for more positions).
    """

    def __init__(
        self, *, value_source: str, expected_planes: int, random_salt: int = 0,
    ) -> None:
        if value_source not in VALUE_SOURCES:
            raise ValueError(
                f"value_source must be one of {VALUE_SOURCES}, got {value_source!r}",
            )
        self.value_source = str(value_source)
        self.expected_planes = int(expected_planes)
        self.random_salt = int(random_salt)
        self.eval_calls = 0
        self.eval_rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # `relations` belongs to the protocol; compute_relations stays off here.
        del relations
        arr = np.asarray(x)
        if arr.ndim != 4 or int(arr.shape[1]) != self.expected_planes:
            # A width mismatch here means the encoding flags did not reach the
            # search. Fail loudly: silently evaluating the wrong planes is the
            # failure this whole tool would be useless under.
            raise ValueError(
                f"evaluator expected (N, {self.expected_planes}, 8, 8) planes, "
                f"got {tuple(arr.shape)}",
            )
        n = int(arr.shape[0])
        self.eval_calls += 1
        self.eval_rows += n
        policy_logits = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        if self.value_source == VALUE_SOURCE_ZERO:
            q = np.zeros((n,), dtype=np.float64)
        elif self.value_source == VALUE_SOURCE_MATERIAL:
            q = material_q(arr)
        else:
            q = random_q(arr, salt=self.random_salt)
        return policy_logits, q_to_wdl_logits(q)


@dataclass(frozen=True)
class GenConfig:
    """Everything the game loop is allowed to depend on, in one picklable box."""

    out_dir: Path
    games: int = 100
    workers: int = 1
    sims: int = DEFAULT_SIMS
    topk: int = DEFAULT_TOPK
    c_scale: float = SELFPLAY_GUMBEL_C_SCALE
    policy_temp: float = DEFAULT_POLICY_TEMP
    temperature: float = DEFAULT_TEMPERATURE
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE
    value_source: str = VALUE_SOURCE_ZERO
    max_plies: int = DEFAULT_MAX_PLIES
    shard_size: int = DEFAULT_SHARD_SIZE
    seed: int = 0
    nice: int = DEFAULT_NICE
    openings: Path | None = None
    opening_plies: int = DEFAULT_OPENING_PLIES
    opening_max_games: int = DEFAULT_OPENING_MAX_GAMES
    random_start_plies: int = 0
    input_history_encoding: str = LC0_HISTORY_ROOT_LEGACY_META
    input_extra_features: str = EXTRA_FEATURES_V2_THREATS
    history_rep_fix: bool = True
    run_id: str = DEFAULT_RUN_ID


@dataclass(frozen=True)
class WorkerSpec:
    cfg: GenConfig
    worker_id: int
    games: int
    seed: int
    shard_index_start: int


@dataclass(frozen=True)
class PlyRecord:
    """One stored ply: the position BEFORE the move, and the search's answer."""

    x: np.ndarray
    policy_probs: np.ndarray  # (4672,) improved policy, search space
    legal_mask: np.ndarray    # (4672,) bool
    pov_white: bool           # side to move AT THIS PLY
    ply_index: int


@dataclass(frozen=True)
class GameOutcome:
    records: list[PlyRecord]
    result: str
    plies: int
    termination: str
    start_fen: str
    opening_source: str
    move_trace: str
    end_ply_index: int


@dataclass
class WorkerResult:
    worker_id: int
    realized: dict[str, Any]
    games: int = 0
    rows: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    plies: list[int] = field(default_factory=list)
    terminations: dict[str, int] = field(default_factory=dict)
    shards: list[dict[str, Any]] = field(default_factory=list)
    eval_calls: int = 0
    eval_rows: int = 0
    seconds: float = 0.0


def build_gumbel_config(cfg: GenConfig) -> GumbelConfig:
    """The search config the game loop will hand to the C tree, and nothing else.

    Validated here, at the construction boundary, by the repo's own
    ``validate_gumbel_config``: an out-of-band knob would otherwise be recorded
    in the realized line and the sidecar while the hot path quietly ignored it,
    which is a worse outcome than a refusal for a tool whose whole product is
    data labelled with the settings that produced it.
    """
    gcfg = GumbelConfig(
        simulations=int(cfg.sims),
        topk=int(cfg.topk),
        temperature=float(cfg.temperature),
        policy_temp=float(cfg.policy_temp),
        c_scale=float(cfg.c_scale),
        add_noise=float(cfg.gumbel_scale) > 0.0,
        gumbel_scale=float(cfg.gumbel_scale),
        input_history_encoding=str(cfg.input_history_encoding),
        input_extra_features=str(cfg.input_extra_features),
        policy_encoding=POLICY_ENCODING_LC0_1858,
    )
    validate_gumbel_config(gcfg, where="gen_random_selfplay_shards")
    assert_c_path_can_run(gcfg, where="gen_random_selfplay_shards")
    return gcfg


def build_opening_config(cfg: GenConfig) -> OpeningConfig:
    return OpeningConfig(
        opening_book_path=None if cfg.openings is None else str(cfg.openings),
        opening_book_max_plies=int(cfg.opening_plies),
        opening_book_max_games=int(cfg.opening_max_games),
        opening_book_prob=1.0 if cfg.openings is not None else 0.0,
        random_start_plies=int(cfg.random_start_plies),
    )


def realized_config(
    *,
    gcfg: GumbelConfig,
    evaluator: UniformPriorEvaluator,
    opening_cfg: OpeningConfig,
    cfg: GenConfig,
    worker_id: int,
) -> dict[str, Any]:
    """Read the realized settings back off the objects that will consume them.

    Deliberately NOT built from ``args``: every field here is fetched from the
    ``GumbelConfig`` the search receives, the evaluator instance it calls, the
    ``OpeningConfig`` the sampler reads, or the OS. A flag that got dropped on
    the way in is therefore visible in this line, not just in the parser.
    """
    return {
        "worker_id": int(worker_id),
        "simulations": int(gcfg.simulations),
        "topk": int(gcfg.topk),
        "c_scale": float(gcfg.c_scale),
        "policy_temp": float(gcfg.policy_temp),
        "temperature": float(gcfg.temperature),
        "add_noise": bool(gcfg.add_noise),
        "gumbel_scale": float(gcfg.gumbel_scale),
        "input_history_encoding": str(gcfg.input_history_encoding),
        "input_extra_features": str(gcfg.input_extra_features),
        "input_planes": int(evaluator.expected_planes),
        "policy_encoding": str(gcfg.policy_encoding),
        "policy_width": int(COMPACT_POLICY_SIZE),
        "value_source": str(evaluator.value_source),
        "material_cp_scale": float(MATERIAL_CP_SCALE),
        "random_salt": int(evaluator.random_salt),
        "opening_book_path": opening_cfg.opening_book_path,
        "opening_book_prob": float(opening_cfg.opening_book_prob),
        "opening_book_max_plies": int(opening_cfg.opening_book_max_plies),
        "random_start_plies": int(opening_cfg.random_start_plies),
        "history_rep_fix": bool(rep_fix.current() or False),
        "max_plies": int(cfg.max_plies),
        "shard_size": int(cfg.shard_size),
        "nice": int(os.getpriority(os.PRIO_PROCESS, 0)),
        "torch_threads": int(torch.get_num_threads()),
    }


def format_realized(realized: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in realized.items())


def _termination(cb: CBoard, board: chess.Board) -> str:
    """Why the game stopped, in ``CBoard.is_game_over()``'s own terms."""
    if not cb.is_game_over():
        return "max_plies"
    if cb.is_checkmate():
        return "checkmate"
    if cb.is_stalemate():
        return "stalemate"
    if int(cb.halfmove_clock) >= 100:
        return "fifty_moves"
    if board.is_insufficient_material():
        return "insufficient_material"
    return "threefold"


def play_game(
    *,
    cfg: GenConfig,
    gcfg: GumbelConfig,
    evaluator: UniformPriorEvaluator,
    rng: np.random.Generator,
    opening_cfg: OpeningConfig,
) -> GameOutcome:
    """Play one complete game with the production C Gumbel search."""
    start = sample_starting_board(rng=rng, cfg=opening_cfg)
    board = start.board
    cb = CBoard.from_board(board)
    start_fen = board.fen()
    records: list[PlyRecord] = []
    actions: list[int] = []

    while not cb.is_game_over() and len(actions) < int(cfg.max_plies):
        x = encode_cboard(
            cb,
            input_history_encoding=cfg.input_history_encoding,
            input_extra_features=cfg.input_extra_features,
        )
        probs, acts, _values, masks, _tree, _root_ids = run_gumbel_root_many_c(
            None,
            [board],
            device="cpu",
            rng=rng,
            cfg=gcfg,
            evaluator=evaluator,
            cboards=[cb],
        )
        action = int(acts[0])
        records.append(
            PlyRecord(
                x=x,
                policy_probs=np.asarray(probs[0]),
                legal_mask=np.asarray(masks[0]),
                pov_white=bool(cb.turn),
                ply_index=int(cb.ply),
            ),
        )
        cb.push_index(action)
        # STRICT, unlike production selfplay's lenient `index_to_move`: an
        # undecodable action there costs one played move, here it would desync
        # the two boards and mislabel every row after the split.
        board.push(index_to_move_strict(action, board))
        actions.append(action)

    # The two boards are pushed independently from one action index, exactly as
    # selfplay/network_turn.py does. If they ever disagreed, every label after
    # the split would be wrong with no shape change and no failing assertion.
    if position_key(cb.fen()) != position_key(board.fen()):
        raise RuntimeError(
            f"CBoard/python-chess divergence: {cb.fen()!r} vs {board.fen()!r}",
        )
    return GameOutcome(
        records=records,
        result=cb.result(),
        plies=len(actions),
        termination=_termination(cb, board),
        start_fen=start_fen,
        opening_source=start.source,
        move_trace=",".join(str(a) for a in actions),
        end_ply_index=int(cb.ply),
    )


def rows_from_game(outcome: GameOutcome, *, cfg: GenConfig) -> list[ReplaySample]:
    """Turn one finished game into replay rows, production conventions only."""
    game_id = _stable_game_id(
        start_fen=outcome.start_fen,
        opening_source=outcome.opening_source,
        move_trace=outcome.move_trace,
        result=outcome.result,
        total_plies_played=outcome.plies,
    )
    source_code = opening_source_code(outcome.opening_source)
    max_plies = max(1.0, float(cfg.max_plies))
    return [
        ReplaySample(
            x=np.asarray(rec.x, dtype=np.float32),
            policy_target=policy_vector_to_encoding(rec.policy_probs),
            wdl_target=int(_result_to_wdl(outcome.result, pov_white=rec.pov_white)),
            legal_mask=policy_mask_to_encoding(rec.legal_mask).astype(
                np.uint8, copy=False,
            ),
            moves_left=float(max(0, outcome.end_ply_index - rec.ply_index)) / max_plies,
            has_policy=True,
            is_selfplay=True,
            is_network_turn=True,
            game_id=game_id,
            ply_index=int(rec.ply_index),
            opening_source_code=source_code,
            input_history_encoding=cfg.input_history_encoding,
            history_rep_fix=bool(cfg.history_rep_fix),
        )
        for rec in outcome.records
    ]


def shard_digest(path: str | Path) -> str:
    """Content hash of a written shard, read back through the real loader."""
    arrs, _meta = load_shard_arrays(path)
    digest = hashlib.sha256()
    for name in sorted(arrs):
        arr = np.asarray(arrs[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def next_shard_index(out_dir: Path) -> int:
    """One past the highest shard already in ``out_dir`` (0 when empty)."""
    indices = [shard_index(p) for p in iter_shard_paths(out_dir)]
    live = [i for i in indices if i >= 0]
    return max(live) + 1 if live else 0


def write_shard(
    *,
    out_dir: Path,
    index: int,
    rows: list[ReplaySample],
    cfg: GenConfig,
    wins: int,
    draws: int,
    losses: int,
    games: int,
    total_plies: int,
) -> dict[str, Any]:
    path = local_shard_path(out_dir, index)
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(rows),
        meta=ShardMeta(
            run_id=str(cfg.run_id),
            generated_at_unix=int(time.time()),
            input_history_encoding=str(cfg.input_history_encoding),
            history_rep_fix=bool(cfg.history_rep_fix),
            policy_encoding=POLICY_ENCODING_LC0_1858,
            policy_size=int(COMPACT_POLICY_SIZE),
            positions=len(rows),
            games=int(games),
            wins=int(wins),
            draws=int(draws),
            losses=int(losses),
            total_game_plies=int(total_plies),
            selfplay_games=int(games),
        ),
    )
    return {
        "index": int(index),
        "path": path.name,
        "rows": len(rows),
        "digest": shard_digest(path),
    }


def _apply_nice(delta: int) -> None:
    if int(delta) == 0:
        return
    try:
        os.nice(int(delta))
    except OSError as exc:  # pragma: no cover - only when privileges are missing
        _LOG.warning("could not renice by %d: %s", int(delta), exc)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def run_worker(spec: WorkerSpec) -> WorkerResult:
    """Play this worker's games and write its shards. Runs in a child process."""
    setup_logging()
    cfg = spec.cfg
    _apply_nice(cfg.nice)
    torch.set_num_threads(1)
    # Before any CBoard exists: per-slot repetition flags are recorded at push
    # time and never recomputed (encoding/rep_fix.py).
    rep_fix.apply(bool(cfg.history_rep_fix))

    gcfg = build_gumbel_config(cfg)
    evaluator = UniformPriorEvaluator(
        value_source=cfg.value_source,
        expected_planes=input_plane_count(cfg.input_extra_features),
        random_salt=int(cfg.seed),
    )
    opening_cfg = build_opening_config(cfg)
    rng = np.random.default_rng(spec.seed)

    realized = realized_config(
        gcfg=gcfg, evaluator=evaluator, opening_cfg=opening_cfg,
        cfg=cfg, worker_id=spec.worker_id,
    )
    _LOG.info("realized %s", format_realized(realized))

    result = WorkerResult(worker_id=spec.worker_id, realized=realized)
    # Shards close on a GAME boundary at or past `--shard-size` rows, so a shard
    # holds whole games and its meta game counters (games/wins/draws/losses/
    # total_game_plies) are exact rather than apportioned across a split game.
    pending: list[ReplaySample] = []
    pending_games = 0
    pending_results = [0, 0, 0]  # white wins / draws / black wins
    pending_plies = 0
    shard_idx = int(spec.shard_index_start)
    stride = max(1, int(cfg.workers))
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending, pending_games, pending_results, pending_plies, shard_idx
        if not pending:
            return
        while local_shard_path(cfg.out_dir, shard_idx).exists():
            shard_idx += stride
        result.shards.append(
            write_shard(
                out_dir=cfg.out_dir, index=shard_idx, rows=pending, cfg=cfg,
                wins=pending_results[0], draws=pending_results[1],
                losses=pending_results[2], games=pending_games,
                total_plies=pending_plies,
            ),
        )
        shard_idx += stride
        pending = []
        pending_games = 0
        pending_results = [0, 0, 0]
        pending_plies = 0

    for _ in range(int(spec.games)):
        outcome = play_game(
            cfg=cfg, gcfg=gcfg, evaluator=evaluator, rng=rng, opening_cfg=opening_cfg,
        )
        rows = rows_from_game(outcome, cfg=cfg)
        result.games += 1
        result.rows += len(rows)
        result.plies.append(outcome.plies)
        result.terminations[outcome.termination] = (
            result.terminations.get(outcome.termination, 0) + 1
        )
        bucket = 0 if outcome.result == "1-0" else (2 if outcome.result == "0-1" else 1)
        pending_results[bucket] += 1
        result.wins += int(bucket == 0)
        result.draws += int(bucket == 1)
        result.losses += int(bucket == 2)
        pending.extend(rows)
        pending_games += 1
        pending_plies += outcome.plies
        if len(pending) >= int(cfg.shard_size):
            flush()
    flush()

    result.eval_calls = evaluator.eval_calls
    result.eval_rows = evaluator.eval_rows
    result.seconds = time.perf_counter() - started
    return result


def plies_summary(plies: list[int]) -> dict[str, float]:
    if not plies:
        return {}
    arr = np.asarray(plies, dtype=np.float64)
    quantiles = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": float(arr.size),
        "min": float(arr.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def plies_histogram(plies: list[int], *, max_plies: int) -> dict[str, int]:
    """Fixed 50-ply buckets up to ``max_plies``, so two runs are comparable."""
    hist: dict[str, int] = {}
    width = 50
    for value in plies:
        lo = min(int(value) // width * width, int(max_plies))
        hist[f"{lo}-{lo + width - 1}"] = hist.get(f"{lo}-{lo + width - 1}", 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0])))


def build_worker_specs(cfg: GenConfig, *, shard_index_start: int) -> list[WorkerSpec]:
    workers = max(1, int(cfg.workers))
    base, extra = divmod(max(0, int(cfg.games)), workers)
    specs: list[WorkerSpec] = []
    for wid in range(workers):
        games = base + (1 if wid < extra else 0)
        seed = int(
            np.random.SeedSequence([int(cfg.seed), wid]).generate_state(
                1, dtype=np.uint64,
            )[0],
        )
        specs.append(
            WorkerSpec(
                cfg=cfg,
                worker_id=wid,
                games=games,
                seed=seed,
                shard_index_start=shard_index_start + wid,
            ),
        )
    return specs


def summarize(
    *, cfg: GenConfig, results: list[WorkerResult], wall_seconds: float,
    shard_index_start: int,
) -> dict[str, Any]:
    plies: list[int] = []
    terminations: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    games = rows = wins = draws = losses = eval_calls = eval_rows = 0
    for res in results:
        plies.extend(res.plies)
        for key, count in res.terminations.items():
            terminations[key] = terminations.get(key, 0) + count
        shards.extend(res.shards)
        games += res.games
        rows += res.rows
        wins += res.wins
        draws += res.draws
        losses += res.losses
        eval_calls += res.eval_calls
        eval_rows += res.eval_rows
    hours = max(wall_seconds, 1e-9) / 3600.0
    return {
        "run_id": str(cfg.run_id),
        "config": {**asdict(cfg), "out_dir": str(cfg.out_dir),
                   "openings": None if cfg.openings is None else str(cfg.openings)},
        "realized_per_worker": [res.realized for res in results],
        "games": games,
        "rows": rows,
        "results": {"white_wins": wins, "draws": draws, "black_wins": losses},
        "plies": plies_summary(plies),
        "plies_histogram": plies_histogram(plies, max_plies=int(cfg.max_plies)),
        "terminations": dict(sorted(terminations.items())),
        "shards_written": len(shards),
        "shard_index_start": int(shard_index_start),
        "shards": sorted(shards, key=lambda s: int(s["index"])),
        "evaluator": {"calls": eval_calls, "rows": eval_rows},
        "wall_seconds": float(wall_seconds),
        "games_per_hour": games / hours,
        "rows_per_hour": rows / hours,
        # These rows carry no Stockfish label and no search value estimate; see
        # the module docstring for why the training config has to say so.
        "sf_fields": "absent",
        "search_wdl": "absent",
        "required_run_config": {
            "sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.0,
        },
    }


def generate(cfg: GenConfig) -> dict[str, Any]:
    """Run the whole generation and return the summary dict."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shard_index_start = next_shard_index(cfg.out_dir)
    specs = build_worker_specs(cfg, shard_index_start=shard_index_start)
    started = time.perf_counter()
    if len(specs) == 1:
        results = [run_worker(specs[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(specs), mp_context=ctx) as pool:
            results = list(pool.map(run_worker, specs))
    return summarize(
        cfg=cfg, results=results, wall_seconds=time.perf_counter() - started,
        shard_index_start=shard_index_start,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--c-scale", type=float, default=SELFPLAY_GUMBEL_C_SCALE)
    parser.add_argument("--policy-temp", type=float, default=DEFAULT_POLICY_TEMP)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gumbel-scale", type=float, default=DEFAULT_GUMBEL_SCALE)
    parser.add_argument(
        "--value-source", choices=VALUE_SOURCES, default=VALUE_SOURCE_ZERO,
    )
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nice", type=int, default=DEFAULT_NICE)
    parser.add_argument(
        "--openings", type=Path, default=None,
        help="PGN/PGN.zip/Polyglot opening book; default is the start position.",
    )
    parser.add_argument("--opening-plies", type=int, default=DEFAULT_OPENING_PLIES)
    parser.add_argument(
        "--opening-max-games", type=int, default=DEFAULT_OPENING_MAX_GAMES,
    )
    parser.add_argument(
        "--random-start-plies", type=int, default=0,
        help="Random legal plies from the start position when no book is given.",
    )
    parser.add_argument(
        "--input-history-encoding", type=str, default=LC0_HISTORY_ROOT_LEGACY_META,
    )
    parser.add_argument(
        "--input-extra-features", type=str, default=EXTRA_FEATURES_V2_THREATS,
    )
    parser.add_argument(
        "--no-history-rep-fix", dest="history_rep_fix", action="store_false",
        help="Encode without the lc0-root repetition-plane fix (production: on).",
    )
    parser.set_defaults(history_rep_fix=True)
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--summary-json", type=Path, default=None,
        help="Sidecar path; default <out-dir>/gen0_summary_<start index>.json",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GenConfig:
    return GenConfig(
        out_dir=Path(args.out_dir),
        games=int(args.games),
        workers=int(args.workers),
        sims=int(args.sims),
        topk=int(args.topk),
        c_scale=float(args.c_scale),
        policy_temp=float(args.policy_temp),
        temperature=float(args.temperature),
        gumbel_scale=float(args.gumbel_scale),
        value_source=str(args.value_source),
        max_plies=int(args.max_plies),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        nice=int(args.nice),
        openings=None if args.openings is None else Path(args.openings),
        opening_plies=int(args.opening_plies),
        opening_max_games=int(args.opening_max_games),
        random_start_plies=int(args.random_start_plies),
        input_history_encoding=str(args.input_history_encoding),
        input_extra_features=str(args.input_extra_features),
        history_rep_fix=bool(args.history_rep_fix),
        run_id=str(args.run_id),
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.games <= 0:
        raise SystemExit("--games must be > 0")
    if cfg.shard_size <= 0:
        raise SystemExit("--shard-size must be > 0")
    summary = generate(cfg)
    sidecar = (
        Path(args.summary_json)
        if args.summary_json is not None
        else cfg.out_dir / f"gen0_summary_{int(summary['shard_index_start']):06d}.json"
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    res = summary["results"]
    plies = summary["plies"]
    print(
        f"games={summary['games']} rows={summary['rows']} "
        f"shards={summary['shards_written']} "
        f"W/D/L={res['white_wins']}/{res['draws']}/{res['black_wins']}",
    )
    if plies:
        print(
            f"plies: min={plies['min']:.0f} p25={plies['p25']:.0f} "
            f"median={plies['median']:.0f} p75={plies['p75']:.0f} "
            f"max={plies['max']:.0f} mean={plies['mean']:.1f}",
        )
    print(f"terminations: {summary['terminations']}")
    print(
        f"throughput: {summary['games_per_hour']:.0f} games/h "
        f"{summary['rows_per_hour']:.0f} rows/h "
        f"in {summary['wall_seconds']:.1f}s",
    )
    print(f"summary: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
