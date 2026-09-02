"""``scripts/gen_sf_rooted_corpus.py`` -- the NNUE-bootstrap corpus generator.

Every test but one runs against a SCRIPTED UCI DOUBLE: the real
``StockfishUCI`` with only ``_send`` / ``_readline_with_deadline`` replaced (the
pattern ``tests/test_stockfish_searchmoves.py`` and
``tests/test_audit_label_candidates.py`` already use), driving a fake engine
that answers the ``go`` line it was ACTUALLY HANDED.  A fake that replied from a
fixed script would pass whether or not the generator emitted the narrowing at
all, which is the one thing several of these tests exist to check.

⚑ THE SELECTION IS MADE DETERMINISTIC BY THE TEMPERATURE, not by patching the
RNG.  ``gumbel_choice`` is ``argmax(q/tau + gumbel)``, so a tiny ``tau`` lets a
scripted +900cp preference dominate any noise the seeded generator produces.
That keeps the Gumbel path -- the thing under test in
``test_the_temperature_knob_changes_what_gets_played`` -- live in every scripted
game rather than stubbed out of them.

The one real-engine test is skipped when no binary is discoverable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, cast

import chess
import chess.polyglot
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts import audit_label_candidates as gate
from scripts import gen_random_selfplay_shards as gen
from scripts import gen_sf_rooted_corpus as corpus
from tests.stockfish_binary import find_stockfish

# ── the scripted engine ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def production_rep_fix() -> None:
    """Production's repetition-plane regime, which ``corpus.row_key`` REQUIRES.

    The generator sets it itself at ``run``/``run_worker`` start; a test that
    drives ``play_game`` or ``row_key`` directly stands in for that call.  The
    suite-wide hookwrapper in ``conftest.py`` rewinds the flag after each test.
    ⚑ This is why the in-process tests cannot see a generator that FORGOT the
    call -- ``test_a_fresh_process_generates_and_derives_under_one_regime``
    runs the generator in a SUBPROCESS for exactly that reason.
    """
    rep_fix.apply(True)


#: Score handed to a move named in a fake's preference list.  Far enough above
#: the hashed scores that q/tau at the test temperature is a landslide.
PREFERRED_CP = 900

def smoke_syzygy() -> Path | None:
    """The 3-4 man SMOKE tablebase, in this checkout or the main one.

    NOT the production pair: a test that needs 151G of 6-man DTZ is a test
    nobody runs, and the point here is the wiring, not the table.  ⚑ The
    two-root lookup is the same one ``engine_discovery`` does for the binary and
    ``corpus.default_syzygy_path`` does for the tables, and for the same reason:
    ``data/`` is untracked runtime output, so a WORKTREE resolves a
    checkout-relative path to nothing and every test below would skip silently.
    """
    roots = [corpus.REPO_ROOT]
    main = corpus.main_checkout()
    if main is not None and main != corpus.REPO_ROOT:
        roots.append(main)
    for root in roots:
        candidate = root / "data" / "syzygy_3-4man"
        if candidate.is_dir():
            return candidate
    return None


def production_syzygy() -> str | None:
    """The PRODUCTION pair, when both of its directories are present.

    The smoke set stops at 4 men, and a 6-man verdict is a different claim: the
    black-to-move adjudication test below needs a table only the production pair
    holds.  ``default_syzygy_path`` already does the two-root lookup, so this is
    only the "is it actually there" half.
    """
    path = corpus.default_syzygy_path()
    if all(Path(part).is_dir() for part in path.split(os.pathsep)):
        return path
    return None


SMOKE_SYZYGY = smoke_syzygy()
PRODUCTION_SYZYGY = production_syzygy()

MATE_GAME_FEN = "6k1/5ppp/8/3n4/8/8/7P/R3K3 b - - 0 1"
MATE_GAME_SCRIPT = ("d5c3", "a1a8")

CAPTURE_CHAIN_FEN = "7k/8/6n1/8/1n1b4/2R5/1Q6/K7 w - - 0 1"
CAPTURE_CHAIN_SCRIPT = ("b2b4", "d4c3", "b4c3")

#: 7 men, WHITE to move.  ``Qxa5`` reaches a 6-man KQRvKPP with BLACK to move,
#: which the production pair calls a loss for black -- so the game's result is
#: "1-0" and the adjudicated position's own ``wdl`` is 0.  Those two numbers
#: disagreeing is the whole point of the fixture; see the test.
BLACK_ADJUDICATION_FEN = "7k/6pp/8/n7/8/8/Q7/K6R w - - 0 1"
BLACK_ADJUDICATION_SCRIPT = ("a2a5",)

#: What a scripted engine raises when a test asks it to die mid-game.
SCRIPTED_ENGINE_DEATH = "the scripted engine died mid-search"


def hashed_cp(uci: str) -> int:
    """A deterministic, process-stable cp for one move.

    ``hash()`` is salted per process and would make a spawned worker disagree
    with its parent about the fake engine's ranking.
    """
    return int(hashlib.sha256(uci.encode()).hexdigest()[:4], 16) % 401 - 200


class ScriptedEngine:
    """Answers ``go`` from the position and the ``searchmoves`` it was given.

    ``preferred`` names moves that, when legal at the root, are scored
    ``PREFERRED_CP`` and therefore rank first -- which is how a test scripts a
    whole game without touching the selection code.
    """

    def __init__(
        self, *, multipv: int = 1, preferred: tuple[str, ...] = (),
        final_depth_offset: int = 0, deep_favourite: str | None = None,
        deep_from_depth: int = 0, raise_on_go: int | None = None,
    ) -> None:
        self.commands: list[str] = []
        self.multipv = int(multipv)
        self.preferred = tuple(preferred)
        #: Subtracted from every requested depth, to script a search that stops
        #: short of its ask.
        self.final_depth_offset = int(final_depth_offset)
        #: A move scored ``+PREFERRED_CP`` from ``deep_from_depth`` and
        #: ``-PREFERRED_CP`` below it, i.e. a ranking that CHANGES WITH DEPTH.
        #: Without one this fake's ordering is depth-invariant, and "the deepest
        #: iteration's ranking" is then indistinguishable from "the shallowest
        #: iteration's ranking" -- which is exactly the choice
        #: ``deepest_block_with_width`` makes.
        self.deep_favourite = deep_favourite
        self.deep_from_depth = int(deep_from_depth)
        #: 1-based index of the ``go`` command to die on, to script an engine
        #: that fails PART WAY THROUGH a worker rather than at startup.
        self.raise_on_go = raise_on_go
        #: Like ``raise_on_go`` but a WEDGE (StockfishTimeoutError), the
        #: recoverable failure ``EngineLease`` replaces the engine over.
        self.wedge_on_go: int | None = None
        self.go_count = 0
        self.fen = chess.STARTING_FEN
        #: The last ``position`` line, split into what it actually said.  Kept
        #: so a test can assert the WINDOW rather than only the position it
        #: happens to reconstruct to.
        self.position_root = chess.STARTING_FEN
        self.position_moves: tuple[str, ...] = ()
        self._pending: list[str] = []

    # -- driver-facing surface --------------------------------------------
    def send(self, cmd: str) -> None:
        self.commands.append(cmd)
        if cmd in ("isready",):
            self._pending.append("readyok\n")
        elif cmd.startswith("setoption name MultiPV value "):
            self.multipv = int(cmd.split()[-1])
        elif cmd.startswith("position fen "):
            # ⚑ THE FAKE REPLAYS THE WINDOW, exactly as the engine does.  It
            # answers `go` from `self.fen`, so a `position fen <root> moves ...`
            # line that named the wrong root or the wrong moves would produce
            # PV lines for a DIFFERENT position and every scripted game would
            # come apart -- rather than the fake quietly scoring the root while
            # the generator believed it had asked about the leaf.
            rest = cmd[len("position fen "):]
            root, _, moves = rest.partition(" moves ")
            board = chess.Board(root)
            self.position_root = root
            self.position_moves = tuple(moves.split())
            for uci in self.position_moves:
                board.push(chess.Move.from_uci(uci))
            self.fen = board.fen()
        elif cmd.startswith("go "):
            self.go_count += 1
            if self.raise_on_go is not None and self.go_count == self.raise_on_go:
                raise RuntimeError(SCRIPTED_ENGINE_DEATH)
            if self.wedge_on_go is not None and self.go_count == self.wedge_on_go:
                raise corpus.StockfishTimeoutError("scripted wedge")
            self._pending.extend(self._reply_to(cmd))

    def readline(self, _deadline: float) -> str:
        return self._pending.pop(0)

    # -- engine behaviour --------------------------------------------------
    def score_of(self, uci: str, *, depth: int) -> int:
        if uci == self.deep_favourite:
            return PREFERRED_CP if depth >= self.deep_from_depth else -PREFERRED_CP
        return PREFERRED_CP if uci in self.preferred else hashed_cp(uci)

    def _reply_to(self, go_cmd: str) -> list[str]:
        toks = go_cmd.split()
        depth = int(toks[toks.index("depth") + 1])
        if "searchmoves" in toks:
            # Per the UCI spec searchmoves runs to the end of the line -- read
            # that way on purpose, so a driver that appended a parameter after
            # it would feed junk moves in here rather than be forgiven.
            root = toks[toks.index("searchmoves") + 1:]
        else:
            root = [m.uci() for m in chess.Board(self.fen).legal_moves]
        lines: list[str] = []
        ranked: list[str] = []
        for d in range(1, max(1, depth - self.final_depth_offset) + 1):
            # Ranked PER ITERATION, like a real MultiPV list: rank 1 is the best
            # score AT THIS DEPTH.  Ties break on the uci so the order is total.
            # With no `deep_favourite` the key is depth-invariant and every
            # iteration reproduces the single ordering this used to emit.
            ranked = sorted(
                root, key=lambda uci, d=d: (-self.score_of(uci, depth=d), uci),
            )[: max(1, self.multipv)]
            for rank, mv in enumerate(ranked, start=1):
                lines.append(
                    f"info depth {d} seldepth {d + 2} multipv {rank} "
                    f"score cp {self.score_of(mv, depth=d)} "
                    f"nodes {1000 * d + rank} pv {mv}\n",
                )
        lines.append(f"bestmove {ranked[0] if ranked else '0000'}\n")
        return lines

    # -- assertion helpers -------------------------------------------------
    @property
    def go_lines(self) -> list[str]:
        return [c for c in self.commands if c.startswith("go ")]

    @property
    def position_lines(self) -> list[str]:
        return [c for c in self.commands if c.startswith("position ")]

    @property
    def multipv_lines(self) -> list[str]:
        return [c for c in self.commands if c.startswith("setoption name MultiPV")]


def uci_double(engine: ScriptedEngine, **attrs: Any) -> StockfishUCI:
    """The REAL ``StockfishUCI`` with only its two I/O methods replaced."""
    sf = cast(Any, object.__new__(StockfishUCI))
    sf.path = "/nonexistent/stockfish"
    sf.nodes = 2000
    sf.multipv = engine.multipv
    sf.hash_mb = 64
    sf.threads = 1
    sf.syzygy_path = "/nonexistent/syzygy"
    sf.nice = 15
    sf.read_timeout_s = 5.0
    sf._lock = threading.Lock()
    sf._send = engine.send
    sf._readline_with_deadline = engine.readline
    # The real `close()` reaches for a pty fd and a Popen this double never
    # had. Shadowing it keeps `run_worker`'s `finally` honest without building
    # a subprocess to tear down.
    sf.close = lambda: None
    for name, value in attrs.items():
        setattr(sf, name, value)
    return cast(StockfishUCI, sf)


class InlineExecutor:
    """A ``ProcessPoolExecutor`` stand-in that runs each submission in process.

    Same surface ``run`` uses -- context manager, ``submit`` returning a
    ``Future``, results read through ``as_completed`` -- so the multi-worker
    branch is exercised rather than bypassed.  A spawn pool cannot carry a
    monkeypatched module into its children, which is why the scripted-engine
    tests cannot use the real one.
    """

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> InlineExecutor:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # a worker whose PROCESS would have died
            future.set_exception(exc)
        return future


def searcher_for(
    engine: ScriptedEngine,
    *,
    staircase: str = corpus.DEFAULT_STAIRCASE,
    staircase_policy: str = corpus.STAIRCASE_POLICY_FIXED,
    **attrs: Any,
) -> corpus.StaircaseSearcher:
    return corpus.StaircaseSearcher(
        engine=uci_double(engine, **attrs),
        staircase=corpus.parse_staircase(staircase),
        cp_slope=gen.NNUE_CP_SLOPE,
        cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        staircase_policy=staircase_policy,
    )


def worker_spec(tmp_path: Path, **overrides: Any) -> corpus.WorkerSpec:
    values: dict[str, Any] = {
        "worker_id": 0,
        "game_ids": (0,),
        "out_dir": tmp_path,
        "sf_binary": "/nonexistent/stockfish",
        "sf_hash_mb": 64,
        # Matches `uci_double`'s, so a test that does not care about the
        # timeout sees the double it built rather than a disagreement.
        "sf_read_timeout_s": 5.0,
        # Under the read timeout, as `run` validates; the scripted engine
        # never blocks, so tests that do not care never feel either bound.
        "sf_search_timeout_s": 4.0,
        "syzygy_path": "/nonexistent/syzygy",
        "staircase": corpus.DEFAULT_STAIRCASE,
        "staircase_policy": corpus.STAIRCASE_POLICY_FIXED,
        "seed": 7,
        "dedup_cache_max": corpus.DEFAULT_DEDUP_CACHE_MAX,
        "temp_plies": 20,
        # Tiny on both rungs: the scripted preference then decides every move.
        "temp_high": 0.01,
        "temp_low": 0.01,
        "max_plies": 50,
        "shard_rows": 100,
        "nice": 0,
        "cp_slope": gen.NNUE_CP_SLOPE,
        "cp_draw_width": gen.NNUE_CP_DRAW_WIDTH,
        "book": None,
        "book_plies": 16,
        "book_max_games": 10,
        "run_id": "test",
        "config_sha256": "0" * 64,
        "resume": False,
    }
    values.update(overrides)
    return corpus.WorkerSpec(**values)


def fen_opening(fen: str, tmp_path: Path) -> OpeningConfig:
    """A one-position opening "book", through the PRODUCTION sampler.

    ``opening_fen_list_path`` + ``opening_fen_prob=1.0`` is a real branch of
    ``sample_starting_board``; the generator itself never sets it (see
    ``build_opening_config``), so this is a fixture rather than a setting.
    """
    path = tmp_path / "seeds.txt"
    path.write_text(fen + "\n", encoding="utf-8")
    return OpeningConfig(
        opening_fen_list_path=str(path), opening_fen_prob=1.0,
    )


def play(
    fen: str, tmp_path: Path, *, engine: ScriptedEngine, **spec_overrides: Any,
) -> tuple[corpus.GameOutcome, corpus.StaircaseSearcher, corpus.DedupStats]:
    spec = worker_spec(tmp_path, **spec_overrides)
    searcher = searcher_for(
        engine,
        staircase=spec.staircase,
        staircase_policy=spec.staircase_policy,
    )
    dedup = corpus.DedupStats()
    outcome = corpus.play_game(
        spec=spec, searcher=searcher,
        opening_cfg=fen_opening(fen, tmp_path),
        game_id=spec.game_ids[0],
        cache=corpus.DedupCache(max_entries=spec.dedup_cache_max),
        dedup=dedup, progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )
    return outcome, searcher, dedup


# ── staircase parsing ────────────────────────────────────────────────────────


def test_the_default_staircase_is_the_three_rung_scout() -> None:
    phases = corpus.parse_staircase(corpus.DEFAULT_STAIRCASE)
    assert [(p.width, p.depth) for p in phases] == [(None, 9), (16, 11), (4, 13)]
    assert corpus.format_staircase(phases) == corpus.DEFAULT_STAIRCASE


def test_a_two_rung_staircase_parses() -> None:
    phases = corpus.parse_staircase("all:9,16:11")
    assert [(p.width, p.depth) for p in phases] == [(None, 9), (16, 11)]


def test_g10_is_a_named_exact_shape_not_a_generic_threshold_knob() -> None:
    phases = corpus.parse_staircase(corpus.G10_STAIRCASE)
    assert corpus.validate_staircase_policy(
        corpus.STAIRCASE_POLICY_G10,
        phases,
    ) == corpus.STAIRCASE_POLICY_G10
    assert corpus.staircase_gate_stamp(corpus.STAIRCASE_POLICY_G10) == {
        "policy": "g10",
        "adaptive": True,
        "decision_after_phase": 1,
        "decision_depth": 10,
        "metric": "effective_cp_rank1_minus_rank2",
        "extend_when": "margin_cp<=threshold_cp",
        "threshold_cp": 10.0,
        "no_margin_action": "stop",
        "extended_phase": 2,
        "extended_width": 4,
        "extended_depth": 12,
    }
    with pytest.raises(ValueError, match="validated only"):
        corpus.validate_staircase_policy(
            corpus.STAIRCASE_POLICY_G10,
            corpus.parse_staircase(corpus.DEFAULT_STAIRCASE),
        )


@pytest.mark.parametrize(
    "spec",
    [
        "16:11,16:13",     # equal widths: not a narrowing
        "4:9,16:11",       # widening
        "16:11,all:13",    # `all` after a number is the widest possible rung
        "all:9,all:11",    # two full-width rungs
    ],
)
def test_a_staircase_whose_widths_do_not_strictly_descend_is_refused(
    spec: str,
) -> None:
    with pytest.raises(ValueError, match="strictly descend"):
        corpus.parse_staircase(spec)


@pytest.mark.parametrize("spec", ["all:9,16:9", "all:11,16:9"])
def test_a_staircase_whose_depths_do_not_strictly_ascend_is_refused(
    spec: str,
) -> None:
    with pytest.raises(ValueError, match="strictly ascend"):
        corpus.parse_staircase(spec)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("all", "not '<width>:<depth>'"),
        ("x:9", "neither an integer"),
        ("all:y", "is not an integer"),
        ("all:0", "depth must be positive"),
        ("0:9", "width must be positive"),
        ("", "selected no phases"),
    ],
)
def test_a_malformed_staircase_rung_is_refused(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        corpus.parse_staircase(spec)


def test_the_all_rung_resolves_per_position() -> None:
    phase = corpus.parse_staircase("all:9")[0]
    assert phase.width_for(20) == 20
    assert phase.width_for(3) == 3
    narrow = corpus.parse_staircase("all:9,16:11")[1]
    assert narrow.width_for(20) == 16
    # Clamped, never widened past what the root actually offers.
    assert narrow.width_for(5) == 5


# ── stream parsing ───────────────────────────────────────────────────────────


def info(depth: int, rank: int, cp: int, move: str, nodes: int) -> str:
    return (
        f"info depth {depth} seldepth {depth} multipv {rank} score cp {cp} "
        f"nodes {nodes} pv {move}"
    )


def test_the_first_emission_wins_and_the_abort_re_emission_is_counted() -> None:
    """The banked block is the CLEAN one; the contamination is an anomaly.

    Mutation caught: keeping the LAST line seen per (depth, rank) -- which is
    what the driver's own accumulator does, correctly, for its single-PV
    purpose.  Here it splices the aborted search's -900cp into depth 6's rank 1
    and every number derived from that block is a blend of two searches.
    """
    stream = [
        info(5, 1, 30, "e2e4", 100),
        info(5, 2, 20, "d2d4", 110),
        info(6, 1, 35, "e2e4", 200),
        info(6, 2, 25, "d2d4", 210),
        # The abort: an updated line re-emitted under the OLD depth label.
        info(6, 1, -900, "g1f3", 300),
    ]
    parsed = corpus.parse_depth_blocks(stream, expected_lines=2)

    by_depth = {b.depth: b for b in parsed.blocks}
    assert [pv.move for pv in by_depth[6].lines] == ["e2e4", "d2d4"]
    assert by_depth[6].lines[0].effective_cp == 35.0
    assert by_depth[6].emissions == 3
    assert by_depth[5].emissions == 2
    assert parsed.re_emissions == 1
    # ⚑ The number that matters: the dropped line DISAGREED with the banked
    # one, so a last-emission-wins parser would have banked a different block.
    assert parsed.re_emissions_disagreeing == 1
    # Depth 6 saw three emissions where a clean iteration emits exactly two.
    assert parsed.emission_count_violations == 1
    assert parsed.duplicate_iteration_flushes == 0


def test_the_measured_end_of_search_flush_is_classified_not_alarmed_on() -> None:
    """Stockfish's benign double-emission of a completed iteration.

    MEASURED 2026-08-27 against the production binary: a cold ``go depth 4`` and
    ``go depth 6`` at MultiPV 20 each emit their FINAL iteration twice, with
    identical ranks, moves and scores.  ``emission_count_violations`` fires --
    correctly, the count is not what a clean iteration emits -- so the parser
    also publishes the signature separately, and ``re_emissions_disagreeing``
    stays at zero.  Without that split the anomaly counter is noise on every
    real corpus and stops being readable as a defect signal.
    """
    block = [info(3, 1, 40, "e2e4", 90), info(3, 2, 12, "d2d4", 95)]
    parsed = corpus.parse_depth_blocks([*block, *block], expected_lines=2)

    assert [pv.move for pv in parsed.blocks[0].lines] == ["e2e4", "d2d4"]
    assert parsed.re_emissions == 2
    assert parsed.re_emissions_disagreeing == 0
    assert parsed.emission_count_violations == 1
    assert parsed.duplicate_iteration_flushes == 1


def test_an_agreeing_partial_re_emission_is_a_violation_but_not_a_flush() -> None:
    """The flush classifier's EXACT condition: ``2 x width``, not ``> width``.

    Mutation caught: relaxing the classifier to "more emissions than expected
    with nothing disagreeing".  The signature the module docstring measured is a
    WHOLE ITERATION emitted twice; a PARTIAL re-emission -- one rank arriving
    again, agreeing -- is a different stream shape and is not what was measured.
    Under the relaxed rule it would be filed as the benign flush, and the
    corpus-level identity ``emission_count_violations == duplicate_iteration_
    flushes`` would then read "we saw nothing but the flush" about a run in
    which we saw something else.  Agreement alone is not the signature.
    """
    block = [
        info(4, 1, 40, "e2e4", 90),
        info(4, 2, 12, "d2d4", 95),
        info(4, 3, 5, "g1f3", 99),
    ]
    parsed = corpus.parse_depth_blocks([*block, block[0]], expected_lines=3)

    assert [pv.move for pv in parsed.blocks[0].lines] == ["e2e4", "d2d4", "g1f3"]
    assert parsed.blocks[0].emissions == 4
    assert parsed.re_emissions == 1
    assert parsed.re_emissions_disagreeing == 0
    assert parsed.emission_count_violations == 1
    assert parsed.duplicate_iteration_flushes == 0, "4 is not 2 x 3"


def test_a_bound_line_never_wins_a_rank_and_is_counted() -> None:
    """Mutation caught: dropping the upperbound/lowerbound filter.

    The bound line arrives FIRST here, so a parser without the filter would let
    an aspiration-window edge become the move's banked score.
    """
    stream = [
        "info depth 7 multipv 1 score cp 999 upperbound nodes 10 pv e2e4",
        info(7, 1, 42, "e2e4", 20),
    ]
    parsed = corpus.parse_depth_blocks(stream, expected_lines=1)

    assert parsed.blocks[0].lines[0].effective_cp == 42.0
    assert parsed.bound_lines == 1
    assert parsed.emission_count_violations == 0
    assert parsed.re_emissions == 0


def test_info_string_lines_are_skipped_without_becoming_anomalies() -> None:
    parsed = corpus.parse_depth_blocks(
        [
            "info string NNUE evaluation using nn-f68ec79f0fe3.nnue",
            "info string Using 1 thread",
            info(3, 1, 12, "e2e4", 50),
        ],
        expected_lines=1,
    )
    assert len(parsed.blocks) == 1
    assert parsed.emission_count_violations == 0
    assert parsed.unscored_lines == 0


def test_every_depth_is_banked_whole_with_its_node_count() -> None:
    engine = ScriptedEngine(multipv=3)
    searcher = searcher_for(engine, staircase="all:4,3:6")
    board = chess.Board(MATE_GAME_FEN)

    search = searcher.search_position(board)

    scout = search.phases[0]
    assert [b.depth for b in scout.parse.blocks] == [1, 2, 3, 4]
    legal = board.legal_moves.count()
    assert all(len(b.lines) == legal for b in scout.parse.blocks)
    assert all(b.complete for b in scout.parse.blocks)
    # `nodes` is cumulative, so the depth's number is the largest on its lines.
    assert [b.nodes_at_depth for b in scout.parse.blocks] == [
        1000 * d + legal for d in (1, 2, 3, 4)
    ]
    assert all(pv.nodes is not None for b in scout.parse.blocks for pv in b.lines)


def test_a_search_that_stops_short_of_its_ask_is_disclosed_not_repaired() -> None:
    engine = ScriptedEngine(final_depth_offset=2)
    searcher = searcher_for(engine, staircase="all:9,16:11,4:13")
    search = searcher.search_position(chess.Board(MATE_GAME_FEN))

    # Depth 7 rather than the requested 9: the ASK is stamped, the realized
    # depths are read off the stream, and the two are allowed to differ.
    assert search.phases[0].depth_requested == 9
    assert max(b.depth for b in search.phases[0].parse.blocks) == 7
    assert search.value_depth == 7


def test_the_deepest_full_width_block_wins_and_a_narrow_one_is_flagged() -> None:
    blocks = corpus.parse_depth_blocks(
        [
            info(1, 1, 10, "e2e4", 5), info(1, 2, 5, "d2d4", 6),
            info(2, 1, 12, "e2e4", 9), info(2, 2, 7, "d2d4", 10),
            info(3, 1, 15, "e2e4", 14),          # depth 3 is short
        ],
        expected_lines=2,
    ).blocks

    block, full = corpus.deepest_block_with_width(blocks, want=2)
    assert (block.depth, full) == (2, True)
    narrow, full_narrow = corpus.deepest_block_with_width(blocks, want=5)
    assert (narrow.depth, full_narrow) == (2, False)


def test_a_stream_with_no_scored_line_is_refused_rather_than_imputed() -> None:
    with pytest.raises(RuntimeError, match="no scored MultiPV line"):
        corpus.deepest_block_with_width((), want=1)


# ── searchmoves narrowing ────────────────────────────────────────────────────


def test_every_rung_runs_and_narrows_through_searchmoves() -> None:
    """The exact protocol the staircase emits, rung by rung.

    Mutation caught: silently skipping a phase (e.g. ``self.staircase[:1]``),
    and emitting the narrowing without ``searchmoves`` -- Stockfish would then
    search the full move list at the deep phase's width and the phase would
    report a narrowing that never happened.
    """
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    board = chess.Board()
    legal = board.legal_moves.count()
    assert legal > 16, "the fixture must be wide enough for the 16-move rung"

    search = searcher.search_position(board)

    assert engine.multipv_lines == [
        f"setoption name MultiPV value {legal}",
        "setoption name MultiPV value 16",
        "setoption name MultiPV value 4",
    ]
    assert engine.go_lines[0] == "go depth 9"
    assert "searchmoves" not in engine.go_lines[0]
    assert engine.go_lines[1].startswith("go depth 11 searchmoves ")
    assert engine.go_lines[2].startswith("go depth 13 searchmoves ")

    scout_order = [pv.move for pv in search.phases[0].parse.blocks[-1].lines]
    assert engine.go_lines[1].split("searchmoves ")[1].split() == scout_order[:16]
    mid_order = [pv.move for pv in search.phases[1].parse.blocks[-1].lines]
    assert engine.go_lines[2].split("searchmoves ")[1].split() == mid_order[:4]

    assert [p.width_realized for p in search.phases] == [legal, 16, 4]
    assert [p.width_requested for p in search.phases] == ["all", "16", "4"]
    assert [p.depth_requested for p in search.phases] == [9, 11, 13]
    assert search.phases[0].searchmoves is None
    assert search.phases[1].searchmoves is not None
    assert len(search.phases[1].searchmoves) == 16


def test_g10_stops_after_d10_when_the_top_two_margin_is_wide() -> None:
    engine = ScriptedEngine(preferred=("e2e4",))
    searcher = searcher_for(
        engine,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    )

    search = searcher.search_position(chess.Board())

    assert [phase.depth_requested for phase in search.phases] == [9, 10]
    assert len(engine.go_lines) == 2
    assert engine.go_lines[0] == "go depth 9"
    assert engine.go_lines[1].startswith("go depth 10 searchmoves ")
    assert all("depth 12" not in line for line in engine.go_lines)
    assert search.staircase_gate is not None
    assert search.staircase_gate.extended is False
    assert search.staircase_gate.margin_cp is not None
    assert search.staircase_gate.margin_cp > corpus.G10_MARGIN_CP
    assert search.staircase_gate.reason == "margin_above_threshold"
    assert searcher.stats.staircase_gate_stopped == 1


def test_g10_extends_at_or_below_the_frozen_margin() -> None:
    class BoundaryEngine(ScriptedEngine):
        def score_of(self, uci: str, *, depth: int) -> int:
            del depth
            return {"e2e4": 100, "d2d4": 90}.get(uci, -1_000)

    # The d10 top two are exactly 10 cp apart: the inclusive boundary extends.
    engine = BoundaryEngine()
    searcher = searcher_for(
        engine,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    )

    search = searcher.search_position(chess.Board())

    assert [phase.depth_requested for phase in search.phases] == [9, 10, 12]
    assert engine.go_lines[2].startswith("go depth 12 searchmoves ")
    assert search.staircase_gate is not None
    assert search.staircase_gate.extended is True
    assert search.staircase_gate.margin_cp == corpus.G10_MARGIN_CP
    assert search.staircase_gate.reason == "margin_at_or_below_threshold"
    assert searcher.stats.staircase_gate_extended == 1


def test_g10_stops_when_there_is_no_second_move_to_compare() -> None:
    board = chess.Board("7k/8/6K1/8/8/8/8/7R b - - 0 1")
    assert [move.uci() for move in board.legal_moves] == ["h8g8"]
    engine = ScriptedEngine()
    searcher = searcher_for(
        engine,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    )

    search = searcher.search_position(board)

    assert len(search.phases) == 2
    assert search.staircase_gate is not None
    assert search.staircase_gate.margin_cp is None
    assert search.staircase_gate.extended is False
    assert search.staircase_gate.reason == "fewer_than_two_moves"
    assert search.staircase_gate.decision_depth_observed == 10
    assert searcher.stats.staircase_gate_forced_stops == 1


def test_g10_never_calls_an_earlier_complete_block_d10() -> None:
    engine = ScriptedEngine(
        preferred=("e2e4", "d2d4"),
        final_depth_offset=2,
    )
    searcher = searcher_for(
        engine,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    )

    search = searcher.search_position(chess.Board())

    assert [phase.depth_requested for phase in search.phases] == [9, 10]
    assert max(block.depth for block in search.phases[-1].parse.blocks) == 8
    assert search.staircase_gate is not None
    assert search.staircase_gate.margin_cp is None
    assert search.staircase_gate.extended is False
    assert search.staircase_gate.reason == "decision_block_incomplete"
    assert search.staircase_gate.decision_depth_observed == 8
    assert searcher.stats.staircase_gate_forced_stops == 1
    assert len(engine.go_lines) == 2


def test_the_g10_shape_remains_fixed_without_the_named_policy() -> None:
    engine = ScriptedEngine(preferred=("e2e4",))
    searcher = searcher_for(engine, staircase=corpus.G10_STAIRCASE)

    search = searcher.search_position(chess.Board())

    assert len(search.phases) == 3
    assert search.staircase_gate is None


def test_a_narrow_position_clamps_every_rung_to_its_legal_moves() -> None:
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    # Two legal moves for black: the king's only squares out of the check.
    board = chess.Board("7k/8/8/8/8/8/8/K6R b - - 0 1")
    assert board.legal_moves.count() == 2

    search = searcher.search_position(board)
    assert [p.width_realized for p in search.phases] == [2, 2, 2]
    assert search.value_full_width is True


def test_an_illegal_narrowing_move_is_refused_by_the_drivers_own_validator() -> None:
    """Stockfish SILENTLY IGNORES an illegal root move, so this must raise.

    An unvalidated ``searchmoves`` list whose entries are all ignored widens the
    search back to full width with no error anywhere -- the phase then reports a
    narrowing that did not happen, which is this repo's signature defect.
    """
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    with pytest.raises(ValueError, match="not legal in this position"):
        searcher.stream(
            corpus.history_for(chess.Board()),
            depth=5, multipv=1, searchmoves=["e7e5"],
        )


def test_the_narrowing_reads_the_deepest_iteration_not_the_shallowest() -> None:
    """A ranking that CHANGES WITH DEPTH, so the choice of block is visible.

    ⚑ THIS IS WHY THE FAKE GREW A DEPTH-DEPENDENT MODE.  With a depth-invariant
    engine every iteration of a phase emits the same order, so
    ``deepest_block_with_width`` returning the SHALLOWEST full-width block is
    indistinguishable from it returning the deepest -- and the deep phases of
    this staircase exist entirely to correct the shallow scout's ranking.  Here
    ``d5c3`` is refuted below depth 4 and best at depth 4, so the shallow answer
    and the deep answer are opposite.

    Mutation caught: ``min(full, key=...)`` in ``deepest_block_with_width``.
    The narrowing then hands the deep rung the moves the SCOUT'S FIRST GUESS
    liked, and every phase after it spends its budget on refuted moves while
    reporting a perfectly well-formed staircase.
    """
    engine = ScriptedEngine(deep_favourite="d5c3", deep_from_depth=4)
    searcher = searcher_for(engine, staircase="all:4,3:6")

    search = searcher.search_position(chess.Board(MATE_GAME_FEN))

    scout = search.phases[0]
    assert [b.depth for b in scout.parse.blocks] == [1, 2, 3, 4]
    assert scout.parse.blocks[0].lines[0].move != "d5c3", "refuted at depth 1"
    assert scout.parse.blocks[-1].lines[0].move == "d5c3", "best at depth 4"

    # The narrowing carries the DEEPEST iteration's order into `searchmoves`...
    assert engine.go_lines[1].split("searchmoves ")[1].split()[0] == "d5c3"
    # ... and selection reads the same block.
    assert search.value_depth == 4
    assert search.values[0].move == "d5c3"


def test_selection_reads_the_full_width_scout_not_the_narrowed_rung() -> None:
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    board = chess.Board(MATE_GAME_FEN)

    search = searcher.search_position(board)

    assert len(search.values) == board.legal_moves.count()
    assert search.value_depth == 9
    assert search.value_full_width is True
    assert searcher.stats.selection_not_full_width == 0


# ── search cost aggregates ───────────────────────────────────────────────────


def test_the_node_stats_are_running_aggregates_with_a_bounded_sample() -> None:
    """Exact where it can be, sampled where it cannot, and it says which.

    Mutation caught: keeping every observation (the shape this replaced).  A
    per-search list is ~84 MB per million positions per worker, held for the
    life of the run to produce four numbers -- and it never appears in any
    output, so nothing but this assertion notices it is there.
    """
    samples = corpus.NodeSamples(rng=np.random.default_rng(0))
    for value in range(1, 10_001):
        samples.add(value)
    cell = samples.summary()

    assert (cell["n"], cell["min"], cell["max"]) == (10_000, 1, 10_000)
    assert cell["total"] == sum(range(1, 10_001))
    assert cell["mean"] == pytest.approx(5000.5)
    assert cell["log2_buckets"]["0"] == 1, "only the value 1 lands in bucket 0"

    # ⚑ The sample is BOUNDED and the key says the median is an estimate.
    assert len(samples.reservoir) == corpus.NODES_RESERVOIR_MAX
    assert cell["median_est_reservoir_n"] == corpus.NODES_RESERVOIR_MAX
    assert cell["median_est_reservoir_capacity"] == corpus.NODES_RESERVOIR_MAX
    assert "median" not in cell, "an estimate must not wear the exact name"
    assert cell["median_est_reservoir"] == pytest.approx(5000.5, rel=0.05)


def test_an_empty_node_cell_reports_nothing_rather_than_zero() -> None:
    cell = corpus.NodeSamples().summary()
    assert cell["n"] == 0
    assert math.isnan(cell["mean"])
    assert math.isnan(cell["median_est_reservoir"])
    assert (cell["min"], cell["max"], cell["total"]) == (0, 0, 0)


def test_the_search_stats_bucket_node_counts_per_phase(tmp_path: Path) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    _, searcher, _ = play(MATE_GAME_FEN, tmp_path, engine=engine)

    cells = searcher.stats.summary()["nodes_by_phase"]
    assert set(cells) == {"0", "1", "2"}
    assert all(cell["n"] == 2 for cell in cells.values()), "two searched plies"
    assert all(cell["median_est_reservoir_n"] == 2 for cell in cells.values())
    assert all(cell["max"] >= cell["min"] > 0 for cell in cells.values())


# ── the shared cp -> q mapping ───────────────────────────────────────────────


def test_the_generator_and_the_gate_share_one_cp_mapping_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proved by EXECUTION: replace the one object, watch this file's q move.

    ``gate.q_from_effective_cp`` resolves ``gen.cp_to_wdl_array`` as a module
    attribute at call time, which is what lets the native arms, the gate's
    Stockfish arms and this generator be one mapping rather than three.
    """
    searcher = searcher_for(ScriptedEngine())
    values = corpus.SelectionValues.from_lines(
        (corpus.PvLine(rank=1, move="e2e4", effective_cp=50.0, nodes=1),),
    )
    before = searcher.q_of(values)

    def reversed_map(eff_cp, *, slope, draw_width_cp):
        del slope, draw_width_cp
        arr = np.asarray(eff_cp, dtype=np.float64)
        # W and L swapped: any consumer of the shared object inverts with it.
        return np.stack(
            [np.zeros_like(arr), np.zeros_like(arr), np.ones_like(arr)], axis=-1,
        ).astype(np.float32)

    monkeypatch.setattr(gen, "cp_to_wdl_array", reversed_map)
    after = searcher.q_of(values)

    assert float(before[0]) > 0.0
    assert float(after[0]) == -1.0


def test_the_mate_band_is_the_gates_and_score_mate_zero_is_a_real_score() -> None:
    parsed = corpus.parse_depth_blocks(
        ["info depth 5 multipv 1 score mate 0 nodes 7 pv e2e4"], expected_lines=1,
    )
    assert parsed.blocks[0].lines[0].effective_cp == gate.effective_cp_from_score(
        None, 0,
    )
    assert parsed.unscored_lines == 0


# ── Gumbel selection ─────────────────────────────────────────────────────────


def test_the_temperature_knob_changes_what_gets_played() -> None:
    """Fixed seed, fixed values, two temperatures, two distributions.

    Mutation caught: ignoring ``temp`` (e.g. ``argmax(q + noise)``).  The cold
    arm then still looks fine -- it is the HOT arm that collapses onto the
    argmax and stops exploring, so a test that only checked the cold arm would
    pass on the mutant.
    """
    q = np.array([1.0, 0.9, 0.2, -0.5])
    cold = [
        corpus.gumbel_choice(
            q, temp=0.01,
            rng=corpus.selection_rng(seed=1, worker_id=0, game_id=0, ply=ply),
        )
        for ply in range(200)
    ]
    hot = [
        corpus.gumbel_choice(
            q, temp=5.0,
            rng=corpus.selection_rng(seed=1, worker_id=0, game_id=0, ply=ply),
        )
        for ply in range(200)
    ]

    assert set(cold) == {0}, "a cold temperature is the argmax"
    assert len(set(hot)) > 1, "a hot temperature must actually explore"
    assert hot != cold


def test_the_selection_draw_is_reproducible_and_has_no_wall_clock() -> None:
    """Same material, same draw; DIFFERENT worker, different draw.

    Mutation caught: dropping ``worker_id`` from the seed material.  Two workers
    would then walk the same games move for move from any shared position, and
    the corpus would silently be worth a fraction of its wall time.  ⚑ The
    worker arm compares the SEQUENCES, not one ``gumbel_choice`` index: over
    three values the argmax collides often enough that a single draw is not
    evidence, which is how the previous ``isinstance(..., int)`` assertion came
    to be vacuous.
    """
    q = np.array([0.4, 0.35, 0.3])
    first = corpus.gumbel_choice(
        q, temp=1.0,
        rng=corpus.selection_rng(seed=5, worker_id=2, game_id=9, ply=11),
    )
    again = corpus.gumbel_choice(
        q, temp=1.0,
        rng=corpus.selection_rng(seed=5, worker_id=2, game_id=9, ply=11),
    )
    assert first == again

    worker_a = corpus.selection_rng(seed=5, worker_id=2, game_id=9, ply=11)
    worker_b = corpus.selection_rng(seed=5, worker_id=3, game_id=9, ply=11)
    assert not np.array_equal(worker_a.random(16), worker_b.random(16))


def test_the_book_stream_and_the_selection_stream_are_tagged_apart() -> None:
    book = corpus.book_rng(seed=3, worker_id=1, game_id=4)
    select = corpus.selection_rng(seed=3, worker_id=1, game_id=4, ply=0)
    assert not np.array_equal(book.random(8), select.random(8))


@pytest.mark.parametrize("temp", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_infinite_temperature_is_refused(temp: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        corpus.gumbel_choice(
            np.array([1.0, 0.0]), temp=temp, rng=np.random.default_rng(0),
        )


def test_the_temperature_schedule_switches_at_temp_plies() -> None:
    kwargs: dict[str, Any] = {
        "temp_plies": 20, "temp_high": 1.0, "temp_low": 0.3,
    }
    assert corpus.temperature_for(0, **kwargs) == (1.0, "high")
    assert corpus.temperature_for(19, **kwargs) == (1.0, "high")
    assert corpus.temperature_for(20, **kwargs) == (0.3, "low")
    assert corpus.temperature_for(400, **kwargs) == (0.3, "low")
    # 0 means "never hot", a usable setting rather than an off-by-one.
    assert corpus.temperature_for(0, temp_plies=0, temp_high=1.0, temp_low=0.3) == (
        0.3, "low",
    )


def test_the_realized_temperature_is_stamped_on_every_row(tmp_path: Path) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    outcome, _, _ = play(
        MATE_GAME_FEN, tmp_path, engine=engine,
        temp_plies=1, temp_high=0.01, temp_low=0.02,
    )
    assert [row["ply"] for row in outcome.rows] == [0, 1]
    assert [row["selection"]["temp"] for row in outcome.rows] == [0.01, 0.02]
    assert [row["selection"]["schedule_phase"] for row in outcome.rows] == [
        "high", "low",
    ]
    assert all(row["selection"]["temp_plies"] == 1 for row in outcome.rows)


# ── dedup ────────────────────────────────────────────────────────────────────


def test_the_dedup_key_drops_the_fullmove_number_and_keeps_the_halfmove_clock(
) -> None:
    a = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 4 20")
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 4 99")
    c = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 9 20")
    assert corpus.dedup_key(a) == corpus.dedup_key(b)
    # The clock decides how close the fifty-move rule is, so it changes the
    # search and must not be folded out of the key.
    assert corpus.dedup_key(a) != corpus.dedup_key(c)


def one_value(move: str, cp: float = 0.0) -> corpus.SelectionValues:
    return corpus.SelectionValues.from_lines(
        (corpus.PvLine(rank=1, move=move, effective_cp=cp, nodes=1),),
    )


def test_the_cache_entry_is_the_compact_pair_not_the_banked_lines() -> None:
    """What the cache holds, and why it is ~10x smaller than what gets banked.

    Mutation caught: caching the ``tuple[PvLine, ...]`` (the shape this
    replaced) -- ~10.3 KiB per 35-move position against ~0.8 KiB here, which at
    the default bound is the difference between a worker that fits on the box
    and one that does not.  Two of the three savings are asserted directly: the
    per-line rank/nodes are simply not here, and the uci spellings are INTERNED,
    so a million cached positions share one object per spelling rather than
    holding a million copies of ``"e2e4"``.
    """
    lines = (
        corpus.PvLine(rank=1, move="e2e4", effective_cp=31.0, nodes=100),
        corpus.PvLine(rank=2, move="d2d4", effective_cp=-12.5, nodes=110),
    )
    values = corpus.SelectionValues.from_lines(lines)

    assert values.moves == ("e2e4", "d2d4")
    assert values.effective_cp.dtype == np.float32
    np.testing.assert_array_equal(
        values.effective_cp, np.array([31.0, -12.5], dtype=np.float32),
    )

    # A spelling built at RUNTIME is a distinct object until it is interned.
    joined = "".join(("e2", "e4"))
    again = corpus.SelectionValues.from_lines(
        (corpus.PvLine(rank=1, move=joined, effective_cp=0.0, nodes=1),),
    )
    assert again.moves[0] is values.moves[0], "the uci spellings must be interned"


def test_the_cached_q_is_the_same_number_the_first_visit_selected_on() -> None:
    """float32 storage must not move the selection between visit 1 and visit 2.

    The narrow dtype is what makes the entry small; it is safe only because
    effective cp is integral centipawns and the mate band is exact in float32.
    ⚑ And because selection reads the COMPACT object on the first visit too --
    a generator that selected off the ``PvLine`` list first and off the cache
    afterwards could play a different move on a repeat of the same position and
    nothing would say so.
    """
    searcher = searcher_for(ScriptedEngine())
    lines = tuple(
        corpus.PvLine(rank=i + 1, move=m, effective_cp=cp, nodes=10 * i)
        for i, (m, cp) in enumerate(
            (("e2e4", 31.0), ("d2d4", -12.0), ("g1f3", 4000.0)),
        )
    )
    compact = corpus.SelectionValues.from_lines(lines)

    exact = gate.q_from_effective_cp(
        np.asarray([pv.effective_cp for pv in lines], dtype=np.float64),
        slope=gen.NNUE_CP_SLOPE, draw_width_cp=gen.NNUE_CP_DRAW_WIDTH,
    )
    np.testing.assert_array_equal(searcher.q_of(compact), exact)


def test_the_dedup_cache_bound_evicts_the_oldest_and_counts_it() -> None:
    """FIFO, not LRU, and the eviction is COUNTED rather than absorbed."""
    cache = corpus.DedupCache(max_entries=2)
    values = {key: one_value(move) for key, move in (
        ("k0", "a2a3"), ("k1", "b2b3"), ("k2", "c2c3"), ("k3", "d2d3"),
    )}
    for key in ("k0", "k1", "k2"):
        cache.put(key, values[key])

    assert len(cache) == 2
    assert cache.evictions == 1
    assert cache.get("k0") is None, "the OLDEST entry is the one that goes"
    assert cache.get("k1") is values["k1"]

    # ⚑ A hit does not save an entry: serving k1 must not reorder it, or the
    # cache would be an LRU and would keep the opening tree resident forever.
    cache.put("k3", values["k3"])
    assert cache.get("k1") is None
    assert cache.evictions == 2

    summary = cache.summary()
    assert summary["dedup_cache_max_entries"] == 2
    assert summary["dedup_cache_entries"] == 2
    assert summary["dedup_cache_evictions"] == 2
    assert summary["dedup_cache_eviction_policy"] == "fifo"
    assert "RE-BANKED" in summary["dedup_cache_eviction_semantics"]
    # The realized cost of the bound, measured off the entries it holds.
    per_entry = summary["dedup_cache_bytes_per_entry_est"]
    assert per_entry > 0.0
    assert summary["dedup_cache_bytes_est"] == pytest.approx(2 * per_entry)


def test_a_re_put_of_a_cached_key_is_not_a_second_entry() -> None:
    cache = corpus.DedupCache(max_entries=2)
    first = one_value("a2a3")
    cache.put("k0", first)
    cache.put("k0", one_value("b2b3"))
    assert len(cache) == 1
    assert cache.get("k0") is first, "the first-seen values are the ones kept"
    assert cache.evictions == 0


def test_a_non_positive_dedup_cache_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="dedup-cache-max must be positive"):
        corpus.DedupCache(max_entries=0)


def test_an_evicted_position_is_re_searched_and_re_banked(tmp_path: Path) -> None:
    """The BOUND's semantic, end to end, and the counters that disclose it.

    Mutation caught: an unbounded cache (dropping the eviction loop, which is
    what the plain ``dict`` this replaced was).  Game 1 then serves both of game
    0's positions from cache, banks nothing, and ``dedup_cache_evictions``
    stays 0 -- so this test fails on all three of its counts.

    ⚑ The re-banked rows are NOT a defect: they are two genuine independent
    searches of one position, exactly like the two rows two workers already
    produce for a shared position, and the summary says how many to expect.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    spec = worker_spec(tmp_path, dedup_cache_max=1)
    searcher = searcher_for(engine)
    dedup = corpus.DedupStats()
    cache = corpus.DedupCache(max_entries=spec.dedup_cache_max)
    opening = fen_opening(MATE_GAME_FEN, tmp_path)

    first = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup, progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )
    second = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup, progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )

    assert [row["dedup_key"] for row in second.rows] == [
        row["dedup_key"] for row in first.rows
    ]
    assert len(second.rows) == 2, "an evicted position is banked again"
    assert cache.evictions >= 2
    assert sum(dedup.hits.values()) == 0, "the bound let every repeat through"
    assert searcher.stats.positions == 4
    # ... and the corpus is otherwise unchanged: same moves, same result.
    assert (second.plies, second.result_pgn) == (first.plies, first.result_pgn)


def test_a_repeated_position_is_served_from_cache_and_never_re_banked(
    tmp_path: Path,
) -> None:
    """Mutation caught: banking a row on a cache hit (a duplicate observation
    published as an independent one), and never consulting the cache at all.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    spec = worker_spec(tmp_path)
    searcher = searcher_for(engine)
    dedup = corpus.DedupStats()
    cache = corpus.DedupCache(max_entries=spec.dedup_cache_max)
    opening = fen_opening(MATE_GAME_FEN, tmp_path)

    first = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup, progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )
    go_lines_after_first = len(engine.go_lines)
    searched_after_first = searcher.stats.positions
    seen_after_first = dict(dedup.first_seen)

    second = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup, progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )

    assert len(first.rows) == 2
    assert second.rows == [], "a cache-served position is never re-banked"
    assert len(engine.go_lines) == go_lines_after_first, "no second search"
    assert searcher.stats.positions == searched_after_first
    assert dict(dedup.first_seen) == seen_after_first
    assert sum(dedup.hits.values()) == 2
    # Cache-served selection is the same selection: same plies, same result.
    assert (second.plies, second.result_pgn) == (first.plies, first.result_pgn)


# ── the two dedup keys ───────────────────────────────────────────────────────
#
# Every route below is a knight shuffle from the standard start, so the
# halfmove clock never resets and the FEN (with clock) at the end is shared by
# the pair.  The routes are pre-pushed through the production seed grammar
# (``<fen> | <uci> ...``) so the game's OWN ply 0 is the position under test and
# ``max_plies=1`` searches exactly that one position per game.

#: T1: one FEN, one clock, different prior frames.
T1_ROUTE_A = "g1f3 g8f6 b1c3 b8c6"
T1_ROUTE_B = "b1c3 b8c6 g1f3 g8f6"
#: T2: one FEN, one clock; A arrives at a position it has ALREADY occupied
#: (a 2-fold, 4 plies apart), B arrives there for the first time.
T2_ROUTE_REPEATED = "g1f3 g8f6 b1c3 b8c6 f3g1 f6g8 g1f3 g8f6"
T2_ROUTE_DIRECT = "g1h3 g8h6 h3g5 h6g4 g5f3 g4f6 b1c3 b8c6"
#: T4: identical last 8 positions (so identical tensors) after a prefix that
#: in D repeats a position (plies 2 and 6) and in E repeats nothing.  The
#: repeat sits 8+ plies behind the searched position -- outside every encoded
#: frame, inside the reversible segment the engine reads.
T4_TAIL = "b1a3 b8c6 f3g1 f6g8 g1f3 c6a5 a3b1 a5c4"
T4_ROUTE_OLD_REPEAT = f"g1f3 g8f6 b1c3 b8c6 c3b1 c6b8 {T4_TAIL}"
T4_ROUTE_NO_REPEAT = f"g1f3 b8a6 b1c3 g8f6 c3b1 a6b8 {T4_TAIL}"


def board_after(moves: str) -> chess.Board:
    board = chess.Board()
    for uci in moves.split():
        board.push_uci(uci)
    return board


def live_input_key(board: chess.Board) -> str:
    """The hash of EXACTLY what live search encodes -- spelled here, not imported."""
    return corpus.input_tensor_key(encode_cboard(
        CBoard.from_board(board),
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    ))


class SeededGames:
    """One engine, one cache, one stats object; a game per pre-pushed route."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.engine = ScriptedEngine()
        self.spec = worker_spec(tmp_path, max_plies=1)
        self.searcher = searcher_for(self.engine, staircase=self.spec.staircase)
        self.cache = corpus.DedupCache(max_entries=self.spec.dedup_cache_max)
        self.dedup = corpus.DedupStats()
        self.seq = corpus.WorkerSeq()
        self.games = 0

    def play(self, moves: str, *, tag: str) -> corpus.GameOutcome:
        line = f"{chess.STARTING_FEN} | {moves}"
        outcome = corpus.play_game(
            spec=self.spec, searcher=self.searcher,
            opening_cfg=fen_list_opening([line], self.tmp_path / f"{tag}.txt"),
            game_id=self.games, cache=self.cache, dedup=self.dedup,
            progress=corpus.WorkerProgress(), seq=self.seq,
        )
        self.games += 1
        return outcome

    @property
    def go_count(self) -> int:
        return len(self.engine.go_lines)

    def counters(self) -> dict[str, int]:
        summary = self.dedup.summary()
        return {
            name: summary[name] for name in (
                "dup_hits", "row_key_hits", "search_key_hits",
                "search_key_hit_on_new_input", "search_key_miss_on_seen_input",
                "searches", "rows_banked",
            )
        }


def test_the_search_key_is_the_dedup_key_plus_the_reversible_segments_repeats() -> None:
    """The exact string, because it is what the cache is keyed on."""
    direct = board_after(T2_ROUTE_DIRECT)
    repeated = board_after(T2_ROUTE_REPEATED)
    assert direct.fen() == repeated.fen(), "the pair no longer shares a FEN"
    assert corpus.dedup_key(direct) == corpus.dedup_key(repeated)
    assert corpus.search_key(direct) == f"{corpus.dedup_key(direct)}|"
    assert corpus.search_key(repeated) == (
        f"{corpus.dedup_key(repeated)}|"
        f"{chess.polyglot.zobrist_hash(repeated):016x}:2"
    )
    # The count is CAPPED: a 3-fold and a 4-fold are the same engine state.
    # (A third cycle also makes every intermediate position a 2-fold, so the
    # signature holds several entries; the current position's is read by hash.)
    three = board_after(T2_ROUTE_REPEATED + " f3g1 f6g8 g1f3 g8f6")
    four = board_after(T2_ROUTE_REPEATED + " f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 g1f3 g8f6")
    assert three.is_repetition(3)
    assert four.is_repetition(4)
    assert f"{chess.polyglot.zobrist_hash(three):016x}:3" in corpus.search_key(three)
    assert f"{chess.polyglot.zobrist_hash(four):016x}:3" in corpus.search_key(four)
    assert ":4" not in corpus.search_key(four)
    # An irreversible move ends the segment: the start position repeated
    # BEFORE e2e4, and the engine cannot see it, so neither does the key.
    cut = board_after("g1f3 g8f6 f3g1 f6g8 e2e4")
    assert cut.halfmove_clock == 0
    assert corpus.search_key(cut) == f"{corpus.dedup_key(cut)}|"


#: ⚑ THE ROUTE WHERE THE TWO REGIMES DISAGREE: two knight cycles (every position
#: a repeat) then ``e2e4 e7e5``.  The irreversible move clears the hash stack
#: the UNFIXED encoder rebuilds repetition planes from, so at plies 9-10 the
#: older frames' repetition planes are set only under the FIXED regime.
#: Measured in fresh subprocesses (2026-09-01): of 102 positions across seven
#: routes, exactly these two hash differently.
REGIME_SENSITIVE_ROUTE = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 e2e4 e7e5"


def unfixed_key_in_a_fresh_process(moves: str) -> str:
    """``input_tensor_key`` of the C planes under the UNFIXED regime.

    A fresh interpreter, because the fixed flag cannot be flipped back under a
    live board in this one (``rep_fix.RepFixFlipError``) -- and because the
    unfixed regime is the C DEFAULT, which is the whole hazard.
    """
    code = (
        "import chess, sys\n"
        "from chess_anti_engine.encoding import rep_fix\n"
        "from chess_anti_engine.encoding._lc0_ext import CBoard\n"
        "from chess_anti_engine.encoding.cboard_encode import encode_cboard\n"
        "from scripts import gen_sf_rooted_corpus as corpus\n"
        "rep_fix.apply(False)\n"
        "b = chess.Board()\n"
        f"for u in {moves.split()!r}: b.push_uci(u)\n"
        "print(corpus.input_tensor_key(encode_cboard(CBoard.from_board(b), "
        "input_history_encoding=corpus.INPUT_HISTORY_ENCODING, "
        "input_extra_features=corpus.INPUT_EXTRA_FEATURES)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=str(corpus.REPO_ROOT), env={**os.environ, "PYTHONPATH": str(corpus.REPO_ROOT)},
    )
    return proc.stdout.strip().splitlines()[-1]


def test_the_generator_writes_the_fixed_input_key_where_the_regimes_differ(
    tmp_path: Path,
) -> None:
    """⚑⚑ The regime is part of the key (#498 reviewer, blocking).

    The row's ``input_key`` must be the FIXED hash -- the one the deriver's
    python encoder reproduces -- and not the C default's.
    """
    board = board_after(REGIME_SENSITIVE_ROUTE)
    unfixed = unfixed_key_in_a_fresh_process(REGIME_SENSITIVE_ROUTE)
    fixed = fixed_key(REGIME_SENSITIVE_ROUTE)
    assert unfixed != fixed, "the route no longer separates the regimes"

    games = SeededGames(tmp_path)
    outcome = games.play(REGIME_SENSITIVE_ROUTE, tag="regime")
    (row,) = outcome.rows
    assert row["input_key"] == fixed
    assert row["input_key"] != unfixed
    assert row["run"][corpus.KEY_HISTORY_REP_FIX] is True
    assert corpus.row_key(board) == fixed


def test_row_key_refuses_a_process_in_the_wrong_regime() -> None:
    """The precondition is loud: an unset or unfixed flag cannot hash a row."""
    code = (
        "import chess\n"
        "from scripts import gen_sf_rooted_corpus as corpus\n"
        "try:\n"
        "    corpus.row_key(chess.Board())\n"
        "except RuntimeError as exc:\n"
        "    print('REFUSED', 'history_rep_fix is None' in str(exc))\n"
        "else:\n"
        "    print('ACCEPTED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=str(corpus.REPO_ROOT), env={**os.environ, "PYTHONPATH": str(corpus.REPO_ROOT)},
    )
    assert proc.stdout.strip().splitlines()[-1] == "REFUSED True"


def test_the_generators_regime_is_productions() -> None:
    """``HISTORY_REP_FIX`` is read from the same place production reads it."""
    import yaml

    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    cfg = flatten_run_config_defaults(yaml.safe_load(
        (corpus.REPO_ROOT / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8"),
    ))
    assert corpus.HISTORY_REP_FIX is bool(cfg["history_rep_fix"])


#: The other route where the regimes differ (delta reviewer, measured over all
#: 522 gate positions): the irreversible ``e2e4`` lands INSIDE the 8 frames
#: after the ``c3b1 c6b8`` repeat, and ``f3g1`` at ply 11 keeps it in the window.
IRREVERSIBLE_AFTER_LONG_RUN = "g1f3 g8f6 b1c3 b8c6 c3b1 c6b8 b1a3 b8a6 e2e4 e7e5 f3g1"
#: ``(route, plies)`` -> the board the subprocess gate must key like the deriver:
#: ``repetition_then_irreversible_in_window[9]`` and
#: ``irreversible_after_long_run[10]``, plus ``[10]`` of the first.
REGIME_SENSITIVE_POSITIONS = (
    (REGIME_SENSITIVE_ROUTE, 9),
    (REGIME_SENSITIVE_ROUTE, 10),
    (IRREVERSIBLE_AFTER_LONG_RUN, 10),
)


def fixed_key(moves: str) -> str:
    """The DERIVER's hash of the position: python encoder, always fixed."""
    return corpus.input_tensor_key(np.asarray(encode_position(
        board_after(moves), add_features=True,
        input_history_encoding=corpus.INPUT_HISTORY_ENCODING,
        input_extra_features=corpus.INPUT_EXTRA_FEATURES,
    ), dtype=np.float32))


def test_a_fresh_process_generates_and_derives_under_one_regime(tmp_path: Path) -> None:
    """⚑⚑ THE SUBPROCESS GATE, THROUGH THE REAL POOL.  A fresh interpreter (C
    default: UNFIXED) runs the generator through ``corpus.run`` with
    ``--workers 2`` -- the production ``spawn`` pool, so ``run_worker`` runs in
    CHILD interpreters that inherit no C global and no monkeypatch -- on the
    three regime-sensitive positions with a scripted engine, then the deriver
    over the whole output with ``enforce_input_key_take_effect`` armed.  No
    fixture applies the flag anywhere here: only ``run_worker``'s own
    ``apply_history_rep_fix`` can make every banked ``input_key`` the
    deriver's hash, and each of the three is also checked by value against
    the FIXED key computed in this process and the UNFIXED key from a third
    interpreter -- so the gate is one that can fail.

    ⚑ The seams are applied at the DRIVER's module level, not under its main
    guard: a spawned child re-imports the parent's ``__main__`` as
    ``__mp_main__`` (module body runs, main guard does not), which is exactly
    what re-applies the scripted engine and the dealt openings in each child.
    Every other ``--workers 2`` test in this file swaps the pool for an inline
    executor, and every direct ``run_worker`` test runs under this file's
    autouse ``production_rep_fix`` fixture, so this is the one test that can
    see a worker that forgot the call (Fable, round 3 delta: the mutant
    "drop ``apply_history_rep_fix()`` from ``run_worker``" survived the whole
    suite before it existed).  Mutant: that line dropped -> ``row_key``
    raises in the child, the run dies, this test fails.  ⚑ Dropping the call
    from ``run()`` alone is an EQUIVALENT mutant: ``run`` builds no CBoard
    before dispatch, and the in-process single-worker branch calls
    ``run_worker`` too.
    """
    lines = [
        f"{chess.STARTING_FEN} | {' '.join(route.split()[:plies])}"
        for route, plies in REGIME_SENSITIVE_POSITIONS
    ]
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "from scripts import derive_corpus_targets as derive\n"
        "from scripts import gen_sf_rooted_corpus as corpus\n"
        "from tests.test_gen_sf_rooted_corpus import (\n"
        "    SMOKE_SYZYGY, ScriptedEngine, fen_list_opening, uci_double,\n"
        ")\n"
        "# MODULE LEVEL: re-executed by every spawned worker (__mp_main__).\n"
        f"root = Path({str(tmp_path)!r})\n"
        f"lines = {lines!r}\n"
        "mp = pytest.MonkeyPatch()\n"
        "mp.setattr(corpus, 'StockfishUCI', lambda *_a, **_kw: uci_double(ScriptedEngine()))\n"
        "mp.setattr(corpus, 'build_opening_config',\n"
        "           lambda _spec: fen_list_opening(lines, root / 'seeds.txt'))\n"
        "# The production sampler draws WITH replacement; deal line game_id %\n"
        "# len(lines) so every game gets its own whichever worker plays it.\n"
        "real_rng, real_sample = corpus.book_rng, corpus.sample_starting_board\n"
        "class Dealt:  # numpy Generators take no attributes: wrap one\n"
        "    def __init__(self, rng, line): self.rng, self.dealt_line = rng, line\n"
        "    def __getattr__(self, name): return getattr(self.rng, name)\n"
        "def dealing_rng(*, seed, worker_id, game_id):\n"
        "    rng = real_rng(seed=seed, worker_id=worker_id, game_id=game_id)\n"
        "    return Dealt(rng, lines[int(game_id) % len(lines)])\n"
        "mp.setattr(corpus, 'book_rng', dealing_rng)\n"
        "mp.setattr(corpus, 'sample_starting_board', lambda *, rng, cfg: real_sample(\n"
        "    rng=rng, cfg=fen_list_opening([rng.dealt_line], root / f'seed{rng.random()}.txt')))\n"
        "# The deriver drops result-less rows before it verifies them, and a\n"
        "# ply-capped scripted game has no result: stamp a draw so every row\n"
        "# reaches the input_key check. A test seam, not a generator setting.\n"
        "mp.setattr(corpus, 'result_from_pov', lambda _r, *, white_to_move: 0.0)\n"
        "if __name__ == '__main__':\n"
        "    out = root / 'corpus'\n"
        "    summary = corpus.run(corpus.build_parser().parse_args([\n"
        "        '--out-dir', str(out), '--games', str(len(lines)), '--workers', '2',\n"
        "        '--syzygy-path', str(SMOKE_SYZYGY or corpus.REPO_ROOT),\n"
        "        '--temp-high', '0.01', '--temp-low', '0.01', '--nice', '0', '--max-plies', '1',\n"
        "    ]))\n"
        "    rows = [{'fen': r['fen'], 'input_key': r['input_key'], 'worker': p.name[:3],\n"
        "             'stamp': r['run'][corpus.KEY_HISTORY_REP_FIX]}\n"
        "            for p in sorted(out.glob('w*.jsonl.zst')) for r in derive.iter_corpus_rows(p)]\n"
        "    rc = derive.main(['--corpus', str(out), '--out', str(root / 'derived'),\n"
        "                      '--scheme', 'uniform-d9', '--temp', '1.0'])\n"
        "    d = json.loads((root / 'derived' / derive.SUMMARY_NAME).read_text())\n"
        "    print(json.dumps({'rc': rc, 'rows': rows, 'banked': summary['rows'],\n"
        "                      'workers': len(summary['config_realized_by_worker']),\n"
        "                      'stamp': summary[corpus.KEY_HISTORY_REP_FIX],\n"
        "                      'verified': d['realized']['input_key_verified'],\n"
        "                      'written': d['realized']['rows_written']}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(driver), str(tmp_path)], capture_output=True, text=True,
        check=False, cwd=str(corpus.REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(corpus.REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["stamp"] is True
    assert result["workers"] == 2, "the real spawn pool, not the in-process branch"
    assert {row["worker"] for row in result["rows"]} == {"w00", "w01"}, (
        "both children banked rows"
    )
    assert result["banked"] == len(lines)
    assert result["verified"] == result["written"] == len(lines)
    keyed = {row["fen"]: row for row in result["rows"]}
    assert len(keyed) == len(lines)
    for route, plies in REGIME_SENSITIVE_POSITIONS:
        moves = " ".join(route.split()[:plies])
        row = keyed[board_after(moves).fen()]
        assert row["stamp"] is True
        unfixed = unfixed_key_in_a_fresh_process(moves)
        assert row["input_key"] == fixed_key(moves), (route, plies)
        assert row["input_key"] != unfixed, (route, plies, "route no longer separates")


def test_the_row_key_is_the_hash_of_the_tensor_live_play_encodes() -> None:
    for moves in (T1_ROUTE_A, T1_ROUTE_B, T2_ROUTE_REPEATED, T4_ROUTE_OLD_REPEAT):
        board = board_after(moves)
        assert corpus.row_key(board) == live_input_key(board), moves
    assert corpus.row_key(board_after(T1_ROUTE_A)) != corpus.row_key(
        board_after(T1_ROUTE_B),
    ), "two routes with different prior frames hashed alike"
    assert corpus.row_key(board_after(T1_ROUTE_A)) == corpus.row_key(
        board_after(T1_ROUTE_A),
    )


def test_t1_two_routes_to_one_fen_with_different_frames_both_bank(
    tmp_path: Path,
) -> None:
    """T1: the second route is a fresh search and a second row, not a hit."""
    games = SeededGames(tmp_path)
    first = games.play(T1_ROUTE_A, tag="a")
    go_after_first = games.go_count
    second = games.play(T1_ROUTE_B, tag="b")

    assert len(first.rows) == 1
    assert len(second.rows) == 1
    a, b = first.rows[0], second.rows[0]
    assert a["fen"] == b["fen"]
    assert a["dedup_key"] == b["dedup_key"]
    assert a["search_key"] == b["search_key"], "no repeat on either route"
    assert a["input_key"] != b["input_key"], "different frames, one hash"
    assert a["input_key"] == live_input_key(board_after(T1_ROUTE_A))
    assert b["input_key"] == live_input_key(board_after(T1_ROUTE_B))
    assert b["history_uci"] == T1_ROUTE_B.split(), "the row banks ITS route"
    assert games.go_count == 2 * go_after_first, "the second route was searched"
    assert games.counters() == {
        "dup_hits": 0, "row_key_hits": 0, "search_key_hits": 1,
        "search_key_hit_on_new_input": 1, "search_key_miss_on_seen_input": 0,
        "searches": 2, "rows_banked": 2,
    }


def test_t2_a_route_that_repeats_its_position_is_not_served_the_direct_label(
    tmp_path: Path,
) -> None:
    """T2: same FEN, different repetition state -- a fresh search, a new row."""
    assert board_after(T2_ROUTE_REPEATED).is_repetition(2)
    assert not board_after(T2_ROUTE_DIRECT).is_repetition(2)
    games = SeededGames(tmp_path)
    direct = games.play(T2_ROUTE_DIRECT, tag="direct")
    go_after_first = games.go_count
    repeated = games.play(T2_ROUTE_REPEATED, tag="repeated")

    assert len(direct.rows) == 1
    assert len(repeated.rows) == 1
    d, r = direct.rows[0], repeated.rows[0]
    assert d["dedup_key"] == r["dedup_key"]
    assert d["search_key"] != r["search_key"], "the repeat is in the label key"
    assert r["search_key"].endswith(":2")
    assert games.go_count == 2 * go_after_first, "the repeated route was searched"
    assert games.counters() == {
        "dup_hits": 0, "row_key_hits": 0, "search_key_hits": 0,
        "search_key_hit_on_new_input": 0, "search_key_miss_on_seen_input": 0,
        "searches": 2, "rows_banked": 2,
    }
    assert games.cache.get(d["search_key"]) is not games.cache.get(r["search_key"])


def test_t3_a_true_duplicate_is_one_row_and_one_search(tmp_path: Path) -> None:
    """T3: the cache still works -- identical book exit, identical first move."""
    games = SeededGames(tmp_path)
    first = games.play(T1_ROUTE_A, tag="same")
    go_after_first = games.go_count
    second = games.play(T1_ROUTE_A, tag="same")

    assert len(first.rows) == 1
    assert second.rows == [], "a duplicate tensor under a cached label banks nothing"
    assert games.go_count == go_after_first, "served from the cache, no search"
    assert games.counters() == {
        "dup_hits": 1, "row_key_hits": 1, "search_key_hits": 1,
        "search_key_hit_on_new_input": 0, "search_key_miss_on_seen_input": 0,
        "searches": 1, "rows_banked": 1,
    }


def test_t4_an_older_repeat_outside_the_frames_is_searched_and_not_banked(
    tmp_path: Path,
) -> None:
    """T4: same tensor, different engine state -- no row, but no served label either."""
    old_repeat = board_after(T4_ROUTE_OLD_REPEAT)
    no_repeat = board_after(T4_ROUTE_NO_REPEAT)
    # The premise, read off the boards: same FEN and clock, same tensor, and
    # the repeat in D is a position that is NOT among the last 8.
    assert old_repeat.fen() == no_repeat.fen()
    assert corpus.row_key(old_repeat) == corpus.row_key(no_repeat)
    assert corpus.search_key(old_repeat) != corpus.search_key(no_repeat)
    assert corpus.search_key(no_repeat).endswith("|")
    keys = [board_after(" ".join(T4_ROUTE_OLD_REPEAT.split()[:n]))._transposition_key()
            for n in range(len(T4_ROUTE_OLD_REPEAT.split()) + 1)]
    repeated_at = [i for i, k in enumerate(keys) if keys.count(k) > 1]
    assert repeated_at == [2, 6], repeated_at
    assert len(keys) - 1 - max(repeated_at) >= corpus.HISTORY_WINDOW_PLIES + 1

    games = SeededGames(tmp_path)
    first = games.play(T4_ROUTE_NO_REPEAT, tag="no_repeat")
    go_after_first = games.go_count
    second = games.play(T4_ROUTE_OLD_REPEAT, tag="old_repeat")

    assert len(first.rows) == 1
    assert second.rows == [], "the tensor is already in the corpus"
    assert games.go_count == 2 * go_after_first, "the cached label was NOT served"
    assert games.counters() == {
        "dup_hits": 0, "row_key_hits": 1, "search_key_hits": 0,
        "search_key_hit_on_new_input": 0, "search_key_miss_on_seen_input": 1,
        "searches": 2, "rows_banked": 1,
    }
    # ... and the second label is now cached under ITS key, so a third route
    # with the same engine state and tensor is served.
    go_after_second = games.go_count
    third = games.play(T4_ROUTE_OLD_REPEAT, tag="old_repeat")
    assert third.rows == []
    assert games.go_count == go_after_second
    assert games.counters()["dup_hits"] == 1


def test_dup_hits_are_split_by_phase_of_game() -> None:
    assert corpus.game_phase(ply=0, piece_count=32) == corpus.PHASE_OPENING
    assert corpus.game_phase(ply=20, piece_count=32) == corpus.PHASE_OPENING
    assert corpus.game_phase(ply=21, piece_count=32) == corpus.PHASE_MIDDLEGAME
    assert corpus.game_phase(ply=200, piece_count=9) == corpus.PHASE_ENDGAME
    # Precedence: a small-material position that arrived early is an ENDGAME,
    # not an opening, so it cannot be read as a book exit.
    assert corpus.game_phase(ply=3, piece_count=9) == corpus.PHASE_ENDGAME


def test_the_dedup_summary_publishes_every_phase_and_the_rate() -> None:
    stats = corpus.DedupStats()
    stats.first_seen[corpus.PHASE_OPENING] += 3
    stats.hits[corpus.PHASE_ENDGAME] += 1
    summary = stats.summary()
    assert summary["positions_visited"] == 4
    assert summary["dup_hits"] == 1
    assert summary["dup_rate"] == pytest.approx(0.25)
    assert set(summary["dup_hits_by_phase"]) == set(corpus.GAME_PHASES)


# ── the game loop: piece filter, adjudication, result sign ───────────────────


def test_a_row_is_banked_only_at_seven_or_more_pieces(tmp_path: Path) -> None:
    engine = ScriptedEngine(preferred=CAPTURE_CHAIN_SCRIPT)
    outcome, searcher, dedup = play(
        CAPTURE_CHAIN_FEN, tmp_path, engine=engine, max_plies=3,
    )

    # Three plies searched (7, 6 and 5 pieces); only the 7-piece one is banked.
    assert searcher.stats.positions == 3
    assert sum(dedup.first_seen.values()) == 3
    assert [row["piece_count"] for row in outcome.rows] == [7]
    assert outcome.rows[0]["ply"] == 0


def test_a_sub_seven_position_is_searched_for_its_move_and_still_not_banked(
    tmp_path: Path,
) -> None:
    engine = ScriptedEngine()
    outcome, searcher, _ = play(
        "6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1", tmp_path, engine=engine, max_plies=1,
    )
    assert searcher.stats.positions == 1, "the move still needs a search"
    assert outcome.rows == []
    assert outcome.termination == "max_plies"
    assert outcome.result_pgn is None, "a capped game has no result to fabricate"


@pytest.mark.skipif(
    SMOKE_SYZYGY is None, reason="the 3-4 man smoke tablebase is absent",
)
def test_a_four_man_position_is_adjudicated_exactly_and_backfilled_with_sign(
    tmp_path: Path,
) -> None:
    """The <=4-man verdict, and the sign it writes onto a 7-piece row.

    The scripted captures walk 7 -> 6 -> 5 pieces.  Only ply 0 is banked
    (7 pieces, WHITE to move); the game ends at ply 2 on the tablebase verdict,
    and the banked row must carry ``+1.0`` -- the WHITE row of a game white won.

    ⚑ The verdict lands at FIVE men against a 3-4 man set, and that is exact
    rather than lucky: Syzygy's ``probe_wdl`` alpha-betas over CAPTURES first,
    and ``Qxc3`` resolves this position into a won ``KQvKN`` that the set does
    hold.  A test that assumed "5 men needs a 5-man table" would have called
    this a bug in the adjudicator.

    Mutation caught: flipping the result sign.  ``-1.0`` here is a well-formed,
    plausible, silently inverted value target on every row of the corpus.
    """
    engine = ScriptedEngine(preferred=CAPTURE_CHAIN_SCRIPT)
    outcome, _, _ = play(
        CAPTURE_CHAIN_FEN, tmp_path, engine=engine,
        syzygy_path=str(SMOKE_SYZYGY),
    )

    assert outcome.termination == "syzygy"
    assert outcome.result_pgn == "1-0"
    assert outcome.adjudication is not None
    assert outcome.adjudication["kind"] == "syzygy"
    # WHITE is to move at the adjudicated position and is WINNING there. The
    # wdl is that position's OWN side-to-move label, never the banked row's --
    # they coincide here only because both movers happen to be white.
    assert outcome.adjudication["wdl"] == 2
    assert outcome.adjudication["pov"] == "terminal_position_side_to_move"
    assert outcome.adjudication["piece_count"] == 5

    assert len(outcome.rows) == 1
    row = outcome.rows[0]
    assert (row["stm"], row["piece_count"]) == ("w", 7)
    assert row["result"] == 1.0
    assert row["result_pgn"] == "1-0"
    assert row["adjudication"] == outcome.adjudication


@pytest.mark.skipif(
    PRODUCTION_SYZYGY is None, reason="the production 6-man pair is absent",
)
def test_a_black_to_move_adjudication_is_a_white_win_not_a_black_one(
    tmp_path: Path,
) -> None:
    """The other side of ``_adjudicate``: the probe's seat is BLACK's.

    ``Qxa5`` walks 7 -> 6 men into a KQRvKPP that Syzygy calls a LOSS for the
    side to move -- and the side to move is black, so the game's result is
    "1-0".  Those two numbers pointing opposite ways is the whole fixture:
    ``probe_wdl`` answers from the ADJUDICATED POSITION's own seat and
    ``tb_adjudicate_result`` is what turns that into a PGN result.

    ⚑ Mutation caught: restating the wdl -> result convention locally
    (``{0: "0-1", 1: "1/2-1/2", 2: "1-0"}``) instead of delegating, which is
    precisely what ``_adjudicate``'s docstring warns against.  Every existing
    adjudication test here reaches a WHITE-to-move terminal position, where the
    two conventions agree and the mutant survives; this one has black to move,
    so the mutant writes ``-1.0`` onto a row whose side WON.  A well-formed,
    plausible, silently inverted value target.
    """
    assert PRODUCTION_SYZYGY is not None
    engine = ScriptedEngine(preferred=BLACK_ADJUDICATION_SCRIPT)
    outcome, _, _ = play(
        BLACK_ADJUDICATION_FEN, tmp_path, engine=engine,
        syzygy_path=PRODUCTION_SYZYGY,
    )

    assert outcome.termination == "syzygy"
    assert outcome.adjudication is not None
    terminal = chess.Board(outcome.adjudication["fen"])
    assert terminal.turn == chess.BLACK, "the fixture's whole point"
    assert outcome.adjudication["piece_count"] == 6
    # BLACK is to move and is LOSING there ...
    assert outcome.adjudication["wdl"] == 0
    # ... which is a WHITE win for the game.
    assert outcome.result_pgn == "1-0"

    # Every banked row's sign is its OWN mover's.
    assert [(row["stm"], row["result"]) for row in outcome.rows] == [("w", 1.0)]
    for row in outcome.rows:
        assert row["result"] == corpus.result_from_pov(
            outcome.result_pgn, white_to_move=row["stm"] == "w",
        )


def test_both_seats_of_one_game_get_opposite_result_signs(tmp_path: Path) -> None:
    """Mutation caught: writing WHITE's POV onto every row.

    Both banked rows below share one game result and have opposite movers, so a
    white-POV backfill makes the black row exactly backwards.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    outcome, _, _ = play(MATE_GAME_FEN, tmp_path, engine=engine)

    assert outcome.termination == "natural"
    assert outcome.result_pgn == "1-0"
    assert [(row["stm"], row["result"]) for row in outcome.rows] == [
        ("b", -1.0), ("w", 1.0),
    ]
    assert all(row["adjudication"] is None for row in outcome.rows)


@pytest.mark.parametrize(
    ("result", "white", "expected"),
    [
        ("1-0", True, 1.0), ("1-0", False, -1.0),
        ("0-1", True, -1.0), ("0-1", False, 1.0),
        ("1/2-1/2", True, 0.0), ("1/2-1/2", False, 0.0),
        (None, True, None), (None, False, None),
        ("*", True, None),
    ],
)
def test_the_result_pov_table(
    result: str | None, white: bool, expected: float | None,
) -> None:
    assert corpus.result_from_pov(result, white_to_move=white) == expected


def test_an_unfinished_game_never_gets_a_fabricated_result(tmp_path: Path) -> None:
    engine = ScriptedEngine()
    outcome, _, _ = play(MATE_GAME_FEN, tmp_path, engine=engine, max_plies=1)
    assert outcome.termination == "max_plies"
    assert outcome.result_pgn is None
    assert [row["result"] for row in outcome.rows] == [None]
    assert [row["result_pgn"] for row in outcome.rows] == [None]


def test_a_terminal_position_is_never_searched() -> None:
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    with pytest.raises(RuntimeError, match="terminal position"):
        searcher.search_position(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert engine.go_lines == []


def test_the_engine_table_is_cleared_once_per_game(tmp_path: Path) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    play(MATE_GAME_FEN, tmp_path, engine=engine)
    assert engine.commands.count("ucinewgame") == 1
    # ... and NOT between the rungs: the scout warms the table the deep rungs
    # spend, which is the scheme this corpus is generated under.
    assert engine.commands.index("ucinewgame") < engine.commands.index("go depth 9")


# ── shards ───────────────────────────────────────────────────────────────────


def read_shard(path: Path) -> list[dict[str, Any]]:
    """The generator's OWN reader -- a second decoder here is how a codec
    choice comes to disagree with itself between the writer and the resume."""
    return list(corpus.iter_shard_rows(path))


def read_progress(out_dir: Path, worker_id: int) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out_dir / corpus.progress_name(worker_id)).read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def test_a_shard_roundtrips_and_rotates_at_the_next_game_boundary(
    tmp_path: Path,
) -> None:
    """The row bound is a FLOOR crossed at a game boundary, not a hard cap.

    Mutation caught: rotating inside ``write`` on the row count (what this did
    before ``--resume``).  Game 1 below would then be split across two shards
    and the second shard would name no games at all -- and a resume, whose only
    unit is "a game a progress line claims", would have no honest answer for a
    game that is half in a kept shard and half in a deleted one.
    """
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=3, shard_rows=2)
    rows = [{"schema": corpus.ROW_SCHEMA, "ply": i} for i in range(5)]
    for row in rows[:1]:
        writer.write(row)
    writer.end_game(7)                     # 1 row: under the bound, no rotation
    for row in rows[1:4]:
        writer.write(row)
    writer.end_game(8)                     # 4 rows: over the bound, rotates
    for row in rows[4:]:
        writer.write(row)
    writer.end_game(9)
    writer.close()

    assert [shard["rows"] for shard in writer.shards] == [4, 1]
    assert [shard["games"] for shard in writer.shards] == [[7, 8], [9]]
    assert [Path(shard["path"]).name for shard in writer.shards] == [
        f"w03-{i:05d}{writer.suffix}" for i in range(2)
    ]
    read_back = [
        row for shard in writer.shards for row in read_shard(Path(shard["path"]))
    ]
    assert read_back == rows
    assert all(row["schema"] == corpus.ROW_SCHEMA for row in read_back)
    assert read_progress(tmp_path, 3) == writer.shards


def test_a_shard_holding_a_game_that_never_ended_is_left_unlisted(
    tmp_path: Path,
) -> None:
    """The worker died between banking a game's rows and ending it.

    Mutation caught: listing it anyway.  The inventory then names a shard whose
    ``games`` list does not mention the half game inside it, so the next resume
    replays that game and banks its head A SECOND TIME -- the one thing the
    protocol promises cannot happen.  Leaving the file unlisted costs the
    replay of every game it held and cannot duplicate a row.
    """
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=0, shard_rows=100)
    writer.write({"schema": corpus.ROW_SCHEMA, "ply": 0})
    writer.end_game(0)
    writer.write({"schema": corpus.ROW_SCHEMA, "ply": 1})  # game 1 -- never ends
    writer.close()

    assert writer.shards == [], "nothing half-owned reaches the inventory"
    assert [Path(s["path"]).name for s in writer.abandoned] == [
        writer.path_for(0).name,
    ]
    assert writer.abandoned[0]["uncommitted_rows"] == 1
    assert not (tmp_path / corpus.progress_name(0)).exists()
    assert writer.path_for(0).exists(), "the file is there for the resume to eat"

    state = corpus.resume_worker_state(
        out_dir=tmp_path, worker_id=0, cache=corpus.DedupCache(max_entries=4),
    )

    assert state.completed_games == frozenset()
    assert state.deleted_partials == (writer.path_for(0).name,)
    assert not writer.path_for(0).exists()


def test_a_shard_index_continues_from_where_a_resume_says_it_should(
    tmp_path: Path,
) -> None:
    """``first_index`` is a resume's promise not to overwrite a banked shard.

    Mutation caught: ignoring it and restarting the counter at 0 -- the first
    shard the resumed worker writes then collides with the killed session's
    ``w00-00000`` on ``open("x")``, which is a crash on day 14 rather than a
    corruption, but only because the leaf banks refuse to overwrite.
    """
    writer = corpus.ShardWriter(
        out_dir=tmp_path, worker_id=0, shard_rows=1, first_index=4,
    )
    assert writer.first_index == 4
    writer.write({"schema": corpus.ROW_SCHEMA})
    writer.end_game(0)
    assert Path(writer.shards[0]["path"]).name == f"w00-00004{writer.suffix}"
    with pytest.raises(ValueError, match="first_index must be >= 0"):
        corpus.ShardWriter(
            out_dir=tmp_path, worker_id=0, shard_rows=1, first_index=-1,
        )


def test_the_writer_falls_back_to_gzip_without_zstandard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "zstandard_module", lambda: None)
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=0, shard_rows=10)
    writer.write({"schema": corpus.ROW_SCHEMA})
    writer.end_game(0)
    writer.close()

    assert writer.codec == "gzip"
    assert writer.suffix == ".jsonl.gz"
    assert read_shard(Path(writer.shards[0]["path"])) == [
        {"schema": corpus.ROW_SCHEMA},
    ]


def test_a_shard_path_that_already_exists_is_refused(tmp_path: Path) -> None:
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=0, shard_rows=10)
    writer.path_for(0).write_bytes(b"someone else's rows")
    with pytest.raises(FileExistsError):
        writer.write({"schema": corpus.ROW_SCHEMA})


def test_a_populated_out_dir_is_refused(tmp_path: Path) -> None:
    (tmp_path / "w00-00000.jsonl.zst").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already holds files"):
        corpus.refuse_populated_dir(tmp_path)
    # An absent or empty directory is fine.
    corpus.refuse_populated_dir(tmp_path / "fresh")


def test_a_non_positive_shard_rotation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard-rows must be positive"):
        corpus.ShardWriter(out_dir=tmp_path, worker_id=0, shard_rows=0)


def test_the_row_schema_carries_everything_a_join_needs(tmp_path: Path) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    outcome, _, _ = play(MATE_GAME_FEN, tmp_path, engine=engine)
    row = outcome.rows[0]

    assert set(row) == {
        "schema", "run", "fen", "dedup_key", "search_key", "input_key",
        "worker_id", "game_id", "ply", "seq",
        "stm", "piece_count", "game_phase", "played_move", "selection",
        "phases", "result", "result_pgn", "adjudication",
        "history_root_fen", "history_uci", "history_plies",
        "history_root_reason",
    }
    assert row["schema"] == 3
    assert row["search_key"].startswith(row["dedup_key"] + "|")
    assert len(row["input_key"]) == 32
    assert int(row["input_key"], 16) >= 0
    assert row["run"][corpus.KEY_TT_CARRIED] is True
    assert row["run"]["config_sha256"] == "0" * 64
    assert row["played_move"] in {m.uci() for m in chess.Board(row["fen"]).legal_moves}
    assert len(row["phases"]) == 3
    phase = row["phases"][0]
    assert set(phase) == {
        "index", "width_requested", "width_realized", "width_streamed",
        "depth_requested", "searchmoves", "per_depth", "nodes_at_depth",
        "anomalies",
    }
    assert set(phase["per_depth"][0]) == {
        "depth", "complete", "emissions", "nodes_at_depth", "lines",
    }
    # (rank, move, effective_cp, cumulative nodes) -- the lowest-level thing the
    # search reported, banked so a re-analysis is a re-read.
    assert len(phase["per_depth"][0]["lines"][0]) == 4
    assert json.loads(json.dumps(row, sort_keys=True)) == row


def test_a_g10_row_banks_the_decision_that_controls_its_phase_count(
    tmp_path: Path,
) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    outcome, searcher, _ = play(
        MATE_GAME_FEN,
        tmp_path,
        engine=engine,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    )

    assert outcome.rows
    for row in outcome.rows:
        gate = row["staircase_gate"]
        assert gate["policy"] == corpus.STAIRCASE_POLICY_G10
        assert gate["threshold_cp"] == corpus.G10_MARGIN_CP
        assert gate["metric"] == "effective_cp_rank1_minus_rank2"
        assert gate["extended"] is False
        assert gate["margin_cp"] > gate["threshold_cp"]
        assert gate["reason"] == "margin_above_threshold"
        assert len(row["phases"]) == 2
    assert searcher.stats.staircase_gate_evaluations == len(outcome.rows)


# ── the banked move window ───────────────────────────────────────────────────


def test_every_banked_row_carries_the_window_the_engine_was_given(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE TAKE-EFFECT PROOF FOR THE WHOLE SCHEMA, on both halves at once.

    The row's window and the engine's ``position`` line come from ONE object
    (``PositionSearch.history``), so the assertion that the exact command the
    window builds is in the transcript is a structural check that the label was
    computed under the history the row banks -- not two spellings that agree
    today.

    A bare-FEN start (the blind-spot branch) is used on purpose: its early plies
    are the short-history case, and its later ones must reach the full 7.
    """
    engine = ScriptedEngine()
    outcome, _, _ = play(chess.STARTING_FEN, tmp_path, engine=engine)
    assert len(outcome.rows) > 10

    for row in outcome.rows:
        label = f"ply {row['ply']}"
        assert row["schema"] == corpus.ROW_SCHEMA, label
        assert row["history_root_reason"] in {
            corpus.HISTORY_ROOT_IRREVERSIBLE, corpus.HISTORY_ROOT_GAME_START,
        }, label
        assert row["history_plies"] == len(row["history_uci"]), label
        # A bare-FEN start has an empty stack, so the window can be no longer
        # than the ply -- and past the 7 frames it must be at least that long.
        assert row["history_plies"] <= row["ply"], label
        if row["ply"] >= corpus.HISTORY_WINDOW_PLIES:
            assert row["history_plies"] >= corpus.HISTORY_WINDOW_PLIES, label

        replayed = chess.Board(row["history_root_fen"])
        for uci in row["history_uci"]:
            replayed.push(chess.Move.from_uci(uci))
        assert replayed.fen() == row["fen"], label

        command = corpus.position_command(corpus.RowHistory(
            fen=row["fen"],
            root_fen=row["history_root_fen"],
            uci=tuple(row["history_uci"]),
            reason=row["history_root_reason"],
        ))
        # One send per staircase rung, and no other position line for this row.
        assert engine.commands.count(command) == 3, f"{label}: {command}"

    # ⚑ THE OLD FORM IS GONE FROM THE WIRE, not merely joined by a new one.
    with_moves = [c for c in engine.position_lines if " moves " in c]
    assert len(with_moves) == len(engine.position_lines) - 3, (
        "exactly one position (the game's own ply 0, whose window is empty) "
        "may be sent without moves"
    )
    assert max(len(c.split(" moves ")[1].split()) for c in with_moves) >= 7


#: A seed line in the production ``<start_fen> | <uci> ...`` grammar: eight
#: REVERSIBLE king moves, so the board the game starts on already carries a
#: halfmove clock of 8 and a real move stack.  A "last 7 moves" banker reports
#: 7 here; the clock-aware one reports 8.
SEEDED_HISTORY_LINE = (
    "1n4k1/5ppp/8/8/8/8/5PPP/1N4K1 w - - 0 1 | g1f1 g8f8 f1e1 f8e8 e1d1 e8d8 "
    "d1c1 d8c8"
)


def test_the_window_reaches_back_past_the_seven_frames_when_the_clock_allows(
    tmp_path: Path,
) -> None:
    """The root is the last irreversible move, NOT ply-7.

    Driven through the production seeding grammar, so the board the first row is
    banked from arrives with real history on its stack -- the same shape a book
    opening produces.  A 7-move banker would report a flat 7 for every row here.
    """
    engine = ScriptedEngine()
    spec = worker_spec(tmp_path, max_plies=12)
    outcome = corpus.play_game(
        spec=spec,
        searcher=searcher_for(engine, staircase=spec.staircase),
        opening_cfg=fen_list_opening([SEEDED_HISTORY_LINE], tmp_path / "seeded.txt"),
        game_id=0,
        cache=corpus.DedupCache(max_entries=spec.dedup_cache_max),
        dedup=corpus.DedupStats(),
        progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )
    windows = [int(row["history_plies"]) for row in outcome.rows]
    assert windows, "no rows banked"
    assert max(windows) > corpus.HISTORY_WINDOW_PLIES, windows
    assert outcome.rows[0]["history_plies"] == 8, outcome.rows[0]
    reasons = {row["history_root_reason"] for row in outcome.rows}
    assert corpus.HISTORY_ROOT_GAME_START in reasons, reasons


def test_the_worker_summary_counts_the_windows_it_banked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE SUMMARY FIELD'S TAKE-EFFECT PROOF, read off the ROWS.

    The histogram is built where the rows are HANDED TO THE WRITER, not where
    the windows are constructed, so it cannot report a window that was computed
    and then dropped (a sub-``MIN_BANKED_PIECES`` ply and a dedup-served ply
    both build one and bank nothing).  ``sum(histogram) == rows`` is what makes
    that a reading rather than a claim.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *_a, **_kw: uci_double(ScriptedEngine()),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(chess.STARTING_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0,), max_plies=20))

    histogram = result["history_plies_histogram"]
    assert result["rows"] > 0
    assert sum(histogram.values()) == result["rows"]
    assert sum(result["history_root_reasons"].values()) == result["rows"]
    assert max(int(k) for k in histogram) >= corpus.HISTORY_WINDOW_PLIES
    assert set(result["history_root_reasons"]) <= {
        corpus.HISTORY_ROOT_IRREVERSIBLE, corpus.HISTORY_ROOT_GAME_START,
    }


def test_the_window_counters_come_off_committed_shards_not_a_counter_beside_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Codex P2: ``sum(histogram) == rows`` has to hold on the FAILURE path.

    The writer dies on the second row of the first game -- AFTER writing it, so
    the shard holds rows of a game that never ended and is abandoned unlisted.
    ``rows`` reads 0 off the inventory; the histograms must read 0 too.  A
    counter incremented beside ``writer.write`` reads 1 here (the first row's
    increment happened, the second's raise skipped it) and the summary claims
    a window for a row the corpus does not hold.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(corpus, "StockfishUCI", lambda *_a, **_kw: uci_double(engine))
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    original_write = corpus.ShardWriter.write
    writes = {"n": 0}

    def write_then_die(self: corpus.ShardWriter, row: dict[str, Any]) -> None:
        original_write(self, row)
        writes["n"] += 1
        if writes["n"] == 2:
            raise RuntimeError("disk full after the second row")

    monkeypatch.setattr(corpus.ShardWriter, "write", write_then_die)
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))

    assert result["failed"] is not None
    assert result["failed"]["exception"] == "disk full after the second row"
    assert result["rows"] == 0
    assert result["shards"] == []
    assert result["history_plies_histogram"] == {}
    assert result["history_root_reasons"] == {}
    # The rows are not lost silently: the abandoned record discloses them,
    # tallies included, and a resume deletes the file.
    (abandoned,) = result["shards_abandoned"]
    assert abandoned["rows"] == 2
    assert abandoned["uncommitted_rows"] == 2
    assert sum(abandoned["tallies"]["history_plies"].values()) == 2
    assert sum(abandoned["tallies"]["history_root_reason"].values()) == 2
    # ... and the counters the game loop kept say two rows were HANDED OVER,
    # which is exactly the number the inventory is allowed to disagree with.
    assert result["dedup"]["rows_banked"] == 2


def test_a_committed_shards_tallies_are_the_rows_it_holds(tmp_path: Path) -> None:
    """The per-shard tally is committed WITH the shard, on its progress line."""
    writer = corpus.ShardWriter(
        out_dir=tmp_path, worker_id=0, shard_rows=2, tally_keys=("history_plies",),
    )
    for game, plies in ((0, (3, 7)), (1, (7,))):
        for n in plies:
            writer.write({"game_id": game, "history_plies": n})
        writer.end_game(game)
    writer.close()

    assert [s["tallies"] for s in writer.shards] == [
        {"history_plies": {"3": 1, "7": 1}}, {"history_plies": {"7": 1}},
    ]
    assert [line["tallies"] for line in read_progress(tmp_path, 0)] == [
        s["tallies"] for s in writer.shards
    ]
    assert corpus.merge_shard_tallies(writer.shards, "history_plies") == {"3": 1, "7": 2}
    # A row missing a tallied key is a writer fault, never a silent zero.
    (tmp_path / "other").mkdir()
    with pytest.raises(KeyError):
        corpus.ShardWriter(
            out_dir=tmp_path / "other", worker_id=1, shard_rows=2,
            tally_keys=("history_plies",),
        ).write({"game_id": 0})


# ── run assembly ─────────────────────────────────────────────────────────────


def test_games_are_dealt_so_no_two_workers_share_a_game_id() -> None:
    buckets = corpus.split_games(7, 3)
    assert buckets == [[0, 3, 6], [1, 4], [2, 5]]
    assert corpus.split_games(2, 5) == [[0], [1]], "no empty workers"


def test_a_nonpositive_worker_count_is_refused_not_clamped() -> None:
    """A silent ``max(1, workers)`` runs one worker while the requested stamp
    says 0 -- the accepted-then-ignored shape, caught by grok review."""
    for workers in (0, -1):
        with pytest.raises(ValueError, match="--workers must be >= 1"):
            corpus.split_games(3, workers)


def test_a_run_with_no_stated_game_count_is_refused() -> None:
    with pytest.raises(ValueError, match="--games must be positive"):
        corpus.split_games(0, 1)


def test_the_realized_stamp_is_read_off_the_live_engine_not_the_flags() -> None:
    """Every field comes from the object that talked to Stockfish.

    Mutation caught: echoing the requested knob back as "realized".  The double
    below is built with a DIFFERENT hash size from the one a caller would ask
    for, so an echo would report the ask and this test would see it.
    """
    engine = ScriptedEngine()
    searcher = searcher_for(engine, hash_mb=7, syzygy_path="/tb/a:/tb/b", nice=11)

    realized = searcher.realized()
    assert realized["sf_hash_mb"] == 7
    assert realized["sf_syzygy_path"] == "/tb/a:/tb/b"
    assert realized["sf_nice"] == 11
    assert realized["sf_threads"] == 1
    assert realized["staircase"] == corpus.DEFAULT_STAIRCASE
    assert realized[corpus.KEY_TT_CARRIED] is True
    assert realized["ucinewgame_calls"] == 0
    # `tt_cleared_per_game` is deliberately NOT here: it needs the games count,
    # so the worker stamps it -- see the skipped-clear test below.
    assert "tt_cleared_per_game" not in realized
    assert realized["cp_slope"] == gen.NNUE_CP_SLOPE


def test_a_mid_position_clear_voids_the_carried_tt_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tt_carried_across_phases`` is an observation that can FAIL.

    Mutation caught: hardcoding the stamp back to ``True`` -- the wrapped
    stream below clears the table mid-staircase, which the counter must see.
    """
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    real_stream = searcher.stream

    def clearing_stream(history: corpus.RowHistory, **kw: Any) -> list[str]:
        searcher.new_game()
        return real_stream(history, **kw)

    monkeypatch.setattr(searcher, "stream", clearing_stream)
    searcher.search_position(chess.Board())
    assert searcher.tt_cleared_mid_position == 1
    assert searcher.realized()[corpus.KEY_TT_CARRIED] is False


def test_width_streamed_reports_the_stream_not_the_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An under-delivering engine shows up as streamed < realized.

    Mutation caught: stamping the requested width as the streamed one.  Rank 3
    is filtered out of every iteration, so no depth ever carries the asked-for
    3 ranks and the consumed block is 2 wide.
    """
    engine = ScriptedEngine()
    searcher = searcher_for(engine, staircase="3:5")
    real_stream = searcher.stream

    def dropping_stream(history: corpus.RowHistory, **kw: Any) -> list[str]:
        return [
            line for line in real_stream(history, **kw)
            if " multipv 3 " not in f" {line} "
        ]

    monkeypatch.setattr(searcher, "stream", dropping_stream)
    search = searcher.search_position(chess.Board())
    phase = search.phases[0]
    assert phase.width_realized == 3
    assert phase.width_streamed == 2
    assert phase.as_row()["width_streamed"] == 2
    assert all(not b.complete for b in phase.parse.blocks)


def test_a_gapped_rank_set_is_not_complete() -> None:
    """Ranks {1, 2, 4} against W=3: right COUNT, missing line.

    The emissions counter cannot see this either (three lines arrived), so
    cardinality-as-completeness would pass it -- grok review's repro.
    """
    parse = corpus.parse_depth_blocks(
        [
            info(3, 1, 30, "e2e4", 100),
            info(3, 2, 20, "d2d4", 100),
            info(3, 4, 10, "g1f3", 100),
        ],
        expected_lines=3,
    )
    (block,) = parse.blocks
    assert len(block.lines) == 3
    assert block.complete is False


def test_the_default_syzygy_path_is_the_production_pair() -> None:
    """Pinned to ``configs/pbt2_small.yaml`` without embedding an absolute path.

    The production pair is two directory NAMES under a checkout root; spelling
    the names in the module and resolving the root at runtime is what keeps a
    username out of a public repo while keeping the default correct.
    """
    config = corpus.REPO_ROOT / "configs" / "pbt2_small.yaml"
    text = config.read_text(encoding="utf-8")
    line = next(
        ln for ln in text.splitlines() if ln.strip().startswith("syzygy_path:")
    )
    configured = [Path(p) for p in line.split(":", 1)[1].strip().split(":")]
    assert tuple(p.name for p in configured) == corpus.SYZYGY_DIR_NAMES
    assert {p.parent for p in configured} == {configured[0].parent}

    default = corpus.default_syzygy_path().split(":")
    assert tuple(Path(p).name for p in default) == corpus.SYZYGY_DIR_NAMES


@pytest.mark.skipif(
    SMOKE_SYZYGY is None, reason="the 3-4 man smoke tablebase is absent",
)
def test_every_syzygy_component_must_open_not_just_one() -> None:
    """⚑⚑ A HALF-OPEN PAIR IS THE FAILURE THIS CHECK EXISTS FOR.

    ``get_tablebase`` returns a handle when AT LEAST ONE listed directory
    opened, so the old ``get_tablebase(pair) is None`` test passed on
    ``<real dir>:/typo/syzygy_6`` -- and production's path IS a pair whose
    second half holds the 6-man DTZ.  The burn would then run to completion with
    every 6-man probe silently answering ``None``: no adjudication, no result on
    the rows, and nothing in any log.  Both orders are checked because "the
    first component works" is exactly the state that made the old check pass.

    Mutation caught: the previous whole-path ``get_tablebase(...) is None``.
    """
    assert SMOKE_SYZYGY is not None
    live = str(SMOKE_SYZYGY)
    dead = "/nonexistent/syzygy_6"

    assert corpus.refuse_unopenable_syzygy(live) == (live,)
    assert corpus.refuse_unopenable_syzygy(f"{live}{os.pathsep}{live}") == (live, live)

    with pytest.raises(ValueError, match=dead):
        corpus.refuse_unopenable_syzygy(f"{live}{os.pathsep}{dead}")
    with pytest.raises(ValueError, match=dead):
        corpus.refuse_unopenable_syzygy(f"{dead}{os.pathsep}{live}")
    with pytest.raises(ValueError, match="names no directory"):
        corpus.refuse_unopenable_syzygy("")


@pytest.mark.skipif(
    PRODUCTION_SYZYGY is None, reason="the production 6-man pair is absent",
)
def test_the_production_pair_opens_on_both_halves() -> None:
    assert PRODUCTION_SYZYGY is not None
    components = corpus.refuse_unopenable_syzygy(PRODUCTION_SYZYGY)
    assert len(components) == 2


@pytest.mark.skipif(
    SMOKE_SYZYGY is None, reason="the 3-4 man smoke tablebase is absent",
)
def test_a_run_refuses_a_half_open_syzygy_pair_before_it_writes_anything(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    args = corpus.build_parser().parse_args([
        "--out-dir", str(out_dir), "--games", "1",
        "--syzygy-path", f"{SMOKE_SYZYGY}{os.pathsep}/nonexistent/syzygy_6",
    ])
    with pytest.raises(ValueError, match="Every component must open"):
        corpus.run(args)
    assert not out_dir.exists(), "the refusal lands before the corpus directory"


def test_the_config_stamp_hash_moves_with_every_knob() -> None:
    parser = corpus.build_parser()
    base = parser.parse_args(["--out-dir", "/tmp/x", "--games", "1"])
    stamp = corpus.config_stamp(base, sf_binary="/bin/sf")
    changed = parser.parse_args(
        ["--out-dir", "/tmp/x", "--games", "1", "--temp-low", "0.9"],
    )
    assert corpus.stamp_sha256(stamp) != corpus.stamp_sha256(
        corpus.config_stamp(changed, sf_binary="/bin/sf"),
    )
    assert corpus.stamp_sha256(stamp) == corpus.stamp_sha256(dict(stamp))
    g10 = parser.parse_args([
        "--out-dir", "/tmp/x", "--games", "1",
        "--staircase", corpus.G10_STAIRCASE,
        "--staircase-policy", corpus.STAIRCASE_POLICY_G10,
    ])
    g10_stamp = corpus.config_stamp(g10, sf_binary="/bin/sf")
    assert g10_stamp["staircase_policy"] == corpus.STAIRCASE_POLICY_G10
    assert corpus.stamp_sha256(stamp) != corpus.stamp_sha256(g10_stamp)


def test_g10_with_the_wrong_staircase_is_refused_before_output_exists(
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"
    args = corpus.build_parser().parse_args([
        "--out-dir", str(out), "--games", "1",
        "--staircase-policy", corpus.STAIRCASE_POLICY_G10,
    ])
    with pytest.raises(ValueError, match="validated only"):
        corpus.run(args)
    assert not out.exists()


def test_the_summary_merges_workers_without_losing_a_counter() -> None:
    workers = [
        {
            "worker_id": 0, "failed": None, "games": 2, "rows": 5,
            "plies_total": 40,
            "terminations": {"natural": 2}, "adjudications": {"none": 2},
            "opening_sources": {"start": 2}, "adjudication_unavailable_plies": 1,
            "history_plies_histogram": {"7": 4, "11": 1},
            "history_root_reasons": {"irreversible": 4, "game_start": 1},
            "history_plies_histogram_prior": {},
            "history_root_reasons_prior": {},
            "history_tallies_unknown_rows_prior": 0,
            "dedup": {
                "positions_first_seen": 8, "dup_hits": 2,
                "first_seen_by_phase": {"opening": 8, "middlegame": 0, "endgame": 0},
                "dup_hits_by_phase": {"opening": 2, "middlegame": 0, "endgame": 0},
                "dedup_cache_entries": 8, "dedup_cache_evictions": 3,
                "dedup_cache_max_entries": 5, "dedup_cache_bytes_est": 8000,
                "row_key_hits": 3, "search_key_hits": 3,
                "search_key_hit_on_new_input": 1,
                "search_key_miss_on_seen_input": 1, "searches": 8,
                "rows_banked": 5, "dedup_input_set_entries": 7,
                "dedup_input_set_evictions": 2,
            },
            "search": {
                "positions_searched": 8, "searches": 24, "search_s": 4.0,
                "anomalies": {"re_emissions": 1, "duplicate_iteration_flushes": 2},
                "staircase_gate": {
                    "evaluations": 0, "extended": 0, "stopped": 0,
                    "forced_stops": 0,
                },
                "nodes_by_phase": {
                    "0": {"n": 8, "total": 80, "min": 5, "max": 15,
                          "median_est_reservoir": 9.5,
                          "log2_buckets": {"3": 8}},
                },
            },
            "shards": [{"path": "a", "rows": 5, "codec": "zstd", "games": [0, 2]}],
            "shards_prior": [], "shards_abandoned": [], "games_completed_prior": 0,
            "dedup_rewarmed": 0, "dedup_cache_events_rewarmed": 0,
            "dedup_rewarmed_resident": 0, "resumed": False,
            "resume_partials_deleted": [], "resume_progress_torn_tail": False,
            "resume_progress_repair": corpus.PROGRESS_ABSENT,
            "resume_legacy_progress_lines": 0,
            "realized": {"sf_hash_mb": 64},
        },
        {
            "worker_id": 1,
            "failed": {
                "exception_type": "RuntimeError", "exception": "engine died",
                "last_game_id": 4, "last_ply": 17, "games_completed": 1,
            },
            "games": 1, "rows": 3, "plies_total": 20,
            "terminations": {"syzygy": 1}, "adjudications": {"syzygy_wdl_2": 1},
            "opening_sources": {"start": 1}, "adjudication_unavailable_plies": 0,
            "history_plies_histogram": {"7": 3},
            "history_root_reasons": {"irreversible": 3},
            # The adopted prior shard's 4 rows: 3 with tallies, 1 whose record
            # predates them (reported as unknown, never as a bucket).
            "history_plies_histogram_prior": {"9": 3},
            "history_root_reasons_prior": {"game_start": 3},
            "history_tallies_unknown_rows_prior": 1,
            "dedup": {
                "positions_first_seen": 4, "dup_hits": 0,
                "first_seen_by_phase": {"opening": 4, "middlegame": 0, "endgame": 0},
                "dup_hits_by_phase": {"opening": 0, "middlegame": 0, "endgame": 0},
                "dedup_cache_entries": 4, "dedup_cache_evictions": 0,
                "dedup_cache_max_entries": 5, "dedup_cache_bytes_est": 4400,
                "row_key_hits": 0, "search_key_hits": 1,
                "search_key_hit_on_new_input": 1,
                "search_key_miss_on_seen_input": 0, "searches": 4,
                "rows_banked": 3, "dedup_input_set_entries": 4,
                "dedup_input_set_evictions": 0,
            },
            "search": {
                "positions_searched": 4, "searches": 12, "search_s": 2.0,
                "anomalies": {"re_emissions": 0, "bound_lines": 3},
                "staircase_gate": {
                    "evaluations": 0, "extended": 0, "stopped": 0,
                    "forced_stops": 0,
                },
                "nodes_by_phase": {
                    "0": {"n": 4, "total": 20, "min": 2, "max": 9,
                          "median_est_reservoir": 4.0,
                          "log2_buckets": {"3": 3, "4": 1}},
                },
            },
            "shards": [{"path": "b", "rows": 3, "codec": "zstd", "games": [1]}],
            # ⚑ A RESUMED worker: its prior shard's rows and games are part of
            # the corpus and must reach the merged totals, or `summary.json`
            # would be a complete-looking document indexing half a corpus.
            "shards_prior": [
                {"path": "b-prior", "rows": 4, "codec": "zstd", "games": [3, 5]},
            ],
            "shards_abandoned": [
                {"path": "b-lost", "rows": 2, "codec": "zstd", "games": [],
                 "uncommitted_rows": 2},
            ],
            "games_completed_prior": 2,
            "dedup_rewarmed": 4, "dedup_cache_events_rewarmed": 3,
            "dedup_rewarmed_resident": 4, "resumed": True,
            "resume_partials_deleted": ["w01-00001.jsonl.zst"],
            "resume_progress_torn_tail": True,
            "resume_progress_repair": corpus.PROGRESS_TRUNCATED,
            "resume_legacy_progress_lines": 1,
            "realized": {"sf_hash_mb": 64},
        },
    ]
    summary = corpus.build_summary(
        results=workers,
        requested={"run_id": "t"},
        config_sha="abc",
        engine_record={"path": "/bin/sf", "sha256": "d" * 64},
        engine_id_name="Stockfish test",
        staircase=corpus.parse_staircase(corpus.DEFAULT_STAIRCASE),
        started_utc="2026-08-27T00:00:00+00:00",
        wall_s=6.0,
    )

    # ⚑ CORPUS totals: this session's 8 rows / 3 games PLUS worker 1's adopted
    # 4 rows / 2 games. The session-scoped numbers keep their own names.
    assert summary["rows"] == 12
    assert summary["games"] == 5
    assert summary["rows_this_session"] == 8
    assert summary["games_this_session"] == 3
    assert summary["rows_prior"] == 4
    assert summary["games_completed_prior"] == 2
    assert summary["games_completed_prior_by_worker"] == {"0": 0, "1": 2}
    assert summary["resumed"] is True
    assert summary["dedup_rewarmed"] == 4
    # ⚑ ITS OWN COLUMN (Grok round 5): the worker computed it and the summary
    # dropped it. Mutant: dropped again -> KeyError here.
    assert summary["dedup_cache_events_rewarmed"] == 3
    assert summary["dedup_cache_events_rewarmed_by_worker"] == {"0": 0, "1": 3}
    assert "3 cache-only events re-warmed" in corpus.format_summary(summary)
    assert summary["resume_partials_deleted"] == {"1": ["w01-00001.jsonl.zst"]}
    assert summary["resume_progress_torn_tail_workers"] == [1]
    assert summary["resume_progress_repaired"] == {"1": corpus.PROGRESS_TRUNCATED}
    assert summary["resume_legacy_progress_lines"] == 1
    # ⚑ Rows this run wrote and then dropped, because their shard also held a
    # game that never ended. Reported, never absorbed into the row count.
    assert list(summary["shards_abandoned"]) == ["1"]
    # Wall clock is this session's, so the rate it divides must be too.
    assert summary["s_per_row"] == pytest.approx(6.0 / 8)
    # Prior first, then this session's: the order they were banked in.
    assert [shard["path"] for shard in summary["shards"]] == ["a", "b-prior", "b"]
    assert summary["terminations"] == {"natural": 2, "syzygy": 1}
    # The window histograms merge like every other per-worker counter; a
    # merge rule that was never added would read {} here rather than 0s.
    assert summary["history_plies_histogram_this_session"] == {"7": 7, "11": 1}
    assert summary["history_root_reasons_this_session"] == {
        "irreversible": 7, "game_start": 1,
    }
    # ⚑ CORPUS level: prior + session, and the invariant the reader holds it
    # to -- sum(histogram) + unknown == rows (12 = 8 session + 4 prior).
    assert summary["history_plies_histogram"] == {"7": 7, "11": 1, "9": 3}
    assert summary["history_root_reasons"] == {
        "irreversible": 7, "game_start": 4,
    }
    assert summary["history_tallies_unknown_rows"] == 1
    assert (
        sum(summary["history_plies_histogram"].values())
        + summary["history_tallies_unknown_rows"]
    ) == summary["rows"]
    assert summary["adjudication_unavailable_plies"] == 1
    assert summary["dedup"]["dup_hits"] == 2
    assert summary["dedup"]["positions_visited"] == 14
    # The two-key counters merge as sums, every one of them: a counter the
    # merge forgot would be absent here, not 0.
    assert {name: summary["dedup"][name] for name in corpus._DEDUP_SCALAR_COUNTERS} == {
        "row_key_hits": 3, "search_key_hits": 4,
        "search_key_hit_on_new_input": 2, "search_key_miss_on_seen_input": 1,
        "searches": 12, "rows_banked": 8, "dedup_input_set_entries": 11,
        "dedup_input_set_evictions": 2,
    }
    # The bound's counters merge as sums; the bound itself is per worker.
    assert summary["dedup"]["dedup_cache_evictions"] == 3
    assert summary["dedup"]["dedup_cache_entries"] == 12
    assert summary["dedup"]["dedup_cache_max_entries_per_worker"] == 5
    assert summary["dedup"]["dedup_cache_bytes_per_entry_est"] == pytest.approx(
        12400 / 12,
    )
    # ⚑ One worker died. The merged rows/games above are still ITS SURVIVORS'
    # numbers, which is exactly why the failure has to be at the top level.
    assert summary["failed_workers"] == [
        {
            "worker_id": 1, "exception_type": "RuntimeError",
            "exception": "engine died", "last_game_id": 4, "last_ply": 17,
            "games_completed": 1,
        },
    ]
    assert summary["search"]["anomalies"] == {
        "re_emissions": 1, "bound_lines": 3, "duplicate_iteration_flushes": 2,
    }
    assert summary["search"]["s_per_position"] == pytest.approx(0.5)
    assert summary["search"]["staircase_gate"] == {
        "evaluations": 0, "extended": 0, "stopped": 0, "forced_stops": 0,
    }
    assert "staircase policy=fixed" in corpus.format_summary(summary)
    adaptive = {
        **summary,
        "staircase_gate": corpus.staircase_gate_stamp(corpus.STAIRCASE_POLICY_G10),
        "search": {
            **summary["search"],
            "staircase_gate": {
                "evaluations": 12,
                "extended": 7,
                "stopped": 5,
                "forced_stops": 1,
            },
        },
    }
    assert "staircase policy=g10 extended=7/12 stopped=5 forced_stops=1" in (
        corpus.format_summary(adaptive)
    )
    nodes = summary["search"]["nodes_by_phase"]["0"]
    assert (nodes["n"], nodes["total"], nodes["min"], nodes["max"]) == (12, 100, 2, 15)
    assert nodes["log2_buckets"] == {"3": 11, "4": 1}
    # ⚑ The median estimate REACHES the merged summary -- per worker, because
    # pooling equal-size reservoirs over unequal n would invent a population.
    assert nodes["median_est_reservoir_by_worker"] == {"0": 9.5, "1": 4.0}
    assert "median_est_reservoir" not in nodes, "no pooled median is claimed"
    assert summary["engine"]["sha256"] == "d" * 64
    assert summary["engine"]["id_name"] == "Stockfish test"
    assert set(summary["config_realized_by_worker"]) == {"0", "1"}
    assert summary["staircase_parsed"] == [
        {"width": "all", "depth": 9},
        {"width": "16", "depth": 11},
        {"width": "4", "depth": 13},
    ]
    assert json.loads(json.dumps(summary, default=corpus._json_default))


def test_a_worker_writes_shards_and_reports_its_own_realized_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_worker`` end to end with the engine constructor stubbed out."""
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0, 1)))

    assert result["games"] == 2
    assert result["failed"] is None
    # Game 1 replays game 0's positions, so every one of them is cache-served.
    assert result["rows"] == 2
    assert result["dedup"]["dup_hits"] == 2
    assert result["dedup"]["dedup_cache_entries"] == 2
    assert result["dedup"]["dedup_cache_evictions"] == 0
    assert result["dedup"]["dedup_cache_bytes_per_entry_est"] > 0.0
    assert result["terminations"] == {"natural": 2}
    assert result["realized"]["sf_hash_mb"] == 64
    # Observed off the counter, not echoed: two games started, two clears.
    assert result["realized"]["tt_cleared_per_game"] is True
    assert result["realized"]["ucinewgame_calls"] == 2
    assert result["realized"]["dedup_cache_max"] == corpus.DEFAULT_DEDUP_CACHE_MAX
    # The incremental inventory: one progress line per CLOSED shard, appended
    # as it closed, byte-agreeing with what the summary will aggregate — the
    # record a crashed run still has. Mutation caught: deleting the append.
    progress = [
        json.loads(line)
        for line in (out_dir / "w00.progress.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert progress == result["shards"]
    assert result["realized"]["max_plies"] == 50
    assert result["realized"]["seed"] == 7
    assert result["realized"]["opening_book_path"] is None

    rows = [
        row for shard in result["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert len(rows) == 2
    assert [row["result"] for row in rows] == [-1.0, 1.0]


def test_the_g10_policy_reaches_the_worker_and_the_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus,
        "StockfishUCI",
        lambda *_args, **_kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus,
        "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(
        out_dir,
        staircase=corpus.G10_STAIRCASE,
        staircase_policy=corpus.STAIRCASE_POLICY_G10,
    ))

    assert result["failed"] is None
    assert result["realized"]["staircase"] == corpus.G10_STAIRCASE
    assert result["realized"]["staircase_policy"] == corpus.STAIRCASE_POLICY_G10
    assert result["realized"]["staircase_gate"]["threshold_cp"] == 10.0
    assert result["search"]["staircase_gate"] == {
        "evaluations": 2,
        "extended": 0,
        "stopped": 2,
        "forced_stops": 0,
    }
    rows = [
        row for shard in result["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert len(rows) == 2
    assert all(row["staircase_gate"]["extended"] is False for row in rows)
    assert all(len(row["phases"]) == 2 for row in rows)


def test_the_cli_run_banks_and_reports_the_g10_policy_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus,
        "StockfishUCI",
        lambda *_args, **_kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus,
        "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(
        corpus,
        "refuse_unopenable_syzygy",
        lambda _path: ("scripted",),
    )
    monkeypatch.setattr(
        corpus.audit_targets,
        "engine_identity",
        lambda _path: "ScriptedEngine",
    )
    out_dir = tmp_path / "run"
    args = corpus.build_parser().parse_args([
        "--out-dir", str(out_dir),
        "--games", "1",
        "--workers", "1",
        "--stockfish", "/bin/true",
        "--syzygy-path", "/nonexistent",
        "--staircase", corpus.G10_STAIRCASE,
        "--staircase-policy", corpus.STAIRCASE_POLICY_G10,
        "--max-plies", "2",
        "--nice", "0",
    ])

    summary = corpus.run(args)
    manifest = json.loads(
        (out_dir / corpus.MANIFEST_NAME).read_text(encoding="utf-8"),
    )
    rows = [
        row for shard in summary["shards"] for row in read_shard(Path(shard["path"]))
    ]

    assert summary["config_requested"]["staircase_policy"] == "g10"
    assert summary["staircase_gate"] == corpus.staircase_gate_stamp("g10")
    assert manifest["staircase_gate"] == summary["staircase_gate"]
    assert summary["config_realized_by_worker"]["0"]["staircase_policy"] == "g10"
    assert summary["search"]["staircase_gate"]["evaluations"] == len(rows) == 2
    assert all(row["staircase_gate"]["policy"] == "g10" for row in rows)
    assert all(row["staircase_gate"]["decision_depth_observed"] == 10 for row in rows)
    assert all(len(row["phases"]) == 2 for row in rows)


def test_a_wedged_engine_is_replaced_and_the_position_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEASURED failure mode: dev Stockfish wedges at 100% CPU with a hot TT
    (`stop` ignored), and the same command finishes in seconds on a cold one.
    One wedge ⇒ fresh engine, cold-TT retry of the SAME position, the game and
    the worker live on, and the retried row discloses its cold table.

    Mutation caught: deleting the retry from ``EngineLease.search_position``
    — the game is then abandoned as engine_wedge and every assert here moves.
    """
    first = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    first.wedge_on_go = 4  # ply 0's three rungs pass; ply 1's first go wedges
    second = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    engines = [first, second]
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engines.pop(0)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))

    assert result["failed"] is None
    assert result["games"] == 1
    assert result["terminations"] == {"natural": 1}
    assert result["search"]["engine_respawns"] == 1
    assert result["realized"]["tt_cleared_per_game"] is True
    assert not engines, "the lease actually spawned the replacement"
    rows = [
        row for shard in result["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert sum(1 for r in rows if r.get("cold_tt_retry")) == 1


def test_a_double_wedge_abandons_the_game_and_the_worker_plays_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry wedging TOO ends the GAME (engine_wedge, no result), never
    the worker: rows banked before the wedge keep their labels, and the next
    game plays on the surviving replacement engine."""
    first = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    first.wedge_on_go = 4
    second = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    second.wedge_on_go = 1  # the retry's own first go wedges as well
    # A double-wedged lease replaces the engine AGAIN before re-raising (a
    # desynced process would be refused on its next use), so game 1 plays on
    # a third, clean engine.
    third = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    engines = [first, second, third]
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engines.pop(0)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0, 1)))

    assert result["failed"] is None
    assert result["games"] == 2
    assert result["terminations"] == {"engine_wedge": 1, "natural": 1}
    assert result["search"]["engine_respawns"] == 2
    assert not engines, "the post-double-wedge replacement was actually spawned"
    rows = [
        row for shard in result["shards"] for row in read_shard(Path(shard["path"]))
    ]
    # Game 0's pre-wedge row is banked RESULTLESS (same shape as a ply cap);
    # game 1 finished on the surviving engine and its row carries the result.
    by_game: dict[int, list[Any]] = {}
    for r in rows:
        by_game.setdefault(r["game_id"], []).append(r)
    assert all(r["result"] is None for r in by_game[0])
    assert all(r["result"] is not None for r in by_game[1])


def test_a_skipped_per_game_clear_fails_the_realized_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting ``new_game`` from the game loop must show in the summary.

    Mutation caught: stamping ``tt_cleared_per_game: True`` as a constant --
    grok review found exactly that, a stamp that could not fail.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(
        corpus.StaircaseSearcher, "new_game", lambda self: None,
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))

    assert result["failed"] is None
    assert result["realized"]["tt_cleared_per_game"] is False
    assert result["realized"]["ucinewgame_calls"] == 0
    assert "ucinewgame" not in engine.commands


@pytest.mark.skipif(
    SMOKE_SYZYGY is None, reason="the 3-4 man smoke tablebase is absent",
)
def test_a_whole_run_writes_a_summary_and_refuses_a_second_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run`` end to end: shards, ``summary.json``, and the rerun refusal."""
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "run"
    argv = [
        "--out-dir", str(out_dir), "--games", "1", "--workers", "1",
        "--syzygy-path", str(SMOKE_SYZYGY), "--temp-high", "0.01",
        "--temp-low", "0.01", "--nice", "0",
    ]
    args = corpus.build_parser().parse_args(argv)

    summary = corpus.run(args)

    assert summary["games"] == 1
    assert summary["rows"] == 2
    assert summary["schema"] == corpus.SUMMARY_SCHEMA
    assert summary["failed_workers"] == []
    assert summary["config_requested"]["games"] == 1
    assert summary["config_realized_by_worker"]["0"][corpus.KEY_TT_CARRIED] is True
    assert summary["banked_rows_min_piece_count"] == corpus.MIN_BANKED_PIECES
    # The launch manifest: written BEFORE the first game so a crashed run's
    # rows keep their cp map, staircase and config sha. `complete: false`
    # forever — summary.json is the only completion record.
    manifest = json.loads(
        (out_dir / corpus.MANIFEST_NAME).read_text(encoding="utf-8"),
    )
    assert manifest["complete"] is False
    assert manifest["config_sha256"] == summary["config_sha256"]
    assert manifest["config_requested"] == summary["config_requested"]
    assert manifest["staircase_parsed"] == summary["staircase_parsed"]
    assert manifest["staircase_gate"] == summary["staircase_gate"] == {
        "policy": corpus.STAIRCASE_POLICY_FIXED,
        "adaptive": False,
    }
    on_disk = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["config_sha256"] == summary["config_sha256"]
    rows = [
        row for shard in summary["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert [row["run"]["config_sha256"] for row in rows] == [
        summary["config_sha256"],
    ] * 2

    with pytest.raises(FileExistsError, match="already holds files"):
        corpus.run(corpus.build_parser().parse_args(argv))


def test_the_read_timeout_flag_reaches_the_engine_it_configures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Take-effect, not acceptance: the value is read BACK off the engine.

    Mutation caught: accepting ``--sf-read-timeout`` and never passing it to
    ``StockfishUCI`` -- the flag then parses, stamps itself into
    ``config_requested``, and the engine quietly keeps the driver's 60 s
    default.  This repo's signature defect, and a stamp echoed from the args
    could not see it, which is why ``realized`` reads
    ``engine.read_timeout_s``.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    constructed: dict[str, Any] = {}

    def build(*_args: Any, **kwargs: Any) -> StockfishUCI:
        constructed.update(kwargs)
        # The double is built FROM what run_worker passed, so a plumbing that
        # never happened cannot be papered over here either.
        return uci_double(engine, read_timeout_s=float(kwargs["read_timeout_s"]))

    monkeypatch.setattr(corpus, "StockfishUCI", build)
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, sf_read_timeout_s=12.5))

    assert constructed["read_timeout_s"] == 12.5
    assert result["realized"]["sf_read_timeout_s"] == 12.5


def test_the_read_timeout_default_is_stated_and_a_bad_one_is_refused() -> None:
    args = corpus.build_parser().parse_args(["--out-dir", "/tmp/x", "--games", "1"])
    assert args.sf_read_timeout == corpus.DEFAULT_SF_READ_TIMEOUT_S
    assert corpus.config_stamp(args, sf_binary="/bin/sf")["sf_read_timeout_s"] == (
        corpus.DEFAULT_SF_READ_TIMEOUT_S
    )
    for bad in ("0", "-1", "nan"):
        with pytest.raises(ValueError, match="finite and positive"):
            corpus.run(corpus.build_parser().parse_args(
                ["--out-dir", "/tmp/x", "--games", "1", "--sf-read-timeout", bad],
            ))


def test_the_search_timeout_flag_reaches_the_searcher_and_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Take-effect, not acceptance: read BACK off the searcher that used it.

    Mutation caught: accepting ``--sf-search-timeout`` into the spec and never
    passing it to ``StaircaseSearcher`` -- the flag then parses, stamps itself
    into ``config_requested``, and every search quietly keeps the 8 s default.
    ``realized`` reads ``searcher.search_timeout_s``, the very attribute
    ``stream`` computes its deadline from, so the stamp cannot echo the args.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *_a, **_kw: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, sf_search_timeout_s=3.25))

    assert result["realized"]["sf_search_timeout_s"] == 3.25


def test_the_search_timeout_default_is_stated_and_a_bad_one_is_refused() -> None:
    args = corpus.build_parser().parse_args(["--out-dir", "/tmp/x", "--games", "1"])
    assert args.sf_search_timeout == corpus.DEFAULT_SF_SEARCH_TIMEOUT_S
    assert corpus.config_stamp(args, sf_binary="/bin/sf")["sf_search_timeout_s"] == (
        corpus.DEFAULT_SF_SEARCH_TIMEOUT_S
    )
    for bad in ("0", "-1", "nan"):
        with pytest.raises(ValueError, match="finite and positive"):
            corpus.run(corpus.build_parser().parse_args(
                ["--out-dir", "/tmp/x", "--games", "1", "--sf-search-timeout", bad],
            ))
    # A tripwire looser than the outer deadline is a stamp naming a bound the
    # engine never enforces -- refused, not silently reordered.
    with pytest.raises(ValueError, match="exceeds"):
        corpus.run(corpus.build_parser().parse_args(
            ["--out-dir", "/tmp/x", "--games", "1",
             "--sf-search-timeout", "20", "--sf-read-timeout", "10"],
        ))


def test_the_search_deadline_bounds_the_go_and_the_handshake_keeps_its_own() -> None:
    """⚑ Take-effect on the wire: the deadline handed to the reader during a
    search is the tripwire, while an option handshake keeps the engine-wide
    ``read_timeout_s``.

    Mutation caught: ``stream`` computing its deadline from
    ``engine.read_timeout_s`` -- the tripwire then silently reverts to the
    handshake bound, and an exploded search wedges the worker for the outer
    deadline, exactly the wait the tripwire exists to cut short (ledger
    AMENDMENT 4).
    """
    engine = ScriptedEngine(multipv=1)
    searcher = corpus.StaircaseSearcher(
        engine=uci_double(engine, read_timeout_s=500.0),
        staircase=corpus.parse_staircase("all:2"),
        cp_slope=gen.NNUE_CP_SLOPE,
        cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        search_timeout_s=2.0,
    )
    remaining: list[float] = []

    def recording_readline(deadline: float) -> str:
        remaining.append(deadline - time.monotonic())
        return engine.readline(deadline)

    searcher.engine._readline_with_deadline = recording_readline

    # multipv=4 differs from the engine's current 1, so the stream opens with
    # a setoption/isready handshake before the go.
    searcher.stream(corpus.history_for(chess.Board()), depth=2, multipv=4)

    handshake, search_reads = remaining[0], remaining[1:]
    assert handshake > 100.0, "the readyok wait must run on read_timeout_s"
    assert search_reads, "the go must stream through the recorded reader"
    assert all(0.0 < d <= 2.0 for d in search_reads), (
        f"every search read must be bounded by the 2 s tripwire, got "
        f"{search_reads!r}"
    )


def test_a_worker_that_dies_mid_game_records_its_slot_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead worker returns a FAILED slot with everything it earned.

    Mutation caught: letting the exception out of ``run_worker``.  With a pool
    that is one dead engine taking the whole run's ``summary.json`` -- every
    other worker's searches included -- and with a single worker it is a
    traceback where a partial corpus with a stamp should have been.
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT, raise_on_go=4)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engine),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    result = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))

    failed = result["failed"]
    assert failed is not None
    assert failed["exception_type"] == "RuntimeError"
    assert failed["exception"] == SCRIPTED_ENGINE_DEATH
    # Ply 0's three rungs ran; the fourth `go` is ply 1's first.
    assert (failed["last_game_id"], failed["last_ply"]) == (0, 1)
    assert failed["games_completed"] == 0

    # ... and the slot is still a slot: every counter the merge reads is there.
    assert result["games"] == 0
    assert result["search"]["positions_searched"] == 1, "ply 0 finished"
    # 1, not 2: first_seen counts AFTER the search since the wedge-recovery
    # change — ply 1's search died before producing values, so that position
    # was never actually seen into the cache.
    assert result["dedup"]["positions_first_seen"] == 1
    assert result["shards"] == []


def test_a_dead_worker_does_not_take_the_other_workers_summary_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two workers, one dies: the survivor's rows and the summary both land.

    ⚑ The pool is replaced by an INLINE executor rather than skipped: ``run``'s
    multi-worker branch is the code under test (submit / ``as_completed`` /
    per-future result), and a spawn pool cannot carry a monkeypatched module
    into its children, so this is the only way to drive that branch against a
    scripted engine.
    """
    healthy = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    dying = ScriptedEngine(preferred=MATE_GAME_SCRIPT, raise_on_go=4)
    engines = [healthy, dying]  # popped in submission order: worker 0, then 1
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *args, **kwargs: uci_double(engines.pop(0)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", InlineExecutor)
    out_dir = tmp_path / "run"

    code = corpus.main([
        "--out-dir", str(out_dir), "--games", "2", "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
    ])

    assert code == 1, "a run that lost a worker did not do what it was asked"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [f["worker_id"] for f in summary["failed_workers"]] == [1]
    assert summary["failed_workers"][0]["exception"] == SCRIPTED_ENGINE_DEATH
    assert summary["failed_workers"][0]["last_ply"] == 1
    # The survivor's game is complete and banked.
    assert summary["games"] == 1
    assert summary["rows"] == 2
    rows = [
        row for shard in summary["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert [row["worker_id"] for row in rows] == [0, 0]
    assert "FAILED worker 1" in corpus.format_summary(summary)


def test_a_worker_whose_process_dies_still_gets_a_mergeable_slot(
    tmp_path: Path,
) -> None:
    """The other half: an OOM kill leaves nothing for ``run_worker`` to record.

    The synthesised slot has to carry every key the merge functions subscript,
    or one dead process still takes the summary -- just from ``merge_dedup``
    instead of from the worker.
    """
    spec = worker_spec(tmp_path, worker_id=3)
    failure = corpus.worker_failure(
        MemoryError("killed"), progress=corpus.WorkerProgress(), games_completed=0,
    )
    slot = corpus.failed_worker_slot(spec, failure)

    summary = corpus.build_summary(
        results=[slot], requested={"run_id": "t"}, config_sha="abc",
        engine_record={"path": "/bin/sf"}, engine_id_name=None,
        staircase=corpus.parse_staircase(corpus.DEFAULT_STAIRCASE),
        started_utc="2026-08-27T00:00:00+00:00", wall_s=1.0,
    )

    assert summary["rows"] == 0
    assert summary["failed_workers"] == [
        {
            "worker_id": 3, "exception_type": "MemoryError", "exception": "killed",
            "last_game_id": None, "last_ply": None, "games_completed": 0,
        },
    ]
    assert summary["config_realized_by_worker"]["3"] == {
        "unavailable_worker_process_died": True,
    }
    assert json.loads(json.dumps(summary, default=corpus._json_default))


# ── kill and resume ──────────────────────────────────────────────────────────


def opening_fens() -> tuple[str, ...]:
    """Eight distinct 32-man openings, one per scripted first move.

    Built by PUSHING the move rather than by spelling the FEN out, so the
    en-passant and clock fields are whatever ``python-chess`` writes and the
    seed list cannot be rejected for a hand-typed field.  Distinct starts are
    what make a multi-game resume test mean anything: replaying one opening
    every game would have every game after the first dedup-served, and "the
    resumed run banked the same rows" would then be true of a run that banked
    nothing at all.
    """
    fens: list[str] = []
    for uci in (
        "e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "g2g3", "b2b3", "f2f4",
    ):
        board = chess.Board()
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())
    return tuple(fens)


#: A two-rung staircase: these tests are about the shard/progress protocol, and
#: the production three-rung scout would spend their whole runtime in the fake.
RESUME_STAIRCASE = "all:2,4:3"


def fen_list_opening(fens: Sequence[str], path: Path) -> OpeningConfig:
    """A multi-position seed list, through the PRODUCTION sampler.

    ⚑ ``_load_fen_list`` is ``lru_cache``d BY PATH, so every distinct list needs
    a distinct file name; a test that reused one would silently draw from
    another test's openings.
    """
    path.write_text("\n".join(fens) + "\n", encoding="utf-8")
    return OpeningConfig(
        opening_fen_list_path=str(path), opening_fen_prob=1.0,
    )


def scripted_worker(
    out_dir: Path, *, monkeypatch: pytest.MonkeyPatch, opening: OpeningConfig,
    **overrides: Any,
) -> dict[str, Any]:
    """``run_worker`` end to end against a fresh scripted engine per spawn."""
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *_a, **_kw: uci_double(ScriptedEngine()),
    )
    monkeypatch.setattr(corpus, "build_opening_config", lambda _spec: opening)
    values: dict[str, Any] = {
        "staircase": RESUME_STAIRCASE, "max_plies": 8, "shard_rows": 12,
    }
    values.update(overrides)
    return corpus.run_worker(worker_spec(out_dir, **values))


def next_shard_index(result: dict[str, Any]) -> int:
    """Where the killed session's in-flight shard would have been.

    ⚑ Read off the shard names the session actually closed, through the
    module's own parser -- NOT by counting progress lines.  A path-less
    completion record is a line that is not a shard, so the two counts differ,
    and a fixture that simulates a kill has to name the file the kill would
    really have caught.
    """
    return max(
        corpus.shard_index_of(Path(shard["path"]).name) or 0
        for shard in result["shards"]
    ) + 1


def rows_by_game(shards: Sequence[dict[str, Any]]) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    for shard in shards:
        for row in read_shard(Path(shard["path"])):
            out.setdefault(int(row["game_id"]), []).append(row)
    return out


def test_every_closed_shard_holds_only_whole_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rotation invariant the whole resume protocol rests on.

    Mutation caught: putting the rotation back in ``ShardWriter.write`` (the
    row-count rotation this file shipped before ``--resume``).  A game's rows
    then straddle a shard boundary, so ``a game appears in exactly one shard``
    below fails -- and with it the resume's only unit of work.  The mutant also
    closes shards with an EMPTY games list, which the first assert catches.
    """
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_list_opening(opening_fens(), tmp_path / "t1.txt")

    result = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=(0, 1, 2, 3), shard_rows=12,
    )

    assert result["failed"] is None
    assert len(result["shards"]) >= 2, "the row bound must have rotated"
    assert read_progress(out_dir, 0) == result["shards"]
    owner: dict[int, str] = {}
    for shard in result["shards"]:
        rows = read_shard(Path(shard["path"]))
        assert shard["games"], "a closed shard names the games it holds"
        assert shard["games"] == sorted(shard["games"])
        assert shard["rows"] == len(rows)
        for game_id in {int(row["game_id"]) for row in rows}:
            assert game_id not in owner, (
                f"game {game_id} is split across {owner.get(game_id)} and "
                f"{shard['path']}: a shard rotated mid-game"
            )
            owner[game_id] = shard["path"]
            assert game_id in shard["games"]
    # Every shard but the last crossed the bound; the last one is the tail.
    assert all(shard["rows"] >= 12 for shard in result["shards"][:-1])
    assert sum(shard["rows"] for shard in result["shards"]) == result["rows"]
    # The four openings really are four different games -- see `opening_fens`.
    banked = rows_by_game(result["shards"])
    assert set(banked) == {0, 1, 2, 3}
    keys = [row["dedup_key"] for rows in banked.values() for row in rows]
    assert len(set(keys)) == len(keys), "the games visited disjoint positions"


def test_a_killed_run_resumes_without_replaying_or_losing_a_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``kill -9``, then ``--resume``: the corpus a single run would have made.

    The kill is simulated exactly as ``SIGKILL`` leaves a worker: a shard file
    the progress file does not list (the one being written when the signal
    landed) and a torn final progress line (the append that was in flight).

    Mutation caught (a): NOT deleting the unlisted partial.  Its rows are then
    a half-game nothing indexes, the resumed worker's ``open("x")`` collides
    with it, and the run dies on its first shard.
    Mutation caught (b): keying completion off banked rows instead of the
    progress line -- games get replayed and the corpus gains duplicates.
    """
    opening = fen_list_opening(opening_fens(), tmp_path / "t2.txt")
    every_game = (0, 1, 2, 3)

    # The corpus an UNINTERRUPTED run produces, for the determinism compare.
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = scripted_worker(
        reference_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game,
    )

    # Session 1 dies after games 0 and 1.
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    first = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening, game_ids=(0, 1),
    )
    assert first["shards"], "session 1 must have closed at least one shard"
    killed_index = next_shard_index(first)
    partial = out_dir / f"w00-{killed_index:05d}.jsonl.zst"
    partial.write_bytes(b'{"schema": 1, "game_id": 2, "ply": 0')
    with open(out_dir / corpus.progress_name(0), "a", encoding="utf-8") as fh:
        fh.write('{"path": "w00-00009.jsonl.zst", "rows": 3, "co')

    second = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game, resume=True,
    )

    assert second["failed"] is None
    assert second["resume_progress_torn_tail"] is True
    assert second["resume_partials_deleted"] == [partial.name]
    # The resumed worker continues INTO that index, so "the junk is gone" is
    # proved by the file now reading as rows (the row compare below) rather
    # than by its absence -- and if the delete were skipped, `open("x")` would
    # refuse it and `failed` above would not be None.
    assert Path(second["shards"][0]["path"]).name == partial.name
    assert second["games_completed_prior"] == 2
    assert second["games"] == 2, "only the two unplayed games were played"
    assert second["realized"]["shard_index_start"] == killed_index
    assert second["resumed"] is True

    banked = rows_by_game([*second["shards_prior"], *second["shards"]])
    expected = rows_by_game(reference["shards"])
    assert set(banked) == set(expected) == set(every_game)
    for game_id in every_game:
        assert banked[game_id] == expected[game_id], (
            f"game {game_id} differs from the uninterrupted run"
        )
    plied = [(row["game_id"], row["ply"]) for rows in banked.values() for row in rows]
    assert len(set(plied)) == len(plied), "a row was banked twice"
    assert second["rows"] + first["rows"] == reference["rows"]


def tear_progress_tail(out_dir: Path, worker_id: int, fragment: str) -> None:
    """What ``kill -9`` leaves mid-append: a line with no newline after it."""
    with open(out_dir / corpus.progress_name(worker_id), "a",
              encoding="utf-8") as fh:
        fh.write(fragment)


def test_two_kills_in_a_row_still_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REPEATED ``kill -9`` IS THIS FEATURE'S CONTRACT, not a corner.

    Tolerating a torn tail on READ while leaving the bytes on DISK is not a
    resume: ``_append_progress`` opens ``"a"``, so session 2's first record
    lands on the end of session 1's fragment and the two become one line that
    is neither.  That line is then MID-FILE, so session 3 hits the "damaged
    some other way" refusal -- and the record swallowed inside it was a closed
    shard, so session 2's games are unknown as well.  One kill worked; two
    bricked the worker.

    Mutation caught: disabling ``repair_worker_progress`` (returning
    ``PROGRESS_INTACT`` without touching the file).  Session 3 then raises
    instead of adopting session 2's shard.
    """
    opening = fen_list_opening(opening_fens(), tmp_path / "t5.txt")
    every_game = (0, 1, 2, 3)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = scripted_worker(
        reference_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game,
    )

    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    first = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening, game_ids=(0, 1),
    )
    assert first["shards"], "session 1 closed a shard before the first kill"
    partial_one = out_dir / f"w00-{next_shard_index(first):05d}.jsonl.zst"
    partial_one.write_bytes(b'{"schema": 1, "game_id": 2, "ply": 0')
    tear_progress_tail(out_dir, 0, '{"path": "w00-00009.jsonl.zst", "rows": 3, "co')

    second = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=(0, 1, 2), resume=True,
    )

    assert second["failed"] is None
    assert second["games"] == 1, "session 2 played only game 2"
    assert second["shards"], "session 2 closed a shard the third must adopt"

    # ... and the SECOND kill, in the same place.
    partial_two = out_dir / f"w00-{next_shard_index(second):05d}.jsonl.zst"
    partial_two.write_bytes(b'{"schema": 1, "game_id": 3')
    tear_progress_tail(out_dir, 0, '{"path": "w00-00009.jsonl.zst", "rows"')

    third = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game, resume=True,
    )

    assert third["failed"] is None
    # ⚑ THE CLAIM: session 2's shard survived session 3's read. Without the
    # repair, session 3 does not get this far -- its pre-flight refuses the
    # corrupt mid-file line that session 2's append created.
    assert third["games_completed_prior"] == 3
    assert third["games"] == 1, "only game 3 was left"
    assert third["resume_partials_deleted"] == [partial_two.name]

    banked = rows_by_game([*third["shards_prior"], *third["shards"]])
    expected = rows_by_game(reference["shards"])
    assert set(banked) == set(expected) == set(every_game)
    for game_id in every_game:
        assert banked[game_id] == expected[game_id], (
            f"game {game_id} differs from the uninterrupted run after two kills"
        )
    plied = [(row["game_id"], row["ply"]) for rows in banked.values() for row in rows]
    assert len(set(plied)) == len(plied), "a row was banked twice"
    # The file the three sessions shared is still a clean append-only log.
    assert all(
        set(line) >= {"path", "rows", "codec", "games"}
        for line in read_progress(out_dir, 0)
    )
    # ... and both resumes said what they repaired.
    assert second["resume_progress_repair"] == corpus.PROGRESS_TRUNCATED
    assert second["resume_progress_torn_tail"] is True
    assert third["resume_progress_repair"] == corpus.PROGRESS_TRUNCATED


def test_a_kill_that_steals_only_the_newline_keeps_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variant with no torn record at all -- and it is the worse one.

    A kill can land after a whole progress line and before its newline.  The
    reader ACCEPTS that record, correctly: it is complete.  Then the next
    append concatenates onto it and destroys it, so a record that was accepted
    -- naming a closed shard and its games -- is gone, and the games it owned
    would be replayed on top of rows the corpus already holds.

    Mutation caught: repairing only the unparseable case (truncate-only, no
    newline restore).  ``games_completed_prior`` below then drops to 0 in
    session 2 -- the whole first shard is forgotten and both its games replay.
    """
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_list_opening(opening_fens(), tmp_path / "t6.txt")
    first = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening, game_ids=(0, 1),
    )
    assert len(first["shards"]) == 1

    path = out_dir / corpus.progress_name(0)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    path.write_text(text[:-1], encoding="utf-8")  # the kill stole the newline

    second = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=(0, 1, 2), resume=True,
    )

    assert second["games_completed_prior"] == 2, "the accepted record survived"
    assert second["games"] == 1

    # ⚑ The proof that it survived the APPEND, not just the read: a third
    # session reads the same file after session 2 wrote to it.
    third = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=(0, 1, 2, 3), resume=True,
    )

    assert third["games_completed_prior"] == 3
    assert third["games"] == 1
    assert second["resume_progress_repair"] == corpus.PROGRESS_NEWLINE_RESTORED
    # Nothing was LOST -- the record was whole, only its terminator was gone.
    assert second["resume_progress_torn_tail"] is False
    assert third["resume_progress_repair"] == corpus.PROGRESS_INTACT


def test_the_tail_repair_is_idempotent_and_leaves_whole_lines_alone(
    tmp_path: Path,
) -> None:
    """The repair's own contract, on each of its four inputs.

    Idempotence is what makes it safe to be killed inside: a second call must
    be a no-op, so a kill mid-repair leaves either the old state (repaired next
    time) or the repaired one, never a third thing.
    """
    path = tmp_path / corpus.progress_name(0)
    whole = json.dumps(
        {"path": "w00-00000.jsonl.zst", "rows": 3, "codec": "zstd", "games": [0],
         "cache_events": []},
        sort_keys=True,
    )

    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_ABSENT

    path.write_text(whole + "\n", encoding="utf-8")
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_INTACT
    assert path.read_text(encoding="utf-8") == whole + "\n"

    path.write_text(whole, encoding="utf-8")
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_NEWLINE_RESTORED
    assert path.read_text(encoding="utf-8") == whole + "\n"
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_INTACT

    path.write_text(whole + "\n" + '{"path": "w00-000', encoding="utf-8")
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_TRUNCATED
    assert path.read_text(encoding="utf-8") == whole + "\n"
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_INTACT

    # A fragment that is valid JSON but not a whole RECORD is still a fragment:
    # a truncated line can parse and still be missing the keys that make it
    # mean something.
    path.write_text('{"path": "w00-00000.jsonl.zst", "rows": 3}', encoding="utf-8")
    assert corpus.repair_worker_progress(path) == corpus.PROGRESS_TRUNCATED
    assert path.read_text(encoding="utf-8") == ""


def test_a_resumed_worker_re_warms_its_dedup_cache_from_its_own_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Take-effect, not a counter: the re-warmed positions are OBSERVED to
    suppress the searches they would have suppressed in one long run.

    Every game here opens on the SAME position, so game 1 replays game 0 move
    for move.  In an uninterrupted run every one of its plies is cache-served
    and it banks nothing.  A resume that started cold would re-search and
    RE-BANK all of them -- so ``rows == 0`` and ``positions_searched == 0``
    are the observation, and ``dedup_rewarmed`` is only its label.

    Mutation caught: dropping the re-warm (or handing it a cache the game loop
    never sees -- then ``dedup_rewarmed`` still reads 2 while
    ``dedup_rewarmed_resident`` reads 0).
    """
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_opening(MATE_GAME_FEN, tmp_path)
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(corpus, "build_opening_config", lambda _spec: opening)

    first = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))
    assert first["rows"] == 2

    second = corpus.run_worker(
        worker_spec(out_dir, game_ids=(0, 1), resume=True),
    )

    assert second["dedup_rewarmed"] == first["rows"]
    assert second["dedup_rewarmed_resident"] == first["rows"]
    assert second["games"] == 1, "game 0 was already banked"
    assert second["rows"] == 0, "every position came back from the re-warm"
    assert second["dedup"]["dup_hits"] == 2
    assert second["search"]["positions_searched"] == 0


class StatefulEngine(ScriptedEngine):
    """A scripted engine whose values depend on how many searches it has
    answered -- the stand-in for Stockfish's carried TT: a RE-search of a
    position is not a replay of the first search, so a resumed worker that
    searches where the uninterrupted one served produces different values,
    a different seeded move, and different rows after it."""

    def score_of(self, uci: str, *, depth: int) -> int:
        del depth  # the base signature's; this fake ranks by search count only
        return hashed_cp(f"{uci}#{self.go_count}")


def deal_openings_by_game(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines: Sequence[str],
) -> None:
    """Game ``g`` opens on ``lines[g % len(lines)]``, whichever session plays it."""
    real_rng, real_sample = corpus.book_rng, corpus.sample_starting_board

    class Dealt:
        def __init__(self, rng: Any, line: str) -> None:
            self.rng, self.dealt_line = rng, line

        def __getattr__(self, name: str) -> Any:
            return getattr(self.rng, name)

    def dealing_rng(*, seed: int, worker_id: int, game_id: int) -> Any:
        rng = real_rng(seed=seed, worker_id=worker_id, game_id=game_id)
        return Dealt(rng, lines[int(game_id) % len(lines)])

    monkeypatch.setattr(corpus, "book_rng", dealing_rng)
    monkeypatch.setattr(
        corpus, "sample_starting_board",
        lambda *, rng, cfg: real_sample(
            rng=rng, cfg=fen_list_opening([rng.dealt_line], tmp_path / f"seed{rng.random()}.txt"),
        ),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_list_opening(list(lines), tmp_path / "seeds.txt"),
    )


def test_a_games_record_without_cache_events_is_refused_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ FAIL CLOSED (Grok/Fable round 5): a progress record that names games
    but carries no ``cache_events`` predates the contract; adopting it as
    "zero events" would silently lose its label-only entries.  Refused by
    name on both record shapes."""
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_opening(MATE_GAME_FEN, tmp_path)
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(corpus, "build_opening_config", lambda _spec: opening)
    first = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))
    assert first["rows"] == 2
    progress = out_dir / corpus.progress_name(0)
    records = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert all("cache_events" in r for r in records), "the writer commits the key"
    for record in records:
        del record["cache_events"]
    progress.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match=r"games \[0\] with no cache_events"):
        corpus.resume_worker_state(
            out_dir=out_dir, worker_id=0,
            cache=corpus.DedupCache(max_entries=corpus.DEFAULT_DEDUP_CACHE_MAX),
        )
    # A null-path completion record is held to the same rule.
    progress.write_text(
        json.dumps({"path": None, "rows": 0, "codec": "zstd", "games": [1]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"games \[1\] with no cache_events"):
        corpus.resume_worker_state(
            out_dir=out_dir, worker_id=0,
            cache=corpus.DedupCache(max_entries=corpus.DEFAULT_DEDUP_CACHE_MAX),
        )


def banked_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("w*-*.jsonl.*")):
        rows.extend(corpus.iter_shard_rows(path))
    return sorted(rows, key=lambda r: (int(r["game_id"]), int(r["ply"])))


@pytest.mark.parametrize(
    ("max_plies", "game_ids", "dedup_cache_max"),
    [
        (1, (0, 1, 2, 3), corpus.DEFAULT_DEDUP_CACHE_MAX),
        (2, (0, 1, 2, 3), corpus.DEFAULT_DEDUP_CACHE_MAX),
        # ⚑ THE ORDER PROBE (Grok round 5): games dealt DESCENDING with a
        # one-entry FIFO. Live: game 2 (old-repeat route) searches and banks,
        # then game 0 (no-repeat route, same tensor) searches label-only and
        # EVICTS game 2's label -- resident = no_repeat. A replay sorted by
        # (game_id, ply) would apply game 0's event first and game 2's row
        # second, leaving old_repeat resident, and game 3 (no-repeat) would
        # then SEARCH where the live run served. Replay by seq keeps it.
        (1, (2, 0, 3), 1),
    ],
)
def test_a_label_only_cache_entry_survives_a_resume_and_the_resumed_run_equals_the_uninterrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_plies: int,
    game_ids: tuple[int, ...], dedup_cache_max: int,
) -> None:
    """⚑⚑ Codex P2 / operator ruling (#497 round 3): resume equivalence must
    not depend on unbanked cache state.

    Game 1 reaches game 0's tensor by a route with an older repeat (T4): a
    seen ``input_key``, a new ``search_key`` -- it SEARCHES, caches the
    values under the new label, banks no row.  That entry lived only in RAM.
    Run A plays games 0-3 uninterrupted; run B is stopped right after game 1
    (the label-only entry just created) and resumed for games 2-3.  Game 2
    takes game 1's route again: A serves it from the label-only entry; a
    resume that could not rebuild the entry would SEARCH it -- and under a
    stateful engine (the carried TT) the re-search gives other values, another
    seeded move and other rows after it.  B must equal A: every hit/miss
    decision, every chosen move, every row and value.  Mutant: the events
    not committed with the record -> B searches, the rows diverge, this fails.

    ⚑ Two ply caps, two replay paths: at 2 plies game 1 banks its ply-1 row
    AFTER the event, so the event is merged in ahead of a row; at 1 ply game
    1 banks nothing and the event trails the record's last row.  A rewarm
    that replayed only one of the two passed the other (measured: the
    tail-replay mutant survived the 2-ply case alone).
    """
    lines = [
        f"{chess.STARTING_FEN} | {T4_ROUTE_NO_REPEAT}",
        f"{chess.STARTING_FEN} | {T4_ROUTE_OLD_REPEAT}",
        f"{chess.STARTING_FEN} | {T4_ROUTE_OLD_REPEAT}",
        f"{chess.STARTING_FEN} | {T4_ROUTE_NO_REPEAT}",
    ]
    deal_openings_by_game(monkeypatch, tmp_path, lines)
    monkeypatch.setattr(
        corpus, "StockfishUCI", lambda *_a, **_kw: uci_double(StatefulEngine()),
    )
    overrides: dict[str, Any] = {
        "staircase": RESUME_STAIRCASE, "max_plies": max_plies, "shard_rows": 100,
        "dedup_cache_max": dedup_cache_max,
    }
    stop_after = game_ids[:2]
    label_only_game = game_ids[1]
    out_a = tmp_path / "a"
    out_a.mkdir()
    a = corpus.run_worker(worker_spec(out_a, game_ids=game_ids, **overrides))
    assert a["failed"] is None
    assert a["dedup"]["search_key_miss_on_seen_input"] == 1, "the label-only search"

    out_b = tmp_path / "b"
    out_b.mkdir()
    stopped = corpus.run_worker(worker_spec(out_b, game_ids=stop_after, **overrides))
    assert stopped["failed"] is None
    assert stopped["dedup"]["search_key_miss_on_seen_input"] == 1
    # The entry is ON DISK, committed with the game's record, with its seq.
    records = [
        json.loads(line) for line in
        (out_b / corpus.progress_name(0)).read_text(encoding="utf-8").splitlines()
    ]
    events = [e for r in records for e in r.get("cache_events", [])]
    assert [(e["game_id"], e["ply"], e["remember_input"]) for e in events] == [
        (label_only_game, 0, False),
    ]
    # One seq per SEARCH, rows and events together, gapless: the event sits
    # exactly where the live cache changed.
    stopped_seqs = sorted(
        [row["seq"] for row in banked_rows(out_b)] + [e["seq"] for e in events],
    )
    assert stopped_seqs == list(range(stopped["search"]["positions_searched"]))
    label_only_board = board_after(lines[label_only_game % len(lines)].split("| ")[1])
    assert events[0]["search_key"] == corpus.search_key(label_only_board)
    assert events[0]["input_key"] == corpus.row_key(label_only_board)
    assert [row["seq"] for row in banked_rows(out_b)][:1] == [0]

    resumed = corpus.run_worker(
        worker_spec(out_b, game_ids=game_ids, resume=True, **overrides),
    )
    assert resumed["failed"] is None
    assert resumed["dedup_cache_events_rewarmed"] == 1
    assert resumed["games"] == len(game_ids) - 2
    # Every decision after the resume is the one A made: game 2's label is
    # SERVED (no search), and so is everything after it.
    assert resumed["search"]["positions_searched"] == 0
    assert resumed["dedup"]["search_key_miss_on_seen_input"] == 0
    assert (
        resumed["dedup"]["dup_hits"] + stopped["dedup"]["dup_hits"]
        == a["dedup"]["dup_hits"]
    )
    assert (
        resumed["search"]["positions_searched"] + stopped["search"]["positions_searched"]
        == a["search"]["positions_searched"]
    )
    rows_a, rows_b = banked_rows(out_a), banked_rows(out_b)
    assert [(r["game_id"], r["ply"]) for r in rows_a] == [(r["game_id"], r["ply"]) for r in rows_b]
    assert rows_a == rows_b, "same rows, same played moves, same values"
    assert a["rows"] == stopped["rows"] + resumed["rows"]


def test_a_zero_work_resume_reports_the_corpus_windows_it_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Codex P2 (round 2): a resume that plays nothing new still has rows.

    Session 1 banks the corpus; session 2 is dealt the same games, replays
    none, and its summary must still say what windows the corpus's rows carry
    -- ``sum(histogram) == rows`` at the CORPUS level, off the tallies the
    progress records committed with each shard.  Before this round the
    histograms were session-only, so this summary read ``rows: 2`` beside
    ``history_plies_histogram: {}``.
    """
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_opening(MATE_GAME_FEN, tmp_path)
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(corpus, "build_opening_config", lambda _spec: opening)

    first = corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))
    assert first["rows"] == 2
    second = corpus.run_worker(worker_spec(out_dir, game_ids=(0,), resume=True))
    assert second["games"] == 0
    assert second["rows"] == 0
    assert second["history_plies_histogram"] == {}, "session-scoped: nothing new"
    assert second["history_plies_histogram_prior"] == first["history_plies_histogram"]
    assert second["history_root_reasons_prior"] == first["history_root_reasons"]
    assert second["history_tallies_unknown_rows_prior"] == 0

    summary = corpus.build_summary(
        results=[second],
        requested={"run_id": "t"},
        config_sha="abc",
        engine_record={"path": "/bin/sf", "sha256": "d" * 64},
        engine_id_name="Stockfish test",
        staircase=corpus.parse_staircase(corpus.DEFAULT_STAIRCASE),
        started_utc="2026-09-01T00:00:00+00:00",
        wall_s=1.0,
    )
    assert summary["rows"] == 2
    assert sum(summary["history_plies_histogram"].values()) == summary["rows"]
    assert sum(summary["history_root_reasons"].values()) == summary["rows"]
    assert summary["history_plies_histogram_this_session"] == {}
    assert summary["history_tallies_unknown_rows"] == 0


def test_the_re_warmed_value_vector_is_the_one_the_live_visit_cached(
    tmp_path: Path,
) -> None:
    """The re-warm has to rebuild the CACHED object, not something like it.

    Selection is ``argmax(q/tau + gumbel)`` over ``effective_cp``, so a value
    vector that differs in one float -- a rank order, a float64 round trip, the
    wrong depth's block -- moves the played move, and the resumed corpus stops
    being the corpus an uninterrupted run would have written.  Nothing else
    would report that: the rows would still be well formed.

    Mutation caught: reading the DEEPEST phase's block instead of phase 0's at
    ``selection.value_depth`` (the narrowed rung has 4 of the 20 moves).
    """
    engine = ScriptedEngine(preferred=MATE_GAME_SCRIPT)
    spec = worker_spec(tmp_path)
    searcher = searcher_for(engine)
    cache = corpus.DedupCache(max_entries=spec.dedup_cache_max)
    outcome = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=fen_opening(MATE_GAME_FEN, tmp_path),
        game_id=0, cache=cache, dedup=corpus.DedupStats(),
        progress=corpus.WorkerProgress(), seq=corpus.WorkerSeq(),
    )

    assert outcome.rows
    for row in outcome.rows:
        live = cache.get(row["search_key"])
        assert live is not None
        assert cache.input_seen(row["input_key"])
        rebuilt = corpus.selection_values_from_row(row)
        assert rebuilt.moves == live.moves
        assert np.array_equal(rebuilt.effective_cp, live.effective_cp)
        assert rebuilt.effective_cp.dtype == live.effective_cp.dtype
        # The number selection actually reads, not just the bytes behind it.
        assert np.array_equal(searcher.q_of(rebuilt), searcher.q_of(live))


def test_a_game_that_banked_no_rows_is_still_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed game with zero rows survives the kill as a path-less record.

    Mutation caught: dropping ``close()``'s null-path flush (or only recording
    games on a shard that has rows).  Game 1 banks nothing because every one of
    its positions is dedup-served, so a resume with no completion record for it
    replays it -- forever, every session, and each replay re-runs the searches.
    """
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    opening = fen_opening(MATE_GAME_FEN, tmp_path)
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(corpus, "build_opening_config", lambda _spec: opening)

    corpus.run_worker(worker_spec(out_dir, game_ids=(0,)))
    second = corpus.run_worker(worker_spec(out_dir, game_ids=(0, 1), resume=True))
    assert second["rows"] == 0, "the fixture's point: game 1 banks nothing"

    third = corpus.run_worker(
        worker_spec(out_dir, game_ids=(0, 1, 2), resume=True),
    )

    assert third["games_completed_prior"] == 2
    assert third["games"] == 1, "the row-less game 1 was not replayed"
    assert third["search"]["positions_searched"] == 0
    # ... and the records that carried it: one per row-less game, path-less
    # because there is no file to index.
    assert [
        line for line in read_progress(out_dir, 0) if line["path"] is None
    ] == [
        {"path": None, "rows": 0, "codec": second["codec"], "games": [1],
         "cache_events": []},
        {"path": None, "rows": 0, "codec": third["codec"], "games": [2],
         "cache_events": []},
    ]


def test_a_progress_file_written_before_end_game_is_adopted_by_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LEGACY ADOPTION: lines with no ``games`` list, and a mid-game shard cut.

    The production burn opened before game-boundary rotation existed, so its
    progress lines are ``{path, rows, codec}`` and its shards rotate on the row
    bound -- the last game in a closed shard may hold only part of its plies.
    Those game ids are DERIVED by reading the shard, truncation included: a
    shard is immutable, and replaying the game to heal its tail would duplicate
    every row already banked.

    Mutation caught: treating a line with no ``games`` as claiming no games.
    Every game in the adopted shard is then replayed and the corpus doubles.
    """
    opening = fen_list_opening(opening_fens(), tmp_path / "t3.txt")
    every_game = (0, 1, 2, 3)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = scripted_worker(
        reference_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game, shard_rows=10**6,
    )
    banked = rows_by_game(reference["shards"])
    assert {0, 1, 2, 3} <= set(banked)
    # ... a shard that ends HALF WAY THROUGH game 2, which is what a row-bound
    # rotation does and what game-boundary rotation exists to stop.
    head = len(banked[2]) // 2
    assert head >= 1
    kept = [*banked[0], *banked[1], *banked[2][:head]]

    out_dir = tmp_path / "legacy"
    out_dir.mkdir()
    writer = corpus.ShardWriter(out_dir=out_dir, worker_id=0, shard_rows=10**6)
    for row in kept:
        writer.write(row)
    # The pre-`end_game` writer had no notion of a game boundary; ending them
    # here only keeps the FILE from being abandoned, and the progress line it
    # produces is thrown away for the legacy one below.
    for game_id in (0, 1, 2):
        writer.end_game(game_id)
    writer.close()
    legacy = {k: v for k, v in writer.shards[0].items() if k != "games"}
    (out_dir / corpus.progress_name(0)).write_text(
        json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8",
    )
    partial = out_dir / f"w00-00001{writer.suffix}"
    partial.write_bytes(b'{"schema": 1, "game_id": 3')

    result = scripted_worker(
        out_dir, monkeypatch=monkeypatch, opening=opening,
        game_ids=every_game, resume=True, shard_rows=10**6,
    )

    # The substantive claims first, so a mutant fails on the CORPUS rather
    # than on the bookkeeping key that happens to be declared above it.
    assert result["failed"] is None
    assert result["games_completed_prior"] == 3, "0, 1 and the truncated 2"
    assert result["games"] == 1, "only game 3 was left to play"
    assert result["shards_prior"][0]["games"] == [0, 1, 2]
    assert result["shards_prior"][0]["games_derived"] is True
    assert result["resume_legacy_progress_lines"] == 1
    assert result["resume_partials_deleted"] == [partial.name]

    adopted = rows_by_game([*result["shards_prior"], *result["shards"]])
    assert adopted[0] == banked[0]
    assert adopted[1] == banked[1]
    # The truncation is ACCEPTED, not healed and not replayed.
    assert adopted[2] == banked[2][:head]
    assert adopted[3], "the unplayed game really did generate"
    plied = [(row["game_id"], row["ply"]) for rows in adopted.values() for row in rows]
    assert len(set(plied)) == len(plied), "a row was banked twice"


def test_a_torn_progress_line_anywhere_but_the_tail_is_refused(
    tmp_path: Path,
) -> None:
    """A kill can only tear the LAST line, and only short of its newline.

    Mutation caught: skipping every unparseable line.  Damage in the middle
    then silently drops every shard listed BELOW it, so their games are
    replayed and the corpus gains duplicates -- the failure mode the tolerance
    exists to avoid, reintroduced by widening the tolerance.  The second case
    is the same widening one notch smaller: a garbled LAST line that ends in a
    newline was written whole and is not a torn tail.
    """
    path = tmp_path / corpus.progress_name(0)
    good = json.dumps({"path": None, "rows": 0, "codec": "zstd", "games": [1],
                       "cache_events": []})
    path.write_text(
        '{"path": "w00-000' + "\n" + good + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not the torn tail"):
        corpus.read_worker_progress(path)

    path.write_text(good + "\n" + '{"path": "w00-000' + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="this file ends in one"):
        corpus.read_worker_progress(path)

    path.write_text(good + "\n" + '{"path": "w00-000', encoding="utf-8")
    records, torn = corpus.read_worker_progress(path)
    assert torn is True
    assert [record["games"] for record in records] == [[1]]

    path.write_text(json.dumps({"path": "a", "rows": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing \['codec'\]"):
        corpus.read_worker_progress(path)


def test_a_listed_shard_that_is_gone_refuses_the_resume(tmp_path: Path) -> None:
    """Its games would be marked complete against rows that no longer exist."""
    (tmp_path / corpus.progress_name(0)).write_text(
        json.dumps({
            "path": str(tmp_path / "w00-00000.jsonl.zst"), "rows": 3,
            "codec": "zstd", "games": [0], "cache_events": [],
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is not in"):
        corpus.resume_worker_state(
            out_dir=tmp_path, worker_id=0,
            cache=corpus.DedupCache(max_entries=10),
        )


def test_a_listed_shard_that_is_truncated_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A shard a progress line LISTS is claimed complete.

    MEASURED on a copy of a live production shard: a truncated one
    decompresses cleanly up to a partial final line, and the bare
    ``JSONDecodeError`` that follows names neither the file nor the corpus.
    Mutation caught: letting that exception out unwrapped -- a day-13 operator
    then reads ``Expecting property name ... char 4347``.
    """
    whole = {
        "schema": corpus.ROW_SCHEMA, "game_id": 0, "ply": 0, "seq": 0, "dedup_key": "k",
        "search_key": "k|", "input_key": "0" * 32,
        "selection": {"value_depth": 1, "value_width": 1},
        "phases": [{"per_depth": [{"depth": 1, "lines": [[1, "e2e4", 0.0, 9]]}]}],
    }
    shard = tmp_path / "w00-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(whole, sort_keys=True) + "\n")
        fh.write('{"schema": 1, "game_id": 0, "dedup')
    (tmp_path / corpus.progress_name(0)).write_text(
        json.dumps({
            "path": str(shard), "rows": 2, "codec": "gzip", "games": [0],
            "cache_events": [],
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"w00-00000\.jsonl\.gz line 2"):
        corpus.resume_worker_state(
            out_dir=tmp_path, worker_id=0,
            cache=corpus.DedupCache(max_entries=10),
        )


def test_a_resume_whose_game_deal_disagrees_with_the_progress_file_is_refused(
    tmp_path: Path,
) -> None:
    """A worker cannot adopt games this run never dealt it.

    ``--workers`` is compared against the manifest, so re-dealing is refused
    one level up -- this is the belt: a progress file that claims game 7 while
    this worker owns {0, 1} would leave game 7 unplayed by ANY worker, and the
    run would report a complete corpus that is missing a game.
    """
    (tmp_path / corpus.progress_name(0)).write_text(
        json.dumps({"path": None, "rows": 0, "codec": "zstd", "games": [7],
                    "cache_events": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not deal it"):
        corpus.run_worker(worker_spec(tmp_path, game_ids=(0, 1), resume=True))


def test_a_resume_that_changes_a_generation_setting_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate that makes ``--resume`` safe, end to end through ``run``.

    Mutation caught: dropping the field-by-field comparison against the
    manifest.  A resume with a different ``--temp-high`` then appends rows
    sampled at another temperature to a corpus whose stamp says otherwise --
    two configurations under one ``config_sha256``, and nothing downstream can
    tell which rows are which.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    out_dir = tmp_path / "run"
    argv = [
        "--out-dir", str(out_dir), "--games", "1", "--workers", "1",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
    ]
    corpus.run(corpus.build_parser().parse_args(argv))

    # A run that WROTE ITS SUMMARY finished; there is nothing to resume.
    with pytest.raises(ValueError, match="Nothing to resume"):
        corpus.run(corpus.build_parser().parse_args([*argv, "--resume"]))
    (out_dir / corpus.SUMMARY_NAME).unlink()  # ... now it looks killed.

    with pytest.raises(ValueError, match=r"temp_high: 0\.01 -> 0\.5"):
        corpus.run(corpus.build_parser().parse_args(
            [*argv[:argv.index("--temp-high") + 1], "0.5",
             *argv[argv.index("--temp-high") + 2:], "--resume"],
        ))
    # ... and a plain rerun is still refused, --resume or not.
    with pytest.raises(FileExistsError, match="already holds files"):
        corpus.run(corpus.build_parser().parse_args(argv))

    (out_dir / corpus.MANIFEST_NAME).unlink()
    with pytest.raises(ValueError, match="no run in this directory"):
        corpus.run(corpus.build_parser().parse_args([*argv, "--resume"]))


def test_a_manifest_that_does_not_hash_its_own_config_is_refused(
    tmp_path: Path,
) -> None:
    """The integrity half: every later comparison trusts this dict."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / corpus.MANIFEST_NAME).write_text(
        json.dumps({
            "config_requested": {"games": 4}, "config_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inconsistent with itself"):
        corpus.load_resume_manifest(out_dir)


def test_the_resume_gate_compares_the_manifests_own_keys_not_a_fresh_sha(
    tmp_path: Path,
) -> None:
    """⚑ Why this is a field compare and not ``sha == sha``.

    A stamp recomputed by TODAY's code and hashed against a manifest written
    by a build that predates a stamp key refuses every legacy resume -- a run
    that has burned for days becomes unresumable for a reason that has nothing
    to do with its configuration.  So a key the manifest does not carry is not
    a claim anyone made, a key it does carry is compared exactly, and
    ``out_dir`` is compared RESOLVED because a corpus is its files.

    Mutation caught: comparing ``stamp_sha256(requested)`` to
    ``manifest["config_sha256"]`` -- the first assert below then raises.
    """
    manifest = {
        "config_requested": {"games": 4, "seed": 7, "out_dir": str(tmp_path)},
        "config_sha256": "unused-by-this-function",
    }
    requested = {
        "games": 4, "seed": 7, "out_dir": f"{tmp_path}/../{tmp_path.name}",
        # A key the manifest predates: not a claim it ever made.
        "sf_search_timeout_s": 8.0,
    }
    corpus.refuse_resume_config_drift(manifest, requested=requested)

    # A missing newly introduced key is ordinarily no old claim, but the
    # pre-G10 meaning is knowable: every old staircase was fixed.  Resuming an
    # old exact-shape fixed corpus under G10 must not mix the two labelers.
    corpus.refuse_resume_config_drift(
        manifest,
        requested={
            **requested,
            "staircase_policy": corpus.STAIRCASE_POLICY_FIXED,
        },
    )
    with pytest.raises(ValueError, match="pre-policy default"):
        corpus.refuse_resume_config_drift(
            manifest,
            requested={
                **requested,
                "staircase_policy": corpus.STAIRCASE_POLICY_G10,
            },
        )

    with pytest.raises(ValueError, match="seed: 7 -> 9"):
        corpus.refuse_resume_config_drift(
            manifest, requested={**requested, "seed": 9},
        )
    with pytest.raises(ValueError, match="out_dir"):
        corpus.refuse_resume_config_drift(
            manifest, requested={**requested, "out_dir": str(tmp_path / "other")},
        )
    with pytest.raises(ValueError, match="does not stamp it"):
        corpus.refuse_resume_config_drift(
            manifest, requested={"games": 4, "out_dir": str(tmp_path)},
        )


def resumable_dir(tmp_path: Path) -> Path:
    """A directory holding a self-consistent manifest and nothing else.

    Everything the resume gate checks BEFORE it looks at ``summary.json``, so a
    test of the summary rule fails on the summary rule or not at all.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    requested = {"games": 4, "seed": 7, "out_dir": str(out_dir)}
    (out_dir / corpus.MANIFEST_NAME).write_text(
        json.dumps({
            "row_schema": corpus.ROW_SCHEMA,
            corpus.KEY_HISTORY_REP_FIX: corpus.HISTORY_REP_FIX,
            "config_requested": requested,
            "config_sha256": corpus.stamp_sha256(requested),
        }),
        encoding="utf-8",
    )
    return out_dir


def rewrite_manifest_row_schema(out_dir: Path, row_schema: Any) -> None:
    """Stamp another row schema on an otherwise self-consistent manifest."""
    path = out_dir / corpus.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["row_schema"] = row_schema
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_resume_onto_an_older_row_schema_is_refused_at_the_manifest(
    tmp_path: Path,
) -> None:
    """⚑ Codex P1: the schema gate is the MANIFEST's, before any worker.

    The per-row check in the cache re-warm only fires for a worker that has
    shards to re-warm from; the gate that protects every worker is this one.
    """
    out_dir = resumable_dir(tmp_path)
    corpus.load_resume_manifest(out_dir)  # the current schema passes

    rewrite_manifest_row_schema(out_dir, corpus.ROW_SCHEMA - 1)
    with pytest.raises(ValueError, match=r"row schema 2 .* schema 3"):
        corpus.load_resume_manifest(out_dir)
    rewrite_manifest_row_schema(out_dir, 1)
    with pytest.raises(ValueError, match=r"row schema 1 .* schema 3"):
        corpus.load_resume_manifest(out_dir)

    rewrite_manifest_row_schema(out_dir, None)
    with pytest.raises(ValueError, match="row schema None"):
        corpus.load_resume_manifest(out_dir)


def test_a_schema_2_manifest_is_refused_by_name_on_resume(tmp_path: Path) -> None:
    """The keyless intermediate shape, named in the refusal (Fable, round 2 delta)."""
    out_dir = resumable_dir(tmp_path)
    rewrite_manifest_row_schema(out_dir, corpus.ROW_SCHEMA_HISTORY_WITHOUT_KEYS)
    with pytest.raises(ValueError, match=r"search_key/input_key.*Regenerate") as excinfo:
        corpus.load_resume_manifest(out_dir)
    assert "row schema 2" in str(excinfo.value)


def test_a_manifest_in_another_repetition_regime_is_refused_on_resume(
    tmp_path: Path,
) -> None:
    out_dir = resumable_dir(tmp_path)
    path = out_dir / corpus.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[corpus.KEY_HISTORY_REP_FIX] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="history_rep_fix=False"):
        corpus.load_resume_manifest(out_dir)
    del manifest[corpus.KEY_HISTORY_REP_FIX]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="history_rep_fix=None"):
        corpus.load_resume_manifest(out_dir)


@pytest.mark.parametrize(
    "progress_record",
    [
        pytest.param(
            {"path": "w00-00000.jsonl.zst", "rows": 3, "codec": "zstd", "games": [0],
         "cache_events": []},
            id="worker_with_a_schema_1_shard",
        ),
        pytest.param(
            {"path": None, "rows": 0, "codec": "zstd", "games": [0], "cache_events": []},
            id="worker_with_only_zero_row_completion_records",
        ),
    ],
)
def test_a_schema_1_corpus_cannot_be_resumed_by_any_worker_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, progress_record: dict[str, Any],
) -> None:
    """⚑⚑ Through ``run``: refused BEFORE a record is archived or a worker starts.

    The second shape is the one the row-level check cannot catch: a worker whose
    progress file holds nothing but zero-row completion records re-warms no row,
    so nothing in it ever reads a schema -- and it would append schema-2 rows
    beside the other workers' schema-1 shards.  The refusal has to come from
    the manifest, and it has to leave the directory exactly as it found it.
    """
    out_dir = resumable_dir(tmp_path)
    rewrite_manifest_row_schema(out_dir, corpus.ROW_SCHEMA - 1)
    write_summary(out_dir, run_finished=False, failed_workers=[{"worker_id": 0}])
    (out_dir / corpus.progress_name(0)).write_text(
        json.dumps(progress_record) + "\n", encoding="utf-8",
    )
    before = {p.name: p.read_bytes() for p in out_dir.iterdir()}

    def no_worker(spec: corpus.WorkerSpec) -> dict[str, Any]:
        raise AssertionError(f"a worker was dispatched onto a refused resume: {spec}")

    monkeypatch.setattr(corpus, "run_worker", no_worker)
    argv = [
        "--out-dir", str(out_dir), "--games", "1", "--workers", "1",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT), "--resume",
    ]
    with pytest.raises(ValueError, match="row schema 2"):
        corpus.run(corpus.build_parser().parse_args(argv))

    after = {p.name: p.read_bytes() for p in out_dir.iterdir()}
    assert after == before, "a refused resume must not archive or write a byte"


def test_a_two_worker_schema_1_resume_is_refused_before_either_worker_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Grok B1/B2, the multi-worker shape: one worker has a listed schema-1
    shard, the other an EMPTY progress file.  Without the manifest gate the
    first would raise inside its re-warm (outside ``run_worker``'s ``try``, so
    the parent records a failed slot with nothing in it) and the second would
    start cold and bank schema-2 rows into the schema-1 corpus.  The refusal
    has to come before the pool is even built.
    """
    out_dir = resumable_dir(tmp_path)
    rewrite_manifest_row_schema(out_dir, corpus.ROW_SCHEMA - 1)
    (out_dir / corpus.progress_name(0)).write_text(
        json.dumps({
            "path": "w00-00000.jsonl.zst", "rows": 3, "codec": "zstd", "games": [0],
            "cache_events": [],
        }) + "\n",
        encoding="utf-8",
    )
    (out_dir / corpus.progress_name(1)).write_text("", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in out_dir.iterdir()}

    class NoPool:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("a worker pool was built for a refused resume")

    monkeypatch.setattr(corpus, "ProcessPoolExecutor", NoPool)
    monkeypatch.setattr(
        corpus, "run_worker",
        lambda spec: (_ for _ in ()).throw(AssertionError(f"worker started: {spec}")),
    )
    argv = [
        "--out-dir", str(out_dir), "--games", "2", "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT), "--resume",
    ]
    with pytest.raises(ValueError, match="row schema 2"):
        corpus.run(corpus.build_parser().parse_args(argv))

    after = {p.name: p.read_bytes() for p in out_dir.iterdir()}
    assert after == before


def write_summary(out_dir: Path, **keys: Any) -> Path:
    """A ``summary.json`` carrying exactly the keys a case is about."""
    path = out_dir / corpus.SUMMARY_NAME
    path.write_text(json.dumps(keys), encoding="utf-8")
    return path


def test_the_resume_gate_reads_the_summarys_verdict_not_its_existence(
    tmp_path: Path,
) -> None:
    """⚑ A ``summary.json`` is a run RECORD, not a completion claim.

    OBSERVED 2026-08-29 on a multi-day burn: an OOM broke the worker pool.
    ``run`` SURVIVED it -- that is the whole point of the synthesised failed
    slot -- so it wrote the summary it always writes, with a full
    ``failed_workers`` block and every this-session total at zero.  The gate
    keyed on the FILE EXISTING, so it answered "that run completed and wrote
    its summary" about a run that had crashed, and the corpus could not be
    resumed until an operator moved the crash record aside by hand.

    Three shapes, one rule -- the summary's own ``run_finished``:

    * crashed (``false``)  -> the gate lets the resume through;
    * completed (``true``) -> refused, in the words operators already know;
    * no verdict at all    -> refused.  ⚑ FAIL CLOSED: a legacy summary is
      ambiguous, and reading ambiguity as "crashed" would append games to a
      corpus that really had finished.  The cost of this direction is one
      manual rename; the cost of the other is a corpus nothing can un-mix.

    Mutation caught: keying the refusal on ``summary_path.exists()`` again --
    the crashed case then raises instead of returning the manifest.
    """
    out_dir = resumable_dir(tmp_path)
    assert corpus.load_resume_manifest(out_dir)["config_sha256"], "no summary yet"

    write_summary(out_dir, run_finished=False, failed_workers=[{"worker_id": 0}])
    assert corpus.load_resume_manifest(out_dir)["config_requested"]["seed"] == 7

    write_summary(out_dir, run_finished=True, failed_workers=[])
    with pytest.raises(ValueError, match="Nothing to resume"):
        corpus.load_resume_manifest(out_dir)

    # The banked 2026-08-29 crash record's own shape: written before the key
    # existed, so it makes no claim -- and is refused for that reason alone,
    # not for what its `failed_workers` implies.
    write_summary(out_dir, rows=0, games=0, failed_workers=[{"worker_id": 0}])
    with pytest.raises(ValueError, match="states no run_finished verdict"):
        corpus.load_resume_manifest(out_dir)


@pytest.mark.parametrize(
    "claim", [None, "false", "true", 0, 1, [], {"run_finished": False}],
)
def test_only_a_real_bool_is_a_completion_verdict(claim: Any) -> None:
    """⚑ Truthiness is not a verdict, in either direction.

    ``"false"`` is a true string and ``0`` is a false int, so a gate written as
    ``if summary.get("run_finished")`` would read a hand-edited or
    wrong-typed value as a claim nobody made -- accepted and then silently
    meaning something else, which is this repo's signature defect.  Anything
    that is not a ``bool`` states nothing, and stating nothing is refused.
    """
    assert corpus.summary_run_finished({"run_finished": claim}) is None
    assert corpus.summary_run_finished({}) is None
    assert corpus.summary_run_finished([1, 2, 3]) is None
    assert corpus.summary_run_finished({"run_finished": True}) is True
    assert corpus.summary_run_finished({"run_finished": False}) is False


def test_an_unreadable_summary_states_no_verdict_rather_than_raising(
    tmp_path: Path,
) -> None:
    """Truncated, absent, or not an object: all ambiguous, all refused."""
    out_dir = resumable_dir(tmp_path)
    (out_dir / corpus.SUMMARY_NAME).write_text('{"run_finished": fal', "utf-8")
    assert corpus.read_summary_run_finished(out_dir / corpus.SUMMARY_NAME) is None
    with pytest.raises(ValueError, match="states no run_finished verdict"):
        corpus.load_resume_manifest(out_dir)

    (out_dir / corpus.SUMMARY_NAME).write_text("[]", "utf-8")
    assert corpus.read_summary_run_finished(out_dir / corpus.SUMMARY_NAME) is None
    assert corpus.read_summary_run_finished(out_dir / "nothing-here.json") is None


def test_a_crash_record_is_kept_rather_than_overwritten_by_the_resume(
    tmp_path: Path,
) -> None:
    """⚑ What makes the relaxed gate SAFE rather than merely permissive.

    ``run`` banks its summary with ``open("x")``.  A resume let past the gate
    with the crashed session's summary still in place would search for however
    many days and die on the very last line -- the days-late failure the old
    blanket refusal was really buying.  So the crash record is MOVED, never
    clobbered: it is the only copy of the ``failed_workers`` block that says
    what killed the run, and three crashes keep three records in order.

    Mutation caught: dropping the ``archive_unfinished_summary`` call from
    ``run`` -- the resumed session then raises ``FileExistsError`` at the end
    (proved end to end in
    ``test_a_crashed_run_resumes_and_its_crash_record_survives``).
    """
    out_dir = resumable_dir(tmp_path)
    assert corpus.archive_unfinished_summary(out_dir) is None, "nothing to move"

    write_summary(out_dir, run_finished=False, failed_workers=[{"worker_id": 3}])
    first = corpus.archive_unfinished_summary(out_dir)
    assert first is not None
    assert first.name == "summary.unfinished_00.json"
    assert not (out_dir / corpus.SUMMARY_NAME).exists()
    assert json.loads(first.read_text("utf-8"))["failed_workers"][0]["worker_id"] == 3

    write_summary(out_dir, run_finished=False, failed_workers=[{"worker_id": 4}])
    second = corpus.archive_unfinished_summary(out_dir)
    assert second is not None
    assert second.name == "summary.unfinished_01.json"
    assert json.loads(first.read_text("utf-8"))["failed_workers"][0]["worker_id"] == 3

    # ... and the archiver is not a way around the gate: a finished or
    # verdict-less summary is refused here too, not quietly renamed.
    for keys in ({"run_finished": True}, {"rows": 0}):
        write_summary(out_dir, **keys)
        with pytest.raises(ValueError, match="refusing to move"):
            corpus.archive_unfinished_summary(out_dir)
        assert (out_dir / corpus.SUMMARY_NAME).exists(), "left exactly as found"


def test_a_crashed_run_resumes_and_its_crash_record_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-08-29 incident end to end, through ``main`` both times.

    Session 1 loses worker 1's PROCESS -- ``BrokenProcessPool``, the pool path
    ``run_worker`` cannot record from -- so the summary lands with
    ``run_finished: false``.  Session 2 passes ``--resume``, replays only the
    game no progress line claims, and banks a corpus of both games with its own
    ``open("x")``.

    Mutation caught: either half alone.  Without the gate change session 2
    raises "Nothing to resume"; without the archive it dies on
    ``FileExistsError`` writing the summary, after the games are already
    played.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", InlineExecutor)
    healthy = corpus.run_worker
    oom_kills: set[int] = {1}

    def pool_may_die(spec: corpus.WorkerSpec) -> dict[str, Any]:
        if spec.worker_id in oom_kills:
            raise BrokenProcessPool(
                "A process in the process pool was terminated abruptly",
            )
        return healthy(spec)

    monkeypatch.setattr(corpus, "run_worker", pool_may_die)
    out_dir = tmp_path / "run"
    argv = [
        "--out-dir", str(out_dir), "--games", "2", "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
    ]

    assert corpus.main(argv) == 1, "a run that lost a worker did not finish"
    crashed = json.loads((out_dir / corpus.SUMMARY_NAME).read_text("utf-8"))
    assert crashed["run_finished"] is False
    assert [f["exception_type"] for f in crashed["failed_workers"]] == [
        "BrokenProcessPool",
    ]
    assert crashed["games"] == 1, "worker 0's game is banked, worker 1's is not"
    assert "RUN DID NOT FINISH" in corpus.format_summary(crashed)

    oom_kills.clear()  # the OOM is over; the same corpus, resumed
    assert corpus.main([*argv, "--resume"]) == 0

    resumed = json.loads((out_dir / corpus.SUMMARY_NAME).read_text("utf-8"))
    assert resumed["run_finished"] is True
    assert resumed["resumed"] is True
    assert resumed["games"] == 2, "worker 1's game was replayed, worker 0's was not"
    assert resumed["games_this_session"] == 1
    assert resumed["games_completed_prior"] == 1
    assert resumed["failed_workers"] == []
    # The crash record is still on disk, under a name no consumer's inventory
    # globs, and still says what killed the run.
    kept = json.loads(
        (out_dir / "summary.unfinished_00.json").read_text("utf-8"),
    )
    assert kept["failed_workers"] == crashed["failed_workers"]
    # ⚑ No --json in this invocation, so nothing else was touched: exactly one
    # archived record, and no stray `.unfinished_` beside it.
    assert sorted(p.name for p in out_dir.glob("*.unfinished_*")) == [
        "summary.unfinished_00.json",
    ]


@pytest.mark.parametrize(
    ("name", "archived"),
    [
        ("summary.json", "summary.unfinished_00.json"),
        ("run02.json", "run02.unfinished_00.json"),
        ("report.tar.json", "report.tar.unfinished_00.json"),
        ("no_suffix", "no_suffix.unfinished_00"),
    ],
)
def test_an_archived_record_keeps_its_extension(
    tmp_path: Path, name: str, archived: str,
) -> None:
    """One naming rule for the summary and for any ``--json`` copy.

    The index is an INFIX rather than a suffix so an archived record is still a
    ``.json`` file to whatever opens one -- and so the name lands outside both
    ``shard_glob`` and the ``.jsonl.*`` suffixes a consumer's inventory globs.
    """
    (tmp_path / name).write_text("{}", encoding="utf-8")
    assert corpus.unfinished_archive_path(tmp_path / name).name == archived


def test_the_json_copy_is_freed_by_the_resume_and_written_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE SAME EXPLOSION ONE FUNCTION FURTHER OUT (review finding, P1).

    ``main`` writes the ``--json`` copy with its OWN ``open("x")`` after ``run``
    returns, so a crashed session that had ``--json`` left TWO files behind.
    Freeing only the in-directory summary would let the resume archive it,
    search for however many days, bank a correct corpus AND its ``summary.json``
    -- and then traceback on ``main``'s last line, handing automation an error
    code beside a complete corpus.  ⚑ ``--json`` is not in ``config_stamp``, so
    the drift gate cannot see it either; this is the only thing that can.

    Mutation caught: dropping the ``archive_json_copy_for_resume`` call from
    ``run`` -- the resumed session then raises ``FileExistsError`` on the aux
    path with both games already played and banked.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", InlineExecutor)
    healthy = corpus.run_worker
    oom_kills: set[int] = {1}

    def pool_may_die(spec: corpus.WorkerSpec) -> dict[str, Any]:
        if spec.worker_id in oom_kills:
            raise BrokenProcessPool("terminated abruptly")
        return healthy(spec)

    monkeypatch.setattr(corpus, "run_worker", pool_may_die)
    out_dir = tmp_path / "run"
    aux = tmp_path / "reports" / "burn.json"
    aux.parent.mkdir()
    argv = [
        "--out-dir", str(out_dir), "--games", "2", "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
        "--json", str(aux),
    ]

    assert corpus.main(argv) == 1
    assert json.loads(aux.read_text("utf-8"))["run_finished"] is False

    oom_kills.clear()
    assert corpus.main([*argv, "--resume"]) == 0

    # The fresh copy landed at the path the operator asked for ...
    assert json.loads(aux.read_text("utf-8"))["run_finished"] is True
    assert json.loads((out_dir / corpus.SUMMARY_NAME).read_text("utf-8"))[
        "run_finished"
    ] is True
    # ... and the crashed one is beside it, under one name, still readable.
    kept = aux.parent / "burn.unfinished_00.json"
    assert json.loads(kept.read_text("utf-8"))["run_finished"] is False
    assert sorted(p.name for p in aux.parent.iterdir()) == [
        "burn.json", "burn.unfinished_00.json",
    ]


def test_a_refused_resume_moves_neither_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE ORDERING GUARANTEE: nothing moves until every refusal has passed.

    Both archive steps sit AFTER ``refuse_resume_config_drift`` inside ``run``,
    which is the only reason a resume that is turned away leaves the directory
    it was pointed at exactly as it found it.  Move either one up into ``main``
    -- the obvious place, since ``main`` owns the ``--json`` write -- and a
    refused resume silently renames the operator's files on its way out.

    Mutation caught: archiving before the drift check.  Both digests below then
    hold a file that is no longer there.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=MATE_GAME_SCRIPT)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(MATE_GAME_FEN, tmp_path),
    )
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", InlineExecutor)
    healthy = corpus.run_worker
    oom_kills: set[int] = {1}

    def pool_may_die(spec: corpus.WorkerSpec) -> dict[str, Any]:
        if spec.worker_id in oom_kills:
            raise BrokenProcessPool("terminated abruptly")
        return healthy(spec)

    monkeypatch.setattr(corpus, "run_worker", pool_may_die)
    out_dir = tmp_path / "run"
    aux = tmp_path / "burn.json"
    argv = [
        "--out-dir", str(out_dir), "--games", "2", "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
        "--json", str(aux),
    ]
    assert corpus.main(argv) == 1
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out_dir / corpus.SUMMARY_NAME, aux)
    }

    oom_kills.clear()
    with pytest.raises(ValueError, match=r"temp_high: 0\.01 -> 0\.5"):
        corpus.main([
            *argv[:argv.index("--temp-high") + 1], "0.5",
            *argv[argv.index("--temp-high") + 2:], "--resume",
        ])

    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    } == before, "a refused resume must not touch a byte"
    assert list(tmp_path.glob("*.unfinished_*")) == []
    assert list(out_dir.glob("*.unfinished_*")) == []


def test_a_resumed_run_summarises_the_whole_corpus_not_the_last_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``summary.json`` after a resume must inventory BOTH sessions.

    Mutation caught: leaving ``shards``/``rows``/``games`` session-scoped.  The
    summary then reads like a complete record of a corpus twice its size, and
    a consumer that iterates ``summary["shards"]`` silently trains on half of
    it.  The rows carry the ORIGINAL stamp too -- one corpus, one
    ``config_sha256``.
    """
    monkeypatch.setattr(
        corpus, "StockfishUCI",
        lambda *_a, **_kw: uci_double(ScriptedEngine(preferred=CAPTURE_CHAIN_SCRIPT)),
    )
    monkeypatch.setattr(
        corpus, "build_opening_config",
        lambda _spec: fen_opening(CAPTURE_CHAIN_FEN, tmp_path),
    )
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", InlineExecutor)
    out_dir = tmp_path / "run"
    argv = [
        "--out-dir", str(out_dir), "--workers", "2",
        "--syzygy-path", str(SMOKE_SYZYGY or corpus.REPO_ROOT),
        "--temp-high", "0.01", "--temp-low", "0.01", "--nice", "0",
        "--max-plies", "1", "--staircase", RESUME_STAIRCASE,
    ]
    first = corpus.run(corpus.build_parser().parse_args([*argv, "--games", "2"]))
    assert first["resumed"] is False
    assert first["games"] == 2
    (out_dir / corpus.SUMMARY_NAME).unlink()  # the kill

    second = corpus.run(
        corpus.build_parser().parse_args([*argv, "--games", "2", "--resume"]),
    )

    assert second["resumed"] is True
    assert second["games_this_session"] == 0, "both games were already banked"
    assert second["games"] == 2, "the corpus still holds two games"
    assert second["rows"] == first["rows"]
    assert second["games_completed_prior_by_worker"] == {"0": 1, "1": 1}
    assert [shard["path"] for shard in second["shards"]] == [
        shard["path"] for shard in first["shards"]
    ]
    assert second["config_sha256"] == first["config_sha256"]
    rows = [
        row for shard in second["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert rows
    assert all(
        row["run"]["config_sha256"] == first["config_sha256"] for row in rows
    )
    assert "RESUMED:" in corpus.format_summary(second)


# ── the real engine ──────────────────────────────────────────────────────────


#: ⚑ A REAL 3-FOLD, built by shuffling knights.  Black is a whole rook up, so a
#: draw and a search are numbers nobody can confuse: ``c6b8`` from the position
#: after these moves completes the THIRD occurrence of the root.
REPETITION_ROOT = "1n4k1/8/8/8/8/8/r7/1N4K1 w - - 0 1"
REPETITION_MOVES = "b1c3 b8c6 c3b1 c6b8 b1c3 b8c6 c3b1"
REPETITION_MOVE = "c6b8"


@pytest.mark.skipif(find_stockfish() is None, reason="no Stockfish binary")
def test_real_stockfish_scores_the_repetition_as_a_draw_only_with_the_window() -> None:
    """⚑⚑ THE LABEL-PATH TAKE-EFFECT PROOF, against the real engine.

    Same position, same ``searchmoves``, same depth -- only the ``position``
    line differs.  With the window the engine sees its own repetition and
    returns the DRAW score; without it the same move is scored as the material
    count, which for a rook-up side is hundreds of centipawns away.  That gap is
    the label error the history-blind form was banking.

    ⚑ The blind arm is built as a ``RowHistory`` with an EMPTY move list rather
    than by calling a removed code path, so both arms run through exactly the
    same sender and the only difference on the wire is the window.
    """
    board = chess.Board(REPETITION_ROOT)
    for uci in REPETITION_MOVES.split():
        board.push_uci(uci)
    after = board.copy(stack=True)
    after.push_uci(REPETITION_MOVE)
    assert after.is_repetition(3), "the fixture no longer 3-folds"

    binary = find_stockfish()
    assert binary is not None
    engine = StockfishUCI(binary, multipv=1, hash_mb=16, nice=15)
    try:
        searcher = corpus.StaircaseSearcher(
            engine=engine,
            staircase=corpus.parse_staircase("all:12"),
            cp_slope=gen.NNUE_CP_SLOPE,
            cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        )
        aware = corpus.history_for(board)
        blind = corpus.RowHistory(
            fen=aware.fen, root_fen=aware.fen, uci=(),
            reason=corpus.HISTORY_ROOT_GAME_START,
        )
        assert " moves " in corpus.position_command(aware)
        assert " moves " not in corpus.position_command(blind)
        scores = {
            name: score_of_last_block(
                searcher.stream(
                    window, depth=12, multipv=1, searchmoves=[REPETITION_MOVE],
                ),
            )
            for name, window in (("aware", aware), ("blind", blind))
        }
    finally:
        engine.close()

    assert scores["aware"] == 0, scores
    assert scores["blind"] >= 200, scores


def score_of_last_block(lines: Sequence[str]) -> int:
    """The cp of the deepest ``multipv 1`` line in a stream, through the
    generator's OWN parser rather than a second one."""
    parse = corpus.parse_depth_blocks(lines, expected_lines=1)
    block, _ = corpus.deepest_block_with_width(parse.blocks, want=1)
    return round(block.lines[0].effective_cp)


@pytest.mark.skipif(find_stockfish() is None, reason="no Stockfish binary")
def test_real_stockfish_emits_exactly_one_line_per_rank_per_depth() -> None:
    """The measured stream shape, against the binary this checkout publishes.

    The whole per-depth block structure rests on it, and a Stockfish upgrade
    that started emitting twice per rank would silently turn every block into an
    anomaly.  Small budgets so the test costs a second, not a minute.
    """
    binary = find_stockfish()
    assert binary is not None
    engine = StockfishUCI(binary, multipv=1, hash_mb=16, nice=15)
    try:
        searcher = corpus.StaircaseSearcher(
            engine=engine,
            staircase=corpus.parse_staircase("all:4,4:6,2:8"),
            cp_slope=gen.NNUE_CP_SLOPE,
            cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        )
        search = searcher.search_position(chess.Board())
    finally:
        engine.close()

    assert [p.width_realized for p in search.phases] == [20, 4, 2]
    assert [p.depth_requested for p in search.phases] == [4, 6, 8]
    for phase in search.phases:
        assert phase.parse.blocks, "every phase must produce depth blocks"
        for block in phase.parse.blocks:
            assert len(block.lines) == phase.width_realized
            assert block.complete
            # Either one clean iteration or the measured end-of-search flush;
            # nothing else has ever been observed here.
            assert block.emissions in (
                phase.width_realized, 2 * phase.width_realized,
            )
        # ⚑ The verdict is that NO re-emission disagreed and every emission-count
        # violation was the flush -- not that the counts were pristine. See the
        # module docstring's 2026-08-27 measurement.
        assert phase.parse.re_emissions_disagreeing == 0
        assert (
            phase.parse.emission_count_violations
            == phase.parse.duplicate_iteration_flushes
        )
    assert search.value_full_width is True
    assert len(search.values) == 20
