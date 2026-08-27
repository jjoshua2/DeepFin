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
import threading
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts import audit_label_candidates as gate
from scripts import gen_random_selfplay_shards as gen
from scripts import gen_sf_rooted_corpus as corpus
from tests.stockfish_binary import find_stockfish

# ── the scripted engine ──────────────────────────────────────────────────────

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


SMOKE_SYZYGY = smoke_syzygy()

MATE_GAME_FEN = "6k1/5ppp/8/3n4/8/8/7P/R3K3 b - - 0 1"
MATE_GAME_SCRIPT = ("d5c3", "a1a8")

CAPTURE_CHAIN_FEN = "7k/8/6n1/8/1n1b4/2R5/1Q6/K7 w - - 0 1"
CAPTURE_CHAIN_SCRIPT = ("b2b4", "d4c3", "b4c3")


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
        final_depth_offset: int = 0,
    ) -> None:
        self.commands: list[str] = []
        self.multipv = int(multipv)
        self.preferred = tuple(preferred)
        #: Subtracted from every requested depth, to script a search that stops
        #: short of its ask.
        self.final_depth_offset = int(final_depth_offset)
        self.fen = chess.STARTING_FEN
        self._pending: list[str] = []

    # -- driver-facing surface --------------------------------------------
    def send(self, cmd: str) -> None:
        self.commands.append(cmd)
        if cmd in ("isready",):
            self._pending.append("readyok\n")
        elif cmd.startswith("setoption name MultiPV value "):
            self.multipv = int(cmd.split()[-1])
        elif cmd.startswith("position fen "):
            self.fen = cmd[len("position fen "):]
        elif cmd.startswith("go "):
            self._pending.extend(self._reply_to(cmd))

    def readline(self, _deadline: float) -> str:
        return self._pending.pop(0)

    # -- engine behaviour --------------------------------------------------
    def score_of(self, uci: str) -> int:
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
        # Ordered like a real MultiPV list: rank 1 is the best score.  Ties
        # break on the uci so the ordering is total.
        root.sort(key=lambda uci: (-self.score_of(uci), uci))
        root = root[: max(1, self.multipv)]
        lines: list[str] = []
        for d in range(1, max(1, depth - self.final_depth_offset) + 1):
            for rank, mv in enumerate(root, start=1):
                lines.append(
                    f"info depth {d} seldepth {d + 2} multipv {rank} "
                    f"score cp {self.score_of(mv)} nodes {1000 * d + rank} "
                    f"pv {mv}\n",
                )
        lines.append(f"bestmove {root[0] if root else '0000'}\n")
        return lines

    # -- assertion helpers -------------------------------------------------
    @property
    def go_lines(self) -> list[str]:
        return [c for c in self.commands if c.startswith("go ")]

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


def searcher_for(
    engine: ScriptedEngine, *, staircase: str = corpus.DEFAULT_STAIRCASE, **attrs: Any,
) -> corpus.StaircaseSearcher:
    return corpus.StaircaseSearcher(
        engine=uci_double(engine, **attrs),
        staircase=corpus.parse_staircase(staircase),
        cp_slope=gen.NNUE_CP_SLOPE,
        cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
    )


def worker_spec(tmp_path: Path, **overrides: Any) -> corpus.WorkerSpec:
    values: dict[str, Any] = {
        "worker_id": 0,
        "game_ids": (0,),
        "out_dir": tmp_path,
        "sf_binary": "/nonexistent/stockfish",
        "sf_hash_mb": 64,
        "syzygy_path": "/nonexistent/syzygy",
        "staircase": corpus.DEFAULT_STAIRCASE,
        "seed": 7,
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
    searcher = searcher_for(engine, staircase=spec.staircase)
    dedup = corpus.DedupStats()
    outcome = corpus.play_game(
        spec=spec, searcher=searcher,
        opening_cfg=fen_opening(fen, tmp_path),
        game_id=spec.game_ids[0], cache={}, dedup=dedup,
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
            chess.STARTING_FEN, depth=5, multipv=1, searchmoves=["e7e5"],
        )


def test_selection_reads_the_full_width_scout_not_the_narrowed_rung() -> None:
    engine = ScriptedEngine()
    searcher = searcher_for(engine)
    board = chess.Board(MATE_GAME_FEN)

    search = searcher.search_position(board)

    assert len(search.values) == board.legal_moves.count()
    assert search.value_depth == 9
    assert search.value_full_width is True
    assert searcher.stats.selection_not_full_width == 0


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
    values = (corpus.PvLine(rank=1, move="e2e4", effective_cp=50.0, nodes=1),)
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
    q = np.array([0.4, 0.35, 0.3])
    first = corpus.gumbel_choice(
        q, temp=1.0,
        rng=corpus.selection_rng(seed=5, worker_id=2, game_id=9, ply=11),
    )
    again = corpus.gumbel_choice(
        q, temp=1.0,
        rng=corpus.selection_rng(seed=5, worker_id=2, game_id=9, ply=11),
    )
    other_worker = corpus.gumbel_choice(
        q, temp=1.0,
        rng=corpus.selection_rng(seed=5, worker_id=3, game_id=9, ply=11),
    )
    assert first == again
    # Different workers draw different noise, so two workers on one position do
    # not play the same move.
    assert isinstance(other_worker, int)


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
    cache: dict[str, tuple[corpus.PvLine, ...]] = {}
    opening = fen_opening(MATE_GAME_FEN, tmp_path)

    first = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup,
    )
    go_lines_after_first = len(engine.go_lines)
    searched_after_first = searcher.stats.positions
    seen_after_first = dict(dedup.first_seen)

    second = corpus.play_game(
        spec=spec, searcher=searcher, opening_cfg=opening,
        game_id=0, cache=cache, dedup=dedup,
    )

    assert len(first.rows) == 2
    assert second.rows == [], "a cache-served position is never re-banked"
    assert len(engine.go_lines) == go_lines_after_first, "no second search"
    assert searcher.stats.positions == searched_after_first
    assert dict(dedup.first_seen) == seen_after_first
    assert sum(dedup.hits.values()) == 2
    # Cache-served selection is the same selection: same plies, same result.
    assert (second.plies, second.result_pgn) == (first.plies, first.result_pgn)


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
    if path.suffix == ".zst":
        module = corpus.zstandard_module()
        assert module is not None
        with open(path, "rb") as raw, module.ZstdDecompressor().stream_reader(
            raw,
        ) as stream:
            text = stream.read().decode("utf-8")
    else:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read()
    return [json.loads(line) for line in text.splitlines() if line]


def test_a_shard_roundtrips_and_rotates_at_shard_rows(tmp_path: Path) -> None:
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=3, shard_rows=2)
    rows = [{"schema": corpus.ROW_SCHEMA, "ply": i} for i in range(5)]
    for row in rows:
        writer.write(row)
    writer.close()

    assert [shard["rows"] for shard in writer.shards] == [2, 2, 1]
    assert [Path(shard["path"]).name for shard in writer.shards] == [
        f"w03-{i:05d}{writer.suffix}" for i in range(3)
    ]
    read_back = [
        row for shard in writer.shards for row in read_shard(Path(shard["path"]))
    ]
    assert read_back == rows
    assert all(row["schema"] == corpus.ROW_SCHEMA for row in read_back)


def test_the_writer_falls_back_to_gzip_without_zstandard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "zstandard_module", lambda: None)
    writer = corpus.ShardWriter(out_dir=tmp_path, worker_id=0, shard_rows=10)
    writer.write({"schema": corpus.ROW_SCHEMA})
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
        "schema", "run", "fen", "dedup_key", "worker_id", "game_id", "ply",
        "stm", "piece_count", "game_phase", "played_move", "selection",
        "phases", "result", "result_pgn", "adjudication",
    }
    assert row["run"][corpus.KEY_TT_CARRIED] is True
    assert row["run"]["config_sha256"] == "0" * 64
    assert row["played_move"] in {m.uci() for m in chess.Board(row["fen"]).legal_moves}
    assert len(row["phases"]) == 3
    phase = row["phases"][0]
    assert set(phase) == {
        "index", "width_requested", "width_realized", "depth_requested",
        "searchmoves", "per_depth", "nodes_at_depth", "anomalies",
    }
    assert set(phase["per_depth"][0]) == {
        "depth", "complete", "emissions", "nodes_at_depth", "lines",
    }
    # (rank, move, effective_cp, cumulative nodes) -- the lowest-level thing the
    # search reported, banked so a re-analysis is a re-read.
    assert len(phase["per_depth"][0]["lines"][0]) == 4
    assert json.loads(json.dumps(row, sort_keys=True)) == row


# ── run assembly ─────────────────────────────────────────────────────────────


def test_games_are_dealt_so_no_two_workers_share_a_game_id() -> None:
    buckets = corpus.split_games(7, 3)
    assert buckets == [[0, 3, 6], [1, 4], [2, 5]]
    assert corpus.split_games(2, 5) == [[0], [1]], "no empty workers"


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
    assert realized["tt_cleared_per_game"] is True
    assert realized["cp_slope"] == gen.NNUE_CP_SLOPE


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


def test_the_summary_merges_workers_without_losing_a_counter() -> None:
    workers = [
        {
            "worker_id": 0, "games": 2, "rows": 5, "plies_total": 40,
            "terminations": {"natural": 2}, "adjudications": {"none": 2},
            "opening_sources": {"start": 2}, "adjudication_unavailable_plies": 1,
            "dedup": {
                "positions_first_seen": 8, "dup_hits": 2,
                "first_seen_by_phase": {"opening": 8, "middlegame": 0, "endgame": 0},
                "dup_hits_by_phase": {"opening": 2, "middlegame": 0, "endgame": 0},
            },
            "search": {
                "positions_searched": 8, "searches": 24, "search_s": 4.0,
                "anomalies": {"re_emissions": 1, "duplicate_iteration_flushes": 2},
                "nodes_by_phase": {
                    "0": {"n": 8, "total": 80, "min": 5, "max": 15,
                          "log2_buckets": {"3": 8}},
                },
            },
            "shards": [{"path": "a", "rows": 5, "codec": "zstd"}],
            "realized": {"sf_hash_mb": 64},
        },
        {
            "worker_id": 1, "games": 1, "rows": 3, "plies_total": 20,
            "terminations": {"syzygy": 1}, "adjudications": {"syzygy_wdl_2": 1},
            "opening_sources": {"start": 1}, "adjudication_unavailable_plies": 0,
            "dedup": {
                "positions_first_seen": 4, "dup_hits": 0,
                "first_seen_by_phase": {"opening": 4, "middlegame": 0, "endgame": 0},
                "dup_hits_by_phase": {"opening": 0, "middlegame": 0, "endgame": 0},
            },
            "search": {
                "positions_searched": 4, "searches": 12, "search_s": 2.0,
                "anomalies": {"re_emissions": 0, "bound_lines": 3},
                "nodes_by_phase": {
                    "0": {"n": 4, "total": 20, "min": 2, "max": 9,
                          "log2_buckets": {"3": 3, "4": 1}},
                },
            },
            "shards": [{"path": "b", "rows": 3, "codec": "zstd"}],
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

    assert summary["rows"] == 8
    assert summary["games"] == 3
    assert summary["terminations"] == {"natural": 2, "syzygy": 1}
    assert summary["adjudication_unavailable_plies"] == 1
    assert summary["dedup"]["dup_hits"] == 2
    assert summary["dedup"]["positions_visited"] == 14
    assert summary["search"]["anomalies"] == {
        "re_emissions": 1, "bound_lines": 3, "duplicate_iteration_flushes": 2,
    }
    assert summary["search"]["s_per_position"] == pytest.approx(0.5)
    nodes = summary["search"]["nodes_by_phase"]["0"]
    assert (nodes["n"], nodes["total"], nodes["min"], nodes["max"]) == (12, 100, 2, 15)
    assert nodes["log2_buckets"] == {"3": 11, "4": 1}
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
    # Game 1 replays game 0's positions, so every one of them is cache-served.
    assert result["rows"] == 2
    assert result["dedup"]["dup_hits"] == 2
    assert result["terminations"] == {"natural": 2}
    assert result["realized"]["sf_hash_mb"] == 64
    assert result["realized"]["max_plies"] == 50
    assert result["realized"]["seed"] == 7
    assert result["realized"]["opening_book_path"] is None

    rows = [
        row for shard in result["shards"] for row in read_shard(Path(shard["path"]))
    ]
    assert len(rows) == 2
    assert [row["result"] for row in rows] == [-1.0, 1.0]


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
    assert summary["config_requested"]["games"] == 1
    assert summary["config_realized_by_worker"]["0"][corpus.KEY_TT_CARRIED] is True
    assert summary["banked_rows_min_piece_count"] == corpus.MIN_BANKED_PIECES
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


# ── the real engine ──────────────────────────────────────────────────────────


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
