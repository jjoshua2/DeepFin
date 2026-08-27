"""The audit-first kill/keep gate for NNUE bootstrap-label candidates.

Every test here is aimed at one of the four ways this tool can be wrong while
looking right, because none of them raises on its own:

* the SEAT -- a label built on the un-negated child value ranks the root's
  moves backwards and is still a well-formed distribution over the right moves;
* the TIE RULE -- scoring against a single deep-SF best move instead of the
  score-tied set measures MultiPV ordering, not move quality;
* the CENSORING RULE -- imputing zero regret for a move the deep MultiPV never
  listed turns an unknown cost into a perfect one;
* the BANNED SOURCE -- the shallow-SF sidecar beside the audit set ran on a
  dirty shared transposition table and must never be read.

And three more that a review found the tool could be wrong about while every
one of the tests above stayed green, because each is a check or a value that
LOOKS present and does not act:

* the UNWIRED GUARD -- ``_refuse_move_set_drift`` was exercised only by a direct
  call, so deleting its call site in ``run`` broke nothing here;
* the INERT SLICE -- ``--limit`` was consumed nowhere a test could see, and
  ``provenance.limit`` echoed the REQUEST, so a skipped slice left no trace;
* the KNOB WITH NO CONSUMER -- ``--nnue-cp-per-unit``, ``--nnue-pack``,
  ``--bank-observations`` and the DAG store bounds were accepted on runs whose
  selected arms read none of them, and the first was stamped into the report.

The mutants that pin them are recorded in the PR description; each was made,
watched to fail these tests, and reverted.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.eval.audit import legal_full_indices
from chess_anti_engine.nnue import _nnue_ext
from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from scripts import audit_label_candidates as gate
from scripts import audit_targets
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_shadow_label_readout as shadow
from tests.test_nnue_gumbel_readout import FakeExt
from tests.test_nnue_native_eval import write_synthetic_pack

#: A quiet middlegame-ish position with a wide, unambiguous move list.
_ROOT_FEN = "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1"
#: White mates in one with Ra1a8.
_MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
#: White can stalemate with several queen moves, or mate with others.
_STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
#: The SAME position with BLACK to move -- i.e. NOT side-to-move canonical.
#: `eval.audit.legal_full_indices` encodes every audit row as white
#: (`uci_to_policy_index(uci, True)`) while `CBoard` encodes from the side to
#: move, so the two disagree about EVERY action id here: measured 37 legal moves
#: each, symmetric difference 74, zero overlap.
_BLACK_TO_MOVE_FEN = "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R b KQkq - 0 1"
#: Three distinct white-to-move rows, for the ``--limit`` slice.
_THREE_FENS = (
    _ROOT_FEN,
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    _MATE_IN_ONE_FEN,
)


@pytest.fixture(autouse=True)
def _extension_defaults():
    """Leave the compiled globals as they were found.

    `set_arm_config` / `fastq_set_config` are PROCESS-wide, and the knob tests
    below open contexts at non-default settings; a test that inherited another's
    configuration would assert the wrong knobs by accident.
    """
    def restore() -> None:
        _nnue_ext.set_arm_config(
            _nnue_ext.RESOLVER_MAX_DEPTH,
            _nnue_ext.QSEARCH_MAX_PLY,
            _nnue_ext.QSEARCH_CHECK_PLIES,
            _nnue_ext.QSEARCH_DAG_NODE_CAP,
        )
        _nnue_ext.fastq_set_config(
            _nnue_ext.FASTQ_MAX_QPLY,
            _nnue_ext.FASTQ_NODE_CAP,
            _nnue_ext.FASTQ_DELTA_MARGIN,
            _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
        )

    restore()
    yield
    restore()


@pytest.fixture(scope="module")
def synthetic_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real-layout, all-zero pack.

    ⚑ IT IS ENOUGH FOR EVERYTHING IT IS USED FOR HERE AND NOT FOR MORE. Zero
    weights make every non-terminal position evaluate to 0, so the arms are
    indistinguishable on quality -- these tests use it for wiring, for the
    requested-vs-realized knob proof, and for TERMINAL values, which the C
    resolver produces from the position and not from the weights.
    """
    path = tmp_path_factory.mktemp("audit_gate") / "tiny.pack"
    write_synthetic_pack(path)
    return path


# ── audit-set fixtures ───────────────────────────────────────────────────────


def audit_row(
    fen: str,
    *,
    listed: list[tuple[str, int]],
    key: str | None = None,
    phase: int = 1,
    source: int = 0,
) -> str:
    """One frozen-audit JSONL line, in ``build_audit_set.py``'s own shape."""
    return json.dumps({
        "key": key or hashlib.sha1(fen.encode()).hexdigest(),
        "fen": fen,
        "phase": phase,
        "source": source,
        "multipv": [{"move": m, "cp": cp, "mate": None, "wdl": None}
                    for m, cp in listed],
        "wdl": [333.0, 334.0, 333.0],
        "bestmove": listed[0][0],
        "nodes": 1_000_000,
        "depth": 40,
    })


def write_audit_set(path: Path, rows: list[str]) -> Path:
    path.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return path


def legal_ucis_of(fen: str) -> list[str]:
    return legal_full_indices(chess.Board(fen))[0]


# ── fake arms and engines ────────────────────────────────────────────────────


def _placement(board: CBoard) -> str:
    """A child's piece placement -- unique among the siblings of one root."""
    return board.fen().split()[0]


class ScriptedChildArm:
    """A ``ChildArm`` whose value for a board is looked up by its placement.

    The values are the CHILD's own seat, exactly as a real arm returns them, so
    ``probe_root`` supplies the negation and this object can pin the convention
    without an NNUE pack or a Stockfish process.
    """

    def __init__(self, arm: str, values: dict[str, float]) -> None:
        self.arm = arm
        self._values = values
        self._cost = gate.ArmCost()
        self.positions = 0

    def evaluate(
        self, boards: list[CBoard], *, role: str, cluster: tuple[int, int] | None,
    ) -> np.ndarray:
        del cluster
        self._cost.add(role, len(boards), 0.0)
        return np.asarray(
            [self._values.get(_placement(b), 0.0) for b in boards],
            dtype=np.float64,
        )

    def begin_position(self, *, reset_memo: bool) -> None:
        del reset_memo
        self.positions += 1

    def cost(self) -> gate.ArmCost:
        return self._cost

    def stamp(self) -> dict[str, Any]:
        return {"kind": "scripted", "construction": "oneply_child"}

    def close(self) -> None:
        return None


class FakeEngine:
    """A ``SearchEngine`` returning a scripted result, keyed by FEN placement."""

    hash_mb: int | None = 16
    multipv = 1
    read_timeout_s = 5.0

    def __init__(self, results: dict[str, StockfishResult] | None = None) -> None:
        self._results = dict(results or {})
        self.searched: list[str] = []
        self.new_games = 0
        self.closed = False

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:
        del nodes
        self.searched.append(fen)
        key = fen.split()[0]
        if key not in self._results:
            raise AssertionError(f"unscripted search for {fen!r}")
        return self._results[key]

    def new_game(self) -> None:
        self.new_games += 1

    def close(self) -> None:
        self.closed = True


class RefusingEngine(FakeEngine):
    """Any search at all is a failure -- for the terminal-child parity test."""

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:
        del nodes
        raise AssertionError(
            f"the arm searched {fen!r}; a terminal child must be resolved from "
            "the board, because Stockfish reports a checkmated position as "
            "`score mate 0`, which through the mate home is a WIN for the side "
            "that was just mated",
        )


def sf_result(*, cp: int | None = None, mate: int | None = None) -> StockfishResult:
    return StockfishResult(
        bestmove_uci="0000", wdl=None, pvs=[], cp=cp, mate=mate,
        nodes=512, depth=8,
    )


def sf_child_arm(engine: FakeEngine, *, nodes: int = 512) -> gate.StockfishCandidateArm:
    spec = gate.parse_sf_arm(f"sf-{nodes}")
    assert spec is not None
    return gate.StockfishCandidateArm(
        spec=spec, engine=cast(gate.SearchEngine, engine),
        cp_slope=gen.NNUE_CP_SLOPE, cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        fresh_per_position=False,
    )


def rooted_arm(
    name: str = "sfroot-2048-mpv20", *, engine: FakeEngine | None = None,
) -> gate.RootedStockfishArm:
    spec = gate.parse_sf_arm(name)
    assert spec is not None
    return gate.RootedStockfishArm(
        spec=spec,
        # The rooted arm drives the UCI stream itself, so its engine is typed as
        # the real class. Every test here replaces `search_lines`, the only
        # method that touches it, so the stand-in never has to satisfy the UCI
        # protocol surface -- hence the `object` hop rather than a structural
        # claim this fake does not make.
        engine=cast(StockfishUCI, cast(object, engine or FakeEngine())),
        cp_slope=gen.NNUE_CP_SLOPE, cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        fresh_per_position=False,
    )


def gate_config(
    audit_set: Path, arms: tuple[str, ...], **overrides: Any,
) -> gate.GateConfig:
    """A config for a run whose arms are injected rather than opened."""
    values: dict[str, Any] = {
        "audit_set": audit_set,
        "pack": Path(),
        "arms": arms,
        "native_configs": {},
        "sf_specs": {},
        "static_resolver_max_depth": None,
        "limit": 0,
        "oneply_sigma": shadow.oneply_sigma_default(),
        "cp_per_internal_unit": gen.NNUE_CP_PER_INTERNAL_UNIT,
        "cp_slope": gen.NNUE_CP_SLOPE,
        "cp_draw_width": gen.NNUE_CP_DRAW_WIDTH,
        "dag_max_nodes": 0,
        "dag_reset_every": 1,
        "sf_binary": None,
        "sf_hash_mb": 16,
        "sf_fresh_per_position": False,
        "nice": 0,
        "run_id": "test",
    }
    values.update(overrides)
    return gate.GateConfig(**values)


def run_with_arms(
    monkeypatch: pytest.MonkeyPatch,
    cfg: gate.GateConfig,
    arms: list[ScriptedChildArm],
) -> dict[str, Any]:
    monkeypatch.setattr(
        gate, "open_arms",
        lambda _cfg, *, pack_sha: (cast(list[gate.ChildArm], list(arms)), []),
    )
    return gate.run(cfg)


# ── 1. the seat ──────────────────────────────────────────────────────────────


def test_the_oneply_label_is_read_from_the_root_movers_seat() -> None:
    """The chosen move is the one whose CHILD is WORST for the child's mover.

    ⚑ THE MUTANT THIS EXISTS FOR is dropping ``probe_root``'s negation. The
    resulting target is a perfectly well-formed distribution over the right move
    set -- it simply ranks the root's moves backwards -- so nothing downstream
    raises and every metric in the report stays in range.
    """
    board = chess.Board(_ROOT_FEN)
    ucis, idxs = legal_full_indices(board)
    assert len(ucis) > 8
    values: dict[str, float] = {}
    for i, uci in enumerate(ucis):
        child = board.copy()
        child.push(chess.Move.from_uci(uci))
        values[child.fen().split()[0]] = float(i)
    arm = ScriptedChildArm("scripted", values)

    probe = shadow.probe_root(
        CBoard.from_board(board), observers=[arm], cluster=(0, 0),
    )
    label = gate.child_label(
        probe, arm="scripted", legal_ucis=ucis, legal_idxs=idxs,
        sigma=shadow.oneply_sigma_default(),
    )
    # ucis[0]'s child scored 0.0 -- the lowest value for the CHILD's mover, so
    # the best move for the ROOT's mover. ucis[-1] is its exact opposite.
    assert label.chosen == ucis[0]
    assert label.chosen != ucis[-1]
    assert label.values[ucis[0]] == pytest.approx(0.0)
    assert label.values[ucis[-1]] == pytest.approx(-float(len(ucis) - 1))


# ── 2. the tie-inclusive top1 set ────────────────────────────────────────────


def test_a_score_tied_deep_sf_best_pair_credits_either_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two moves at the same best cp: an arm picking EITHER agrees.

    ⚑ THE MUTANT THIS EXISTS FOR is building the reference set by SLICE
    (``ranked[:1]``) instead of by SCORE. A sliced set makes the metric report
    which co-best move the deep MultiPV happened to list first, which is
    tie-breaking, not move quality.
    """
    ucis = legal_ucis_of(_ROOT_FEN)
    first, second, third = ucis[0], ucis[1], ucis[2]
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(first, 50), (second, 50), (third, 10)]),
    ])
    # Each arm is scripted so its OWN argmax is one of the two co-best moves:
    # the child value is negated into the root's seat, so the move to prefer is
    # the one whose child scores lowest.
    def picking(uci: str) -> ScriptedChildArm:
        board = chess.Board(_ROOT_FEN)
        values: dict[str, float] = {}
        for candidate in ucis:
            child = board.copy()
            child.push(chess.Move.from_uci(candidate))
            values[child.fen().split()[0]] = -1.0 if candidate == uci else 1.0
        return ScriptedChildArm(f"picks_{uci}", values)

    arm_a, arm_b = picking(first), picking(second)
    report = run_with_arms(
        monkeypatch,
        gate_config(audit, (arm_a.arm, arm_b.arm)),
        [arm_a, arm_b],
    )
    for arm, expected in ((arm_a, first), (arm_b, second)):
        cell = report["arms"][arm.arm]
        assert cell["top1_agree_rate"] == 1.0, f"{arm.arm} picked {expected}"
        assert cell["top1_regret_cp_mean"] == 0.0
    assert audit_targets.sf_reference_sets({first: 50.0, second: 50.0, third: 10.0})[0] == {
        first, second,
    }


# ── 3. the censoring rule ────────────────────────────────────────────────────


def test_an_unlisted_choice_is_censored_at_the_worst_listed_regret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arm choosing a move deep SF never listed gets a FLOOR, not a zero.

    ⚑ THE MUTANT THIS EXISTS FOR is imputing 0 regret for an unlisted move --
    the most flattering possible reading of a move whose true cost is unknown,
    and the one that would make a weak labeller look best.
    """
    ucis = legal_ucis_of(_ROOT_FEN)
    listed = [(ucis[0], 100), (ucis[1], 60), (ucis[2], 20)]
    unlisted = ucis[-1]
    audit = write_audit_set(
        tmp_path / "audit.jsonl", [audit_row(_ROOT_FEN, listed=listed)],
    )
    board = chess.Board(_ROOT_FEN)
    values: dict[str, float] = {}
    for candidate in ucis:
        child = board.copy()
        child.push(chess.Move.from_uci(candidate))
        values[child.fen().split()[0]] = -1.0 if candidate == unlisted else 1.0
    arm = ScriptedChildArm("picks_unlisted", values)
    dump = tmp_path / "rows.jsonl"
    report = run_with_arms(
        monkeypatch,
        gate_config(audit, (arm.arm,), dump_per_position=dump),
        [arm],
    )
    cell = report["arms"][arm.arm]
    # best 100, worst LISTED 20 -> the floor is 80cp, not 0 and not the true
    # (unknown) cost of the unlisted move.
    assert cell["top1_regret_cp_mean"] == pytest.approx(80.0)
    assert cell["top1_move_unlisted_rate"] == pytest.approx(1.0)
    assert cell["top1_move_unlisted_positions"] == 1.0
    row = json.loads(dump.read_text(encoding="utf-8").splitlines()[0])
    assert row["arm"][arm.arm]["move"] == unlisted
    assert row["arm"][arm.arm]["top1_move_listed_by_deep_sf"] is False
    assert report["metric_definitions"]["censoring_rule_for_unlisted_moves"].count(
        "WORST LISTED",
    ) == 1


# ── 4. one cp mapping, shared by every arm ───────────────────────────────────


def test_the_sf_arm_and_the_nnue_arm_share_one_cp_mapping_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Replacing ONE object moves both answers -- proof by execution.

    A test that merely read both call sites would pass on two identical copies,
    and two copies of a mapping is precisely how the Stockfish arm would drift
    away from the label rule it exists to hold fixed.
    """
    seen: list[float] = []
    real = gen.cp_to_wdl_array

    def spy(eff_cp: np.ndarray, *, slope: float, draw_width_cp: float) -> np.ndarray:
        seen.append(float(np.asarray(eff_cp, dtype=np.float64).ravel()[0]))
        return real(eff_cp, slope=slope, draw_width_cp=draw_width_cp)

    monkeypatch.setattr(gen, "cp_to_wdl_array", spy)
    gate.q_from_effective_cp(
        np.asarray([250.0]), slope=gen.NNUE_CP_SLOPE,
        draw_width_cp=gen.NNUE_CP_DRAW_WIDTH,
    )
    source = gen.NnueArmValueSource(
        arm=gen.VALUE_SOURCE_NNUE_STATIC,
        pack=tmp_path / "unused.pack",
        cp_per_internal_unit=1.0,
        cp_slope=gen.NNUE_CP_SLOPE,
        cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        resolver_max_depth=FakeExt.RESOLVER_MAX_DEPTH,
        ext=FakeExt(),
        pack_file_sha256="0" * 64,
    )
    source.q_from_values(np.asarray([700.0]))
    assert seen == [250.0, 700.0], (
        "the Stockfish arm and the NNUE arm did not both route through the "
        "replaced object; one of them holds its own reference"
    )


# ── 5-7. the per-child Stockfish arm ─────────────────────────────────────────


def test_the_sf_arm_answers_from_the_childs_own_seat() -> None:
    """A child the engine scores +300 comes back POSITIVE from this arm.

    ⚑ THE MUTANT THIS EXISTS FOR is negating inside the Stockfish arm. The
    negation to the root mover's seat belongs to ``probe_root`` and is applied
    to EVERY arm; doing it twice for one arm inverts that arm alone, which reads
    as "this labeller is bad" rather than as a bug.
    """
    board = chess.Board(_ROOT_FEN)
    uci = legal_ucis_of(_ROOT_FEN)[0]
    child = board.copy()
    child.push(chess.Move.from_uci(uci))
    engine = FakeEngine({child.fen().split()[0]: sf_result(cp=300)})
    arm = sf_child_arm(engine)
    q = arm.evaluate([CBoard.from_board(child)], role="leaf", cluster=None)
    assert q[0] > 0.0
    assert q[0] == pytest.approx(
        gate.q_from_effective_cp(
            np.asarray([300.0]), slope=gen.NNUE_CP_SLOPE,
            draw_width_cp=gen.NNUE_CP_DRAW_WIDTH,
        )[0],
    )


def test_the_sf_arm_maps_mate_through_the_single_mate_home() -> None:
    """Mate scores go through ``mate_to_effective_cp``, never the cp slope."""
    assert gate.effective_cp_from_score(None, 3) == mate_to_effective_cp(3)
    assert gate.effective_cp_from_score(None, -3) == mate_to_effective_cp(-3)
    # ⚑ `mate 0` is a REAL score. `if mate:` would send it to the cp branch,
    # where a missing cp then reads as "the engine did not score this".
    assert gate.effective_cp_from_score(None, 0) == mate_to_effective_cp(0)
    # Mate wins over a cp that arrived on the same line.
    assert gate.effective_cp_from_score(5, 2) == mate_to_effective_cp(2)
    assert gate.effective_cp_from_score(5, None) == 5.0
    assert gate.effective_cp_from_score(None, None) is None

    board = chess.Board(_ROOT_FEN)
    uci = legal_ucis_of(_ROOT_FEN)[0]
    child = board.copy()
    child.push(chess.Move.from_uci(uci))
    engine = FakeEngine({child.fen().split()[0]: sf_result(mate=3)})
    arm = sf_child_arm(engine)
    assert arm.effective_cp(CBoard.from_board(child)) == mate_to_effective_cp(3)


def test_an_unscored_search_is_refused_rather_than_imputed() -> None:
    board = chess.Board(_ROOT_FEN)
    uci = legal_ucis_of(_ROOT_FEN)[0]
    child = board.copy()
    child.push(chess.Move.from_uci(uci))
    engine = FakeEngine({child.fen().split()[0]: sf_result()})
    arm = sf_child_arm(engine)
    with pytest.raises(RuntimeError, match="no cp and no mate"):
        arm.effective_cp(CBoard.from_board(child))


@pytest.mark.parametrize("fen", [_MATE_IN_ONE_FEN, _STALEMATE_FEN])
def test_the_sf_arm_scores_terminal_children_exactly_as_the_native_arm_does(
    synthetic_pack: Path, fen: str,
) -> None:
    """Terminal children get the NATIVE arm's values, measured against it.

    Stockfish reports a checkmated position as ``score mate 0``, which through
    the mate home is +100000 -- a WIN for the side that was just mated. So the
    arm resolves terminals from the board, and this test is what pins the values
    it resolves them to.
    """
    board = chess.Board(fen)
    ucis, _ = legal_full_indices(board)
    root = CBoard.from_board(board)
    children: list[CBoard] = []
    for uci in ucis:
        child = root.copy()
        child.push_index(_action_of(board, uci))
        children.append(child)
    terminal = [c for c in children if c.is_game_over()]
    assert terminal, f"{fen} has no terminal child to compare on"

    source = gen.NnueArmValueSource(
        arm=gen.VALUE_SOURCE_NNUE_STATIC,
        pack=synthetic_pack,
        cp_per_internal_unit=gen.NNUE_CP_PER_INTERNAL_UNIT,
        cp_slope=gen.NNUE_CP_SLOPE,
        cp_draw_width=gen.NNUE_CP_DRAW_WIDTH,
        resolver_max_depth=int(_nnue_ext.RESOLVER_MAX_DEPTH),
    )
    try:
        native = source.q_for_boards(terminal, role="leaf")
    finally:
        source.close()
    arm = sf_child_arm(RefusingEngine())
    stockfish = arm.evaluate(terminal, role="leaf", cluster=None)
    np.testing.assert_allclose(stockfish, native)
    assert arm.stamp()["terminal_children_resolved_without_search"] == len(terminal)


def _action_of(board: chess.Board, uci: str) -> int:
    ucis, idxs = legal_full_indices(board)
    return int(idxs[ucis.index(uci)])


# ── 8-11. the rooted MultiPV arm ─────────────────────────────────────────────


def info_line(depth: int, rank: int, move: str, cp: int, *, bound: str = "") -> str:
    tail = f" {bound}" if bound else ""
    return (
        f"info depth {depth} seldepth {depth + 2} multipv {rank} "
        f"score cp {cp}{tail} nodes 1234 pv {move} e7e5"
    )


def test_the_rooted_arm_reads_pv1_from_the_deepest_complete_depth() -> None:
    """A later depth's PV1 wins; the earlier depth's is stale.

    ⚑ THE CLASSIC UCI PARSING BUG. A node-limited search prints every depth, so
    "the PV1 line I saw" is the FIRST depth unless the reader keeps looking, and
    "the last PV1 line I saw" is a partial iteration unless the reader checks
    the set is complete.
    """
    lines = [
        info_line(6, 1, "a2a3", 20), info_line(6, 2, "b2b3", 10),
        info_line(6, 3, "c2c3", 5),
        info_line(7, 1, "b2b3", 30), info_line(7, 2, "a2a3", 12),
        info_line(7, 3, "c2c3", 4),
    ]
    ranking = gate.rooted_ranking_from_info_lines(lines, expected_lines=3)
    assert ranking.depth == 7
    assert ranking.complete is True
    assert ranking.moves[0] == (1, "b2b3", 30.0)


def test_the_rooted_arm_falls_back_to_the_deepest_complete_multipv_set() -> None:
    """A truncated final iteration is DISCARDED, never blended with the one below.

    Keeping "the last line per rank" would take rank 1 from depth 7 and ranks 2
    and 3 from depth 6 -- one ranking assembled out of two searches, with every
    downstream number a blend and nothing raising.
    """
    lines = [
        info_line(6, 1, "a2a3", 20), info_line(6, 2, "b2b3", 10),
        info_line(6, 3, "c2c3", 5),
        info_line(7, 1, "b2b3", 30),  # the node limit landed here
    ]
    ranking = gate.rooted_ranking_from_info_lines(lines, expected_lines=3)
    assert ranking.depth == 6
    assert ranking.complete is True
    assert [m for _, m, _ in ranking.moves] == ["a2a3", "b2b3", "c2c3"]
    assert [cp for _, _, cp in ranking.moves] == [20.0, 10.0, 5.0]


def test_a_rooted_search_with_no_complete_depth_reports_the_narrower_ranking() -> None:
    lines = [info_line(4, 1, "a2a3", 20), info_line(4, 2, "b2b3", 10)]
    ranking = gate.rooted_ranking_from_info_lines(lines, expected_lines=3)
    assert ranking.depth == 4
    assert ranking.complete is False
    assert len(ranking.moves) == 2
    with pytest.raises(RuntimeError, match="no scored MultiPV line"):
        gate.rooted_ranking_from_info_lines(["info depth 3 nodes 5"], expected_lines=1)


def test_the_rooted_arm_drops_aspiration_bound_lines() -> None:
    """An ``upperbound`` score is a claim about a window, not about the move."""
    lines = [
        info_line(6, 1, "a2a3", 20), info_line(6, 2, "b2b3", 10),
        info_line(7, 1, "c2c3", 900, bound="upperbound"),
        info_line(7, 2, "d2d4", 800, bound="lowerbound"),
    ]
    ranking = gate.rooted_ranking_from_info_lines(lines, expected_lines=2)
    assert ranking.depth == 6
    assert ranking.moves[0][1] == "a2a3"


def test_the_rooted_arm_reads_scores_from_the_root_movers_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rooted MultiPV score is ALREADY the root mover's -- do not negate.

    ⚑ THE MUTANT THIS EXISTS FOR is running the rooted arm's values through the
    per-child arm's negation. The two constructions differ in exactly this, and
    a negated rooted arm produces a well-formed target that prefers the WORST
    root move the engine listed.
    """
    board = chess.Board(_ROOT_FEN)
    ucis, idxs = legal_full_indices(board)
    best, worst = ucis[0], ucis[1]
    arm = rooted_arm()
    monkeypatch.setattr(
        arm, "search_lines",
        lambda fen, *, multipv: [
            info_line(9, 1, best, 500), info_line(9, 2, worst, -500),
        ],
    )
    label = arm.label(
        board=board, legal_ucis=ucis, legal_idxs=idxs,
        sigma=shadow.oneply_sigma_default(),
    )
    assert label.chosen == best
    assert ucis[int(np.argmax(label.probs))] == best
    assert label.values == {best: 500.0, worst: -500.0}
    assert label.values_kind == "effective_cp_root_seat"
    # Legal moves the search did not list get exactly 0.0, not an imputed value.
    assert float(label.probs[ucis.index(ucis[2])]) == 0.0
    assert arm.stamp()["multipv_realized_mean"] == pytest.approx(2.0)
    assert arm.stamp()["positions_without_a_complete_multipv_depth"] == 1


def test_the_rooted_arm_chooses_pv1_and_counts_the_argmax_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """On an exact cp tie the arm's answer is PV1 and the difference is COUNTED."""
    board = chess.Board(_ROOT_FEN)
    ucis, idxs = legal_full_indices(board)
    # PV1 is the LATER move in audit's own legal order, so np.argmax over the
    # tied probabilities would pick the other one.
    pv1, other = ucis[3], ucis[0]
    arm = rooted_arm()
    monkeypatch.setattr(
        arm, "search_lines",
        lambda fen, *, multipv: [
            info_line(9, 1, pv1, 40), info_line(9, 2, other, 40),
        ],
    )
    label = arm.label(
        board=board, legal_ucis=ucis, legal_idxs=idxs,
        sigma=shadow.oneply_sigma_default(),
    )
    assert label.chosen == pv1
    assert ucis[int(np.argmax(label.probs))] == other
    del tmp_path


def test_a_rooted_arm_name_canonicalises_to_its_default_width() -> None:
    names, specs = gate.parse_arms("sfroot-2048,sfroot-2048-mpv20")
    assert names == (f"sfroot-2048-mpv{gate.DEFAULT_ROOTED_MULTIPV}",)
    assert specs[names[0]].width == gate.DEFAULT_ROOTED_MULTIPV
    assert gate.DEFAULT_ROOTED_MULTIPV == 20

    all_names, all_specs = gate.parse_arms("sfroot-1024-mpvall")
    assert all_names == ("sfroot-1024-mpvall",)
    assert all_specs[all_names[0]].width is None
    assert all_specs[all_names[0]].rooted is True

    per_child, child_specs = gate.parse_arms("sf-512")
    assert per_child == ("sf-512",)
    assert child_specs["sf-512"].rooted is False
    with pytest.raises(ValueError, match="unknown arm"):
        gate.parse_arms("sfroot-abc")


def test_the_all_width_is_resolved_per_position(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = rooted_arm("sfroot-1024-mpvall")
    assert arm.requested_width(37) == 37
    assert arm.requested_width(2) == 2
    fixed = rooted_arm("sfroot-1024-mpv20")
    assert fixed.requested_width(37) == 20
    # ⚑ A fixed width still CLAMPS to the legal move count: Stockfish caps
    # MultiPV at the root move count, so a realized width of 4 on a 4-move
    # position is complete, not truncated.
    assert fixed.requested_width(4) == 4
    del monkeypatch


# ── 12-14. the banned source, and the stamp ──────────────────────────────────


def test_an_audit_set_pointed_at_the_dirty_sidecar_is_refused() -> None:
    args = gate.build_parser().parse_args([
        "--arms", "sf-512",
        "--audit-set", f"data/audit_set_v1.jsonl{audit_targets.SHALLOW_SF_CACHE_SUFFIX}",
    ])
    with pytest.raises(ValueError, match="dirty shared transposition table"):
        gate.config_from_args(args)


def test_the_dirty_tt_shallow_sf_cache_is_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_pack: Path,
) -> None:
    """Every path opened during a REAL run is recorded and inspected.

    A source scan would prove only that this module has no such literal; the
    tool reaches ``audit_targets`` and ``eval.audit``, either of which could
    open the cache on its behalf. The recorder is asserted to be live (it saw
    the audit set) so the check cannot pass vacuously.
    """
    ucis = legal_ucis_of(_ROOT_FEN)
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(u, 100 - 10 * i) for i, u in enumerate(ucis[:10])]),
    ])
    # A decoy with the banned name, beside the audit set exactly where
    # `audit_targets` would look for it.
    decoy = tmp_path / f"audit.jsonl{audit_targets.SHALLOW_SF_CACHE_SUFFIX}"
    decoy.write_text("{}\n", encoding="utf-8")

    opened: list[str] = []
    real_open = builtins.open
    real_path_open = Path.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(os.fspath(file) if not isinstance(file, int) else "<fd>")
        return real_open(file, *args, **kwargs)

    def spy_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)

    args = gate.build_parser().parse_args([
        "--audit-set", str(audit),
        "--nnue-pack", str(synthetic_pack),
        "--arms", "nnue-static,nnue-fastq",
        "--nice", "0",
    ])
    report = gate.run(gate.config_from_args(args))

    assert any(str(audit) in p for p in opened), (
        "the open recorder saw nothing; the check would pass vacuously"
    )
    banned = [p for p in opened if audit_targets.SHALLOW_SF_CACHE_SUFFIX in p]
    assert banned == [], f"the dirty-TT sidecar was opened: {banned}"
    assert decoy.exists()
    assert report["positions_scored"] == 1
    assert set(report["arms"]) == {"nnue-static", "nnue-fastq"}
    # The positive control for the consumer-gated stamps, on the REAL run path:
    # nnue-static consumes the cp-per-unit conversion and nnue-fastq is
    # DAG-backed, so both stamps carry numbers here rather than null.
    assert report["provenance"]["cp_per_internal_unit"] == pytest.approx(
        gen.NNUE_CP_PER_INTERNAL_UNIT,
    )
    assert report["provenance"]["dag_max_nodes"] == shadow.DEFAULT_DAG_MAX_NODES


def test_the_report_stamps_the_audit_set_and_states_the_censoring_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ucis = legal_ucis_of(_ROOT_FEN)
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(u, 100 - 10 * i) for i, u in enumerate(ucis[:10])]),
    ])
    arm = ScriptedChildArm("scripted", {})
    report = run_with_arms(monkeypatch, gate_config(audit, (arm.arm,)), [arm])
    provenance = report["provenance"]
    assert provenance["audit_set_sha256"] == hashlib.sha256(
        audit.read_bytes(),
    ).hexdigest()
    assert provenance["audit_set"] == str(audit)
    assert provenance["sf_binary"] is None
    definitions = report["metric_definitions"]
    assert "WORST LISTED" in definitions["censoring_rule_for_unlisted_moves"]
    assert "FLOOR" in definitions["censoring_rule_for_unlisted_moves"]
    assert definitions["regret_cap_cp"] == 1000.0
    assert "sf_reference_sets" in definitions["top1_agree_rate"]
    assert report["admissible"] is True
    assert report["schema"] == gate.REPORT_SCHEMA


# ── 15-17. the knobs, and the move set ───────────────────────────────────────


@pytest.mark.parametrize("requested", [2, 6])
def test_a_fastq_knob_reaches_the_arms_own_realized_config(
    tmp_path: Path, synthetic_pack: Path, requested: int,
) -> None:
    """``--fastq-max-qply`` is read back out of the CONTEXT, not echoed.

    ⚑ THIS REPO'S SIGNATURE DEFECT IS A VALUE ACCEPTED AND THEN IGNORED, so the
    proof has to come from the consumer. ``arm_config_realized`` is
    ``arm_stats`` on the live FastQ context; the CLI value is in
    ``arm_config_requested`` beside it, and ``NnueArmValueSource`` refuses to
    open when the two disagree.
    """
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(legal_ucis_of(_ROOT_FEN)[0], 10)]),
    ])
    args = gate.build_parser().parse_args([
        "--audit-set", str(audit), "--nnue-pack", str(synthetic_pack),
        "--arms", "nnue-fastq", "--fastq-max-qply", str(requested),
    ])
    child, rooted = gate.open_arms(gate.config_from_args(args), pack_sha=None)
    try:
        assert rooted == []
        stamp = child[0].stamp()
        assert stamp["arm_config_requested"]["max_qply"] == requested
        assert stamp["arm_config_realized"]["max_qply"] == requested
        assert stamp["provider_stats"]["max_qply"] == requested
    finally:
        for arm in child:
            arm.close()


def test_a_stockfish_engine_is_closed_when_its_cold_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``ucinewgame`` must not leak the subprocess it was talking to.

    ⚑ THE MUTANT THIS EXISTS FOR is calling ``clear_transposition_table`` BEFORE
    the arm is appended to ``child``/``rooted``. ``open_arms``'s cleanup walks
    those lists, so an engine that was constructed and not yet registered is a
    live Stockfish nobody owns -- and the clear is exactly the step that can
    fail, because it is the first thing that talks UCI.
    """
    engines: list[FakeEngine] = []

    def fake_stockfish(*args: Any, **kwargs: Any) -> FakeEngine:
        del args, kwargs
        engine = FakeEngine()
        engines.append(engine)
        return engine

    def failing_clear(engine: Any) -> None:
        del engine
        raise RuntimeError("ucinewgame handshake timed out")

    monkeypatch.setattr(gate, "StockfishUCI", fake_stockfish)
    monkeypatch.setattr(gate, "clear_transposition_table", failing_clear)

    names, specs = gate.parse_arms("sf-512")
    cfg = gate_config(
        tmp_path / "unread.jsonl", names,
        sf_specs=specs, sf_binary=Path("stockfish"),
    )
    with pytest.raises(RuntimeError, match="ucinewgame handshake timed out"):
        gate.open_arms(cfg, pack_sha=None)
    assert len(engines) == 1, "the engine constructor was not the one patched"
    assert engines[0].closed is True, "the failed cold start leaked its engine"


def test_the_default_fastq_width_is_the_extensions_own(
    tmp_path: Path, synthetic_pack: Path,
) -> None:
    """The control for the test above: unset means the compiled-in default."""
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(legal_ucis_of(_ROOT_FEN)[0], 10)]),
    ])
    args = gate.build_parser().parse_args([
        "--audit-set", str(audit), "--nnue-pack", str(synthetic_pack),
        "--arms", "nnue-fastq",
    ])
    child, _ = gate.open_arms(gate.config_from_args(args), pack_sha=None)
    try:
        assert child[0].stamp()["arm_config_realized"]["max_qply"] == int(
            _nnue_ext.FASTQ_MAX_QPLY,
        )
    finally:
        for arm in child:
            arm.close()


def refusal_args(arms: str, *extra: str) -> argparse.Namespace:
    """CLI args for one refusal case, with a pack IFF a native arm needs one.

    ⚑ THE PACK IS CONDITIONAL, and that is load-bearing. ``--nnue-pack`` is
    itself one of the knobs a Stockfish-only run refuses, so passing it on every
    row would make each sf-only case raise the PACK's message before reaching
    the knob the row is about -- every row still red, every row proving the
    wrong thing.
    """
    argv = ["--arms", arms, *extra]
    if any(a in gate.NATIVE_ARMS for a in arms.split(",")):
        argv += ["--nnue-pack", "unused.pack"]
    return gate.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("arms", "flag", "value", "message"),
    [
        ("nnue-static", "--fastq-max-qply", "2", "--fastq-"),
        ("nnue-fastq", "--nnue-qsearch-max-ply", "2", "--nnue-qsearch-"),
        ("nnue-fastq", "--dag-node-cap", "8", "--dag-node-cap"),
        ("sf-512", "--nnue-resolver-max-depth", "8", "--nnue-resolver-max-depth"),
        # ⚑ THE WHOLE-RUN KNOBS, which were accepted in silence. The first is
        # the worst of them: it was also STAMPED into `provenance`, so an
        # SF-only report published a native-arm conversion factor beside numbers
        # no arm had used it to produce.
        ("sf-512", "--nnue-cp-per-unit", "300", "--nnue-cp-per-unit"),
        ("sf-512", "--bank-observations", "bank.jsonl", "--bank-observations"),
        ("sf-512", "--dag-max-nodes", "1000", "--dag-max-nodes"),
        ("sf-512", "--dag-reset-every", "4", "--dag-reset-every"),
        ("sfroot-2048-mpv20", "--nnue-cp-per-unit", "300", "--nnue-cp-per-unit"),
        # ⚑ NATIVE IS NOT THE SAME SET AS DAG-BACKED. `nnue-static` and
        # `nnue-qsearch` intern no canonical store, so the store knobs have no
        # consumer on those runs either.
        ("nnue-static", "--dag-max-nodes", "1000", "--dag-max-nodes"),
        ("nnue-qsearch", "--dag-reset-every", "4", "--dag-reset-every"),
    ],
)
def test_a_knob_no_selected_arm_consumes_is_refused(
    arms: str, flag: str, value: str, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        gate.config_from_args(refusal_args(arms, flag, value))


def test_a_stockfish_only_run_refuses_the_native_pack() -> None:
    """``--nnue-pack`` on a run with no native arm: read by nothing, not stamped.

    It gets its own test rather than a parametrize row because ``refusal_args``
    is what decides whether to pass a pack, and a helper cannot test itself.
    """
    args = gate.build_parser().parse_args([
        "--arms", "sf-512", "--nnue-pack", "unused.pack",
    ])
    with pytest.raises(ValueError, match="--nnue-pack"):
        gate.config_from_args(args)


def test_a_dag_backed_run_still_accepts_every_whole_run_knob() -> None:
    """The control for the refusals: an arm that DOES consume them gets them.

    A refusal that fired on the consuming arm too would pass every test above
    and make the tool unusable, so the accepting direction is asserted on the
    resolved config rather than assumed.
    """
    cfg = gate.config_from_args(gate.build_parser().parse_args([
        "--arms", "nnue-qsearch-dag", "--nnue-pack", "unused.pack",
        "--nnue-cp-per-unit", "300", "--dag-max-nodes", "4096",
        "--dag-reset-every", "4", "--bank-observations", "bank.jsonl",
    ]))
    assert cfg.cp_per_internal_unit == pytest.approx(300.0)
    assert cfg.dag_max_nodes == 4096
    assert cfg.dag_reset_every == 4
    assert cfg.bank_observations == Path("bank.jsonl")
    assert cfg.dag_arms == ("nnue-qsearch-dag",)
    # nnue-fastq is DAG-backed too, read off ArmSpec.uses_dag rather than named.
    assert sorted(gate.DAG_BACKED_ARMS) == ["nnue-fastq", "nnue-qsearch-dag"]


def test_the_sentinel_defaults_resolve_to_the_values_they_replaced() -> None:
    """``default=None`` must not change what an UNSET run does.

    The three knobs moved to a "not supplied" sentinel so the refusals above can
    tell an explicit value from argparse's own default. This is the check that
    the move did not quietly redefine the defaults themselves.
    """
    cfg = gate.config_from_args(gate.build_parser().parse_args([
        "--arms", "nnue-qsearch-dag", "--nnue-pack", "unused.pack",
    ]))
    assert cfg.cp_per_internal_unit == gen.NNUE_CP_PER_INTERNAL_UNIT
    assert cfg.dag_max_nodes == shadow.DEFAULT_DAG_MAX_NODES
    assert cfg.dag_reset_every == gate.DEFAULT_DAG_RESET_EVERY == 1


def test_a_report_stamps_a_whole_run_knob_only_when_an_arm_consumed_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No native arm, no DAG arm -> the knobs stamp NULL, not their defaults.

    ⚑ THE MUTANT THIS EXISTS FOR is stamping ``float(cfg.cp_per_internal_unit)``
    unconditionally. A number on the face of a report reads as a setting that
    shaped it; on a run with no native arm nothing read this one, and a default
    printed there is indistinguishable from a value that was applied.
    """
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(legal_ucis_of(_ROOT_FEN)[0], 10)]),
    ])
    arm = ScriptedChildArm("scripted", {})
    provenance = run_with_arms(
        monkeypatch, gate_config(audit, (arm.arm,)), [arm],
    )["provenance"]
    assert provenance["cp_per_internal_unit"] is None
    assert provenance["dag_max_nodes"] is None
    assert provenance["dag_reset_every_positions"] is None
    assert provenance["pack_path"] is None
    # The cp logistic IS on every arm's path, Stockfish arms included, so these
    # two are real numbers on the same report. The null above is a consumer
    # claim, not a blanket "no native arm, no cp fields".
    assert provenance["cp_slope"] == pytest.approx(gen.NNUE_CP_SLOPE)
    assert provenance["cp_draw_width"] == pytest.approx(gen.NNUE_CP_DRAW_WIDTH)


def test_the_static_arm_still_consumes_the_resolver_depth() -> None:
    """The one case ``readout._validate_matrix_knobs`` would refuse wrongly."""
    args = gate.build_parser().parse_args([
        "--arms", "nnue-static", "--nnue-pack", "unused.pack",
        "--nnue-resolver-max-depth", "8",
    ])
    cfg = gate.config_from_args(args)
    assert cfg.static_resolver_max_depth == 8


def test_the_probe_and_the_audit_scorer_must_agree_on_the_move_set() -> None:
    board = chess.Board(_ROOT_FEN)
    _, idxs = legal_full_indices(board)
    probe = shadow.PlyProbe(
        game=0, ply_ordinal=0,
        legal_full_indices=tuple(int(a) for a in idxs[:-1]),
        q_mover={},
    )
    position = argparse.Namespace(key="k", fen=_ROOT_FEN)
    with pytest.raises(RuntimeError, match="different move set"):
        gate._refuse_move_set_drift(cast(Any, position), probe, idxs)


def test_a_black_to_move_audit_row_is_refused_through_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drift guard is WIRED INTO ``run``, proved by a row that trips it.

    ⚑ THE MUTANT THIS EXISTS FOR is deleting the ``_refuse_move_set_drift``
    call from ``run``. The test above calls the guard directly, so it passes on
    an unwired guard -- a check that cannot fire is this repo's signature defect
    wearing a helper's name.

    ⚑ AND THE DRIFT IS REAL, NOT HYPOTHETICAL. ``eval.audit`` hard-codes
    ``uci_to_policy_index(uci, True)`` because audit rows are meant to be
    side-to-move canonical; ``CBoard`` encodes from the side to move. On this
    row the two index sets are DISJOINT -- 37 legal moves each, symmetric
    difference 74 -- so the ``full[legal_idxs]`` gather in ``child_label`` reads
    4672-vector entries the softmax never wrote.

    ⚑ WHAT THE UNWIRED MUTANT ACTUALLY DOES, MEASURED rather than assumed: it
    dies one line further on in ``probe_order_for`` with a bare
    ``KeyError: 4641``. So the guard is not buying "raises instead of not
    raising" -- it is buying a NAMED position, its FEN and the symmetric
    difference in place of a dict lookup blowing up two helpers deep. The gather
    IS silent in the other direction: where the audit set is a strict SUBSET of
    the probe's (a legal move ``uci_to_policy_index`` declines to encode), every
    later lookup succeeds and the label is simply built on less than the full
    move mass, with nothing raising anywhere.
    """
    fen = _BLACK_TO_MOVE_FEN
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(fen, listed=[(u, 100 - 10 * i)
                               for i, u in enumerate(legal_ucis_of(fen)[:5])]),
    ])
    arm = ScriptedChildArm("scripted", {})
    with pytest.raises(RuntimeError, match="different move set"):
        run_with_arms(monkeypatch, gate_config(audit, (arm.arm,)), [arm])
    # The arm WAS driven before the refusal, so the guard is inside the scoring
    # loop and not a precondition `run` could have checked on the file alone.
    assert arm.positions == 1


def test_the_same_row_with_white_to_move_scores_without_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: the guard fires on the SEAT, not the row.

    Without this, a ``_refuse_move_set_drift`` that raised unconditionally would
    pass the black-to-move test and break the tool.
    """
    audit = write_audit_set(tmp_path / "audit.jsonl", [
        audit_row(_ROOT_FEN, listed=[(u, 100 - 10 * i)
                                     for i, u in enumerate(legal_ucis_of(_ROOT_FEN)[:5])]),
    ])
    arm = ScriptedChildArm("scripted", {})
    report = run_with_arms(monkeypatch, gate_config(audit, (arm.arm,)), [arm])
    assert report["positions_scored"] == 1


# ── 18-19. the --limit slice ─────────────────────────────────────────────────


def limited_audit_set(path: Path) -> Path:
    return write_audit_set(path, [
        audit_row(fen, listed=[(legal_ucis_of(fen)[0], 10)]) for fen in _THREE_FENS
    ])


def test_the_limit_bounds_the_rows_the_arms_are_actually_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--limit 2`` on a 3-row set: two rows scored, two arm calls, no more.

    ⚑ THE MUTANT THIS EXISTS FOR is dropping the slice
    (``positions = positions[: cfg.limit]``). Nothing downstream notices: the
    run simply scores every row in the file, the report is well-formed, and
    ``provenance.limit`` still echoes the 2 that was asked for -- so the one
    number a reader would check to catch it is the number that lies.
    """
    audit = limited_audit_set(tmp_path / "audit.jsonl")
    arm = ScriptedChildArm("scripted", {})
    dump = tmp_path / "rows.jsonl"
    report = run_with_arms(
        monkeypatch,
        gate_config(audit, (arm.arm,), limit=2, dump_per_position=dump),
        [arm],
    )
    assert report["audit_positions_in_file"] == 3
    assert report["positions_scored"] == 2
    # The arm itself was driven twice, which is the claim `positions_scored`
    # would still make if the slice had been skipped and two rows had failed to
    # encode instead.
    assert arm.positions == 2
    assert len(dump.read_text(encoding="utf-8").splitlines()) == 2
    assert report["provenance"]["limit"] == 2
    assert report["provenance"]["limit_realized"] == 2


def test_the_realized_limit_is_the_slice_and_not_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit larger than the file: the REQUEST is 5 and the REALIZED is 3.

    ⚑ THE MUTANT THIS EXISTS FOR is stamping ``int(cfg.limit)`` under both
    keys. ``provenance.limit`` is what the CLI asked for and cannot be a
    measurement of anything; ``limit_realized`` is how many rows the slice
    yielded, and the two only coincide when the ask happened to fit.
    """
    audit = limited_audit_set(tmp_path / "audit.jsonl")
    arm = ScriptedChildArm("scripted", {})
    report = run_with_arms(
        monkeypatch, gate_config(audit, (arm.arm,), limit=5), [arm],
    )
    assert report["provenance"]["limit"] == 5
    assert report["provenance"]["limit_realized"] == 3
    assert report["positions_scored"] == 3
    assert arm.positions == 3

    # 0 means "all", and the realized value says so rather than repeating 0.
    unlimited = ScriptedChildArm("scripted", {})
    whole = run_with_arms(
        monkeypatch, gate_config(audit, (unlimited.arm,), limit=0), [unlimited],
    )
    assert whole["provenance"]["limit"] == 0
    assert whole["provenance"]["limit_realized"] == 3
