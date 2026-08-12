from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import chess
import chess.engine
import pytest

from tests.script_loading import load_script_module


def _load_match_vs_uci_module():
    return load_script_module("match_vs_uci.py", "match_vs_uci_test_module")


def test_side_limit_prefers_nodes_over_time() -> None:
    module = _load_match_vs_uci_module()

    limit = module._side_limit(time_ms=200, nodes=640_000)

    assert limit.nodes == 640_000
    assert limit.time is None


def test_side_limit_uses_time_without_nodes() -> None:
    module = _load_match_vs_uci_module()

    limit = module._side_limit(time_ms=250, nodes=None)

    assert limit.nodes is None
    assert limit.time == 0.25


def test_clock_limit_sets_both_clocks_and_increments() -> None:
    module = _load_match_vs_uci_module()

    limit = module._clock_limit(
        white_clock_s=30.0,
        black_clock_s=29.5,
        white_inc_s=0.5,
        black_inc_s=0.25,
    )

    assert limit.white_clock == 30.0
    assert limit.black_clock == 29.5
    assert limit.white_inc == 0.5
    assert limit.black_inc == 0.25


def test_play_label_prefers_clock_over_fallback_time() -> None:
    module = _load_match_vs_uci_module()

    assert module._play_label(
        time_ms=200,
        nodes=None,
        clock_base_ms=30_000,
        clock_inc_ms=500,
    ) == "30+0.5s"


def test_score_for_a_maps_results_by_color() -> None:
    module = _load_match_vs_uci_module()

    assert module._score_for_a("1-0", a_is_white=True) == 1.0
    assert module._score_for_a("0-1", a_is_white=True) == 0.0
    assert module._score_for_a("0-1", a_is_white=False) == 1.0
    assert module._score_for_a("1-0", a_is_white=False) == 0.0
    assert module._score_for_a("1/2-1/2", a_is_white=False) == 0.5


def test_play_one_pairing_preserves_side_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_match_vs_uci_module()
    calls = {}
    white_engine: Any = object()
    black_engine: Any = object()

    def fake_play_one_game(eng_w: object, eng_b: object, **kwargs):
        calls["eng_w"] = eng_w
        calls["eng_b"] = eng_b
        calls["kwargs"] = kwargs
        return module.GameRecord("1/2-1/2", 0, chess.Board(), (), ())

    monkeypatch.setattr(module, "play_one_game", fake_play_one_game)
    white = module.EngineSide(
        label="A",
        engine=white_engine,
        time_ms=100,
        nodes=11,
        clock_base_ms=30_000,
        clock_inc_ms=500,
        enforce_nodes=True,
    )
    black = module.EngineSide(
        label="B",
        engine=black_engine,
        time_ms=250,
        nodes=None,
        clock_base_ms=15_000,
        clock_inc_ms=250,
        enforce_nodes=False,
    )

    record = module._play_one_pairing(
        white=white,
        black=black,
        max_plies=3,
        start_board=chess.Board(),
        move_callback=None,
        syzygy_adjudicator=None,
        syzygy_adjudicate_max_pieces=6,
    )

    assert record.result == "1/2-1/2"
    assert calls["eng_w"] is white_engine
    assert calls["eng_b"] is black_engine
    assert calls["kwargs"]["limit_w"].nodes == 11
    assert calls["kwargs"]["limit_b"].time == 0.25
    assert calls["kwargs"]["enforce_nodes_w"] is True
    assert calls["kwargs"]["enforce_nodes_b"] is False
    assert calls["kwargs"]["clock_base_w_ms"] == 30_000
    assert calls["kwargs"]["clock_base_b_ms"] == 15_000
    assert calls["kwargs"]["clock_inc_w_ms"] == 500
    assert calls["kwargs"]["clock_inc_b_ms"] == 250
    assert calls["kwargs"]["syzygy_adjudicate_max_pieces"] == 6


def test_parse_opening_line_accepts_six_field_fen() -> None:
    module = _load_match_vs_uci_module()

    board = module._parse_opening_line("rn1q1rk1/p1ppbppp/b3pn2/1B6/2PP4/2N2N2/PP3PPP/R1BQR1K1 w - - 3 9")

    assert board is not None
    assert board.turn is True
    assert board.fullmove_number == 9


def test_parse_opening_line_accepts_four_field_epd() -> None:
    module = _load_match_vs_uci_module()

    board = module._parse_opening_line("8/8/8/8/8/8/K7/6k1 w - - bm Ka2;")

    assert board is not None
    assert board.fullmove_number == 1


def test_score_ci_and_elo_helpers() -> None:
    module = _load_match_vs_uci_module()

    ci = module._score_ci([1.0, 0.5, 0.0, 0.5])

    assert ci is not None
    assert ci[0] < 0.5 < ci[1]
    assert module._elo_from_score(0.5) == 0.0
    assert module._elo_from_score(0.0) is None


def test_tb_adjudicate_result_maps_wdl_from_side_to_move() -> None:
    module = _load_match_vs_uci_module()

    class FakeTablebase:
        def probe_wdl(self, board: chess.Board) -> int:
            del board
            return 2

    white_to_move = chess.Board("6k1/8/8/8/8/8/8/KQ6 w - - 0 1")
    black_to_move = chess.Board("6k1/8/8/8/8/8/8/KQ6 b - - 0 1")

    assert module._tb_adjudicate_result(white_to_move, FakeTablebase(), max_pieces=4) == "1-0"
    assert module._tb_adjudicate_result(black_to_move, FakeTablebase(), max_pieces=4) == "0-1"


def test_validate_syzygy_adjudicator_requires_wdl_tables() -> None:
    module = _load_match_vs_uci_module()

    class DtzOnlyTablebase:
        wdl: ClassVar[dict[str, object]] = {}
        dtz: ClassVar[dict[str, object]] = {"KQvK": object()}

    with pytest.raises(SystemExit, match="0 WDL"):
        module._validate_syzygy_adjudicator("/tb", DtzOnlyTablebase(), max_pieces=6)


def test_validate_syzygy_adjudicator_requires_requested_wdl_coverage() -> None:
    module = _load_match_vs_uci_module()

    class FiveManOnlyTablebase:
        wdl: ClassVar[dict[str, object]] = {"KQvKRR": object()}
        dtz: ClassVar[dict[str, object]] = {}

    with pytest.raises(SystemExit, match="only has WDL up to 5 pieces"):
        module._validate_syzygy_adjudicator("/tb", FiveManOnlyTablebase(), max_pieces=6)


def test_validate_syzygy_adjudicator_reports_table_coverage() -> None:
    module = _load_match_vs_uci_module()

    class CoveredTablebase:
        wdl: ClassVar[dict[str, object]] = {"KQvKR": object(), "KQBNvKP": object()}
        dtz: ClassVar[dict[str, object]] = {"KQvKRR": object()}

    assert module._validate_syzygy_adjudicator(
        "/tb",
        CoveredTablebase(),
        max_pieces=6,
    ) == (2, 1, 6, 5)


def test_warmup_engine_uses_external_node_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_match_vs_uci_module()
    calls: list[int] = []

    def fake_play_node_limited(
        eng: object, board: chess.Board, *, nodes: int, timeout_s: float | None = None,
    ):
        del eng, timeout_s
        assert board == chess.Board()
        calls.append(nodes)
        return module.MoveResult(chess.Move.from_uci("e2e4"), {"nodes": nodes})

    monkeypatch.setattr(module, "_play_node_limited", fake_play_node_limited)
    module._warmup_engine(object(), nodes=7, label="fake")

    assert calls == [7]
    assert "nodes=7" in capsys.readouterr().out


def test_play_node_limited_preserves_latest_analysis_info() -> None:
    module = _load_match_vs_uci_module()
    move = chess.Move.from_uci("e2e4")

    class FakeAnalysis:
        def __init__(self) -> None:
            self.stop_calls = 0

        def would_block(self) -> bool:
            return False

        def get(self) -> dict[str, Any]:
            return {
                "nodes": 7,
                "time": 0.125,
                "score": chess.engine.PovScore(chess.engine.Cp(23), chess.WHITE),
                "pv": [move],
            }

        def stop(self) -> None:
            self.stop_calls += 1

        def wait(self) -> chess.engine.BestMove:
            return chess.engine.BestMove(None, None)

    class FakeEngine:
        def __init__(self) -> None:
            self.analysis_obj = FakeAnalysis()

        def analysis(
            self,
            board: chess.Board,
            limit: chess.engine.Limit,
            *,
            game: object = None,
            info: chess.engine.Info = chess.engine.INFO_NONE,
        ) -> FakeAnalysis:
            del game
            assert board == chess.Board()
            assert limit == chess.engine.Limit()
            assert info == chess.engine.INFO_ALL
            return self.analysis_obj

    result = module._play_node_limited(FakeEngine(), chess.Board(), nodes=7)

    assert result.move == move
    assert result.info["nodes"] == 7
    assert result.info["time"] == 0.125
    assert module._score_cp(result.info["score"]) == 23


def test_play_node_limited_handles_analysis_complete() -> None:
    module = _load_match_vs_uci_module()
    move = chess.Move.from_uci("e2e4")

    class FakeAnalysis:
        def __init__(self) -> None:
            self.stop_calls = 0

        def would_block(self) -> bool:
            return False

        def get(self) -> dict[str, object]:
            raise chess.engine.AnalysisComplete

        def stop(self) -> None:
            self.stop_calls += 1

        def wait(self) -> chess.engine.BestMove:
            return chess.engine.BestMove(move, None)

    class FakeEngine:
        def __init__(self) -> None:
            self.analysis_obj = FakeAnalysis()

        def analysis(
            self,
            board: chess.Board,
            limit: chess.engine.Limit,
            *,
            game: object = None,
            info: chess.engine.Info = chess.engine.INFO_NONE,
        ) -> FakeAnalysis:
            del board, limit, info, game
            return self.analysis_obj

    engine = FakeEngine()
    result = module._play_node_limited(engine, chess.Board(), nodes=10)

    assert result.move == move
    assert result.info == {}
    assert engine.analysis_obj.stop_calls == 1


def test_play_one_game_logs_null_move_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_match_vs_uci_module()
    callbacks = []

    def fake_play_move(*args: object, **kwargs: object):
        del args, kwargs
        return module.MoveResult(None, {"nodes": 3, "time": 0.01})

    monkeypatch.setattr(module, "_play_move", fake_play_move)

    record = module.play_one_game(
        object(),
        object(),
        limit_w=chess.engine.Limit(nodes=1),
        limit_b=chess.engine.Limit(nodes=1),
        enforce_nodes_w=False,
        enforce_nodes_b=False,
        max_plies=1,
        move_callback=callbacks.append,
    )

    assert record.result == "0-1"
    assert record.termination == "illegal_or_null"
    assert record.plies == 0
    assert record.moves == ()
    assert len(record.move_records) == 1
    assert record.move_records[0].move == "0000"
    assert record.move_records[0].nodes == 3
    assert callbacks == [record.move_records[0]]


def test_play_one_game_logs_time_loss_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_match_vs_uci_module()
    move = chess.Move.from_uci("e2e4")

    def fake_play_move(*args: object, **kwargs: object):
        del args, kwargs
        return module.MoveResult(move, {"nodes": 5, "time": 0.2})

    times = iter([10.0, 10.5])
    monkeypatch.setattr(module, "_play_move", fake_play_move)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))

    record = module.play_one_game(
        object(),
        object(),
        limit_w=chess.engine.Limit(time=1),
        limit_b=chess.engine.Limit(time=1),
        enforce_nodes_w=False,
        enforce_nodes_b=False,
        clock_base_w_ms=100,
        clock_base_b_ms=100,
        max_plies=1,
    )

    assert record.result == "0-1"
    assert record.termination == "time"
    assert record.plies == 0
    assert record.moves == ()
    assert len(record.move_records) == 1
    assert record.move_records[0].move == "e2e4"
    assert record.move_records[0].nodes == 5
    assert record.move_records[0].white_clock_before_s == 0.1
    assert record.move_records[0].white_clock_after_s == pytest.approx(-0.4)


def test_main_quits_engine_a_if_engine_b_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_match_vs_uci_module()
    opened = []

    class FakeEngine:
        id: ClassVar[dict[str, str]] = {"name": "fake"}
        options: ClassVar[dict[str, object]] = {}

        def __init__(self) -> None:
            self.quit_called = False

        def quit(self) -> None:
            self.quit_called = True

    def fake_open_engine(spec: str, cwd: str | None = None, *, timeout_s: float = 0.0):
        del cwd, timeout_s
        if spec == "engine-b":
            raise RuntimeError("engine B failed")
        engine = FakeEngine()
        opened.append(engine)
        return engine

    monkeypatch.setattr(module, "_open_engine", fake_open_engine)
    monkeypatch.setattr(sys, "argv", ["match_vs_uci.py", "--engine-a", "engine-a", "--engine-b", "engine-b"])

    with pytest.raises(RuntimeError, match="engine B failed"):
        module.main()

    assert len(opened) == 1
    assert opened[0].quit_called is True


@pytest.mark.parametrize("spec_kind", ["multi_token", "single_binary"])
def test_open_engine_forwards_timeout_to_popen_uci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec_kind: str,
) -> None:
    """The leg that proves the value reaches python-chess, not just our own signature.

    ⚑ Parametrized because ``_open_engine`` has TWO popen_uci call sites, and the
    single-binary branch is the one every real engine-B invocation takes (Cheese and
    rofChade are bare paths). A version of this test that only exercised the multi-token
    branch left a mutation of the single-binary branch completely undetected.
    """
    module = _load_match_vs_uci_module()
    seen: dict[str, object] = {}

    def fake_popen_uci(command: object, cwd: object = None, timeout: float | None = None):
        del command, cwd
        seen["timeout"] = timeout
        return object()

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", staticmethod(fake_popen_uci))

    if spec_kind == "single_binary":
        binary = tmp_path / "engine_binary"
        binary.write_text("#!/bin/sh\n")
        spec = str(binary)
    else:
        spec = "some-engine --flag"

    module._open_engine(spec, timeout_s=1800.0)

    assert seen["timeout"] == 1800.0


def test_relaxed_timeout_is_restored_after_the_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ The relaxation must NOT outlive the warmup it was granted for.

    ``SimpleEngine.timeout`` is set once at popen and re-read by every later command, so
    a timeout left raised would give the whole match -- each per-game ``analysis()``, each
    ``configure()``, and the ``quit()`` calls in teardown -- a multi-minute deadline
    instead of the finite backstop that catches a wedged engine. Failure scenario: an
    engine wedges mid-match and the driver hangs for ``--warmup-timeout-s`` instead of
    300s, then hangs that long AGAIN inside quit(), holding the GPU.
    """
    module = _load_match_vs_uci_module()
    during: dict[str, float] = {}

    class FakeEngine:
        id: ClassVar[dict[str, str]] = {"name": "fake"}
        options: ClassVar[dict[str, object]] = {}

        def __init__(self) -> None:
            self.timeout = 0.0

    def fake_open_engine(spec: str, cwd: str | None = None, *, timeout_s: float = 0.0):
        del spec, cwd
        eng = FakeEngine()
        eng.timeout = timeout_s
        return eng

    def fake_warmup(eng: object, *, nodes: int | None, label: str, timeout_s: float = -1.0):
        del nodes, timeout_s
        during[label] = eng.timeout  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr(module, "_open_engine", fake_open_engine)
    monkeypatch.setattr(module, "_warmup_engine", fake_warmup)

    eng = module._open_warm_engine(
        "engine-a", {}, nodes=3000, label="A", warmup_timeout_s=2400.0,
    )

    assert during["A"] == 2400.0, "the warmup must actually run under the relaxed deadline"
    assert eng.timeout == 300.0, "and the backstop must be back in force once it returns"


def test_relaxed_timeout_is_restored_even_when_the_warmup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-open engine still gets quit()ed, so the restore must be in a finally."""
    module = _load_match_vs_uci_module()

    class FakeEngine:
        id: ClassVar[dict[str, str]] = {"name": "fake"}
        options: ClassVar[dict[str, object]] = {}

        def __init__(self) -> None:
            self.timeout = 0.0

    captured: list[FakeEngine] = []

    def fake_open_engine(spec: str, cwd: str | None = None, *, timeout_s: float = 0.0):
        del spec, cwd
        eng = FakeEngine()
        eng.timeout = timeout_s
        captured.append(eng)
        return eng

    def fake_warmup(eng: object, *, nodes: int | None, label: str, timeout_s: float = -1.0):
        del eng, nodes, label, timeout_s
        raise RuntimeError("warmup blew up")

    monkeypatch.setattr(module, "_open_engine", fake_open_engine)
    monkeypatch.setattr(module, "_warmup_engine", fake_warmup)

    with pytest.raises(RuntimeError, match="warmup blew up"):
        module._open_warm_engine("engine-a", {}, nodes=3000, label="A", warmup_timeout_s=2400.0)

    assert captured[0].timeout == 300.0


def test_warmup_timeout_raises_the_deadline_that_actually_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REGRESSION: --warmup-timeout-s used to reach ONLY the post-``analysis()`` node
    loop, while a cold torch.compile stalls INSIDE ``analysis()``, which python-chess
    bounds with the popen_uci timeout. So the flag could not extend a compile window and
    a 60-game Cheese block died at exactly 300s under ``--warmup-timeout-s 1800``.

    Asserted on BOTH engines: B's warmup never received the flag at all.
    """
    module = _load_match_vs_uci_module()
    timeouts: dict[str, float] = {}
    warmups: dict[str, float] = {}

    class FakeEngine:
        id: ClassVar[dict[str, str]] = {"name": "fake"}
        options: ClassVar[dict[str, object]] = {}

        def quit(self) -> None:
            return None

    def fake_open_engine(spec: str, cwd: str | None = None, *, timeout_s: float = 0.0):
        del cwd
        timeouts[spec] = timeout_s
        return FakeEngine()

    def fake_warmup(eng: object, *, nodes: int | None, label: str, timeout_s: float = -1.0):
        del eng, nodes
        warmups[label] = timeout_s
        if label == "B":  # both engines opened and both warmups reached by now
            raise RuntimeError("stop after warmup")

    monkeypatch.setattr(module, "_open_engine", fake_open_engine)
    monkeypatch.setattr(module, "_warmup_engine", fake_warmup)
    monkeypatch.setattr(sys, "argv", [
        "match_vs_uci.py",
        "--engine-a", "engine-a", "--engine-b", "engine-b",
        "--warmup-nodes", "3000", "--warmup-timeout-s", "1800",
    ])

    with pytest.raises(RuntimeError, match="stop after warmup"):
        module.main()

    # The deadline that aborts the compile is raised, not left at the 300s default.
    assert timeouts["engine-a"] == 1800.0
    assert timeouts["engine-b"] == 1800.0
    # And the flag reaches BOTH warmups; B's call site used to omit it entirely.
    assert warmups["A"] == 1800.0
    assert warmups["B"] == 1800.0


def test_engine_command_timeout_stays_at_the_backstop_without_a_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: no warmup means no reason to relax the hung-engine backstop.

    ⚑ This catches ONE mutation specifically: dropping the ``if nodes is not None``
    conditional so the timeout is relaxed unconditionally, which would silently disarm
    the guard that catches a genuinely wedged engine. Verified: that mutation fails this
    test and ONLY this test.

    It does NOT catch a bumped ``_ENGINE_COMMAND_TIMEOUT_S`` -- an earlier version of
    this docstring claimed it did, and the claim was false because the assertion compared
    against the constant itself, so raising the constant moved the expectation with it.
    Asserting the LITERAL 300.0 is what makes the backstop's VALUE pinned rather than
    self-referential.
    """
    module = _load_match_vs_uci_module()
    timeouts: dict[str, float] = {}

    class FakeEngine:
        id: ClassVar[dict[str, str]] = {"name": "fake"}
        options: ClassVar[dict[str, object]] = {}

        def quit(self) -> None:
            return None

    def fake_open_engine(spec: str, cwd: str | None = None, *, timeout_s: float = 0.0):
        del cwd
        timeouts[spec] = timeout_s
        if spec == "engine-b":
            raise RuntimeError("stop before playing")
        return FakeEngine()

    monkeypatch.setattr(module, "_open_engine", fake_open_engine)
    monkeypatch.setattr(sys, "argv", [
        "match_vs_uci.py",
        "--engine-a", "engine-a", "--engine-b", "engine-b",
        "--warmup-timeout-s", "1800",
    ])

    with pytest.raises(RuntimeError, match="stop before playing"):
        module.main()

    assert timeouts["engine-a"] == 300.0
    assert timeouts["engine-b"] == 300.0
    assert module._ENGINE_COMMAND_TIMEOUT_S == 300.0
