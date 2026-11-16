"""The last `info ... pv` line before `bestmove X` leads with X.

GUIs and tournament logs show the last PV and score as the engine's choice.
In-search lines rank root moves by visits; `bestmove` is the Gumbel
sequential-halving survivor, which can differ. The search therefore ends on a
line set led by the played move (`SearchWorker.run(final_info=True)`), with
that move's own score, and `Engine._publish_bestmove` covers answers that
replace a searched one (watchdog, parked search, fault fallback, stop during
ponder) with a score-less line for the played move.

Invariant checked on every `go` that ends in `bestmove X` (X != 0000): the last
line with a `pv` and `multipv` absent or 1 has `pv` starting with X. With
MultiPV > 1 the played move is `multipv 1` of the final set.
"""
from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdSetOption, parse_command
from chess_anti_engine.uci.score import q_to_cp
from chess_anti_engine.uci.search import SearchResult, _multipv_lines, _proven_line_mate
from tests.test_uci_compliance_p2 import _MATE_IN_2, _go, _Out, _tiny_worker
from tests.test_uci_degrade_gaps_subprocess import _fault_env, _Session, _tiny_checkpoint
from tests.test_uci_tb_root_c import _TABLES, needs_tables

_INT_FIELDS = ("depth", "seldepth", "multipv", "nodes", "nps", "time", "hashfull", "tbhits")


def _parse_info(line: str) -> dict[str, Any]:
    """Fields of one `info` line; asserts the grammar the engine emits."""
    toks = line.split()
    assert toks[0] == "info", line
    out: dict[str, Any] = {}
    i = 1
    while i < len(toks):
        key = toks[i]
        if key in _INT_FIELDS:
            out[key] = int(toks[i + 1])
            i += 2
        elif key == "score":
            assert toks[i + 1] in ("cp", "mate"), line
            out["score"] = (toks[i + 1], int(toks[i + 2]))
            i += 3
        elif key == "wdl":
            out["wdl"] = tuple(int(t) for t in toks[i + 1:i + 4])
            assert 999 <= sum(out["wdl"]) <= 1001, line  # per-mille rounding
            i += 4
        elif key == "pv":
            j = i + 1
            while j < len(toks) and toks[j] != "string":
                chess.Move.from_uci(toks[j])
                j += 1
            out["pv"] = toks[i + 1:j]
            i = j
        elif key == "string":
            out["string"] = " ".join(toks[i + 1:])
            break
        else:
            raise AssertionError(f"unknown info field {key!r}: {line}")
    return out


def _segments(lines: list[str]) -> list[tuple[list[str], str]]:
    """Split a transcript into (lines before, bestmove move) per answered go."""
    segs: list[tuple[list[str], str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.startswith("bestmove "):
            segs.append((cur, ln.split()[1]))
            cur = []
        else:
            cur.append(ln)
    return segs


def _rank1_pv_lines(seg: list[str]) -> list[dict[str, Any]]:
    infos = [_parse_info(ln) for ln in seg if ln.startswith("info ") and " pv " in ln]
    return [f for f in infos if f.get("pv") and f.get("multipv", 1) == 1]


def _assert_aligned(lines: list[str], *, require_pv: bool = True) -> list[dict[str, Any]]:
    """Assert the invariant on every go in ``lines``; return each go's last rank-1 line."""
    lasts = []
    segs = _segments(lines)
    assert segs, lines
    for seg, best in segs:
        rank1 = _rank1_pv_lines(seg)
        if best == "0000" or (not rank1 and not require_pv):
            continue
        assert rank1, "no info pv line before bestmove:\n" + "\n".join(seg[-20:])
        assert rank1[-1]["pv"][0] == best, (
            f"last info pv {rank1[-1]['pv']} vs bestmove {best}:\n" + "\n".join(seg[-12:]))
        lasts.append(rank1[-1])
    return lasts


# ------------------------------------------------------- the bestmove guard


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> _Out:
    sink = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=sink, stderr=sys.stderr))
    return sink


class _NoWorker:
    """Enough of a worker for `Engine.__init__` and the print helpers."""

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


def _bare_engine() -> Engine:
    return Engine(cast(Any, _NoWorker()))


def _info(engine: Engine, gen: int, pv: tuple[str, ...], multipv: int | None = None) -> None:
    engine._emit_info(nodes=10, elapsed_ms=5, score_cp=12, pv=pv, tbhits=0, score_mate=None,
                      multipv=multipv, wdl=None, gen=gen)


def test_guard_prints_a_scoreless_line_for_a_replaced_answer(out: _Out) -> None:
    engine = _bare_engine()
    _info(engine, 7, ("e2e4", "e7e5"))
    engine._publish_bestmove(7, "d2d4")
    assert out.lines[-2:] == ["info depth 1 pv d2d4", "bestmove d2d4"], out.lines
    _assert_aligned(out.lines)


def test_guard_keeps_a_continuation_that_starts_with_the_move(out: _Out) -> None:
    engine = _bare_engine()
    _info(engine, 3, ("e2e4", "e7e5"))
    engine._publish_bestmove(3, "g1f3", pv=("g1f3", "b8c6"), nodes=40)
    assert out.lines[-2:] == ["info depth 2 nodes 40 pv g1f3 b8c6", "bestmove g1f3"]


@pytest.mark.parametrize("case", ["same_head", "other_go", "no_line", "null_move"])
def test_guard_is_silent_when_nothing_is_stale(out: _Out, case: str) -> None:
    engine = _bare_engine()
    if case != "no_line":
        _info(engine, 4 if case != "other_go" else 3, ("e2e4",))
    engine._publish_bestmove(4, "0000" if case == "null_move" else "e2e4")
    assert [ln for ln in out.lines if ln.startswith("info depth 1 pv")] == []


def test_guard_line_is_multipv_1_under_multipv(out: _Out) -> None:
    engine = _bare_engine()
    engine._options.multi_pv = 3
    _info(engine, 5, ("e2e4",), multipv=1)
    _info(engine, 5, ("d2d4",), multipv=2)
    engine._publish_bestmove(5, "c2c4")
    assert out.lines[-2:] == ["info depth 1 multipv 1 pv c2c4", "bestmove c2c4"]
    # A multipv 2+ line does not count as the last PV.
    _info(engine, 6, ("e2e4",), multipv=1)
    _info(engine, 6, ("c2c4",), multipv=2)
    engine._publish_bestmove(6, "e2e4")
    assert out.lines[-1] == "bestmove e2e4"
    assert "pv e2e4" in out.lines[-3]


def test_guard_ignores_a_line_the_silenced_thread_never_printed(out: _Out) -> None:
    engine = _bare_engine()
    _info(engine, 2, ("e2e4",))
    t = threading.Thread(target=_info, args=(engine, 2, ("d2d4",)))
    engine_module._SILENCED_THREADS.add(t)
    t.start()
    t.join()
    engine._publish_bestmove(2, "e2e4")
    assert out.lines[-1] == "bestmove e2e4", out.lines
    # The printed e2e4 line still stands: no aligned line was needed.
    assert out.lines[-2].startswith("info depth 1 nodes 10 "), out.lines


# ------------------------------------------- in-process real search (tiny net)


def _random_positions(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    while len(fens) < n:
        b = chess.Board()
        for _ in range(rng.randint(6, 50)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over(claim_draw=True):
            fens.append(b.fen())
    return fens


def _expected_line_score(worker: Any, board: chess.Board, move: str) -> tuple[str, int] | None:
    """`score` of ``move``'s own line from the searched tree (None: unvisited)."""
    tree, root = worker._tree, worker._root_id
    idx = worker_index(board, move)
    actions, visits, qs = tree.get_children_q(root, 0.0)
    hit = [i for i, a in enumerate(actions.tolist()) if a == idx]
    if not hit or int(visits[hit[0]]) == 0:
        return None
    mate = _proven_line_mate(tree, root, board, idx)
    if mate is not None:
        return ("mate", mate)
    return ("cp", q_to_cp(0.5 * (float(qs[hit[0]]) + 1.0)))


def worker_index(board: chess.Board, move: str) -> int:
    from chess_anti_engine.moves import move_to_index
    return int(move_to_index(chess.Move.from_uci(move), board))


def _search_positions(out: _Out, fens: list[str], go: str, multipv: int = 1):
    worker = _tiny_worker()
    results: list[SearchResult] = []
    orig = worker.run

    def run(*a: Any, **k: Any) -> SearchResult:
        r = orig(*a, **k)
        results.append(r)
        return r

    worker.run = run  # type: ignore[method-assign]
    engine = Engine(worker)
    try:
        if multipv > 1:
            engine.dispatch(CmdSetOption(name="MultiPV", value=str(multipv)))
        for fen in fens:
            out.lines.clear()
            engine.dispatch(parse_command(f"position fen {fen}"))
            _go(engine, go)
            yield fen, list(out.lines), results[-1], worker
    finally:
        engine.close()
        worker.close()


def test_property_last_pv_and_score_are_the_played_moves_over_200_positions(out: _Out) -> None:
    """Fixed nodes, 200 positions: the last PV leads with bestmove, carries that
    move's own score and the final node count, and the field set matches the
    in-search lines. Also counts the positions where bestmove is NOT the visit
    leader the in-search lines rank first -- the case this guards (non-vacuity)."""
    survivor_not_leader = 0
    score_checked = 0
    for fen, lines, result, worker in _search_positions(
            out, _random_positions(200, 20261116), "go nodes 256"):
        board = chess.Board(fen)
        (last,) = _assert_aligned(lines)
        rank1 = _rank1_pv_lines(_segments(lines)[0][0])
        assert last["nodes"] == result.nodes, (fen, last, result.nodes)
        assert worker._tree is not None
        assert worker._root_id is not None
        leader = _multipv_lines(worker._tree, worker._root_id, 1, 0.0)[0][2][0]
        survivor_not_leader += leader != worker_index(board, result.bestmove_uci)
        if len(rank1) >= 2:
            assert set(rank1[-2]) == set(last), (rank1[-2], last)
        else:
            assert {"depth", "seldepth", "nodes", "nps", "time", "hashfull", "tbhits",
                    "score", "pv"} <= set(last), last
        expected = _expected_line_score(worker, board, result.bestmove_uci)
        if expected is not None:
            assert last["score"] == expected, (fen, last, expected)
            score_checked += 1
    assert survivor_not_leader >= 20, survivor_not_leader
    assert score_checked >= 150, score_checked


def test_final_set_is_not_a_repeat_of_the_last_in_search_set(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Info due every chunk: when the last in-search set already led with the
    played move at the final node count, the final set is not printed again;
    when it led with another move, the final set is printed (and aligned)."""
    import chess_anti_engine.uci.search as search_module
    from chess_anti_engine.uci.time_manager import Deadline

    monkeypatch.setattr(search_module, "_INFO_EMIT_INTERVAL_MS", -10**9)
    worker = _tiny_worker()
    repeats = reprinted = 0
    try:
        for fen in _random_positions(60, 11):
            emitted: list[tuple[int, tuple[str, ...]]] = []

            def record(*, nodes: int, pv: tuple[str, ...], sink=emitted, **_f: Any) -> None:
                sink.append((nodes, pv))

            result = worker.run(
                chess.Board(fen), stop_event=threading.Event(), deadline=Deadline(None),
                max_nodes=256, info_cb=record)
            if result.nodes <= 1:
                continue
            assert emitted[-1][1][0] == result.bestmove_uci, (fen, emitted[-2:])
            final_nodes = [e for e in emitted if e[0] == result.nodes]
            assert len(final_nodes) in (1, 2), (fen, final_nodes)
            if len(final_nodes) == 2:
                assert final_nodes[0][1][0] != result.bestmove_uci, final_nodes
                reprinted += 1
            else:
                repeats += 1
    finally:
        worker.close()
    assert repeats >= 10, (repeats, reprinted)
    assert reprinted >= 10, (repeats, reprinted)


class _VariedEvaluator:
    """Position-dependent WDL, flat policy: child Qs differ from the root Q,
    which the tiny random net (near-constant value) cannot provide."""

    def evaluate_encoded(self, x: Any, relations: Any = None) -> tuple[Any, Any]:
        import numpy as np

        from chess_anti_engine.moves import POLICY_SIZE
        del relations
        n = len(x)
        w = 3.0 * np.sin(0.37 * np.asarray(x).reshape(n, -1).sum(axis=1))
        wdl = np.stack([w, np.zeros(n), -w], axis=1).astype(np.float32)
        return np.zeros((n, POLICY_SIZE), np.float32), wdl


@pytest.mark.parametrize("walkers", [1, 3])
def test_final_line_score_is_the_played_moves_not_the_roots(walkers: int) -> None:
    """The final line's score is the played move's own Q. On classic Gumbel the
    search's returned value already is the survivor's Q; on the walker pool it
    is the root's average, so there the two must differ (non-vacuity)."""
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import Deadline

    worker = SearchWorker(_VariedEvaluator(), device="cpu", n_walkers=walkers,  # type: ignore[arg-type]
                          chunk_sims=64, gumbel_cfg=GumbelConfig(add_noise=False))
    checked = differs_from_root = 0
    try:
        for fen in _random_positions(60, 5):
            board = chess.Board(fen)
            emitted: list[dict[str, Any]] = []

            def record(*, sink=emitted, **f: Any) -> None:
                sink.append(f)

            result = worker.run(board, stop_event=threading.Event(), deadline=Deadline(None),
                                max_nodes=256, info_cb=record)
            if result.nodes <= 1:
                continue
            last = emitted[-1]
            assert last["pv"][0] == result.bestmove_uci
            expected = _expected_line_score(worker, board, result.bestmove_uci)
            if expected is None or expected[0] != "cp":
                continue
            checked += 1
            assert (last["score_mate"], last["score_cp"]) == (None, expected[1]), (fen, last)
            differs_from_root += expected[1] != result.score_cp
    finally:
        worker.close()
    assert checked >= 40, checked
    if walkers > 1:
        assert differs_from_root >= 10, differs_from_root


def test_multipv_final_set_leads_with_the_played_move(out: _Out) -> None:
    checked = 0
    for _fen, lines, result, _w in _search_positions(
            out, _random_positions(40, 7), "go nodes 256", multipv=3):
        _assert_aligned(lines)
        if result.nodes <= 1:  # mate-in-one shortcut: one line, no tree
            continue
        checked += 1
        seg = _segments(lines)[0][0]
        infos = [_parse_info(ln) for ln in seg if " pv " in ln]
        tail = infos[-3:]
        assert [f["multipv"] for f in tail] == [1, 2, 3], tail
        heads = [f["pv"][0] for f in tail]
        assert heads[0] == result.bestmove_uci, heads
        assert len(set(heads)) == 3, heads
        assert all(f["nodes"] == result.nodes for f in tail)
    assert checked >= 30, checked


def test_show_wdl_final_line_has_wdl(out: _Out) -> None:
    worker = _tiny_worker()
    engine = Engine(worker)
    try:
        engine.dispatch(CmdSetOption(name="UCI_ShowWDL", value="true"))
        engine.dispatch(parse_command("position startpos moves e2e4"))
        _go(engine, "go nodes 256")
        (last,) = _assert_aligned(out.lines)
        assert "wdl" in last, last
    finally:
        engine.close()
        worker.close()


def test_solved_root_final_line_is_a_mate_score_for_the_played_move(out: _Out) -> None:
    worker = _tiny_worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command(f"position fen {_MATE_IN_2}"))
        _go(engine, "go nodes 2000")
        (last,) = _assert_aligned(out.lines)
        assert last["score"] == ("mate", 2), last
    finally:
        engine.close()
        worker.close()


def test_fault_fallback_after_info_lines_ends_on_the_fallback_move(
        out: _Out, monkeypatch: pytest.MonkeyPatch) -> None:
    """The search prints a PV for one move and then raises; the fallback plays
    another. The published line names the fallback move, without a score."""
    worker = _tiny_worker()
    engine = Engine(worker)

    def run(_board: chess.Board, **kw: Any) -> SearchResult:
        kw["info_cb"](nodes=64, elapsed_ms=3, score_cp=5, pv=("e2e4", "e7e5"), tbhits=0,
                      score_mate=None, multipv=None, wdl=None)
        raise RuntimeError("injected search fault")

    monkeypatch.setattr(worker, "run", run)
    monkeypatch.setattr(engine, "_bestmove_fallback",
                        lambda *a, **k: ("d2d4", "root_policy"))
    try:
        engine.dispatch(parse_command("position startpos"))
        _go(engine, "go nodes 100")
        _assert_aligned(out.lines)
        seg = _segments(out.lines)[0][0]
        assert _rank1_pv_lines(seg)[-1] == {"depth": 1, "nodes": 0, "pv": ["d2d4"]}
    finally:
        engine.close()
        worker.close()


def _ponder_session(out: _Out, finish: str, calls: list[bool]) -> list[str]:
    worker = _tiny_worker()
    orig = worker.run

    def run(*a: Any, **k: Any) -> SearchResult:
        calls.append(k["final_info"])
        return orig(*a, **k)

    worker.run = run  # type: ignore[method-assign]
    engine = Engine(worker)
    try:
        engine.dispatch(CmdSetOption(name="Ponder", value="true"))
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 4000 btime 4000"))
        time.sleep(0.6)
        engine.dispatch(parse_command(finish))
        thread = engine._search_thread
        if thread is not None:
            thread.join(15)
        return list(out.lines)
    finally:
        engine.close()
        worker.close()


@pytest.mark.parametrize("finish", ["ponderhit", "stop"])
def test_ponder_answer_matches_its_last_pv(out: _Out, finish: str) -> None:
    calls: list[bool] = []
    lines = _ponder_session(out, finish, calls)
    assert sum(ln.startswith("bestmove ") for ln in lines) == 1, lines
    # The ponder phase (the position before the predicted reply) asks for no
    # final line set; the phase after ponderhit does.
    assert calls == ([False, True] if finish == "ponderhit" else [False]), calls
    # During ponder the in-search lines are for the position before the
    # predicted reply (they lead with e7e5); the answer still ends aligned.
    _assert_aligned(lines)


# ------------------------------------------------- subprocess path matrix


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tiny_checkpoint(tmp_path_factory.mktemp("final_info_ckpt"))


_START = "position startpos moves e2e4"
# (position, go, stop_after_s, require_pv) -- every way a classic search ends.
_CLASSIC_GOS: list[tuple[str, str, float | None, bool]] = [
    (_START, "go nodes 300", None, True),
    (_START, "go movetime 800", None, True),
    (_START, "go wtime 3000 btime 3000 winc 0 binc 0", None, True),
    (_START, "go infinite", 1.2, True),
    (_START, "go depth 3", None, True),
    (_START, "go mate 2 movetime 1500", None, True),
    (_START, "go movetime 800 searchmoves a7a6 h7h6 g8f6", None, True),
    (f"position fen {_MATE_IN_2}", "go nodes 3000", None, True),        # solved root
    (f"position fen {_MATE_IN_2}", "go infinite", 1.5, True),           # solved, waits for stop
    ("position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "go nodes 50", None, True),  # mate in 1
    ("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "go nodes 50", None, False),  # mated
    ("position fen 7k/8/8/8/8/8/5P1P/r5K1 w - - 0 1", "go wtime 2000 btime 2000", None, True),  # forced
]


def _run_session(ckpt: Path, tmp_path: Path, args: list[str], gos, *,
                 setup: tuple[str, ...] = (), env: dict[str, str] | None = None,
                 timeout: float = 90.0) -> list[str]:
    sess = _Session(ckpt, tmp_path, args, env or {"FAULT_MODE": "none"})
    try:
        sess.ready()
        for cmd in setup:
            sess.send(cmd)
        sess.send("isready")
        sess.read_until(lambda ln: ln == "readyok", 60)
        transcript: list[str] = []
        for position, go, stop_after, require_pv in gos:
            _move, got = sess.go(position, go, stop_after=stop_after, timeout=timeout)
            _assert_aligned(got, require_pv=require_pv)
            transcript.extend(got)
        sess.quit()
    finally:
        sess.close()
    return transcript


def test_classic_single_device_every_go_kind(ckpt: Path, tmp_path: Path) -> None:
    lines = _run_session(ckpt, tmp_path, ["--device", "cpu"], _CLASSIC_GOS)
    assert len(_segments(lines)) == len(_CLASSIC_GOS)
    mated = _segments(lines)[10]
    assert mated[1] == "0000"
    assert any("score mate 0" in ln for ln in mated[0]), mated


def test_classic_multipv_and_show_wdl(ckpt: Path, tmp_path: Path) -> None:
    gos = [(_START, "go nodes 400", None, True), (_START, "go movetime 700", None, True),
           (_START, "go infinite", 1.0, True)]
    lines = _run_session(
        ckpt, tmp_path, ["--device", "cpu"], gos,
        setup=("setoption name MultiPV value 3", "setoption name UCI_ShowWDL value true"))
    for seg, best in _segments(lines):
        infos = [_parse_info(ln) for ln in seg if ln.startswith("info ") and " pv " in ln]
        assert [f["multipv"] for f in infos[-3:]] == [1, 2, 3]
        assert infos[-3]["pv"][0] == best
        assert all("wdl" in f for f in infos[-3:])


@pytest.mark.parametrize("path", ["walkers", "pool", "rpg", "chain"])
def test_parallel_paths(ckpt: Path, tmp_path: Path, path: str) -> None:
    args = {
        "walkers": ["--device", "cpu", "--walkers", "3"],
        "pool": ["--devices", "cpu,cpu", "--vl-gather", "32"],
        "rpg": ["--devices", "cpu,cpu", "--search-parallel", "gumbel", "--vl-gather", "32"],
        "chain": ["--devices", "cpu,cpu", "--no-multi-gpu-pucv"],
    }[path]
    # The clocked gos may be answered by the deadline watchdog before any line on
    # a slow CPU (root-parallel Gumbel's first schedule is large): no line is
    # required there, only that a printed one is aligned.
    gos = [(_START, "go nodes 400", None, True), (_START, "go movetime 800", None, False),
           (_START, "go wtime 3000 btime 3000", None, False), (_START, "go infinite", 1.0, True),
           (_START, "go movetime 800 searchmoves a7a6 h7h6 g8f6", None, True)]
    _run_session(ckpt, tmp_path, args, gos)


@pytest.mark.parametrize(("path", "where"), [("pool", "pool:1"), ("rpg", "rpg:1")])
def test_after_a_device_fault(ckpt: Path, tmp_path: Path, path: str, where: str) -> None:
    """A device raises mid-search: the pool keeps one device (a 1-device pool),
    root-parallel Gumbel retires to classic Gumbel. Every answer stays aligned,
    the faulted go and the later ones on the degraded path."""
    args = {"pool": ["--devices", "cpu,cpu", "--vl-gather", "32"],
            "rpg": ["--devices", "cpu,cpu", "--search-parallel", "gumbel", "--vl-gather", "32"],
            }[path]
    gos = [(_START, "go movetime 1500", None, False),
           ("position startpos moves e2e4 e7e5", "go nodes 300", None, True),
           ("position startpos moves e2e4 e7e5 g1f3", "go wtime 3000 btime 3000", None, True),
           ("position startpos moves e2e4 e7e5 g1f3 b8c6", "go infinite", 1.0, True)]
    lines = _run_session(ckpt, tmp_path, args, gos, env=_fault_env("raise", where))
    assert any("DEGRADED" in ln or "retir" in ln for ln in lines), lines[-40:]


def test_every_device_failing_mid_search_answers_aligned(ckpt: Path, tmp_path: Path) -> None:
    gos = [(_START, "go movetime 1500", None, False),
           ("position startpos moves e2e4 e7e5", "go nodes 300", None, False),
           ("position startpos moves e2e4 e7e5 g1f3", "go wtime 2500 btime 2500", None, False)]
    lines = _run_session(
        ckpt, tmp_path, ["--devices", "cpu,cpu", "--vl-gather", "32"], gos,
        env={"FAULT_MODE": "raise", "FAULT_WHERE": "all-pool,primary", "FAULT_AFTER": "6"})
    assert any("bestmove_fallback_used" in ln for ln in lines), lines[-40:]


def test_deadline_watchdog_answer_is_aligned(ckpt: Path, tmp_path: Path) -> None:
    gos = [(_START, "go movetime 2500", None, False)]
    lines = _run_session(
        ckpt, tmp_path, ["--devices", "cpu,cpu", "--vl-gather", "32"], gos,
        env={**_fault_env("block", "all-pool,primary"), "FAULT_AFTER": "6"}, timeout=120)
    assert any("deadline watchdog answers" in ln or "did not stop in time" in ln
               or "bestmove_fallback_used" in ln for ln in lines), lines[-40:]


@pytest.mark.parametrize("finish", ["ponderhit", "stop"])
def test_ponder_subprocess(ckpt: Path, tmp_path: Path, finish: str) -> None:
    sess = _Session(ckpt, tmp_path, ["--device", "cpu"], {"FAULT_MODE": "none"})
    try:
        sess.ready()
        sess.send("setoption name Ponder value true")
        sess.send("position startpos moves e2e4 e7e5")
        sess.send("go ponder wtime 3000 btime 3000")
        time.sleep(1.2)
        sess.send(finish)
        got = sess.read_until(lambda ln: ln.startswith("bestmove"), 60)
        _assert_aligned(got)
        sess.quit()
    finally:
        sess.close()


# A 7-man root (no root shortcut) whose leaves reach the tables: the case that
# printed `pv e1e2 ...` then `bestmove e1d2` before the fix.
_TB_SEARCHED = "8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 0 40"
_TB_ROOT = "7k/8/8/8/8/8/8/KR6 w - - 0 1"


@needs_tables
def test_tablebase_root_shortcut_and_tb_leaves(ckpt: Path, tmp_path: Path) -> None:
    gos = [(f"position fen {_TB_SEARCHED}", "go nodes 400", None, True),
           (f"position fen {_TB_SEARCHED}", "go movetime 1000", None, True),
           (f"position fen {_TB_ROOT}", "go nodes 400", None, True),
           (f"position fen {_TB_ROOT}", "go wtime 2000 btime 2000", None, True)]
    _run_session(ckpt, tmp_path, ["--device", "cpu"], gos,
                 setup=(f"setoption name SyzygyPath value {_TABLES}",))
