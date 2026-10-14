"""Live VLGather changes reach the executing native PUCT chunker."""
from __future__ import annotations

from chess_anti_engine.uci.protocol import parse_command
from tests.test_uci_search_options import FEN, _DetAsyncEval, _make_engine

import pytest


class _RecordingEval(_DetAsyncEval):
    def __init__(self) -> None:
        super().__init__(175, max_batch=1024)
        self.batches: list[int] = []

    def evaluate_inplace_async(self, bsz: int, *, slot: int = 0):
        self.batches.append(bsz)
        return super().evaluate_inplace_async(bsz, slot=slot)


@pytest.mark.parametrize('commands', [
    ['setoption name UseVL value true', 'setoption name VLGather value 64'],
    ['setoption name VLGather value 64', 'setoption name UseVL value true'],
    ['setoption name UseVL value true', 'setoption name VLGather value 64', 'setoption name UseVL value false', 'setoption name UseVL value true'],
])
def test_live_gather_reaches_native_dispatch(commands: list[str]) -> None:
    ev = _RecordingEval()
    engine = _make_engine(threads=1, evaluator=ev)
    try:
        engine.dispatch(parse_command('setoption name ChunkSims value 128'))
        for command in commands:
            engine.dispatch(parse_command(command))
        engine.dispatch(parse_command('isready'))
        ev.batches.clear()
        engine.dispatch(parse_command('ucinewgame'))
        engine.dispatch(parse_command('position fen ' + FEN))
        engine.dispatch(parse_command('go nodes 128'))
        thread = engine._search_thread
        assert thread is not None
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert engine.bestmove_fallback_used == 0
        assert ev.batches == [64, 64]
        assert engine._worker._pucv is not None
        assert engine._worker._pucv._gather == 64
    finally:
        engine.close()


def test_repeating_same_gather_preserves_chunker() -> None:
    engine = _make_engine(threads=1, evaluator=_RecordingEval())
    try:
        engine.dispatch(parse_command('setoption name UseVL value true'))
        original = engine._worker._pucv
        assert original is not None
        engine.dispatch(parse_command('setoption name VLGather value 512'))
        assert engine._worker._pucv is original
    finally:
        engine.close()
