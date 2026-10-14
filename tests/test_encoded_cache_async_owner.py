"""Production-factory cache wrappers preserve async failure owner identity."""
from __future__ import annotations

import contextlib
import threading
from typing import Any, cast

import chess
import pytest
import torch

from chess_anti_engine import async_safety
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts import puct_vl
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.uci.__main__ import _make_evaluator_factory
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


class Model(torch.nn.Module):
    input_extra_features = "v2_threats"

    def forward(self, x, **_kwargs):
        return {"policy": torch.zeros((len(x), 4672)), "wdl": torch.zeros((len(x), 3))}


class Stream:
    def __init__(self, **_kwargs):
        self.syncs = 0
        self.fail = False

    def wait_event(self, event):
        del event

    def synchronize(self):
        self.syncs += 1
        if self.fail:
            raise TimeoutError("cache owner stream control")


class Event:
    def __init__(self, **_kwargs):
        self.syncs = 0
        self.fail = False
        self.error = TimeoutError("cache owner completion control")

    def record(self, stream=None):
        del stream

    def synchronize(self):
        self.syncs += 1
        if self.fail:
            raise self.error


def usable(owner):
    try:
        async_safety.ensure_usable(owner)
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("cache", [0, 8])
@pytest.mark.parametrize("fail_drain", [False, True])
def test_actual_factory_cache_owner_failure_and_retry(monkeypatch, cache, fail_drain):
    # Real factory construction and CPU warmup; fake only CUDA primitives.
    factory = _make_evaluator_factory([Model()], ["cpu"], False, 1, 2)
    owner = cast(Any, factory(128, eval_cache_entries=cache))
    assert isinstance(owner, EncodedEvalCache if cache else ThreadSafeGPUDispatcher)
    dispatcher = cast(Any, owner)._inner if cache else owner
    assert type(dispatcher) is ThreadSafeGPUDispatcher
    backend = dispatcher._eval
    assert type(backend) is DirectGPUEvaluator
    backend._use_cuda = True
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args, **_kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "stream", contextlib.nullcontext)

    receipts = []
    releases = []
    closes = []
    original_submit = backend.evaluate_inplace_async
    original_release = puct_vl.release_batch_rows

    def submit(*args, **kwargs):
        raw = cast(Any, original_submit(*args, **kwargs))
        assert type(raw) is tuple
        assert len(raw) == 3
        receipts.append(raw)
        if len(receipts) == 1:
            raw[2]._event.fail = True
            raw[2]._stream.fail = fail_drain
        return raw

    def release(tree, buffers, rows, **kwargs):
        assert type(tree) is MCTSTree
        releases.append(([raw[2]._event.syncs for raw in receipts], buffers["path_lens"][:rows].copy()))
        return original_release(tree, buffers, rows, **kwargs)

    monkeypatch.setattr(backend, "evaluate_inplace_async", submit)
    monkeypatch.setattr(puct_vl, "release_batch_rows", release)
    monkeypatch.setattr(owner, "close", lambda: closes.append(True), raising=False)
    worker = SearchWorker(owner, device="cpu", chunk_sims=4, eval_cache_entries=cache)
    worker.set_use_pucv(True, gather=2)

    def search():
        return worker.run(chess.Board(), stop_event=threading.Event(),
                          deadline=Deadline(None), max_nodes=4, allow_terminal_shortcuts=False)

    try:
        with pytest.raises(TimeoutError, match="cache owner completion control") as caught:
            search()
        assert caught.value is receipts[0][2]._event.error
        assert [raw[2]._event.syncs for raw in receipts] == [2, 1]
        assert receipts[0][2]._stream.syncs == 2
        assert releases
        assert all(syncs == [2, 1] for syncs, _paths in releases)
        assert all(max(paths) > 0 for _syncs, paths in releases)
        if fail_drain:
            assert id(backend) not in getattr(caught.value, "_async_quiesced_owners", ())
            submits = len(receipts)
            with pytest.raises(RuntimeError, match="poisoned; restart required"):
                search()
            assert len(receipts) == submits
            assert any(receipts[0] is value for retained in async_safety._quarantine for value in retained)
        else:
            result = search()
            assert result.nodes == 4
            assert chess.Move.from_uci(result.bestmove_uci) in chess.Board().legal_moves
            assert id(backend) in getattr(caught.value, "_async_quiesced_owners", ())
            assert async_safety._backend(owner) is backend
        assert usable(backend) == (not fail_drain)
        assert usable(owner) == (not fail_drain)
        assert usable(worker._pucv) == (not fail_drain)
    finally:
        worker.close(close_evaluator=False)
        assert closes == [], "the caller borrows the factory evaluator"
