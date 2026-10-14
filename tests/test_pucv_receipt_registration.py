"""Real direct tuples and native loans, with CPU-only CUDA primitives."""
from __future__ import annotations

import contextlib
import dis
import linecache
import sys
import threading
from collections.abc import Generator
from types import FrameType
from typing import Any
from unittest.mock import patch

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine import async_safety
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.mcts import puct_vl
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.uci import multi_gpu_pucv_pool as multi


class AllocationFailure(MemoryError):
    pass


class CpuModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        return {"policy": torch.zeros((x.shape[0], 4672)), "wdl": torch.zeros((x.shape[0], 3))}


class CpuStream:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.syncs = 0
        self.fail = False

    def wait_event(self, event: Any) -> None:
        del event

    def synchronize(self) -> None:
        self.syncs += 1
        if self.fail:
            raise TimeoutError("controlled stream failure")


class CpuEvent:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.syncs = 0
        self.fail = False

    def record(self, stream: Any = None) -> None:
        del stream

    def synchronize(self) -> None:
        self.syncs += 1
        if self.fail and self.syncs == 1:
            raise TimeoutError("controlled completion failure")


class RecordingDirect(DirectGPUEvaluator):
    def __init__(self) -> None:
        super().__init__(CpuModel(), device="cpu", max_batch=4, n_slots=2, use_amp=False, legal_bf16=False)
        self._use_cuda = True
        self.receipts: list[Any] = []

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        receipt = super().evaluate_inplace_async(*args, **kwargs)
        assert type(receipt) is tuple
        assert len(receipt) == 3
        assert all(t.device.type == "cpu" for t in receipt[:2])
        self.receipts.append(receipt)
        return receipt


@contextlib.contextmanager
def cuda_mocks() -> Generator[None]:
    with patch.object(torch.cuda, "Stream", CpuStream), patch.object(torch.cuda, "Event", CpuEvent), \
         patch.object(torch.cuda, "current_stream", return_value=CpuStream()), \
         patch.object(torch.cuda, "stream", side_effect=lambda stream: contextlib.nullcontext()):
        yield


def native_root() -> tuple[Any, int, Any]:
    cb = CBoard.from_board(chess.Board())
    tree = MCTSTree()
    root_id = tree.add_root(1, 0.0, cb)
    legal = cb.legal_move_indices().astype(np.int32)
    tree.expand(root_id, legal, np.full(len(legal), 1.0 / len(legal), dtype=np.float64))
    return tree, root_id, cb


def run_control(kind: str, cache: int, boundary: str, completion_failure: str = "") -> dict[str, Any]:
    ev = RecordingDirect()
    tree, root_id, cb = native_root()
    pipeline = puct_vl.PucvChunker._run_pipelined if kind == "single" else multi.MultiGpuPucvPool._pipeline_until_done
    prepare = puct_vl.PucvChunker._prepare_cached_submit if kind == "single" else multi.MultiGpuPucvPool._prepare_submit
    codes = {pipeline.__code__, prepare.__code__}
    instructions = {code: {i.offset: i for i in dis.get_instructions(code)} for code in codes}
    fired: list[Any] = []
    released: list[Any] = []
    primary = AllocationFailure("controlled receipt ownership boundary")
    original_release = puct_vl.release_batch_rows

    def release(tree_arg: Any, buffers: Any, n_rows: int, **kwargs: Any) -> None:
        # Quiescence must precede native loan release, even on completion failure.
        assert all(r[2]._event.syncs == 1 for r in ev.receipts)
        released.append((type(tree_arg) is MCTSTree, buffers["path_lens"][:n_rows].copy()))
        original_release(tree_arg, buffers, n_rows, **kwargs)

    def tracer(frame: FrameType, event: str, arg: Any) -> Any:
        del arg
        if frame.f_code in codes:
            frame.f_trace_opcodes = True
            ins = instructions[frame.f_code].get(frame.f_lasti)
            local = frame.f_locals
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            is_registration = event == "opcode" and ins is not None and ins.opname == "STORE_SUBSCR" and line.startswith("receipts[")
            is_alias = event == "line" and line == "unregistered_receipt[0] = None"
            cell = local.get("unregistered_receipt", local.get("receipt_cell"))
            is_metadata = event == "opcode" and ins is not None and ins.opname == "BUILD_TUPLE" and cell and cell[0] is not None
            if not fired and len(ev.receipts) == 2 and ((boundary == "registration" and is_registration) or (boundary == "alias" and is_alias) or (boundary == "metadata" and is_metadata)):
                assert cell is not None
                assert cell[0] is ev.receipts[-1]
                assert type(cell[0]) is tuple
                if completion_failure:
                    cell[0][2]._event.fail = True
                    cell[0][2]._stream.fail = completion_failure == "stream"
                fired.append((frame.f_code.co_name, frame.f_lineno))
                raise primary
        return tracer

    pool = None
    old_trace, old_thread_trace = sys.gettrace(), threading.gettrace()
    with cuda_mocks(), patch.object(puct_vl, "release_batch_rows", release):
        helper: Any = None
        caught = None
        threading.settrace(tracer)
        sys.settrace(tracer)
        try:
            if kind == "single":
                helper = puct_vl.PucvChunker(ev, gather=2, eval_cache_entries=cache)
                result = helper.run(tree, root_id, cb, 4)
            else:
                pool = multi.MultiGpuPucvPool(multi.MultiGpuPucvConfig(n_gpus=1, gather=2, eval_cache_entries=cache), evaluators=[ev])
                helper = pool
                result = pool.run(tree=tree, root_id=root_id, root_cboard=cb, target_sims=4, stop_event=threading.Event())
        except AllocationFailure as error:
            caught = error
            result = None
        finally:
            sys.settrace(old_trace)
            threading.settrace(old_thread_trace)
            if pool is not None:
                pool.close()
        if boundary:
            assert caught is primary
            assert len(fired) == 1
            assert result is None
            assert released
            assert all(native and max(lens) > 0 for native, lens in released)
        else:
            assert result == 4
            assert not fired
        assert len(ev.receipts) == 2
        assert all(r[2]._event.syncs == 1 for r in ev.receipts)
        stream_syncs = ev.receipts[-1][2]._stream.syncs
        assert stream_syncs == int(bool(completion_failure))
        if completion_failure:
            assert any("pending receipt drain failed" in note for note in getattr(primary, "__notes__", ()))
        if completion_failure == "stream":
            for owner in (ev, helper):
                with pytest.raises(RuntimeError, match="poisoned"):
                    async_safety.ensure_usable(owner)
            with pytest.raises(RuntimeError, match="poisoned"):
                puct_vl.PucvChunker(ev, gather=2).run(*native_root(), 2)
            assert any(ev.receipts[-1] is item for retained in async_safety._quarantine for item in retained)
        else:
            async_safety.ensure_usable(ev)
            async_safety.ensure_usable(helper)
            ev.get_input_buffer(2, slot=0).fill(17)
        return {"kind": kind, "cache": cache, "boundary": boundary, "completion_failure": completion_failure,
                "result": result, "event_syncs": [r[2]._event.syncs for r in ev.receipts],
                "stream_syncs": stream_syncs, "native_releases": len(released)}


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("cache", [0, 8])
@pytest.mark.parametrize("boundary", ["", "registration", "metadata", "alias"])
def test_exact_receipt_survives_registration_and_metadata(kind: str, cache: int, boundary: str) -> None:
    run_control(kind, cache, boundary)


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("cache", [0, 8])
@pytest.mark.parametrize("completion_failure", ["event", "stream"])
def test_unregistered_receipt_uses_canonical_failure_drain(kind: str, cache: int, completion_failure: str) -> None:
    run_control(kind, cache, "registration", completion_failure)
