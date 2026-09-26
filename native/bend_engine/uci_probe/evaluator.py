"""Exact-package native evaluator for UCI. No eager reference or model export.

A hard abort poisons the native stream: restart the UCI process to reload it.
Never silently substitute diagnostic evaluations for failed neural inference.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import chess
import numpy as np

from native.bend_engine.neural_probe.adapter import HistoryEncoder, probabilities
from native.bend_engine.neural_probe.backend import NativeEvaluator
from native.bend_engine.neural_probe.checkpoint import load_checkpoint, target
from native.bend_engine.neural_probe.qualification import verified_package
from native.bend_engine.session_probe import run_probe as sessions


class PackageEvaluator:
    def __init__(self, checkpoint: Path, package: Path, worker: Path, *, weights: str,
                 device: str, device_index: int, batch: int):
        loaded = load_checkpoint(checkpoint, weights_key=weights)
        target(device, device_index)  # explicit target, never fall back from CUDA
        verified_package(package, loaded, batch=batch, device=device, device_index=device_index)
        self.encoding = loaded.encoding
        self.encoder: HistoryEncoder | None = None
        self.identity: tuple[str, tuple[chess.Move, ...]] | None = None
        self.lock = Lock()
        # Startup is deliberately completed before the UCI handshake/readyok.
        self.worker: NativeEvaluator | None = NativeEvaluator(worker.resolve(), package.resolve())

    def evaluate(self, root: chess.Board, path: list[int], board: sessions.rules.Position,
                 actions: list[int]) -> tuple[list[float], list[float]]:
        identity = (root.root().fen(en_passant='fen'), tuple(root.move_stack))
        if identity != self.identity:
            self.encoder, self.identity = HistoryEncoder(root, self.encoding), identity
        assert self.encoder is not None
        x, full, _ = self.encoder.encode(path, board, actions)
        with self.lock:
            worker = self.worker
            if worker is None or worker.failed:
                raise RuntimeError('neural worker stopped; restart the UCI process')
        padded = np.zeros((worker.batch, self.encoding.channels, 8, 8), dtype=np.float32)
        padded[0] = x[0]
        policy, wdl = worker.evaluate(padded)
        return probabilities(policy[:1], wdl[:1], full)

    def interrupt(self) -> None:
        with self.lock:
            if self.worker is not None:
                self.worker.failed = True
                if self.worker.proc.poll() is None:
                    self.worker.proc.kill()

    def close(self) -> None:
        with self.lock:
            worker, self.worker = self.worker, None
        if worker is not None:
            worker.close()
        self.encoder, self.identity = None, None
