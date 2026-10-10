"""Opt-in finite teacher pool using canonical BT4 semantics and raw writer.

Game IDs and RNG seeds depend on the frozen finite budget, never dispatch order.
Individual completed slots refill after durable acknowledgment, following the
selfplay manager's finalizer. Done outcomes stay owned across writer failures.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import math
import queue
import threading
from typing import Any, Generic, TypeVar

import chess
import numpy as np

from chess_anti_engine.teacher_dispatch import TeacherDispatcher
from scripts.bt4_generation_evaluator import BT4RootOutput
from scripts.bt4_root_policy_stepper import BT4FinalizedGame, BT4RootPolicyStepper, PreparedBatch


class DurableWriter:
    """One daemon writer with a bounded admission count and finite close wait.

    A timeout retains ownership; it neither kills a writer during fsync nor
    releases resources for a replacement owner. The caller decides recovery.
    """

    def __init__(self, *, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("finite positive writer capacity required")
        self._queue: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue(maxsize=capacity)
        self._capacity = capacity
        self._owned = 0
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="TeacherWriter", daemon=True)
        self._thread.start()

    def submit(self, operation: Callable[[], Any]) -> Future[Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("writer closed")
            if self._owned >= self._capacity:
                raise BufferError("durable writer backpressure")
            result: Future[Any] = Future()
            result.set_running_or_notify_cancel()
            self._owned += 1
            self._queue.put_nowait((operation, result))
            return result

    def _run(self) -> None:
        while True:
            try:
                operation, result = self._queue.get(timeout=0.01)
            except queue.Empty:
                with self._lock:
                    if self._closed and self._owned == 0:
                        return
                continue
            try:
                value = operation()
            except BaseException as exc:
                with self._lock:
                    self._owned -= 1
                result.set_exception(exc)
            else:
                with self._lock:
                    self._owned -= 1
                result.set_result(value)

    def close(self, *, timeout: float) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("finite nonnegative writer shutdown timeout required")
        with self._lock:
            self._closed = True
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("durable writer still owned; shutdown budget expired")


@dataclass
class _Game:
    stepper: BT4RootPolicyStepper
    batch: PreparedBatch | None = None
    result: Future[tuple[BT4RootOutput, ...]] | None = None
    finalized: BT4FinalizedGame | None = None
    writing: Future[dict[str, Any]] | None = None


class BT4GamePool:
    """Persistent ready games with bounded, separately scheduled durable writes.

    STOP freezes admission and move/RNG application; prepared roots and results
    remain owned for in-process resume. Completed publication continues at STOP.
    The commit callback must reconcile already durable artifacts on retry.
    """

    def __init__(
        self, *, games: int, capacity: int, seed: int, temperature: float,
        make_stepper: Callable[[int, np.random.Generator], BT4RootPolicyStepper],
        dispatcher: TeacherDispatcher[tuple[chess.Board, np.ndarray], BT4RootOutput],
        commit: Callable[[BT4FinalizedGame], dict[str, Any]], max_writes: int,
        completed: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if (any(type(value) is not int for value in (games, capacity, seed, max_writes))
                or seed < 0 or not math.isfinite(temperature) or temperature < 0
                or not 1 <= capacity <= games or not 1 <= max_writes <= capacity):
            raise ValueError("finite game and writer capacities required")
        self.games = games
        self.capacity = capacity
        self.seed = seed
        self.temperature = temperature
        self.make_stepper = make_stepper
        self.dispatcher = dispatcher
        self.commit = commit
        self.max_writes = max_writes
        self.next_id = 0
        self.active: dict[int, _Game] = {}
        self.receipts = dict(completed or {})
        if any(type(i) is not int or not 0 <= i < games for i in self.receipts):
            raise ValueError("checkpoint game IDs exceed frozen budget")
        self._ready: deque[int] = deque()
        self._writer = DurableWriter(capacity=max_writes)

    def advance(self, *, stopped: bool = False) -> bool:
        """Poll completed work without waiting on any game or backend call."""
        for game_id, game in tuple(self.active.items()):
            if game.writing is not None and game.writing.done():
                writing = game.writing
                game.writing = None  # failures retain finalized outcome for retry
                receipt = writing.result()
                self.receipts[game_id] = receipt
                del self.active[game_id]
        writing_count = sum(game.writing is not None for game in self.active.values())
        for game_id, game in tuple(self.active.items()):
            if game.finalized is not None:
                if game.writing is None and writing_count < self.max_writes:
                    finalized = game.finalized
                    game.writing = self._writer.submit(lambda outcome=finalized: self.commit(outcome))
                    writing_count += 1
                continue
            if stopped:
                continue
            if game.result is not None:
                if not game.result.done():
                    continue
                outputs = game.result.result()
                assert game.batch is not None
                game.stepper.apply_root_outputs(
                    game.batch, outputs, temperatures={game_id: self.temperature},
                )
                game.result = None
                game.batch = None
            if game.batch is None:
                game.batch, finished = game.stepper.prepare_roots()
                if finished:
                    if len(finished) != 1 or finished[0].slot_id != game_id:
                        raise ValueError("canonical per-game outcome identity changed")
                    game.finalized = finished[0]
                    continue
                if game.batch is not None:
                    self._ready.append(game_id)
        while self._ready and not stopped:
            game_id = self._ready[0]
            game = self.active[game_id]
            if game.batch is not None:
                boards, inputs = game.batch.inference_inputs()
                try:
                    game.result = self.dispatcher.submit(
                        f"bt4:{game_id}:{game.batch.generation}", list(zip(boards, inputs)),
                    )
                except BufferError:
                    # Keep prepared root; no second prepare or RNG/move advance.
                    break
                self._ready.popleft()
        while not stopped and len(self.active) < self.capacity and self.next_id < self.games:
            game_id = self.next_id
            if game_id in self.receipts:
                self.next_id += 1
                continue
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, game_id]))
            stepper = self.make_stepper(game_id, rng)
            self.active[game_id] = _Game(stepper)
            self.next_id += 1
        return bool(self.active or self.next_id < self.games)

    def close(self, *, timeout: float = 30) -> None:
        self._writer.close(timeout=timeout)


Root = TypeVar("Root")
Label = TypeVar("Label")


class CompletedGameLabels(Generic[Root, Label]):
    """Pipeline immutable completed-game label requests across shared batches.

    The roots are supplied by the existing saved-history verifier and the
    backend keeps its uint8 TPG / raw FP16 three-head contract. Publication
    consumes a complete per-game result and must reconcile NPZ orphans before
    returning. Failure retains that exact result; it never resubmits inference.
    """

    def __init__(
        self, dispatcher: TeacherDispatcher[Root, Label], *, max_games: int,
        total_games: int, max_game_rows: int,
        publish: Callable[[int, str, tuple[Label, ...]], Any],
    ) -> None:
        if (any(type(value) is not int for value in (max_games, total_games, max_game_rows))
                or not 1 <= max_games <= total_games or max_game_rows < 1):
            raise ValueError("bounded game label publication required")
        self.dispatcher = dispatcher
        self.max_games = max_games
        self.max_game_rows = max_game_rows
        self.total_games = total_games
        self.publish = publish
        self.pending: dict[int, tuple[str, Future[tuple[Label, ...]]]] = {}
        self.durable: dict[int, str] = {}
        self._writer = DurableWriter(capacity=1)
        self._writing: tuple[int, Future[Any]] | None = None

    def submit(self, game_id: int, raw_sha: str, roots: Sequence[Root]) -> None:
        if (game_id in self.pending or game_id in self.durable or type(game_id) is not int
                or not 0 <= game_id < self.total_games or len(raw_sha) != 64
                or any(c not in "0123456789abcdef" for c in raw_sha)
                or len(roots) > self.max_game_rows):
            raise ValueError("distinct exact immutable game request required")
        if len(self.pending) >= self.max_games:
            raise BufferError("completed-game publication backpressure")
        future = self.dispatcher.submit(f"ceres:{game_id}:{raw_sha}", roots)
        self.pending[game_id] = raw_sha, future

    def drain(self, *, stopped: bool = False) -> list[int]:
        if stopped:
            return []
        completed = []
        if self._writing is not None:
            game_id, writing = self._writing
            if not writing.done():
                return completed
            self._writing = None
            writing.result()  # failure retains exact completed result for retry
            raw_sha, _future = self.pending[game_id]
            self.durable[game_id] = raw_sha
            del self.pending[game_id]
            completed.append(game_id)
        for game_id, (raw_sha, future) in tuple(self.pending.items()):
            if future.done():
                values = future.result()
                writing = self._writer.submit(
                    lambda identity=game_id, digest=raw_sha, labels=values: self.publish(identity, digest, labels),
                )
                self._writing = game_id, writing
                break
        return completed

    def close(self, *, timeout: float = 30) -> None:
        self._writer.close(timeout=timeout)
