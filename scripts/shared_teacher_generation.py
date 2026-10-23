"""Canonical teacher domains on the shared ready/completion session runtime.

Unit pools namespace IDs, RNG seeds and durable receipts, never CPU batches.
Individual games refill after durable acknowledgment; outcomes remain owned
across writer/ACK failures. Backend and writer progress is independent of CPU
producer capacity, including when another domain blocks on neural inference.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import chess
import numpy as np

from chess_anti_engine.completion import AcceptedCompletion, drain_completed_slots
from chess_anti_engine.teacher_dispatch import TeacherDispatcher
from chess_anti_engine.producer_executor import ProducerExecutor, ProducerJoinTimeout
from chess_anti_engine.generation_runtime import (
    AdvanceResult, ExclusiveLaneRegistry, LaneLease, ReservationCredits,
)
if TYPE_CHECKING:
    from scripts.bt4_generation_evaluator import BT4RootOutput
    from scripts.bt4_root_policy_stepper import BT4FinalizedGame, BT4RootPolicyStepper, PreparedBatch


class DurableWriter:
    """Bounded daemon writers; default one worker preserves FIFO execution.

    A timeout retains ownership; it neither kills a writer during fsync nor
    releases resources for a replacement owner. The caller decides recovery.
    Explicit parallel workers preserve FIFO admission, not completion order.
    """

    def __init__(self, *, capacity: int, workers: int = 1) -> None:
        if (type(capacity) is not int or capacity < 1
                or type(workers) is not int or workers not in (1, 2, 4) or workers > capacity):
            raise ValueError("finite writer capacity and 1/2/4 workers within capacity required")
        self._queue: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue(maxsize=capacity)
        self._capacity = capacity
        self._owned = 0
        self._closed = False
        self._lock = threading.Lock()
        self._threads = tuple(threading.Thread(target=self._run,
            name="TeacherWriter" if workers == 1 else f"TeacherWriter-{index}", daemon=True)
            for index in range(workers))
        self._thread = self._threads[0]  # Legacy single-worker inspection only.
        try:
            for thread in self._threads:
                thread.start()
        except BaseException as exc:
            with self._lock:
                self._closed = True
            deadline = time.monotonic() + 1
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join(max(0, deadline - time.monotonic()))
            if self.is_alive:
                raise ProducerJoinTimeout("writer startup failed; ownership retained", self) from exc
            raise

    @property
    def available_capacity(self) -> int:
        """Advisory credit snapshot; only submit atomically admits an operation."""
        with self._lock:
            if self._closed:
                raise RuntimeError("writer closed")
            return self._capacity - self._owned

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
                    if self._closed:
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

    @property
    def is_alive(self) -> bool:
        """Whether background work still owns its backend/resource state."""
        return any(thread.is_alive() for thread in self._threads)

    def close(self, *, timeout: float) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("finite nonnegative writer shutdown timeout required")
        with self._lock:
            self._closed = True
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        if self.is_alive:
            raise TimeoutError("durable writer still owned; shutdown budget expired")


@dataclass
class _Game:
    stepper: BT4RootPolicyStepper
    batch: PreparedBatch | None = None
    result: Future[tuple[BT4RootOutput, ...]] | None = None
    finalized: BT4FinalizedGame | None = None
    writing: Future[dict[str, Any]] | None = None
    raw_receipt: dict[str, Any] | None = None
    acknowledgment: Future[dict[str, Any]] | None = None


class BT4GamePool:
    """Per-unit quota, receipt and durable-writer ownership for a shared driver.

    This container does not schedule boards or implement another actor loop.
    The commit callback must reconcile already durable artifacts on retry.
    """

    def __init__(
        self, *, games: int, capacity: int, seed: int, temperature: float,
        make_stepper: Callable[[int, np.random.Generator], BT4RootPolicyStepper],
        dispatcher: TeacherDispatcher[tuple[chess.Board, np.ndarray], BT4RootOutput],
        commit: Callable[[BT4FinalizedGame], dict[str, Any]], max_writes: int,
        completed: dict[int, dict[str, Any]] | None = None,
        namespace: str = "", writer: DurableWriter | None = None, writer_threads: int = 1,
        acknowledge: Callable[[int, dict[str, Any]], dict[str, Any] | Future[dict[str, Any]]] | None = None,
    ) -> None:
        if (any(type(value) is not int for value in (games, capacity, seed, max_writes))
                or seed < 0 or not math.isfinite(temperature) or temperature < 0
                or not 1 <= capacity <= games or not 1 <= max_writes <= capacity):
            raise ValueError("finite game and writer capacities required")
        if type(namespace) is not str or ":" in namespace or len(namespace.encode()) > 128:
            raise ValueError("bounded unit namespace required")
        self.games = games
        self.capacity = capacity
        self.seed = seed
        self.temperature = temperature
        self.make_stepper = make_stepper
        self.dispatcher = dispatcher
        self.commit = commit
        self.max_writes = max_writes
        self.namespace = namespace
        self.writer_seconds = 0.0
        self._writer_seconds_lock = threading.Lock()
        self.acknowledge = acknowledge
        self.next_id = 0
        self.active: dict[int, _Game] = {}
        self.receipts = dict(completed or {})
        if any(type(i) is not int or not 0 <= i < games for i in self.receipts):
            raise ValueError("checkpoint game IDs exceed frozen budget")
        self._owns_writer = writer is None
        self._writer = writer if writer is not None else DurableWriter(capacity=max_writes, workers=writer_threads)

    def _commit(self, game: BT4FinalizedGame) -> dict[str, Any]:
        started = time.monotonic()
        try:
            return self.commit(game)
        finally:
            elapsed = time.monotonic() - started
            with self._writer_seconds_lock:
                self.writer_seconds += elapsed

    def drain_writes(
        self, *, stopped: bool = False, should_stop: Callable[[], bool] = lambda: False,
        game_ids: Sequence[int] | None = None,
    ) -> None:
        """Collect acknowledgments without admitting writes, roots or games."""
        def poll(item: tuple[int, _Game]) -> AcceptedCompletion[dict[str, Any]] | None:
            game_id, game = item
            if game.writing is not None and game.writing.done():
                writing = game.writing
                game.writing = None  # failures retain finalized outcome for retry
                game.raw_receipt = writing.result()
            if game.raw_receipt is None:
                return None
            receipt = game.raw_receipt
            if self.acknowledge is not None:
                if game.acknowledgment is None:
                    if stopped or should_stop():
                        return None  # no new companion admission after STOP
                    try:
                        ack = self.acknowledge(game_id, dict(receipt))
                    except BufferError:
                        return None
                    if isinstance(ack, Future):
                        game.acknowledgment = ack
                    else:
                        if any(ack.get(key) != value for key, value in receipt.items()):
                            raise ValueError("acknowledgment changed canonical raw receipt")
                        return AcceptedCompletion(ack)
                if not game.acknowledgment.done():
                    return None
                ack = game.acknowledgment.result()
                if any(ack.get(key) != value for key, value in receipt.items()):
                    raise ValueError("acknowledgment changed canonical raw receipt")
                receipt = ack
            return AcceptedCompletion(receipt)

        def release(item: tuple[int, _Game], receipt: dict[str, Any]) -> None:
            game_id, _game = item
            self.receipts[game_id] = receipt
            del self.active[game_id]

        items = (self.active.items() if game_ids is None else
                 ((game_id, self.active[game_id]) for game_id in game_ids))
        drain_completed_slots(items, poll=poll, release=release)

    def _admit_writes(self, game_ids: Sequence[int] | None = None) -> None:
        writing_count: int | None = None
        games = (self.active.values() if game_ids is None else
                 (self.active[game_id] for game_id in game_ids))
        for game in games:
            if game.finalized is None or game.raw_receipt is not None or game.writing is not None:
                continue
            # Saturated shared writers need no active-map scan or failed submit
            # per waiting game. Completion Futures already wake the controller.
            if not self._writer.available_capacity:
                return
            if writing_count is None:
                writing_count = sum(active.writing is not None for active in self.active.values())
            if writing_count >= self.max_writes:
                return
            finalized = game.finalized
            try:
                game.writing = self._writer.submit(lambda outcome=finalized: self._commit(outcome))
            except BufferError:
                return  # Snapshot raced another submit; ownership remains here.
            writing_count += 1

    def close(self, *, timeout: float = 30) -> None:
        if self._owns_writer:
            self._writer.close(timeout=timeout)


class BT4GenerationDriver:
    """Ready individual games, not units, lease bounded native CPU microbatches.

    Unit pools remain namespaces for deterministic seeds, quota, canonical
    writer/ACK and resume receipts. Only this owner-thread driver changes those
    registries. A CPU lease owns its individual game fields until settlement.
    The sole neural FIFO accepts independent game Futures in one bounded prefix.
    """

    def __init__(
        self, pools: dict[str, BT4GamePool], *, producers: ProducerExecutor,
        native_microbatch: int, max_live_games: int, retained_byte_budget: int,
        reservation_bytes: dict[str, int], should_stop: Callable[[], bool],
        wake: Callable[[], None] = lambda: None,
    ) -> None:
        if type(native_microbatch) is not int or not 1 <= native_microbatch <= 256:
            raise ValueError("native CPU microbatch must be within 1..256")
        if set(reservation_bytes) != set(pools):
            raise ValueError("exact per-unit trajectory reservations required")
        requested = sum(pool.capacity * reservation_bytes[name] for name, pool in pools.items())
        if requested > retained_byte_budget:
            raise ValueError("requested live reservoir exceeds retained trajectory byte budget")
        self.pools = pools
        self.producers = producers
        self.native_microbatch = native_microbatch
        self.max_live_games = max_live_games
        self.credits = ReservationCredits(retained_byte_budget)
        self.reservation_bytes = reservation_bytes
        self.should_stop = should_stop
        self.wake = wake
        self.lanes = ExclusiveLaneRegistry()
        self._ready: deque[tuple[str, int]] = deque()
        self._inference: deque[tuple[str, int]] = deque()
        self._notifications: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        self._tasks: dict[Future[Any], tuple[LaneLease, ...]] = {}
        self._publications: dict[tuple[str, int], _Game] = {}
        self._watched: set[Future[Any]] = set()
        self.paused = False
        self.started = False
        self.live = 0
        self.peak_live = 0
        self.cpu_microbatches: dict[int, int] = {}

    def _observe_stop(self) -> bool:
        self.paused = self.paused or self.should_stop() or self.producers.stop_event.is_set()
        if self.paused:
            self.producers.request_stop()
        return self.paused

    def _fill(self, name: str) -> bool:
        pool = self.pools[name]
        progress = False
        while len(pool.active) < pool.capacity and pool.next_id < pool.games:
            if self._observe_stop():
                break
            game_id = pool.next_id
            if game_id in pool.receipts:
                pool.next_id += 1
                continue
            key = (name, game_id)
            if self.live >= self.max_live_games:
                raise RuntimeError("cross-unit live game capacity violated")
            if not self.credits.reserve(key, self.reservation_bytes[name]):
                break
            try:
                with self.producers.admission() as admitted:
                    if not admitted:
                        self.credits.release(key)
                        break
                    rng = np.random.default_rng(np.random.SeedSequence([pool.seed, game_id]))
                    game = _Game(pool.make_stepper(game_id, rng))
            except BaseException:
                self.credits.release(key)
                raise
            pool.active[game_id] = game
            pool.next_id += 1
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)
            self._ready.append(key)
            progress = True
        return progress

    def _watch(self, future: Future[Any]) -> None:
        if future not in self._watched:
            self._watched.add(future)
            future.add_done_callback(lambda _future: self.wake())

    def _advance_games(self, keys: tuple[tuple[str, int], ...]) -> None:
        from scripts.bt4_root_policy_stepper import BT4RootPolicyStepper

        games = [(self.pools[name], game_id, self.pools[name].active[game_id])
                 for name, game_id in keys]
        if self.producers.stop_event.is_set():
            return
        applications = []
        for pool, game_id, game in games:
            if game.result is not None:
                if not game.result.done() or game.batch is None:
                    raise RuntimeError("unready game entered native CPU lease")
                applications.append((game.stepper, game.batch, game.result.result(),
                                     {game_id: pool.temperature}))
        def admit_transition() -> bool:
            # Called AFTER whole-microbatch validation, immediately before its
            # first RNG draw. A bounded admitted transition may finish after
            # STOP, but STOP never waits on a native/control lock.
            with self.producers.admission() as admitted:
                return admitted
        if BT4RootPolicyStepper.apply_many(applications, admit=admit_transition) is None:
            return
        for _pool, _game_id, game in games:
            if game.result is not None:
                game.result = None
                game.batch = None
        prepared = BT4RootPolicyStepper.prepare_many([game.stepper for _, _, game in games])
        for (_pool, game_id, game), (batch, finished) in zip(games, prepared, strict=True):
            game.batch = batch
            if finished:
                if len(finished) != 1 or finished[0].slot_id != game_id:
                    raise ValueError("canonical per-game outcome identity changed")
                game.finalized = finished[0]

    def _settle_cpu(self) -> bool:
        progress = False
        for future, leases in tuple(self._tasks.items()):
            if not future.done():
                continue
            # An exception retains leases and all dependencies for diagnosis;
            # the producer join confirms quiescence before ordinary teardown.
            if not future.cancelled():
                future.result()
            self.lanes.complete(leases)
            del self._tasks[future]
            for lease in leases:
                # This driver's only lane admissions use (unit, game ID) keys;
                # the shared registry intentionally accepts any Hashable.
                name, game_id = cast(tuple[str, int], lease.identity)
                game = self.pools[name].active[game_id]
                if game.finalized is not None:
                    self._publications[(name, game_id)] = game
                elif game.batch is not None and game.result is None:
                    self._inference.append((name, game_id))
                elif not self.paused:
                    self._ready.append((name, game_id))
            progress = True
        return progress

    def _publish(self) -> bool:
        progress = False
        for key, game in tuple(self._publications.items()):
            name, game_id = key
            pool = self.pools[name]
            pool.drain_writes(stopped=self.paused, should_stop=self._observe_stop,
                              game_ids=(game_id,))
            if game_id not in pool.active:
                del self._publications[key]
                self.credits.release(key)
                self.lanes.forget(key)
                self.live -= 1
                self._fill(name)
                progress = True
                continue
            was_writing = game.writing
            pool._admit_writes((game_id,))
            progress = progress or was_writing is not game.writing
            for future in (game.writing, game.acknowledgment):
                if future is not None:
                    self._watch(future)
        return progress

    def settle_stopped(self) -> None:
        """After producer quiescence, publish admitted outcomes, never new moves."""
        self.paused = True
        self._settle_cpu()
        self._publish()

    def advance(self) -> AdvanceResult:
        self.producers.raise_if_failed()
        self._observe_stop()
        progress = self._settle_cpu()
        self._watched.intersection_update(future for future in self._watched if not future.done())
        if not self.started:
            self.started = True
            for name in self.pools:
                progress = self._fill(name) or progress
        while not self._notifications.empty():
            self._ready.append(self._notifications.get_nowait())
            progress = True
        progress = self._publish() or progress
        if self._observe_stop():
            return AdvanceResult(progress=progress, terminal=True)

        if self._inference:
            keys = tuple(self._inference)
            requests = []
            for name, game_id in keys:
                game = self.pools[name].active[game_id]
                assert game.batch is not None
                requests.append((f"bt4:{name}:{game_id}:{game.batch.generation}",
                                 game.batch.dispatch_inputs()))
            if not self._observe_stop():
                dispatcher = self.pools[keys[0][0]].dispatcher
                futures = dispatcher.submit_many(requests)
                for key, future in zip(keys, futures):
                    name, game_id = key
                    self.pools[name].active[game_id].result = future
                    self._inference.popleft()
                    def completed(_future: Future[Any], identity: tuple[str, int] = key) -> None:
                        self._notifications.put(identity)
                        self.wake()
                    future.add_done_callback(completed)
                progress = progress or bool(futures)

        # CPU credit is independent of unit quota and inference/publication
        # capacity. Never submit more than one operation for the same game.
        while self._ready and len(self._tasks) < self.producers.capacity:
            if self._observe_stop():
                break
            keys = tuple(self._ready.popleft() for _ in range(min(len(self._ready), self.native_microbatch)))
            leases = self.lanes.admit(keys)
            future = self.producers.submit(self._advance_games, keys)
            self._tasks[future] = leases
            self._watch(future)
            self.cpu_microbatches[len(keys)] = self.cpu_microbatches.get(len(keys), 0) + 1
            progress = True
        remaining = self.live > 0 or any(pool.next_id < pool.games for pool in self.pools.values())
        waits = tuple(self._tasks) + tuple(self._watched)
        return AdvanceResult(progress=progress, terminal=not remaining, waits=waits)


def ceres_game_reservation(max_game_rows: int) -> int:
    """Conservative immutable saved-game inputs/results including full histories.

    Every Ceres Root may retain a separate Python board and restoration stack;
    reserve quadratic history storage instead of assuming those states alias.
    No observed RSS or allocator/model guarantee is implied by this schema cap.
    """
    row_bytes = 2 * 64 * 137 + 2 * (1858 + 3 + 3) + 8192
    history_bytes = (max_game_rows * (max_game_rows + 1) // 2) * 2048
    return max_game_rows * row_bytes + history_bytes


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
        retained_byte_budget: int | None = None,
    ) -> None:
        if (any(type(value) is not int for value in (max_games, total_games, max_game_rows))
                or not 1 <= max_games <= total_games or max_game_rows < 1):
            raise ValueError("bounded game label publication required")
        self.dispatcher = dispatcher
        self.max_games = max_games
        self.max_game_rows = max_game_rows
        self.total_games = total_games
        self.publish = publish
        # uint8 TPG feed, its retained output copy, all raw FP16 heads and
        # bounded routing/history metadata. Reserve the whole immutable game.
        self.reservation_bytes = ceres_game_reservation(max_game_rows)
        required = self.reservation_bytes * max_games
        if retained_byte_budget is not None and retained_byte_budget < required:
            raise ValueError("Ceres requested reservoir exceeds retained label byte budget")
        self.credits = ReservationCredits(required if retained_byte_budget is None else retained_byte_budget)
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
        if not self.credits.reserve(game_id, self.reservation_bytes):
            raise BufferError("completed-game retained label backpressure")
        try:
            future = self.dispatcher.submit(f"ceres:{game_id}:{raw_sha}", roots)
        except BaseException:
            self.credits.release(game_id)
            raise
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
            self.credits.release(game_id)
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

    @property
    def is_alive(self) -> bool:
        return self._writer.is_alive

    def close(self, *, timeout: float = 30) -> None:
        self._writer.close(timeout=timeout)
