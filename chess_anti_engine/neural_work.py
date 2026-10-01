"""Neural-work accounting, independent of simulations and backend call counts.

A ledger is owned by ONE bounded search/benchmark run, not a lifetime broker.
Logical cancellation and physical completion are deliberately separate. Budget
admission charges real rows at dispatch and never refunds a failed/cancelled
forward. A backend-only observer must leave acceptance and simulations unknown.
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

Outcome = Literal["accepted", "cancelled", "stale", "rejected", "failed"]
SCHEMA = "deepfin.neural-work.v1"
PHASES = ("selection", "encoding", "queue_wait", "h2d", "gpu", "d2h", "backup")


def _count(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


@dataclass
class WorkCounters:
    """Single-owner aggregate counters; no request history or GPU synchronization.

    padded_rows counts dispatched padding, including unsuccessful forwards.
    executed_real_rows is confirmed completed computation, NOT submitted rows.
    A failed forward's actual completed work can be unknown; never estimate it.
    """

    forward_calls: int = 0
    dispatched_real_rows: int = 0
    executed_real_rows: int = 0
    padded_rows: int = 0
    failed_forward_rows: int = 0
    real_batch_histogram: Counter[int] = field(default_factory=Counter)
    physical_batch_histogram: Counter[int] = field(default_factory=Counter)

    def dispatch(self, real_rows: int, physical_rows: int) -> None:
        _count(real_rows, "real_rows", positive=True)
        _count(physical_rows, "physical_rows", positive=True)
        if physical_rows < real_rows:
            raise ValueError("physical_rows cannot be smaller than real_rows")
        self.forward_calls += 1
        self.dispatched_real_rows += real_rows
        self.padded_rows += physical_rows - real_rows
        self.real_batch_histogram[real_rows] += 1
        self.physical_batch_histogram[physical_rows] += 1

    def snapshot(self, wall_seconds: float) -> dict[str, Any]:
        if not math.isfinite(wall_seconds) or wall_seconds < 0:
            raise ValueError("wall_seconds must be finite and nonnegative")
        return {
            "schema": SCHEMA,
            "wall_seconds": wall_seconds,
            "completed_simulations": None,
            "forward_calls": self.forward_calls,
            "dispatched_real_rows": self.dispatched_real_rows,
            "executed_real_rows": self.executed_real_rows,
            "accepted_neural_rows": None,
            "padded_rows": self.padded_rows,
            "failed_forward_rows": self.failed_forward_rows,
            "cancelled_rows": None,
            "stale_rows": None,
            "rejected_rows": None,
            "failed_rows": None,
            "useful_eps": None,
            "executed_eps": self.executed_real_rows / wall_seconds if wall_seconds else None,
            "padding_fraction": self.padded_rows / (self.dispatched_real_rows + self.padded_rows)
            if self.dispatched_real_rows + self.padded_rows else 0.0,
            "real_batch_histogram": dict(sorted(self.real_batch_histogram.items())),
            "physical_batch_histogram": dict(sorted(self.physical_batch_histogram.items())),
            "phase_seconds": dict.fromkeys(PHASES),
        }


class NeuralBudgetExceeded(RuntimeError):
    """The whole batch would exceed the real-row budget; nothing was dispatched."""


@dataclass
class _Request:
    rows: int
    batch: int | None = None
    executed: bool = False
    outcome: Outcome | None = None


@dataclass
class _Batch:
    requests: tuple[int, ...]
    finished: bool = False


class NeuralWorkLedger:
    """Exactly-once observer and admission guard for a bounded run.

    IDs are caller-supplied, unique within this ledger, and never recycled.
    Submit rejection, duplicate completion and late cancellation cannot mint
    useful work. The lock protects accounting only; no model/search call occurs
    under it. Discard the ledger after the run to bound tombstone storage.
    """

    def __init__(self, *, neural_budget: int | None = None) -> None:
        if neural_budget is not None:
            _count(neural_budget, "neural_budget", positive=True)
        self.neural_budget = neural_budget
        self.counters = WorkCounters()
        self._requests: dict[int, _Request] = {}
        self._batches: dict[int, _Batch] = {}
        self._simulations = 0
        self._phases: dict[str, float] = {}
        self._lock = threading.Lock()

    def submit(self, request_id: int, rows: int = 1) -> None:
        _count(request_id, "request_id")
        _count(rows, "rows", positive=True)
        with self._lock:
            if request_id in self._requests:
                raise ValueError("request ID already used in this run")
            self._requests[request_id] = _Request(rows)

    def dispatch(self, batch_id: int, requests: tuple[int, ...], *, physical_rows: int) -> None:
        _count(batch_id, "batch_id")
        _count(physical_rows, "physical_rows", positive=True)
        with self._lock:
            if batch_id in self._batches or not requests or len(set(requests)) != len(requests):
                raise ValueError("batch ID reused or request list empty/duplicated")
            entries = [self._requests[key] for key in requests]
            if any(entry.batch is not None or entry.outcome is not None for entry in entries):
                raise ValueError("request already dispatched or disposed")
            rows = sum(entry.rows for entry in entries)
            if physical_rows < rows:
                raise ValueError("physical_rows cannot be smaller than real_rows")
            if self.neural_budget is not None and self.counters.dispatched_real_rows + rows > self.neural_budget:
                raise NeuralBudgetExceeded("batch exceeds remaining real neural rows")
            self.counters.dispatch(rows, physical_rows)
            for entry in entries:
                entry.batch = batch_id
            self._batches[batch_id] = _Batch(requests)

    def complete(self, batch_id: int, *, success: bool = True) -> bool:
        with self._lock:
            batch = self._batches[batch_id]
            if batch.finished:
                return False
            batch.finished = True
            for key in batch.requests:
                entry = self._requests[key]
                if success:
                    entry.executed = True
                    self.counters.executed_real_rows += entry.rows
                else:
                    self.counters.failed_forward_rows += entry.rows
                    if entry.outcome is None:
                        entry.outcome = "failed"
            return True

    def resolve(self, request_id: int, outcome: str) -> bool:
        if outcome not in ("accepted", "cancelled", "stale", "rejected", "failed"):
            raise ValueError("unknown request outcome")
        with self._lock:
            entry = self._requests[request_id]
            if entry.outcome is not None:
                return False
            if outcome == "accepted" and not entry.executed:
                raise ValueError("cannot accept a row before confirmed execution")
            entry.outcome = outcome
            return True

    def simulations_completed(self, count: int = 1) -> None:
        _count(count, "count")
        with self._lock:
            self._simulations += count

    def record_phase(self, name: str, seconds: float) -> None:
        if name not in PHASES or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid phase measurement")
        with self._lock:
            self._phases[name] = self._phases.get(name, 0.0) + seconds

    def snapshot(self, wall_seconds: float) -> dict[str, Any]:
        with self._lock:
            report = self.counters.snapshot(wall_seconds)
            outcomes: Counter[str] = Counter()
            unresolved = wasted = 0
            for entry in self._requests.values():
                if entry.outcome is not None:
                    outcomes[entry.outcome] += entry.rows
                else:
                    unresolved += entry.rows
                if entry.executed and entry.outcome not in (None, "accepted"):
                    wasted += entry.rows
            report.update({
                "completed_simulations": self._simulations,
                "accepted_neural_rows": outcomes["accepted"],
                "cancelled_rows": outcomes["cancelled"],
                "stale_rows": outcomes["stale"],
                "rejected_rows": outcomes["rejected"],
                "failed_rows": outcomes["failed"],
                "unresolved_rows": unresolved,
                "executed_wasted_rows": wasted,
                "useful_eps": outcomes["accepted"] / wall_seconds if wall_seconds else None,
                "neural_budget": self.neural_budget,
                "phase_seconds": {name: self._phases.get(name) for name in PHASES},
            })
            return report
