"""Bounded, single-owner batching state for independent Bend search processes.

One outstanding request per session, one native batch in flight. No within-tree
parallelism or virtual loss. Model execution happens outside this object, allowing
its owner to cancel/expire requests before results return. No locks/background IO
here: all state transitions belong to one event-loop thread.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np

from .adapter import probabilities
from .backend import BATCHES


@dataclass(frozen=True)
class Key:
    session: int
    epoch: int
    request: int
    node: int

    def __post_init__(self) -> None:
        if any(type(x) is not int or not 0 <= x < 2**32
               for x in (self.session, self.epoch, self.request, self.node)):
            raise ValueError('request identity must contain U32 integers')
        if not self.epoch or not self.request:
            raise ValueError('epoch and request must be positive')


@dataclass(frozen=True, eq=False)
class Pending:
    key: Key
    x: np.ndarray
    actions: np.ndarray
    queued_at: float
    deadline: float


@dataclass(frozen=True, eq=False)
class Batch:
    jobs: tuple[Pending, ...]
    x: np.ndarray


@dataclass(frozen=True)
class Completion:
    key: Key
    status: str
    wdl: tuple[float, ...] = ()
    policy: tuple[float, ...] = ()


class Batcher:
    """Queue capacity counts cancelled-but-not-yet-returned in-flight rows too."""
    def __init__(self, batch: int, channels: int, *, capacity: int = 8,
                 max_sessions: int = 8, max_wait: float = 0.002):
        if type(batch) is not int or batch not in BATCHES or channels not in (146, 175):
            raise ValueError('unsupported batch or channels')
        if (type(capacity) is not int or not batch <= capacity <= 64
                or type(max_sessions) is not int or not 1 <= max_sessions <= 64):
            raise ValueError('invalid bounded capacity/session count')
        if not math.isfinite(max_wait) or max_wait < 0:
            raise ValueError('invalid batching wait')
        self.batch, self.channels = batch, channels
        self.capacity, self.max_sessions, self.max_wait = capacity, max_sessions, max_wait
        self.epochs: dict[int, tuple[int, int]] = {}
        self.pending: dict[int, Pending] = {}
        self.queue: OrderedDict[Key, Pending] = OrderedDict()
        self.flight: Batch | None = None

    @property
    def reserved(self) -> int:
        return len(self.queue) + (len(self.flight.jobs) if self.flight else 0)

    def register(self, session: int, epoch: int) -> None:
        Key(session, epoch, 1, 0)
        if session in self.pending:
            raise ValueError('session still has an outstanding request')
        if session not in self.epochs and len(self.epochs) >= self.max_sessions:
            raise BufferError('session capacity reached')
        if epoch <= self.epochs.get(session, (0, 0))[0]:
            raise ValueError('non-increasing session epoch')
        self.epochs[session] = (epoch, 0)

    def submit(self, key: Key, x: np.ndarray, actions: np.ndarray,
               *, now: float, deadline: float) -> None:
        if not math.isfinite(now) or not math.isfinite(deadline) or deadline <= now:
            raise ValueError('request deadline must be finite and future')
        epoch, previous = self.epochs.get(key.session, (0, 0))
        if key.epoch != epoch or key.request <= previous:
            raise ValueError('stale or duplicate request identity')
        if key.session in self.pending:
            raise ValueError('session already has an outstanding request')
        if self.reserved >= self.capacity:
            raise BufferError('batch queue capacity reached')
        if x.shape != (1, self.channels, 8, 8) or not np.isfinite(x).all():
            raise ValueError('invalid request tensor')
        # Validate the full/compact action map before queue admission.
        probabilities(np.zeros((1, 1858)), np.zeros((1, 3)), actions)
        owned = np.array(x, dtype=np.float32, order='C', copy=True)
        if not np.isfinite(owned).all():
            raise ValueError('request tensor overflows F32')
        indices = np.array(actions, copy=True)
        owned.setflags(write=False)
        indices.setflags(write=False)
        job = Pending(key, owned, indices, now, deadline)
        self.pending[key.session] = job
        self.queue[key] = job
        self.epochs[key.session] = (epoch, key.request)

    def record_local(self, key: Key) -> None:
        """Consume a rule-adjudication identity without reserving an inference row.

        Sequence gaps are allowed for cached terminal visits. A previous cancelled
        batch may still own storage, but no live pending request may be bypassed.
        """
        epoch, previous = self.epochs.get(key.session, (0, 0))
        if key.epoch != epoch or key.request <= previous:
            raise ValueError('stale or duplicate local request identity')
        if key.session in self.pending:
            raise ValueError('local reply would bypass an outstanding request')
        self.epochs[key.session] = (epoch, key.request)

    def cancel(self, key: Key, *, status: str = 'cancelled') -> Completion:
        if status not in ('cancelled', 'expired'):
            raise ValueError('invalid cancellation status')
        job = self.pending.get(key.session)
        if job is None or job.key != key:
            raise ValueError('request is not active')
        del self.pending[key.session]
        self.queue.pop(key, None)
        # A cancelled flight still owns its inputs/slot until complete/fail.
        return Completion(key, status)

    def expire(self, now: float) -> list[Completion]:
        if not math.isfinite(now):
            raise ValueError('invalid clock value')
        return [self.cancel(p.key, status='expired') for p in tuple(self.pending.values())
                if now >= p.deadline]

    def dispatch(self, now: float, *, max_rows: int | None = None) -> Batch | None:
        limit = self.batch if max_rows is None else max_rows
        if type(limit) is not int or not 1 <= limit <= self.batch:
            raise ValueError("invalid dispatch row limit")
        if not math.isfinite(now):
            raise ValueError('invalid clock value')
        if self.flight is not None or not self.queue:
            return None
        oldest = next(iter(self.queue.values()))
        if len(self.queue) < limit and now < oldest.queued_at + self.max_wait:
            return None
        # Fixed-shape packages receive zero padding. Only real rows get replies.
        jobs = tuple(list(self.queue.values())[:limit])
        x = np.zeros((self.batch, self.channels, 8, 8), dtype=np.float32)
        for i, p in enumerate(jobs):
            x[i] = p.x[0]
            del self.queue[p.key]
        x.setflags(write=False)
        self.flight = Batch(jobs, x)
        return self.flight

    def _current(self, batch: Batch) -> None:
        if self.flight is None or batch is not self.flight:
            raise ValueError('stale, duplicate or foreign batch completion')

    def complete(self, batch: Batch, policy: np.ndarray, wdl: np.ndarray,
                 *, now: float) -> list[Completion]:
        self._current(batch)
        if not math.isfinite(now):
            raise ValueError('invalid clock value')
        if (policy.shape != (self.batch, 1858) or wdl.shape != (self.batch, 3)
                or not np.isfinite(policy).all() or not np.isfinite(wdl).all()):
            raise ValueError('invalid batched model outputs')
        # Validate/convert the entire batch before changing any pending state.
        results = []
        for i, job in enumerate(batch.jobs):
            if self.pending.get(job.key.session) is not job:
                continue  # cancelled, or a newer epoch now owns the session
            if now >= job.deadline:
                results.append(Completion(job.key, 'expired'))
            else:
                value, prior = probabilities(policy[i:i+1], wdl[i:i+1], job.actions)
                results.append(Completion(job.key, 'ok', tuple(value), tuple(prior)))
        self._retire(batch)
        return results

    def fail(self, batch: Batch) -> list[Completion]:
        self._current(batch)
        results = [Completion(p.key, 'backend_error') for p in batch.jobs
                   if self.pending.get(p.key.session) is p]
        self._retire(batch)
        return results

    def _retire(self, batch: Batch) -> None:
        for p in batch.jobs:
            if self.pending.get(p.key.session) is p:
                del self.pending[p.key.session]
        self.flight = None
