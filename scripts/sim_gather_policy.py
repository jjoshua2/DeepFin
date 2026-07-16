#!/usr/bin/env python3
"""Offline gather-policy simulator for SlotBroker arrival traces.

Replays a policy-independent arrival stream (from CAE_ARRIVAL_TRACE JSONL, or
a synthetic Poisson selftest) through several batch-gather policies against a
single-GPU serial forward model. Pure Python + numpy — does not import the
broker.

Usage:
  python3 scripts/sim_gather_policy.py trace.jsonl
  python3 scripts/sim_gather_policy.py --selftest
  # If both a trace path and --selftest are given, --selftest wins (no file load).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Copied from chess_anti_engine/inference.py ``_COMPILED_BATCH_BUCKETS`` —
# keep in sync manually; this script deliberately does not import the broker.
_COMPILED_BATCH_BUCKETS: tuple[int, ...] = (
    16,
    32,
    64,
    96,
    128,
    170,
    256,
    340,
    384,
    512,
    680,
    768,
    1020,
    1024,
    1190,
    1536,
    1792,
    2048,
    2336,
    2720,
    4096,
)

# Measured AOT forward latency (ms) by padded bucket size (docs / bench notes).
# Sizes 16/32/64/96/128 use table values; >=96 missing keys use the linear fit
# through (96, 7.10) and (128, 10.77). Keep in sync with measured tables manually.
_LATENCY_TABLE_MS: dict[int, float] = {
    16: 3.89,
    32: 4.26,
    64: 4.90,
    96: 7.10,
    128: 10.77,
}

# Linear fit through (96, 7.10) and (128, 10.77) for buckets not in the table.
_LAT_SLOPE = (10.77 - 7.10) / (128 - 96)
_LAT_FLOOR = 7.10 - _LAT_SLOPE * 96

_MAX_BUCKET = _COMPILED_BATCH_BUCKETS[-1]
_BUCKET_INDEX: dict[int, int] = {b: i for i, b in enumerate(_COMPILED_BATCH_BUCKETS)}


def pad_bucket(rows: int) -> int:
    """Smallest ladder bucket >= rows; clamp to largest if rows exceed ladder."""
    if rows <= 0:
        return _COMPILED_BATCH_BUCKETS[0]
    for b in _COMPILED_BATCH_BUCKETS:
        if b >= rows:
            return b
    return _MAX_BUCKET


def fwd_ms(rows: int) -> float:
    """Forward latency (ms) for a batch of ``rows`` after ladder padding."""
    bucket = pad_bucket(rows)
    table = _LATENCY_TABLE_MS.get(bucket)
    if table is not None:
        return float(table)
    return float(_LAT_FLOOR + _LAT_SLOPE * bucket)


def next_bucket(bucket: int) -> int:
    """Next larger ladder bucket, or the same bucket if already at the top."""
    idx = _BUCKET_INDEX.get(bucket)
    if idx is None:
        return pad_bucket(bucket)
    if idx + 1 < len(_COMPILED_BATCH_BUCKETS):
        return _COMPILED_BATCH_BUCKETS[idx + 1]
    return bucket


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------


def load_trace(path: str) -> list[tuple[float, int]]:
    """Flatten JSONL arrival records into a sorted (t_arrival, rows) stream."""
    out: list[tuple[float, int]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                print(
                    f"warning: skip malformed JSON at {path}:{lineno}: {exc}",
                    file=sys.stderr,
                )
                continue
            arrivals = obj.get("arrivals") if isinstance(obj, dict) else None
            if not isinstance(arrivals, list):
                print(
                    f"warning: skip line {lineno}: missing/invalid 'arrivals'",
                    file=sys.stderr,
                )
                continue
            for pair in arrivals:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                try:
                    t_arr = float(pair[0])
                    rows = int(pair[1])
                except (TypeError, ValueError):
                    continue
                if rows > 0:
                    out.append((t_arr, rows))
    out.sort(key=lambda x: x[0])
    return out


def synthetic_poisson_stream(
    *,
    n_arrivals: int = 5000,
    rate_per_ms: float = 0.35,
    mean_rows: float = 18.0,
    seed: int = 0,
) -> list[tuple[float, int]]:
    """Poisson inter-arrival process (absolute seconds, synthetic clock).

    Default load ~0.35 arr/ms × 18 rows ≈ 6300 rows/s — high enough that
    padding waste affects makespan (so rows/s ranks policies) but below the
    ~8–9k rows/s GPU ceiling of the latency model (avoids unbounded backlog).
    """
    rng = np.random.default_rng(seed)
    gaps_ms = rng.exponential(1.0 / rate_per_ms, size=n_arrivals)
    t_ms = np.cumsum(gaps_ms)
    rows = np.maximum(1, rng.poisson(mean_rows, size=n_arrivals)).astype(np.int64)
    t0 = 1000.0
    return [
        (t0 + float(t) / 1000.0, int(r)) for t, r in zip(t_ms, rows, strict=True)
    ]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class GatherPolicy(Protocol):
    """Common interface: return absolute sim time (ms) to dispatch."""

    def choose_dispatch_ms(
        self,
        *,
        now_ms: float,
        queue_rows: int,
        queue_oldest_ms: float,
        future: Sequence[tuple[float, int]],
        past_arrivals: Sequence[tuple[float, int]],
        queue: Sequence[tuple[float, int]],
    ) -> float:
        """Return dispatch time >= now_ms. GPU is free at now_ms; queue non-empty."""
        ...


@dataclass(frozen=True)
class FixedWindow:
    """Wait up to ``window_ms`` from GPU-free for the queue to fill, then dispatch.

    When the queue is empty at GPU-free the simulator waits for the first
    arrival and dispatches immediately (this class is only invoked with a
    non-empty queue).
    """

    window_ms: float

    @property
    def name(self) -> str:
        return f"FixedWindow({self.window_ms:g}ms)"

    def choose_dispatch_ms(
        self,
        *,
        now_ms: float,
        queue_rows: int,
        queue_oldest_ms: float,
        future: Sequence[tuple[float, int]],
        past_arrivals: Sequence[tuple[float, int]],
        queue: Sequence[tuple[float, int]],
    ) -> float:
        del queue_rows, queue_oldest_ms, future, past_arrivals, queue
        return now_ms + self.window_ms


@dataclass(frozen=True)
class AdaptiveIdle:
    """Production-style adaptive idle + hard cap.

    Once the queue is non-empty, an idle countdown of ``idle_ms`` restarts on
    every new arrival after gather start; dispatch when idle expires or total
    wait since gather start reaches ``cap_ms``. Matches
    ``SlotBroker._gather_more_within_window`` with adaptive_idle_ms / batch_wait_ms.
    """

    idle_ms: float
    cap_ms: float

    @property
    def name(self) -> str:
        return f"AdaptiveIdle(idle={self.idle_ms:g},cap={self.cap_ms:g})"

    def choose_dispatch_ms(
        self,
        *,
        now_ms: float,
        queue_rows: int,
        queue_oldest_ms: float,
        future: Sequence[tuple[float, int]],
        past_arrivals: Sequence[tuple[float, int]],
        queue: Sequence[tuple[float, int]],
    ) -> float:
        del queue_rows, queue_oldest_ms, past_arrivals, queue
        idle_deadline = now_ms + self.idle_ms
        cap_deadline = now_ms + self.cap_ms
        for t_arr, _rows in future:
            if t_arr > cap_deadline or t_arr > idle_deadline:
                break
            # New arrival within the idle window restarts the idle countdown.
            idle_deadline = t_arr + self.idle_ms
        return min(idle_deadline, cap_deadline)


@dataclass
class Economic:
    """Dispatch when waiting no longer earns back >1ms of amortized value per ms.

    Exact rule (auditable):
      λ = (rows arrived in trailing 1000ms ending at the evaluation time) / 1000
          — rows/ms, not count/ms.
      Q = current queue rows (including arrivals already joined by eval time).
      B = pad_bucket(Q), B' = next ladder bucket (or B if at top).
      marginal_ms_per_row = (fwd_ms(B') - fwd_ms(B)) / max(B' - B, 1)
          (0 when B' == B — cannot grow the padded bucket further).
      per_row_now = fwd_ms(B) / max(Q, 1)
      expected_gain_per_ms = λ * max(0.0, per_row_now - marginal_ms_per_row)
      Stop waiting (dispatch) when expected_gain_per_ms < 1.0.

    Fallbacks:
      - Q == 0: should not be called (simulator waits for first arrival,
        matching FixedWindow empty-queue behavior).
      - λ == 0: gain is 0 → dispatch immediately (no benefit to waiting).
      - Safety: never wait more than 100ms from gather start.
    """

    trail_ms: float = 1000.0
    max_wait_ms: float = 100.0

    @property
    def name(self) -> str:
        return "Economic"

    def _lambda(
        self,
        eval_ms: float,
        past_arrivals: Sequence[tuple[float, int]],
        joined: Sequence[tuple[float, int]],
    ) -> float:
        lo = eval_ms - self.trail_ms
        total_rows = 0
        for t, r in past_arrivals:
            if lo < t <= eval_ms:
                total_rows += r
        for t, r in joined:
            if lo < t <= eval_ms:
                total_rows += r
        return float(total_rows) / self.trail_ms

    def _gain(self, queue_rows: int, lam: float) -> float:
        if queue_rows <= 0 or lam <= 0.0:
            return 0.0
        b = pad_bucket(queue_rows)
        b_next = next_bucket(b)
        if b_next <= b:
            marginal = 0.0
        else:
            marginal = (fwd_ms(b_next) - fwd_ms(b)) / float(b_next - b)
        per_row_now = fwd_ms(b) / float(queue_rows)
        return lam * max(0.0, per_row_now - marginal)

    def choose_dispatch_ms(
        self,
        *,
        now_ms: float,
        queue_rows: int,
        queue_oldest_ms: float,
        future: Sequence[tuple[float, int]],
        past_arrivals: Sequence[tuple[float, int]],
        queue: Sequence[tuple[float, int]],
    ) -> float:
        del queue_oldest_ms, queue
        t = now_ms
        q = queue_rows
        joined: list[tuple[float, int]] = []
        fi = 0
        hard = now_ms + self.max_wait_ms
        while True:
            lam = self._lambda(t, past_arrivals, joined)
            if self._gain(q, lam) < 1.0:
                return t
            while fi < len(future) and future[fi][0] <= t:
                fi += 1
            if fi >= len(future):
                return t
            t_next = float(future[fi][0])
            if t_next > hard:
                return t
            t = t_next
            while fi < len(future) and future[fi][0] <= t:
                joined.append(future[fi])
                q += int(future[fi][1])
                fi += 1


@dataclass(frozen=True)
class OracleClairvoyant:
    """Upper-bound reference: look ahead ``horizon_ms`` and pick best dispatch time.

    Evaluates candidate dispatch instants (now + every arrival within the
    horizon). For each candidate, forms a batch of queue + arrivals with
    ``t_arrival <= candidate`` (oldest first, capped at one forward of 4096
    rows) and scores mean per-row latency for the included requests. Picks the
    candidate with the lowest score.

    This is allowed to look ahead; it is not a real-time policy.
    """

    horizon_ms: float = 50.0

    @property
    def name(self) -> str:
        return f"OracleClairvoyant(h={self.horizon_ms:g}ms)"

    def choose_dispatch_ms(
        self,
        *,
        now_ms: float,
        queue_rows: int,
        queue_oldest_ms: float,
        future: Sequence[tuple[float, int]],
        past_arrivals: Sequence[tuple[float, int]],
        queue: Sequence[tuple[float, int]],
    ) -> float:
        del queue_rows, queue_oldest_ms, past_arrivals
        horizon_end = now_ms + self.horizon_ms
        candidates = [now_ms]
        for t_arr, _r in future:
            if t_arr > horizon_end:
                break
            if t_arr > now_ms:
                candidates.append(float(t_arr))

        best_t = now_ms
        best_cost = float("inf")
        for t_d in candidates:
            included: list[tuple[float, int]] = []
            rows_sum = 0
            for t_a, r in queue:
                if rows_sum > 0 and rows_sum + r > _MAX_BUCKET:
                    break
                included.append((t_a, r))
                rows_sum += r
                if rows_sum >= _MAX_BUCKET:
                    break
            if rows_sum < _MAX_BUCKET:
                for t_a, r in future:
                    if t_a > t_d:
                        break
                    if rows_sum > 0 and rows_sum + r > _MAX_BUCKET:
                        break
                    included.append((t_a, r))
                    rows_sum += r
                    if rows_sum >= _MAX_BUCKET:
                        break
            if rows_sum <= 0:
                continue
            t_done = t_d + fwd_ms(rows_sum)
            cost = sum((t_done - t_a) * r for t_a, r in included) / float(rows_sum)
            if cost < best_cost:
                best_cost = cost
                best_t = t_d
        return best_t


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class SimMetrics:
    label: str
    mean_batch_rows: float
    forwards_per_s: float
    rows_per_s: float
    mean_latency_ms: float
    p95_latency_ms: float
    gpu_busy_frac: float


def _to_ms_stream(arrivals: Sequence[tuple[float, int]]) -> list[tuple[float, int]]:
    """Convert absolute perf_counter seconds to ms relative to first arrival."""
    if not arrivals:
        return []
    t0 = arrivals[0][0]
    return [((t - t0) * 1000.0, rows) for t, rows in arrivals]


def run_simulation(
    arrivals_sec: Sequence[tuple[float, int]],
    policy: GatherPolicy,
    *,
    label: str | None = None,
) -> SimMetrics:
    """Serial single-GPU simulation over a sorted arrival stream."""
    stream = _to_ms_stream(arrivals_sec)
    n = len(stream)
    policy_label = label if label is not None else type(policy).__name__
    if n == 0:
        return SimMetrics(
            label=policy_label,
            mean_batch_rows=0.0,
            forwards_per_s=0.0,
            rows_per_s=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            gpu_busy_frac=0.0,
        )

    queue: deque[tuple[float, int]] = deque()
    idx = 0
    now = 0.0
    gpu_free_at = 0.0
    past: list[tuple[float, int]] = []
    latencies: list[float] = []
    batch_row_counts: list[int] = []
    total_fwd_ms = 0.0
    total_rows = 0
    n_forwards = 0

    def enqueue_until(t_end: float) -> None:
        nonlocal idx
        while idx < n and stream[idx][0] <= t_end:
            queue.append(stream[idx])
            past.append(stream[idx])
            idx += 1

    while idx < n or queue:
        if now < gpu_free_at:
            enqueue_until(gpu_free_at)
            now = gpu_free_at

        if not queue:
            if idx >= n:
                break
            now = max(now, stream[idx][0])
            enqueue_until(now)

        queue_list = list(queue)
        queue_rows = sum(r for _t, r in queue_list)
        queue_oldest = queue_list[0][0]
        future = stream[idx:]
        t_dispatch = policy.choose_dispatch_ms(
            now_ms=now,
            queue_rows=queue_rows,
            queue_oldest_ms=queue_oldest,
            future=future,
            past_arrivals=past,
            queue=queue_list,
        )
        if t_dispatch < now:
            t_dispatch = now

        enqueue_until(t_dispatch)
        now = t_dispatch

        if not queue:
            continue

        # Dispatch residual present at decision time as consecutive forwards
        # (cap 4096 rows each). Arrivals after t_dispatch that join mid-split
        # stay for the next policy decision after the GPU frees.
        residual = [(t_a, r) for t_a, r in queue if t_a <= t_dispatch]
        later = [(t_a, r) for t_a, r in queue if t_a > t_dispatch]
        queue.clear()
        queue.extend(later)

        res_i = 0
        while res_i < len(residual):
            batch: list[tuple[float, int]] = []
            rows_sum = 0
            while res_i < len(residual):
                t_a, r = residual[res_i]
                if rows_sum > 0 and rows_sum + r > _MAX_BUCKET:
                    break
                batch.append((t_a, r))
                rows_sum += r
                res_i += 1
                if rows_sum >= _MAX_BUCKET:
                    break
            if not batch:
                batch.append(residual[res_i])
                rows_sum = batch[0][1]
                res_i += 1

            dur = fwd_ms(rows_sum)
            t_done = now + dur
            for t_a, _r in batch:
                latencies.append(t_done - t_a)
            batch_row_counts.append(rows_sum)
            total_fwd_ms += dur
            total_rows += rows_sum
            n_forwards += 1
            # Mid-split: arrivals during this forward enqueue for later.
            enqueue_until(t_done)
            gpu_free_at = t_done
            now = t_done

    wall_ms = max(gpu_free_at, stream[-1][0], now)
    if wall_ms <= 0.0:
        wall_ms = 1.0
    wall_s = wall_ms / 1000.0
    lat = np.asarray(latencies, dtype=np.float64) if latencies else np.zeros(0)
    return SimMetrics(
        label=policy_label,
        mean_batch_rows=(
            float(sum(batch_row_counts) / len(batch_row_counts)) if batch_row_counts else 0.0
        ),
        forwards_per_s=float(n_forwards) / wall_s,
        rows_per_s=float(total_rows) / wall_s,
        mean_latency_ms=float(lat.mean()) if lat.size else 0.0,
        p95_latency_ms=float(np.percentile(lat, 95)) if lat.size else 0.0,
        gpu_busy_frac=float(total_fwd_ms / wall_ms),
    )


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def _print_table(rows: Sequence[SimMetrics]) -> None:
    headers = (
        "policy",
        "mean_batch",
        "fwd/s",
        "rows/s",
        "mean_lat_ms",
        "p95_lat_ms",
        "gpu_busy",
    )
    body = [
        (
            m.label,
            f"{m.mean_batch_rows:.1f}",
            f"{m.forwards_per_s:.1f}",
            f"{m.rows_per_s:.0f}",
            f"{m.mean_latency_ms:.2f}",
            f"{m.p95_latency_ms:.2f}",
            f"{m.gpu_busy_frac:.3f}",
        )
        for m in rows
    ]
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in body:
        print(fmt.format(*row))


def _build_policies(families: set[str]) -> list[tuple[str, GatherPolicy]]:
    """Return (table_label, policy) pairs for the requested families."""
    out: list[tuple[str, GatherPolicy]] = []
    # PRODUCTION: configs/pbt2_small.yaml distributed_inference_adaptive_idle_ms=4.0
    # (2026-07-16 coalescing arm; cap batch_wait_ms=20). CLI --adaptive-idle-ms
    # default is 0.0 (fixed window); live prod config sets 4.0.
    if "adaptive_idle" in families:
        for idle in (1.0, 2.0, 4.0, 8.0):
            pol = AdaptiveIdle(idle_ms=idle, cap_ms=20.0)
            label = pol.name + (" [PRODUCTION]" if idle == 4.0 else "")
            out.append((label, pol))
    if "fixed_window" in families:
        for w in (2.0, 5.0, 10.0):
            pol_fw = FixedWindow(window_ms=w)
            out.append((pol_fw.name, pol_fw))
    if "economic" in families:
        pol_ec = Economic()
        out.append((pol_ec.name, pol_ec))
    if "oracle" in families:
        pol_or = OracleClairvoyant(horizon_ms=50.0)
        out.append((f"{pol_or.name} [UPPER BOUND]", pol_or))
    return out


def _parse_policies(raw: str) -> set[str]:
    allowed = {"fixed_window", "adaptive_idle", "economic", "oracle"}
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if not parts:
        return set(allowed)
    bad = parts - allowed
    if bad:
        raise SystemExit(
            f"unknown policy family {sorted(bad)}; choose from {sorted(allowed)}"
        )
    return parts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Offline gather-policy simulator over CAE_ARRIVAL_TRACE JSONL "
            "(or a synthetic Poisson stream with --selftest)."
        ),
    )
    ap.add_argument(
        "trace",
        nargs="?",
        default=None,
        help="Path to arrival-trace JSONL (required unless --selftest).",
    )
    ap.add_argument(
        "--policies",
        default="fixed_window,adaptive_idle,economic,oracle",
        help=(
            "Comma-separated policy families to run "
            "(fixed_window,adaptive_idle,economic,oracle). Default: all."
        ),
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Run on a seeded synthetic Poisson stream (no file needed). "
            "If both a trace path and --selftest are given, --selftest wins."
        ),
    )
    args = ap.parse_args(argv)

    families = _parse_policies(args.policies)

    if args.selftest:
        arrivals = synthetic_poisson_stream()
    elif args.trace:
        arrivals = load_trace(args.trace)
        if not arrivals:
            print("error: no arrivals loaded from trace", file=sys.stderr)
            return 1
    else:
        ap.error("trace path is required unless --selftest is passed")

    policy_rows = _build_policies(families)
    metrics: list[SimMetrics] = []
    for row_label, pol in policy_rows:
        metrics.append(run_simulation(arrivals, pol, label=row_label))

    _print_table(metrics)

    if args.selftest:
        oracle_ms = [m for m in metrics if "OracleClairvoyant" in m.label]
        economic_ms = [m for m in metrics if m.label == "Economic"]
        adaptive_ms = [m for m in metrics if m.label.startswith("AdaptiveIdle")]
        if oracle_ms and economic_ms:
            assert oracle_ms[0].rows_per_s >= economic_ms[0].rows_per_s - 1e-6, (
                f"oracle rows/s {oracle_ms[0].rows_per_s} < "
                f"economic {economic_ms[0].rows_per_s}"
            )
        if oracle_ms and adaptive_ms:
            best_ad = max(m.rows_per_s for m in adaptive_ms)
            assert oracle_ms[0].rows_per_s >= best_ad - 1e-6, (
                f"oracle rows/s {oracle_ms[0].rows_per_s} < "
                f"best AdaptiveIdle {best_ad}"
            )
        print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
