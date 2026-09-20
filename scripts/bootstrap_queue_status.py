"""Compact read-only report for a bootstrap operator queue.

Estimates and timeout caps are separate. A queued descriptor is not a guarantee
that its inputs will pass launch checks. This tool never dispatches or polls jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
from pathlib import Path
import time
from typing import Any


def summarize(queue: dict[str, Any], state: dict[str, Any], now: float) -> dict[str, Any]:
    items = queue['items']
    runnable = [job for job in items if job['status'] == 'queued']
    estimates = [job for job in runnable if isinstance(job.get('estimated_active_seconds'), (int, float))
                 and job['estimated_active_seconds'] > 0]
    completed = [job for job in items if job['status'] in ('logged', 'complete', 'completed')]
    active = [job for job in items if job['status'] in ('running', 'launching', 'needs_recovery')]
    fields = ('id', 'kind', 'status', 'pid', 'out', 'max_seconds', 'estimated_active_seconds')
    recent = []
    for job in completed[-5:]:
        result = job.get('result') or {}
        recent.append({'id': job['id'], **{key: result[key] for key in
                       ('status', 'elo', 'elo_ci95', 'games', 'rows', 'elapsed_seconds', 'duration_s')
                       if key in result}})
    deadline = state.get('deadline_unix')
    return {
        'checked_utc': dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
        'status_counts': dict(Counter(job['status'] for job in items)),
        'active': [{key: job[key] for key in fields if key in job} for job in active],
        'queued': [{key: job[key] for key in fields if key in job} for job in runnable],
        'estimated_queued_hours_known_jobs_only': sum(job['estimated_active_seconds'] for job in estimates) / 3600,
        'queued_jobs_without_estimates': [job['id'] for job in runnable if job not in estimates],
        'queued_timeout_cap_hours': sum(float(job.get('max_seconds', 0)) for job in runnable) / 3600,
        'deadline_hours_remaining': None if deadline is None else (float(deadline) - now) / 3600,
        'recent_completed_in_queue_order': recent,
        'needs_attention': [job['id'] for job in items if job['status'] in ('needs_recovery', 'launching')],
        'scope': 'Snapshot of scheduler records; not process/GPU health or input validation. Recent means queue order, not completion time.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', type=Path, required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    # Existing lock only: read-only command must not create a new scheduler root.
    with (args.loop / 'gpu.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        queue = json.loads((args.loop / 'queue.json').read_text())
        state = json.loads((args.loop / 'STATE.json').read_text())
    report = summarize(queue, state, time.time())
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(report['checked_utc'])
    print('Status:', report['status_counts'])
    for job in report['active']:
        print(f"{job['status'].upper()}: {job['id']} pid={job.get('pid', 'unknown')}")
    print(f"Queued: {len(report['queued'])} jobs; known estimates {report['estimated_queued_hours_known_jobs_only']:.1f}h")
    print(f"Missing estimates: {len(report['queued_jobs_without_estimates'])}; timeout caps {report['queued_timeout_cap_hours']:.1f}h (not ETA)")
    for job in report['queued']:
        seconds = job.get('estimated_active_seconds')
        estimate = f'{seconds / 3600:.2f}h estimated' if seconds else 'duration unestimated'
        print(f"  {job['id']}: {estimate}")
    if report['needs_attention']:
        print('Inspect recorded states:', ', '.join(report['needs_attention']))


if __name__ == '__main__':
    main()
