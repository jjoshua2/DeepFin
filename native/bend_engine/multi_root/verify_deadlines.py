"""Opt-in deadline checks on the actual coordinator with a held test callback.

Uses the existing control harness. No Python search, timing-based release, or
mocked successful model qualification. Acknowledgments are not physical drain.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from native.bend_engine.multi_root.verify import environment, parse
from native.bend_engine.multi_root.verify_control import Controlled, POSITIONS


def send(c: Controlled, text: str, reply: str) -> None:
    c.send(text + '\n')
    c.until('info string cohort_control ' + reply)


def expire(c: Controlled, root: int, milliseconds: int = 0) -> None:
    send(c, f'deadline {root} {milliseconds}', f'deadline {root} {milliseconds}')
    c.until(f'info string cohort_control expired {root}')


def run(binary: Path, reference: Path) -> dict[str, Any]:
    control = subprocess.run(
        [str(reference.resolve()), '--threads', '1', '--', '1', '2', '1', '1', *POSITIONS],
        env=environment(), text=True, capture_output=True, timeout=30, check=True,
    )
    assert not control.stderr
    baseline = parse(control.stdout, 6, 4, 1, 1, True)
    observed = []
    c = Controlled(binary, POSITIONS)
    try:
        c.started()
        for text in ('deadline 0 1', 'deadline 7 1', 'deadline 1 -1',
                     'deadline 1 3600001', 'deadline 1 4294967296', 'deadline x 0',
                     'deadline 1 nan', 'deadline 1 1 extra'):
            send(c, text, 'error')
        send(c, 'deadline 1 10000', 'deadline 1 10000')
        send(c, 'deadline 1 3600000', 'error deadline-extension')
        # Shortening is accepted. One dispatched and one undispatched root expire.
        expire(c, 1)
        expire(c, 6)
        send(c, 'deadline 1 0', 'error inactive-root')
        send(c, 'deadline 2 3600000', 'deadline 2 3600000')
        c.held()
        c.release()
        c.started()  # only root five is left for a second forward
        send(c, 'deadline 2 0', 'error inactive-root')  # root two completed
        c.release()
        r = parse(c.finish(), 6, 4, 1, 1, True, asynchronous=True)
        assert (r['work']['executed_real_rows'], r['work']['accepted_neural_rows'],
                r['work']['executed_wasted_rows'], r['work']['padded_rows']) == (5, 4, 1, 3)
        assert r['work']['deadline_expired_mask'] == (1 << 1) | (1 << 6)
        for ep in (1, 6):
            assert r['roots'][ep]['deadline_expired']
            assert r['roots'][ep]['accepted_neural_rows'] == 0
            assert r['roots'][ep]['cancelled_rows'] == (1 if ep == 1 else 0)
            assert r['nodes'][ep][0][23:25] == [0, 0]
        for ep in (2, 3, 4, 5):
            assert not r['roots'][ep]['deadline_expired']
            assert r['nodes'][ep] == baseline['nodes'][ep]
        observed.append({'case': 'selective_expiry', 'work': r['work'], 'roots': r['roots']})
    finally:
        c.close()

    c = Controlled(binary, POSITIONS[:4], sims=2, budget=2)
    try:
        c.started()
        c.release()
        c.started()
        expire(c, 1)
        c.release()
        r = parse(c.finish(), 4, 4, 2, 2, True, asynchronous=True)
        assert r['nodes'][1] == baseline['nodes'][1], 'expiry changed previously accepted work'
        assert r['roots'][1]['accepted_neural_rows'] == r['roots'][1]['cancelled_rows'] == 1
        assert (r['work']['executed_real_rows'], r['work']['accepted_neural_rows']) == (8, 7)
        observed.append({'case': 'accepted_work_preserved', 'work': r['work']})
    finally:
        c.close()

    # A real monotonic deadline must continue ticking after stdin is closed.
    c = Controlled(binary, POSITIONS[:4])
    try:
        c.started()
        send(c, 'deadline 1 25', 'deadline 1 25')
        assert c.proc.stdin
        c.proc.stdin.close()
        c.until('info string cohort_control expired 1')
        c.held()
        c.release()
        r = parse(c.finish(), 4, 4, 1, 1, True, asynchronous=True)
        assert (r['work']['accepted_neural_rows'], r['work']['executed_wasted_rows']) == (3, 1)
        observed.append({'case': 'expiry_after_eof', 'work': r['work']})
    finally:
        c.close()

    # Expiry must be observed between buffered commands, not postponed until
    # stdin becomes idle. Process suspension is test orchestration, not a clock hook.
    c = Controlled(binary, POSITIONS[:4])
    try:
        c.started()
        send(c, 'deadline 1 100', 'deadline 1 100')
        os.kill(c.proc.pid, signal.SIGSTOP)
        time.sleep(0.15)
        c.send('isready\n' * 200)
        os.kill(c.proc.pid, signal.SIGCONT)
        c.until('info string cohort_control expired 1')
        send(c, 'deadline 1 1', 'error inactive-root')  # barrier after the backlog
        ix = c.lines.index('info string cohort_control expired 1')
        assert c.lines[:ix].count('info string cohort_ready') < 200
        c.release()
        r = parse(c.finish(), 4, 4, 1, 1, True, asynchronous=True)
        assert r['work']['accepted_neural_rows'] == 3
        observed.append({'case': 'expiry_across_input_backlog', 'work': r['work']})
    finally:
        c.close()

    # A manual stop must not later acquire a misleading deadline-expired reason.
    c = Controlled(binary, POSITIONS[:4])
    try:
        c.started()
        send(c, 'deadline 1 3600000', 'deadline 1 3600000')
        send(c, 'cancel 1', 'cancel 1')
        send(c, 'deadline 1 0', 'error inactive-root')
        c.release()
        r = parse(c.finish(), 4, 4, 1, 1, True, asynchronous=True)
        assert r['roots'][1]['cancel_requested']
        assert not r['roots'][1]['deadline_expired']
        assert r['work']['deadline_expired_mask'] == 0
        observed.append({'case': 'manual_stop_reason_preserved', 'work': r['work']})
    finally:
        c.close()

    # Expiry must not hide corrupt output in its own cancelled row.
    for fault in ('fail', 'nan'):
        c = Controlled(binary, POSITIONS[:4], fault=fault)
        try:
            c.started()
            expire(c, 4)
            c.release()
            text = c.finish(2)
            assert 'cohort_work ' not in text
            assert 'native_reply ' not in text
        finally:
            c.close()

    # An immediate deadline before any gather consumes zero neural work.
    result = subprocess.run(
        [str(reference.resolve()), '--threads', '1', '--', '1', '2', '1', '1', POSITIONS[0]],
        env={**environment(), 'DEEPFIN_COHORT_ASYNC': '1'}, input='deadline 1 0\n',
        text=True, capture_output=True, timeout=30, check=True,
    )
    assert not result.stderr
    r = parse(result.stdout, 1, 4, 1, 1, True, asynchronous=True)
    assert r['roots'][1]['deadline_expired']
    assert r['work']['forward_calls'] == 0
    assert r['roots'][1]['neural_budget_met'] is False
    observed.append({'case': 'zero_work_pre_gather', 'work': r['work']})
    return {'status': 'passed', 'scope': 'actual Bend coordinator and worker, blocked test callback only',
            'cases': observed, 'expired_fault_controls': 2, 'speed_qualified': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.binary, args.reference)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
