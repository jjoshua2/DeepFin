"""Opt-in full Bend application lifecycle checks using an explicitly blocked callback.

No real model, performance measurement, or Python search controller. Both event
and release pipes are test-only; the production engine has no test hooks.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import select
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import chess

from scripts.bench_neural_search import parse_report

WORK = "info string neural_work "
RETIRED = "info string neural_retired "


def one_report(lines: list[str], prefix: str, epoch: int) -> dict[str, Any]:
    found = [json.loads(s[len(prefix):]) for s in lines if s.startswith(prefix)]
    assert len(found) == 1, lines
    result = found[0]
    assert isinstance(result, dict), result
    assert type(result.get('search_epoch')) is int, result
    assert result['search_epoch'] == epoch, result
    return result


class Gate:
    def __init__(self, command: list[str], asynchronous: bool = True, diagnostics: bool = False):
        self.events, events_writer = os.pipe()
        release_reader, self.release_writer = os.pipe()
        self.rows: queue.Queue[str | None] = queue.Queue()
        self.errors: list[str] = []
        self.transcript: list[str] = []
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, pass_fds=(events_writer, release_reader),
            env={**os.environ, 'DEEPFIN_BEND_ASYNC': str(int(asynchronous)),
                 'DEEPFIN_BEND_NATIVE_DIAGNOSTICS': str(int(diagnostics)),
                 'DEEPFIN_TEST_EVENT_FD': str(events_writer),
                 'DEEPFIN_TEST_RELEASE_FD': str(release_reader)},
        )
        os.close(events_writer)
        os.close(release_reader)
        assert self.proc.stdin
        assert self.proc.stdout
        assert self.proc.stderr

        def output() -> None:
            assert self.proc.stdout
            for row in self.proc.stdout:
                text = row.rstrip('\n')
                self.transcript.append(text)
                self.rows.put(text)
            self.rows.put(None)

        def errors() -> None:
            assert self.proc.stderr
            self.errors.extend(self.proc.stderr)

        self.readers = [threading.Thread(target=output, daemon=True), threading.Thread(target=errors, daemon=True)]
        for reader in self.readers:
            reader.start()

    def send(self, text: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()

    def until(self, prefix: str, timeout: float = 5) -> list[str]:
        end = time.monotonic() + timeout
        result = []
        while True:
            row = self.rows.get(timeout=max(0.001, end - time.monotonic()))
            assert row is not None, (result, self.errors)
            result.append(row)
            if row.startswith(prefix):
                return result
            if time.monotonic() >= end:
                raise TimeoutError(result)

    def init(self) -> None:
        self.send('uci\nisready')
        assert 'uciok' in self.until('readyok')

    def started(self) -> None:
        assert select.select([self.events], [], [], 5)[0], self.transcript
        assert os.read(self.events, 1) == b'S'

    def no_start(self) -> None:
        assert not select.select([self.events], [], [], 0.03)[0], self.transcript

    def release(self, code: bytes = b'R') -> None:
        assert len(code) == 1
        assert os.write(self.release_writer, code) == 1

    def ready(self) -> list[str]:
        self.send('isready')
        # This timeout is a generous functional bound, not a tournament latency claim.
        return self.until('readyok', timeout=0.75)

    def no_bestmove(self) -> None:
        assert not any(row.startswith('bestmove ') for row in self.ready())

    def finish(self, code: int = 0) -> None:
        self.proc.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
            assert not reader.is_alive()
        assert self.proc.returncode == code, (self.proc.returncode, self.errors)
        if code == 0:
            assert not self.errors, self.errors

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        os.close(self.events)
        os.close(self.release_writer)


def snapshot(lines: list[str], epoch: int, *, dispatched: int, executed: int,
             accepted: int, cancelled: int) -> dict[str, Any]:
    report = one_report(lines, WORK, epoch)
    for key, value in {'dispatched_real_rows': dispatched, 'executed_real_rows': executed,
                       'accepted_neural_rows': accepted, 'cancelled_rows': cancelled,
                       'unconfirmed_forward_rows': dispatched - executed}.items():
        assert type(report.get(key)) is int, report
        assert report[key] == value, report
    assert len([x for x in lines if x.startswith('bestmove ')]) == 1, lines
    return report


def run_cases(command: list[str]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    c = Gate(command)
    try:
        c.init()
        c.send('position startpos\ngo evals 2 depth 2')
        c.started()
        assert c.proc.stdin
        c.proc.stdin.write('isre')
        c.proc.stdin.flush()
        c.send('ady')
        c.until('readyok', 0.75) # partial input while blocked
        c.no_bestmove() # readiness is observable while callback is blocked
        c.send('stop\nisready')
        lines = c.until('readyok', 0.75)
        observations.append(snapshot(lines, 1, dispatched=1, executed=0, accepted=0, cancelled=1))
        parsed = parse_report(lines, kind='evals', budget=2)
        assert not parsed['comparable'], parsed
        c.send('stop')
        c.no_bestmove()
        c.send('position startpos moves e2e4\ngo evals 1 depth 2')
        c.no_bestmove()
        c.no_start() # cannot dispatch into old storage before physical retirement
        c.release()
        c.started()
        c.release()
        lines = c.until('bestmove ')
        retired = one_report(lines, RETIRED, 1)
        assert (retired['executed_real_rows'], retired['executed_wasted_rows'],
                retired['accepted_neural_rows'], retired['cancelled_rows']) == (1, 1, 0, 1)
        observations.append(snapshot(lines, 2, dispatched=1, executed=1, accepted=1, cancelled=0))
        board = chess.Board()
        board.push_uci('e2e4')
        assert chess.Move.from_uci(lines[-1].split()[1]) in board.legal_moves, lines[-1]
        assert parse_report(lines, kind='evals', budget=1)['comparable']
        c.send('position invalid\nisready')
        assert any('previous root preserved' in row for row in c.until('readyok'))
        c.send('ucinewgame\ngo evals 1 depth 2')
        c.started()
        c.release()
        observations.append(snapshot(c.until('bestmove '), 3, dispatched=1, executed=1, accepted=1, cancelled=0))
        c.send('position fen k7/1Q6/2K5/8/8/8/8/8 b - - 150 1\ngo nodes 4')
        lines = c.until('bestmove ')
        observations.append(snapshot(lines, 4, dispatched=0, executed=0, accepted=0, cancelled=0))
        assert lines[-1] == 'bestmove 0000', lines
        c.no_start()
        c.send('quit')
        c.finish()
    finally:
        c.close()

    # A pending request's deadline expires before the callback is released.
    c = Gate(command)
    try:
        c.init()
        c.send('go movetime 200 depth 8')
        c.started()
        # Suspend all engine threads past the deadline, queue complete commands,
        # then resume. Bestmove must precede the end of the backlog: merely
        # checking the timer when stdin becomes empty cannot pass this test.
        os.kill(c.proc.pid, signal.SIGSTOP)
        time.sleep(0.25)
        c.send('isready\n' * 400)
        os.kill(c.proc.pid, signal.SIGCONT)
        lines = c.until('bestmove ', 1.5)
        assert 0 < lines.count('readyok') < 400, lines
        observations.append(snapshot(lines, 1, dispatched=1, executed=0, accepted=0, cancelled=1))
        # Drain remaining ready responses before a new queued search.
        c.send('d')
        c.until('info string state_end')
        c.send('go movetime 30 depth 8')
        lines = c.until('bestmove ', 0.75)
        observations.append(snapshot(lines, 2, dispatched=0, executed=0, accepted=0, cancelled=0))
        c.no_start()
        c.release()
        retired = one_report(c.until(RETIRED), RETIRED, 1)
        assert retired['unconfirmed_forward_rows'] == 0
        c.send('quit')
        c.finish()
    finally:
        c.close()

    # A naturally completed infinite search holds its decision until stop.
    c = Gate(command, diagnostics=True)
    try:
        c.init()
        c.send('go nodes 1 infinite')
        c.started()
        c.release()
        c.until('info string native_reply ')
        c.no_bestmove()
        c.send('stop')
        lines = c.until('bestmove ', 0.75)
        observations.append(snapshot(lines, 1, dispatched=1, executed=1, accepted=1, cancelled=0))
        c.send('stop')
        c.no_bestmove()
        c.send('quit')
        c.finish()
    finally:
        c.close()

    # Quit cannot destroy the model or worker storage while a callback owns it.
    c = Gate(command)
    try:
        c.init()
        c.send('go evals 1 depth 2')
        c.started()
        c.send('quit')
        try:
            c.proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError('quit returned before callback completion')
        c.release()
        c.finish()
        assert not any(x.startswith('bestmove ') for x in c.transcript), c.transcript
        retired = one_report(c.transcript, RETIRED, 1)
        assert (retired['executed_real_rows'], retired['cancelled_rows']) == (1, 1)
    finally:
        c.close()

    for cancelled, outcome in ((False, b'F'), (True, b'F'), (False, b'N')):
        c = Gate(command)
        try:
            c.init()
            c.send('go evals 1 depth 2')
            c.started()
            if cancelled:
                c.send('stop')
                snapshot(c.until('bestmove ', 0.75), 1, dispatched=1, executed=0, accepted=0, cancelled=1)
            c.release(outcome)
            c.finish(2)
            assert len([x for x in c.transcript if x.startswith('bestmove ')]) == int(cancelled), c.transcript
            assert not any(x.startswith('info string native_reply ') for x in c.transcript)
            if cancelled:
                retired = one_report(c.transcript, RETIRED, 1)
                assert (retired['failed_forward_rows'], retired['executed_real_rows']) == (1, 0)
        finally:
            c.close()

    # Negative control: same generated engine in sync mode cannot satisfy the
    # blocked-callback readiness test. Release explicitly so cleanup is bounded.
    c = Gate(command, asynchronous=False)
    try:
        c.init()
        c.send('go evals 1 depth 2')
        c.started()
        c.send('isready')
        try:
            c.until('readyok', 0.1)
        except queue.Empty:
            pass
        else:
            raise AssertionError('synchronous blocked callback unexpectedly returned readiness')
        c.release()
        c.until('readyok')
        c.until('bestmove ')
        c.send('quit')
        c.finish()
    finally:
        c.close()
    return {'status': 'passed', 'scope': 'actual Bend UCI/search with deterministic blocked callback; not a model or speed test',
            'decision_reports': observations, 'failure_cases': 3, 'sync_negative_control': True,
            'stop_before_release': True, 'readiness_before_release': True, 'quit_waits_for_physical_completion': True}


def verify_accounting(text: str) -> None:
    rows = text.splitlines()
    assert len(rows) == 3, rows
    decision = one_report([rows[0]], WORK, 7)
    success = one_report([rows[1]], RETIRED, 7)
    failed = one_report([rows[2]], RETIRED, 7)
    for r in (decision, success, failed):
        assert r['forward_calls'] == r['dispatched_real_rows'] == 2
        assert r['accepted_neural_rows'] == r['completed_simulations'] == 1
        assert r['cancelled_rows'] == 1
        assert r['unresolved_rows'] == 0
        assert r['phase_seconds']['encoding'] == 0.009
        assert r['backend_and_transport_seconds'] is None
    assert (decision['executed_real_rows'], decision['executed_wasted_rows'],
            decision['failed_forward_rows'], decision['unconfirmed_forward_rows']) == (1, 0, 0, 1)
    assert (success['executed_real_rows'], success['executed_wasted_rows'],
            success['failed_forward_rows'], success['unconfirmed_forward_rows']) == (2, 1, 0, 0)
    assert (failed['executed_real_rows'], failed['executed_wasted_rows'],
            failed['failed_forward_rows'], failed['unconfirmed_forward_rows']) == (1, 0, 1, 0)
    parsed = parse_report([rows[0], 'info nodes 1 string cancelled', 'bestmove e2e4'], kind='evals', budget=2)
    assert not parsed['comparable']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--accounting-output', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    verify_accounting(args.accounting_output.read_text())
    report = run_cases(args.command)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
