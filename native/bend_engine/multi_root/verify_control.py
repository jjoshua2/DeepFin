"""Explicit blocked-forward tests of the actual Bend cohort and native worker.

Never linked into normal model execution. The pipes release callbacks only;
all cancellation, root ownership and result routing execute inside Bend.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import threading
import time
from typing import Any

from .verify import environment, parse

POSITIONS = ['startpos', 'startpos moves e2e4', 'startpos moves d2d4',
             'startpos moves g1f3', 'startpos moves c2c4', 'startpos moves b1c3']


class Controlled:
    def __init__(self, binary: Path, positions: list[str], *, asynchronous: bool = True, fault: str = '', sims: int = 1, budget: int = 1):
        self.events, event_writer = os.pipe()
        release_reader, self.release_writer = os.pipe()
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.queue: queue.Queue[str | None] = queue.Queue()
        env = {**environment(), 'DEEPFIN_COHORT_ASYNC': str(int(asynchronous)),
               'DEEPFIN_TEST_EVENT_FD': str(event_writer), 'DEEPFIN_TEST_RELEASE_FD': str(release_reader)}
        if fault:
            env['DEEPFIN_MULTI_TEST_FAULT'] = fault
        self.proc = subprocess.Popen([str(binary.resolve()), '--threads', '1', '--', str(sims), '2', str(budget), '1', *positions],
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1,
                                     pass_fds=(event_writer, release_reader))
        os.close(event_writer)
        os.close(release_reader)

        def output() -> None:
            assert self.proc.stdout
            for line in self.proc.stdout:
                self.lines.append(line.rstrip('\n'))
                self.queue.put(line.rstrip('\n'))
            self.queue.put(None)

        def errors() -> None:
            assert self.proc.stderr
            self.errors.extend(self.proc.stderr)

        self.readers = [threading.Thread(target=output, daemon=True), threading.Thread(target=errors, daemon=True)]
        for t in self.readers:
            t.start()

    def started(self) -> None:
        assert select.select([self.events], [], [], 5)[0], self.lines
        assert os.read(self.events, 1) == b'S'

    def send(self, command: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(command)
        self.proc.stdin.flush()

    def until(self, expected: str, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            item = self.queue.get(timeout=max(0.001, deadline-time.monotonic()))
            assert item is not None, (self.lines, self.errors)
            if item == expected or (expected.endswith(' error') and item.startswith(expected+' ')):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(self.lines)

    def release(self) -> None:
        assert os.write(self.release_writer, b'R') == 1

    def held(self) -> None:
        assert self.proc.poll() is None
        assert not any(('cohort_root ' in s or 'cohort_work ' in s or 'native_reply ' in s) for s in self.lines)
        assert not select.select([self.events], [], [], 0.02)[0]

    def finish(self, code: int = 0) -> str:
        self.proc.wait(timeout=5)
        for t in self.readers:
            t.join(timeout=2)
            assert not t.is_alive()
        assert self.proc.returncode == code, (self.proc.returncode, self.errors, self.lines[-8:])
        if code == 0:
            assert not self.errors, self.errors
        return '\n'.join(self.lines)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        for t in self.readers:
            t.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        os.close(self.events)
        os.close(self.release_writer)


def controls(binary: Path, reference: Path) -> dict[str, Any]:
    plain = subprocess.run([str(reference.resolve()), '--threads', '1', '--', '1','2','1','1', *POSITIONS],
                           env=environment(), text=True, capture_output=True, timeout=30, check=False)
    assert plain.returncode == 0, plain.stderr
    assert not plain.stderr, plain.stderr
    baseline = parse(plain.stdout, 6, 4, 1, 1, True)
    c = Controlled(binary, POSITIONS)
    try:
        c.started()
        c.send('isre')
        c.send('ady\n')
        c.until('info string cohort_ready')
        for command in ('cancel 0', 'cancel 7', 'cancel -1', 'cancel 4294967296', 'cancel 1 extra', 'garbage'):
            c.send(command+'\n')
            c.until('info string cohort_control error')
        for command in ('cancel 1', 'cancel 1', 'cancel 6'):
            c.send(command+'\n')
            c.until('info string cohort_control '+command)
        c.held()
        c.release()
        c.started()  # Root five still needs a row; cancelled root six never dispatches.
        c.send('isready\n')
        c.until('info string cohort_ready')
        assert not any('cohort_work ' in s for s in c.lines)
        c.release()
        result = parse(c.finish(), 6, 4, 1, 1, True, asynchronous=True)
        assert result['work']['executed_real_rows'] == 5
        assert result['work']['accepted_neural_rows'] == 4
        assert result['work']['executed_wasted_rows'] == 1
        assert result['work']['padded_rows'] == 3
        for ep in (1, 6):
            r = result['roots'][ep]
            assert r['cancel_requested']
            assert r['accepted_neural_rows'] == 0
            assert r['cancelled_rows'] == (1 if ep == 1 else 0)
            assert r['used_nodes'] == 1
            assert not r['searched_move']
            # No value/visit updates were applied to either cancelled tree.
            assert result['nodes'][ep][0][23:25] == [0, 0]
        for ep in (2, 3, 4, 5):
            r = result['roots'][ep]
            assert not r['cancel_requested']
            assert all(r[k] == baseline['roots'][ep][k] for k in baseline['roots'][ep])
            assert result['nodes'][ep] == baseline['nodes'][ep], 'cancellation changed an unaffected tree'
        selective = result['work']
    finally:
        c.close()
    # Cancel only after an earlier sweep was accepted: preserve banked tree work.
    c = Controlled(binary, POSITIONS[:4], sims=2, budget=2)
    try:
        c.started()
        c.release()
        c.started()
        c.send('cancel 1\n')
        c.until('info string cohort_control cancel 1')
        c.release()
        result = parse(c.finish(),4,4,2,2,True,asynchronous=True)
        r = result['roots'][1]
        assert r['dispatched_real_rows'] == 2
        assert r['accepted_neural_rows'] == r['cancelled_rows'] == 1
        assert not r['neural_budget_met']
        assert result['nodes'][1] == baseline['nodes'][1]
        late = result['work']
    finally:
        c.close()
    stopped = []
    for command in ('stop', 'quit'):
        c = Controlled(binary, POSITIONS)
        try:
            c.started()
            c.send(command+'\n')
            c.until('info string cohort_control '+command)
            c.held()
            c.release()
            r = parse(c.finish(), 6, 4, 1, 1, True, asynchronous=True)
            assert r['work']['forward_calls'] == 1
            assert r['work']['accepted_neural_rows'] == 0
            assert r['work']['cancelled_rows'] == r['work']['executed_real_rows'] == 4
            assert all(x['cancel_requested'] for x in r['roots'].values())
            stopped.append(r['work'])
        finally:
            c.close()
    for fault in ('fail', 'nan'):
        c = Controlled(binary, POSITIONS, fault=fault)
        try:
            c.started()
            c.send('cancel 4\n')  # The nonfinite final row belongs to a cancelled root.
            c.until('info string cohort_control cancel 4')
            c.release()
            text = c.finish(2)
            assert 'native_reply ' not in text
            assert 'cohort_work ' not in text
            assert 'cohort_root ' not in text
        finally:
            c.close()
    c = Controlled(binary, POSITIONS[:2], asynchronous=False)
    try:
        c.started()
        c.send('isready\n')
        try:
            c.until('info string cohort_ready', 0.1)
        except queue.Empty:
            pass
        else:
            raise AssertionError('synchronous blocked callback returned control readiness')
        c.release()
        parse(c.finish(), 2, 4, 1, 1, True)
    finally:
        c.close()
    # No stdin / EOF is explicitly not cancellation: compare the full no-control oracle separately.
    for flag in ('true', '2', '-1'):
        run = subprocess.run([str(reference.resolve()), '--threads','1','--','1','2','1','0','startpos'],
                             env={**environment(), 'DEEPFIN_COHORT_ASYNC': flag}, capture_output=True, timeout=10, check=False)
        assert run.returncode == 2
        assert not run.stdout
    return {'status':'passed', 'scope':'actual Bend cohort with blocked deterministic callback; no GPU or speed test',
            'selective_cancellation':selective, 'cancel_after_acceptance':late, 'stop_and_quit':stopped, 'unaffected_trees_bit_identical':True,
            'invalid_commands_rejected':6,'failed_batches_rejected':2,'invalid_flags_rejected':3,
            'synchronous_negative_control':True}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = controls(args.binary, args.reference)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
