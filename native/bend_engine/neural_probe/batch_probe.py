"""Opt-in cross-search batching qualification, not a production inference server."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty
import shutil
import tempfile
import time

import chess
import numpy as np
import torch

from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.search_draws import automatic_draw, draw_reply, reconstruct_leaf
from .adapter import HistoryEncoder, board_position, decode_key
from .backend import BATCHES, NativeEvaluator, build_worker
from .batching import Batch, Batcher, Completion, Key
from .package import export_smoke
from .run_probe import check_encoding, worker_failures

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Observation:
    summary: dict[str, int]
    structure: list[tuple[int, ...]]


class Actor:
    def __init__(self, peer: sessions.Peer, root: chess.Board, encoder: HistoryEncoder,
                 oracle: sessions.Oracle, session: int, budget: int):
        self.peer, self.root, self.encoder, self.oracle = peer, root, encoder, oracle
        self.session, self.budget = session, budget
        self.epoch = 0
        self.waiting: Key | None = None
        self.actions: list[int] = []
        self.done = False
        self.results: list[Observation] = []
        self.draw_leaves: list[dict[str, object]] = []
        self.start()

    def start(self) -> None:
        self.epoch += 1
        self.ref = sessions.Reference(board_position(self.root), self.oracle,
                                      cap=4096, depth=4, budget=self.budget)
        self.peer.write(f'config {self.epoch:x} {self.budget:x} 1000 4\n')

    def receive(self, line: str, broker: Batcher) -> None:
        wanted = self.ref.next()
        if wanted is None:
            result = sessions.numbers(line, 'result', 6)
            rows = [sessions.numbers(self.peer.line(), 'node', 30) for _ in self.ref.nodes]
            best = sessions.numbers(self.peer.line(), 'best', 1)[0]
            self.ref.check_snapshot(rows, result, best, self.epoch)
            self.peer.expect('ready')
            if best != sessions.SENTINEL:
                decode_key(self.root, best)
            self.results.append(Observation(
                {'session': self.session, 'epoch': self.epoch, 'completed': self.ref.completed,
                 'nodes': len(rows), 'stop': self.ref.stop, 'best': best},
                [tuple(r[:6] + r[9:]) for r in rows]))
            self.after_search(broker)
            return
        header = sessions.numbers(line, 'eval', 4)
        if header[:3] != [self.epoch, self.ref.seq, wanted] or not 1 <= header[3] <= 256:
            raise AssertionError('batched search request identity/count mismatch')
        supplied = sessions.position(sessions.numbers(self.peer.line(), 'board', 19))
        path = sessions.parse_path(self.peer.line())
        expected_path = []
        ancestor = wanted
        while ancestor:
            expected_path.append(self.ref.nodes[ancestor].key)
            ancestor = self.ref.nodes[ancestor].parent
        if path != list(reversed(expected_path)) or supplied != self.ref.nodes[wanted].board:
            raise AssertionError('batched search path/board mismatch')
        actions = [sessions.numbers(self.peer.line(), 'action', 1)[0] for _ in range(header[3])]
        self.peer.expect('end_eval')
        board = reconstruct_leaf(self.root, path, supplied, actions)
        reason = automatic_draw(board)
        if reason is not None:
            key = Key(self.session, self.epoch, header[1], wanted)
            broker.record_local(key)
            self.ref.accept_draw(wanted)
            self.peer.write(draw_reply(self.epoch, header[1], wanted))
            self.draw_leaves.append({'epoch': self.epoch, 'request': header[1],
                                     'node': wanted, 'path': path, 'reason': reason})
            return
        x, full, board = self.encoder.encode(path, supplied, actions)
        check_encoding(x, board, self.encoder.encoding)
        key = Key(self.session, self.epoch, header[1], wanted)
        now = time.monotonic()
        broker.submit(key, x, full, now=now, deadline=now + 30)
        self.waiting, self.actions = key, actions

    def after_search(self, broker: Batcher) -> None:
        """Default qualification behavior; game controllers may advance at ready."""
        if self.ref.stop:
            # Exercise a fresh epoch before late cancelled rows are scattered.
            self.start()
            broker.register(self.session, self.epoch)
        else:
            self.done = True

    def deliver(self, reply: Completion) -> None:
        if reply.key != self.waiting:
            raise AssertionError('wrong-session/epoch or duplicate completion reached an actor')
        if reply.status == 'ok':
            wdl, policy = list(reply.wdl), list(reply.policy)
            self.ref.accept(reply.key.node, self.actions, wdl, policy)
            status = 0
        else:
            status = 2 if reply.status == 'cancelled' else 1
            self.ref.stop = 2 if status == 2 else 3
            wdl, policy = [0.0, 1.0, 0.0], [0.0] * len(self.actions)
        fields = [reply.key.epoch, reply.key.request, reply.key.node, status,
                  *(sessions.bits(v) for v in wdl), len(policy), *(sessions.bits(v) for v in policy)]
        self.peer.write('reply ' + ' '.join(f'{v:x}' for v in fields) + '\n')
        self.waiting = None


def roots() -> list[chess.Board]:
    history = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2:
        history.push_uci(uci)
    return [chess.Board(), chess.Board(rules.CANONICAL[1][1]),
            chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
            chess.Board('4k3/8/8/8/8/8/p7/4K3 b - - 0 1'), history]


def group_limits(batch: int, max_rows: int | None) -> tuple[int, int]:
    """Match the package bucket without losing the bounded queue contract."""
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported group batch')
    limit = batch if max_rows is None else max_rows
    if type(limit) is not int or not 1 <= limit <= batch:
        raise ValueError('invalid group row limit')
    return limit, max(8, batch)


def group(binary: Path, oracle: sessions.Oracle, evaluator: NativeEvaluator,
          eager: torch.nn.Module, *, max_rows: int | None = None, faults: bool = False,
          atol: float = 2e-6, rtol: float = 2e-5
          ) -> tuple[dict[str, object], list[Observation]]:
    max_rows, capacity = group_limits(evaluator.batch, max_rows)
    broker = Batcher(evaluator.batch, evaluator.encoding.channels, capacity=capacity, max_wait=0.002)
    actors: list[Actor] = []
    peer_pool: list[sessions.Peer] = []
    future: Future[tuple[np.ndarray, np.ndarray]] | None = None
    flight: Batch | None = None
    histogram: dict[int, int] = {}
    cancellations: set[int] = set()
    max_error, rows, peak_reserved, peak_queued = 0.0, 0, 0, 0
    calls_before = evaluator.sequence - 1
    deadline = time.monotonic() + 120
    try:
        for i, (root, budget) in enumerate(zip(roots(), (8, 12, 4, 6, 10), strict=True)):
            encoder = HistoryEncoder(root, evaluator.encoding)
            peer = sessions.Peer(binary, board_position(root))
            peer_pool.append(peer)
            actors.append(Actor(peer, root, encoder, oracle, i, budget))
            broker.register(i, 1)
        # One worker thread owns the native stream. Main thread owns the broker.
        with ThreadPoolExecutor(max_workers=1) as executor:
            while not all(a.done for a in actors) or future is not None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('batched search group deadline')
                for actor in actors:
                    if actor.done or actor.waiting is not None:
                        continue
                    try:
                        line = actor.peer.queue.get_nowait()
                    except Empty:
                        continue
                    if line is None:
                        raise RuntimeError('batched search peer closed unexpectedly')
                    actor.receive(line, broker)
                    if (faults and actor.session == 0 and actor.waiting is not None
                            and actor.waiting.request == 3 and actor.session not in cancellations):
                        actor.deliver(broker.cancel(actor.waiting))
                        cancellations.add(actor.session)
                for reply in broker.expire(time.monotonic()):
                    actors[reply.key.session].deliver(reply)
                peak_reserved = max(peak_reserved, broker.reserved)
                peak_queued = max(peak_queued, len(broker.queue))
                if future is not None and future.done():
                    assert flight is not None
                    try:
                        policy, wdl = future.result()
                    except Exception:
                        for reply in broker.fail(flight):
                            actors[reply.key.session].deliver(reply)
                        raise
                    # Check every real row against independent eager SINGLETON inference,
                    # including cancelled rows. This catches wrong lanes/padding effects.
                    for i, job in enumerate(flight.jobs):
                        with torch.no_grad():
                            expected = eager(torch.from_numpy(job.x.copy()))
                        for actual, name in ((policy[i:i+1], 'policy'), (wdl[i:i+1], 'wdl')):
                            want = expected[name].detach().numpy()
                            np.testing.assert_allclose(actual, want, atol=atol, rtol=rtol)
                            max_error = max(max_error, float(np.abs(actual - want).max()))
                    for reply in broker.complete(flight, policy, wdl, now=time.monotonic()):
                        actors[reply.key.session].deliver(reply)
                    future, flight = None, None
                if future is None:
                    flight = broker.dispatch(time.monotonic(), max_rows=max_rows)
                    if flight is not None:
                        n = len(flight.jobs)
                        histogram[n] = histogram.get(n, 0) + 1
                        rows += n
                        future = executor.submit(evaluator.evaluate, flight.x)
                        # Cancellation after submission but before scatter. It does not
                        # claim to interrupt a running native forward or CUDA kernel.
                        for job in flight.jobs:
                            if (faults and job.key.session == 1 and job.key.request == 3
                                    and job.key.session not in cancellations):
                                actors[1].deliver(broker.cancel(job.key))
                                cancellations.add(1)
                time.sleep(0.0005)
        if faults and cancellations != {0, 1}:
            raise AssertionError('both queued and in-flight cancellation must execute')
        if broker.reserved or broker.pending:
            raise AssertionError('batched run leaked pending work')
        for actor in actors:
            actor.peer.finish()
        completed = [a.results[-1] for a in actors]
        report: dict[str, object] = {
            'max_rows': max_rows, 'faults': faults,
            'native_batch_calls': evaluator.sequence - 1 - calls_before,
            'real_rows': rows, 'fill_histogram': histogram,
            'peak_reserved_rows': peak_reserved, 'peak_queued_rows': peak_queued,
            'cancelled_sessions': sorted(cancellations),
            'max_logit_absolute_error': max_error,
            'epochs': [o.summary for a in actors for o in a.results],
        }
        return report, completed
    finally:
        for peer in peer_pool:
            peer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--cxx', default=shutil.which('clang++'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc or not args.cxx:
        parser.error('Bun, Clang and Clang++ required')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='bend-batched-') as temp:
        work = Path(temp)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(work / 'inductor')
        package = work / 'batch4.pt2'
        eager = export_smoke(package, batch=4)
        worker = build_worker(work / 'worker', args.cxx)
        binaries = sessions.build(args.compiler_root, work / 'bend', args.bun, args.cc, args.modes)
        evaluator = NativeEvaluator(worker, package)
        results = []
        try:
            negatives = worker_failures(worker, package, evaluator.encoding.channels, batch=4)
            oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
            for mode in args.modes:
                baseline_report, baseline = group(binaries[mode], oracle, evaluator, eager, max_rows=1)
                for inject in (False, True):
                    observed, final = group(binaries[mode], oracle, evaluator, eager, faults=inject)
                    for original, candidate in zip(baseline, final, strict=True):
                        if original.structure != candidate.structure:
                            raise AssertionError('batching/reset changed the single-row search structure or visits')
                        for name in ('best', 'nodes', 'completed', 'stop'):
                            if original.summary[name] != candidate.summary[name]:
                                raise AssertionError(f'batching/reset changed {name}')
                    results.append({'mode': mode, **observed})
                results.append({'mode': mode, **baseline_report})
            evaluator.finish()
            report = {'scope': 'cross-search CPU batching correctness, NOT throughput or trained weights',
                      'package': evaluator.manifest,
                      'compiler_revision': sessions.check_compiler(args.compiler_root)['revision'],
                      'results': results, 'native_calls': evaluator.sequence - 1,
                      'native_invalid_inputs_rejected': negatives}
        finally:
            evaluator.close()
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
