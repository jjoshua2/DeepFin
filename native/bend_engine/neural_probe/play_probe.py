"""Opt-in sustained neural play through the already-exported checkpoint boundary.

Uses an exact checkpoint/package pair from checkpoint_probe --work-dir, avoiding
another model export. Native logits, all tree snapshots and played roots remain
checked: this is a correctness controller, not a throughput benchmark or UCI.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import io
import json
from pathlib import Path
from queue import Empty
import shutil
import time

import chess
import chess.pgn
import numpy as np
import torch

from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe import run_probe as sessions
from .adapter import board_position
from .backend import BATCHES, NativeEvaluator, build_worker
from .batch_probe import group_limits
from .batching import Batch, Batcher, Key
from .checkpoint import load_checkpoint
from .checkpoint_probe import tolerances, validate_report_destination
from .game import CLAIM_POLICIES, GameActor, GameSpec
from .qualification import (
    compilation_cache, device_snapshot, protect_inputs, reuse_reference,
    tools, verified_package, workspace, write_report,
)

ROOT = Path(__file__).resolve().parents[3]


def fixtures() -> list[GameSpec]:
    cycle = ('g1f3', 'g8f6', 'f3g1', 'f6g8')
    history = chess.Board()
    for uci in cycle * 2:
        history.push_uci(uci)
    return [
        GameSpec('start', chess.Board(), 6, scripted=('e2e4',)),
        GameSpec('kiwipete', chess.Board(rules.CANONICAL[1][1]), 4, scripted=('e1g1',)),
        GameSpec('played-history', history, 4),
        GameSpec('black-promotion', chess.Board('4k3/8/8/8/8/8/p7/4K3 b - - 0 1'), 1),
        GameSpec('scripted-mate', chess.Board(), 4, scripted=('f2f3', 'e7e5', 'g2g4', 'd8h4')),
        GameSpec('fivefold', chess.Board(), 16, scripted=cycle * 4),
        GameSpec('seventyfive', chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 149 1'), 1, scripted=('a1b1',)),
        GameSpec('claim-at-root', history, 1, claims='claim_available'),
    ]


def check_fixture_results(actors: list[GameActor]) -> None:
    expected = {
        'scripted-mate': ('checkmate', '0-1', 4),
        'fivefold': ('fivefold_repetition', '1/2-1/2', 16),
        'seventyfive': ('seventyfive_moves', '1/2-1/2', 1),
        'claim-at-root': ('threefold_repetition', '1/2-1/2', 0),
    }
    for actor in actors:
        name = actor.spec.name
        if name in expected and (actor.end is None or (actor.end.reason, actor.end.result, len(actor.moves)) != expected[name]):
            raise AssertionError('unexpected rule result: ' + name)
        if len(actor.results) != len(actor.moves):
            raise AssertionError('a terminal played root started an extra neural search')
        if actor.rebound_encoders != len(actor.moves):
            raise AssertionError('not every played move rebuilt the encoder')


def same_games(reference: list[GameActor], candidate: list[GameActor], *, faults: bool) -> None:
    for old, new in zip(reference, candidate, strict=True):
        if old.end != new.end or old.root.fen(en_passant='fen') != new.root.fen(en_passant='fen'):
            raise AssertionError('batching/cancellation changed game outcome or final board')
        if old.root.move_stack != new.root.move_stack or len(old.results) != len(new.results):
            raise AssertionError('batching/cancellation changed played history or search count')
        for i, (a, b) in enumerate(zip(old.results, new.results, strict=True)):
            # The first scripted move after deliberate cancellation ignores the
            # partial tree. All subsequent searches must match the completed control.
            if faults and new.session in (0, 1) and i == 0:
                if b.summary['completed'] != 2 or b.summary['stop'] != 2:
                    raise AssertionError('cancelled pre-advance search was partially committed')
                continue
            if a.summary != b.summary or a.structure != b.structure:
                raise AssertionError('new-root search differs from the same-package control')


def replay_check(actor: GameActor) -> None:
    report = actor.report()
    game = chess.pgn.read_game(io.StringIO(str(report['pgn'])))
    if game is None or game.errors:
        raise AssertionError('game PGN did not parse')
    final = game.end().board()
    if final.fen(en_passant='fen') != actor.root.fen(en_passant='fen') or final.move_stack != actor.root.move_stack:
        raise AssertionError('PGN lost played or pre-root history')
    if actor.end is None or game.headers['Result'] != actor.end.result:
        raise AssertionError('PGN did not preserve explicit termination result')


def play_games(binary: Path, oracle: sessions.Oracle, evaluator: NativeEvaluator,
               eager: torch.nn.Module, specs: list[GameSpec], *, budget: int = 4,
               max_rows: int | None = None, faults: bool = False,
               atol: float = 2e-6, rtol: float = 2e-5) -> tuple[dict[str, object], list[GameActor]]:
    if not 1 <= len(specs) <= 8 or len({s.name for s in specs}) != len(specs):
        raise ValueError('one to eight distinctly named games are required')
    if faults and (len(specs) < 2 or any(not s.scripted or s.max_plies < 2 for s in specs[:2]) or budget < 3):
        raise ValueError('fault contrast requires scripted first moves and a later search')
    max_rows, capacity = group_limits(evaluator.batch, max_rows)
    broker = Batcher(evaluator.batch, evaluator.encoding.channels, capacity=max(capacity, len(specs) + int(faults)), max_wait=0.002)
    actors: list[GameActor] = []
    peers: list[sessions.Peer] = []
    future: Future[tuple[np.ndarray, np.ndarray]] | None = None
    flight: Batch | None = None
    held: Key | None = None
    cancelled: set[int] = set()
    histogram: dict[int, int] = {}
    rows = late_discarded = 0
    max_error = 0.0
    peak_reserved = 0
    before = evaluator.sequence - 1
    deadline = time.monotonic() + 180
    try:
        for i, spec in enumerate(specs):
            peer = sessions.Peer(binary, board_position(spec.root))
            peers.append(peer)
            actor = GameActor(peer, spec, evaluator.encoding, oracle, i, budget)
            actors.append(actor)
            if not actor.done:
                broker.register(i, actor.epoch)
        # Exactly one worker owns the native stream. This thread alone changes
        # roots, epochs, encoders and queue state. Cancellation cannot kill kernels.
        with ThreadPoolExecutor(max_workers=1) as executor:
            while not all(a.done for a in actors) or future is not None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('neural game group deadline')
                for actor in actors:
                    if actor.done or actor.waiting is not None:
                        continue
                    try:
                        line = actor.peer.queue.get_nowait()
                    except Empty:
                        continue
                    if line is None:
                        raise RuntimeError('neural game peer closed unexpectedly')
                    actor.receive(line, broker)
                    if (faults and actor.session == 0 and actor.waiting is not None
                            and actor.waiting.epoch == 1 and actor.waiting.request == 3 and 0 not in cancelled):
                        actor.deliver(broker.cancel(actor.waiting))
                        cancelled.add(0)
                for reply in broker.expire(time.monotonic()):
                    actors[reply.key.session].deliver(reply)
                peak_reserved = max(peak_reserved, broker.reserved)
                # Deliberately hold scatter of one cancelled old-root row until
                # the new root/encoder has queued its first evaluation. This
                # makes the late-result test independent of CPU arrival timing.
                can_scatter = held is None or (
                    actors[held.session].waiting is not None
                    and actors[held.session].epoch > held.epoch
                    and bool(actors[held.session].moves))
                if future is not None and future.done() and can_scatter:
                    assert flight is not None
                    try:
                        policy, wdl = future.result()
                    except Exception:
                        for reply in broker.fail(flight):
                            actors[reply.key.session].deliver(reply)
                        raise
                    for i, job in enumerate(flight.jobs):
                        with torch.no_grad():
                            expected = eager(torch.from_numpy(job.x.copy()))
                        for actual, name in ((policy[i:i+1], 'policy'), (wdl[i:i+1], 'wdl')):
                            want = expected[name].detach().numpy()
                            np.testing.assert_allclose(actual, want, atol=atol, rtol=rtol)
                            max_error = max(max_error, float(np.abs(actual - want).max()))
                    completions = broker.complete(flight, policy, wdl, now=time.monotonic())
                    if held is not None:
                        if any(r.key == held for r in completions):
                            raise AssertionError('old-root native output reached new-root scatter')
                        late_discarded += 1
                    for reply in completions:
                        actors[reply.key.session].deliver(reply)
                    future, flight, held = None, None, None
                if future is None:
                    flight = broker.dispatch(time.monotonic(), max_rows=max_rows)
                    if flight is not None:
                        n = len(flight.jobs)
                        histogram[n] = histogram.get(n, 0) + 1
                        rows += n
                        future = executor.submit(evaluator.evaluate, flight.x)
                        for job in flight.jobs:
                            if (faults and job.key.session == 1 and job.key.epoch == 1
                                    and job.key.request == 3 and 1 not in cancelled):
                                actors[1].deliver(broker.cancel(job.key))
                                held = job.key
                                cancelled.add(1)
                time.sleep(0.0005)
        if faults and (cancelled != {0, 1} or late_discarded != 1):
            raise AssertionError('queued and old-root in-flight cancellation were not exercised')
        if broker.pending or broker.reserved or broker.flight is not None:
            raise AssertionError('played-game controller leaked evaluator work')
        for actor in actors:
            if actor.end is None or actor.end.reason in ('cancelled', 'search_stopped'):
                raise RuntimeError('neural game ended on an unexpected search failure')
            replay_check(actor)
            actor.peer.finish()
        return {'games': [a.report() for a in actors], 'real_rows': rows,
                'native_calls': evaluator.sequence - 1 - before, 'fill_histogram': histogram,
                'peak_reserved': peak_reserved, 'faults': faults,
                'late_old_root_rows_discarded': late_discarded,
                'max_logit_absolute_error': max_error}, actors
    finally:
        for peer in peers:
            peer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reuse-package', type=Path, required=True)
    parser.add_argument('--weights-key', choices=['model', 'swa_model'], default='model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--batch', choices=BATCHES, type=int, default=4)
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    parser.add_argument('--fen', default=rules.START)
    parser.add_argument('--moves', nargs='*', default=[], help='Played pre-root UCI moves, preserving their history')
    parser.add_argument('--max-plies', type=int, default=8)
    parser.add_argument('--simulations', type=int, default=4)
    parser.add_argument('--claims', choices=CLAIM_POLICIES, default='automatic')
    parser.add_argument('--qualification-suite', action='store_true', help='Bounded fixture/control/fault contrasts instead of a single game')
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--cxx', default=shutil.which('clang++'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=['native'])
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.qualification_suite and (args.fen != rules.START or args.moves or args.max_plies != 8 or args.claims != 'automatic'):
        parser.error('--qualification-suite owns its root/ply/claim fixtures; do not combine with game overrides')
    if not 2 <= args.simulations <= 64:
        parser.error('--simulations must be 2..64')
    validate_report_destination(args.report, args.checkpoint)
    protect_inputs(args.report, [args.reuse_package, args.reuse_package.with_suffix('.json')])
    report: dict[str, object] = {'status': 'failed', 'scope': 'played-root diagnostic neural lifecycle; not production search, strength or throughput evidence', 'results': []}
    observations: list[dict[str, object]] = []
    stage = 'preflight'
    try:
        found = tools(args.bun, args.cc, args.cxx)
        compiler = sessions.check_compiler(args.compiler_root)
        a, r = tolerances(args.device, args.atol, args.rtol)
        report['device'] = device_snapshot(args.device, args.device_index)
        torch.set_num_threads(2)
        loaded = load_checkpoint(args.checkpoint, weights_key=args.weights_key)
        stage = 'package-verification'
        manifest = verified_package(args.reuse_package, loaded, batch=args.batch, device=args.device, device_index=args.device_index)
        eager = reuse_reference(loaded, device=args.device, device_index=args.device_index)
        report.update(package=manifest, compiler=compiler, atol=a, rtol=r, claims=args.claims)
        if args.qualification_suite:
            specs = fixtures()
        else:
            root = chess.Board(args.fen)
            for uci in args.moves:
                if root.outcome(claim_draw=False) is not None:
                    raise ValueError('pre-root moves continue past an automatic game result')
                root.push_uci(uci)
            specs = [GameSpec('neural-play', root, args.max_plies, args.claims)]
        with workspace(args.work_dir) as work, compilation_cache(work):
            stage = 'native-build'
            worker = build_worker(work / 'worker', found['cxx'])
            binaries = sessions.build(args.compiler_root, work / 'bend', found['bun'], found['cc'], args.modes)
            evaluator = NativeEvaluator(worker, args.reuse_package)
            try:
                oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
                stage = 'native-play'
                for mode in args.modes:
                    control, baseline = play_games(binaries[mode], oracle, evaluator, eager, specs,
                                                  budget=args.simulations, max_rows=1, atol=a, rtol=r)
                    if args.qualification_suite:
                        check_fixture_results(baseline)
                    observations.append({'mode': mode, 'contrast': 'one_real_row', **control})
                    if args.qualification_suite:
                        for faults in (False, True):
                            observed, actors = play_games(binaries[mode], oracle, evaluator, eager, specs,
                                                          budget=args.simulations, faults=faults, atol=a, rtol=r)
                            same_games(baseline, actors, faults=faults)
                            check_fixture_results(actors)
                            observations.append({'mode': mode, 'contrast': 'faults' if faults else 'batched', **observed})
                    report['results'] = observations
                    write_report(args.report, report)
                evaluator.finish()
                report.update(status='passed', native_calls=evaluator.sequence - 1,
                              oracle_positions=len(oracle.cache), results=observations)
            finally:
                report['native_stderr_tail'] = evaluator.diagnostics()
                evaluator.close()
    except Exception as error:
        report.update(failed_stage=stage, error=str(error), error_type=type(error).__name__, results=observations)
        raise
    finally:
        write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
