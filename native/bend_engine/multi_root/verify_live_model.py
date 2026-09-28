"""Opt-in CPU model composition for the persistent Bend owner, not a scheduler.

The existing independent selected-leaf oracle consumes original model trace bytes.
Only diagnostic slot identities are mapped to unique generation ordinals; no
model output, path, tree or work count is regenerated from a reference result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from .verify import BATCHES, environment, integer, obj, oracle_check
from .verify_live import CONTROL, ROOT, WORK, Client, tree_snapshots, validate

EVENTS = ('cohort_batch', 'cohort_rule', 'native_path', 'native_reply')
BEGIN = 'info string live_result_begin '
OPEN = 'info string live_open deepfin.live-cohort.v1'


def project(lines: list[str], identities: list[tuple[int, int]], batch: int,
            diagnostics: bool) -> dict[str, Any]:
    """Require a complete, uncancelled run before adapting identity namespaces."""
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('empty or duplicate planned identities')
    for slot, gen in identities:
        integer(slot, 1, 16)
        integer(gen, 1, 65535)
    integer(batch, 1, 16)
    report = validate(lines)
    work = report['work']
    if work['batch_size'] != batch or batch not in BATCHES:
        raise ValueError('bound batch mismatch')
    if (work['cancelled_rows'] or work['executed_wasted_rows']
            or work.get('useful_eps', 'missing') is not None
            or work.get('warmup_excluded') is not False):
        raise ValueError('unexpected cancellation or performance claim')
    expected = {identity: i for i, identity in enumerate(identities, 1)}
    seen: list[tuple[int, int]] = []
    active: dict[int, tuple[int, int]] = {}
    retired: set[tuple[int, int]] = set()
    events: list[tuple[str, list[int]]] = []
    row_counts: list[int] = []
    opened = 0
    pending_remove: tuple[int, int] | None = None
    for line in lines:
        if line == OPEN:
            opened += 1
        elif line.startswith(CONTROL + 'admitted '):
            slot, gen = map(int, line.removeprefix(CONTROL + 'admitted ').split())
            identity = (slot, gen)
            if pending_remove is not None or identity not in expected or (slot in active and active[slot] not in retired):
                raise ValueError('unplanned or premature admission')
            active[slot] = identity
            seen.append(identity)
        elif line.startswith(CONTROL + 'pending-remove '):
            slot, gen = map(int, line.removeprefix(CONTROL + 'pending-remove ').split())
            if pending_remove is not None or active.get(slot) != (slot, gen) or (slot, gen) not in retired:
                raise ValueError('unscoped or premature removal reservation')
            pending_remove = (slot, gen)
        elif line.startswith(CONTROL + 'removed '):
            slot, gen = map(int, line.removeprefix(CONTROL + 'removed ').split())
            if (pending_remove != (slot, gen) or active.get(slot) != (slot, gen)
                    or (slot, gen) not in retired):
                raise ValueError('premature removal')
            del active[slot]
            pending_remove = None
        elif line.startswith(ROOT):
            r = obj(line.removeprefix(ROOT))
            identity = (integer(r['slot'], 1, 16), integer(r['generation'], 1, 65535))
            if active.get(identity[0]) != identity:
                raise ValueError('root result has stale identity')
            if (r.get('cancel_requested') is not False or r.get('deadline_expired') is not False
                    or r.get('deadline_offset_ms', 'missing') is not None):
                raise ValueError('model parity run has cancellation/deadline state')
            if (r.get('arena_capacity') != 4096 or r.get('arena_physical_slots') != 4096
                    or r.get('simulation_budget') != 4 or r.get('neural_budget') != 0):
                raise ValueError('unexpected search configuration')
            retired.add(identity)
        elif line.startswith(BEGIN):
            fields = list(map(int, line.removeprefix(BEGIN).split()))
            if len(fields) != 2:
                raise ValueError('malformed result marker')
            identity = (fields[0], fields[1])
            if identity not in expected or active.get(identity[0]) != identity:
                raise ValueError('unscoped result marker')
        elif line.startswith('info string cohort_node '):
            if not diagnostics:
                raise ValueError('diagnostic tree in quiet run')
        elif line.startswith('info string ') and line.split(' ', 3)[2] in EVENTS:
            if not diagnostics:
                raise ValueError('diagnostic event in quiet run')
            _, _, name, text = line.split(' ', 3)
            values = [integer(int(x)) for x in text.split()]
            if name == 'cohort_batch':
                if len(values) != 3 or values[0] != len(row_counts) + 1 or values[2] != batch:
                    raise ValueError('invalid physical batch identity')
                row_counts.append(integer(values[1], 1, batch))
            else:
                if len(values) < 3 or values[0] not in active or active[values[0]] in retired:
                    raise ValueError('unscoped or retired-generation neural event')
                values[0] = expected[active[values[0]]]
            events.append((name, values))
        elif line == CONTROL + 'quit' or line == 'info string live_ready':
            continue
        elif line.startswith(CONTROL + 'pending-install '):
            parts = line.removeprefix(CONTROL + 'pending-install ').split()
            if len(parts) != 2 or tuple(map(int, parts)) not in expected:
                raise ValueError('unplanned install reservation')
        elif line.startswith(WORK):
            obj(line.removeprefix(WORK))
        else:
            raise ValueError('unexpected live-model output: ' + line[:160])
    if pending_remove is not None or opened != 1 or seen != identities or retired != set(identities):
        raise ValueError('incomplete or reordered generation plan')
    snapshots = tree_snapshots(lines) if diagnostics else {}
    roots, nodes = {}, {}
    for identity, ep in expected.items():
        r = report['roots'][f'{identity[0]}:{identity[1]}']
        roots[ep] = {**r, 'root': ep}
        if diagnostics:
            nodes[ep] = dict(enumerate(snapshots[identity]))
    if diagnostics and (len(row_counts) != work['forward_calls']
                        or sum(row_counts) != work['executed_real_rows']):
        raise ValueError('trace events do not reconcile with lifetime work')
    return {'roots': roots, 'nodes': nodes, 'events': events, 'work': work,
            'row_counts': row_counts}


def audit(text: str, calls: int) -> None:
    want = f'native-buffer-audit calls={calls} input_changes=0 output_changes=0 input_tensor_allocations=1'
    if text.strip() != want:
        raise ValueError('native bridge reuse audit mismatch: ' + text[:200])


def model_identity(package: Path, checkpoint: Path) -> tuple[dict[str, Any], Any, Any]:
    import torch
    from native.bend_engine.neural_probe.backend import CHECKPOINT_FORMAT, package_manifest
    from native.bend_engine.neural_probe.checkpoint import EagerReference, load_checkpoint
    manifest, encoding = package_manifest(package)
    if (manifest['format'] != CHECKPOINT_FORMAT or manifest['device'] != 'cpu'
            or manifest['dtype'] != 'float32' or not encoding.history_rep_fix):
        raise ValueError('requires an exact trusted v3 CPU-F32 checkpoint package')
    loaded = load_checkpoint(checkpoint, weights_key=str(manifest['weights_key']))
    if loaded.encoding != encoding or loaded.identity['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise ValueError('checkpoint/encoding mismatch')
    return manifest, loaded, EagerReference(loaded.model, torch.device('cpu'), torch.float32)


def session(binary: Path, package: Path, positions: list[str], directory: Path,
            diagnostics: bool, *, shared: bool, dispatch_environment: dict[str, str] | None = None) -> tuple[list[str], list[tuple[int, int]], Path]:
    directory.mkdir()
    trace = directory / 'model.trace'
    env = {**environment(), 'CUDA_VISIBLE_DEVICES': '',
           'DEEPFIN_BEND_MODEL_PACKAGE': str(package.resolve()), 'DEEPFIN_BEND_BUFFER_AUDIT': '1'}
    env = {k: v for k, v in env.items() if not k.startswith('DEEPFIN_TEST_')}
    if dispatch_environment is not None:
        if set(dispatch_environment) != {'DEEPFIN_COHORT_DISPATCH', 'DEEPFIN_COHORT_PROFILE_PACKAGE_SHA256'}:
            raise ValueError('unexpected dispatch test environment')
        env.update(dispatch_environment)
    if diagnostics:
        env['DEEPFIN_BEND_MODEL_TRACE'] = str(trace)
    c = Client(binary.resolve(), simulations=4, environment=env, diagnostics=diagnostics)
    plan: list[tuple[int, int]] = []
    try:
        if shared:
            if len(positions) != 2:
                raise ValueError('shared case needs exactly two roots')
            plan = [(1, 1), (2, 2)]
            c.send('add ' + positions[0] + '\nadd ' + positions[1])
            found = set()
            while len(found) != 2:
                r = obj(c.until(ROOT, 60).removeprefix(ROOT))
                found.add((r['slot'], r['generation']))
            if found != set(plan):
                raise ValueError('wrong shared roots')
        else:
            for gen, position in enumerate(positions, 1):
                # Exercise both replacement and explicit remove/re-add in one process.
                if gen == 8:
                    c.send(f'remove 1 {gen - 1}')
                    c.until(CONTROL + f'removed 1 {gen - 1}', 60)
                command = 'add ' if gen in (1, 8) else f'replace 1 {gen - 1} '
                c.send(command + position)
                r = obj(c.until(ROOT, 60).removeprefix(ROOT))
                if (r['slot'], r['generation']) != (1, gen):
                    raise ValueError('incorrect replacement identity')
                plan.append((1, gen))
        c.send('quit')
        c.proc.wait(timeout=60)
        for reader in c.readers:
            reader.join(timeout=5)
            if reader.is_alive():
                raise ValueError('unjoined output reader')
        if c.proc.returncode:
            raise ValueError(f'native model exit {c.proc.returncode}: {c.errors}')
        lines = c.lines.copy()
        parsed = validate(lines)
        audit(''.join(c.errors), parsed['work']['forward_calls'])
        if diagnostics != trace.exists():
            raise ValueError('trace-mode mismatch')
        (directory / 'stdout.txt').write_text('\n'.join(lines) + '\n')
        return lines, plan, trace
    finally:
        c.close()


def qualify(binary: Path, package: Path, checkpoint: Path, oracle: Path) -> dict[str, Any]:
    import torch
    from chess_anti_engine.encoding import rep_fix
    from native.bend_engine.standalone.verify_neural import fixtures
    from native.bend_engine.standalone.verify_rules import position
    torch.set_num_threads(2)
    rep_fix.apply(True)
    manifest, loaded, eager = model_identity(package, checkpoint)
    batch, channels = manifest['batch'], manifest['channels']
    results = []
    with tempfile.TemporaryDirectory(prefix='deepfin-live-model-') as temporary:
        directory = Path(temporary)
        for name, roots, shared in [('reuse', fixtures(), False), ('shared', fixtures()[:2], True)]:
            positions = [position(r).removeprefix('position ') for r in roots]
            lines, identities, trace = session(binary, package, positions, directory / name, True, shared=shared)
            parsed = project(lines, identities, batch, True)
            checked = oracle_check(parsed, roots, trace, oracle, channels, batch, 4, 2, 0,
                                   eager, loaded.encoding.input_history_encoding)
            # A native-trace corruption must fail the same independent oracle.
            damaged = bytearray(trace.read_bytes())
            output_offset = 24 + batch * channels * 64 * 4
            if len(damaged) < output_offset + 4:
                raise ValueError('missing native model trace for negative control')
            damaged[output_offset:output_offset + 4] = bytes.fromhex('0000c07f')
            negative = directory / (name + '-nonfinite.trace')
            negative.write_bytes(damaged)
            try:
                oracle_check(parsed, roots, negative, oracle, channels, batch, 4, 2, 0,
                             eager, loaded.encoding.input_history_encoding)
            except (ValueError, AssertionError):
                pass
            else:
                raise ValueError('nonfinite native trace escaped independent oracle')
            quiet, quiet_ids, _ = session(binary, package, positions, directory / (name + '-quiet'), False, shared=shared)
            plain = project(quiet, quiet_ids, batch, False)
            if plain['roots'] != parsed['roots']:
                raise ValueError('diagnostics changed per-generation result')
            for key in ('forward_calls', 'executed_real_rows', 'accepted_neural_rows', 'padded_rows'):
                if plain['work'][key] != parsed['work'][key]:
                    raise ValueError('diagnostics changed physical work: ' + key)
            if shared and batch > 1 and max(parsed['row_counts'], default=0) < 2:
                raise ValueError('shared model call was not exercised')
            results.append({'case': name, **checked, 'generations': identities,
                            'physical_batch_rows': parsed['row_counts'], 'quiet_matches': True,
                            'tree_sha256': hashlib.sha256(json.dumps(parsed['nodes'], sort_keys=True).encode()).hexdigest(),
                            'transcript_sha256': hashlib.sha256('\n'.join(lines).encode()).hexdigest(),
                            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
                            'root_results': parsed['roots'], 'bridge_audit_passed': True,
                            'nonfinite_trace_rejected': True})
    return {'status': 'passed', 'scope': 'persistent live Bend owner with actual CPU-F32 AOTI model',
            'live_model_qualified': True, 'batch': batch, 'channels': channels, 'cases': results,
            'parameter_count': loaded.identity['parameter_count'], 'torch_version': str(torch.__version__),
            'checkpoint_sha256': loaded.identity['checkpoint_sha256'], 'package_sha256': manifest['sha256'],
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'atol': 2e-6, 'rtol': 2e-5, 'cancellation_injected': False, 'gpu_qualified': False,
            'speed_or_strength_qualified': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'package', 'checkpoint', 'oracle', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.report.exists():
        p.error('report already exists; use a new evidence path')
    try:
        report = qualify(a.binary, a.package, a.checkpoint, a.oracle)
    except Exception as error:
        a.report.write_text(json.dumps({'status': 'failed', 'live_model_qualified': False,
                                       'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')
        raise
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'batch': report['batch'], 'cases': len(report['cases'])}))


if __name__ == '__main__':
    main()
