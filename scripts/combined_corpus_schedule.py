"""Admit an ordered matched B100/SF versus B100/native-V50 corpus set.

No training launcher. The optional prospective pass uses the historical sampler,
not main's replay implementation. Existing single-source tools are unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

FROZEN_PINS = {
    'chess_anti_engine/replay/game_epoch.py': '621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3',
    'scripts/lc0_control_train.py': '52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337',
}
TEACHER = '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0'
ARMS = ('source', 'B100', 'V50')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pin(ref: dict[str, Any]) -> dict[str, Any]:
    path = Path(ref['path'])
    require(path.is_absolute() and sha(path) == ref['sha256'], f'input pin differs: {path}')
    return json.loads(path.read_text())


def same_json(a: Any, b: Any) -> bool:
    # Historical source summaries retain NaN diagnostics, not numeric evidence.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def subset(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    require(all(actual.get(k) == v for k, v in expected.items()), f'{label} differs')


def summary_ref(cohort: dict[str, Any], arm: str) -> dict[str, Any]:
    return cohort['roots'][arm]['summary']


def root_path(cohort: dict[str, Any], arm: str) -> Path:
    return Path(summary_ref(cohort, arm)['path']).parent


def validate_recipe(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    """Check unchanged original SF lineage and the two fixed target recipes."""
    source, policy, value = (read_pin(summary_ref(cohort, arm)) for arm in ARMS)
    require('value_target_postprocess' not in source and 'policy_target_postprocess' not in source,
            'original SF source is already target-modified')
    specs = source['shards']
    require(specs and all(type(s['rows']) is int and s['rows'] > 0 for s in specs), 'empty/invalid layout')
    names = [s['path'] for s in specs]
    require(len(set(names)) == len(names) and all(re.fullmatch(r'shard_\d{6}\.zarr', n) for n in names),
            'duplicate/invalid shard name')
    require(names == sorted(names), 'source shard order differs')
    rows = sum(s['rows'] for s in specs)
    require(rows == cohort['rows'], 'cohort row count differs')
    p = read_pin(cohort['policy_recipe'])
    v = read_pin(cohort['value_recipe'])
    subset(p, {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
               'alpha': 1.0, 'bt4_temperature': .5, 'rows': rows,
               'expected_shards': len(specs), 'source_dir': str(root_path(cohort, 'source')),
               'source_derive_summary_sha256': summary_ref(cohort, 'source')['sha256'],
               'mutated_arrays': ['policy_target']}, 'B100 recipe')
    require(same_json(policy.get('policy_target_postprocess'), p)
            and same_json({k: x for k, x in policy.items() if k != 'policy_target_postprocess'}, source),
            'B100 changed original source/value lineage')
    subset(v, {'schema': 1, 'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
               'algorithm': 'normalized-wdl-arithmetic-float16-v1', 'sf_weight': .5, 'bt4_weight': .5,
               'wdl_order': 'WDL', 'wdl_pov': 'side_to_move', 'wdl_kind': 'probabilities',
               'wdl_output': '/output/wdl', 'onnx_sha256': TEACHER, 'rows': rows, 'shards': len(specs),
               'source_dir': str(root_path(cohort, 'B100')), 'sf_source_dir': str(root_path(cohort, 'source')),
               'source_derive_summary_sha256': summary_ref(cohort, 'B100')['sha256'],
               'source_policy_summary_sha256': cohort['policy_recipe']['sha256'],
               'sf_derive_summary_sha256': summary_ref(cohort, 'source')['sha256'],
               'mutated_arrays': ['search_wdl'], 'value_scheme': 'sf-bt4-native-alpha=0.5',
               'value_source': 'stored-sf-search-and-derived-bt4-wdl;onnx=' + TEACHER
                   + ';output=/output/wdl;bt4_weight=0.5'}, 'V50 recipe')
    expected = dict(policy)
    expected['value_scheme'] = {'name': v['value_scheme'], 'source': v['value_source']}
    expected['value_target_postprocess'] = {k: x for k, x in v.items() if k != 'outputs'}
    require(same_json(value, expected), 'V50 changed policy/history/source lineage')
    require([(s['path'], s['rows']) for s in v['outputs']] == [(s['path'], s['rows']) for s in specs],
            'V50 output coverage/order differs')
    require(sha(root_path(cohort, 'V50') / 'bt4_policy_mix_summary.json') == cohort['policy_recipe']['sha256'],
            'V50 copied policy recipe differs')
    return specs


def admit(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Metadata admission only; pinned qualification is inherited, not rerun."""
    subset(manifest, {'schema': 1, 'kind': 'matched-b100-sf-native50-corpus-set',
                      'batch_size': 512,
                      'game_identity_contract': 'disjoint-whole-game-raw-shard-selections-v1'}, 'manifest')
    require(type(manifest['seed']) is int and 0 <= manifest['seed'] < 2**32, 'invalid schedule seed')
    require(set(manifest['runtime_pins']) == set(FROZEN_PINS), 'frozen runtime pin set differs')
    require(manifest['runtime_pins'] == FROZEN_PINS, 'frozen runtime changed')
    require(manifest['cohorts'], 'empty corpus set')
    ids: set[str] = set()
    roots: set[Path] = set()
    raw_seen: set[tuple[str, str]] = set()
    historical_namespaces: set[str] = set()
    mapping = []
    for c in manifest['cohorts']:
        require(isinstance(c['id'], str) and re.fullmatch(r'[A-Za-z0-9_-]+', c['id']) is not None
                and c['id'] not in ids, 'duplicate/invalid logical cohort')
        ids.add(c['id'])
        for arm in ARMS:
            root = root_path(c, arm)
            require(root.is_absolute() and root.is_dir() and not root.is_symlink()
                    and root.resolve() not in roots, 'duplicate/aliased corpus root')
            require(not root.name.endswith('.writing') and not root.with_name(root.name + '.writing').exists()
                    and not (root / 'failed.json').exists(), 'unpublished corpus')
            roots.add(root.resolve())
        qualification = read_pin(c['source_qualification'])
        require(bool(qualification), 'missing source qualification')
        require(re.fullmatch(r'[0-9a-f]{64}', c['source_namespace']) is not None,
                'invalid source namespace')
        if c['identity_kind'] == 'qualified-g10-selection':
            # This is the existing metadata assessment, including its historical
            # source-version caveat, not a fabricated generation attestation.
            identity = read_pin(c['identity_receipt'])
            require(identity['status'] == 'METADATA_ONLY_DISJOINT_RAW_ROSTER_AND_SOURCE_IDENTITY_MAPPING',
                    'source identity mapping status differs')
            matches = [x for x in identity['cohorts'] if x['cohort'] == c['id']]
            require(len(matches) == 1, 'logical cohort missing/ambiguous in identity mapping')
            recorded = matches[0]
            subset(recorded, {'source_namespace': c['source_namespace'],
                              'derived_summary': summary_ref(c, 'source'),
                              'qualification': c['source_qualification'],
                              'raw_shards': c['raw_shards'], 'rows': c['rows']}, 'source identity mapping')
            require(qualification['status'] == 'complete' and qualification['rows'] == c['rows'],
                    'source qualification is incomplete')
            require(set(qualification['per_raw_shard_survivors']) == set(c['raw_shards'])
                    and sum(qualification['per_raw_shard_survivors'].values()) == c['rows'],
                    'qualified physical selection differs')
            require(c['raw_shards'], 'empty raw selection')
        else:
            require(c['identity_kind'] == 'historical-single-source', 'unknown identity contract')
            require(c['source_namespace'] not in historical_namespaces, 'historical source repeated')
            # A single historical corpus has its existing qualified game namespace.
            # Never fabricate its raw-shard roster or merge it into a G10 run.
            subset(qualification, {'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION',
                                   'corpus': str(root_path(c, 'B100')), 'rows': c['rows'],
                                   'source': {'path': str(root_path(c, 'source')),
                                              'derive_sha256': summary_ref(c, 'source')['sha256']}},
                   'historical source qualification')
            historical_namespaces.add(c['source_namespace'])
            require(not c['raw_shards'], 'historical raw roster must not be invented')
        for raw in c['raw_shards']:
            require(isinstance(raw, str) and Path(raw).name == raw and raw not in ('', '.', '..'),
                    'invalid raw shard identity')
            key = (c['source_namespace'], raw)
            require(key not in raw_seen, 'overlapping physical raw shard selection')
            raw_seen.add(key)
        specs = validate_recipe(c)
        for arm in ARMS:
            root = root_path(c, arm)
            actual = sorted(root.glob('shard_*.zarr'))
            require([p.name for p in actual] == [x['path'] for x in specs]
                    and all(p.is_dir() and not p.is_symlink() for p in actual), 'actual shard roster differs')
        for index, spec in enumerate(specs):
            mapping.append({'cohort': c['id'], 'shard_index': index, 'rows': spec['rows'],
                            'namespace': c['source_namespace'],
                            'paths': {arm: str(root_path(c, arm) / spec['path']) for arm in ARMS}})
    require(len(historical_namespaces) <= 1, 'only one qualified historical source is supported')
    require(not historical_namespaces.intersection(ns for ns, _ in raw_seen), 'historical/G10 namespaces collide')
    require(sum(s['rows'] for s in mapping) == manifest['expected_rows'], 'union row count differs')
    require(len(mapping) == manifest['expected_shards'], 'union shard count differs')
    return mapping


def ordered_paths(mapping: list[dict[str, Any]], arm: str) -> list[Path]:
    return [Path(row['paths'][arm]) for row in mapping]


def canonical_records(records: Any, mapping: list[dict[str, Any]], arm: str) -> list[Any]:
    """Full path lookup, never basename lookup; preserve actual assigned keys."""
    by_path = {Path(row['paths'][arm]).resolve(): row for row in mapping}
    require(len(by_path) == len(mapping) == len(records), 'planner roster count differs')
    result = []
    for record, row in zip(records, mapping, strict=True):
        require(record.path.resolve() == Path(row['paths'][arm]).resolve(), 'planner source order differs')
        require(record.rows == row['rows'], 'planner row count differs')
        result.append(replace(record, path=Path(row['paths']['source'])))
    return result


def same_records(left: Any, right: Any, np: Any) -> None:
    require(len(left) == len(right), 'planner record count differs')
    for a, b in zip(left, right, strict=True):
        require(a.path == b.path and a.rows == b.rows and all(
            np.array_equal(getattr(a, name), getattr(b, name))
            for name in ('game_ids', 'game_keys', 'game_counts')), 'canonical game grouping differs')


def scan_columns(epoch: Any, paths: list[Path], mapping: list[dict[str, Any]], guard: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Witness the historical scan's inputs; decode each small column once."""
    import numpy as np

    loader = epoch.load_shard_arrays
    wanted = {path.resolve(): row['rows'] for path, row in zip(paths, mapping, strict=True)}
    witnesses: dict[Path, dict[str, Any]] = {}
    def witnessed_loader(path: Path, **kwargs: Any) -> Any:
        guard()
        arrays, attrs = loader(path, **kwargs)
        n = int(arrays['x'].shape[0])  # Shape only; never feature decode.
        game = np.asarray(arrays['game_id'], dtype='<i8')
        present = np.asarray(arrays['has_game_id'], dtype=np.bool_)
        require(n == wanted[path.resolve()] and game.shape == present.shape == (n,) and bool(present.all()),
                'incomplete game identity')
        require(path.resolve() not in witnesses, 'shard scanned twice')
        witnesses[path.resolve()] = {'rows': n, 'game_id_sha256': hashlib.sha256(game.tobytes()).hexdigest(),
                                    'has_game_id_sha256': hashlib.sha256(present.tobytes()).hexdigest()}
        # Original scanner consumes these same decoded values; no target/feature
        # access or planner modification, and no second storage read.
        return dict(arrays, game_id=game, has_game_id=present), attrs
    epoch.load_shard_arrays = witnessed_loader
    try:
        records = epoch._scan_shards(paths, workers=2)
    finally:
        epoch.load_shard_arrays = loader
    require(len(witnesses) == len(paths), 'metadata witness coverage differs')
    return records, [witnesses[path.resolve()] for path in paths]


def prospective(manifest: dict[str, Any], mapping: list[dict[str, Any]], guard: Any) -> dict[str, Any]:
    runtime = Path(manifest['runtime'])
    for suffix, digest in FROZEN_PINS.items():
        require(sha(runtime / suffix) == digest, 'historical runtime pin differs')
    require('chess_anti_engine.replay.game_epoch' not in sys.modules, 'sampler already imported')
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    blosc: Any = importlib.import_module('numcodecs.blosc')
    epoch: Any = importlib.import_module('chess_anti_engine.replay.game_epoch')

    require(Path(epoch.__file__).resolve() == runtime.resolve() / 'chess_anti_engine/replay/game_epoch.py',
            'wrong sampler imported')
    torch.set_num_threads(2)
    blosc.set_nthreads(1)
    source_records = None
    source_columns = None
    source_plan: Any = None
    arms = {}
    for arm in ARMS:
        paths = ordered_paths(mapping, arm)
        records, columns = scan_columns(epoch, paths, mapping, guard)
        guard()
        normalized = canonical_records(records, mapping, arm)
        if source_records is None:
            source_records, source_columns = normalized, columns
            source_plan = epoch._plan_epoch(records, batch_size=manifest['batch_size'], seed=manifest['seed'])
            plan = source_plan
        else:
            require(columns == source_columns, 'ordered full game columns differ')
            same_records(normalized, source_records, np)
            plan = epoch._plan_epoch(records, batch_size=manifest['batch_size'], seed=manifest['seed'])
        require(source_plan is not None, 'missing source plan')
        require(np.array_equal(plan.load_counts, source_plan.load_counts)
                and np.array_equal(plan.batch_rows, source_plan.batch_rows), 'physical batch/load schedule differs')
        arms[arm] = {'physical_plan': plan.as_dict(), 'canonical_plan_sha256': source_plan.plan_sha256,
                     'metadata_matches_source': True, 'training_completed': False}
        guard()
    return {'arms': arms, 'ordered_source_columns': source_columns,
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
            'proof': 'Equal canonical planner records and full ordered game columns under the pinned sampler; row-offset equality is code-backed inference, not a realized training observation.'}


def resource_guard(output_parent: Path) -> None:
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    require(available >= 32 * 1024**3, 'available memory below32GiB')
    require(shutil.disk_usage(output_parent).free >= 150 * 1024**3, 'disk reserve below150GiB')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='read only game columns and compute prospective schedules')
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize and os.environ.get('PYTHONOPTIMIZE') in (None, '', '0'), 'optimized Python refused')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be hidden')
    require(sha(args.manifest) == args.expected_manifest_sha256, 'manifest pin differs')
    require(not args.output.exists(), 'output must be fresh')
    sampled_at = 0.0
    def guard() -> None:
        nonlocal sampled_at
        require(time.time() < args.deadline_unix, 'deadline exhausted')
        require(not (args.output.parent / 'STOP').exists(), 'STOP requested')
        if time.monotonic() - sampled_at >= 5:
            resource_guard(args.output.parent)
            sampled_at = time.monotonic()
    guard()
    manifest = json.loads(args.manifest.read_text())
    mapping = admit(manifest)
    report = {'status': 'PASS_CORPUS_SET_METADATA_NOT_TRAINING', 'manifest_sha256': args.expected_manifest_sha256,
              'mapping': mapping, 'rows': sum(r['rows'] for r in mapping),
              'seed': manifest['seed'], 'batch_size': manifest['batch_size'],
              'trainer_shards': {arm: [str(root_path(c, arm)) for c in manifest['cohorts']] for arm in ('B100', 'V50')},
              'training_role_to_arm': {'Combined35M_SF100': 'B100', 'Combined35M_V50': 'V50'},
              'limits': ['Producer/source qualification is inherited from pinned receipts, not repeated.',
                         'No feature/target read or full training admission; trainer history/value gates remain required.',
                         'Whole-game equivalence inherits the qualified generator contract and exact disjoint rosters; historical per-shard source-code attestation was not added.']}
    if args.execute:
        report.update(prospective(manifest, mapping, guard))
        report['status'] = 'PASS_CORPUS_SET_PROSPECTIVE_NOT_TRAINING'
    guard()
    require(sha(args.manifest) == args.expected_manifest_sha256, 'manifest changed')
    # Recheck small binding files after the pass, without repeating array reads.
    require(admit(manifest) == mapping, 'corpus metadata changed')
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
