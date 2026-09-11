#!/usr/bin/env python3
"""Offline three-teacher WDL mixture; copy B100 and change only search_wdl.

Completed original-source BT4 WDL and Ceres dual-head banks are consumed without
inference. Historical BT4 producer identity is preserved, never rebound to this
checkout. Output is admissible only after a final atomic publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ceres_target_mix as policy
from scripts import bt4_value_rewrite as copies
from scripts.sf_policy_rewrite import ARRAYS, require

wdl = policy.shared
ceres = policy.ceres
SUMMARY = 'ceres_value_mix_summary.json'
DERIVE_SUMMARY = 'derive_targets_summary.json'
POLICY_SUMMARY = 'bt4_policy_mix_summary.json'
ALGORITHM = 'normalized-wdl-probability-mixture-float16-v1'
VALUE_SCHEME = 'sf50-bt4-native25-ceres-dual25'
BT4_MODEL = '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0'
HISTORICAL_BT4_PRODUCER = {
    'chess_anti_engine/encoding/lc0.py': 'a20b56a7c0666e134791855c0f124a66504013983f8f97948539a015dac9c3ee',
    'scripts/bt4_derived_wdl_sidecar.py': '6fdaf63038f58e71d6ee6ab8fc6bbc04cca422b1469ecd0f65eabf890085a370',
    'scripts/bt4_raw_corpus_sidecar.py': '1c63aa4147ef8717855d226f30094cbc03431e556cd7106d0619adee7945363a',
}
WEIGHTS = {'sf': .5, 'bt4': .25, 'ceres': .25}
CERES_VALUE_PROFILE = {'primary_temperature': .55, 'secondary_temperature': 1.5,
    'primary_weight': .6, 'secondary_weight': .4, 'arithmetic': 'float64-probability-mixture',
    'parity': 'mathematical-analogue-not-native-fp16'}


def value_source() -> str:
    return f'{VALUE_SCHEME};bt4={BT4_MODEL};ceres={ceres.MODEL_SHA};primary=0.6@0.55;secondary=0.4@1.5'


def producer_pins() -> dict[str, str]:
    return {**policy.producer_pins(), str(Path(__file__).resolve()): wdl.file_sha256(__file__)}


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 3 and bool(np.isfinite(values).all()),
            'Ceres WDL logits must be finite Nx3')
    scaled = (values - values.max(axis=1, keepdims=True)) / temperature
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def target(sf: np.ndarray, bt4: np.ndarray, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    require(sf.shape == bt4.shape == primary.shape == secondary.shape, 'WDL row shapes differ')
    return (.5 * copies.normalized(sf) + .25 * copies.normalized(bt4)
            + .25 * (.6 * softmax(primary, .55) + .4 * softmax(secondary, 1.5)))


def omit_array(files: dict[str, str], array: str) -> dict[str, str]:
    return {k: v for k, v in files.items() if k != '.zattrs' and k.split('/')[0] != array}


def read_manifest(path: Path, digest: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    require(wdl.file_sha256(path) == digest, 'manifest pin differs')
    m: dict[str, Any] = json.loads(path.read_text())
    require(m.get('schema') == 1, 'unsupported value manifest')
    sf_root, source = Path(m['sf_source']).resolve(), Path(m['source']).resolve()
    sf, specs = wdl.source_inventory(argparse.Namespace(source=str(sf_root),
        expected_source_summary_sha256=m['sf_summary_sha256'], start_shard=0, max_shards=sys.maxsize))
    for name, key in ((DERIVE_SUMMARY, 'source_summary_sha256'),
                      (POLICY_SUMMARY, 'source_policy_summary_sha256')):
        require(wdl.file_sha256(source / name) == m[key], 'B100 summary pin differs')
    base: dict[str, Any] = json.loads((source / DERIVE_SUMMARY).read_text())
    mix: dict[str, Any] = json.loads((source / POLICY_SUMMARY).read_text())
    require(copies.equal_json({k: v for k, v in base.items() if k != 'policy_target_postprocess'}, sf)
            and copies.equal_json(base.get('policy_target_postprocess'), mix),
            'B100 changed original value/history lineage')
    expected = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
        'alpha': 1.0, 'bt4_temperature': .5, 'rows': sum(s['rows'] for s in specs),
        'expected_shards': len(specs), 'source_dir': str(sf_root),
        'source_derive_summary_sha256': m['sf_summary_sha256'], 'mutated_arrays': ['policy_target']}
    require(all(mix.get(k) == v for k, v in expected.items()), 'requires original B100 policy')
    copies.inventory(source, specs)
    entries = m['entries']
    require(isinstance(entries, list) and [e['shard'] for e in entries] == [s['path'] for s in specs],
            'full source shard coverage required exactly once')
    for key in ('bt4', 'ceres'):
        require(all(Path(e[key]).is_absolute() and not Path(e[key]).name.endswith('.writing') for e in entries),
                'absolute completed teacher shard required')
        require(len({str(Path(e[key]).resolve()) for e in entries}) == len(specs), 'duplicate teacher shard')
    teacher = m['teachers']['bt4']
    require(teacher['model_sha256'] == BT4_MODEL and teacher['producer'] == HISTORICAL_BT4_PRODUCER
            and teacher['requested_wdl'] == {'kind': 'probabilities', 'output': '/output/wdl'},
            'historical BT4 collector/teacher differs')
    expected_ceres = {'model_sha256': ceres.MODEL_SHA, 'profile': ceres.EXTENDED_PROFILE,
        'backend': ceres.backend(argparse.Namespace(pad_final_batch=True, retain_value2=True))}
    require(m['teachers']['ceres'] == expected_ceres, 'Ceres dual-head profile differs')
    return m, sf, base, specs


def verify_teachers(m: dict[str, Any], entry: dict[str, Any], spec: dict[str, Any],
                    original: Any, source_state: str, batch_size: int) -> tuple[Any, Any]:
    sf_root = Path(m['sf_source']).resolve()
    teacher = m['teachers']['bt4']
    expected = {'schema': 1, 'source_dir': str(sf_root), 'source_shard': spec['path'],
        'source_summary_sha256': m['sf_summary_sha256'], 'source_storage_identity': source_state,
        'rows': spec['rows'], 'onnx': teacher['onnx'], 'onnx_sha256': teacher['model_sha256'],
        'requested_wdl': teacher['requested_wdl'], 'history_lineage': wdl.LINEAGE,
        'producer': teacher['producer']}
    # verify_cached compares actual historical binding verbatim and verifies array contents.
    battrs = wdl.verify_cached(Path(entry['bt4']), expected, batch_size)
    bg: Any = zarr.open_group(entry['bt4'], mode='r')
    source_hashes = {name: wdl.raw.sha_array(np.asarray(original[name][:])) for name in wdl.COLUMNS}
    require(source_hashes == battrs['source_array_sha256'], 'BT4 source content differs')
    for start in range(0, spec['rows'], batch_size):
        end = min(start + batch_size, spec['rows'])
        for name in ('game_id', 'ply_index'):
            require(bool(np.all(original['has_' + name][start:end] == 1))
                    and np.array_equal(bg[name][start:end], original[name][start:end]),
                    'BT4 row identity differs')
        feed = wdl.stored_feed(original['x'][start:end]).astype(battrs['input']['dtype'])
        require(np.array_equal(wdl.row_digests(feed), bg['lc0_feed_sha256'][start:end]),
                'BT4 actual LC0 feed differs')
    binding = entry['ceres_binding']
    ct = m['teachers']['ceres']
    require('selected_rows' not in binding and binding['source'] == str(sf_root)
            and binding['shard'] == spec['path'] and binding['rows'] == spec['rows']
            and binding['source_storage_identity'] == source_state
            and binding['summary_sha256'] == m['sf_summary_sha256']
            and all(binding[k] == ct[k] for k in ('model_sha256', 'profile', 'backend')),
            'Ceres source/history/shard binding differs')
    cattrs = ceres.verify_cached(Path(entry['ceres']), binding)
    cg: Any = zarr.open_group(entry['ceres'], mode='r')
    require('value2_logits' in cg, 'Ceres secondary head missing')
    policy.check_ceres_alignment(original, cg, cattrs, spec['rows'])
    return bg, cg


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    wdl.set_nthreads(2)
    require(type(args.batch_size) is int and 0 < args.batch_size <= 512, 'invalid batch size')
    require(math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0, 'invalid reserve')
    require(math.isfinite(args.max_seconds) and args.max_seconds > 0, 'invalid deadline')
    manifest = Path(args.manifest).resolve()
    m, sf, base, specs = read_manifest(manifest, args.expected_manifest_sha256)
    source, sf_root = Path(m['source']).resolve(), Path(m['sf_source']).resolve()
    out = Path(args.out).absolute()
    writing = out.with_name(out.name + '.writing')
    require(not os.path.lexists(out) and not os.path.lexists(writing), 'output or partial exists')
    inputs = [source, sf_root, manifest, *(Path(e[k]).resolve() for e in m['entries'] for k in ('bt4', 'ceres'))]
    require(all(dest != p and p not in dest.parents and dest not in p.parents
                for dest in (out, writing) for p in inputs), 'output overlaps input')
    require(out.parent.is_dir() and out.parent.resolve() == out.parent, 'canonical existing output parent required')
    started = time.monotonic()
    pins = producer_pins()
    states: dict[Path, str] = {}

    def guard() -> None:
        require(time.monotonic() - started < args.max_seconds, 'value rewrite deadline')
        require(shutil.disk_usage(out.parent).free >= args.minimum_free_gib * 2**30, 'disk reserve')
        require(not args.stop or not Path(args.stop).exists(), 'STOP requested')

    guard()
    writing.mkdir()
    outputs: list[dict[str, Any]] = []
    max_error = max_l1 = 0.0
    changed_total = 0
    for spec, entry in zip(specs, m['entries'], strict=True):
        guard()
        original_path, src = sf_root / spec['path'], source / spec['path']
        local_states = {p: wdl.storage_identity(p) for p in
                        (original_path, src, Path(entry['bt4']), Path(entry['ceres']))}
        original = wdl.source_arrays(original_path, sf, spec['rows'])
        group: Any = zarr.open_group(str(src), mode='r')
        require(set(original.array_keys()) == set(group.array_keys()) == set(ARRAYS), '17-array corpus required')
        attrs = dict(group.attrs)
        require({k: v for k, v in attrs.items() if not k.startswith('policy_target_mix_')}
                == dict(original.attrs), 'B100 changed original metadata')
        require(attrs.get('policy_target_mix_kind') == 'global'
                and attrs.get('policy_target_mix_alpha') == 1.0
                and attrs.get('policy_target_mix_bt4_temperature') == .5, 'B100 policy stamp differs')
        for name in ARRAYS:
            wdl.complete_chunks(original[name])
            wdl.complete_chunks(group[name])
            require(group[name].shape[0] == spec['rows'], 'array row count differs')
        require(group['search_wdl'].shape == (spec['rows'], 3)
                and group['search_wdl'].dtype == np.dtype('float16')
                and bool(np.all(group['has_search_wdl'][:] == 1)), 'source WDL coverage/layout differs')
        source_files = copies.file_map(src)
        require(omit_array(source_files, 'policy_target') == omit_array(copies.file_map(original_path), 'policy_target'),
                'B100 nonpolicy bytes differ from original SF')
        bg, cg = verify_teachers(m, entry, spec, original, local_states[original_path], args.batch_size)
        guard()
        dest_path = writing / spec['path']
        shutil.copytree(src, dest_path)
        require(copies.file_map(dest_path) == source_files, 'copied bytes differ')
        dest: Any = zarr.open_group(str(dest_path), mode='a')
        digest = hashlib.sha256()
        changed = 0
        for start in range(0, spec['rows'], args.batch_size):
            guard()
            end = min(start + args.batch_size, spec['rows'])
            old = original['search_wdl'][start:end]
            ideal = target(old, bg['bt4_wdl_raw'][start:end], cg['value_logits'][start:end], cg['value2_logits'][start:end])
            stored = ideal.astype(np.float16)
            error = float(np.abs(stored.astype(np.float64).sum(1) - 1).max())
            require(error <= 2**-10, 'stored WDL mass differs')
            max_error = max(max_error, error)
            max_l1 = max(max_l1, float(np.abs(ideal - stored).sum(1).max()))
            changed += int(np.count_nonzero(np.any(stored != old, axis=1)))
            dest['search_wdl'][start:end] = stored
            require(np.array_equal(dest['search_wdl'][start:end], stored), 'WDL readback differs')
            digest.update(stored.tobytes(order='C'))
        stamp = {'schema': 1, 'kind': 'sf-bt4-ceres-value', 'weights': WEIGHTS,
                 'ceres_value_profile': CERES_VALUE_PROFILE, 'manifest_sha256': args.expected_manifest_sha256}
        dest.attrs.update(derive_value_scheme=VALUE_SCHEME, derive_value_source=value_source(),
                          value_target_postprocess=stamp)
        final_files = copies.file_map(dest_path)
        require(omit_array(source_files, 'search_wdl') == omit_array(final_files, 'search_wdl'),
                'nonvalue compressed bytes changed')
        require(all(wdl.storage_identity(p) == v for p, v in local_states.items()), 'input changed during rewrite')
        states.update(local_states)
        outputs.append({'path': spec['path'], 'rows': spec['rows'], 'changed_rows': changed,
            'source_storage_identity': local_states[src], 'sf_storage_identity': local_states[original_path],
            'bt4_storage_identity': local_states[Path(entry['bt4'])], 'ceres_storage_identity': local_states[Path(entry['ceres'])],
            'search_wdl_sha256': digest.hexdigest(), 'attrs_sha256': wdl.file_sha256(dest_path / '.zattrs'),
            'files_manifest_sha256': hashlib.sha256(json.dumps(final_files, sort_keys=True).encode()).hexdigest(),
            'output_storage_identity': wdl.storage_identity(dest_path)})
        changed_total += changed
    for path, identity in states.items():
        guard()
        require(wdl.storage_identity(path) == identity, 'previous input changed before completion')
    read_manifest(manifest, args.expected_manifest_sha256)
    require(producer_pins() == pins, 'producer changed during rewrite')
    result = {'schema': 1, 'status': 'COMPLETE', 'complete': True, 'kind': 'sf-bt4-ceres-value',
        'algorithm': ALGORITHM, 'weights': WEIGHTS, 'ceres_value_profile': CERES_VALUE_PROFILE,
        'wdl_order': 'WDL', 'wdl_pov': 'side_to_move', 'wdl_kind': 'probabilities',
        'mutated_arrays': ['search_wdl'], 'unchanged_arrays': sorted(ARRAYS - {'search_wdl'}),
        'source_dir': str(source), 'sf_source_dir': str(sf_root),
        'source_summary_sha256': m['source_summary_sha256'],
        'source_policy_summary_sha256': m['source_policy_summary_sha256'], 'sf_summary_sha256': m['sf_summary_sha256'],
        'manifest_sha256': args.expected_manifest_sha256, 'producer_sha256': pins, 'teachers': m['teachers'],
        'value_scheme': VALUE_SCHEME, 'value_source': value_source(), 'rows': sum(s['rows'] for s in specs),
        'shards': len(specs), 'changed_rows': changed_total, 'max_stored_mass_error': max_error,
        'max_stored_l1_error': max_l1, 'outputs': outputs}
    shutil.copyfile(source / POLICY_SUMMARY, writing / POLICY_SUMMARY)
    require(wdl.file_sha256(writing / POLICY_SUMMARY) == m['source_policy_summary_sha256'], 'policy summary copy differs')
    derived = {**base, 'value_scheme': {'name': VALUE_SCHEME, 'source': value_source()},
               'value_target_postprocess': {k: v for k, v in result.items() if k != 'outputs'}}
    for name, value in ((SUMMARY, result), (DERIVE_SUMMARY, derived)):
        (writing / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    for proof in outputs:
        guard()
        require(wdl.storage_identity(writing / proof['path']) == proof['output_storage_identity'],
                'completed output changed before publication')
    guard()
    require(not os.path.lexists(out), 'output appeared before publication')
    writing.rename(out)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--minimum-free-gib', type=float, default=150)
    parser.add_argument('--max-seconds', type=float, default=10800)
    parser.add_argument('--stop')
    parser.add_argument('--execute', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.execute:
        result = rewrite(args)
        print(json.dumps({k: v for k, v in result.items() if k != 'outputs'}, sort_keys=True))
    else:
        _, _, _, specs = read_manifest(Path(args.manifest), args.expected_manifest_sha256)
        print(json.dumps({'status': 'MANIFEST_VALIDATED_NOT_EXECUTED', 'shards': len(specs)}))


if __name__ == '__main__':
    main()
