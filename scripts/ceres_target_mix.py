#!/usr/bin/env python3
"""Offline, full-corpus BT4/Ceres probability mixing; never performs inference.

A SHA-pinned manifest names original SF source and one completed Ceres and raw
BT4 shard per source shard. Output becomes admissible only after atomic rename.
Only policy_target changes; ordinary copies preserve the historical consumer.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
from scripts import bt4_policy_mix as bt4
from scripts import bt4_value_rewrite as copies
from scripts import ceres_derived_sidecar as ceres
from scripts.sf_policy_rewrite import ARRAYS, require

shared = ceres.shared
SUMMARY = 'ceres_target_mix_summary.json'
DERIVE_SUMMARY = 'derive_targets_summary.json'
ALGORITHM = 'separately-normalized-probability-mixture-float16-v1'
LEGACY_BT4_SUMMARY_SHA = '68b32a41e89c03737aa28c89310d9ac744f6b1e5afcbfba198d2a0155bd646b3'


def producer_pins() -> dict[str, str]:
    modules = (bt4, copies, ceres, shared, ceres.tpg, ceres.mapping, ceres.selected,
               shared.raw, shared.lc0, copies.sf_rewrite, copies.reused, bt4.sf_ranks)
    paths = {Path(__file__).resolve()}
    for module in modules:
        filename = module.__file__
        if filename is None:
            raise ValueError('producer module source absent')
        paths.add(Path(filename).resolve())
    for function in (bt4.shard_encoding, bt4.position_fingerprints, shared.storage_identity,
                     shared.shard_contract):
        filename = inspect.getsourcefile(function)
        if filename is None:
            raise ValueError('producer helper source absent')
        paths.add(Path(str(filename)).resolve())
    return {str(p): shared.file_sha256(p) for p in sorted(paths)}


def policy_target(raw_bt4: np.ndarray, ceres_logits: np.ndarray, legal: np.ndarray,
                  *, bt4_weight: float, bt4_temperature: float, ceres_temperature: float
                  ) -> np.ndarray:
    require(type(bt4_weight) in (int, float) and math.isfinite(bt4_weight)
            and 0 <= bt4_weight <= 1, 'invalid BT4 weight')
    temperature = bt4.validate_bt4_temperature(ceres_temperature)
    legal = np.asarray(legal)
    require(bool(legal.ndim == 2 and np.all((legal == 0) | (legal == 1))), 'invalid legal mask')
    require(bool(ceres_logits.shape == legal.shape and np.isfinite(ceres_logits).all()),
            'invalid Ceres logits')
    b = bt4._tempered_bt4_policy(raw_bt4, legal, temperature=bt4_temperature)
    logits = np.where(legal != 0, ceres_logits.astype(np.float64), -np.inf)
    logits = (logits - logits.max(axis=1, keepdims=True)) / temperature
    c = bt4._normalized_legal(np.exp(logits), legal, name='Ceres policy')
    return bt4_weight * b + (1 - bt4_weight) * c


def check_ceres_alignment(group: Any, bank: Any, attrs: dict[str, Any], rows: int) -> None:
    """Compare actual source arrays, not just collector claims about them."""
    hashes = {}
    for name in ceres.COLUMNS:
        shared.complete_chunks(group[name])
        hashes[name] = shared.raw.sha_array(np.asarray(group[name][:]))
    require(hashes == attrs['source_array_sha256'], 'Ceres source array digest differs')
    require(bool(np.array_equal(bank['row_index'][:], np.arange(rows))), 'Ceres row order differs')
    for name in ('game_id', 'ply_index'):
        require(bool(np.all(group['has_' + name][:] == 1)
                and np.array_equal(bank[name][:], group[name][:])), 'Ceres row identity differs')
    legal = np.asarray(group['legal_mask'][:])
    require(bool(legal.shape == (rows, 1858) and np.all((legal == 0) | (legal == 1))
            and np.all(group['has_legal_mask'][:] == 1)), 'source legal mask differs')
    counts = np.count_nonzero(legal, axis=1)
    require(bool(np.all(counts > 0)), 'empty legal roster')
    require(bool(np.array_equal(bank['legal_offsets'][:], np.r_[0, np.cumsum(counts)])
            and np.array_equal(bank['legal_indices'][:], np.nonzero(legal)[1])),
            'Ceres legal roster differs')
    for start in range(0, rows, 128):
        feed = ceres.tpg.stored_x_to_ceres_tpg_bytes(
            group['x'][start:start + 128], input_history_encoding=shared.HISTORY,
            history_rep_fix=True)
        require(bool(np.array_equal(shared.row_digests(feed),
                               bank['tpg_feed_sha256'][start:start + 128])),
                'Ceres feed identity differs')


def verify_bt4_lineage(manifest: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    lineage = manifest['bt4_lineage']
    mode, pin = lineage['mode'], lineage['summary']
    require(mode in ('legacy-root-position-v1', 'stored-x-v1'), 'unsupported BT4 lineage mode')
    path = Path(pin['path'])
    require(path.is_absolute() and path.name == bt4.SIDECAR_SUMMARY
            and shared.file_sha256(path) == pin['sha256'], 'BT4 summary pin differs')
    if mode == 'legacy-root-position-v1':
        require(pin['sha256'] == LEGACY_BT4_SUMMARY_SHA, 'unregistered legacy BT4 collection')
    summary = json.loads(path.read_text())
    teacher = manifest['teachers']['bt4']
    expected = {'schema': 1, 'kind': 'bt4_raw_legal_policy_sidecar',
        'source_dir': str(Path(manifest['source']).resolve()), 'source_shards': len(specs),
        'sidecar_shards': len(specs), 'rows': sum(s['rows'] for s in specs),
        'policy_encoding': 'lc0_1858', 'policy_output': teacher['policy_output'],
        'providers': teacher['providers'], 'teacher_evaluations_per_position': 1,
        'search_nodes': 0, 'stored_dtype': 'float32'}
    require(all(summary.get(k) == v for k, v in expected.items())
            and summary.get('onnx', {}).get('sha256') == teacher['model_sha256'],
            'BT4 collection summary source/teacher/counts differ')
    remap = bt4.functional_remap_identity(summary.get('remap'))
    root = path.parent.resolve()
    require([p.name for p in sorted(root.glob('shard_*.zarr'))] == [s['path'] for s in specs]
            and all(Path(e['bt4']).resolve() == root / e['shard'] for e in manifest['entries']),
            'BT4 collection directory membership differs')
    return {'mode': mode, 'summary': pin, 'remap': remap,
            'root_position_and_history_regime_verified': True,
            'full_historical_input_provenance': ('inherited_from_pinned_collection'
                if mode == 'legacy-root-position-v1' else 'stored_x_digest_verified'),
            'full_input_digest_verified_shards': 0}


def verify_bt4_full_input(group: Any, attrs: dict[str, Any], manifest: dict[str, Any]) -> bool:
    names = ('source_stored_x_sha256', 'source_derive_summary_sha256')
    present = [name in attrs for name in names]
    if any(present) or manifest['bt4_lineage']['mode'] == 'stored-x-v1':
        require(all(present), 'BT4 full input proof is incomplete')
        require(bool(attrs[names[0]] == shared.raw.sha_array(np.asarray(group['x'][:]))
                and attrs[names[1]] == manifest['source_summary_sha256']),
                'BT4 full input digest differs')
        return True
    return False


def read_manifest(path: Path, digest: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    require(shared.file_sha256(path) == digest, 'manifest pin differs')
    manifest = json.loads(path.read_text())
    require(manifest.get('schema') == 1, 'unsupported manifest schema')
    source = Path(manifest['source']).resolve()
    args = argparse.Namespace(source=str(source),
        expected_source_summary_sha256=manifest['source_summary_sha256'],
        start_shard=0, max_shards=sys.maxsize)
    summary, specs = shared.source_inventory(args)
    entries = manifest['entries']
    require(isinstance(entries, list) and len(entries) == len(specs)
            and [e['shard'] for e in entries] == [s['path'] for s in specs],
            'manifest must cover every source shard exactly once in order')
    for field in ('bt4', 'ceres'):
        paths = [str(Path(e[field]).resolve()) for e in entries]
        require(len(set(paths)) == len(paths), 'duplicate teacher shard')
    require(manifest['teachers']['ceres']['model_sha256'] == ceres.MODEL_SHA,
            'unqualified Ceres model')
    for entry, spec in zip(entries, specs, strict=True):
        binding = entry['ceres_binding']
        require('selected_rows' not in binding and binding['profile'] != ceres.selected.PROFILE,
                'selected fragments are not full coverage')
        require(binding['source'] == str(source) and binding['shard'] == spec['path']
                and binding['rows'] == spec['rows']
                and binding['summary_sha256'] == manifest['source_summary_sha256']
                and binding['model_sha256'] == manifest['teachers']['ceres']['model_sha256']
                and binding['profile'] == manifest['teachers']['ceres']['profile']
                and binding['backend'] == manifest['teachers']['ceres']['backend'],
                'Ceres source/teacher binding differs')
        ceres.binding_options(binding)
    verify_bt4_lineage(manifest, specs)
    return manifest, summary, specs


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    shared.set_nthreads(2)
    require(type(args.batch_size) is int and 0 < args.batch_size <= 4096, 'invalid batch size')
    require(math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0,
            'invalid reserve')
    require(math.isfinite(args.max_seconds) and args.max_seconds > 0, 'invalid deadline')
    weight = copies.checked_alpha(args.bt4_weight)
    temperatures = {k: bt4.validate_bt4_temperature(getattr(args, k + '_temperature'))
                    for k in ('bt4', 'ceres')}
    manifest_path = Path(args.manifest).resolve()
    manifest, original, specs = read_manifest(manifest_path, args.expected_manifest_sha256)
    source = Path(manifest['source']).resolve()
    out = Path(args.out).absolute()
    writing = out.with_name(out.name + '.writing')
    require(not os.path.lexists(out) and not os.path.lexists(writing), 'output or partial exists')
    roots = [source, manifest_path, *(Path(e[k]).resolve() for e in manifest['entries']
                                     for k in ('bt4', 'ceres'))]
    require(all(out != p and p not in out.parents and out not in p.parents
                and writing != p and p not in writing.parents and writing not in p.parents
                for p in roots), 'output overlaps inputs')
    require(out.parent.is_dir() and out.parent.resolve() == out.parent,
            'output parent must exist without symlink aliases')
    lineage = verify_bt4_lineage(manifest, specs)
    pins = producer_pins()
    started = time.monotonic()

    def guard() -> None:
        require(time.monotonic() - started < args.max_seconds, 'rewrite deadline')
        require(shutil.disk_usage(out.parent).free >= args.minimum_free_gib * 2**30,
                'disk reserve')
        require(not args.stop or not Path(args.stop).exists(), 'STOP requested')

    guard()
    writing.mkdir()
    outputs = []
    verified_states: dict[Path, str] = {}
    max_mass_error = max_tv = 0.0
    support_lost = 0
    for spec, entry in zip(specs, manifest['entries'], strict=True):
        guard()
        src = source / spec['path']
        bpath, cpath = Path(entry['bt4']).resolve(), Path(entry['ceres']).resolve()
        states = {p: shared.storage_identity(p) for p in (src, bpath, cpath)}
        require(states[src] == entry['ceres_binding']['source_storage_identity'],
                'Ceres source storage identity differs')
        require(states[bpath] == entry['bt4_storage_identity'], 'BT4 storage identity differs')
        group = shared.source_arrays(src, original, spec['rows'])
        require(set(group.array_keys()) == set(ARRAYS), 'source array inventory differs')
        for name in ARRAYS:
            shared.complete_chunks(group[name])
        require(bool(group['policy_target'].shape == (spec['rows'], 1858)
                and group['policy_target'].dtype == np.dtype('float16')
                and np.all(group['has_policy'][:] == 1)), 'source policy contract differs')
        encoding, keys, key_sha, policy_sha = bt4._sidecar_identity(group, src)
        teacher = manifest['teachers']['bt4']
        battrs = bt4._validate_sidecar(bpath, source_path=src, source_keys=keys,
            source_key_sha=key_sha, source_policy_sha=policy_sha,
            onnx_sha=teacher['model_sha256'], providers=teacher['providers'],
            policy_output=teacher['policy_output'])
        require(battrs.get('input_history_encoding') == encoding
                and battrs.get('source_dir') == str(source), 'BT4 source/history differs')
        lineage['full_input_digest_verified_shards'] += int(verify_bt4_full_input(group, battrs, manifest))
        bg: Any = zarr.open_group(str(bpath), mode='r')
        for name in (bt4.SIDECAR_KEY_FIELD, bt4.SIDECAR_POLICY_FIELD):
            shared.complete_chunks(bg[name])
        if 'bt4_policy_sha256' in battrs:
            require(bool(shared.raw.sha_array(np.asarray(bg[bt4.SIDECAR_POLICY_FIELD][:]))
                    == battrs['bt4_policy_sha256']), 'BT4 policy digest differs')
        cattrs = ceres.verify_cached(cpath, entry['ceres_binding'])
        cg: Any = zarr.open_group(str(cpath), mode='r')
        check_ceres_alignment(group, cg, cattrs, spec['rows'])
        before = copies.file_map(src)
        dest_path = writing / spec['path']
        shutil.copytree(src, dest_path)
        require(copies.file_map(dest_path) == before, 'copied bytes differ')
        dest: Any = zarr.open_group(str(dest_path), mode='a')
        policy_hash = hashlib.sha256()
        changed = 0
        for start in range(0, spec['rows'], args.batch_size):
            guard()
            end = min(start + args.batch_size, spec['rows'])
            legal = group['legal_mask'][start:end]
            logits = np.zeros(legal.shape, dtype=np.float64)
            offsets = cg['legal_offsets'][start:end + 1]
            indices = cg['legal_indices'][int(offsets[0]):int(offsets[-1])]
            raw_logits = cg['policy_logits'][int(offsets[0]):int(offsets[-1])]
            for row in range(end - start):
                lo, hi = int(offsets[row] - offsets[0]), int(offsets[row + 1] - offsets[0])
                logits[row, indices[lo:hi]] = raw_logits[lo:hi]
            ideal = policy_target(bg[bt4.SIDECAR_POLICY_FIELD][start:end], logits, legal,
                bt4_weight=weight, bt4_temperature=temperatures['bt4'],
                ceres_temperature=temperatures['ceres'])
            stored = ideal.astype(np.float16)
            mass = stored.astype(np.float64).sum(axis=1, keepdims=True)
            require(bool(np.all(mass > 0)), 'stored policy lost all mass')
            error = float(np.max(np.abs(mass - 1)))
            require(error <= 2**-10, 'stored policy mass differs')
            max_mass_error = max(max_mass_error, error)
            max_tv = max(max_tv, float(np.max(np.abs(ideal - stored / mass).sum(axis=1) / 2)))
            support_lost += int(np.count_nonzero((ideal > 0) & (stored == 0)))
            changed += int(np.count_nonzero(np.any(stored != group['policy_target'][start:end], axis=1)))
            dest['policy_target'][start:end] = stored
            require(bool(np.array_equal(dest['policy_target'][start:end], stored)), 'policy readback differs')
            policy_hash.update(stored.tobytes(order='C'))
        stamp = {'schema': 1, 'kind': 'bt4-ceres-policy', 'algorithm': ALGORITHM,
                 'weights': {'bt4': weight, 'ceres': 1 - weight}, 'temperatures': temperatures,
                 'manifest_sha256': args.expected_manifest_sha256,
                 'source_storage_identity': states[src]}
        dest.attrs['ceres_policy_postprocess'] = stamp
        after = copies.file_map(dest_path)
        def unchanged(files: dict[str, str]) -> dict[str, str]:
            return {k: v for k, v in files.items()
                    if k != '.zattrs' and k.split('/')[0] != 'policy_target'}
        require(unchanged(before) == unchanged(after), 'nonpolicy arrays changed')
        require(all(shared.storage_identity(p) == state for p, state in states.items()),
                'input changed during rewrite')
        verified_states.update(states)
        outputs.append({'path': spec['path'], 'rows': spec['rows'], 'changed_rows': changed,
            'source_storage_identity': states[src], 'bt4_storage_identity': states[bpath],
            'ceres_storage_identity': states[cpath], 'policy_target_sha256': policy_hash.hexdigest(),
            'files_manifest_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
            'attrs_sha256': shared.file_sha256(dest_path / '.zattrs'),
            'output_storage_identity': shared.storage_identity(dest_path)})
    guard()
    require(shared.file_sha256(manifest_path) == args.expected_manifest_sha256
            and shared.file_sha256(source / DERIVE_SUMMARY) == manifest['source_summary_sha256']
            and producer_pins() == pins, 'final input or producer pin differs')
    verify_bt4_lineage(manifest, specs)
    for path, state in verified_states.items():
        guard()
        require(shared.storage_identity(path) == state, 'previous input changed before completion')
    result = {'schema': 1, 'status': 'COMPLETE', 'complete': True, 'kind': 'bt4-ceres-policy',
        'algorithm': ALGORITHM, 'weights': {'bt4': weight, 'ceres': 1 - weight},
        'temperatures': temperatures, 'mutated_arrays': ['policy_target'],
        'unchanged_arrays': sorted(ARRAYS - {'policy_target'}), 'source_dir': str(source),
        'source_summary_sha256': manifest['source_summary_sha256'],
        'manifest_sha256': args.expected_manifest_sha256, 'producer_sha256': pins,
        'bt4_lineage': lineage, 'teachers': manifest['teachers'], 'rows': sum(s['rows'] for s in specs),
        'shards': len(specs), 'max_stored_mass_error': max_mass_error,
        'max_stored_total_variation': max_tv, 'support_lost_move_entries': support_lost,
        'outputs': outputs}
    postprocess = {k: v for k, v in result.items() if k != 'outputs'}
    for name, value in ((SUMMARY, result), (DERIVE_SUMMARY,
            {**original, 'policy_target_postprocess': postprocess})):
        (writing / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    for proof in outputs:
        guard()
        require(shared.storage_identity(writing / proof['path']) == proof['output_storage_identity'],
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
    parser.add_argument('--bt4-weight', type=float, default=0.5)
    parser.add_argument('--bt4-temperature', type=float, default=0.5)
    parser.add_argument('--ceres-temperature', type=float, default=0.5)
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
        _, _, specs = read_manifest(Path(args.manifest), args.expected_manifest_sha256)
        print(json.dumps({'status': 'MANIFEST_VALIDATED_NOT_EXECUTED', 'shards': len(specs)}))


if __name__ == '__main__':
    main()
