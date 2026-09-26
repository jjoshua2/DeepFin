#!/usr/bin/env python3
"""Build the SF-free E overlay from original authenticated BT4/Ceres banks.

The policy is identical to factorial D. Only the main prepared WDL differs:
half native BT4 plus half calibrated Ceres, without reconstructing BT4 from V50.
A pinned cohort manifest extends the original factorial manifest with a native_bt4
path, native_bt4_binding and native_bt4_attrs_sha256 for every exact base shard.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import zarr

from scripts import bootstrap_factorial_targets as factorial
from scripts import bt4_derived_wdl_sidecar as native
from scripts import ceres_value_mix as value

require = factorial.policy.require
SCHEMA = 'factorial58-sffree-v1'
RECIPE = {
    'kind': SCHEMA, 'arm': 'E',
    'policy': factorial.RECIPE['policy'],
    'value': 'half-normalized-native-bt4-plus-half-ceres-dual',
    'weights': {'sf': 0., 'bt4': .5, 'ceres': .5},
    'ceres_value': value.CERES_VALUE_PROFILE,
    'bt4_model_sha256': value.BT4_MODEL,
}


def mixed_targets(bt4_policy: np.ndarray, bt4_wdl: np.ndarray, legal: np.ndarray,
                  ceres_logits: np.ndarray, primary: np.ndarray,
                  secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    policy = factorial.policy.policy_target(bt4_policy, ceres_logits, legal,
        bt4_weight=.5, bt4_temperature=1., ceres_temperature=.5)
    wdl = .5 * value.copies.normalized(bt4_wdl) + .5 * (
        .6 * value.softmax(primary, .55) + .4 * value.softmax(secondary, 1.5))
    require(wdl.shape == (len(policy), 3), 'teacher row counts differ')
    return policy.astype('float16'), wdl.astype('float16')


def verify_native(entry: dict[str, Any], source: Any, rows: int) -> tuple[Any, dict[str, Any]]:
    path = Path(entry['native_bt4']).resolve(strict=True)
    binding = entry['native_bt4_binding']
    require(binding.get('onnx_sha256') == value.BT4_MODEL
            and binding.get('requested_wdl') == {'kind': 'probabilities', 'output': '/output/wdl'}
            and binding.get('rows') == rows, 'native BT4 teacher contract differs')
    require(native.file_sha256(path / '.zattrs') == entry['native_bt4_attrs_sha256'],
            'native BT4 provenance changed')
    attrs = native.verify_cached(path, binding, 512)
    group: Any = zarr.open_group(str(path), mode='r')
    hashes = {name: hashlib.sha256() for name in native.COLUMNS}
    for start in range(0, rows, 512):
        end = min(rows, start + 512)
        batch = {name: np.asarray(source[name][start:end]) for name in native.COLUMNS}
        for name, digest in hashes.items():
            digest.update(batch[name].tobytes(order='C'))
        require(bool((batch['has_game_id'] == 1).all() and (batch['has_ply_index'] == 1).all()),
                'missing base row identity')
        require(np.array_equal(group['game_id'][start:end], batch['game_id'])
                and np.array_equal(group['ply_index'][start:end], batch['ply_index']),
                'native BT4 row identity differs')
        feed = native.stored_feed(batch['x']).astype(attrs['input']['dtype'])
        require(np.array_equal(native.row_digests(feed), group['lc0_feed_sha256'][start:end]),
                'native BT4 feed identity differs')
    require({name: h.hexdigest() for name, h in hashes.items()} == attrs['source_array_sha256'],
            'native BT4 source content differs')
    return group, attrs


def build_cohort(manifest: dict[str, Any], out: Path, *, minimum_free_gib: float = 150.) -> dict[str, Any]:
    from chess_anti_engine.replay import target_overlay as old
    from chess_anti_engine.replay import target_overlay_v2 as overlay

    require(manifest.get('schema') == SCHEMA, 'explicit SF-free manifest required')
    require(np.isfinite(minimum_free_gib) and minimum_free_gib >= 0, 'invalid disk reserve')
    base = Path(manifest['base']).resolve(strict=True)
    summary = factorial.pinned(manifest['base_summary'])
    require(Path(manifest['base_summary']['path']).resolve() == base / 'derive_targets_summary.json',
            'foreign base summary')
    factorial.validate_base(summary)
    context = old.BaseSeal(manifest['base_seal'])
    seal = old.require_base_corpus(manifest['base_seal'], base, context=context)
    entries = manifest['entries']
    require([e['shard'] for e in entries] == [s['name'] for s in seal['shards']],
            'teacher roster differs from sealed base')
    require(bool(entries), 'empty cohort')
    for key in ('native_bt4', 'ceres'):
        paths = [Path(e[key]).resolve(strict=True) for e in entries]
        require(len(set(paths)) == len(paths), 'duplicate teacher shard')
        require(all(not p.name.endswith('.writing') for p in paths), 'partial teacher shard')
    out = out.resolve()
    inputs = [base, *(Path(e[key]).resolve() for e in entries for key in ('native_bt4', 'ceres'))]
    require(all(out != p and out not in p.parents and p not in out.parents for p in inputs),
            'output overlaps inputs')
    require(not os.path.lexists(out), 'fresh output required')
    out.parent.mkdir(parents=True, exist_ok=True)

    def guard() -> None:
        require(not (out / 'STOP').exists() and not (out.parent / 'STOP').exists(), 'STOP requested')
        require(shutil.disk_usage(out.parent).free >= minimum_free_gib * 2**30, 'disk reserve')

    guard()
    out.mkdir()
    (out / 'E').mkdir()
    rows = 0
    for entry in entries:
        guard()
        src = base / entry['shard']
        source: Any = zarr.open_group(str(src), mode='r')
        n = int(source['x'].shape[0])
        for start in range(0, n, 512):
            end = min(n, start + 512)
            require(bool((np.asarray(source['has_policy'][start:end]) == 1).all()
                         and (np.asarray(source['has_search_wdl'][start:end]) == 1).all()),
                    'full policy and value supervision required')
        bpath, cpath = (Path(entry[key]).resolve() for key in ('native_bt4', 'ceres'))
        states = {p: native.storage_identity(p) for p in (bpath, cpath)}
        bt4, _ = verify_native(entry, source, n)
        binding = entry['ceres_binding']
        require(binding['model_sha256'] == factorial.policy.ceres.MODEL_SHA, 'Ceres model differs')
        require(native.file_sha256(cpath / '.zattrs') == entry['ceres_attrs_sha256'],
                'Ceres provenance changed')
        attrs = factorial.policy.ceres.verify_cached(cpath, binding)
        ceres: Any = zarr.open_group(str(cpath), mode='r')
        require('value2_logits' in ceres, 'missing secondary Ceres values')
        factorial.policy.check_ceres_alignment(source, ceres, attrs, n)
        dest = out / 'E' / src.name
        overlay.begin_target_shard(src, dest, manifest['base_seal'],
                                   replacements=('policy_target', 'search_wdl'), seal=context)
        target: Any = zarr.open_group(str(dest), mode='a')
        offsets = np.asarray(ceres['legal_offsets'][:])
        for start in range(0, n, 512):
            guard()
            end = min(n, start + 512)
            legal = np.asarray(source['legal_mask'][start:end])
            logits = np.zeros(legal.shape, dtype=np.float64)
            lo, hi = int(offsets[start]), int(offsets[end])
            logits[np.nonzero(legal)] = np.asarray(ceres['policy_logits'][lo:hi])
            p, w = mixed_targets(np.asarray(source['policy_target'][start:end]),
                np.asarray(bt4['bt4_wdl_raw'][start:end]), legal, logits,
                np.asarray(ceres['value_logits'][start:end]), np.asarray(ceres['value2_logits'][start:end]))
            target['policy_target'][start:end] = p
            target['search_wdl'][start:end] = w
        require(all(native.storage_identity(p) == state for p, state in states.items()),
                'teacher bank changed during build')
        overlay.finish_target_shard(src, dest, manifest['base_seal'], recipe={**RECIPE,
            'base_summary': manifest['base_summary'], 'ceres_binding': binding,
            'ceres_attrs_sha256': entry['ceres_attrs_sha256'],
            'native_bt4_binding': entry['native_bt4_binding'],
            'native_bt4_attrs_sha256': entry['native_bt4_attrs_sha256']}, seal=context)
        rows += n
    guard()
    factorial.pinned(manifest['base_summary'])
    old.require_base_corpus(manifest['base_seal'], base, context=context)
    result = {'status': 'COMPLETE_SFFREE_TARGET_COHORT', 'rows': rows,
              'shards': len(entries), 'base': str(base), 'root': str(out / 'E'), 'recipe': RECIPE}
    old._atomic_new_json(out / 'complete.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--minimum-free-gib', type=float, default=150.)
    args = parser.parse_args()
    manifest = factorial.pinned({'path': str(args.manifest), 'sha256': args.sha256})
    build_cohort(manifest, args.out, minimum_free_gib=args.minimum_free_gib)


if __name__ == '__main__':
    main()
