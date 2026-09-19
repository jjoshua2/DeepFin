#!/usr/bin/env python3
"""Build B/C/D immutable teacher-target overlays from qualified V50 and Ceres.

A is the unmodified base. This module performs no teacher inference. Each cohort
manifest pins the base summary, base seal and exact Ceres entries. Full Ceres input
alignment is checked once per shard before writing all three arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from scripts import ceres_target_mix as policy
from scripts import ceres_value_mix as value

ARMS = {'B': ('policy_target',), 'C': ('search_wdl',),
        'D': ('policy_target', 'search_wdl')}
RECIPE = {
    'kind': 'bootstrap-factorial58-v1',
    'policy': 'half-normalized-stored-bt4-t05-plus-half-ceres-t05',
    'value': 'two-thirds-normalized-stored-v50-plus-one-third-ceres-dual',
    'ceres_value': value.CERES_VALUE_PROFILE,
    'rounding': 'inherits-base-float16-before-normalization',
}


def normalized(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    policy.require(a.ndim == 2 and a.shape[1] == 3 and bool(np.isfinite(a).all())
                   and bool((a >= 0).all()), 'invalid base WDL')
    mass = a.sum(axis=1, keepdims=True)
    policy.require(bool((mass > 0).all()), 'zero base WDL mass')
    return a / mass


def mixed_targets(bt4_t05: np.ndarray, v50: np.ndarray, legal: np.ndarray,
                  ceres_logits: np.ndarray, primary: np.ndarray,
                  secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Do not sharpen already-T0.5 BT4 again. Ceres dual calibration is fixed."""
    p = policy.policy_target(bt4_t05, ceres_logits, legal,
                             bt4_weight=.5, bt4_temperature=1., ceres_temperature=.5)
    c = .6 * value.softmax(primary, .55) + .4 * value.softmax(secondary, 1.5)
    w = (2. / 3.) * normalized(v50) + (1. / 3.) * c
    return p.astype('float16'), w.astype('float16')


def pinned(ref: dict[str, str]) -> dict[str, Any]:
    policy.require(policy.shared.file_sha256(ref['path']) == ref['sha256'],
                   'manifest input pin differs: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def validate_base(summary: dict[str, Any]) -> None:
    p, v = summary['policy_target_postprocess'], summary['value_target_postprocess']
    policy.require(p.get('kind') == 'global' and p.get('alpha') == 1
                   and p.get('bt4_temperature') == .5, 'base must be BT4 T0.5')
    policy.require(v.get('status') == 'COMPLETE' and v.get('kind') == 'bt4_value_rewrite'
                   and v.get('bt4_weight') == .5 and v.get('sf_weight') == .5
                   and v.get('mutated_arrays') == ['search_wdl'], 'base must be V50')


def build_cohort(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    from chess_anti_engine.replay import target_overlay as old
    from chess_anti_engine.replay import target_overlay_v2 as overlay

    base = Path(manifest['base']).resolve(strict=True)
    summary = pinned(manifest['base_summary'])
    policy.require(Path(manifest['base_summary']['path']).resolve()
                   == base / 'derive_targets_summary.json', 'foreign base summary')
    validate_base(summary)
    context = old.BaseSeal(manifest['base_seal'])
    seal = old.require_base_corpus(manifest['base_seal'], base, context=context)
    entries = manifest['entries']
    policy.require([e['shard'] for e in entries] == [s['name'] for s in seal['shards']],
                   'teacher roster differs from sealed base')
    policy.require(not out.exists(), 'refuse existing cohort output')
    out.mkdir(parents=True)
    for arm in ARMS:
        (out / arm).mkdir()
    rows = 0
    for entry in entries:
        src = base / entry['shard']
        group: Any = zarr.open_group(str(src), mode='r')
        cpath = Path(entry['ceres']).resolve(strict=True)
        binding = entry['ceres_binding']
        policy.require(binding['model_sha256'] == policy.ceres.MODEL_SHA,
                       'Ceres model differs')
        ceres_state = policy.shared.storage_identity(cpath)
        attrs = policy.ceres.verify_cached(cpath, binding)
        bank: Any = zarr.open_group(str(cpath), mode='r')
        policy.require('value2_logits' in bank, 'missing secondary Ceres values')
        n = int(group['x'].shape[0])
        # This compares actual x/history/row IDs/legal arrays to the teacher's
        # source and input digests; directory names alone do not prove alignment.
        policy.check_ceres_alignment(group, bank, attrs, n)
        destinations = {}
        for arm, fields in ARMS.items():
            dest = out / arm / src.name
            overlay.begin_target_shard(src, dest, manifest['base_seal'],
                                       replacements=fields, seal=context)
            destinations[arm] = zarr.open_group(str(dest), mode='a')
        offsets = np.asarray(bank['legal_offsets'][:])
        for start in range(0, n, 512):
            end = min(start + 512, n)
            legal = np.asarray(group['legal_mask'][start:end])
            logits = np.zeros(legal.shape, dtype=np.float64)
            lo, hi = int(offsets[start]), int(offsets[end])
            logits[np.nonzero(legal)] = np.asarray(bank['policy_logits'][lo:hi])
            p, w = mixed_targets(np.asarray(group['policy_target'][start:end]),
                                 np.asarray(group['search_wdl'][start:end]), legal,
                                 logits, np.asarray(bank['value_logits'][start:end]),
                                 np.asarray(bank['value2_logits'][start:end]))
            for arm, fields in ARMS.items():
                for name in fields:
                    destinations[arm][name][start:end] = p if name == 'policy_target' else w
        policy.require(policy.shared.storage_identity(cpath) == ceres_state,
                       'Ceres bank changed while building targets')
        for arm in ARMS:
            overlay.finish_target_shard(src, out / arm / src.name,
                manifest['base_seal'], recipe={**RECIPE, 'arm': arm,
                    'base_summary': manifest['base_summary'], 'ceres_binding': binding},
                seal=context)
        rows += n
    pinned(manifest['base_summary'])
    old.require_base_corpus(manifest['base_seal'], base, context=context)
    result = {'status': 'COMPLETE_FACTORIAL_TARGET_COHORT', 'rows': rows,
              'shards': len(entries), 'base': str(base),
              'roots': {arm: str(out / arm) for arm in ARMS}, 'recipe': RECIPE}
    old._atomic_new_json(out / 'complete.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build_cohort(pinned({'path': str(a.manifest), 'sha256': a.sha256}), a.out)


if __name__ == '__main__':
    main()
