"""Admit a complete matched adaptive-SF cohort without rebinding its teachers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import zarr

from scripts import bt4_derived_wdl_sidecar as wdl
from scripts import corpus_row_provenance as refs
from scripts.raw_wdl_adaptation import pin
from scripts.sf_policy_rewrite import ARRAYS, require

PROFILE = 'matched-g10-adaptive-sf-value-v1'
SELECTOR = 'g10-adaptive-final-or-d9-v1'
VALUE_SOURCE = 'g10-adaptive-final-root-candidates-with-latest-d9-fallback-v1'
SUMMARY = 'derive_targets_summary.json'


def admit(path: Path, digest: str, *, original: Path, original_sha: str,
          summary: dict[str, Any], specs: list[dict[str, Any]],
          pins: dict[Path, str]) -> dict[str, Any]:
    def checked(item: dict[str, str]) -> Path:
        result = pin(item)
        pins[result] = item['sha256']
        return result

    manifest = json.loads(checked({'path': str(path), 'sha256': digest}).read_text())
    require(manifest['schema'] == 1 and manifest['profile'] == PROFILE,
            'matched SF profile differs')
    require(manifest['original_source'] == str(original)
            and manifest['original_summary_sha256'] == original_sha,
            'matched SF original source differs')
    candidate = Path(manifest['candidate_source'])
    require(candidate.is_absolute() and candidate == candidate.resolve()
            and candidate != original and original not in candidate.parents
            and candidate not in original.parents and not candidate.name.endswith('.writing'),
            'matched SF candidate path differs')
    candidate_summary = checked(manifest['candidate_summary'])
    require(candidate_summary == candidate / SUMMARY, 'matched SF summary path differs')
    after = json.loads(candidate_summary.read_text())
    plan_path = checked(manifest['launch_plan'])
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(checked(manifest['completed_receipt']).read_text())
    require(receipt['status'] == 'COMPLETE_MATCHED_DERIVATION_NOT_TRAINING_ADMISSION'
            and receipt['returncode'] == 0
            and receipt['plan_sha256'] == manifest['launch_plan']['sha256']
            and receipt['derive_summary_sha256'] == manifest['candidate_summary']['sha256'],
            'matched SF completion differs')
    require(plan['baseline'] == str(original)
            and plan['pins'][str(original / SUMMARY)] == original_sha,
            'matched SF launch original differs')
    argv = plan['derive_argv']
    def option(name: str) -> str:
        require(argv.count(name) == 1, 'matched SF launch option missing or repeated')
        return argv[argv.index(name) + 1]
    require(option('--out') == str(candidate)
            and option('--sf-value-selector') == SELECTOR
            and option('--value-scheme') == 'search'
            and option('--scheme') == 'uniform-d9'
            and option('--policy-observation') == 'phase0'
            and option('--value-observation') == 'latest-phase'
            and int(option('--limit')) == summary['limit_requested'],
            'matched SF launch selector differs')
    require(receipt['argv'][-len(argv):] == argv, 'matched SF executed argv differs')
    runtime = Path(plan['runtime_cwd'])
    require(argv[1] == str(runtime / 'scripts/derive_corpus_targets.py'),
            'matched SF executed producer differs')
    for role in ('scripts/derive_corpus_targets.py', 'scripts/adaptive_sf_value.py'):
        producer = runtime / role
        checked({'path': str(producer), 'sha256': plan['pins'][str(producer)]})
    for key in ('source_selection', 'limit_requested', 'rows_per_shard', 'seed', 'cp_map',
                'temp_requested', 'floor_requested'):
        require(after[key] == summary[key], 'matched SF derivation differs: ' + key)
    for key in ('rows_read', 'rows_written', 'rows_dropped_no_result',
                'rows_dropped_policy_support', 'rows_dropped_envelope', 'policy_support_exclusions'):
        require(after['realized'].get(key, [] if key == 'policy_support_exclusions' else 0)
                == summary['realized'].get(key, [] if key == 'policy_support_exclusions' else 0),
                'matched SF omissions differ: ' + key)
    layout = [(s['path'], s['rows']) for s in specs]
    require([(s['path'], s['rows']) for s in after['shards']] == layout
            and receipt['rows'] == sum(s['rows'] for s in specs),
            'matched SF complete cohort required')
    checked_shards = receipt['checked_shards']
    require([(s['shard'], s['rows']) for s in checked_shards] == layout
            and set(manifest['shards']) == {s['path'] for s in specs},
            'matched SF receipt coverage differs')
    require(after['value_scheme']['name'] == 'search', 'matched SF value scheme differs')
    for record in checked_shards:
        require(frozenset(record['nonvalue_decoded_sha256']) == ARRAYS - {'search_wdl'},
                'matched SF nonvalue receipt coverage differs')
        attrs = checked(manifest['shards'][record['shard']]['attrs'])
        require(attrs == candidate / record['shard'] / '.zattrs', 'matched SF attrs path differs')
    return {'root': candidate, 'manifest': manifest, 'records': {r['shard']: r for r in checked_shards},
            'provenance': {'profile': PROFILE, 'selector': SELECTOR,
                           'manifest': {'path': str(path), 'sha256': digest},
                           'candidate_source': str(candidate),
                           'candidate_summary': manifest['candidate_summary'],
                           'completed_receipt': manifest['completed_receipt'],
                           'value_witness': 'Separately pinned current WDL hash; completion attests matching nonvalue arrays only'}}


def verify_shard(admitted: dict[str, Any], original: Path, spec: dict[str, Any],
                 original_group: Any, batch_size: int, guard: Callable[[], None]) -> Any:
    name, n = spec['path'], spec['rows']
    candidate = admitted['root'] / name
    record = admitted['records'][name]
    require(wdl.file_sha256(candidate / refs.FILENAME) == record['provenance_sha256'],
            'matched SF provenance hash differs')
    require(wdl.file_sha256(original / refs.FILENAME) == spec['row_provenance']['sha256'],
            'matched SF original provenance changed')
    require(refs.read(candidate / refs.FILENAME, rows=n) == refs.read(original / refs.FILENAME, rows=n),
            'matched SF source-qualified row order differs')
    group: Any = zarr.open_group(str(candidate), mode='r')
    require(frozenset(group.array_keys()) == ARRAYS, 'matched SF array coverage differs')
    attrs = dict(group.attrs)
    params = dict(attrs['derive_scheme_params'])
    require(attrs['derive_value_source'] == VALUE_SOURCE and attrs['derive_value_scheme'] == 'search'
            and params.pop('sf_value_selector', None) == SELECTOR
            and params.get('value_source') == VALUE_SOURCE, 'matched SF selector stamp differs')
    baseline_attrs = dict(original_group.attrs)
    params['value_source'] = baseline_attrs['derive_scheme_params']['value_source']
    attrs['derive_scheme_params'] = params
    attrs['derive_value_source'] = baseline_attrs['derive_value_source']
    require(attrs == baseline_attrs, 'matched SF nonvalue metadata differs')
    for column in ARRAYS:
        wdl.complete_chunks(group[column])
        require(group[column].shape == original_group[column].shape
                and group[column].dtype == original_group[column].dtype,
                'matched SF array layout differs')
        hashed = hashlib.sha256()
        for start in range(0, n, batch_size):
            guard()
            value = np.asarray(group[column][start:start + batch_size])
            if column == 'search_wdl':
                require(bool(np.isfinite(value).all() and (value >= 0).all())
                        and bool((np.abs(value.astype('float64').sum(axis=1) - 1) <= 2**-10).all()),
                        'matched SF invalid WDL')
            else:
                baseline = np.asarray(original_group[column][start:start + batch_size])
                require(value.tobytes() == baseline.tobytes(), 'matched SF nonvalue bytes differ: ' + column)
            hashed.update(value.tobytes())
        expected = (admitted['manifest']['shards'][name]['search_wdl_sha256']
                    if column == 'search_wdl' else record['nonvalue_decoded_sha256'][column])
        require(hashed.hexdigest() == expected, 'matched SF decoded hash differs: ' + column)
    return group
