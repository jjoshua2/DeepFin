"""Explicit admission of a finalized, audited subset of a growing raw corpus.

This verifies saved audit/derivation/adapter receipts. It never turns the growing
source into a complete corpus and performs no teacher inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import baseline_row_exclusions as exclusions

PROFILE = 'audited-frozen-derived-v1'
require = exclusions.require


def same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def process(ref: dict[str, Any], source: Path, stage: str) -> dict[str, Any]:
    result = exclusions.read(ref)
    require(result.get('process_complete') is True and result.get('exit_code') == 0,
            stage + ' process not complete')
    require(result.get('gpu_seconds') == 0, stage + ' unexpectedly used GPU')
    command = result['command']
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), 'invalid process command')
    require(command.count('--out') == 1 and command[command.index('--out') + 1] == str(source),
            stage + ' output differs')
    return result


def admit(path: Path, digest: str, source: Path, summary: dict[str, Any]) -> dict[str, Any]:
    reference = {'path': str(path.resolve()), 'sha256': digest}
    manifest = exclusions.read(reference)
    require(manifest.get('schema') == 1 and manifest.get('profile') == PROFILE, 'unsupported audited source')
    expected_summary = {'path': str(source / 'derive_targets_summary.json'),
                        'sha256': exclusions.sha(source / 'derive_targets_summary.json')}
    require(manifest['derived_summary'] == expected_summary
            and same(exclusions.read(expected_summary), summary), 'derived source binding differs')
    proof = summary['baseline_exclusions']
    rule = exclusions.load(Path(proof['path']))
    require(same(rule.proof, proof), 'realized baseline exclusion proof differs')
    rule.bind(summary['source_selection'])
    rows = summary['realized']['rows_written']
    require(type(rows) is int and rows > 0 and rows == rule.proof['eligible_rows'], 'audited retained row count differs')
    for key, expected in {'rows_read': rule.proof['physical_rows'],
                          'rows_dropped_no_result': rule.proof['no_result_rows'],
                          'rows_dropped_baseline_audit': rule.proof['excluded_rows'],
                          'rows_dropped_envelope': 0, 'input_key_verified': rows}.items():
        require(summary['realized'].get(key) == expected, 'derived ' + key + ' differs')
    scheme = summary['scheme']
    require(scheme['canonical'] == 'uniform-d9' and scheme['policy_observation'] == 'phase0'
            and scheme['value_observation'] == 'latest-phase' and summary['value_scheme']['name'] == 'search'
            and summary['temp_requested'] == .0005 and summary['floor_requested'] == 0,
            'audited SF target semantics differ')
    selected = rule.selection
    process_result = process(manifest['derivation_process'], source, 'derive')
    command = process_result['command']
    for flag, expected in {'--corpus': selected['source_dir'], '--source-shards': summary['source_selection']['path'],
                           '--baseline-exclusions': proof['path']}.items():
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == expected,
                'derived command ' + flag + ' differs')
    adapter = exclusions.read(manifest['adapter_summary'])
    adapter_root = Path(manifest['adapter_summary']['path']).parent
    exclusions.read(adapter['adapter']['manifest'])
    adapter_command = process(manifest['adapter_process'], adapter_root, 'adapt')['command']
    for flag, expected in {'--manifest': adapter['adapter']['manifest']['path'],
                           '--expected-manifest-sha256': adapter['adapter']['manifest']['sha256']}.items():
        require(adapter_command.count(flag) == 1
                and adapter_command[adapter_command.index(flag) + 1] == expected, 'adapter manifest command differs')
    require(adapter['adapter']['derived_summary'] == expected_summary and adapter['rows'] == rows,
            'adapter belongs to another derived source')
    require(adapter['adapted_wdl']['new_teacher_evaluations'] == 0
            and adapter['adapted_wdl']['rows'] == rows, 'missing reused native WDL')
    specs = summary['shards']
    written = adapter['adapter']['written_shards']
    require([(s['path'], s['rows']) for s in specs] == [(s['path'], s['rows']) for s in written],
            'adapter/derived shard order differs')
    for spec, item in zip(specs, written, strict=True):
        require(spec['row_provenance']['sha256'] == item['row_provenance_sha256'], 'physical offset proof differs')
    audit_manifest = json.loads(Path(proof['path']).read_text())
    audit = exclusions.read(audit_manifest['audit'])
    relevant = [s for s in audit['shards'] if s['source_dir'] == selected['source_dir']]
    counts = {s['source_shard']: s['counts']['eligible_rows'] for s in relevant}
    require(sum(counts.values()) == rows and set(counts) == {s['source_shard'] for s in selected['shards']},
            'audited source shard coverage differs')
    namespace = hashlib.sha256(json.dumps([selected['source_dir'], selected['source_config_sha256']],
                                          separators=(',', ':')).encode()).hexdigest()
    return {'profile': PROFILE, 'qualification': reference, 'derived_summary': expected_summary,
            'rows': rows, 'adapter_manifest': adapter['adapter']['manifest'],
            'wdl_dir': str(adapter_root / 'wdl'), 'source_namespace': namespace, 'raw_source_dir': selected['source_dir'],
            'source_config_sha256': selected['source_config_sha256'],
            'raw_shards': [s['source_shard'] for s in selected['shards']],
            'per_raw_shard_survivors': counts, 'source_selection': summary['source_selection'],
            'summary_pins': {manifest[k]['path']: manifest[k]['sha256'] for k in
                             ('derived_summary', 'derivation_process', 'adapter_summary', 'adapter_process')},
            'baseline_exclusion_pins': rule.proof['pins']}
