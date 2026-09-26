"""Admit pinned historical direct-WDL receipts without rebinding their source.

The caller-reviewed manifest pins actual invocation receipts, shard attributes and
historical producer files separately: completion receipts do not attest producers.
Only complete, original G10 cohorts are supported; schema 2 routes explicitly
across multiple historical output directories without moving their contents.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from scripts import bt4_derived_wdl_sidecar as wdl
from scripts.sf_policy_rewrite import require

PROFILE = 'historical-g10-native-wdl-reuse-v1'
MULTI_PROFILE = 'historical-g10-native-wdl-multi-output-v1'
PRODUCERS = {'scripts/bt4_derived_wdl_sidecar.py', 'scripts/bt4_raw_corpus_sidecar.py',
             'chess_anti_engine/encoding/lc0.py'}


def pinned(ref: Any, pins: dict[Path, str]) -> Path:
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'native WDL pin schema')
    path = Path(ref['path'])
    digest = ref['sha256']
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'native WDL pin path')
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
            'native WDL pin digest')
    require(wdl.file_sha256(path) == digest, 'native WDL proof pin differs')
    require(path not in pins or pins[path] == digest, 'native WDL conflicting pin')
    pins[path] = digest
    return path


def read(ref: Any, pins: dict[Path, str]) -> tuple[Path, dict[str, Any]]:
    path = pinned(ref, pins)
    body = json.loads(path.read_text())
    require(isinstance(body, dict), 'native WDL receipt object required')
    return path, body


def admit(manifest: Path, expected_sha: str, *, source: Path, sidecar: Path,
          summary_sha: str, model_sha: str, head: str, admission: dict[str, Any],
          specs: list[dict[str, Any]], pins: dict[Path, str],
          shard_roots: dict[str, Path] | None = None) -> dict[str, dict[str, Any]]:
    """Return exact historical bindings; caller still verifies all cached arrays."""
    _, body = read({'path': str(manifest), 'sha256': expected_sha}, pins)
    multi = body.get('schema') == 2
    directory_key = 'wdl_dirs' if multi else 'wdl_dir'
    require(set(body) == {'schema', 'profile', 'source_dir', directory_key,
                         'source_summary_sha256', 'onnx_sha256', 'wdl_output', 'invocations'},
            'native WDL manifest schema')
    require(body['schema'] == (2 if multi else 1)
            and body['profile'] == (MULTI_PROFILE if multi else PROFILE)
            and body['source_dir'] == str(source)
            and body['source_summary_sha256'] == summary_sha
            and body['onnx_sha256'] == model_sha and body['wdl_output'] == head,
            'native WDL manifest source/model/head differs')
    directory_names = body['wdl_dirs'] if multi else [body['wdl_dir']]
    require(isinstance(directory_names, list) and all(isinstance(p, str) for p in directory_names)
            and len(directory_names) >= (2 if multi else 1), 'native WDL directories required')
    directories = [Path(p) for p in directory_names]
    require(directories[0] == sidecar, 'native WDL anchor directory differs')
    if multi:
        require(shard_roots is not None, 'native WDL multi-output routing required')
        require(all(p.is_absolute() and p.resolve() == p and p.is_dir() and not p.is_symlink()
                    for p in directories), 'native WDL directory path')
        require(len(set(directories)) == len(directories)
                and all(a not in b.parents for a in directories for b in directories if a != b),
                'native WDL overlapping directories')
    used_directories: set[Path] = set()
    entries = body['invocations']
    require(isinstance(entries, list) and bool(entries), 'native WDL invocations required')
    by_name = {s['path']: s for s in specs}
    bindings: dict[str, dict[str, Any]] = {}
    for entry in entries:
        expected_keys = {'completed', 'started', 'producer', 'g10_admission_script', 'attributes'}
        if multi:
            expected_keys.add('wdl_dir')
        require(isinstance(entry, dict) and set(entry) == expected_keys,
                'native WDL invocation schema')
        directory = Path(entry['wdl_dir']) if multi else sidecar
        require(directory in directories, 'native WDL undeclared invocation directory')
        used_directories.add(directory)
        namespace_path = directory / 'g10_common_source.json'
        namespace_sha = wdl.file_sha256(namespace_path)
        namespace = json.loads(namespace_path.read_text())
        require(namespace_path not in pins or pins[namespace_path] == namespace_sha,
                'native WDL conflicting namespace pin')
        pins[namespace_path] = namespace_sha
        completed_path, completed = read(entry['completed'], pins)
        started_path, started = read(entry['started'], pins)
        require(completed_path.name == 'completed.json'
                and completed_path.parent.parent == directory / 'invocations'
                and started_path == completed_path.parent / 'started.json'
                and not (completed_path.parent / 'failed.json').exists(),
                'native WDL invocation namespace or failure')
        require(completed.get('schema') == 1 and completed.get('complete') is True,
                'native WDL invocation incomplete')
        begin, end = started.get('started_unix'), completed.get('ended_unix')
        if not isinstance(begin, (float, int)) or not isinstance(end, (float, int)):
            raise ValueError('native WDL invocation timing')
        require(type(begin) in (float, int) and type(end) in (float, int)
                and math.isfinite(begin) and math.isfinite(end) and end >= begin,
                'native WDL invocation timing')
        argv = started['argv']
        required_args = {'source': str(source), 'out': str(directory),
                         'invocation': str(completed_path.parent),
                         'expected_source_summary_sha256': summary_sha,
                         'expected_onnx_sha256': model_sha, 'wdl_output': head,
                         'wdl_output_kind': 'probabilities',
                         'g10_common_qualification': admission['qualification']['path'],
                         'expected_g10_common_qualification_sha256': admission['qualification']['sha256']}
        require(all(argv.get(k) == v for k, v in required_args.items()),
                'native WDL invocation source/model/head differs')
        start, count = argv['start_shard'], argv['max_shards']
        require(type(start) is int and type(count) is int
                and 0 <= start < len(specs) and count > 0, 'native WDL selection range')
        selection = specs[start:start + count]
        require(completed.get('selection') == selection
                and completed.get('rows') == sum(s['rows'] for s in selection)
                and completed.get('shards') == len(selection)
                and completed.get('source_summary_sha256') == summary_sha
                and completed.get('history_lineage') == wdl.LINEAGE
                and completed.get('g10_common_admission') == admission,
                'native WDL completed selection or G10 lineage differs')
        producer_refs = entry['producer']
        require(isinstance(producer_refs, dict) and set(producer_refs) == PRODUCERS,
                'native WDL historical producer proof roles')
        historical = {role: pins[pinned(ref, pins)] for role, ref in producer_refs.items()}
        g10_path = pinned(entry['g10_admission_script'], pins)
        g10_sha = pins[g10_path]
        require(completed.get('g10_admission_script_sha256') == g10_sha
                and namespace == {'g10_common_admission': admission,
                                  'g10_admission_script_sha256': g10_sha},
                'native WDL historical G10 producer proof differs')
        attrs_refs = entry['attributes']
        require(isinstance(attrs_refs, dict)
                and set(attrs_refs) == {s['path'] for s in selection},
                'native WDL attribute proof coverage')
        for spec in selection:
            name = spec['path']
            require(name in by_name and name not in bindings, 'native WDL overlapping selection')
            attrs_path, attrs = read(attrs_refs[name], pins)
            require(attrs_path == directory / name / '.zattrs', 'native WDL attribute namespace')
            expected = {
                'schema': 1, 'g10_common_admission': admission,
                'g10_admission_script_sha256': g10_sha,
                'source_dir': str(source), 'source_shard': name,
                'source_summary_sha256': summary_sha,
                'source_storage_identity': wdl.storage_identity(source / name),
                'rows': spec['rows'], 'onnx': argv['onnx'], 'onnx_sha256': model_sha,
                'requested_wdl': {'output': head, 'kind': 'probabilities'},
                'history_lineage': wdl.LINEAGE, 'producer': historical,
            }
            require(Path(argv['onnx']).is_absolute()
                    and attrs.get('complete') is True and attrs.get('binding') == expected,
                    'native WDL historical binding differs')
            bindings[name] = expected
            if shard_roots is not None:
                shard_roots[name] = directory
    require(set(bindings) == set(by_name), 'native WDL missing complete-cohort coverage')
    require(used_directories == set(directories), 'native WDL unused directory')
    return bindings
