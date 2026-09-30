#!/usr/bin/env python3
"""Qualify published Ceres corpus metadata against completed producer witnesses.

Payload checks are inherited from the pinned producer, with its own storage
identities checked before and after inspection. No tensor/history replay occurs.
The caller supplies the enclosing process/resource deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_one_epoch_screen as epoch
from scripts import ceres_target_mix as policy
from scripts import ceres_value_mix as value

ROWS, SHARDS = 18910484, 2309
PROFILES = {'Ceres100': policy.SUMMARY, 'CeresB50': policy.SUMMARY, 'B100CeresV25': value.SUMMARY}
require = policy.require


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(path: Path) -> tuple[int, ...]:
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode), f'nonregular input: {path}')
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def qualify(plan_path: Path, expected: str, out: Path, *, execute: bool) -> dict[str, Any]:
    started = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be hidden')
    policy.shared.set_nthreads(2)
    seen: dict[Path, tuple[int, ...]] = {}

    def read(path: Path, digest: str | None = None) -> dict[str, Any]:
        before = stamp(path)
        require(path.is_file() and before[2] <= 32 * 2**20, 'metadata read exceeds bound')
        raw = path.read_bytes()
        require(stamp(path) == before and (path not in seen or seen[path] == before), 'metadata changed')
        require(digest is None or hashlib.sha256(raw).hexdigest() == digest, f'metadata hash differs: {path}')
        seen[path] = before
        result = json.loads(raw)
        require(isinstance(result, dict), 'metadata object required')
        return result

    plan = read(plan_path, expected)
    require(plan.get('schema') == 1 and plan.get('profile') in PROFILES, 'unsupported qualification plan')
    profile = plan['profile']
    corpus = Path(plan['corpus'])
    require(corpus == epoch.corpus_for({'profile': profile}) and corpus.resolve() == corpus,
            'noncanonical registered corpus')
    require(math.isfinite(plan['max_seconds']) and 0 < plan['max_seconds'] <= 1800, 'invalid deadline')
    require(math.isfinite(plan['minimum_free_gib']) and plan['minimum_free_gib'] >= 0, 'invalid reserve')
    require(isinstance(plan['stop_paths'], list) and all(isinstance(p, str) for p in plan['stop_paths']), 'invalid STOP paths')
    out = out.absolute()
    partial = out.with_name(out.name + '.writing')
    require(out.parent.is_dir() and out.parent.resolve() == out.parent, 'canonical output parent required')
    require(not os.path.lexists(out) and not os.path.lexists(partial), 'qualification output already exists')

    def guard() -> None:
        require(time.monotonic() - started < plan['max_seconds'], 'qualification deadline')
        require(shutil.disk_usage(out.parent).free >= plan['minimum_free_gib'] * 2**30, 'disk reserve')
        require(not any(os.path.lexists(p) for p in plan['stop_paths']), 'STOP requested')

    def reference(key: str) -> dict[str, Any]:
        ref = plan[key]
        require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute(), 'invalid metadata reference')
        return read(Path(ref['path']), ref['sha256'])

    guard()
    require(plan['derive_summary']['path'] == str(corpus / policy.DERIVE_SUMMARY)
            and plan['rewrite_summary']['path'] == str(corpus / PROFILES[profile]), 'wrong publication paths')
    require(not os.path.lexists(corpus.with_name(corpus.name + '.writing'))
            and not (corpus / 'failed.json').exists(), 'incomplete publication')
    manifest, rewritten, derived = reference('producer_manifest'), reference('rewrite_summary'), reference('derive_summary')
    terminal = reference('materialization')
    pins = plan['ceres_producer_pins']
    require(isinstance(pins, dict) and bool(pins), 'producer freeze required')
    require(rewritten['producer_sha256'] == pins and rewritten['manifest_sha256'] == plan['producer_manifest']['sha256'], 'producer/manifest differs')
    terminal_expected = {'status': 'COMPLETE', 'returncode': 0, 'profile': profile, 'corpus': str(corpus),
                        'producer_manifest_sha256': plan['producer_manifest']['sha256'],
                        'derive_summary_sha256': plan['derive_summary']['sha256'],
                        'rewrite_summary_sha256': plan['rewrite_summary']['sha256'], 'producer_sha256': pins}
    require(all(terminal.get(k) == v for k, v in terminal_expected.items())
            and type(terminal.get('returncode')) is int and not terminal.get('error')
            and not terminal.get('stop_reason'), 'materialization incomplete or binding differs')
    for path, digest in pins.items():
        p = Path(path)
        require(p.is_absolute() and p.resolve() == p and sha(p) == digest, 'producer pin differs')
        seen[p] = stamp(p)
    source = Path(manifest['source'])
    sf = Path(manifest.get('sf_source', manifest['source']))
    require(sf == epoch.SOURCE and source == (sf if profile in epoch.CERES_POLICY_PROFILES else epoch.CORPORA['B100']), 'wrong source')
    originals = read(sf / policy.DERIVE_SUMMARY, epoch.COMMON_PINS[str(sf / policy.DERIVE_SUMMARY)])
    read(source / policy.DERIVE_SUMMARY, manifest['source_summary_sha256'])
    specs = originals['shards']
    require(len(specs) == SHARDS and sum(s['rows'] for s in specs) == ROWS, 'wrong full source counts')
    names = [s['path'] for s in specs]
    require(len(set(names)) == SHARDS and all(Path(n).name == n and n.startswith('shard_') and n.endswith('.zarr') for n in names), 'invalid source shard names')
    require([(p['path'], p['rows']) for p in rewritten['outputs']] == [(s['path'], s['rows']) for s in specs]
            and [e['shard'] for e in manifest['entries']] == names, 'output/teacher coverage differs')
    roots = [corpus, source, sf, *(Path(e[k]) for e in manifest['entries'] for k in ('bt4', 'ceres'))]
    require(all(out != p and p not in out.parents and out not in p.parents
                and partial != p and p not in partial.parents and partial not in p.parents
                for p in [*roots, *seen]), 'qualification output overlaps input')
    for root in {corpus, source, sf}:
        seen[root] = stamp(root)
        require(sorted(p.name for p in root.glob('shard_*.zarr')) == sorted(names), 'shard membership differs')
    metadata = []
    states: dict[Path, str] = {}
    for spec, proof, entry in zip(specs, rewritten['outputs'], manifest['entries'], strict=True):
        guard()
        name, rows = spec['path'], spec['rows']
        dest, parent = corpus / name, source / name
        mappings = [(dest, 'output_storage_identity'), (parent, 'source_storage_identity'),
                    (Path(entry['bt4']), 'bt4_storage_identity'), (Path(entry['ceres']), 'ceres_storage_identity')]
        if profile == 'B100CeresV25':
            mappings.append((sf / name, 'sf_storage_identity'))
        for path, key in mappings:
            require(path.resolve() == path and path.is_dir(), 'aliased/missing shard')
            actual = policy.shared.storage_identity(path)
            require(actual == proof.get(key), f'storage identity differs: {key}: {name}')
            states[path] = actual
        require({p.name for p in dest.iterdir() if p.is_dir()} == set(policy.ARRAYS), '17-array membership differs')
        require(read(dest / '.zgroup') == read(parent / '.zgroup') == {'zarr_format': 2}, 'group format differs')
        attrs = read(dest / '.zattrs', proof['attrs_sha256'])
        expected_attrs = read(parent / '.zattrs')
        if profile in epoch.CERES_POLICY_PROFILES:
            expected_attrs['ceres_policy_postprocess'] = {
                'schema': 1, 'kind': 'bt4-ceres-policy', 'algorithm': policy.ALGORITHM,
                'weights': rewritten['weights'], 'temperatures': rewritten['temperatures'],
                'manifest_sha256': rewritten['manifest_sha256'], 'source_storage_identity': proof['source_storage_identity']}
        else:
            expected_attrs.update(derive_value_scheme=rewritten['value_scheme'], derive_value_source=rewritten['value_source'],
                value_target_postprocess={'schema': 1, 'kind': 'sf-bt4-ceres-value', 'weights': rewritten['weights'],
                                          'ceres_value_profile': rewritten['ceres_value_profile'], 'manifest_sha256': rewritten['manifest_sha256']})
        require(attrs == expected_attrs and attrs['positions'] == rows, 'actual recipe attrs differ')
        layouts = {}
        for column in sorted(policy.ARRAYS):
            layout = read(dest / column / '.zarray')
            require(layout == read(parent / column / '.zarray') and layout['shape'][0] == rows, 'array layout differs')
            if column in ('policy_target', 'search_wdl'):
                require(layout['shape'] == [rows, 1858 if column == 'policy_target' else 3]
                        and layout['dtype'] == '<f2', 'target layout differs')
            layouts[column] = sha(dest / column / '.zarray')
        digest = proof['files_manifest_sha256']
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'file proof missing')
        metadata.append({'shard': name, 'rows': rows, 'attrs_sha256': proof['attrs_sha256'],
                         'layouts': layouts, 'files_manifest_sha256': digest,
                         'output_storage_identity': proof['output_storage_identity']})
    m = {'profile': profile, 'ceres_producer_pins': pins}
    (epoch.verify_ceres_recipe if profile in epoch.CERES_POLICY_PROFILES else epoch.verify_ceres_value_recipe)(m, rewritten, derived)
    require(derived.get('policy_target_postprocess') == {k: v for k, v in rewritten.items() if k != 'outputs'}
            if profile in epoch.CERES_POLICY_PROFILES else derived.get('value_target_postprocess') == {k: v for k, v in rewritten.items() if k != 'outputs'}, 'derive postprocess differs')
    for path, expected_state in states.items():
        guard()
        require(policy.shared.storage_identity(path) == expected_state, 'storage changed during qualification')
    for path, before in seen.items():
        require(stamp(path) == before, 'input replaced/changed during qualification')
    guard()
    result = {'schema': 1, 'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'profile': profile,
              'corpus': str(corpus), 'rows': ROWS, 'shards': SHARDS,
              'source': {'path': str(sf), 'derive_sha256': epoch.COMMON_PINS[str(sf / policy.DERIVE_SUMMARY)]},
              'derive_summary': plan['derive_summary'], 'rewrite_summary': plan['rewrite_summary'],
              'materialization_status': plan['materialization'], 'plan_sha256': expected,
              'inspector_sha256': sha(Path(__file__)), 'metadata': metadata,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Actual attrs/layouts and stable producer-bound storage identities. Payload and history checks inherited from pinned completed producer; no payload reread.'}
    if execute:
        with partial.open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        guard()
        # Publish without overwriting even if another writer raced the initial check.
        os.link(partial, out)
        partial.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = qualify(args.plan, args.expected_plan_sha256, args.out, execute=args.execute)
    print(json.dumps({'status': result['status'] if args.execute else 'QUALIFIED_NOT_PUBLISHED', 'rows': result['rows']}))


if __name__ == '__main__':
    main()
