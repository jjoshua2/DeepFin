#!/usr/bin/env python3
"""Join completed CeresB50 policy and CeresV25 WDL without teacher inference."""
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
from scripts import bt4_one_epoch_screen as recipes
from scripts import bt4_value_rewrite as copies
from scripts import ceres_value_mix as value
from scripts.sf_policy_rewrite import ARRAYS, require

PROFILE = 'CeresB50V25'
ROWS, SHARDS = 18910484, 2309
SUMMARY = 'ceres_joint_recipe_summary.json'
DERIVE = 'derive_targets_summary.json'
MUTATED = {'policy_target', 'search_wdl'}


def sha(path: Path) -> str:
    return value.wdl.file_sha256(path)


def read(ref: dict[str, Any]) -> dict[str, Any]:
    require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute(), 'invalid reference')
    require(sha(Path(ref['path'])) == ref['sha256'], 'reference digest differs')
    return json.loads(Path(ref['path']).read_text())


def ref(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': sha(path)}


def producer_pins() -> dict[str, str]:
    return {**value.producer_pins(), str(Path(__file__).resolve()): sha(Path(__file__)),
            str(Path(recipes.__file__).resolve()): sha(Path(recipes.__file__))}


def file_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def unchanged(files: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in files.items() if k != '.zattrs' and k.split('/')[0] not in MUTATED}


def validate_parents(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(manifest.get('schema') == 1 and manifest.get('profile') == PROFILE, 'unsupported joint recipe')
    source = read(manifest['source'])
    require(manifest['source'] == {'path': str(recipes.SOURCE / DERIVE),
        'sha256': recipes.COMMON_PINS[str(recipes.SOURCE / DERIVE)]}, 'original SF source differs')
    found = []
    for role, profile, summary_name, verifier in (
        ('policy', 'CeresB50', 'ceres_target_mix_summary.json', recipes.verify_ceres_recipe),
        ('value', 'B100CeresV25', value.SUMMARY, recipes.verify_ceres_value_recipe),
    ):
        parent = manifest[role]
        root = recipes.CORPORA[profile]
        require(parent['derive']['path'] == str(root / DERIVE)
                and parent['recipe']['path'] == str(root / summary_name), 'parent path differs')
        derived, recipe = read(parent['derive']), read(parent['recipe'])
        verifier({'profile': profile, 'ceres_producer_pins': recipe['producer_sha256']}, recipe, derived)
        require(derived.get('policy_target_postprocess') ==
                ({k: v for k, v in recipe.items() if k != 'outputs'} if role == 'policy'
                 else json.loads((recipes.CORPORA['B100'] / DERIVE).read_text())['policy_target_postprocess']),
                'parent policy lineage differs')
        copies.inventory(root, source['shards'])
        found.append(recipe)
    return source, found[0], found[1]


def combine_shard(original: Path, policy: Path, values: Path, dest: Path,
                  spec: dict[str, Any], policy_proof: dict[str, Any], value_proof: dict[str, Any]) -> dict[str, Any]:
    require(not os.path.lexists(dest), 'output shard exists')
    roots = (original, policy, values)
    stamps = {str(p): value.wdl.storage_identity(p) for p in roots}
    maps = {str(p): copies.file_map(p) for p in roots}
    for root, proof in ((policy, policy_proof), (values, value_proof)):
        require(proof['path'] == spec['path'] and proof['rows'] == spec['rows'], 'parent shard order differs')
        require(file_digest(maps[str(root)]) == proof['files_manifest_sha256'], 'parent published bytes differ')
    require(unchanged(maps[str(original)]) == unchanged(maps[str(policy)]) == unchanged(maps[str(values)]),
            'parent non-target bytes differ')
    groups = [zarr.open_group(str(p), mode='r') for p in roots]
    for group in groups:
        require(set(group.array_keys()) == set(ARRAYS), '17-array corpus required')
        for name in ARRAYS:
            value.wdl.complete_chunks(group[name])
            require(group[name].shape[0] == spec['rows'], 'array row count differs')
    for group, name, width in ((groups[1], 'policy_target', 1858), (groups[2], 'search_wdl', 3)):
        a = np.asarray(group[name][:])
        require(a.shape == (spec['rows'], width) and a.dtype == np.float16
                and bool(np.isfinite(a).all()) and bool((a >= 0).all()), 'target layout or values differ')
        require(bool((np.abs(a.astype(np.float64).sum(1) - 1) <= 2**-10).all()), 'target mass differs')
        if name == 'policy_target':
            require(bool((a[np.asarray(group['legal_mask'][:]) == 0] == 0).all()), 'illegal policy mass')
        flag = 'has_policy' if name == 'policy_target' else 'has_search_wdl'
        require(bool(np.all(group[flag][:] == 1)), 'target coverage differs')
    shutil.copytree(policy, dest)
    shutil.rmtree(dest / 'search_wdl')
    shutil.copytree(values / 'search_wdl', dest / 'search_wdl')
    out: Any = zarr.open_group(str(dest), mode='a')
    attrs = dict(groups[1].attrs)
    for key in ('derive_value_scheme', 'derive_value_source', 'value_target_postprocess'):
        require(key in groups[2].attrs, 'value provenance absent')
        attrs[key] = groups[2].attrs[key]
    attrs['joint_recipe'] = PROFILE
    out.attrs.put(attrs)
    final = copies.file_map(dest)
    require(unchanged(final) == unchanged(maps[str(original)]), 'joint copy changed non-target bytes')
    for name, parent in (('policy_target', policy), ('search_wdl', values)):
        def subset(files: dict[str, str], array: str = name) -> dict[str, str]:
            return {k: v for k, v in files.items() if k.split('/')[0] == array}
        require(subset(final) == subset(maps[str(parent)]), 'joint target bytes differ')
    require(all(value.wdl.storage_identity(p) == stamps[str(p)] for p in roots), 'parent changed during copy')
    return {'path': spec['path'], 'rows': spec['rows'], 'parents_storage_identity': stamps,
            'files_manifest_sha256': file_digest(final), 'output_storage_identity': value.wdl.storage_identity(dest),
            'allocated_bytes': sum(p.stat().st_blocks * 512 for p in dest.rglob('*'))}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    require(math.isfinite(args.max_seconds) and args.max_seconds > 0, 'invalid deadline')
    require(math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0, 'invalid disk floor')
    require(math.isfinite(args.max_output_gib) and args.max_output_gib > 0, 'invalid output cap')
    started = time.monotonic()
    manifest_path = Path(args.manifest).resolve()
    manifest = read({'path': str(manifest_path), 'sha256': args.expected_manifest_sha256})
    source, policy_recipe, value_recipe = validate_parents(manifest)
    out = Path(args.out).absolute()
    writing = out.with_name(out.name + '.writing')
    roots = (recipes.SOURCE, recipes.CORPORA['CeresB50'], recipes.CORPORA['B100CeresV25'])
    require(out.parent.is_dir() and out.parent.resolve() == out.parent, 'canonical output parent required')
    require(not os.path.lexists(out) and not os.path.lexists(writing), 'output or partial exists')
    require(all(p != q and p not in q.parents and q not in p.parents
                for p in (out, writing) for q in (*roots, manifest_path)), 'output overlaps input')
    producer = producer_pins()
    def guard() -> None:
        require(time.monotonic() - started < args.max_seconds, 'materialization deadline')
        require(shutil.disk_usage(out.parent).free >= args.minimum_free_gib * 1024**3, 'disk floor')
    guard()
    writing.mkdir()
    outputs = []
    for spec, pp, vp in zip(source['shards'], policy_recipe['outputs'], value_recipe['outputs'], strict=True):
        guard()
        outputs.append(combine_shard(roots[0] / spec['path'], roots[1] / spec['path'],
                                     roots[2] / spec['path'], writing / spec['path'], spec, pp, vp))
        require(sum(p['allocated_bytes'] for p in outputs) <= args.max_output_gib * 1024**3, 'output allocation cap')
    validate_parents(manifest)
    require(sha(manifest_path) == args.expected_manifest_sha256, 'manifest changed')
    require(all(sha(Path(p)) == h for p, h in producer.items()), 'producer changed')
    for proof in outputs:
        guard()
        for path, identity in proof['parents_storage_identity'].items():
            require(value.wdl.storage_identity(Path(path)) == identity, 'previous parent changed')
        require(value.wdl.storage_identity(writing / proof['path']) == proof['output_storage_identity'], 'previous output changed')
    result = {'schema': 1, 'status': 'COMPLETE', 'profile': PROFILE, 'manifest': ref(manifest_path),
              'parents': manifest, 'producer_sha256': producer, 'rows': sum(s['rows'] for s in source['shards']),
              'shards': len(outputs), 'mutated_arrays': sorted(MUTATED), 'unchanged_arrays': sorted(ARRAYS - MUTATED),
              'new_teacher_evaluations': 0, 'outputs': outputs}
    pd = read(manifest['policy']['derive'])
    vd = read(manifest['value']['derive'])
    derived = {**pd, 'value_scheme': vd['value_scheme'], 'value_target_postprocess': vd['value_target_postprocess'],
               'joint_recipe': {k: v for k, v in result.items() if k != 'outputs'}}
    for name, obj in ((SUMMARY, result), (DERIVE, derived)):
        (writing / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    guard()
    writing.rename(out)
    return result


def verify_training_recipe(prep: dict[str, Any]) -> Path:
    """Explicit joint-profile admission; existing profiles retain their gates."""
    recipe, derived, qualification = (read(prep[k]) for k in ('recipe_summary', 'derive_summary', 'data_qualification'))
    corpus = Path(prep['derive_summary']['path']).parent
    require(prep['recipe_summary']['path'] == str(corpus / SUMMARY) and corpus.is_dir()
            and not corpus.is_symlink() and not os.path.lexists(corpus.with_name(corpus.name + '.writing')),
            'joint corpus incomplete')
    source, _, _ = validate_parents(recipe['parents'])
    require(prep['source'] == recipe['parents']['source'], 'joint original source differs')
    require(read(recipe['manifest']) == recipe['parents'], 'joint manifest differs')
    expected = {'schema': 1, 'status': 'COMPLETE', 'profile': PROFILE, 'rows': ROWS, 'shards': SHARDS,
                'mutated_arrays': sorted(MUTATED), 'unchanged_arrays': sorted(ARRAYS - MUTATED), 'new_teacher_evaluations': 0}
    require(all(recipe.get(k) == v for k, v in expected.items()), 'joint recipe differs')
    require([(x['path'], x['rows']) for x in recipe['outputs']] == [(x['path'], x['rows']) for x in source['shards']], 'joint output roster differs')
    require(recipe['producer_sha256'] == producer_pins(), 'joint producer differs')
    pd, vd = read(recipe['parents']['policy']['derive']), read(recipe['parents']['value']['derive'])
    require(derived == {**pd, 'value_scheme': vd['value_scheme'], 'value_target_postprocess': vd['value_target_postprocess'],
                       'joint_recipe': {k: v for k, v in recipe.items() if k != 'outputs'}}, 'joint derived lineage differs')
    require(qualification == {'schema': 1, 'status': 'PASS_JOINT_CORPUS_QUALIFICATION', 'profile': PROFILE,
                             'corpus': str(corpus), 'source': prep['source'], 'derive_summary': prep['derive_summary'],
                             'recipe_summary': prep['recipe_summary'], 'rows': ROWS, 'shards': SHARDS}, 'joint qualification differs')
    copies.inventory(corpus, source['shards'])
    for proof in recipe['outputs']:
        require(value.wdl.storage_identity(corpus / proof['path']) == proof['output_storage_identity'], 'joint output changed')
    return corpus


def qualify(corpus: Path, output: Path) -> dict[str, Any]:
    corpus, output = corpus.resolve(), output.absolute()
    partial = output.with_name(output.name + '.writing')
    require(not os.path.lexists(output) and not os.path.lexists(partial), 'qualification exists')
    recipe_ref, derive_ref = ref(corpus / SUMMARY), ref(corpus / DERIVE)
    recipe = read(recipe_ref)
    qualification = {'schema': 1, 'status': 'PASS_JOINT_CORPUS_QUALIFICATION', 'profile': PROFILE,
                     'corpus': str(corpus), 'source': recipe['parents']['source'],
                     'derive_summary': derive_ref, 'recipe_summary': recipe_ref, 'rows': ROWS, 'shards': SHARDS}
    partial.write_text(json.dumps(qualification, indent=2, sort_keys=True) + '\n')
    verify_training_recipe({'source': qualification['source'], 'derive_summary': derive_ref,
                            'recipe_summary': recipe_ref, 'data_qualification': ref(partial)})
    partial.rename(output)
    return qualification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    materializer = commands.add_parser('materialize')
    materializer.add_argument('--manifest', required=True)
    materializer.add_argument('--expected-manifest-sha256', required=True)
    materializer.add_argument('--out', required=True)
    materializer.add_argument('--max-seconds', type=float, default=28800)
    materializer.add_argument('--max-output-gib', type=float, default=18)
    materializer.add_argument('--minimum-free-gib', type=float, default=150)
    qualifier = commands.add_parser('qualify')
    qualifier.add_argument('--corpus', required=True, type=Path)
    qualifier.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'qualify':
        qualify(args.corpus, args.out)
    else:
        materialize(args)


if __name__ == '__main__':
    main()
