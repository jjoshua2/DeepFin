#!/usr/bin/env python3
"""Seal ordinary base bytes or qualify a completed policy overlay for exact training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.replay.target_overlay import (
    BASE_STATUS, OVERLAY_STATUS, POLICY, BaseSeal, _atomic_new_json, _open_manifest,
    _plain_content, _root_files, has_overlay, overlay_content_sha256,
    require, require_base_corpus, tree_stamp,
)
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.replay.game_epoch import _input_history_identity


def _identity(arrays: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    n = int(arrays['x'].shape[0])
    require(n > 0, 'overlay bases require nonempty shards')
    digest = hashlib.sha256()
    for key in ('game_id', 'ply_index', 'has_game_id', 'has_ply_index'):
        require(key in arrays, f'missing base row identity {key}')
        value = np.asarray(arrays[key])
        require(value.shape == (n,), f'row identity shape differs: {key}')
        if key.startswith('has_'):
            require(bool((value == 1).all()), 'incomplete base row identity')
        else:
            require(value.dtype.kind in 'iu' and bool((value >= 0).all()), 'invalid game/ply identity')
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    encoding, repfix = _input_history_identity(arrays, path=Path('<storage-base>'))
    return {'rows': n, 'row_identity_sha256': digest.hexdigest(),
            'input_history_encoding': encoding, 'history_rep_fix': repfix,
            'input_shape': list(arrays['x'].shape), 'policy_shape': list(arrays[POLICY].shape),
            'policy_dtype': np.dtype(arrays[POLICY].dtype).str,
            'policy_encoding': meta.get('policy_encoding')}


def seal_base(root: Path, output: Path) -> dict[str, Any]:
    """Validate and seal actual ordinary bytes; never upgrades recipe science."""
    root = root.resolve(strict=True)
    require(root not in output.resolve().parents, 'store base seal outside immutable base')
    paths = iter_shard_paths(root)
    require(bool(paths) and not root.name.endswith('.writing'), 'incomplete/empty base')
    root_files = _root_files(root)
    entries = []
    for path in paths:
        require(not has_overlay(path), 'overlay chains are unsupported')
        before = tree_stamp(path)
        content = _plain_content(path)
        arrays, meta = load_shard_arrays(path, lazy=False)
        identity = _identity(arrays, meta)
        require(tree_stamp(path) == before, 'base changed while sealing')
        entries.append({'name': path.name, 'content_sha256': content, 'storage_stamp': before,
                        'identity': identity})
    require([p.name for p in iter_shard_paths(root)] == [e['name'] for e in entries],
            'base membership changed during seal')
    require(all(tree_stamp(root / e['name']) == e['storage_stamp'] for e in entries),
            'base changed before seal publication')
    result = {'schema': 1, 'status': BASE_STATUS, 'base': str(root),
              'rows': sum(e['identity']['rows'] for e in entries), 'shards': entries,
              'root_files': root_files,
              'scope': 'Immutable storage bytes, declarations, content validity and row/history identity; not recipe qualification.'}
    require(_root_files(root) == root_files, 'base metadata changed during seal')
    _atomic_new_json(output, result)
    return result


def qualify_overlay(root: Path, output: Path) -> dict[str, Any]:
    """Validate actual composed arrays before publishing storage-only admission."""
    root = root.resolve(strict=True)
    require(root not in output.resolve().parents, 'keep qualification outside immutable output')
    require(not root.name.endswith('.writing'), 'cannot qualify unpublished output')
    paths = iter_shard_paths(root)
    require(bool(paths), 'empty overlay corpus')
    before_files = _root_files(root)
    mix = json.loads((root / 'bt4_policy_mix_summary.json').read_bytes())
    derive = json.loads((root / 'derive_targets_summary.json').read_bytes())
    require(derive.get('policy_target_postprocess') == mix and mix.get('kind') == 'global',
            'requires genuine global-policy producer summaries')
    storage = mix.get('storage', {})
    require(storage.get('kind') == 'immutable-policy-overlay', 'missing overlay producer identity')
    context = BaseSeal(storage['base_seal'])
    seal = require_base_corpus(storage['base_seal'], Path(storage['base']), context=context)
    require([p.name for p in paths] == [e['name'] for e in seal['shards']], 'overlay membership differs')
    entries = []
    for path in paths:
        content = overlay_content_sha256(path, seal=context)
        manifest, meta = _open_manifest(path, seal=context)
        require(manifest['base_seal'] == storage['base_seal']
                and Path(manifest['base']) == Path(storage['base']) / path.name, 'mixed overlay bases')
        for field, value in (('kind', 'global'), ('alpha', mix['alpha']),
                             ('bt4_temperature', mix['bt4_temperature'])):
            require(meta.get('policy_target_mix_' + field) == value, 'overlay recipe stamp differs')
        arrays, _ = load_shard_arrays(path, allow_target_overlay=True, overlay_seal=context)
        identity = _identity(arrays, meta)
        require(identity == manifest['identity'], 'composed row/history identity differs')
        require(overlay_content_sha256(path, seal=context) == content, 'overlay changed during qualification')
        entries.append({'name': path.name, 'content_sha256': content, 'identity': identity})
    require(sum(e['identity']['rows'] for e in entries) == mix['rows'] == seal['rows']
            and len(entries) == mix['shards'], 'overlay completion counts differ')
    require([p.name for p in iter_shard_paths(root)] == [e['name'] for e in entries]
            and _root_files(root) == before_files, 'overlay publication changed during qualification')
    require_base_corpus(storage['base_seal'], Path(storage['base']), context=context)
    result = {'schema': 1, 'status': OVERLAY_STATUS, 'root': str(root),
              'base_seal': storage['base_seal'], 'shards': entries, 'rows': seal['rows'],
              'root_files': before_files,
              'scope': 'Storage validity and immutable inheritance only; no recipe, model, or playing-strength qualification.'}
    _atomic_new_json(output, result)
    return result



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('seal-base', 'qualify-overlay'))
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    operation = seal_base if args.operation == 'seal-base' else qualify_overlay
    operation(args.shards, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
