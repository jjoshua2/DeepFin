"""Admission for the immutable qualified Soft-SF sample, never a whole corpus."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import bt4_derived_wdl_sidecar as shared
from scripts.sf_policy_rewrite import require

PROFILE = 'ceres-c3-fixed32-soft-sf-selected-v1'
QUALIFICATION_SHA = '32a329ec01c0b205f9397c72928f407f3ef4131a0b1d165f461b3e4e78ae3dcb'
ROWS, FRAGMENTS = 4096, 64
KEYS = ('source_dir', 'derived_shard', 'derived_row', 'game_id', 'ply', 'stratum', 'weight')
EXTRA_DTYPES = {'selection_index': 'uint64', 'x_sha256': 'uint8'}


def enabled(args: argparse.Namespace) -> bool:
    return getattr(args, 'selected_bank_qualification', None) is not None


def validate(args: argparse.Namespace) -> None:
    pin = getattr(args, 'expected_selected_bank_qualification_sha256', None)
    require(enabled(args) == bool(pin), 'selected bank requires qualification path and SHA256')
    if enabled(args):
        require(pin == QUALIFICATION_SHA, 'unsupported selected-bank qualification')
        require(Path(args.selected_bank_qualification).resolve() == Path(args.source).resolve() / 'complete.json',
                'qualification must belong to selected bank')
        require(not args.expected_source_summary_sha256 and not args.g10_common_qualification
                and not args.expected_g10_common_qualification_sha256
                and args.start_shard == 0 and args.max_shards == FRAGMENTS,
                'selected bank cannot use corpus summary, G10 admission or partial shard range')
        require(args.retain_value2 and args.pad_final_batch,
                'selected bank requires both raw heads and explicit final-batch padding')
    else:
        require(bool(args.expected_source_summary_sha256), 'source summary SHA256 required')


def read_file(path: Path, pin: dict[str, Any]) -> tuple[bytes, str]:
    require(path.is_file() and not path.is_symlink(), 'bank file must be regular and nonsymlink')
    state = shared.storage_identity(path)
    require(path.stat().st_size == pin['bytes'], 'bank file size differs')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin['sha256'], 'bank file digest differs')
    require(shared.storage_identity(path) == state, 'bank file changed during read')
    return data, state


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row['source_dir'], row['derived_shard'], row['derived_row']


def admit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate(args)
    root = Path(args.source)
    complete_path = Path(args.selected_bank_qualification)
    data, complete_state = read_file(complete_path,
        {'sha256': QUALIFICATION_SHA, 'bytes': complete_path.stat().st_size})
    complete = json.loads(data)
    require(complete['status'] == 'COMPLETE_QUALIFIED_TRAINING_SAMPLE'
            and complete['rows'] == ROWS and complete['derived_shards'] == FRAGMENTS,
            'unqualified selected sample')
    pins = complete['outputs']
    states = {'complete.json': complete_state}
    data, states['selection.json'] = read_file(root / 'selection.json', pins['selection.json'])
    selection = json.loads(data)['selected']
    require(len(selection) == ROWS and len({row_key(r) for r in selection}) == ROWS,
            'selected row count or duplicates')
    specs: list[dict[str, Any]] = []
    for index, row in enumerate(selection):
        shard = row['derived_shard']
        require(Path(shard).name == shard and shard.startswith('shard_') and shard.endswith('.zarr')
                and type(row['derived_row']) is int and 0 <= row['derived_row'] < 8192
                and type(row['stratum']) is int and np.isfinite(row['weight']) and row['weight'] > 0,
                'invalid selected identity or weight')
        if not specs or specs[-1]['fragment'] != shard + '.npz':
            specs.append({'path': f'selection_{len(specs):06d}.zarr', 'fragment': shard + '.npz',
                          'rows': 0, 'indices': [], 'identities': []})
        specs[-1]['rows'] += 1
        specs[-1]['indices'].append(index)
        specs[-1]['identities'].append(row)
    require(len(specs) == FRAGMENTS and len({s['fragment'] for s in specs}) == FRAGMENTS
            and {p for p in pins if p.endswith('.npz')} == {s['fragment'] for s in specs},
            'selected fragment inventory/order differs')
    wanted = {row_key(r): r for r in selection}
    data, states['raw_rows.jsonl'] = read_file(root / 'raw_rows.jsonl', pins['raw_rows.jsonl'])
    found: dict[tuple[str, str, int], dict[str, Any]] = {}
    raw_ids = set()
    for line in data.splitlines():
        row = json.loads(line)
        key = row_key(row)
        require(key in wanted and key not in found
                and all(row[k] == wanted[key][k] for k in KEYS), 'raw selected-row join differs')
        raw = row['raw']
        require(raw['schema'] == 3 and all(row[k] == raw[k] for k in
                ('game_id', 'ply', 'worker_id', 'input_key'))
                and len(raw['history_uci']) == raw['history_plies'], 'raw history identity differs')
        raw_id = (row['source_dir'], row['raw_shard'], row['physical_row'])
        require(raw_id not in raw_ids, 'duplicate raw physical identity')
        raw_ids.add(raw_id)
        history = {k: raw[k] for k in ('history_root_fen', 'history_uci', 'history_plies')}
        found[key] = {k: row[k] for k in (*KEYS, 'raw_shard', 'physical_row', 'worker_id', 'input_key')}
        found[key].update(raw_record_sha256=hashlib.sha256(line).hexdigest(),
                          history_sha256=hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest())
    require(set(found) == set(wanted), 'missing selected raw history')
    for spec in specs:
        spec['identities'] = [found[row_key(r)] for r in spec['identities']]
        spec['fragment_pin'] = pins[spec['fragment']]
        states[spec['fragment']] = shared.storage_identity(root / spec['fragment'])
    args.selected_bank_states = states
    guard(args)
    return {}, specs


def guard(args: argparse.Namespace) -> None:
    require(all(shared.storage_identity(Path(args.source) / name) == state
                for name, state in args.selected_bank_states.items()), 'selected bank changed')
    require(shared.file_sha256(args.selected_bank_qualification) == QUALIFICATION_SHA,
            'selected qualification changed')


def binding(args: argparse.Namespace) -> dict[str, Any]:
    return {'selected_bank_qualification_sha256': args.expected_selected_bank_qualification_sha256,
            'selection_qualification': 'soft_sf_qualified_sample_v1',
            'selected_reader_sha256': shared.file_sha256(__file__)}


def arrays(args: argparse.Namespace, spec: dict[str, Any], columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    data, state = read_file(Path(args.source) / spec['fragment'], spec['fragment_pin'])
    require(state == args.selected_bank_states[spec['fragment']], 'selected fragment changed')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        group = {k: archive[k] for k in columns}
        require(np.array_equal(archive['selected_rows'], [r['derived_row'] for r in spec['identities']]),
                'fragment selected row order differs')
    n = spec['rows']
    for k, values in group.items():
        shape = (n, 175, 8, 8) if k == 'x' else (n, 1858) if k == 'legal_mask' else (n,)
        require(values.shape == shape, 'selected array shape differs')
        require(values.dtype == np.float16 if k == 'x' else values.dtype.kind in 'biu',
                'selected array dtype differs')
    require(np.array_equal(group['game_id'], [r['game_id'] for r in spec['identities']])
            and np.array_equal(group['ply_index'], [r['ply'] for r in spec['identities']]),
            'selected game/ply differs')
    return group
