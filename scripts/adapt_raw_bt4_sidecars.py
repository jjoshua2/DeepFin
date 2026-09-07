#!/usr/bin/env python3
"""Join pinned raw BT4 labels to derived rows through verified full-history provenance.

CPU-only preparation. Input manifest: schema1, derived_summary {path,sha256},
teacher {onnx {path,sha256}, policy_output, providers, remap}, sources list of
{source_dir, sidecar_dir, manifest {path,sha256}, receipts {path,sha256}}.
Receipts must be immutable snapshots of closed-shard raw progress, never live logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from scripts import bt4_policy_mix as mix
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import corpus_row_provenance as provenance
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus
from scripts.bt4_policy_dump import file_sha256, remap_provenance


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pin(item: dict[str, Any]) -> Path:
    require(set(item) == {'path', 'sha256'}, 'expected explicit path/SHA256 pin')
    path = Path(item['path'])
    require(path.is_absolute() and path == path.resolve(), 'pinned paths must be canonical absolute paths')
    require(file_sha256(path) == item['sha256'], f'changed pinned input: {path}')
    return path


def storage_identity(path: Path) -> str:
    """Detect changes after content verification without retaining per-file data."""
    digest = hashlib.sha256()
    def entries():
        yield path
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for name in sorted([*dirs, *files]):
                    yield Path(root) / name
    for entry in entries():
        require(not entry.is_symlink(), f'storage symlink refused: {entry}')
        stat = entry.stat()
        require(entry.is_dir() or entry.is_file(), f'special storage entry refused: {entry}')
        digest.update(json.dumps([str(entry.relative_to(path) if entry != path else '.'),
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]).encode())
    return digest.hexdigest()


def namespace(path: Path, config: str) -> str:
    return hashlib.sha256(json.dumps([str(path), config], separators=(',', ':')).encode()).hexdigest()


class RawInputs:
    """Small bounded cache of verified row identities; policies remain on disk."""

    def __init__(self, manifest: dict[str, Any], max_rows: int, max_index_bytes: int):
        self.teacher = manifest['teacher']
        require(set(self.teacher) == {'onnx', 'policy_output', 'providers', 'remap'}, 'teacher fields differ')
        onnx = self.teacher['onnx']
        require(set(onnx) == {'path', 'sha256'} and len(onnx['sha256']) == 64, 'teacher ONNX identity required')
        require(bool(self.teacher['providers']) and isinstance(self.teacher['policy_output'], str), 'teacher runtime identity required')
        self.remap = mix.functional_remap_identity(self.teacher['remap'])
        require(self.remap == mix.functional_remap_identity(remap_provenance()), 'unqualified remap implementation')
        self.max_rows = max_rows
        self.max_index_bytes = max_index_bytes
        self.index_bytes = 0
        self.cache_dir: Path | None = None
        self.disk_cache: dict[tuple[str, str], tuple[Path, str]] = {}
        self.sources: dict[str, tuple[raw.SourceSpec, dict[str, dict[str, Any]]]] = {}
        self.pins: list[dict[str, Any]] = []
        self.verified: dict[str, dict[str, Any]] = {}
        self.stable: dict[Path, str] = {}
        self.cache: OrderedDict[tuple[str, str], tuple[Any, np.ndarray]] = OrderedDict()
        for item in manifest['sources']:
            require(set(item) == {'source_dir', 'sidecar_dir', 'manifest', 'receipts'}, 'raw source fields differ')
            source_dir, sidecar_dir = Path(item['source_dir']), Path(item['sidecar_dir'])
            require(source_dir == source_dir.resolve() and sidecar_dir == sidecar_dir.resolve(), 'raw source aliases refused')
            require(str(source_dir) not in self.sources, 'duplicate raw source mapping')
            manifest_path = pin(item['manifest'])
            require(manifest_path == source_dir / corpus.MANIFEST_NAME, 'source manifest path differs')
            launch = corpus.read_launch_manifest(source_dir)
            require(launch['row_schema'] == corpus.ROW_SCHEMA and launch[corpus.KEY_HISTORY_REP_FIX] is True, 'raw history schema differs')
            receipts = raw.read_receipts(pin(item['receipts']))
            paths = tuple(source_dir / name for name in receipts)
            require(all(path.name == name for path, name in zip(paths, receipts)), 'unsafe raw shard name')
            counts = tuple(int(receipt['positions']) for receipt in receipts.values())
            inventory = derive.ProgressInventory(shards=paths, shard_rows=counts, rows_claimed=sum(counts),
                                                 progress_files=(), torn_tail_files=(), unlisted_on_disk=())
            ids = {receipt['source_id'] for receipt in receipts.values()}
            require(len(ids) == 1, 'receipt snapshot must describe one raw source ID')
            spec = raw.SourceSpec(str(next(iter(ids))), source_dir, sidecar_dir, launch,
                                  item['manifest']['sha256'], inventory)
            self.sources[str(source_dir)] = spec, receipts
            self.pins.extend([item['manifest'], item['receipts']])

    def get(self, ref: dict[str, Any]) -> tuple[Any, np.ndarray]:
        key = str(ref['source_dir']), str(ref['source_shard'])
        require(key[0] in self.sources, 'row provenance source has no explicit raw mapping')
        spec, receipts = self.sources[key[0]]
        require(ref['source_config_sha256'] == spec.manifest['config_sha256'] and
                ref['source_namespace'] == namespace(spec.corpus_dir, str(spec.manifest['config_sha256'])),
                'source namespace/config mismatch')
        require(key[1] in receipts and Path(key[1]).name == key[1], 'raw shard absent from pinned closed receipts')
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key in self.disk_cache:
            path, digest = self.disk_cache[key]
            target = spec.out_dir / raw.sidecar_name(key[1])
            require(storage_identity(target) == self.stable[target] and
                    storage_identity(spec.corpus_dir / key[1]) == self.stable[spec.corpus_dir / key[1]],
                    'verified raw source changed before cache reuse')
            require(file_sha256(path) == digest, 'raw identity cache changed')
            records = np.load(path, mmap_mode='r', allow_pickle=False)
            group = zarr.open_group(str(target), mode='r')
            self.cache[key] = group, records
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            return self.cache[key]
        receipt = receipts[key[1]]
        rows = int(receipt['positions'])
        require(0 < rows <= self.max_rows, 'raw shard exceeds bounded identity-cache row limit')
        require(self.index_bytes + rows * provenance.RECORD_DTYPE.itemsize + 4096 <= self.max_index_bytes,
                'raw identity disk-cache budget exhausted')
        pending = raw.PendingShard(spec, spec.corpus_dir / key[1], rows, spec.out_dir / raw.sidecar_name(key[1]))
        before = {path: storage_identity(path) for path in (pending.path, pending.target)}
        attrs = raw.verify_shard(
            pending, onnx_sha256=self.teacher['onnx']['sha256'],
            expected_policy_output=self.teacher['policy_output'], expected_providers=self.teacher['providers'],
            expected_remap=self.remap, batch_size=512,
        )
        require(raw.receipt_from_attrs(attrs, pending.target) == receipt, 'raw receipt differs from verified sidecar')
        # The existing verifier checks original keys, fingerprints, policy digests,
        # legal support and mass. This additional streaming reconstruction banks
        # quantized history keys and worker identity, which raw sidecars omit.
        records = np.zeros(rows, dtype=provenance.RECORD_DTYPE)
        count = 0
        for count, row in enumerate(derive.iter_corpus_rows(pending.path), 1):
            require(count <= rows, 'raw source grew during reconstruction')
            planes, _boards, keys, gids, plies = raw.encode_rows([row], source=spec)
            record = records[count - 1]
            record['input_key'] = keys[0]
            record['stored_input_key'] = np.frombuffer(bytes.fromhex(corpus.input_tensor_key(
                np.asarray(planes[0], dtype=np.float16))), dtype=np.uint8)
            record['game_id'], record['ply'], record['worker_id'] = gids[0], plies[0], int(row['worker_id'])
        require(count == rows, 'raw row count changed during reconstruction')
        require(all(storage_identity(path) == stamp for path, stamp in before.items()), 'raw storage changed during verification')
        self.stable.update(before)
        assert self.cache_dir is not None
        index_path = self.cache_dir / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + '.npy')
        with index_path.open('xb') as stream:
            np.save(stream, records, allow_pickle=False)
        self.index_bytes += index_path.stat().st_size
        self.disk_cache[key] = index_path, file_sha256(index_path)
        group: Any = zarr.open_group(str(pending.target), mode='r')
        self.verified[f'{key[0]}/{key[1]}'] = {'receipt': receipt, 'source_manifest_sha256': spec.manifest_sha256}
        self.cache[key] = group, records
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return group, records


def adapt(manifest_path: Path, *, expected_manifest_sha256: str, out: Path,
          max_raw_rows: int = 100000, max_index_bytes: int = 8 * 1024**3) -> dict[str, Any]:
    manifest_pin = {'path': str(manifest_path.resolve()), 'sha256': expected_manifest_sha256}
    manifest = json.loads(pin(manifest_pin).read_text())
    require(set(manifest) == {'schema', 'derived_summary', 'teacher', 'sources'} and manifest['schema'] == 1,
            'adapter manifest fields/schema differ')
    require(max_raw_rows > 0 and max_index_bytes > 0, 'raw row/cache limits must be positive')
    summary_path = pin(manifest['derived_summary'])
    require(summary_path.name == derive.SUMMARY_NAME, 'expected derived corpus summary')
    source = summary_path.parent
    summary = json.loads(summary_path.read_text())
    require(summary.get('row_provenance', {}).get('path_in_shard') == provenance.FILENAME, 'derivation has no row provenance')
    paths = sorted(source.glob('shard_*.zarr'))
    written = {entry['path']: entry for entry in summary['shards']}
    require(bool(paths) and len(written) == len(summary['shards']) and set(written) == {path.name for path in paths}, 'derived shard inventory differs from summary')
    inputs = RawInputs(manifest, max_raw_rows, max_index_bytes)
    out = out.resolve()
    protected = [source, manifest_path.resolve()] + [Path(value) for value in inputs.sources]
    protected += [spec.out_dir for spec, _receipts in inputs.sources.values()]
    require(all(out != path and out not in path.parents and path not in out.parents for path in protected), 'output overlaps inputs')
    require(not out.exists() and not out.with_name(out.name + '.writing').exists(), 'new output required; no adoption')
    writing = out.with_name(out.name + '.writing')
    writing.mkdir(parents=True)
    inputs.cache_dir = writing / '._raw_identity_cache'
    inputs.cache_dir.mkdir()
    total = 0
    entropy_sum = top1_sum = 0.0
    legal_moves_sum = 0
    derived_stable: dict[Path, str] = {}
    payloads: list[dict[str, Any]] = []
    for path in paths:
        derived_stable[path] = storage_identity(path)
        group: Any = zarr.open_group(str(path), mode='r')
        x = np.asarray(group['x'][:])
        rows = len(x)
        require(rows == written[path.name]['rows'] and rows > 0, 'derived row count differs from summary')
        x_sha = raw.sha_array(x)
        stamp = dict(group.attrs).get('derive_row_provenance')
        if not isinstance(stamp, dict) or stamp != written[path.name].get('row_provenance'):
            raise ValueError('provenance summary/attribute pin mismatch')
        require(stamp['schema'] == provenance.SCHEMA and stamp['rows'] == rows and
                stamp['record_bytes'] == provenance.RECORD_DTYPE.itemsize and stamp['path'] == provenance.FILENAME,
                'provenance stamp format differs')
        provenance_path = path / provenance.FILENAME
        require(file_sha256(provenance_path) == stamp['sha256'], 'corrupted row provenance')
        refs = provenance.read(provenance_path, rows=rows)
        encoding, keys, key_sha, policy_sha = mix._sidecar_identity(group, path, source_x=x)
        mix._source_game_ply_sha(group, path)
        game_ids = np.asarray(group['game_id'][:])
        ply_indices = np.asarray(group['ply_index'][:])
        legal = np.asarray(group['legal_mask'][:]) != 0
        policy = np.empty((rows, mix.COMPACT_POLICY_SIZE), dtype=np.float32)
        seen: set[tuple[str, str, int]] = set()
        grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        for index, ref in enumerate(refs):
            identity = (ref['source_namespace'], ref['source_shard'], ref['source_row'])
            require(identity not in seen, 'duplicate source-qualified derived row')
            seen.add(identity)
            grouped.setdefault((ref['source_dir'], ref['source_shard']), []).append((index, ref))
        for requests in grouped.values():
            first_ref = requests[0][1]
            raw_group, records = inputs.get(first_ref)
            output_rows, raw_rows = [], []
            for index, ref in requests:
                require(ref['source_namespace'] == first_ref['source_namespace'] and
                        ref['source_config_sha256'] == first_ref['source_config_sha256'],
                        'source namespace/config mismatch within raw shard')
                offset = int(ref['source_row'])
                require(0 <= offset < len(records), 'physical source row outside closed shard')
                record = records[offset]
                require(ref['input_key'] == record['input_key'].tobytes().hex(), 'raw full-history key mismatch')
                require(ref['stored_input_key'] == record['stored_input_key'].tobytes().hex() == corpus.input_tensor_key(x[index]),
                        'stored quantized full-history key mismatch')
                require(all(int(ref[field]) == int(record[field]) for field in ('worker_id', 'game_id', 'ply')),
                        'raw physical row/game/ply/worker identity mismatch')
                require(int(game_ids[index]) == ref['game_id'] and int(ply_indices[index]) == ref['ply'],
                        'derived game/ply alignment mismatch')
                output_rows.append(index)
                raw_rows.append(offset)
            # Zarr groups requested rows by chunk, avoiding one decompression per row.
            policy[output_rows] = raw_group[raw.POLICY_FIELD].oindex[np.asarray(raw_rows), :]
        require(np.isfinite(policy).all() and np.all(policy >= 0) and np.all(policy[~legal] == 0)
                and np.allclose(policy.sum(axis=1, dtype=np.float64), 1, atol=2e-6, rtol=0),
                'adapted policy has invalid mass or legal support')
        positive = policy[policy > 0].astype(np.float64)
        entropy_sum -= float(np.sum(positive * np.log(positive)))
        top1_sum += float(policy.max(axis=1).sum(dtype=np.float64))
        legal_moves_sum += int(legal.sum())
        target = writing / path.name
        result: Any = zarr.open_group(str(target), mode='w')
        result.create_dataset(mix.SIDECAR_KEY_FIELD, data=keys, chunks=(min(512, rows), keys.shape[1]), compressor=raw._COMPRESSOR)
        result.create_dataset(mix.SIDECAR_POLICY_FIELD, data=policy, chunks=(min(512, rows), policy.shape[1]), compressor=raw._COMPRESSOR)
        attrs = {'bt4_policy_sidecar_schema': mix.SIDECAR_SCHEMA, 'source_shard': path.name,
                 'source_dir': str(source), 'positions': rows, 'source_key_sha256': key_sha,
                 'source_policy_sha256': policy_sha, 'input_history_encoding': encoding,
                 'policy_encoding': mix.POLICY_ENCODING_LC0_1858, 'policy_size': mix.COMPACT_POLICY_SIZE,
                 'onnx_path': inputs.teacher['onnx']['path'], 'onnx_sha256': inputs.teacher['onnx']['sha256'],
                 'policy_output': inputs.teacher['policy_output'], 'providers': inputs.teacher['providers'],
                 'teacher_evaluations_per_position': 1, 'search_nodes': 0, 'stored_dtype': 'float32',
                 'adapter_manifest_sha256': expected_manifest_sha256, 'row_provenance_sha256': stamp['sha256'],
                 'source_derive_summary_sha256': manifest['derived_summary']['sha256'],
                 'source_stored_x_sha256': x_sha,
                 'teacher_input': 'original verified float32 full history; exact float16 quantization bound to derived row',
                 'bt4_policy_sha256': raw.sha_array(policy)}
        result.attrs.update(attrs)
        mix._validate_sidecar(target, source_path=path, source_keys=keys, source_key_sha=key_sha,
                              source_policy_sha=policy_sha, onnx_sha=inputs.teacher['onnx']['sha256'],
                              providers=inputs.teacher['providers'], policy_output=inputs.teacher['policy_output'])
        require(raw.sha_array(np.asarray(result[mix.SIDECAR_POLICY_FIELD][:])) == attrs['bt4_policy_sha256'],
                'adapted policy write/readback digest differs')
        require(mix._sidecar_identity(zarr.open_group(str(path), mode='r'), path)[2:] == (key_sha, policy_sha)
                and raw.sha_array(np.asarray(group['x'][:])) == x_sha
                and file_sha256(provenance_path) == stamp['sha256'], 'derived input changed during adaptation')
        payloads.append({'path': path.name, 'rows': rows, 'bt4_policy_sha256': attrs['bt4_policy_sha256'],
                         'source_key_sha256': key_sha, 'source_policy_sha256': policy_sha,
                         'row_provenance_sha256': stamp['sha256']})
        total += rows
    require(total == summary['realized']['rows_written'], 'derived total differs from summary')
    require(all(storage_identity(path) == stamp for path, stamp in (inputs.stable | derived_stable).items()),
            'verified source storage changed before publication')
    for item in [manifest_pin, manifest['derived_summary'], *inputs.pins]:
        pin(item)
    final = {'schema': mix.SIDECAR_SCHEMA, 'kind': 'bt4_raw_legal_policy_sidecar', 'source_dir': str(source),
             'source_shards': len(paths), 'sidecar_shards': len(paths), 'rows': total,
             'policy_encoding': mix.POLICY_ENCODING_LC0_1858, 'onnx': inputs.teacher['onnx'],
             'policy_output': inputs.teacher['policy_output'], 'providers': inputs.teacher['providers'],
             'teacher_evaluations_per_position': 1, 'search_nodes': 0, 'stored_dtype': 'float32',
             'mean_entropy_nats': entropy_sum / total, 'mean_top1_probability': top1_sum / total,
             'mean_legal_moves': legal_moves_sum / total,
             'remap': inputs.teacher['remap'], 'adapter': {'manifest': manifest_pin,
                 'script_sha256': file_sha256(__file__), 'derived_summary': manifest['derived_summary'],
                 'teacher_input': 'original float32 full history, not reinference of quantized replay',
                 'verified_raw_shards': inputs.verified, 'written_shards': payloads,
                 'identity_cache_bytes': inputs.index_bytes, 'identity_cache_limit_bytes': max_index_bytes,
                 'raw_reconstruction': 'one existing verification plus one quantized-key reconstruction per used raw shard'},
             'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    inputs.cache.clear()
    shutil.rmtree(inputs.cache_dir)
    mix._atomic_json(writing / mix.SIDECAR_SUMMARY, final)
    os.replace(writing, out)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-raw-rows', type=int, default=100000)
    parser.add_argument('--max-index-bytes', type=int, default=8 * 1024**3,
                        help='cap temporary disk identity cache (56 bytes/raw row plus headers)')
    args = parser.parse_args(argv)
    adapt(args.manifest, expected_manifest_sha256=args.expected_manifest_sha256,
          out=args.out, max_raw_rows=args.max_raw_rows, max_index_bytes=args.max_index_bytes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
