"""CPU-only paired-wave segment prototype with literal replay UIDs.

This accepts already replayed source-qualified rows enriched by the checked
whole-game source proof. The current child row stream does not carry that
proof/Syzygy pair; checkpointed_cursor_v2 supplies the bounded diagnostic
reader interface. No production child restart, target join or trainer arrays
are established here.

The per-512-row wall/RSS checks are cooperative. A production caller must run
each segment in an owned subprocess with a <=600-second kill/restart watchdog
and an archive/game cursor that can resume after the last sealed segment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import sys
import time
from collections.abc import Iterable, Iterator

import zstandard as zstd


DOMAIN = (b'mixed500m/student-x/v1;history=lc0_root_legacy_meta;'
          b'extras=v2_threats;repfix=1;shape=175x8x8;'
          b'pack=<f2,C;model=<f4,C\n')
CONTEXT_KEYS = ('history_stack_sha256', 'repetition', 'rule50',
                'legal_context_sha256', 'teacher_query_sha256')
SEGMENT_ROWS = 8192
FRAME_ROWS = 512
MAX_META = 4096
ROW_INDEX = struct.Struct('<QII')  # metadata byte offset, size, local prefix ID
FRAME_INDEX = struct.Struct('<QII32s32s')  # zstd offset,size,rows,compressed/plain SHA


class Hold(ValueError):
    pass


def need(ok: bool, why: str) -> None:
    if not ok:
        raise Hold(why)


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(',', ':')) + '\n').encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, raw: bytes) -> None:
    stage = path.with_name(path.name + '.part')
    need(not path.exists(), f'output exists: {path.name}')
    if stage.exists():
        stage.unlink()  # owned unsealed scratch after an interrupted attempt
    with stage.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stage, path)
    fsync_directory(path.parent)


def _hex64(value: object) -> bool:
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _note(row: dict, data: bytes, ordinal: int,
          roster: dict[tuple[str, str], str], syzygy_sha256: str) -> dict:
    uid = row.get('uid')
    if type(uid) is not list:
        raise Hold('source-qualified UID')
    need(type(uid) is list and len(uid) == 5 and
         all(type(x) is str and x and len(x.encode('utf-8')) <= 4096
             for x in uid[:3]) and
         all(type(x) is int and 0 <= x <= (1 << 32) - 1
             for x in uid[3:]), 'source-qualified UID')
    need(row.get('source') == roster.get((str(uid[0]), str(uid[1]))) and
         row.get('source') in ('BT4-v9', 'Ceres-v8', 'SF-d6') and
         _hex64(uid[0]) and
         _hex64(row.get('provenance_locator_sha256'))
         and _hex64(row.get('game_proof_sha256')) and
         row.get('syzygy_inventory_sha256') == syzygy_sha256,
         'source/proof/Syzygy identity')
    context = [row.get(key) for key in CONTEXT_KEYS]
    need(all(type(value) is str and value.strip() and
             value.strip().lower() not in {'unknown', 'none', 'null', 'n/a', 'missing'}
             for value in (context[0], context[1], context[3], context[4]))
         and type(context[2]) is int and context[2] >= 0,
         'complete canonical context')
    need(row.get('outcome') in ('1-0', '0-1', '1/2-1/2'), 'strict outcome')
    digest = sha(DOMAIN + data)
    need(row.get('input_digest') in (None, digest), 'input digest')
    return {'ordinal': ordinal, 'uid': uid, 'source': row['source'],
            'provenance_locator_sha256': row['provenance_locator_sha256'],
            'game_proof_sha256': row['game_proof_sha256'],
            'syzygy_inventory_sha256': row['syzygy_inventory_sha256'],
            'input_digest': digest, 'input_bytes_sha256': sha(data),
            'context': context, 'outcome': row['outcome']}


class SegmentStore:
    def __init__(self, root: Path, *, input_sha256: str, source_sha256: str,
                 config_sha256: str,
                 roster: tuple[tuple[str, str, str], ...],
                 strict_receipt_sha256: str, syzygy_inventory_sha256: str,
                 route_code_sha256: str, row_bytes: int = 44800,
                 segment_rows: int = SEGMENT_ROWS, frame_rows: int = FRAME_ROWS,
                 zstd_level: int = 5, wall_cap_seconds: int = 600,
                 rss_cap_bytes: int = 1 << 30,
                 output_cap_bytes: int = 512 << 20) -> None:
        need(sys.flags.optimize == 0, 'optimized Python mode forbidden')
        need(all(_hex64(x) for x in (input_sha256, source_sha256,
                                   config_sha256, strict_receipt_sha256,
                                   syzygy_inventory_sha256,
                                   route_code_sha256)),
             'source/config pins')
        need(type(roster) is tuple and 0 < len(roster) <= 4096 and
             all(type(item) is tuple and len(item) == 3 and
                 _hex64(item[0]) and type(item[1]) is str and item[1] and
                 item[2] in ('BT4-v9', 'Ceres-v8', 'SF-d6')
                 for item in roster) and
             len({(item[0], item[1]) for item in roster}) == len(roster),
             'exact source roster')
        self.roster = {(item[0], item[1]): item[2] for item in roster}
        self.syzygy_sha256 = syzygy_inventory_sha256
        need(0 < row_bytes <= 44800 and segment_rows == SEGMENT_ROWS
             and frame_rows == FRAME_ROWS and segment_rows % frame_rows == 0
             and 1 <= zstd_level <= 10 and 0 < wall_cap_seconds <= 600
             and 0 < rss_cap_bytes <= (4 << 30)
             and 0 < output_cap_bytes <= (512 << 20), 'bounded segment recipe')
        self.root = Path(root)
        self.row_bytes = row_bytes
        self.segment_rows = segment_rows
        self.frame_rows = frame_rows
        self.level = zstd_level
        self.wall_cap = wall_cap_seconds
        self.rss_cap = rss_cap_bytes
        self.output_cap = output_cap_bytes
        self.claim = {'schema': 'tri_paired_wave_segment_claim_v2',
                      'credit': 'SYNTHETIC_ZERO_CORPUS_AND_TARGET_CREDIT',
                      'input_sha256': input_sha256,
                      'source_sha256': source_sha256,
                      'config_sha256': config_sha256,
                      'producer_sha256': file_sha(Path(__file__)),
                      'roster': [list(item) for item in roster],
                      'strict_receipt_sha256': strict_receipt_sha256,
                      'syzygy_inventory_sha256': syzygy_inventory_sha256,
                      'route_code_sha256': route_code_sha256,
                      'row_bytes': row_bytes, 'segment_rows': segment_rows,
                      'frame_rows': frame_rows, 'zstd_level': zstd_level,
                      'wall_cap_seconds': wall_cap_seconds,
                      'rss_cap_bytes': rss_cap_bytes,
                      'output_cap_bytes': output_cap_bytes}
        self.claim_raw = canonical(self.claim)
        self.root.mkdir(parents=True, exist_ok=True)
        claim_path = self.root / 'CLAIM.json'
        if claim_path.exists():
            need(claim_path.read_bytes() == self.claim_raw, 'claim/source/config changed')
        else:
            atomic_write(claim_path, self.claim_raw)

    def _path(self, number: int) -> Path:
        need(type(number) is int and number >= 0, 'segment number')
        return self.root / f'segment_{number:08d}'

    def _budget(self, start: float, size: int) -> None:
        need(time.monotonic() - start <= self.wall_cap, 'segment wall cap')
        # Linux ru_maxrss is KiB; do not substitute /proc/self/io counters.
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
             <= self.rss_cap, 'segment RSS cap')
        need(size <= self.output_cap, 'segment output cap')

    def verify_segment(self, number: int) -> dict:
        directory = self._path(number)
        receipt_path = directory / 'RECEIPT.json'
        need(receipt_path.is_file(), 'sealed segment receipt absent')
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        need(raw == canonical(receipt) and
             receipt['schema'] == 'tri_paired_wave_segment_v2' and
             receipt['segment'] == number and
             receipt['claim_sha256'] == sha(self.claim_raw) and
             receipt['start_ordinal'] == number * self.segment_rows and
             receipt['rows'] > 0 and receipt['rows'] <= self.segment_rows and
             receipt['frames'] ==
                 (receipt['rows'] + self.frame_rows - 1) // self.frame_rows and
             receipt['native_plain_bytes'] == receipt['rows'] * self.row_bytes,
             'sealed receipt identity')
        need(set(receipt['sha256']) == {'DATA.zst', 'FRAMES.bin', 'ROWS.bin',
                                      'META.jsonl', 'PREFIXES.json'},
             'sealed file membership')
        for name, digest in receipt['sha256'].items():
            need(file_sha(directory / name) == digest, f'sealed file SHA: {name}')
        need((directory / 'DATA.zst').stat().st_size ==
             receipt['compressed_data_bytes'] and
             sum((directory / name).stat().st_size for name in receipt['sha256'])
             == receipt['retained_data_index_meta_bytes'], 'receipt byte accounting')
        need((directory / 'ROWS.bin').stat().st_size ==
             receipt['rows'] * ROW_INDEX.size, 'row-index length')
        need((directory / 'FRAMES.bin').stat().st_size ==
             receipt['frames'] * FRAME_INDEX.size, 'frame-index length')
        need({item.name for item in directory.iterdir()} ==
             {'DATA.zst', 'FRAMES.bin', 'ROWS.bin', 'META.jsonl',
              'PREFIXES.json', 'RECEIPT.json'} and
             receipt['retained_data_index_meta_bytes'] <=
                 self.output_cap, 'segment membership/output cap')
        prefixes_raw = (directory / 'PREFIXES.json').read_bytes()
        prefixes = json.loads(prefixes_raw)
        need(prefixes_raw == canonical(prefixes) and type(prefixes) is list and
             all(type(item) is list and len(item) == 3 for item in prefixes),
             'local prefix table')
        return receipt

    def seal_wave1(self, number: int, rows: Iterable[tuple[dict, bytes]],
                   *, expected_rows: int) -> dict:
        need(0 < expected_rows <= self.segment_rows, 'bounded segment count')
        existing = self._path(number)
        if existing.exists():
            receipt = self.verify_segment(number)
            need(receipt['rows'] == expected_rows, 'resume segment row count')
            return {'status': 'SKIPPED_VERIFIED', **receipt}
        if number:
            need(self._path(number - 1).is_dir(), 'noncontiguous segment')
        stage = self.root / f'.segment_{number:08d}.part'
        if stage.exists():
            shutil.rmtree(stage)  # deterministic, owned unsealed scratch only
        stage.mkdir()
        start = time.monotonic()
        prefixes: list[list[str]] = []
        prefix_ids: dict[tuple[str, str, str], int] = {}
        frame_count = count = plain_bytes = compressed_bytes = 0
        frame = bytearray()
        compressor = zstd.ZstdCompressor(level=self.level)
        with ((stage / 'DATA.zst').open('xb') as payload,
              (stage / 'FRAMES.bin').open('xb') as frames,
              (stage / 'ROWS.bin').open('xb') as index,
              (stage / 'META.jsonl').open('xb') as meta):
            def flush_frame() -> None:
                nonlocal frame_count, plain_bytes, compressed_bytes
                if not frame:
                    return
                raw = bytes(frame)
                encoded = compressor.compress(raw)
                nrows = len(raw) // self.row_bytes
                frames.write(FRAME_INDEX.pack(compressed_bytes, len(encoded),
                                             nrows,
                                             hashlib.sha256(encoded).digest(),
                                             hashlib.sha256(raw).digest()))
                payload.write(encoded)
                compressed_bytes += len(encoded)
                plain_bytes += len(raw)
                frame_count += 1
                frame.clear()

            for row, data in rows:
                need(count < expected_rows and type(data) is bytes and
                     len(data) == self.row_bytes, 'row length/count')
                ordinal = number * self.segment_rows + count
                note = _note(row, data, ordinal, self.roster,
                             self.syzygy_sha256)
                prefix = tuple(note['uid'][:3])
                if prefix not in prefix_ids:
                    prefix_ids[prefix] = len(prefixes)
                    prefixes.append(list(prefix))
                prefix_id = prefix_ids[prefix]
                compact = dict(note)
                compact.pop('uid')
                compact.pop('source')
                compact['game_id'] = note['uid'][3]
                compact['ply_index'] = note['uid'][4]
                encoded_meta = canonical(compact)
                need(0 < len(encoded_meta) <= MAX_META, 'bounded row metadata')
                index.write(ROW_INDEX.pack(meta.tell(), len(encoded_meta), prefix_id))
                meta.write(encoded_meta)
                frame.extend(data)
                count += 1
                if count % self.frame_rows == 0:
                    flush_frame()
                if count % 512 == 0:
                    self._budget(start, payload.tell() + frames.tell() +
                                 index.tell() + meta.tell())
            need(count == expected_rows, 'segment source truncated')
            flush_frame()
            for stream in (payload, frames, index, meta):
                stream.flush()
                os.fsync(stream.fileno())
        atomic_write(stage / 'PREFIXES.json', canonical(prefixes))
        names = ('DATA.zst', 'FRAMES.bin', 'ROWS.bin', 'META.jsonl',
                 'PREFIXES.json')
        size = sum((stage / name).stat().st_size for name in names)
        self._budget(start, size)
        receipt = {'schema': 'tri_paired_wave_segment_v2',
                   'claim_sha256': sha(self.claim_raw), 'segment': number,
                   'start_ordinal': number * self.segment_rows,
                   'rows': count, 'frames': frame_count,
                   'native_plain_bytes': plain_bytes,
                   'compressed_data_bytes': compressed_bytes,
                   'retained_data_index_meta_bytes': size,
                   'sha256': {name: file_sha(stage / name) for name in names},
                   'credit': 'SYNTHETIC_ZERO_CORPUS_AND_TARGET_CREDIT'}
        atomic_write(stage / 'RECEIPT.json', canonical(receipt))
        fsync_directory(stage)
        os.replace(stage, existing)
        fsync_directory(self.root)
        return {'status': 'BUILT', **receipt}

    def iter_rows(self, number: int) -> Iterator[tuple[dict, bytes]]:
        receipt = self.verify_segment(number)
        directory = self._path(number)
        prefixes = json.loads((directory / 'PREFIXES.json').read_bytes())
        index = (directory / 'ROWS.bin').read_bytes()
        frame_index = (directory / 'FRAMES.bin').read_bytes()
        meta = (directory / 'META.jsonl').read_bytes()
        decompressor = zstd.ZstdDecompressor(
            max_window_size=self.frame_rows * self.row_bytes)
        with (directory / 'DATA.zst').open('rb') as payload:
            row = 0
            compressed_end = 0
            meta_end = 0
            for i in range(receipt['frames']):
                offset, length, nrows, encoded_hash, plain_hash = FRAME_INDEX.unpack_from(
                    frame_index, i * FRAME_INDEX.size)
                need(offset == compressed_end and 0 < nrows <= self.frame_rows,
                     'frame offsets/count')
                payload.seek(offset)
                encoded = payload.read(length)
                need(len(encoded) == length and
                     hashlib.sha256(encoded).digest() == encoded_hash,
                     'compressed frame hash')
                plain = decompressor.decompress(
                    encoded, max_output_size=nrows * self.row_bytes,
                    allow_extra_data=False)
                need(len(plain) == nrows * self.row_bytes and
                     hashlib.sha256(plain).digest() == plain_hash,
                     'native frame hash')
                compressed_end += length
                for j in range(nrows):
                    pos, size, prefix_id = ROW_INDEX.unpack_from(
                        index, row * ROW_INDEX.size)
                    need(prefix_id < len(prefixes) and pos == meta_end and
                         0 < size <= MAX_META and pos + size <= len(meta),
                         'row metadata index')
                    meta_end = pos + size
                    raw_meta = meta[pos:pos + size]
                    compact = json.loads(raw_meta)
                    need(raw_meta == canonical(compact), 'row metadata canonical')
                    uid = prefixes[prefix_id] + [compact.pop('game_id'),
                                                  compact.pop('ply_index')]
                    need((uid[0], uid[1]) in self.roster,
                         'row namespace missing from exact roster')
                    note = {**compact, 'uid': uid,
                            'source': self.roster[(uid[0], uid[1])]}
                    native = plain[j * self.row_bytes:(j + 1) * self.row_bytes]
                    need(note == _note({**note,
                                        **dict(zip(CONTEXT_KEYS, note['context']))},
                                       native, number * self.segment_rows + row,
                                       self.roster, self.syzygy_sha256),
                         'row native/context proof')
                    yield note, native
                    row += 1
            need(row == receipt['rows'] and meta_end == len(meta) and compressed_end ==
                 (directory / 'DATA.zst').stat().st_size, 'frame coverage')

    def verify_comparison(self, number: int) -> dict:
        first = self.verify_segment(number)
        path = self.root / f'compare_{number:08d}.json'
        need(path.is_file(), 'sealed comparison absent')
        raw = path.read_bytes()
        try:
            receipt = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise Hold('comparison JSON invalid') from error
        need(type(receipt) is dict and raw == canonical(receipt) and
             set(receipt) == {'schema', 'claim_sha256', 'segment',
                              'segment_receipt_sha256', 'rows',
                              'paired_rows_sha256', 'credit'} and
             receipt['schema'] == 'tri_paired_wave_comparison_v2' and
             receipt['claim_sha256'] == sha(self.claim_raw) and
             receipt['segment'] == number and
             receipt['segment_receipt_sha256'] ==
                 file_sha(self._path(number) / 'RECEIPT.json') and
             receipt['rows'] == first['rows'] and
             _hex64(receipt['paired_rows_sha256']) and
             receipt['credit'] == 'SYNTHETIC_ZERO_CORPUS_AND_TARGET_CREDIT',
             'sealed comparison identity')
        return receipt

    def compare_wave2(self, number: int, rows: Iterable[tuple[dict, bytes]]) -> dict:
        first = self.verify_segment(number)
        path = self.root / f'compare_{number:08d}.json'
        if path.exists():
            receipt = self.verify_comparison(number)
            return {'status': 'SKIPPED_VERIFIED', **receipt}
        start = time.monotonic()
        count = 0
        digest = hashlib.sha256()
        iterator = iter(rows)
        for expected, native in self.iter_rows(number):
            try:
                row, current = next(iterator)
            except StopIteration as error:
                raise Hold('wave 2 truncated') from error
            need(type(current) is bytes and len(current) == self.row_bytes,
                 'wave 2 row bytes')
            note = _note(row, current, number * self.segment_rows + count,
                         self.roster, self.syzygy_sha256)
            need(note == expected and current == native,
                 f'wave 2 byte/order/metadata mismatch at {count}')
            digest.update(canonical(note))
            digest.update(current)
            count += 1
            if count % 512 == 0:
                self._budget(start, 0)
        need(next(iterator, None) is None and count == first['rows'],
             'wave 2 trailing/count')
        receipt = {'schema': 'tri_paired_wave_comparison_v2',
                   'claim_sha256': sha(self.claim_raw), 'segment': number,
                   'segment_receipt_sha256':
                       file_sha(self._path(number) / 'RECEIPT.json'),
                   'rows': count, 'paired_rows_sha256': digest.hexdigest(),
                   'credit': 'SYNTHETIC_ZERO_CORPUS_AND_TARGET_CREDIT'}
        atomic_write(path, canonical(receipt))
        return {'status': 'BUILT', **receipt}
