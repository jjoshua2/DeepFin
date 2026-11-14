"""Bounded CPU preparation only. Creates immutable ZIPs and staged paired roots."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

GIB = 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(cohorts, reused):
    records = []
    for i, cohort in enumerate(cohorts):
        root = Path(cohort['base'])
        paths = sorted(root.glob('shard_*.zarr'))
        if i == 0:
            paths = [root / path.name.removesuffix('.zip')
                     for path in sorted(reused.glob('shard_*.zarr.zip'))]
            if len(paths) != 32:
                raise ValueError('expected exact existing 32-shard bank')
        else:
            count = 7 if i <= 20 else 6
            # Deterministic spread over each source, rather than the first shards.
            paths = [paths[j * (len(paths) - 1) // (count - 1)] for j in range(count)]
        for path in paths:
            rows = json.loads((path / 'x/.zarray').read_text())['shape'][0]
            if rows <= 0:
                raise ValueError(f'empty selected shard: {path}')
            records.append({'cohort': i, 'source': str(path.resolve()), 'rows': rows,
                            'reused': str(reused / (path.name + '.zip')) if i == 0 else None})
    if len(cohorts) != 35 or len(records) != 256:
        raise ValueError('requires exactly 35 cohorts / 256 unique shards')
    if len({r['source'] for r in records}) != 256:
        raise ValueError('duplicate source shard')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort-plan', type=Path, required=True)
    p.add_argument('--cohort-plan-sha256', required=True)
    p.add_argument('--selection-plan', type=Path, required=True)
    p.add_argument('--selection-plan-sha256', required=True)
    p.add_argument('--reuse', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--external', type=Path, required=True)
    p.add_argument('--prepare', action='store_true', help='otherwise metadata inventory only')
    a = p.parse_args()
    if sha(a.cohort_plan) != a.cohort_plan_sha256:
        raise ValueError('cohort plan pin mismatch')
    start = time.monotonic()
    os.sched_setaffinity(0, {12, 13})
    os.nice(19)
    records = select(json.loads(a.cohort_plan.read_text())['cohorts'], a.reuse)
    if sha(a.selection_plan) != a.selection_plan_sha256:
        raise ValueError('selection plan pin mismatch')
    frozen = json.loads(a.selection_plan.read_text())
    if records != frozen['records'] or str(a.out) != frozen['output'] or str(a.external) != frozen['external']:
        raise ValueError('source selection or output differs from immutable plan')
    a.out.mkdir(parents=True, exist_ok=False)
    budget = 0

    def guard():
        if time.monotonic() - start >= 1200 or (a.out / 'STOP').exists():
            raise RuntimeError('CPU preparation deadline or STOP')
        for path in (a.out, a.external.parent):
            if shutil.disk_usage(path).free < 150 * GIB:
                raise RuntimeError('150GiB disk floor')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 40 * GIB:
            raise RuntimeError('40GiB available RAM floor')
        if budget > 10 * GIB:
            raise RuntimeError('10GiB new disk cap')

    receipt = {'status': 'METADATA_ONLY', 'cohort_plan_sha256': a.cohort_plan_sha256,
               'records': records, 'rows': sum(r['rows'] for r in records)}
    try:
        guard()
        if a.prepare:
            a.external.mkdir(parents=True, exist_ok=False)
            for label in ('directory', 'packed'):
                (a.out / label).mkdir()
            for index, record in enumerate(records):
                guard()
                source = Path(record['source'])
                target = Path(record['reused']) if record['reused'] else (
                    a.external / f"cohort{record['cohort']:02d}" / (source.name + '.zip'))
                members = sorted(path for path in source.rglob('*') if path.is_file())
                if not record['reused']:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED) as archive:
                        for member in members:
                            guard()
                            budget += member.stat().st_size + 256
                            guard()
                            archive.write(member, member.relative_to(source).as_posix())
                # Compare the existing compressed Zarr bytes, without decoding arrays.
                with zipfile.ZipFile(target) as archive:
                    expected = [m.relative_to(source).as_posix() for m in members]
                    if sorted(archive.namelist()) != expected:
                        raise ValueError('ZIP member roster mismatch')
                    hashes = {}
                    for member, name in zip(members, expected):
                        guard()
                        if archive.getinfo(name).compress_type != zipfile.ZIP_STORED:
                            raise ValueError('unexpected outer compression')
                        digest = sha(member)
                        if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                            raise ValueError(f'ZIP/source mismatch: {name}')
                        hashes[name] = digest
                record.update(zip=str(target.resolve()), zip_sha256=sha(target), members=hashes)
                (a.out / 'directory' / f'shard_{index:06d}.zarr').symlink_to(source)
                (a.out / 'packed' / f'shard_{index:06d}.zarr.zip').symlink_to(target)
            receipt['status'] = 'PASS_BYTES_REQUIRES_FULL_STREAM_QUALIFICATION'
    except BaseException as error:
        receipt.update(status='INCOMPLETE', error=repr(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - start, new_bytes_upper_bound=budget)
        (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
