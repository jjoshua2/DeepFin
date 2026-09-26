#!/usr/bin/env python3
"""Fixed salvage COPY+VERIFY operation. No source deletion; default is plan-only.

Plan: {schema_version:1,mount:{target,source,fstype},pools:[{name,
 estimated_tar_bytes}]}. Descriptive extra keys are permitted. No adoption of
 previous attempts: an existing per-pool staging/receipt/archive refuses work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tarfile
import time

ROOT = Path('/home/josh/projects/chess/scratchpad/storage_audit_20260906')
SOURCE = Path('/home/josh/projects/chess/data/salvage')
DEST = Path('/mnt/e/chess_archive_20260906/salvage')
EXCLUDED = {'DRIFT_20260728_postC17_iter192', 'bt4heads_iter100_20260815',
            'rolling', 'pre_audit_deploy_20260726'}
GIB = 1024**3
RESERVE = 150 * GIB
RATE = 32 * 1024**2
CHUNK = 1024**2
CHILD = None


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def sha(path):
    with path.open('rb') as handle:
        return hash_stream(handle)


class Pace:
    def __init__(self):
        self.start = time.monotonic()
        self.size = 0

    def add(self, count):
        self.size += count
        delay = self.size / RATE - (time.monotonic() - self.start)
        if delay > 0:
            time.sleep(min(delay, 1))


def hash_stream(handle):
    digest = hashlib.sha256()
    pace = Pace()
    while True:
        block = handle.read(CHUNK)
        if not block:
            return digest.hexdigest()
        digest.update(block)
        pace.add(len(block))


def stamp(path):
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)):
        raise ValueError(f'link/special file refused: {path}')
    if stat.S_ISREG(value.st_mode) and value.st_nlink != 1:
        raise ValueError(f'hard-linked file requires separate review: {path}')
    return {key: getattr(value, key) for key in
            ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def assert_stat(path, expected):
    if stamp(path) != expected:
        raise ValueError(f'source changed: {path}')


def walk(path):
    yield path
    with os.scandir(path) as entries:
        for entry in entries:
            child = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                yield from walk(child)
            else:
                yield child


def mount_check(plan):
    result = subprocess.run(['findmnt', '-J', '-T', '/mnt/e', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'],
                            check=True, capture_output=True, text=True, timeout=15)
    rows = json.loads(result.stdout)['filesystems']
    expected = plan['mount']
    if len(rows) != 1 or any(rows[0].get(k) != expected[k] for k in ('target', 'source', 'fstype')):
        raise ValueError(f'archive mount identity changed: {rows}')
    if rows[0]['target'] != '/mnt/e' or rows[0]['fstype'] != '9p' or 'rw' not in rows[0]['options'].split(','):
        raise ValueError('expected writable E: 9p mount is absent')
    return rows[0]


def resources(estimate=0):
    free = shutil.disk_usage(ROOT).free
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if free < RESERVE + estimate or available < 16 * GIB:
        raise RuntimeError(f'resource guard: free={free}, staging_reservation={estimate}, RAM={available}')
    return {'local_free_bytes': free, 'available_RAM_bytes': available}


def references(pool):
    """Report observed consumers and visibility limits for this copy-only job."""
    found, uninspected, inaccessible_same_user = [], [], []
    prefix = str(pool.resolve())
    def inside(target):
        target = target.removesuffix(' (deleted)')
        return target == prefix or target.startswith(prefix + '/')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                uninspected.append(int(process.name))
                continue
            targets = [process / 'cwd']
            targets.extend((process / 'fd').iterdir())
            for target in targets:
                try:
                    destination = os.readlink(target)
                except FileNotFoundError:
                    continue
                if inside(destination):
                    found.append({'pid': int(process.name), 'reference': str(target), 'target': destination})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A non-dumpable ssh-agent can have a same-user process directory
            # but root-owned fd entries. Copying retains the entire source and
            # independently checks its content/metadata stability; incomplete
            # process visibility is disclosed, never treated as deletion proof.
            inaccessible_same_user.append(int(process.name))
    if found:
        raise RuntimeError(f'active source references: {found}')
    return {'observed_source_references': [],
            'other_user_processes_not_inspected': uninspected,
            'same_user_processes_not_inspected': inaccessible_same_user,
            'complete_process_visibility': not (uninspected or inaccessible_same_user),
            'scope': 'Copy only; source retained and stability independently checked. Not source-removal qualification.'}


def inventory(pool, manifest, names, estimate):
    total, count = 0, 0
    device = pool.stat().st_dev
    pace = Pace()
    with manifest.open('x') as records, names.open('xb') as listing:
        for path in walk(pool):
            before = stamp(path)
            if before['st_dev'] != device:
                raise ValueError(f'source crosses filesystem: {path}')
            relative = str(path.relative_to(SOURCE))
            record = {'name': relative, 'stat': before, 'sha256': None}
            if stat.S_ISREG(before['st_mode']):
                total += before['st_size']
                if total + (count + 1) * 2048 > estimate:
                    raise RuntimeError('pool exceeds reviewed staging estimate')
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                digest = hashlib.sha256()
                with os.fdopen(descriptor, 'rb') as handle:
                    if os.fstat(handle.fileno()).st_ino != before['st_ino']:
                        raise ValueError('source identity changed on open')
                    while True:
                        block = handle.read(CHUNK)
                        if not block:
                            break
                        digest.update(block)
                        pace.add(len(block))
                record['sha256'] = digest.hexdigest()
            assert_stat(path, before)
            records.write(json.dumps(record, separators=(',', ':')) + '\n')
            listing.write(os.fsencode(relative) + b'\0')
            count += 1
            if count % 1024 == 0:
                resources()
    return {'entries': count, 'logical_file_bytes': total}


def check_source(manifest):
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            assert_stat(SOURCE / record['name'], record['stat'])


def create_tar(names, archive, log, estimate, plan):
    global CHILD
    command = ['tar', '--create', '--file=-', '--format=pax', '--numeric-owner', '--acls', '--xattrs',
               '--no-recursion', '--verbatim-files-from', '--null', '-C', str(SOURCE), '--files-from', str(names)]
    pace = Pace()
    written = 0
    checked = 0.0
    with log.open('xb') as errors, archive.open('xb') as output:
        CHILD = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
        try:
            while True:
                block = CHILD.stdout.read(CHUNK)
                if not block:
                    break
                if written + len(block) > estimate:
                    raise RuntimeError('tar exceeds reviewed staging estimate')
                resources()
                if time.monotonic() - checked >= 5:
                    mount_check(plan)
                    checked = time.monotonic()
                output.write(block)
                written += len(block)
                pace.add(len(block))
            code = CHILD.wait()
            if code:
                raise RuntimeError(f'tar exited {code}; see {log}')
            output.flush()
            os.fsync(output.fileno())
        finally:
            if CHILD.poll() is None:
                os.killpg(CHILD.pid, signal.SIGTERM)
                CHILD.wait(timeout=30)
            CHILD.stdout.close()
            CHILD = None
    return written


def verify_tar(archive, manifest):
    """Sequential independent member-content verification; no full archive in RAM."""
    count = 0
    with manifest.open() as records, tarfile.open(archive, mode='r|') as source:
        for member in source:
            line = records.readline()
            if not line:
                raise ValueError('unexpected archive member')
            record = json.loads(line)
            info = record['stat']
            if member.name.rstrip('/') != record['name'] or not (member.isdir() or member.isfile()):
                raise ValueError(f'archive member name/type mismatch: {member.name}')
            if member.isdir() != stat.S_ISDIR(info['st_mode']):
                raise ValueError('archive member type changed')
            if (member.mode != stat.S_IMODE(info['st_mode']) or member.uid != info['st_uid'] or member.gid != info['st_gid']
                    or abs(member.mtime - info['st_mtime_ns'] / 1e9) > 1e-6):
                raise ValueError(f'archive metadata mismatch: {member.name}')
            if member.isfile():
                if member.size != info['st_size']:
                    raise ValueError('archive member size mismatch')
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError('archive regular file has no stream')
                with extracted:
                    if hash_stream(extracted) != record['sha256']:
                        raise ValueError(f'archive content mismatch: {member.name}')
            assert_stat(SOURCE / record['name'], info)
            source.members.clear()
            count += 1
        if records.readline():
            raise ValueError('archive is missing source members')
    return count


def terminate(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise RuntimeError(f'archive driver interrupted by signal {signum}')


def run_pool(item, plan, plan_sha):
    global CHILD
    name, estimate = item['name'], item['estimated_tar_bytes']
    pool = SOURCE / name
    attempt = ROOT / 'staging' / name
    archive_destination = DEST / f'{name}.tar'
    if attempt.exists() or archive_destination.exists():
        raise FileExistsError(f'prior attempt/archive exists for {name}; no automatic adoption')
    if SOURCE.resolve() != SOURCE or pool.resolve() != pool:
        raise ValueError('source root/pool contains symlink components')
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        target = line.split()[4]
        if target == str(pool) or target.startswith(str(pool) + '/'):
            raise ValueError(f'nested source mount refused: {target}')
    source_stat = stamp(pool)
    if not stat.S_ISDIR(source_stat['st_mode']):
        raise ValueError('pool is not a directory')
    mount = mount_check(plan)
    budget = resources(estimate + estimate // 10)
    if shutil.disk_usage('/mnt/e').free < estimate * 2:
        raise RuntimeError('archive destination lacks two estimated copies of free space')
    consumers = references(pool)
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = {'pool': name, 'source': str(pool), 'source_deleted': False, 'plan_sha256': plan_sha,
               'started_unix': time.time(), 'mount': mount, 'resources': budget, 'references': consumers}
    save(attempt / 'started.json', receipt)
    try:
        manifest, names, archive = attempt / 'source.jsonl', attempt / 'names.nul', attempt / f'{name}.tar'
        receipt['inventory'] = inventory(pool, manifest, names, estimate)
        print(f'{name}: source inventory complete', flush=True)
        receipt['tar_bytes'] = create_tar(names, archive, attempt / 'tar.log', estimate, plan)
        receipt['verified_members'] = verify_tar(archive, manifest)
        receipt['tar_sha256'] = sha(archive)
        receipt['source_manifest_sha256'] = sha(manifest)
        check_source(manifest)
        mount_check(plan)
        DEST.mkdir(parents=True, exist_ok=True)
        if DEST.resolve() != DEST or archive_destination.exists():
            raise ValueError('archive destination changed or already exists')
        partial = attempt.name + '.partial'
        receipt['rsync_command'] = ['rsync', '--no-whole-file', '--partial-dir=.' + partial,
                                     '--bwlimit=16384', '--fsync', '--', str(archive), str(archive_destination)]
        with (attempt / 'rsync.log').open('xb') as log:
            CHILD = subprocess.Popen(receipt['rsync_command'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while CHILD.poll() is None:
                    time.sleep(5)
                    mount_check(plan)
                    resources()
                if CHILD.returncode:
                    raise RuntimeError(f'rsync exited {CHILD.returncode}')
                receipt['rsync_exit_code'] = CHILD.returncode
            finally:
                if CHILD.poll() is None:
                    os.killpg(CHILD.pid, signal.SIGTERM)
                    CHILD.wait(timeout=30)
                CHILD = None
        mount_check(plan)
        remote_before = archive_destination.stat()
        receipt['external_tar_sha256'] = sha(archive_destination)
        remote_after = archive_destination.stat()
        def remote_identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if remote_identity(remote_before) != remote_identity(remote_after) or receipt['external_tar_sha256'] != receipt['tar_sha256']:
            raise ValueError('archive readback hash/metadata mismatch')
        mount_check(plan)
        check_source(manifest)
        references(pool)
        if sha(plan['_path']) != plan_sha:
            raise ValueError('plan changed while copying')
        receipt.update({'status': 'COPY_AND_CONTENT_VERIFICATION_COMPLETE', 'finished_unix': time.time(),
                        'external_archive': str(archive_destination), 'external_size': remote_after.st_size,
                        'external_mtime_ns': remote_after.st_mtime_ns,
                        'limitations': ['Source retained; removal requires separate parent verification.',
                                        'Observed fd/cwd consumers rejected; inaccessible process IDs disclosed. Not deletion qualification.',
                                        'File mode/uid/gid/mtime and all contents verified; GNU tar captures ACL/xattrs but these are not independently compared.']})
        save(attempt / 'verified.json', receipt)
        # This is the ONLY unlink in the driver. It targets the just-verified
        # LOCAL staging tar; source and external archive are never removed.
        archive.unlink()
        print(f'{name}: verified, source retained, staging tar removed', flush=True)
    except BaseException as error:
        receipt.update({'status': 'FAILED_SOURCE_AND_PARTIAL_PRESERVED', 'error': repr(error), 'failed_unix': time.time()})
        save(attempt / 'failed.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pool', action='append', help='Restrict to named plan pools, retaining plan order')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get('schema_version', plan.get('schema')) != 1:
        raise ValueError('plan schema must be 1')
    if plan.get('source_root', str(SOURCE)) != str(SOURCE) or plan.get('archive_root', str(DEST)) != str(DEST):
        raise ValueError('plan changes fixed roots')
    names = []
    for item in plan['pools']:
        name = item['name']
        if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'} or name in EXCLUDED:
            raise ValueError(f'unsafe/protected pool: {name}')
        if type(item['estimated_tar_bytes']) is not int or item['estimated_tar_bytes'] <= 0:
            raise ValueError('positive estimated_tar_bytes required')
        names.append(name)
    if len(set(names)) != len(names) or (args.pool and not set(args.pool) <= set(names)):
        raise ValueError('duplicate or unknown pools')
    selected = [item for item in plan['pools'] if not args.pool or item['name'] in args.pool]
    plan_sha = sha(args.plan)
    print(json.dumps({'mode': 'EXECUTE_COPY_VERIFY_ONLY' if args.execute else 'PLAN_ONLY',
                      'plan_sha256': plan_sha, 'pools': selected, 'source': str(SOURCE), 'destination': str(DEST)}), flush=True)
    if not args.execute:
        return
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, {0, 1})
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, terminate)
    plan['_path'] = args.plan.resolve()
    for item in selected:
        if sha(args.plan) != plan_sha:
            raise ValueError('plan changed before next pool')
        run_pool(item, plan, plan_sha)


if __name__ == '__main__':
    main()
