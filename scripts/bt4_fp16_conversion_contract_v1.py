"""Stdlib-only exact-artifact compatibility gate; no authority is minted here.

The manifest hash must come from a separately reviewed, closed provenance audit.
Passing an arbitrary user-authored manifest is not a scientific approval route.
"""
import hashlib
import json
import math
import os
import pathlib
import stat

MODEL = '809db9c3c8063683bf8e7c08de8fec92ea4e37ad8c68c8b1befbde4bd1958961'
SCHEMA = 'BT4_EXACT809_REVIEWED_RAW_PROVENANCE_V1'
KINDS = {'scope', 'descriptor', 'checkpoint', 'producer_source', 'precision_validator'}
_LOADER_TOKEN = object()

class LoadedManifest(dict):
    """Marker issued only after the fixed trust pin and evidence reads pass."""
    def __init__(self, value, token):
        need(token is _LOADER_TOKEN, 'manifest loader token')
        super().__init__(value)

class AuthenticatedContext(dict):
    """In-process marker, never accepted from a JSON roster or raw metadata."""

def need(ok, message):
    if not ok:
        raise ValueError(message)

def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def hexsha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def stable_bytes(path, cap):
    p = pathlib.Path(path)
    need(p.is_absolute(), 'absolute provenance path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'bounded regular provenance file')
        chunks = []
        remaining = cap + 1
        while remaining:
            block = os.read(fd, min(1048576, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        value = b''.join(chunks)
        after = os.fstat(fd)
        need(len(value) <= cap and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'stable provenance read')
        return value
    finally:
        os.close(fd)

def identity(entry):
    r = entry['receipt']
    return (entry['bank'], entry['unit'], entry['game'], r['path'], r['sha256'], r['rows'], entry['model_sha256'])

def validate_manifest(doc, roster_sha):
    need(set(doc) == {'schema', 'status', 'roster_sha256', 'model_sha256', 'arithmetic_precision', 'storage_dtype', 'evidence', 'games'}, 'closed provenance schema')
    need(doc['schema'] == SCHEMA and doc['status'] == 'INDEPENDENTLY_VERIFIED_EXACT809_RAW_PROVENANCE', 'reviewed provenance schema/status')
    need(hexsha(roster_sha) and doc['roster_sha256'] == roster_sha and doc['model_sha256'] == MODEL and
         doc['arithmetic_precision'] == 'fp16' and doc['storage_dtype'] == 'float32', 'exact roster/model/precision')
    need(isinstance(doc['evidence'], dict) and set(doc['evidence']) == KINDS, 'closed evidence kinds')
    for ref in doc['evidence'].values():
        need(set(ref) == {'path', 'sha256'} and pathlib.Path(ref['path']).is_absolute() and hexsha(ref['sha256']), 'evidence ref')
    need(isinstance(doc['games'], list) and 0 < len(doc['games']) <= 65536, 'bounded provenance games')
    seen = set()
    seen_uids = set()
    seen_raw_sha256 = set()
    for g in doc['games']:
        need(set(g) == {'bank', 'unit', 'game', 'receipt', 'model_sha256', 'checkpoint_binding_sha256'}, 'closed game schema')
        r = g['receipt']
        need(set(r) == {'path', 'sha256', 'rows'} and pathlib.Path(r['path']).is_absolute() and hexsha(r['sha256']), 'raw receipt')
        need(type(r['rows']) is int and 0 < r['rows'] <= 400 and type(g['game']) is int and g['game'] >= 0 and
             isinstance(g['bank'], str) and isinstance(g['unit'], str) and hexsha(g['checkpoint_binding_sha256']) and g['model_sha256'] == MODEL, 'game source identity')
        key = identity(g)
        need(key not in seen, 'duplicate provenance game')
        uid = (g['bank'], g['unit'], g['game'])
        need(uid not in seen_uids, 'duplicate physical game')
        need(r['sha256'] not in seen_raw_sha256, 'duplicate raw content')
        seen.add(key)
        seen_uids.add(uid)
        seen_raw_sha256.add(r['sha256'])
    return doc

def load_manifest(path, expected_sha, roster_sha):
    need(hexsha(expected_sha), 'externally pinned provenance hash')
    raw = stable_bytes(path, 16 * 1024 * 1024)
    need(digest_bytes(raw) == expected_sha, 'provenance hash changed')
    doc = validate_manifest(json.loads(raw), roster_sha)
    # Fixed small metadata/source evidence only; checkpoint here is a hash-bound
    # exported checkpoint witness JSON, never a live mutable SQLite database.
    for ref in doc['evidence'].values():
        need(digest_bytes(stable_bytes(ref['path'], 4 * 1024 * 1024)) == ref['sha256'], 'provenance evidence changed')
    return LoadedManifest(doc, _LOADER_TOKEN)

def authenticated_context(doc, entry):
    need(isinstance(doc, LoadedManifest), 'manifest was not loaded through trusted evidence gate')
    matches = [g for g in doc['games'] if identity(g) == identity(entry)]
    need(len(matches) == 1, 'raw game absent from reviewed provenance')
    return AuthenticatedContext(model_sha256=MODEL, raw_identity=identity(matches[0]), checkpoint_binding_sha256=matches[0]['checkpoint_binding_sha256'])

def mass_tolerance(entry):
    context = entry.get('_exact809_context')
    if context is None:
        return 2e-6
    need(isinstance(context, AuthenticatedContext) and set(context) == {'model_sha256', 'raw_identity', 'checkpoint_binding_sha256'} and context['model_sha256'] == MODEL and
         context['raw_identity'] == identity(entry) and hexsha(context['checkpoint_binding_sha256']), 'authenticated context mismatch')
    return 2**-10

def check_mass_rows(rows, tolerance):
    """Pure fixture oracle for the main adapter's NumPy range/mass predicates."""
    for row in rows:
        need(len(row) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in row), 'WDL range/finite/shape')
        need(abs(sum(row) - 1) <= tolerance, 'WDL mass')
