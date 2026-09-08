"""Check the qualified external runtime dependencies without importing CUDA."""
import hashlib
import json
from pathlib import Path

QUALIFICATION = Path('/home/josh/projects/chess/scratchpad/bt4_joint20/label_preparation_profile_v1/runtime_adoption_pr537/qualification.json')
EXPECTED = 'aed87c6f60fdc826380bd01c562f976ead23359fe4566b88ce69adbd7a0a6ac9'

def read_pinned(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError(f'changed qualification evidence: {path}')
    return json.loads(payload)

def check_stat(path, expected):
    actual = Path(path).stat()
    if any(getattr(actual, key) != value for key, value in expected.items() if key.startswith('st_')):
        raise RuntimeError(f'qualified runtime dependency changed: {path}')

q = read_pinned(QUALIFICATION, EXPECTED)
candidate_path = str(QUALIFICATION.parent / 'candidate_imports.json')
candidate = read_pinned(candidate_path, q['evidence'][candidate_path])
check_stat(candidate['interpreter']['path'], candidate['interpreter'])
for library in candidate['libraries'].values():
    check_stat(library['file']['path'], library['file'])
for native in q['native_links']:
    check_stat(native['runtime_path'], native['stat'])
    if hashlib.sha256(Path(native['runtime_path']).read_bytes()).hexdigest() != native['sha256']:
        raise RuntimeError(f'qualified native content changed: {native["runtime_path"]}')
