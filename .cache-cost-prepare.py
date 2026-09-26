"""Temporary transport only; apply and fingerprint exactly seven intended paths."""
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess

BASE = 'ead1abf29f6bbe6245d7ea8428f5fa0b45d6054f'
PATCH_SHA = '6114fcedf4f79094e43857b53dbfd56602918d2922694bae0e003aee48c6ddf5'
EXPECTED = {
    'tests/test_bend_map_cache_benchmark.py': '54fa821627e6b19d2f05d43d7e65e2124464ef02bacf3f73d5ce737889db4aa7',
    'docs/experiments/2026-09-26-move-cache-cost-screen.md': '9ef836ea6323e99534104041a6d6542ea8ee539a61bb2c9927009b7cd7552bdb',
    'docs/experiments/README.md': '0c35ece5100d6589c1d75ec632e777220b0f6ba07cd5637b4e187d862ce4c744',
    'docs/experiments/evidence/move-cache-cost-screen/samples.csv': 'c3acc2e46367bba20b86f17854cf545aa81024d2e46a92c2125d0d99028b83f9',
    'docs/experiments/evidence/move-cache-cost-screen/summary.json': '71a8dec5cd98e1dde9be108380ac2d5c56c20e14c8a53d9a9cebb7975f7ea7a5',
    'native/bend_engine/u64_map_probe/move_cache_benchmark.bend': 'f798521ff0f5394f1df27a7f9c7b7f98630554f80d5c1f6d8a4b0333eb9e9bf9',
    'native/bend_engine/u64_map_probe/move_cache_benchmark.py': 'c64138b065e71a2c55a21cfdd555ce308c403549861a4074c73c630799cbf1a1',
}
assert os.environ['BASE'] == BASE
raw = lzma.decompress(base64.b64decode(''.join(Path(f'.cache-cost-payload-{i}.b64').read_text() for i in range(1, 7)), validate=True))
assert hashlib.sha256(raw).hexdigest() == PATCH_SHA
work = Path(os.environ['RUNNER_TEMP']) / 'cache-cost'
subprocess.run(['git', 'fetch', '--depth=1', 'origin', BASE], check=True)
subprocess.run(['git', 'worktree', 'add', '--detach', str(work), BASE], check=True)
patch = Path(os.environ['RUNNER_TEMP']) / 'cache-cost.patch'
patch.write_bytes(raw)
subprocess.run(['git', '-C', str(work), 'apply', '--check', str(patch)], check=True)
subprocess.run(['git', '-C', str(work), 'apply', '--index', str(patch)], check=True)
assert sorted(subprocess.check_output(['git', '-C', str(work), 'diff', '--cached', '--name-only'], text=True).splitlines()) == sorted(EXPECTED)
for path, digest in EXPECTED.items():
    assert hashlib.sha256((work / path).read_bytes()).hexdigest() == digest, path
subprocess.run(['git', '-C', str(work), 'diff', '--cached', '--check'], check=True)
# Corrections, if ever needed, remain explicit and do not alter the original transport.
fixes = Path('.cache-cost-fixes.py')
if fixes.exists():
    subprocess.run(['python3', str(fixes.resolve()), str(work)], check=True)
    subprocess.run(['git', '-C', str(work), 'add', '--', *EXPECTED], check=True)
manifest = {path: hashlib.sha256((work / path).read_bytes()).hexdigest() for path in EXPECTED}
(work / 'artifacts').mkdir(exist_ok=True)
(work / 'artifacts/candidate-source.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Prepared exact base with seven benchmark/test/evidence paths.')
