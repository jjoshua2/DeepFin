"""Opt-in external dependency test: no Python/code tree in the engine filesystem.

Creates a NEW owned directory with the neural executable, exact package, native
ELF dependencies and /tmp. Does not launch chroot or modify a system installation.
Only use trusted local binaries/packages: dependency discovery invokes ldd.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


def prepare(root: Path, executable: Path, package: Path) -> dict[str, object]:
    executable, package = executable.resolve(strict=True), package.resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'tmp').mkdir(mode=0o700)
    # LibTorch's CPU dispatch reads this OS hardware description, not Python.
    (root / 'proc').mkdir()
    shutil.copyfile('/proc/cpuinfo', root / 'proc/cpuinfo')
    (root / 'proc/cpuinfo').chmod(0o444)
    shutil.copy2(executable, root / 'deepfin-bend-neural')
    shutil.copy2(package, root / 'model.pt2')
    # An extracted package .so has no executable RUNPATH. Resolve the executable
    # first, then inspect package libraries in precisely its resolved library dirs.
    # Missing dependencies still fail; no interpreter or compiler is copied.
    probe = subprocess.run(['ldd', str(executable)], text=True, capture_output=True, check=True, timeout=15).stdout
    if 'not found' in probe:
        raise ValueError('unresolved executable dependency: ' + probe)
    pattern = r'(/[^\s]+)\s+\(0x[0-9a-f]+\)'
    environment = os.environ.copy()
    environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(str(Path(p).parent) for p in re.findall(pattern, probe)))
    pending = [executable]
    seen: set[Path] = set()
    libraries: set[Path] = set()
    # Compiled package .so dependencies need checking too; extracting them here
    # does not place model sources, Python metadata, or a compiler in the runtime.
    with tempfile.TemporaryDirectory(prefix='bend-dependency-check-') as temp:
        with zipfile.ZipFile(package) as z:
            for i, member in enumerate(z.infolist()):
                if member.filename.endswith('.so'):
                    target = Path(temp) / f'model-{i}.so'
                    with z.open(member) as src, target.open('xb') as dest:
                        shutil.copyfileobj(src, dest)
                    pending.append(target)
        while pending:
            binary = pending.pop()
            if binary in seen:
                continue
            seen.add(binary)
            out = subprocess.run(['ldd', str(binary)], text=True, capture_output=True, check=True, timeout=15, env=environment).stdout
            if 'not found' in out:
                raise ValueError('unresolved native dependency: ' + out)
            for text in re.findall(pattern, out):
                path = Path(text)
                if 'libpython' in path.name:
                    raise ValueError('interpreter linked into native runtime')
                if path not in libraries:
                    libraries.add(path)
                    target = root / str(path).lstrip('/')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    pending.append(path)
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    assert all(not p.endswith(('.py', '.pyc', '.json')) for p in files)
    assert len(files) == len(libraries) + 3
    return {'root': str(root), 'files': files, 'native_libraries': len(libraries),
            'scope': 'engine + exact model package + native libraries + CPU description + scratch; no Python interpreter or controller'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    result = prepare(args.root, args.executable, args.package)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
