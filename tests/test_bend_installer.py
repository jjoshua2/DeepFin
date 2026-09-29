"""Installer integration checks with local fake downloads; never uses the network."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _installation(tmp_path: Path, *, checksum_ok: bool = True, version_ok: bool = True,
                  metadata_ok: bool = True) -> tuple[subprocess.CompletedProcess[str], Path]:
    repo = tmp_path / 'repo'
    native = repo / 'native/bend_engine'
    native.mkdir(parents=True)
    for name in ('install_bend.sh', 'resolve_latest.py'):
        shutil.copyfile(ROOT / 'native/bend_engine' / name, native / name)
    archive = tmp_path / 'release.tgz'
    version = '2.0.25' if version_ok else '0.0.0'
    executable = f'#!/bin/sh\n[ "$1" = version ] || exit 17\nprintf "bend {version}\\n"\n'.encode()
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data, mode in [('bend/bin/bend', executable, 0o755),
                                 ('bend/bend2/base.bend', b'# fixture\n', 0o644)]:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    metadata = tmp_path / 'latest.json'
    sha = hashlib.sha256(archive.read_bytes()).hexdigest() if checksum_ok else '0' * 64
    payload = {'ver': '2.0.25', 'sha256': sha, 'url': 'https://bend-lang.com/dl/test.tgz'}
    metadata.write_text(json.dumps(payload) if metadata_ok else 'not JSON')
    commands = tmp_path / 'commands'
    commands.mkdir()
    curl = commands / 'curl'
    curl.write_text('#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n'
                    '  if [ "$1" = -o ]; then cp "$TEST_ARCHIVE" "$2"; exit; fi\n'
                    '  shift\ndone\ncat "$TEST_METADATA"\n')
    curl.chmod(0o755)
    prefix = tmp_path / 'installation'
    env = {**os.environ, 'PATH': str(commands) + os.pathsep + os.environ['PATH'],
           'TEST_ARCHIVE': str(archive), 'TEST_METADATA': str(metadata),
           'BEND_PROBE_HOME': str(prefix), 'BEND_PLATFORM': 'linux-x64',
           'GITHUB_PATH': str(tmp_path / 'github-path')}
    env.pop('GITHUB_TOKEN', None)
    env.pop('GH_TOKEN', None)
    result = subprocess.run(['bash', str(native / 'install_bend.sh')], env=env,
                            capture_output=True, text=True, check=False, timeout=15)
    return result, prefix


def test_native_release_installs_and_reports_real_version(tmp_path: Path) -> None:
    result, prefix = _installation(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'installed bend 2.0.25' in result.stdout
    version = subprocess.run([str(prefix / 'bin/bend'), 'version'],
                             capture_output=True, text=True, check=True, timeout=5)
    assert version.stdout == 'bend 2.0.25\n'


@pytest.mark.parametrize('kind', ['checksum', 'version', 'metadata'])
def test_broken_release_cannot_report_installation_success(tmp_path: Path, kind: str) -> None:
    result, prefix = _installation(tmp_path, checksum_ok=kind != 'checksum',
                                   version_ok=kind != 'version', metadata_ok=kind != 'metadata')
    assert result.returncode != 0
    assert 'installed ' not in result.stdout
    if kind == 'metadata':
        assert 'latest.json is not JSON' in result.stderr
        assert 'invalid Bend version in latest.json' not in result.stderr
        assert not prefix.exists()
    elif kind == 'checksum':
        assert 'sha256 mismatch' in result.stderr
        assert not (prefix / 'bin/bend').exists()
    else:
        assert 'version mismatch' in result.stderr
