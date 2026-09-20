from __future__ import annotations

import shutil

import pytest

from scripts.benchmark_storage_loader import copy_payload_tree


def test_copy_to_filesystem_without_posix_metadata(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'array').mkdir(parents=True)
    (source / 'array' / '0').write_bytes(b'compressed payload')
    (source / '.zattrs').write_text('{}')

    def unsupported(*_args, **_kwargs):
        raise PermissionError('drvfs cannot preserve POSIX metadata')

    monkeypatch.setattr(shutil, 'copystat', unsupported)
    target = tmp_path / 'target'
    copy_payload_tree(source, target)
    assert (target / 'array' / '0').read_bytes() == b'compressed payload'
    assert (target / '.zattrs').read_text() == '{}'


def test_copy_refuses_symlink_and_existing_output(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlink'):
        copy_payload_tree(source, tmp_path / 'out')
    with pytest.raises(FileExistsError):
        copy_payload_tree(source, tmp_path / 'out')
