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


@pytest.mark.parametrize("statm", ["100 7 2 0 0 0 0", "100 0 0 0 0 0 0"])
def test_linux_rss_uses_current_resident_pages(monkeypatch, statm):
    from scripts import benchmark_storage_loader as pilot

    monkeypatch.setattr(pilot.Path, "read_text", lambda _path: statm)
    monkeypatch.setattr(pilot.os, "sysconf", lambda _key: 4096)
    assert pilot.current_rss_bytes() == int(statm.split()[1]) * 4096


@pytest.mark.parametrize("statm", ["", "100", "100 invalid", "100 -1"])
def test_linux_rss_rejects_unreadable_or_invalid_samples(monkeypatch, statm):
    from scripts import benchmark_storage_loader as pilot

    monkeypatch.setattr(pilot.Path, "read_text", lambda _path: statm)
    with pytest.raises((ValueError, IndexError)):
        pilot.current_rss_bytes()


def test_linux_rss_propagates_missing_proc_read(monkeypatch):
    from scripts import benchmark_storage_loader as pilot

    def missing(_path):
        raise FileNotFoundError("statm unavailable")

    monkeypatch.setattr(pilot.Path, "read_text", missing)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        pilot.current_rss_bytes()

