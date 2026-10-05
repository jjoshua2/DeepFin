"""Dual digests preserve the existing byte contract with one physical read."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
from typing import Any

import pytest

from chess_anti_engine.replay import target_overlay as storage


def _legacy_digest(path: Path) -> str:
    # Independent pre-change serialization oracle: ordered names, lengths, bytes.
    digest = hashlib.sha256()
    for directory, directories, files in os.walk(path):
        directories.sort()
        for name in sorted(files):
            item = Path(directory) / name
            relative = item.relative_to(path).as_posix().encode("utf-8", errors="surrogateescape")
            data = item.read_bytes()
            digest.update(struct.pack("<I", len(relative)))
            digest.update(relative)
            digest.update(struct.pack("<Q", len(data)))
            digest.update(data)
    return digest.hexdigest()


def _tree(path: Path) -> None:
    for name, data in {
        '.zgroup': b'{"zarr_format":2}',
        'policy_target/.zarray': b'policy metadata',
        'policy_target/0.0': b'x' * (1024 * 1024 + 17),
        'policy_target/nested/empty': b'',
        'policy_target/nested/unicode-\u03bb': b'policy tail',
        'search_wdl/.zarray': b'value metadata',
        'search_wdl/0.0': bytes(range(256)),
        'unrelated/0': b'whole tree only',
    }.items():
        item = path / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(data)


@pytest.mark.parametrize('names', [(), ('policy_target',), ('policy_target', 'search_wdl')])
def test_exact_legacy_digest_and_one_read_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...],
) -> None:
    _tree(tmp_path)
    expected = _legacy_digest(tmp_path)
    expected_targets = {name: _legacy_digest(tmp_path / name) for name in names}
    original = Path.open
    reads: dict[Path, int] = {}

    def counted(path: Path, *args: Any, **kwargs: Any) -> Any:
        if args and args[0] == 'rb':
            reads[path] = reads.get(path, 0) + 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', counted)
    actual, targets = storage._plain_content_digests(tmp_path, names)
    assert actual == expected
    assert targets == expected_targets
    assert reads
    assert set(reads.values()) == {1}
    assert set(reads) == {p for p in tmp_path.rglob('*') if p.is_file()}


def test_byte_and_name_changes_change_correct_digests(tmp_path: Path) -> None:
    _tree(tmp_path)
    whole, targets = storage._plain_content_digests(tmp_path, ('policy_target', 'search_wdl'))
    (tmp_path / 'policy_target/0.0').write_bytes(b'changed')
    changed, changed_targets = storage._plain_content_digests(tmp_path, ('policy_target', 'search_wdl'))
    assert changed != whole
    assert changed_targets['policy_target'] != targets['policy_target']
    assert changed_targets['search_wdl'] == targets['search_wdl']
    (tmp_path / 'search_wdl/0.0').rename(tmp_path / 'search_wdl/renamed')
    renamed, renamed_targets = storage._plain_content_digests(tmp_path, ('policy_target', 'search_wdl'))
    assert renamed != changed
    assert renamed_targets['search_wdl'] != changed_targets['search_wdl']


def test_rejects_mutation_during_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tree(tmp_path)
    original = os.fstat
    changed = False

    def mutated(fd: int) -> os.stat_result:
        nonlocal changed
        if not changed:
            changed = True
            (tmp_path / '.zgroup').write_bytes(b'longer and changed')
        return original(fd)

    monkeypatch.setattr(os, 'fstat', mutated)
    with pytest.raises(RuntimeError, match='changed while its exact-epoch content was hashed'):
        storage._plain_content_digests(tmp_path, ('policy_target', 'search_wdl'))


@pytest.mark.parametrize('change', ['empty', 'missing', 'chain'])
def test_rejects_invalid_replacement_subtree(tmp_path: Path, change: str) -> None:
    (tmp_path / '.zgroup').write_bytes(b'root')
    subtree = tmp_path / 'policy_target'
    if change != 'missing':
        subtree.mkdir()
    if change == 'chain':
        (subtree / storage.MANIFEST).write_bytes(b'{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        storage._plain_content_digests(tmp_path, ('policy_target',))
