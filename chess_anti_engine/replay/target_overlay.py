"""Explicit, immutable policy overlays for qualified exact-epoch consumers.

The storage seal is a byte/row/history proof, not a recipe or strength verdict.
Only ordinary bases and policy_target replacements are supported in v1.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import struct
from pathlib import Path
from typing import Any

import numpy as np
import zarr

MANIFEST = 'target_overlay.json'
BASE_STATUS = 'PASS_IMMUTABLE_BASE_STORAGE_SEAL'
OVERLAY_STATUS = 'PASS_IMMUTABLE_POLICY_OVERLAY_STORAGE_QUALIFICATION'
POLICY = 'policy_target'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_overlay(path: Path) -> bool:
    candidate = path / MANIFEST
    return candidate.exists() or candidate.is_symlink()


def tree_stamp(path: Path) -> str:
    """Bind membership and regular-file identities, including ctime and inode."""
    require(path.is_dir() and not path.is_symlink(), 'storage must be a plain directory')
    entries = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            item = Path(root) / name
            st = item.lstat()
            require(stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode),
                    'overlay storage forbids links and special descendants')
            entries.append((str(item.relative_to(path)), st.st_mode, st.st_dev, st.st_ino,
                            st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    require(bool(entries), 'empty storage tree')
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def _plain_content(path: Path) -> str:
    require(not has_overlay(path), 'overlay chains are unsupported')
    return plain_content_sha256(path)


def _atomic_new_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), 'refusing existing storage receipt')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.writing')
    with temp.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    # A link publishes the complete file without replacing a concurrent owner.
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def _read_pin(ref: dict[str, str]) -> dict[str, Any]:
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'invalid receipt path')
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'receipt changed during read')
    require(hashlib.sha256(payload).hexdigest() == ref['sha256'], 'storage receipt SHA differs')
    result = json.loads(payload)
    require(isinstance(result, dict), 'storage receipt must be an object')
    return result


def _receipt_stamp(path: Path) -> tuple[int, int, int, int, int]:
    require(path.is_file() and not path.is_symlink(), 'invalid receipt path')
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


class BaseSeal:
    """One operation's validated seal/index; never a global unchecked cache."""

    def __init__(self, ref: dict[str, str]) -> None:
        self._ref = dict(ref)
        self._path = Path(ref['path'])
        self._stamp = _receipt_stamp(self._path)
        self.value = _read_pin(ref)
        require(self.value.get('schema') == 1 and self.value.get('status') == BASE_STATUS,
                'wrong base seal type')
        self.root = Path(self.value['base'])
        self._entries = {entry['name']: entry for entry in self.value['shards']}
        require(len(self._entries) == len(self.value['shards']), 'duplicate base seal shard names')
        self.check(ref)

    def check(self, ref: dict[str, str]) -> None:
        require(ref == self._ref, 'different base seal context')
        require(_receipt_stamp(self._path) == self._stamp, 'base seal identity changed during operation')

    def entry(self, ref: dict[str, str], base: Path) -> dict[str, Any]:
        self.check(ref)
        require(base.parent == self.root and not has_overlay(base), 'wrong base or overlay chain')
        require(base.name in self._entries, 'base shard absent from seal')
        entry = self._entries[base.name]
        require(tree_stamp(base) == entry['storage_stamp'], 'sealed base changed')
        self.check(ref)
        return entry


def base_entry(ref: dict[str, str], base: Path, *, seal: BaseSeal | None = None) -> dict[str, Any]:
    return (seal if seal is not None else BaseSeal(ref)).entry(ref, base)


def seal_for_overlay(path: Path) -> BaseSeal:
    require(not (path / MANIFEST).is_symlink(), 'linked overlay manifest')
    manifest = json.loads((path / MANIFEST).read_bytes())
    return BaseSeal(manifest['base_seal'])


def begin_policy_shard(base: Path, output: Path, seal_ref: dict[str, str], *, seal: BaseSeal | None = None) -> None:
    """Create only local metadata; mixer writes every replacement chunk itself."""
    base_entry(seal_ref, base, seal=seal)
    require(not output.exists(), 'overlay output already exists')
    output.mkdir()
    for name in ('.zgroup', '.zattrs'):
        require((base / name).is_file(), 'missing base group metadata')
        shutil.copyfile(base / name, output / name)
    (output / POLICY).mkdir()
    for name in ('.zarray', '.zattrs'):
        if (base / POLICY / name).is_file():
            shutil.copyfile(base / POLICY / name, output / POLICY / name)


def finish_policy_shard(base: Path, output: Path, seal_ref: dict[str, str], *, seal: BaseSeal | None = None) -> None:
    entry = base_entry(seal_ref, base, seal=seal)
    local: Any = zarr.open_group(str(output), mode='r')
    require(set(local.array_keys()) == {POLICY}, 'overlay permits only policy_target replacement')
    replacement = local[POLICY]
    identity = entry['identity']
    require(list(replacement.shape) == identity['policy_shape']
            and np.dtype(replacement.dtype).str == identity['policy_dtype'], 'replacement layout differs')
    value = {'schema': 1, 'kind': 'immutable-policy-overlay', 'base': str(base),
             'base_seal': seal_ref, 'base_content_sha256': entry['content_sha256'],
             'identity': identity, 'replacement': POLICY,
             'target_content_sha256': _plain_content(output / POLICY),
             'base_lifetime': 'Retain the complete base and seal until every dependent overlay is retired.'}
    _atomic_new_json(output / MANIFEST, value)
    overlay_content_sha256(output, seal=seal)


def _open_manifest(path: Path, *, seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    require(has_overlay(path), 'missing overlay manifest')
    require(not (path / MANIFEST).is_symlink(), 'linked overlay manifest')
    manifest = json.loads((path / MANIFEST).read_bytes())
    require(manifest.get('schema') == 1 and manifest.get('kind') == 'immutable-policy-overlay'
            and manifest.get('replacement') == POLICY, 'unsupported overlay type/replacement')
    base = Path(manifest['base'])
    require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
    entry = base_entry(manifest['base_seal'], base, seal=seal)
    require(manifest['identity'] == entry['identity']
            and manifest['base_content_sha256'] == entry['content_sha256'], 'overlay base identity differs')
    require(_plain_content(path / POLICY) == manifest['target_content_sha256'], 'overlay target changed')
    local: Any = zarr.open_group(str(path), mode='r')
    original: Any = zarr.open_group(str(base), mode='r')
    require(set(local.array_keys()) == {POLICY}, 'unexpected overlay arrays')
    attrs, original_attrs = dict(local.attrs), dict(original.attrs)
    def stable(values: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in values.items() if not k.startswith('policy_target_mix_')}

    require(stable(attrs) == stable(original_attrs), 'overlay changed inherited metadata/history')
    target = local[POLICY]
    require(list(target.shape) == entry['identity']['policy_shape']
            and np.dtype(target.dtype).str == entry['identity']['policy_dtype'], 'overlay target layout changed')
    return manifest, attrs


def overlay_content_sha256(path: Path, *, seal: BaseSeal | None = None) -> str:
    path = path.resolve(strict=True)
    before = tree_stamp(path)
    seal = seal if seal is not None else seal_for_overlay(path)
    manifest, _ = _open_manifest(path, seal=seal)
    # The sealed inherited content digest is usable only while every anchored
    # file stamp still matches. Fresh, unanchored stats never substitute for it.
    local = plain_content_sha256(path)
    require(tree_stamp(path) == before, 'overlay changed during identity read')
    base_entry(manifest['base_seal'], Path(manifest['base']), seal=seal)
    return hashlib.sha256(json.dumps({'kind': 'immutable-policy-overlay-v1',
        'base': manifest['base_content_sha256'], 'base_seal': manifest['base_seal'],
        'local': local}, sort_keys=True).encode()).hexdigest()


def overlay_proxies(path: Path, fields: tuple[str, ...], *, seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, meta = _open_manifest(path, seal=seal)
    base: Any = zarr.open_group(manifest['base'], mode='r')
    local: Any = zarr.open_group(str(path), mode='r')
    arrays = {name: (local[POLICY] if name == POLICY else base[name])
              for name in fields if name in base}
    return arrays, meta


def _root_files(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.iterdir()):
        require(not path.is_symlink(), 'linked corpus metadata/shard')
        if path.is_file():
            files[path.name] = sha(path)
    return files


def require_base_corpus(ref: dict[str, str], root: Path, *, context: BaseSeal | None = None) -> dict[str, Any]:
    context = context if context is not None else BaseSeal(ref)
    context.check(ref)
    seal = context.value
    require(str(root.resolve(strict=True)) == seal['base'], 'base root differs')
    require([p.name for p in shard_paths(root)] == [e['name'] for e in seal['shards']],
            'sealed base membership differs')
    require(_root_files(root) == seal['root_files'], 'sealed base metadata changed')
    for path in shard_paths(root):
        base_entry(ref, path, seal=context)
    context.check(ref)
    return seal


def verify_qualification(ref: dict[str, str], root: Path, *, context: BaseSeal | None = None) -> dict[str, Any]:
    receipt = _read_pin(ref)
    root = root.resolve(strict=True)
    require(receipt.get('schema') == 1 and receipt.get('status') == OVERLAY_STATUS
            and receipt.get('root') == str(root), 'wrong overlay storage qualification')
    require(_root_files(root) == receipt['root_files'], 'qualified overlay metadata changed')
    context = context if context is not None else BaseSeal(receipt['base_seal'])
    require_base_corpus(receipt['base_seal'], context.root, context=context)
    paths = shard_paths(root)
    require(bool(paths) and [p.name for p in paths] == [e['name'] for e in receipt['shards']],
            'qualified overlay membership changed')
    for path, expected in zip(paths, receipt['shards'], strict=True):
        require(overlay_content_sha256(path, seal=context) == expected['content_sha256'],
                'qualified overlay dependency/content changed')
    return receipt


def shard_paths(root: Path) -> list[Path]:
    """The v1 layout uses ordinary local Zarr shard names exclusively."""
    return sorted(root.glob("shard_*.zarr"))


def plain_content_sha256(path: Path) -> str:
    """Stream a deterministic digest over a Zarr tree's names and bytes."""
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"exact-epoch shard is not a directory: {path}")
    digest = hashlib.sha256()
    files_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            relative = file_path.relative_to(root).as_posix().encode(
                "utf-8", errors="surrogateescape",
            )
            before = file_path.stat()
            digest.update(struct.pack("<I", len(relative)))
            digest.update(relative)
            digest.update(struct.pack("<Q", int(before.st_size)))
            bytes_read = 0
            with file_path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    bytes_read += len(block)
                    digest.update(block)
                after = os.fstat(handle.fileno())
            if (
                bytes_read != int(before.st_size)
                or int(after.st_size) != int(before.st_size)
                or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                or int(after.st_ctime_ns) != int(before.st_ctime_ns)
            ):
                raise RuntimeError(
                    f"{path} changed while its exact-epoch content was hashed",
                )
            files_seen += 1
    if files_seen == 0:
        raise ValueError(f"exact-epoch shard contains no files: {path}")
    return digest.hexdigest()


def qualified_paths(ref: dict[str, str], paths: list[Path]) -> tuple[dict[Path, str], BaseSeal]:
    """Bind staging and every new epoch to the actual qualified corpus."""
    receipt = _read_pin(ref)
    root = Path(receipt['root'])
    context = BaseSeal(receipt['base_seal'])
    receipt = verify_qualification(ref, root, context=context)
    expected = {root / entry['name']: entry['content_sha256'] for entry in receipt['shards']}
    require([path.resolve(strict=True) for path in paths] == list(expected),
            'staged overlay paths/order differ from qualified corpus')
    return expected, context
