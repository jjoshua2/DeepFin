"""Explicit, immutable policy overlays for qualified exact-epoch consumers.

The storage seal is a byte/row/history proof, not a recipe or strength verdict.
Schema 1 replaces policy only; schema 2 replaces policy and/or search WDL.
Both inherit ordinary sealed bases without chains.
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
TARGET_OVERLAY_STATUS = 'PASS_IMMUTABLE_TARGET_OVERLAY_STORAGE_QUALIFICATION'
TARGET_FIELDS = frozenset({'policy_target', 'search_wdl'})


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
    if manifest.get('schema') == 2:
        return _open_target_manifest(path, manifest, seal=seal)
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
    replacements = manifest['replacements'] if manifest['schema'] == 2 else [POLICY]
    arrays = {name: (local[name] if name in replacements else base[name])
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
    if receipt.get('schema') == 2:
        expected, contexts = verify_receipt(receipt)
        require([path.resolve(strict=True) for path in paths] == list(expected),
                'staged overlay paths/order differ from qualified corpus')
        return expected, contexts
    root = Path(receipt['root'])
    context = BaseSeal(receipt['base_seal'])
    receipt = verify_qualification(ref, root, context=context)
    expected = {root / entry['name']: entry['content_sha256'] for entry in receipt['shards']}
    require([path.resolve(strict=True) for path in paths] == list(expected),
            'staged overlay paths/order differ from qualified corpus')
    return expected, context


class BaseSeals(BaseSeal):
    """Operation-local seal routing; compatible with exact-epoch storage consumers."""

    def __init__(self, refs: list[dict[str, str]]) -> None:
        require(bool(refs), 'empty base seals')
        self.contexts = {ref['path']: BaseSeal(ref) for ref in refs}
        super().__init__(refs[0])
        require(bool(refs) and len(self.contexts) == len(refs), 'duplicate/empty base seals')

    def check(self, ref: dict[str, str]) -> None:
        require(ref['path'] in self.contexts, 'unqualified base seal')
        self.contexts[ref['path']].check(ref)

    def entry(self, ref: dict[str, str], base: Path) -> dict[str, Any]:
        self.check(ref)
        return self.contexts[ref['path']].entry(ref, base)


def _names(names: list[str] | tuple[str, ...]) -> list[str]:
    require(bool(names) and len(names) == len(set(names)) and set(names) <= TARGET_FIELDS,
                'unsupported target replacements')
    return sorted(names)


def begin_target_shard(base: Path, output: Path, seal_ref: dict[str, str], *,
                       replacements: tuple[str, ...], seal: BaseSeal | None = None) -> None:
    base_entry(seal_ref, base, seal=seal)
    names = _names(replacements)
    require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
    require(not output.exists(), 'overlay output already exists')
    original: Any = zarr.open_group(str(base), mode='r')
    require(all(name in original for name in names), 'base target missing')
    output.mkdir()
    for name in ('.zgroup', '.zattrs'):
        shutil.copyfile(base / name, output / name)
    for field in names:
        (output / field).mkdir()
        for name in ('.zarray', '.zattrs'):
            if (base / field / name).is_file():
                shutil.copyfile(base / field / name, output / field / name)


def _validate_local(path: Path, base: Path, names: list[str]) -> dict[str, Any]:
    local: Any = zarr.open_group(str(path), mode='r')
    original: Any = zarr.open_group(str(base), mode='r')
    require(set(local.array_keys()) == set(names), 'unexpected overlay arrays')
    require(dict(local.attrs) == dict(original.attrs), 'overlay changed inherited metadata/history')
    for name in names:
        a, b = local[name], original[name]
        require(a.shape == b.shape and np.dtype(a.dtype) == np.dtype(b.dtype),
                    'replacement layout differs')
        require(len(a.shape) == 2 and (name != 'search_wdl' or a.shape[1] == 3),
                    'invalid target shape')
        require(a.nchunks_initialized == a.nchunks, "missing replacement target chunk")
        # Validate values as well as chunk presence; a nonzero fill can look normalized.
        for start in range(0, a.shape[0], 8192):
            values = np.asarray(a[start:start + 8192], dtype=np.float64)
            require(bool(np.isfinite(values).all() and (values >= 0).all()
                             and np.allclose(values.sum(axis=1), 1, atol=.002, rtol=0)),
                        'invalid target probabilities')
            if name == 'policy_target':
                legal = np.asarray(original['legal_mask'][start:start + 8192])
                require(bool((values[legal == 0] == 0).all()), 'illegal target mass')
    return dict(local.attrs)


def finish_target_shard(base: Path, output: Path, seal_ref: dict[str, str], *,
                        recipe: dict[str, Any], seal: BaseSeal | None = None) -> None:
    entry = base_entry(seal_ref, base, seal=seal)
    local: Any = zarr.open_group(str(output), mode='r')
    names = _names(list(local.array_keys()))
    _validate_local(output, base, names)
    require(bool(recipe), 'missing target recipe')
    manifest = {'schema': 2, 'kind': 'immutable-target-overlay', 'base': str(base),
                'base_seal': seal_ref, 'base_content_sha256': entry['content_sha256'],
                'identity': entry['identity'], 'replacements': names, 'recipe': recipe,
                'target_content_sha256': {name: _plain_content(output / name) for name in names},
                'base_lifetime': 'Retain base and seal while any overlay depends on them.'}
    _atomic_new_json(output / MANIFEST, manifest)
    overlay_content_sha256(output, seal=seal)


def _open_target_manifest(path: Path, manifest: dict[str, Any], *,
                  seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    require(manifest.get('schema') == 2 and manifest.get('kind') == 'immutable-target-overlay',
                'unsupported overlay type/replacement')
    names = _names(manifest['replacements'])
    require(names == manifest['replacements'], 'noncanonical replacements')
    base = Path(manifest['base'])
    require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
    entry = base_entry(manifest['base_seal'], base, seal=seal)
    require(manifest['identity'] == entry['identity']
                and manifest['base_content_sha256'] == entry['content_sha256'], 'overlay base identity differs')
    require(set(manifest['target_content_sha256']) == set(names), 'replacement hashes differ')
    for name in names:
        require(_plain_content(path / name) == manifest['target_content_sha256'][name],
                    'overlay target changed')
    require(isinstance(manifest.get('recipe'), dict) and bool(manifest['recipe']), 'missing target recipe')
    return manifest, _validate_local(path, base, names)


def qualify_target_roots(roots: list[Path], output: Path) -> dict[str, Any]:
    roots = [root.resolve(strict=True) for root in roots]
    require(bool(roots) and len(set(roots)) == len(roots), 'duplicate/empty target roots')
    require(all(root not in output.resolve().parents and not root.name.endswith('.writing') for root in roots),
                'qualification must be outside completed roots')
    refs: dict[str, dict[str, str]] = {}
    entries = []
    root_files = {str(root): _root_files(root) for root in roots}
    for root in roots:
        paths = shard_paths(root)
        require(bool(paths), 'empty overlay root')
        for path in paths:
            manifest = json.loads((path / MANIFEST).read_bytes())
            require(manifest.get('schema') == 2, 'schema2 qualification requires schema2 shards')
            ref = manifest['base_seal']
            require(ref['path'] not in refs or refs[ref['path']] == ref, 'conflicting base seal')
            refs[ref['path']] = ref
            entries.append({'path': str(path), 'content_sha256': overlay_content_sha256(path),
                            'rows': manifest['identity']['rows']})
    result = {'schema': 2, 'status': TARGET_OVERLAY_STATUS, 'roots': [str(root) for root in roots],
              'root_files': root_files, 'base_seals': list(refs.values()), 'shards': entries,
              'rows': sum(entry['rows'] for entry in entries),
              'scope': 'Storage only; experiment must separately admit the target recipe.'}
    verify_receipt(result)
    _atomic_new_json(output, result)
    return result


def verify_receipt(receipt: dict[str, Any]) -> tuple[dict[Path, str], BaseSeals]:
    require(receipt.get('schema') == 2 and receipt.get('status') == TARGET_OVERLAY_STATUS, 'wrong target qualification')
    roots = [Path(root) for root in receipt['roots']]
    require(bool(roots) and len(set(roots)) == len(roots), 'duplicate/empty target roots')
    require(all(root.is_absolute() and root == root.resolve(strict=True) for root in roots),
                'noncanonical target roots')
    context = BaseSeals(receipt['base_seals'])
    for ref in receipt['base_seals']:
        single = context.contexts[ref['path']]
        require_base_corpus(ref, single.root, context=single)
    require({str(root): _root_files(root) for root in roots} == receipt['root_files'],
                'qualified overlay metadata changed')
    paths = [path for root in roots for path in shard_paths(root)]
    require([str(path) for path in paths] == [entry['path'] for entry in receipt['shards']],
                'qualified overlay membership changed')
    expected = {}
    bases = []
    for path, entry in zip(paths, receipt['shards'], strict=True):
        require(overlay_content_sha256(path, seal=context) == entry['content_sha256'],
                    'qualified overlay dependency/content changed')
        manifest = json.loads((path / MANIFEST).read_bytes())
        require(entry['rows'] == manifest['identity']['rows'], 'qualified row count differs')
        bases.append(manifest['base'])
        expected[path] = entry['content_sha256']
    require(len(set(bases)) == len(bases), 'duplicate base rows in target corpus')
    require(receipt['rows'] == sum(entry['rows'] for entry in receipt['shards']),
            'qualified total row count differs')
    return expected, context
