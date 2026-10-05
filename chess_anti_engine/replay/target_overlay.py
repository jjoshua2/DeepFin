"""Explicit, immutable policy overlays for qualified exact-epoch consumers.

The storage seal is a byte/row/history proof, not a recipe or strength verdict.
Schema 1 replaces policy only; schema 2 replaces policy and/or search WDL.
Both inherit ordinary sealed bases without chains.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import stat
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .codec_safety import _reject_unsafe_shard_codecs

MANIFEST = 'target_overlay.json'
BASE_STATUS = 'PASS_IMMUTABLE_BASE_STORAGE_SEAL'
OVERLAY_STATUS = 'PASS_IMMUTABLE_POLICY_OVERLAY_STORAGE_QUALIFICATION'
POLICY = 'policy_target'
TARGET_OVERLAY_STATUS = 'PASS_IMMUTABLE_TARGET_OVERLAY_STORAGE_QUALIFICATION'
TARGET_FIELDS = frozenset({'policy_target', 'search_wdl'})
# The pre-roster E corpus is accepted only through its independently pinned
# preparation completion and exact storage receipt. New overlays require a
# producer registration and intent roster; this is a frozen read-only adapter.
LEGACY_E_QUALIFIED_SHA256 = '4325d6bb319d278d5c844b0e16a82a3211dab1224d9c6443d0bbc25b82af1f03'
LEGACY_E_COMPLETE_SHA256 = 'ed21767547f8fe0f83918e86e1e1aa899c2aab52f7800f6b949c24059a98916a'
LEGACY_E_MANIFESTS_SHA256 = 'ca4e61bbae176ac0b6d1f32698aad9de299bc4f48de3035f2a430dac60ccf34c'
LEGACY_E_COHORT_COUNT = 35


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
    """Publish a new JSON authority by no-replace link and exact readback.

    The link is the filesystem commit point. A crash after that link can leave
    the final name installed even if this call did not return. Recovery must
    read the named bytes through an independently retained expected SHA pin
    and reverify their semantics; it must never retry by replacing that name.
    Ordinary post-link exceptions attempt rollback, but SIGKILL cannot.
    """
    require(not path.exists() and not path.is_symlink(), 'refusing existing storage receipt')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f'.{path.name}.{os.getpid()}.{time.time_ns()}.writing'
    linked = False
    link_attempted = False
    temporary_exists = False
    created: tuple[int, int] = (-1, -1)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        temporary_exists = True
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            st = os.fstat(stream.fileno())
            created = (st.st_dev, st.st_ino)
        link_attempted = True
        os.link(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        linked = True
        os.fsync(directory)
        os.unlink(temporary, dir_fd=directory)
        temporary_exists = False
        os.fsync(directory)
        # Receipt authority is the named bytes, not just a successful write to
        # the temporary inode. Read them back through the pinned directory.
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode)
                    and (before.st_dev, before.st_ino) == created
                    and before.st_size == len(raw),
                    'published storage receipt identity or size differs')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                observed = stream.read(len(raw) + 1)
            after = os.fstat(fd)
            named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            require((before.st_dev, before.st_ino, before.st_size,
                     before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size,
                     after.st_mtime_ns, after.st_ctime_ns)
                    and (named.st_dev, named.st_ino) == created
                    and observed == raw,
                    'published storage receipt readback differs')
        finally:
            os.close(fd)
    except BaseException:
        rollback_error: BaseException | None = None
        if link_attempted and not linked:
            try:
                named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except BaseException as error:
                rollback_error = error
            else:
                linked = (named.st_dev, named.st_ino) == created
        if linked:
            try:
                named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                require((named.st_dev, named.st_ino) == created,
                        'storage receipt changed before rollback')
                os.unlink(path.name, dir_fd=directory)
                os.fsync(directory)
                try:
                    os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise RuntimeError('storage receipt remained after rollback')
            except BaseException as error:
                rollback_error = error
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
            except BaseException:
                # A leftover .writing inode has no receipt authority.
                pass
        if rollback_error is not None:
            raise RuntimeError('AMBIGUOUS_STORAGE_RECEIPT_AUTHORITY') from rollback_error
        raise
    finally:
        # A close error after successful fsync/readback must not turn an
        # authoritative receipt into an apparent failed publication.
        try:
            os.close(directory)
        except OSError:
            pass


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _json_exact(left: Any, right: Any) -> bool:
    """Compare JSON values by canonical bytes, including scalar JSON types."""
    try:
        return _json_bytes({'value': left}) == _json_bytes({'value': right})
    except (TypeError, ValueError) as error:
        raise ValueError('non-JSON authority value') from error


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


@dataclass(frozen=True)
class _ValidatedOverlay:
    stamp: str
    manifest: dict[str, Any]
    attrs: dict[str, Any]
    content_sha256: str


def _directory_identity(path: Path) -> tuple[int, int, int, int, int]:
    require(path.is_dir() and not path.is_symlink(), 'invalid corpus root')
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


class BaseSeal:
    """One operation's validated seal/index; never a global unchecked cache."""

    def __init__(self, ref: dict[str, str]) -> None:
        self._validated_overlays: dict[Path, _ValidatedOverlay] = {}
        self._operation_roots: dict[Path, tuple[int, int, int, int, int]] = {}
        self._operation_files: dict[Path, tuple[int, int, int, int, int]] = {}
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

    def bind_roots(self, roots: list[Path]) -> None:
        """Anchor root membership/metadata before corpus qualification starts."""
        self.check_operation()
        for root in roots:
            if root in self._operation_roots:
                continue
            before = _directory_identity(root)
            for path in root.iterdir():
                require(not path.is_symlink(), 'linked corpus metadata/shard')
                if path.is_file():
                    self._operation_files[path] = _receipt_stamp(path)
            require(_directory_identity(root) == before, 'corpus membership changed during binding')
            self._operation_roots[root] = before

    def bind_receipt(self, path: Path, stamp: tuple[int, int, int, int, int]) -> None:
        require(_receipt_stamp(path) == stamp, 'qualification receipt identity changed')
        self._operation_files[path] = stamp

    def check_operation(self) -> None:
        for root, stamp in self._operation_roots.items():
            require(_directory_identity(root) == stamp, 'corpus membership changed during operation')
        for path, stamp in self._operation_files.items():
            require(_receipt_stamp(path) == stamp, 'operation receipt/metadata identity changed')

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


def _validate_manifest(path: Path, *, seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any], str]:
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
    local_content, targets = _plain_content_digests(path, (POLICY,))
    require(targets[POLICY] == manifest['target_content_sha256'], 'overlay target changed')
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
    return manifest, attrs, local_content


def _validated_overlay(path: Path, seal: BaseSeal) -> _ValidatedOverlay:
    """Reuse dense checks only while all anchored dependencies still match.

    The context belongs to one caller/epoch, never a process-global cache. Every
    access rechecks local membership/file identities, inherited base identities,
    receipt identities and qualified corpus membership. Changed inputs fail
    closed rather than refreshing the cached identity.
    """
    path = path.resolve(strict=True)
    seal.check_operation()
    before = tree_stamp(path)
    cached = seal._validated_overlays.get(path)
    if cached is not None:
        require(before == cached.stamp, 'validated overlay changed during operation')
        base_entry(cached.manifest['base_seal'], Path(cached.manifest['base']), seal=seal)
        require(tree_stamp(path) == before, 'overlay changed during identity read')
        seal.check_operation()
        return cached
    manifest, attrs, local = _validate_manifest(path, seal=seal)
    require(tree_stamp(path) == before, 'overlay changed during identity read')
    base_entry(manifest['base_seal'], Path(manifest['base']), seal=seal)
    seal.check_operation()
    content = hashlib.sha256(json.dumps({'kind': 'immutable-policy-overlay-v1',
        'base': manifest['base_content_sha256'], 'base_seal': manifest['base_seal'],
        'local': local}, sort_keys=True).encode()).hexdigest()
    validated = _ValidatedOverlay(before, copy.deepcopy(manifest), copy.deepcopy(attrs), content)
    seal._validated_overlays[path] = validated
    return validated


def _open_manifest(path: Path, *, seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    context = seal if seal is not None else seal_for_overlay(path)
    validated = _validated_overlay(path, context)
    # Consumers may attach metadata; never expose mutable cached dictionaries.
    return copy.deepcopy(validated.manifest), copy.deepcopy(validated.attrs)


def overlay_content_sha256(path: Path, *, seal: BaseSeal | None = None) -> str:
    path = path.resolve(strict=True)
    context = seal if seal is not None else seal_for_overlay(path)
    return _validated_overlay(path, context).content_sha256


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
    context.bind_roots([root, context.root])
    require_base_corpus(receipt['base_seal'], context.root, context=context)
    paths = shard_paths(root)
    require(bool(paths) and [p.name for p in paths] == [e['name'] for e in receipt['shards']],
            'qualified overlay membership changed')
    for path, expected in zip(paths, receipt['shards'], strict=True):
        require(overlay_content_sha256(path, seal=context) == expected['content_sha256'],
                'qualified overlay dependency/content changed')
    context.check_operation()
    return receipt


def shard_paths(root: Path) -> list[Path]:
    """The v1 layout uses ordinary local Zarr shard names exclusively."""
    return sorted(root.glob("shard_*.zarr"))


def _plain_content_digests(path: Path, names: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """Read each file once while retaining whole-tree and subtree hash formats."""
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"exact-epoch shard is not a directory: {path}")
    targets = {name: hashlib.sha256() for name in names}
    counts = dict.fromkeys(names, 0)
    for name in names:
        subtree = root / name
        require(not has_overlay(subtree), 'overlay chains are unsupported')
        if not subtree.resolve(strict=True).is_dir():
            raise ValueError(f"exact-epoch shard is not a directory: {subtree}")
    digest = hashlib.sha256()
    files_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            relative_path = file_path.relative_to(root)
            relative = relative_path.as_posix().encode("utf-8", errors="surrogateescape")
            name = relative_path.parts[0]
            target = targets.get(name) if len(relative_path.parts) > 1 else None
            before = file_path.stat()
            digest.update(struct.pack("<I", len(relative)))
            digest.update(relative)
            digest.update(struct.pack("<Q", int(before.st_size)))
            if target is not None:
                target_relative = file_path.relative_to(root / name).as_posix().encode(
                    "utf-8", errors="surrogateescape",
                )
                target.update(struct.pack("<I", len(target_relative)))
                target.update(target_relative)
                target.update(struct.pack("<Q", int(before.st_size)))
                counts[name] += 1
            bytes_read = 0
            with file_path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    bytes_read += len(block)
                    digest.update(block)
                    if target is not None:
                        target.update(block)
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
    if files_seen == 0 or any(count == 0 for count in counts.values()):
        raise ValueError(f"exact-epoch shard contains no files: {path}")
    return digest.hexdigest(), {name: value.hexdigest() for name, value in targets.items()}


def plain_content_sha256(path: Path) -> str:
    """Stream a deterministic digest over a Zarr tree's names and bytes."""
    return _plain_content_digests(path, ())[0]


def qualified_paths(ref: dict[str, str], paths: list[Path]) -> tuple[dict[Path, str], BaseSeal]:
    """Bind staging and every new epoch to the actual qualified corpus."""
    receipt_path = Path(ref['path'])
    receipt_stamp = _receipt_stamp(receipt_path)
    receipt = _read_pin(ref)
    if receipt.get('schema') == 2:
        legacy_ref = ref if 'intent_roster' not in receipt else None
        expected, contexts = verify_receipt(receipt, legacy_ref=legacy_ref)
        require([path.resolve(strict=True) for path in paths] == list(expected),
                'staged overlay paths/order differ from qualified corpus')
        contexts.bind_receipt(receipt_path, receipt_stamp)
        return expected, contexts
    root = Path(receipt['root'])
    context = BaseSeal(receipt['base_seal'])
    receipt = verify_qualification(ref, root, context=context)
    expected = {root / entry['name']: entry['content_sha256'] for entry in receipt['shards']}
    require([path.resolve(strict=True) for path in paths] == list(expected),
            'staged overlay paths/order differ from qualified corpus')
    context.bind_receipt(receipt_path, receipt_stamp)
    context.check_operation()
    return expected, context


class BaseSeals(BaseSeal):
    """Operation-local seal routing; compatible with exact-epoch storage consumers."""

    def __init__(self, refs: list[dict[str, str]], *,
                 legacy_recipes: dict[str, dict[str, Any]] | None = None) -> None:
        require(bool(refs), 'empty base seals')
        self.legacy_recipes = legacy_recipes
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


def _target_intent(base: Path, output: Path, seal_ref: dict[str, str],
                   names: list[str]) -> dict[str, Any]:
    return {'schema': 2, 'kind': 'target-overlay-begin-intent',
            'base': str(base), 'output': str(output), 'base_seal': seal_ref,
            'replacements': names}


def write_target_intent_roster(scopes: list[dict[str, Any]], roster_path: Path, *,
                               registration_ref: dict[str, str]) -> dict[str, str]:
    """Seal the ordered producer plan before beginning any overlay shard.

    The caller retains the returned SHA pin across restarts. If that pin is
    lost, an incomplete output cannot establish its own original intent.
    """
    require(bool(scopes), 'empty target intent roster')
    require(roster_path.is_absolute()
            and roster_path.parent == roster_path.parent.resolve(strict=False),
            'target intent roster path must be canonical')
    intents = []
    recipes = []
    for scope in scopes:
        base, output = Path(scope['base']), Path(scope['output'])
        seal_ref = scope['base_seal']
        names = _names(scope['replacements'])
        base_entry(seal_ref, base)
        require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
        require(output.is_absolute()
                and output.parent == output.parent.resolve(strict=False),
                'target output parent must be canonical')
        require(not output.exists(), 'target output already exists')
        require(output.parent not in roster_path.parents
                and base.parent not in roster_path.parents,
                'target intent roster must be outside overlay and base roots')
        recipe = scope['recipe']
        require(isinstance(recipe, dict) and bool(recipe),
                'target intent requires an exact nonempty recipe')
        intents.append(_target_intent(base, output, seal_ref, names))
        recipes.append(recipe)
    require(len({intent['output'] for intent in intents}) == len(intents),
            'duplicate target intent output')
    registration = _read_pin(registration_ref)
    registered_recipes = registration.get('recipes')
    require(registration.get('schema') == 2
            and registration.get('kind') == 'target-overlay-producer-registration'
            and isinstance(registered_recipes, list)
            and len(registered_recipes) == len(intents)
            and all(isinstance(recipe, dict) and bool(recipe)
                    for recipe in registered_recipes)
            and _json_exact(registered_recipes, recipes)
            and _json_exact(registration.get('intents'), intents),
            'target intent roster differs from pinned producer registration')
    roster = {'schema': 2, 'kind': 'immutable-target-intent-roster',
              'producer_registration': registration_ref, 'intents': intents}
    expected_sha = hashlib.sha256(_json_bytes(roster)).hexdigest()
    _atomic_new_json(roster_path, roster)
    require(sha(roster_path) == expected_sha, 'published target intent roster differs')
    return {'path': str(roster_path), 'sha256': expected_sha}


def _read_target_roster(ref: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(ref['path'])
    require(path == path.resolve(strict=True), 'target intent roster path changed')
    roster = _read_pin(ref)
    require(roster.get('schema') == 2 and roster.get('kind') == 'immutable-target-intent-roster',
            'wrong target intent roster')
    intents = roster['intents']
    if not isinstance(intents, list) or not intents:
        raise ValueError('empty target intent roster')
    registration = _read_pin(roster['producer_registration'])
    recipes = registration.get('recipes')
    if not isinstance(recipes, list) or len(recipes) != len(intents):
        raise ValueError('target intent roster differs from pinned producer registration')
    typed_intents: list[dict[str, Any]] = []
    for intent in intents:
        if not isinstance(intent, dict) or not intent:
            raise ValueError('invalid target intent scope')
        typed_intents.append(intent)
    typed_recipes: list[dict[str, Any]] = []
    for recipe in recipes:
        if not isinstance(recipe, dict) or not recipe:
            raise ValueError('target intent roster differs from pinned producer registration')
        typed_recipes.append(recipe)
    require(registration.get('schema') == 2
            and registration.get('kind') == 'target-overlay-producer-registration'
            and _json_exact(registration.get('intents'), intents),
            'target intent roster differs from pinned producer registration')
    outputs = []
    for intent in typed_intents:
        names = _names(intent['replacements'])
        base, output = Path(intent['base']), Path(intent['output'])
        require(base.is_absolute() and base == base.resolve(strict=True), 'noncanonical roster base')
        require(output.is_absolute() and output.parent == output.parent.resolve(strict=True),
                'noncanonical roster output')
        require(_json_exact(intent, _target_intent(base, output, intent['base_seal'], names)),
                'invalid target intent scope')
        require(output.parent not in path.parents and base.parent not in path.parents,
                'target intent roster must be outside overlay and base roots')
        outputs.append(str(output))
    require(len(set(outputs)) == len(outputs), 'duplicate target intent output')
    return typed_intents, typed_recipes


def _roster_scope(ref: dict[str, str], base: Path, output: Path,
                  seal_ref: dict[str, str]) -> tuple[list[str], dict[str, Any]]:
    intents, recipes = _read_target_roster(ref)
    matches = [(index, scope) for index, scope in enumerate(intents)
               if scope['output'] == str(output)]
    require(len(matches) == 1, 'target absent from authenticated intent roster')
    index, scope = matches[0]
    names = _names(scope['replacements'])
    require(_json_exact(scope, _target_intent(base, output, seal_ref, names)),
            'authenticated begin intent differs from target scope')
    return names, recipes[index]


def begin_target_shard(base: Path, output: Path, seal_ref: dict[str, str], *,
                       replacements: tuple[str, ...], roster_ref: dict[str, str],
                       seal: BaseSeal | None = None) -> None:
    base_entry(seal_ref, base, seal=seal)
    names = _names(replacements)
    require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
    authenticated_names, _recipe = _roster_scope(roster_ref, base, output, seal_ref)
    require(names == authenticated_names,
            'requested replacements differ from authenticated begin intent')
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
    # The ordinary shard loader checks these exact metadata rules before any
    # chunk decode. Qualification also reads chunks, so it must use that guard.
    replacements = {name: local[name] for name in names}
    _reject_unsafe_shard_codecs(replacements)
    require(dict(local.attrs) == dict(original.attrs), 'overlay changed inherited metadata/history')
    for name in names:
        a, b = replacements[name], original[name]
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
                        recipe: dict[str, Any], roster_ref: dict[str, str],
                        seal: BaseSeal | None = None) -> None:
    entry = base_entry(seal_ref, base, seal=seal)
    names, registered_recipe = _roster_scope(roster_ref, base, output, seal_ref)
    require(bool(recipe)
            and _json_exact(recipe, registered_recipe),
            'target recipe differs from pinned producer registration')
    local: Any = zarr.open_group(str(output), mode='r')
    require(set(local.array_keys()) == set(names), 'requested replacements are missing or differ')
    _validate_local(output, base, names)
    manifest = {'schema': 2, 'kind': 'immutable-target-overlay', 'base': str(base),
                'base_seal': seal_ref, 'base_content_sha256': entry['content_sha256'],
                'identity': entry['identity'], 'replacements': names, 'recipe': recipe,
                'intent_roster': roster_ref,
                'target_content_sha256': {name: _plain_content(output / name) for name in names},
                'base_lifetime': 'Retain base and seal while any overlay depends on them.'}
    _atomic_new_json(output / MANIFEST, manifest)
    overlay_content_sha256(output, seal=seal)


def _open_target_manifest(path: Path, manifest: dict[str, Any], *,
                  seal: BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any], str]:
    require(manifest.get('schema') == 2 and manifest.get('kind') == 'immutable-target-overlay',
                'unsupported overlay type/replacement')
    names = _names(manifest['replacements'])
    require(names == manifest['replacements'], 'noncanonical replacements')
    base = Path(manifest['base'])
    require(base.is_absolute() and base == base.resolve(strict=True), 'base must be canonical')
    if 'intent_roster' in manifest:
        roster_ref = manifest['intent_roster']
        canonical_output = path.resolve(strict=True)
        registered_names, registered_recipe = _roster_scope(
            roster_ref, base, canonical_output, manifest['base_seal'])
        require(names == registered_names,
            'target overlay replacements differ from authenticated begin intent')
        require(_json_exact(manifest.get('recipe'), registered_recipe),
            'target recipe differs from pinned producer registration')
    else:
        # The frozen E bank predates intent rosters. Its receipt is accepted
        # only via the exact pinned preparation completion in qualified_paths.
        if not isinstance(seal, BaseSeals) or seal.legacy_recipes is None:
            raise ValueError('pre-roster target requires the frozen E qualification')
        require(names == ['policy_target', 'search_wdl'],
                'frozen E target must replace policy and value')
        scope = seal.legacy_recipes.get(str(path.resolve(strict=True).parent))
        if scope is None or base.parent != Path(scope['base']):
            raise ValueError('frozen E target differs from pinned cohort scope')
        require(isinstance(manifest.get('recipe'), dict) and all(
            manifest['recipe'].get(key) == value for key, value in scope['recipe'].items()),
            'frozen E recipe differs from pinned cohort completion')
    entry = base_entry(manifest['base_seal'], base, seal=seal)
    require(manifest['identity'] == entry['identity']
                and manifest['base_content_sha256'] == entry['content_sha256'], 'overlay base identity differs')
    require(set(manifest['target_content_sha256']) == set(names), 'replacement hashes differ')
    local_content, targets = _plain_content_digests(path, tuple(names))
    for name in names:
        require(targets[name] == manifest['target_content_sha256'][name],
                    'overlay target changed')
    require(isinstance(manifest.get('recipe'), dict) and bool(manifest['recipe']), 'missing target recipe')
    return manifest, _validate_local(path, base, names), local_content


def qualify_target_roots(roots: list[Path], output: Path, *,
                         roster_ref: dict[str, str]) -> dict[str, Any]:
    roster_ref = dict(roster_ref)
    roots = [root.resolve(strict=True) for root in roots]
    require(bool(roots) and len(set(roots)) == len(roots), 'duplicate/empty target roots')
    require(all(root not in output.resolve().parents and not root.name.endswith('.writing') for root in roots),
                'qualification must be outside completed roots')
    refs: dict[str, dict[str, str]] = {}
    entries = []
    root_files = {str(root): _root_files(root) for root in roots}
    expected_paths = [path for root in roots for path in shard_paths(root)]
    intents, _recipe = _read_target_roster(roster_ref)
    require([str(path) for path in expected_paths] == [intent['output'] for intent in intents],
            'target paths/order differ from authenticated intent roster')
    for root in roots:
        paths = shard_paths(root)
        require(bool(paths), 'empty overlay root')
        for path in paths:
            manifest = json.loads((path / MANIFEST).read_bytes())
            require(manifest.get('schema') == 2, 'schema2 qualification requires schema2 shards')
            require(manifest['intent_roster'] == roster_ref,
                    'target shard intent roster differs from authoritative qualification')
            ref = manifest['base_seal']
            require(ref['path'] not in refs or refs[ref['path']] == ref, 'conflicting base seal')
            refs[ref['path']] = ref
            entries.append({'path': str(path), 'content_sha256': overlay_content_sha256(path),
                            'rows': manifest['identity']['rows']})
    result = {'schema': 2, 'status': TARGET_OVERLAY_STATUS, 'roots': [str(root) for root in roots],
              'root_files': root_files, 'base_seals': list(refs.values()), 'shards': entries,
              'intent_roster': roster_ref,
              'rows': sum(entry['rows'] for entry in entries),
              'scope': 'Storage only; experiment must separately admit the target recipe.'}
    verify_receipt(result)
    _atomic_new_json(output, result)
    return result


def _legacy_e_recipes(ref: dict[str, str], receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reconstruct only the historical E scopes from pinned JSON metadata."""
    require(ref['sha256'] == LEGACY_E_QUALIFIED_SHA256,
            'unrecognized pre-roster target qualification')
    # The frozen completion is the sibling of its pinned qualification receipt.
    # Its exact bytes remain independently fixed by this SHA.
    complete = _read_pin({
        'path': str(Path(ref['path']).with_name('complete.json')),
        'sha256': LEGACY_E_COMPLETE_SHA256,
    })
    require(complete.get('status') == 'COMPLETE_SFFREE_35_COHORTS'
            and complete.get('qualified') == ref
            and complete.get('rows') == receipt.get('rows')
            and complete.get('shards') == len(receipt['shards']),
            'frozen E completion differs from qualified receipt')
    plan = _read_pin(complete['plan'])
    require(plan.get('status') == 'READY_REVIEWED_FROZEN'
            and plan['manifests']['sha256'] == LEGACY_E_MANIFESTS_SHA256,
            'frozen E preparation plan differs')
    manifests = _read_pin(plan['manifests'])
    require(manifests.get('rows') == receipt['rows']
            and manifests.get('shards') == len(receipt['shards'])
            and len(manifests['cohorts']) == len(complete['cohorts'])
            == len(receipt['roots']) == LEGACY_E_COHORT_COUNT,
            'frozen E preparation cohort count differs')
    require(sum(item['rows'] for item in manifests['cohorts']) == receipt['rows']
            and sum(item['shards'] for item in manifests['cohorts']) == len(receipt['shards']),
            'frozen E planned row or shard total differs')
    recipes: dict[str, dict[str, Any]] = {}
    for root, cohort_ref, planned in zip(
        receipt['roots'], complete['cohorts'], manifests['cohorts'], strict=True,
    ):
        cohort = _read_pin(cohort_ref)
        recipe = cohort.get('recipe')
        require(cohort.get('status') == 'COMPLETE_SFFREE_TARGET_COHORT'
                and cohort.get('root') == root
                and isinstance(recipe, dict)
                and recipe.get('kind') == 'factorial58-sffree-v1'
                and recipe.get('arm') == 'E'
                and cohort['rows'] == planned['rows']
                and cohort['shards'] == planned['shards']
                and sum(item['rows'] for item in receipt['shards']
                        if Path(item['path']).parent == Path(root)) == cohort['rows']
                and sum(Path(item['path']).parent == Path(root)
                        for item in receipt['shards']) == cohort['shards'],
                'frozen E cohort differs from pinned preparation')
        recipes[root] = {'recipe': recipe, 'base': cohort['base']}
    require(len(recipes) == LEGACY_E_COHORT_COUNT, 'duplicate frozen E cohort roots')
    return recipes


def verify_receipt(receipt: dict[str, Any], *,
                   legacy_ref: dict[str, str] | None = None,
                   ) -> tuple[dict[Path, str], BaseSeals]:
    require(receipt.get('schema') == 2 and receipt.get('status') == TARGET_OVERLAY_STATUS, 'wrong target qualification')
    roots = [Path(root) for root in receipt['roots']]
    require(bool(roots) and len(set(roots)) == len(roots), 'duplicate/empty target roots')
    require(all(root.is_absolute() and root == root.resolve(strict=True) for root in roots),
                'noncanonical target roots')
    if 'intent_roster' in receipt:
        require(legacy_ref is None, 'rostered qualification cannot use legacy scopes')
        legacy_recipes = None
    else:
        if legacy_ref is None or _read_pin(legacy_ref) != receipt:
            raise ValueError('pre-roster qualification requires its exact pinned receipt')
        legacy_recipes = _legacy_e_recipes(legacy_ref, receipt)
        require(list(legacy_recipes) == receipt['roots'],
                'pre-roster qualification requires pinned E cohort scopes')
    context = BaseSeals(receipt['base_seals'], legacy_recipes=legacy_recipes)
    context.bind_roots([*roots, *(single.root for single in context.contexts.values())])
    for ref in receipt['base_seals']:
        single = context.contexts[ref['path']]
        require_base_corpus(ref, single.root, context=single)
    require({str(root): _root_files(root) for root in roots} == receipt['root_files'],
                'qualified overlay metadata changed')
    paths = [path for root in roots for path in shard_paths(root)]
    require([str(path) for path in paths] == [entry['path'] for entry in receipt['shards']],
                'qualified overlay membership changed')
    if 'intent_roster' in receipt:
        intents, _recipe = _read_target_roster(receipt['intent_roster'])
        require([str(path) for path in paths] == [intent['output'] for intent in intents],
                'qualified paths/order differ from authenticated intent roster')
    expected = {}
    bases = []
    for path, entry in zip(paths, receipt['shards'], strict=True):
        require(overlay_content_sha256(path, seal=context) == entry['content_sha256'],
                    'qualified overlay dependency/content changed')
        manifest = json.loads((path / MANIFEST).read_bytes())
        if 'intent_roster' in receipt:
            require(manifest['intent_roster'] == receipt['intent_roster'],
                    'qualified intent roster changed')
        else:
            require('intent_roster' not in manifest,
                    'frozen E target changed its intent type')
        require(entry['rows'] == manifest['identity']['rows'], 'qualified row count differs')
        bases.append(manifest['base'])
        expected[path] = entry['content_sha256']
    require(len(set(bases)) == len(bases), 'duplicate base rows in target corpus')
    require(receipt['rows'] == sum(entry['rows'] for entry in receipt['shards']),
            'qualified total row count differs')
    context.check_operation()
    return expected, context
