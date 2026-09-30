"""Opt-in, source-bound durable receipts for a fixed paired arena.

The JSONL game row commits after the PGN game. A pair receipt is published
only after both JSONL rows have been fsynced; a crash between that commit and
the receipt is repaired from the validated rows on the next attempt.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SEAL_SCHEMA = "arena_durable_source_seal_v1"
CATALOG_SCHEMA = "arena_syzygy_metadata_inventory_v1"
CATALOG_STATUS = "PASS_METADATA_INVENTORY_ONLY"
RECEIPT_SCHEMA = "arena_durable_pair_receipt_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def runtime_record() -> dict[str, object]:
    """Bind the interpreter, numeric libraries and search-affecting environment."""
    import chess
    import numpy as np
    import torch

    names = (
        "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
        "TORCHINDUCTOR_CACHE_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "chess": chess.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "env": {name: os.environ.get(name) for name in names},
    }


def file_span(path: Path, offset: int, length: int) -> dict[str, int | str]:
    """Hash exactly the appended bytes after their writer fsyncs."""
    if offset < 0 or not 0 < length <= 8 * 1024 * 1024:
        raise ValueError("empty, negative or implausibly large durable file span")
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read(length)
    if len(data) != length:
        raise ValueError(f"short durable file span: {path}")
    return {"offset": offset, "length": length, "sha256": _sha(data)}


def verify_span(path: Path, span: object) -> None:
    if not isinstance(span, dict) or set(span) != {"offset", "length", "sha256"}:
        raise ValueError(f"invalid durable span for {path}")
    offset, length, digest = span["offset"], span["length"], span["sha256"]
    if type(offset) is not int or type(length) is not int or not isinstance(digest, str):
        raise ValueError(f"invalid durable span types for {path}")
    if not _SHA.fullmatch(digest) or file_span(path, offset, length)["sha256"] != digest:
        raise ValueError(f"durable span changed: {path}")


def _hash_file(path: Path, *, expected_size: int, expected_sha: str) -> None:
    if type(expected_size) is not int or expected_size < 0 or not _SHA.fullmatch(expected_sha):
        raise ValueError(f"invalid file seal for {path}")
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size:
        raise ValueError(f"sealed file missing or size changed: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"sealed file bytes changed: {path}")


def _file_entry(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source-seal input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _validate_tablebase_catalog(catalog: dict[str, Any], syzygy_path: str) -> None:
    """Recheck metadata identity, with no tablebase-content hash claim."""
    if (catalog.get("schema") != CATALOG_SCHEMA
            or catalog.get("status") != CATALOG_STATUS):
        raise ValueError("tablebase metadata inventory has wrong schema/status")
    roots = [str(Path(p).resolve()) for p in syzygy_path.split(os.pathsep)]
    if catalog.get("roots") != roots or not roots or any(not Path(p).is_dir() for p in roots):
        raise ValueError("tablebase roots differ from frozen inventory")
    entries = catalog.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("tablebase catalog has no files")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "mtime_ns"}:
            raise ValueError("invalid tablebase catalog entry")
        path = Path(entry["path"])
        name = str(path)
        if name in seen or path.suffix.lower() not in {".rtbw", ".rtbz"}:
            raise ValueError(f"duplicate or non-tablebase catalog entry: {path}")
        seen.add(name)
        if not any(path.parent == Path(root) for root in roots):
            raise ValueError(f"catalog entry outside frozen roots: {path}")
        stat = path.stat()
        if (not path.is_file() or path.is_symlink()
                or type(entry["bytes"]) is not int
                or type(entry["mtime_ns"]) is not int
                or stat.st_size != entry["bytes"]
                or stat.st_mtime_ns != entry["mtime_ns"]):
            raise ValueError(f"tablebase identity changed: {path}")
    actual = {
        str(path) for root in roots for path in Path(root).iterdir()
        if path.suffix.lower() in {".rtbw", ".rtbz"}
    }
    if actual != seen:
        raise ValueError("tablebase catalog inventory changed")
    if {Path(path).suffix.lower() for path in seen} != {".rtbw", ".rtbz"}:
        raise ValueError("tablebase inventory lacks WDL or DTZ files")


def prepare_tablebase_inventory(output: Path, syzygy_path: str) -> str:
    """Seal tablebase names/size/mtime without reading tablebase file bytes."""
    if output.exists():
        raise ValueError(f"tablebase inventory already exists: {output}")
    parts = syzygy_path.split(os.pathsep)
    if not parts or any(not part or part != part.strip() for part in parts):
        raise ValueError("invalid Syzygy path components")
    roots = [str(Path(part).resolve()) for part in parts]
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate Syzygy root")
    entries: list[dict[str, object]] = []
    for root in roots:
        directory = Path(root)
        if not directory.is_dir():
            raise ValueError(f"Syzygy root is missing: {root}")
        for path in directory.iterdir():
            if path.suffix.lower() not in {".rtbw", ".rtbz"}:
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Syzygy entry is not a regular file: {path}")
            stat = path.stat()
            entries.append({"path": str(path), "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns})
    entries.sort(key=lambda entry: str(entry["path"]))
    catalog = {
        "schema": CATALOG_SCHEMA, "status": CATALOG_STATUS,
        "roots": roots, "files": entries,
    }
    _validate_tablebase_catalog(catalog, syzygy_path)
    raw = _json_bytes(catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _sha(raw)


def validate_source_seal(
    path: Path, expected_sha: str, *, argv: list[str],
    candidate: str, reference: str, openings: Path, config: Path,
    syzygy_path: str,
) -> str:
    """Read a frozen manifest and validate every cheap per-attempt input.

    Tablebase file *metadata* is sealed by the catalog's own hash. Per attempt
    we recheck exact resolved path/size/mtime inventory, not 235 GB of bytes.
    This does not authenticate tablebase content; runtime strict WDL/DTZ
    availability and probes remain separate checks.
    """
    if not _SHA.fullmatch(expected_sha):
        raise ValueError("invalid expected source-seal SHA256")
    raw = path.read_bytes()
    if _sha(raw) != expected_sha:
        raise ValueError("source-seal bytes changed")
    seal = json.loads(raw)
    if seal.get("schema") != SEAL_SCHEMA or seal.get("argv") != argv:
        raise ValueError("durable arena argv or seal schema changed")
    if seal.get("profile") != {"pairs": 576, "games": 1152, "sprt": False,
                                "syzygy_max_pieces": 6, "rule50_aware": True}:
        raise ValueError("durable arena profile is not the frozen 576-pair route")
    if seal.get("runtime") != runtime_record():
        raise ValueError("durable arena interpreter or numeric runtime changed")
    files = seal.get("files")
    native_roles = {
        "native_encoding": "chess_anti_engine.encoding._lc0_ext",
        "native_features": "chess_anti_engine.encoding._features_ext",
        "native_mcts": "chess_anti_engine.mcts._mcts_tree",
    }
    native_paths: dict[str, Path] = {}
    for role, module in native_roles.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"required native extension is unavailable: {module}")
        native_paths[role] = Path(spec.origin).resolve()
    required = {
        "candidate": Path(candidate).resolve(),
        "reference": Path(reference).resolve(),
        "openings": openings.resolve(),
        "production_config": config.resolve(),
        **native_paths,
    }
    if not isinstance(files, dict) or not set(required).issubset(files):
        raise ValueError("source seal lacks required model/opening/config files")
    for role, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError(f"invalid sealed file entry: {role}")
        item = Path(entry["path"])
        if not item.is_absolute() or item.resolve() != item:
            raise ValueError(f"sealed path must be absolute and resolved: {role}")
        if role in required and item != required[role]:
            raise ValueError(f"sealed {role} path changed")
        _hash_file(item, expected_size=entry["bytes"], expected_sha=entry["sha256"])
    source = seal.get("source")
    if not isinstance(source, dict) or set(source) != {"root", "git_head"}:
        raise ValueError("source seal lacks exact Git source identity")
    root = Path(source["root"])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("source root is not absolute and resolved")
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("source root differs from imported arena package")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != source["git_head"]:
        raise ValueError("arena source commit changed")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise ValueError("arena source tracked files are dirty")
    tablebase = seal.get("tablebase")
    if not isinstance(tablebase, dict) or set(tablebase) != {"path", "sha256"}:
        raise ValueError("source seal lacks tablebase metadata inventory")
    catalog_path = Path(tablebase["path"])
    if not catalog_path.is_absolute() or catalog_path.resolve() != catalog_path:
        raise ValueError("tablebase catalog path must be absolute and resolved")
    catalog_raw = catalog_path.read_bytes()
    if _sha(catalog_raw) != tablebase["sha256"]:
        raise ValueError("tablebase metadata inventory bytes changed")
    _validate_tablebase_catalog(json.loads(catalog_raw), syzygy_path)
    return expected_sha


def prepare_source_seal(
    output: Path, *, argv: list[str], candidate: Path, reference: Path,
    openings: Path, config: Path, catalog_path: Path, syzygy_path: str,
) -> str:
    """Create one immutable seal from a frozen tablebase metadata inventory.

    The caller supplies the exact arena argv minus the seal-path/SHA flags.
    Preparing this seal hashes model bytes once; validation rehashes them at
    every bounded attempt. Tablebase file bytes are not hashed or claimed
    qualified by this helper.
    """
    if output.exists():
        raise ValueError(f"source seal already exists: {output}")
    catalog_path = catalog_path.resolve()
    catalog_raw = catalog_path.read_bytes()
    _validate_tablebase_catalog(json.loads(catalog_raw), syzygy_path)
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise ValueError("arena source tracked files are dirty")
    native_roles = {
        "native_encoding": "chess_anti_engine.encoding._lc0_ext",
        "native_features": "chess_anti_engine.encoding._features_ext",
        "native_mcts": "chess_anti_engine.mcts._mcts_tree",
    }
    files = {
        "candidate": _file_entry(candidate),
        "reference": _file_entry(reference),
        "openings": _file_entry(openings),
        "production_config": _file_entry(config),
    }
    for role, module in native_roles.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"required native extension is unavailable: {module}")
        files[role] = _file_entry(Path(spec.origin))
    seal = {
        "schema": SEAL_SCHEMA,
        "argv": argv,
        "profile": {"pairs": 576, "games": 1152, "sprt": False,
                    "syzygy_max_pieces": 6, "rule50_aware": True},
        "runtime": runtime_record(),
        "files": files,
        "source": {"root": str(root), "git_head": head},
        "tablebase": {"path": str(catalog_path),
                      "sha256": _sha(catalog_raw)},
    }
    raw = _json_bytes(seal)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _sha(raw)


class DurablePairReceipts:
    """Atomically seal both durable game rows and their exact PGN byte spans."""

    def __init__(self, directory: Path, *, seal_sha256: str, log: Path, pgn: Path):
        self.directory = directory
        self.seal_sha256 = seal_sha256
        self.log = log
        self.pgn = pgn
        self.pending: dict[int, dict[str, dict[str, object]]] = {}
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = (directory / ".arena.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise ValueError("another durable arena owns the pair receipts") from exc
        parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def close(self) -> None:
        self._lock.close()

    def _path(self, pair_id: int) -> Path:
        return self.directory / f"pair_{pair_id:06d}.json"

    def _payload(self, pair_id: int, halves: dict[str, dict[str, object]]) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA, "seal_sha256": self.seal_sha256,
            "pair_id": pair_id, "halves": halves,
        }

    def _publish(self, pair_id: int, halves: dict[str, dict[str, object]]) -> None:
        if set(halves) != {"0", "1"}:
            raise ValueError("a pair receipt requires both colorings")
        final = self._path(pair_id)
        raw = _json_bytes(self._payload(pair_id, halves))
        if final.exists():
            if final.is_symlink():
                raise ValueError(f"pair receipt is a symlink: {final}")
            if final.read_bytes() != raw:
                raise ValueError(f"pair receipt changed: {final}")
            return
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.directory, prefix=f".{final.name}.",
            delete=False,
        ) as fh:
            temp = Path(fh.name)
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, final)
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def record_game(
        self, pair_id: int, half: int, *, jsonl_span: dict[str, int | str],
        pgn_span: dict[str, int | str],
    ) -> None:
        halves = self.pending.setdefault(pair_id, {})
        if str(half) in halves:
            raise ValueError(f"same pair half written twice: {pair_id}/{half}")
        halves[str(half)] = {"jsonl": jsonl_span, "pgn": pgn_span}
        if len(halves) == 2:
            self._publish(pair_id, halves)
            del self.pending[pair_id]

    def recover_and_verify(self, complete_pair_ids: list[int]) -> None:
        """Verify retained pairs, recovering only a receipt-missing commit gap."""
        needed = set(complete_pair_ids)
        recorded: dict[int, dict[str, dict[str, object]]] = {}
        offset = 0
        with self.log.open("rb") as fh:
            for line in fh:
                length = len(line)
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Existing reader separately refuses nonterminal corruption.
                    offset += length
                    continue
                if row.get("kind") == "game" and row.get("pair_id") in needed:
                    if not line.endswith(b"\n"):
                        raise ValueError("durable game row lacks commit newline")
                    pair_id = int(row["pair_id"])
                    half = str(int(row["half"]))
                    pgn_span = row.get("durable_pgn_span")
                    recorded.setdefault(pair_id, {})[half] = {
                        "jsonl": {"offset": offset, "length": length,
                                  "sha256": _sha(line)},
                        "pgn": pgn_span,
                    }
                offset += length
        for pair_id in complete_pair_ids:
            halves = recorded.get(pair_id, {})
            if set(halves) != {"0", "1"}:
                raise ValueError(f"complete pair lacks two durable rows: {pair_id}")
            for half in halves.values():
                verify_span(self.pgn, half["pgn"])
            self._publish(pair_id, halves)
        observed = {p.name for p in self.directory.glob("pair_*.json")}
        expected = {self._path(pair_id).name for pair_id in complete_pair_ids}
        if observed != expected:
            raise ValueError("pair receipt inventory differs from complete pairs")
