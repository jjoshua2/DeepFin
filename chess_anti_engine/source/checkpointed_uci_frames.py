"""Lossless, bounded Zstd frames for pre-banked raw Stockfish JSONL blocks.

This is an isolated storage primitive, not a live label recipe. A caller must
provide one independently sealed input block and its source/schema pins.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd

MAX_ROWS = 2048
MAX_RAW_BYTES = 8 << 20
MAX_FRAME_BYTES = MAX_RAW_BYTES + (128 << 10)
LEVEL = 3
SCHEMA = "sf_raw_uci_frame_v1"
_FILES = {".lock", "CLAIM.json", "FRAME.zst", "RECEIPT.json"}
_STAGES = (".CLAIM.json.part-", ".FRAME.zst.part-", ".RECEIPT.json.part-")


class FrameError(ValueError):
    """A source, frame, or checkpoint fails its sealed contract."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise FrameError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hex64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_new(path: Path, data: bytes) -> None:
    _need(not path.exists(), f"sealed file already exists: {path.name}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.part-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
        _fsync_dir(path.parent)
    except BaseException:
        # The uniquely named stage remains for the next locked invocation.
        raise


def _load_canonical(path: Path) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"missing or linked {path.name}")
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"invalid {path.name}") from exc
    _need(isinstance(value, dict) and data == _canonical(value),
          f"noncanonical {path.name}")
    return value


def _check_rows(raw: bytes, expected_rows: int) -> None:
    _need(raw.endswith(b"\n"), "unterminated raw JSONL row")
    lines = raw.splitlines(keepends=True)
    _need(len(lines) == expected_rows and 0 < len(lines) <= MAX_ROWS,
          "raw JSONL row count")
    for line in lines:
        _need(line.endswith(b"\n") and bool(line.strip()),
              "blank or torn raw JSONL row")
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FrameError("invalid raw JSONL row") from exc
        _need(isinstance(value, dict), "raw JSONL row must be an object")


@dataclass(frozen=True)
class FrameSpec:
    source_sha256: str
    schema_sha256: str
    config_sha256: str
    input_sha256: str
    block_index: int
    rows: int
    raw_bytes: int

    def validated(self) -> FrameSpec:
        _need(all(_hex64(value) for value in (
            self.source_sha256, self.schema_sha256,
            self.config_sha256, self.input_sha256)), "source/schema/config/input pins")
        _need(type(self.block_index) is int and 0 <= self.block_index < 1_000_000_000,
              "block index")
        _need(type(self.rows) is int and 0 < self.rows <= MAX_ROWS,
              "frame row cap")
        _need(type(self.raw_bytes) is int and 0 < self.raw_bytes <= MAX_RAW_BYTES,
              "frame raw-byte cap")
        return self


def _read_input(path: Path, spec: FrameSpec) -> bytes:
    _need(path.is_file() and not path.is_symlink(), "missing or linked input block")
    with path.open("rb") as source:
        raw = source.read(MAX_RAW_BYTES + 1)
        _need(len(raw) <= MAX_RAW_BYTES and source.read(1) == b"",
              "input block exceeds byte cap")
    _need(len(raw) == spec.raw_bytes and _sha(raw) == spec.input_sha256,
          "input block byte/hash mismatch")
    _check_rows(raw, spec.rows)
    return raw


def _claim(spec: FrameSpec) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_sha256": spec.source_sha256,
        "schema_sha256": spec.schema_sha256,
        "config_sha256": spec.config_sha256,
        "input_sha256": spec.input_sha256,
        "code_sha256": _sha(Path(__file__).read_bytes()),
        "block_index": spec.block_index,
        "rows": spec.rows,
        "raw_bytes": spec.raw_bytes,
        "codec": "zstd-single-frame",
        "zstandard_python": zstd.__version__,
        "zstd_library": list(zstd.ZSTD_VERSION),
        "level": LEVEL,
        "threads": 0,
        "write_content_size": True,
        "write_checksum": True,
        "write_dict_id": False,
        "max_rows": MAX_ROWS,
        "max_raw_bytes": MAX_RAW_BYTES,
    }


def _receipt(claim: dict[str, Any], frame: bytes) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "claim_sha256": _sha(_canonical(claim)),
        "input_sha256": claim["input_sha256"],
        "raw_bytes": claim["raw_bytes"],
        "rows": claim["rows"],
        "frame_sha256": _sha(frame),
        "frame_bytes": len(frame),
    }


def _decode(frame: bytes, expected_raw_bytes: int) -> bytes:
    _need(0 < len(frame) <= MAX_FRAME_BYTES, "compressed frame byte cap")
    try:
        size = zstd.frame_content_size(frame)
        _need(size == expected_raw_bytes, "Zstd content size")
        decoder = zstd.ZstdDecompressor().decompressobj()
        raw = decoder.decompress(frame)
        _need(decoder.eof and not decoder.unused_data and
              not decoder.unconsumed_tail, "trailing or truncated Zstd frame")
    except zstd.ZstdError as exc:
        raise FrameError("corrupt Zstd frame") from exc
    _need(len(raw) == expected_raw_bytes, "decoded byte count")
    return raw


def _verify_seal(directory: Path, raw: bytes,
                 claim: dict[str, Any]) -> dict[str, Any]:
    _need({part.name for part in directory.iterdir()} == _FILES,
          "unexpected sealed frame files")
    _need(_load_canonical(directory / "CLAIM.json") == claim,
          "source/schema/config/code claim changed")
    receipt = _load_canonical(directory / "RECEIPT.json")
    frame_path = directory / "FRAME.zst"
    _need(frame_path.is_file() and not frame_path.is_symlink(),
          "missing or linked compressed frame")
    with frame_path.open("rb") as source:
        frame = source.read(MAX_FRAME_BYTES + 1)
        _need(len(frame) <= MAX_FRAME_BYTES and source.read(1) == b"",
              "compressed frame byte cap")
    _need(receipt == _receipt(claim, frame), "frame receipt mismatch")
    decoded = _decode(frame, len(raw))
    _need(decoded == raw, "decoded raw UCI bytes differ from source block")
    _check_rows(decoded, claim["rows"])
    return receipt


def _recover_unsealed(directory: Path, claim: dict[str, Any]) -> None:
    names = {part.name for part in directory.iterdir()}
    _need("RECEIPT.json" not in names, "sealed receipt needs verification")
    _need(all(name in _FILES or name.startswith(_STAGES) for name in names),
          "unexpected unsealed frame files")
    _need(".lock" in names, "missing frame lock")
    if "CLAIM.json" in names:
        _need(_load_canonical(directory / "CLAIM.json") == claim,
              "unsealed source/schema/config/code claim changed")
    else:
        _need("FRAME.zst" not in names, "frame without source claim")
    for part in directory.iterdir():
        if part.name.startswith(_STAGES) or part.name == "FRAME.zst":
            _need(not part.is_dir(), "unexpected stage directory")
            part.unlink()
    _fsync_dir(directory)


def pack_frame(input_block: Path, frames_root: Path,
               spec: FrameSpec) -> tuple[dict[str, Any], bool]:
    """Seal one bounded block; return ``(receipt, reused_existing_seal)``.

    No whole-source scan occurs. The caller owns an upstream sealed block
    inventory; each invocation authenticates only this block and its frame.
    """
    spec.validated()
    frames_root = Path(frames_root)
    _need(frames_root.parent.is_dir(), "frame-root parent missing")
    if not frames_root.exists():
        frames_root.mkdir()
        _fsync_dir(frames_root.parent)
    _need(frames_root.is_dir() and not frames_root.is_symlink(),
          "invalid frame root")
    directory = frames_root / f"block-{spec.block_index:09d}"
    if not directory.exists():
        directory.mkdir()
        _fsync_dir(frames_root)
    _need(directory.is_dir() and not directory.is_symlink(),
          "invalid block directory")
    lock_fd = os.open(directory / ".lock",
                      os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrameError("block already has a writer") from exc
        _fsync_dir(directory)
        raw = _read_input(Path(input_block), spec)
        claim = _claim(spec)
        if (directory / "RECEIPT.json").exists():
            return _verify_seal(directory, raw, claim), True
        _recover_unsealed(directory, claim)
        if not (directory / "CLAIM.json").exists():
            _atomic_new(directory / "CLAIM.json", _canonical(claim))
        frame = zstd.ZstdCompressor(
            level=LEVEL, threads=0, write_content_size=True,
            write_checksum=True, write_dict_id=False,
        ).compress(raw)
        _need(len(frame) <= MAX_FRAME_BYTES, "compressed frame exceeds cap")
        _atomic_new(directory / "FRAME.zst", frame)
        receipt = _receipt(claim, frame)
        _atomic_new(directory / "RECEIPT.json", _canonical(receipt))
        return _verify_seal(directory, raw, claim), False
    finally:
        os.close(lock_fd)
