"""Bounded lossless raw-Stockfish-frame and crash-recovery witnesses."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import zstandard as zstd

from chess_anti_engine.source import checkpointed_uci_frames as frames


SOURCE = "a" * 64
SCHEMA = "b" * 64
CONFIG = "c" * 64


def _rows(count: int) -> bytes:
    rows = []
    for n in range(count):
        row = {
            "source_uid": ["SF", "selected", n // 60, n % 60],
            "fen": f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - {n % 60} 1",
            "uci_raw": [
                "info depth 8 seldepth 14 multipv 1 score cp 34 nodes 6431 nps 971000 "
                "pv e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
                "info depth 12 seldepth 20 multipv 2 score cp 18 nodes 45786 nps 1064000 "
                "pv d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
                f"bestmove {'e2e4' if n % 2 else 'd2d4'} ponder e7e5",
            ],
            "searchmoves": ["e2e4", "d2d4", "g1f3"],
            "nodes": 45786 + n,
        }
        # Preserve deliberate noncanonical whitespace as well as every UCI line.
        rows.append(json.dumps(row, ensure_ascii=False).encode() + b" \n")
    return b"".join(rows)


def _block(tmp_path: Path, index: int, count: int) -> tuple[Path, frames.FrameSpec]:
    raw = _rows(count)
    path = tmp_path / f"raw-{index}.jsonl"
    path.write_bytes(raw)
    return path, frames.FrameSpec(
        SOURCE, SCHEMA, CONFIG, hashlib.sha256(raw).hexdigest(),
        index, count, len(raw),
    )


def _frame_path(root: Path, index: int) -> Path:
    return root / f"block-{index:09d}" / "FRAME.zst"


def _rewrite_receipt(root: Path, index: int, frame: bytes) -> None:
    directory = _frame_path(root, index).parent
    path = directory / "RECEIPT.json"
    value = json.loads(path.read_bytes())
    value["frame_sha256"] = hashlib.sha256(frame).hexdigest()
    value["frame_bytes"] = len(frame)
    path.write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def test_independent_decode_preserves_every_raw_byte_and_noop(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 128)
    root = tmp_path / "frames"
    receipt, reused = frames.pack_frame(source, root, spec)
    assert not reused
    compressed = _frame_path(root, 0).read_bytes()
    assert zstd.ZstdDecompressor().decompress(compressed) == source.read_bytes()
    assert receipt["rows"] == 128
    assert receipt["raw_bytes"] == source.stat().st_size
    sealed = (root / "block-000000000" / "RECEIPT.json").read_bytes()
    assert frames.pack_frame(source, root, spec) == (receipt, True)
    assert (root / "block-000000000" / "RECEIPT.json").read_bytes() == sealed


@pytest.mark.parametrize("damage", ["flip", "truncate", "junk", "second-frame"])
def test_corrupted_or_extra_frame_refuses_even_with_rewritten_receipt(
    tmp_path: Path, damage: str,
) -> None:
    source, spec = _block(tmp_path, 0, 17)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    frame_path = _frame_path(root, 0)
    original = frame_path.read_bytes()
    if damage == "flip":
        changed = bytearray(original)
        changed[-5] ^= 0x40
        corrupt = bytes(changed)
    elif damage == "truncate":
        corrupt = original[:-7]
    elif damage == "junk":
        corrupt = original + b"extra bytes"
    else:
        corrupt = original + original
    frame_path.write_bytes(corrupt)
    _rewrite_receipt(root, 0, corrupt)
    with pytest.raises(frames.FrameError):
        frames.pack_frame(source, root, spec)


def test_oversized_expansion_refuses_even_with_rewritten_receipt(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 1)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    oversized = zstd.ZstdCompressor().compress(b"x" * (frames.MAX_RAW_BYTES + 1))
    _frame_path(root, 0).write_bytes(oversized)
    _rewrite_receipt(root, 0, oversized)
    with pytest.raises(frames.FrameError, match="Zstd content size"):
        frames.pack_frame(source, root, spec)


def test_forged_small_content_size_refuses_before_expansion() -> None:
    compressed = zstd.ZstdCompressor(write_checksum=True).compress(b"x" * 100_000)
    assert compressed[4] & 0xE0 == 0xA0  # Four-byte content size, single segment.
    forged = compressed[:5] + (1).to_bytes(4, "little") + compressed[9:]
    assert zstd.get_frame_parameters(forged).content_size == 1
    with pytest.raises(frames.FrameError):
        frames._decode(forged, 1)


def test_oversized_window_header_refuses_before_decompression() -> None:
    compressed = zstd.ZstdCompressor(write_checksum=True).compress(b"x" * 100_000)
    assert compressed[4] & 0x20  # Single-segment frame has no window descriptor.
    forged = compressed[:4] + bytes([compressed[4] & ~0x20, 0x80]) + compressed[5:]
    assert zstd.get_frame_parameters(forged).window_size > frames.MAX_RAW_BYTES
    with pytest.raises(frames.FrameError, match="Zstd window cap"):
        frames._decode(forged, 100_000)


def test_changed_source_schema_or_code_claim_refuses(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    source, spec = _block(tmp_path, 0, 3)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    with pytest.raises(frames.FrameError, match="claim changed"):
        frames.pack_frame(source, root, replace(spec, schema_sha256="d" * 64))
    with pytest.raises(frames.FrameError, match="claim changed"):
        frames.pack_frame(source, root, replace(spec, source_sha256="e" * 64))
    original = frames._claim
    monkeypatch.setattr(frames, "_claim", lambda s: {
        **original(s), "code_sha256": "f" * 64,
    })
    with pytest.raises(frames.FrameError, match="claim changed"):
        frames.pack_frame(source, root, spec)


def test_input_byte_change_and_oversize_refuse(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 2)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    source.write_bytes(source.read_bytes().replace(b"depth 8", b"depth 9", 1))
    with pytest.raises(frames.FrameError, match="input block byte/hash"):
        frames.pack_frame(source, root, spec)
    with pytest.raises(frames.FrameError, match="frame row cap"):
        frames.pack_frame(source, root, replace(spec, rows=2049))
    with pytest.raises(frames.FrameError, match="frame raw-byte cap"):
        frames.pack_frame(source, root, replace(spec, raw_bytes=frames.MAX_RAW_BYTES + 1))


def test_unsealed_owned_stage_recovers_but_foreign_file_refuses(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 8)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    directory = _frame_path(root, 0).parent
    (directory / "RECEIPT.json").unlink()
    (directory / ".FRAME.zst.part-dead-worker").write_bytes(b"incomplete")
    _, reused = frames.pack_frame(source, root, spec)
    assert not reused
    assert not (directory / ".FRAME.zst.part-dead-worker").exists()
    (directory / "foreign-file").write_bytes(b"x")
    with pytest.raises(frames.FrameError, match="unexpected sealed"):
        frames.pack_frame(source, root, spec)


def test_oversized_receipt_refuses_before_parsing(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 1)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    receipt = root / "block-000000000" / "RECEIPT.json"
    receipt.write_bytes(b" " * (frames.MAX_META_BYTES + 1))
    with pytest.raises(frames.FrameError, match="metadata byte cap"):
        frames.pack_frame(source, root, spec)


def test_actual_sigkill_after_first_seal_then_resume(tmp_path: Path) -> None:
    sources = [_block(tmp_path, i, 32)[0] for i in range(2)]
    root = tmp_path / "frames"
    marker = tmp_path / "first-sealed"
    code = """
import hashlib, os, signal, sys
from pathlib import Path
from chess_anti_engine.source.checkpointed_uci_frames import FrameSpec, pack_frame
root, marker = Path(sys.argv[1]), Path(sys.argv[2])
for index, name in enumerate(sys.argv[3:]):
    source = Path(name)
    raw = source.read_bytes()
    spec = FrameSpec('a'*64, 'b'*64, 'c'*64, hashlib.sha256(raw).hexdigest(),
                     index, len(raw.splitlines()), len(raw))
    pack_frame(source, root, spec)
    if index == 0 and os.environ.get('STOP_AFTER_FIRST') == '1':
        marker.write_text('sealed')
        os.kill(os.getpid(), signal.SIGSTOP)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["STOP_AFTER_FIRST"] = "1"
    command = [sys.executable, "-c", code, str(root), str(marker),
               *(str(path) for path in sources)]
    child = subprocess.Popen(command, env=env)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)
    assert child.returncode == -signal.SIGKILL
    first_receipt = (root / "block-000000000" / "RECEIPT.json").read_bytes()
    env["STOP_AFTER_FIRST"] = "0"
    subprocess.run(command, env=env, check=True, timeout=10)
    assert (root / "block-000000000" / "RECEIPT.json").read_bytes() == first_receipt
    assert _frame_path(root, 1).exists()


def test_actual_sigkill_in_unsealed_frame_stage_then_resume(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 25)
    root = tmp_path / "frames"
    marker = tmp_path / "frame-staged"
    code = """
import hashlib, os, signal, sys
from pathlib import Path
from chess_anti_engine.source import checkpointed_uci_frames as frames
source, root, marker = map(Path, sys.argv[1:])
raw = source.read_bytes()
spec = frames.FrameSpec('a'*64, 'b'*64, 'c'*64,
                        hashlib.sha256(raw).hexdigest(), 0,
                        len(raw.splitlines()), len(raw))
original = frames._atomic_new
def stopped_write(path, data):
    if path.name == 'FRAME.zst':
        (path.parent / '.FRAME.zst.part-killed').write_bytes(data[:len(data)//2])
        marker.write_text('staged')
        os.kill(os.getpid(), signal.SIGSTOP)
    return original(path, data)
frames._atomic_new = stopped_write
frames.pack_frame(source, root, spec)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(source), str(root), str(marker)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)
    assert child.returncode == -signal.SIGKILL
    directory = root / "block-000000000"
    assert not (directory / "RECEIPT.json").exists()
    claim = (directory / "CLAIM.json").read_bytes()
    receipt, reused = frames.pack_frame(source, root, spec)
    assert not reused
    assert receipt["rows"] == 25
    assert (directory / "CLAIM.json").read_bytes() == claim
    assert not (directory / ".FRAME.zst.part-killed").exists()


def test_live_writer_lock_is_nonblocking(tmp_path: Path) -> None:
    source, spec = _block(tmp_path, 0, 1)
    root = tmp_path / "frames"
    frames.pack_frame(source, root, spec)
    lock = root / "block-000000000" / ".lock"
    import fcntl

    with lock.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(frames.FrameError, match="already has a writer"):
            frames.pack_frame(source, root, spec)
