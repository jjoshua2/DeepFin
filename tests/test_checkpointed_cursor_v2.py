"""Synthetic only: a bounded fake replay exercises the real reader interface."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import pytest

from chess_anti_engine.source import checkpointed_candidate_v2 as candidate
from chess_anti_engine.source import checkpointed_cursor_v2 as cursor
from chess_anti_engine.source import checkpointed_sort as sort
from chess_anti_engine.source import checkpointed_wave_v2 as wave

MANIFEST = "a" * 64
STRICT = "b" * 64
SYZYGY = "c" * 64
CONFIG = "d" * 64
SOURCE = "e" * 64
ROUTE_CODE = "f" * 64
NAMESPACE = "bt4_run11:" + MANIFEST
CONTEXT = ("history", "repetition", 0, "legal", "teacher-query")


def _archive(game: int) -> bytes:
    return wave.canonical(["synthetic-archive", game])


def _locators(*, large: bool = False) -> tuple[cursor.GameLocator, ...]:
    # 31*256 + 255 = 8,191. A 2-row game crosses 8,192, then two
    # independent games prove that the stored midgame hash is not reused.
    sizes = [256] * 31 + [255, 2, 1, 1] if large else [2, 2]
    return tuple(cursor.GameLocator(
        "BT4-v9", MANIFEST, NAMESPACE, f"root-{index}", index,
        size, wave.sha(_archive(index)), STRICT)
        for index, size in enumerate(sizes))


def _pipeline(root: Path, *, large: bool = False,
              tamper_native: bool = False,
              tamper_proof: bool = False,
              tamper_archive: bool = False,
              locators: tuple[cursor.GameLocator, ...] | None = None
              ) -> tuple[cursor.CursorPipeline, cursor.SourceCursor]:
    locators = locators or _locators(large=large)

    def snapshot(loc: cursor.GameLocator) -> bytes:
        return (_archive(loc.game_id) + b"changed" if tamper_archive
                else _archive(loc.game_id))

    def replay(loc: cursor.GameLocator, raw: bytes, proof_sink):
        source_proof = {"schema": "full512_raw_strict_source_proof_v1",
                        "status": "PASS_SOURCE_PROOF",
                        "source": loc.source,
                        "root_id": loc.root_id,
                        "game_id": loc.game_id,
                        "strict_receipt_sha256": STRICT,
                        "archive_sha256": wave.sha(raw),
                        "gross_rows": loc.rows}
        source_proof_sha = hashlib.sha256(json.dumps(
            source_proof, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n").hexdigest()
        proof_sink({"source": loc.source, "root_id": loc.root_id,
                    "game_id": loc.game_id, "archive_sha256": wave.sha(raw),
                    "source_proof_sha256": ("0" * 64 if tamper_proof
                                            else source_proof_sha),
                    "source_proof": source_proof,
                    "terminal_fact": {"syzygy_inventory_sha256": SYZYGY,
                                      "terminal_replayed": True,
                                      "rows": loc.rows}})
        answer = []
        for ply in range(loc.rows):
            uid = [loc.source_manifest_sha, loc.namespace, loc.root_id,
                   loc.game_id, ply]
            native = hashlib.sha256(wave.canonical(uid)).digest()
            if tamper_native and loc.game_id == 0 and ply == 0:
                native = bytes([native[0] ^ 1]) + native[1:]
            answer.append(({
                "uid": uid, "source": loc.source,
                "provenance_locator_sha256": wave.sha(wave.canonical(uid)),
                "history_stack_sha256": CONTEXT[0],
                "repetition": CONTEXT[1], "rule50": CONTEXT[2],
                "legal_context_sha256": CONTEXT[3],
                "teacher_query_sha256": CONTEXT[4],
                "outcome": "1-0"}, native))
        return tuple(answer)

    reader = cursor.ReplayGameReader(snapshot, replay,
                                     strict_receipt_sha256=STRICT,
                                     syzygy_inventory_sha256=SYZYGY)
    source = cursor.SourceCursor(locators, reader,
                                 strict_receipt_sha256=STRICT,
                                 syzygy_inventory_sha256=SYZYGY,
                                 reader_code_sha256=wave.sha(b"synthetic-reader-v1"))
    store = wave.SegmentStore(root / "waves", input_sha256=source.identity,
                              source_sha256=SOURCE, config_sha256=CONFIG,
                              roster=source.roster, strict_receipt_sha256=STRICT,
                              syzygy_inventory_sha256=SYZYGY,
                              route_code_sha256=ROUTE_CODE, row_bytes=32)
    return cursor.CursorPipeline(root / "pipeline", source, store), source


def _sort(root: Path, source: cursor.SourceCursor) -> sort.CheckpointedSort:
    return sort.CheckpointedSort(root / "sort", source_sha256=source.identity,
                                 config_sha256=CONFIG, kind="digest",
                                 row_cap=2048, byte_cap=256 << 10, fanin=4)


def _run_child(root: str, hook: str) -> None:
    pipeline, _ = _pipeline(Path(root), large=True)
    def kill(_: int) -> None:
        os.kill(os.getpid(), signal.SIGKILL)
    if hook == "wave":
        pipeline.wave1(after_wave_seal=kill)
    elif hook == "comparison":
        pipeline.wave1(after_comparison=kill)
    else:
        raise ValueError("unknown kill hook")


def test_literal_uid_paired_candidate_and_sort(tmp_path: Path) -> None:
    pipeline, source = _pipeline(tmp_path)
    pipeline.wave1()
    runs = pipeline.wave2_metadata_sort(_sort(tmp_path, source))
    assert len(runs) == 1
    assert len(list(_sort(tmp_path, source).iter_run(runs[0]))) == 4
    notes = list(pipeline.store.iter_rows(0))
    assert notes[0][0]["uid"][1] == NAMESPACE
    assert notes[0][0]["source"] == "BT4-v9"
    assert candidate.route(notes[0][0]["uid"]) == "Ceres"
    assert candidate.route(notes[2][0]["uid"]) == "BT4"
    records = (pipeline.root / "candidate_00000000" / "RECORDS.bin").read_bytes()
    assert len(records) == 4 * 100
    for index in range(4):
        item = sort.RECORD.unpack_from(records, index * 100)
        assert sort.TEACHERS[item[11]] == candidate.route(notes[index][0]["uid"])
    before = wave.file_sha(pipeline.root / "candidate_00000000" / "RECEIPT.json")
    assert pipeline.wave2_metadata_sort(_sort(tmp_path, source)) == runs
    assert wave.file_sha(pipeline.root / "candidate_00000000" / "RECEIPT.json") == before


def test_8195_midgame_two_crash_boundaries_resume_and_noop(tmp_path: Path) -> None:
    script = ("import runpy,sys; "
              "runpy.run_path(sys.argv[1])['_run_child'](sys.argv[2],sys.argv[3])")
    exit_code = sort.run_owned_cpu(
        [sys.executable, "-c", script, __file__, str(tmp_path), "wave"],
        log_path=tmp_path / "kill.log", timeout_seconds=90)
    assert exit_code == -signal.SIGKILL
    sealed_wave = tmp_path / "waves" / "segment_00000000" / "RECEIPT.json"
    assert sealed_wave.exists()
    wave_hash = wave.file_sha(sealed_wave)
    wave_mtime = sealed_wave.stat().st_mtime_ns
    first = tmp_path / "pipeline" / "cursor_00000000.json"
    assert not first.exists()
    exit_code = sort.run_owned_cpu(
        [sys.executable, "-c", script, __file__, str(tmp_path), "comparison"],
        log_path=tmp_path / "kill-comparison.log", timeout_seconds=90)
    assert exit_code == -signal.SIGKILL
    comparison = tmp_path / "waves" / "compare_00000000.json"
    assert comparison.exists()
    assert not first.exists()
    compare_hash = wave.file_sha(comparison)
    compare_mtime = comparison.stat().st_mtime_ns
    pipeline, source = _pipeline(tmp_path, large=True)
    pipeline.wave1()
    assert wave.file_sha(sealed_wave) == wave_hash
    assert sealed_wave.stat().st_mtime_ns == wave_mtime
    assert wave.file_sha(comparison) == compare_hash
    assert comparison.stat().st_mtime_ns == compare_mtime
    first_cursor = json.loads(first.read_bytes())
    assert first_cursor["end"][:2] == [32, 1]
    assert first_cursor["end"][2] is not None
    sealed_hash = wave.file_sha(first)
    sealed_mtime = first.stat().st_mtime_ns
    runs = pipeline.wave2_metadata_sort(_sort(tmp_path, source))
    assert len(runs) == 5
    assert sum(_sort(tmp_path, source).verify_run(run)["rows"] for run in runs) == 8195
    assert first.stat().st_mtime_ns == sealed_mtime
    assert wave.file_sha(first) == sealed_hash
    pipeline.wave1()
    assert pipeline.wave2_metadata_sort(_sort(tmp_path, source)) == runs
    assert first.stat().st_mtime_ns == sealed_mtime


def test_source_namespace_native_and_receipt_tamper_refusal(tmp_path: Path) -> None:
    pipeline, source = _pipeline(tmp_path)
    pipeline.wave1()
    changed, _ = _pipeline(tmp_path, tamper_native=True)
    with pytest.raises(wave.Hold, match="wave 2 byte/order/metadata mismatch"):
        changed.wave2_metadata_sort(_sort(tmp_path, source))
    pipeline.wave2_metadata_sort(_sort(tmp_path, source))
    archive, _ = _pipeline(tmp_path / "archive", tamper_archive=True)
    with pytest.raises(wave.Hold, match="archive snapshot byte identity"):
        archive.wave1()
    proof, _ = _pipeline(tmp_path / "proof", tamper_proof=True)
    with pytest.raises(wave.Hold, match="complete checked replay proof"):
        proof.wave1()
    locators = _locators()
    swap = (cursor.GameLocator("Ceres-v8", *(
        locators[0].source_manifest_sha, locators[0].namespace,
        locators[0].root_id, locators[0].game_id, locators[0].rows,
        locators[0].archive_sha, locators[0].strict_receipt_sha)),
            *locators[1:])
    with pytest.raises(wave.Hold, match="conflicting source namespace"):
        _pipeline(tmp_path / "swap", locators=swap)
    namespace = tuple(cursor.GameLocator(
        loc.source, loc.source_manifest_sha, "new-namespace", loc.root_id,
        loc.game_id, loc.rows, loc.archive_sha, loc.strict_receipt_sha)
        for loc in locators)
    with pytest.raises(wave.Hold, match=r"cursor/wave source pins|claim/source/config"):
        _pipeline(tmp_path, locators=namespace)
    coordinated = tuple(cursor.GameLocator(
        loc.source, loc.source_manifest_sha, loc.namespace, loc.root_id,
        loc.game_id, loc.rows,
        wave.sha(_archive(loc.game_id) + b"changed"), loc.strict_receipt_sha)
        for loc in locators)
    with pytest.raises(wave.Hold, match=r"cursor/wave source pins|claim/source/config"):
        _pipeline(tmp_path, locators=coordinated)
    reused_game_id = cursor.GameLocator(
        locators[0].source, locators[0].source_manifest_sha,
        locators[0].namespace, "another-root", locators[0].game_id,
        locators[0].rows, locators[0].archive_sha,
        locators[0].strict_receipt_sha)
    accepted, _ = _pipeline(tmp_path / "reused-game-id",
                            locators=(locators[0], reused_game_id))
    assert accepted.source.total_rows == 4
    with pytest.raises(wave.Hold, match="duplicate literal source-qualified game"):
        _pipeline(tmp_path / "duplicate-game",
                  locators=(locators[0], locators[0]))
    path = pipeline.root / "cursor_00000000.json"
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises((ValueError, json.JSONDecodeError)):
        pipeline.wave1()
    path.write_bytes(original)
    records = pipeline.root / "candidate_00000000" / "RECORDS.bin"
    original = records.read_bytes()
    records.write_bytes(original[:-1])
    with pytest.raises(wave.Hold, match="candidate file hash/size"):
        candidate.verify(pipeline.store, 0, records.parent)
    records.write_bytes(original)
    # A coordinated local data/receipt rewrite still cannot alter winner facts.
    changed = bytearray(original)
    fields = list(sort.RECORD.unpack_from(changed, 0))
    fields[9] = 1  # a valid but false outcome
    sort.RECORD.pack_into(changed, 0, *fields)
    records.write_bytes(changed)
    receipt_path = records.parent / "RECEIPT.json"
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    receipt["files"]["RECORDS.bin"]["sha256"] = wave.sha(bytes(changed))
    receipt_path.write_bytes(wave.canonical(receipt))
    with pytest.raises(wave.Hold, match="candidate differs from paired wave"):
        candidate.verify(pipeline.store, 0, records.parent)
    records.write_bytes(original)
    receipt_path.write_bytes(original_receipt)


def test_valid_but_false_presealed_sort_run_refused(tmp_path: Path) -> None:
    pipeline, source = _pipeline(tmp_path)
    pipeline.wave1()
    pipeline.store.compare_wave2(
        0, source.rows_from(cursor.Cursor(0, 0, None)))
    meta = pipeline.root / "candidate_00000000"
    candidate.seal(pipeline.store, 0, meta)
    sorter = sort.CheckpointedSort(
        tmp_path / "false-sort", source_sha256=source.identity,
        config_sha256=CONFIG, kind="digest", row_cap=2048,
        byte_cap=256 << 10, fanin=4)
    receipt_sha = wave.file_sha(meta / "RECEIPT.json")
    identity = wave.sha(wave.canonical([receipt_sha, 0]))
    sorter.seal_source_run(
        "candidate_00000000",
        [sort.SortEntry("0" * 64, index, (0, index)) for index in range(4)],
        source_identity_sha256=identity)
    with pytest.raises(wave.Hold, match="sort run differs from paired candidate"):
        pipeline.wave2_metadata_sort(sorter)


def test_comparison_without_cursor_rechecks_source_bytes(tmp_path: Path) -> None:
    pipeline, _ = _pipeline(tmp_path)

    def interrupt(_: int) -> None:
        raise RuntimeError("controlled unsealed cursor window")

    with pytest.raises(RuntimeError, match="controlled"):
        pipeline.wave1(after_wave_seal=interrupt)
    with pytest.raises(RuntimeError, match="controlled"):
        pipeline.wave1(after_comparison=interrupt)
    comparison = tmp_path / "waves" / "compare_00000000.json"
    saved = comparison.read_bytes()
    changed, _ = _pipeline(tmp_path, tamper_native=True)
    with pytest.raises(wave.Hold, match="source differs from sealed paired wave"):
        changed.wave1()
    assert comparison.read_bytes() == saved
    assert not (pipeline.root / "cursor_00000000.json").exists()


def test_direct_wave2_refuses_valid_sealed_truncated_last_segment(
        tmp_path: Path) -> None:
    pipeline, source = _pipeline(tmp_path)
    two_rows = source.reader.read_game(source.locators[0])
    pipeline.store.seal_wave1(0, iter(two_rows), expected_rows=2)
    pipeline.store.compare_wave2(0, iter(two_rows))
    receipt = {
        "schema": "source_cursor_segment_v2",
        "claim_sha256": wave.sha(wave.canonical(pipeline.claim)),
        "segment": 0,
        "start": [0, 0, None], "end": [1, 0, None],
        "previous_cursor_receipt_sha256": "0" * 64,
        "wave_receipt_sha256": wave.file_sha(
            pipeline.store._path(0) / "RECEIPT.json"),
    }
    wave.atomic_write(pipeline.root / "cursor_00000000.json",
                      wave.canonical(receipt))
    assert pipeline.store.verify_segment(0)["rows"] == 2
    assert pipeline.store.verify_comparison(0)["rows"] == 2
    with pytest.raises(wave.Hold, match="paired segment truncated"):
        pipeline.wave2_metadata_sort(_sort(tmp_path, source))
