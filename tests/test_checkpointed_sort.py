"""CPU-only contracts for checkpointed candidate metadata and sorted runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tracemalloc

import pytest

from chess_anti_engine.source import checkpointed_sort as cs


SOURCE = "a" * 64
CONFIG = "b" * 64
NATIVE = "c" * 64
DIGEST = "d" * 64
CONTEXT = ("history", "repetition", 0, "legal", "query")


def _candidate(index: int) -> cs.CandidateMetadata:
    return cs.CandidateMetadata(
        hashlib.sha256(f"digest-{index}".encode()).hexdigest(),
        hashlib.sha256(f"bytes-{index}".encode()).hexdigest(), index,
        ("v9", "BT4", "fixture", index // 2, index % 2), CONTEXT,
        hashlib.sha256(f"proof-{index // 2}".encode()).hexdigest(),
        hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        cs.OUTCOMES[index % 3], "BT4", cs.TEACHERS[index % 2],
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def test_fixed_record_segment_seal_and_exact_readback(tmp_path: Path) -> None:
    path = tmp_path / "segment"
    expected = [_candidate(index) for index in range(5)]
    first = expected[0]
    expected[0] = cs.CandidateMetadata(
        first.input_digest_sha256, first.input_bytes_sha256,
        first.ordinal, ("v9", "BT4", "fixture", 70_000, 0),
        first.context, first.game_proof_sha256, first.provenance_sha256,
        first.outcome, first.source, first.teacher)
    receipt = cs.seal_candidate_segment(
        path, iter(expected), segment=0, source_sha256=SOURCE,
        config_sha256=CONFIG, native_source_sha256=NATIVE)
    assert cs.RECORD.size == 100
    assert receipt["rows"] == 5
    assert (path / "RECORDS.bin").stat().st_size == 500
    assert list(cs.iter_candidate_segment(
        path, source_sha256=SOURCE, config_sha256=CONFIG,
        native_source_sha256=NATIVE)) == expected
    original_receipt = (path / "RECEIPT.json").read_bytes()
    cs.seal_candidate_segment(
        path, iter(()), segment=0, source_sha256=SOURCE,
        config_sha256=CONFIG, native_source_sha256=NATIVE)
    assert (path / "RECEIPT.json").read_bytes() == original_receipt
    with pytest.raises(cs.CheckpointError, match="source/config/code"):
        cs.verify_candidate_segment(
            path, source_sha256=SOURCE, config_sha256="f" * 64,
            native_source_sha256=NATIVE)

    records = path / "RECORDS.bin"
    original = records.read_bytes()
    records.write_bytes(original[:-1])
    with pytest.raises(cs.CheckpointError, match="hash/size"):
        cs.verify_candidate_segment(
            path, source_sha256=SOURCE, config_sha256=CONFIG,
            native_source_sha256=NATIVE)
    records.write_bytes(original)

    # A coordinated file/receipt hash update still cannot authorize an invalid
    # segment-local table reference.
    record = list(cs.RECORD.unpack_from(original, 0))
    record[3] = cs.U32
    invalid = bytearray(original)
    cs.RECORD.pack_into(invalid, 0, *record)
    records.write_bytes(invalid)
    claim = json.loads(original_receipt)
    claim["files"]["RECORDS.bin"]["sha256"] = hashlib.sha256(invalid).hexdigest()
    (path / "RECEIPT.json").write_bytes(_canonical(claim))
    with pytest.raises(cs.CheckpointError, match="fixed record"):
        cs.verify_candidate_segment(
            path, source_sha256=SOURCE, config_sha256=CONFIG,
            native_source_sha256=NATIVE)

    overlong = _candidate(0)
    overlong = cs.CandidateMetadata(
        overlong.input_digest_sha256, overlong.input_bytes_sha256,
        overlong.ordinal, ("X" * (cs.MAX_FIELD_BYTES + 1),
                           "BT4", "fixture", 0, 0),
        overlong.context, overlong.game_proof_sha256,
        overlong.provenance_sha256, overlong.outcome,
        overlong.source, overlong.teacher)
    with pytest.raises(cs.CheckpointError, match="typed source-qualified UID"):
        cs.seal_candidate_segment(
            tmp_path / "oversized", [overlong], segment=0,
            source_sha256=SOURCE, config_sha256=CONFIG,
            native_source_sha256=NATIVE)


def _make_runs(store: cs.CheckpointedSort) -> list[Path]:
    runs = []
    for partition in range(5):
        entries = (cs.SortEntry(DIGEST, index, (0, index))
                   for index in range(partition, 8193, 5))
        run = store.seal_source_run(
            f"source_{partition}", entries,
            source_identity_sha256=hashlib.sha256(
                f"source-slice-{partition}".encode()).hexdigest())
        runs.append(run)
    return runs


def _fetch(locator: tuple[int, int]) -> cs.DigestMember:
    index = locator[1]
    return cs.DigestMember(
        b"exact-native" * 4, CONTEXT,
        ((8192 - index).to_bytes(8, "big"), b""), index % 3)


def test_stable_same_key_merge_kill_resume_and_group_bound(tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2048, byte_cap=256 << 10, fanin=4)
    inputs = _make_runs(store)
    child = (
        "import os,signal,sys; from pathlib import Path; "
        "from chess_anti_engine.source.checkpointed_sort import CheckpointedSort; "
        "r=Path(sys.argv[1]); "
        "s=CheckpointedSort(r,source_sha256='a'*64,config_sha256='b'*64,"
        "kind='digest',row_cap=2048,byte_cap=256<<10,fanin=4); "
        "s.merge_all([r/f'source_{i}' for i in range(5)],prefix='merge',"
        "after_part_seal=lambda *_:os.kill(os.getpid(),signal.SIGKILL))"
    )
    code = cs.run_owned_cpu(
        [sys.executable, "-c", child, str(store.root)],
        log_path=tmp_path / "kill.log", timeout_seconds=30)
    assert code == -signal.SIGKILL
    first = store.root / "merge_000_000000" / "part_00000000.receipt.json"
    assert first.is_file()
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    first_bytes = (first.parent / "part_00000000.jsonl").read_bytes()

    # The last sealed input bookmark must be a real line boundary. A forged
    # bookmark with a new receipt hash is refused before resuming.
    raw = first.read_bytes()
    bad = json.loads(raw)
    bad["end_cursors"][0] = [0, 1]
    first.write_bytes(_canonical(bad))
    with pytest.raises(cs.CheckpointError, match="cursor line boundary"):
        store.merge_all(inputs, prefix="merge")
    first.write_bytes(raw)

    final = store.merge_all(inputs, prefix="merge")
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_sha
    assert (first.parent / "part_00000000.jsonl").read_bytes() == first_bytes
    assert store.verify_run(final)["parts"] >= 5
    indexes = (entry.input_index for entry in store.iter_run(final))
    assert all(index == expected for expected, index in enumerate(indexes))
    losers = 0

    def on_loser(_: cs.SortEntry) -> None:
        nonlocal losers
        losers += 1

    result = list(cs.fold_digest_groups(
        store.iter_run(final), fetch=_fetch, on_loser=on_loser))
    assert len(result) == 1
    assert result[0].count == 8193
    assert result[0].winner.input_index == 8192
    assert result[0].outcome_counts == (2731, 2731, 2731)
    assert losers == 8192

    manifest_sha = hashlib.sha256((final / "RUN.json").read_bytes()).hexdigest()
    receipts = {p.relative_to(store.root): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in store.root.glob("*/part_*.receipt.json")}
    assert store.merge_all(inputs, prefix="merge") == final
    assert hashlib.sha256((final / "RUN.json").read_bytes()).hexdigest() == manifest_sha
    assert {p.relative_to(store.root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in store.root.glob("*/part_*.receipt.json")} == receipts

    payload = final / "part_00000000.jsonl"
    original = payload.read_bytes()
    payload.write_bytes(original[:-1])
    with pytest.raises(cs.CheckpointError, match="part hash/size"):
        store.verify_run(final)
    payload.write_bytes(original)
    changed = cs.CheckpointedSort(
        store.root, source_sha256=SOURCE, config_sha256="f" * 64,
        kind="digest", row_cap=2048, byte_cap=256 << 10, fanin=4)
    with pytest.raises(cs.CheckpointError, match="source/config/code claim"):
        changed.verify_run(final)
    store.code_sha256 = "0" * 64
    with pytest.raises(cs.CheckpointError, match="source/config/code claim"):
        store.verify_run(final)


def test_final_receipt_stage_sigkill_resume_preserves_parts(tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2, byte_cap=2048, fanin=2)
    inputs = [store.seal_source_run(
        f"source_{index}", [cs.SortEntry(DIGEST, index, (0, index))],
        source_identity_sha256=str(index) * 64)
        for index in range(2)]
    child = """import os, signal, sys
from pathlib import Path
from chess_anti_engine.source import checkpointed_sort as cs
old_atomic = cs._atomic_bytes
def kill_at_final(path, data):
    if path.name == 'RUN.json':
        stage = path.with_name('.RUN.json.part')
        with stage.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
    old_atomic(path, data)
cs._atomic_bytes = kill_at_final
root = Path(sys.argv[1])
store = cs.CheckpointedSort(root, source_sha256='a'*64,
    config_sha256='b'*64, kind='digest', row_cap=2, byte_cap=2048, fanin=2)
store.merge_run('merged', [root/'source_0', root/'source_1'])
"""
    code = cs.run_owned_cpu(
        [sys.executable, "-c", child, str(store.root)],
        log_path=tmp_path / "final_kill.log", timeout_seconds=30)
    assert code == -signal.SIGKILL
    run = store.root / "merged"
    assert (run / ".RUN.json.part").is_file()
    assert not (run / "RUN.json").exists()
    receipts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run.glob("part_*.receipt.json")}
    assert len(receipts) == 1

    assert store.merge_run("merged", inputs) == run
    assert not (run / ".RUN.json.part").exists()
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run.glob("part_*.receipt.json")} == receipts
    final_sha = hashlib.sha256((run / "RUN.json").read_bytes()).hexdigest()
    assert store.merge_run("merged", inputs) == run
    assert hashlib.sha256((run / "RUN.json").read_bytes()).hexdigest() == final_sha

    # A scratch file alongside an already sealed final receipt is not owned
    # recovery state.
    stage = run / ".RUN.json.part"
    stage.write_bytes(b"unexpected")
    with pytest.raises(cs.CheckpointError, match="file membership"):
        store.verify_run(run)
    assert stage.read_bytes() == b"unexpected"


def test_duplicate_digest_group_accumulator_does_not_grow_with_members() -> None:
    def peak_for(count: int) -> int:
        losers = 0

        def loser(_: cs.SortEntry) -> None:
            nonlocal losers
            losers += 1

        entries = (cs.SortEntry(DIGEST, index, (0, index))
                   for index in range(count))
        tracemalloc.start()
        try:
            groups = list(cs.fold_digest_groups(
                entries, fetch=_fetch, on_loser=loser))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert len(groups) == 1
        assert groups[0].count == count
        assert losers == count - 1
        return peak

    short_peak = peak_for(128)
    long_peak = peak_for(8193)  # >4x the physical 2,048-row cap
    assert long_peak - short_peak < 256 << 10


def test_exact_uid_sort_and_external_wall_watchdog(tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "uid_runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="uid", row_cap=2, byte_cap=1024)
    one = store.seal_source_run(
        "one", [cs.SortEntry(("v9", "BT4", "x", 1, 0), 2, (0, 2)),
                cs.SortEntry(("v9", "BT4", "x", 0, 0), 0, (0, 0))],
        source_identity_sha256="1" * 64)
    two = store.seal_source_run(
        "two", [cs.SortEntry(("v9", "BT4", "x", 1, 0), 1, (0, 1))],
        source_identity_sha256="2" * 64)
    merged = store.merge_run("merged", [one, two])
    assert [entry.input_index for entry in store.iter_run(merged)] == [0, 1, 2]
    oversized = cs.SortEntry(
        ("v9", "BT4", "x" * (cs.MAX_FIELD_BYTES + 1), 2, 0),
        3, (0, 3))
    with pytest.raises(cs.CheckpointError, match="typed source-qualified UID"):
        store.seal_source_run("oversized", [oversized],
                              source_identity_sha256="3" * 64)
    with pytest.raises(TimeoutError, match="owned CPU unit"):
        cs.run_owned_cpu(
            [sys.executable, "-c",
             "import os,time; print(os.getpid(),flush=True); time.sleep(30)"],
            log_path=tmp_path / "timeout.log", timeout_seconds=1)
    child_pid = int((tmp_path / "timeout.log").read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
