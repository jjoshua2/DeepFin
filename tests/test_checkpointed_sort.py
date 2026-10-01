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


@pytest.mark.parametrize("source_count", [4, 16, 64])
def test_shared_noop_verification_reads_each_part_once(
        tmp_path: Path, source_count: int) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2, byte_cap=2048, fanin=4)
    inputs = [store.seal_source_run(
        f"source_{index}", [cs.SortEntry(DIGEST, index, (0, index))],
        source_identity_sha256=hashlib.sha256(
            f"source-{index}".encode()).hexdigest())
        for index in range(source_count)]
    build = cs._VerificationSession(store)
    final = store.merge_all(inputs, prefix="merge", _session=build)
    before = hashlib.sha256((final / "RUN.json").read_bytes()).hexdigest()
    part_receipts = {
        path.relative_to(store.root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.glob("*/part_*.receipt.json")}

    fresh = cs._VerificationSession(store)
    assert store.merge_all(inputs, prefix="merge", _session=fresh) == final
    parts = list(store.root.glob("*/part_*.jsonl"))
    distinct_extent = sum(path.stat().st_size for path in parts)
    assert fresh.payload_reads == len(parts)
    assert fresh.payload_bytes_read == (distinct_extent +
                                       fresh.cursor_probe_reads)
    assert len(fresh.runs) <= cs.MAX_VERIFIED_RUNS
    assert len(fresh.parts) == len(parts) <= cs.MAX_VERIFIED_PARTS
    assert hashlib.sha256((final / "RUN.json").read_bytes()).hexdigest() == before
    assert {
        path.relative_to(store.root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.glob("*/part_*.receipt.json")} == part_receipts
    assert [entry.input_index for entry in store.iter_run(final)] == list(
        range(source_count))

    # The cache is invocation-local: an old ancestor changed after the no-op
    # must be refused by the next full verification.
    ancestor = inputs[0] / "part_00000000.jsonl"
    ancestor.write_bytes(ancestor.read_bytes()[:-1])
    with pytest.raises(cs.CheckpointError, match="part hash/size"):
        store.verify_run(final)


@pytest.mark.parametrize("tamper", ["payload", "receipt"])
def test_input_part_changed_after_preflight_is_refused_before_merge(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        tamper: str) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2, byte_cap=2048, fanin=2)
    inputs = [store.seal_source_run(
        f"source_{index}", [cs.SortEntry(DIGEST, index, (0, index))],
        source_identity_sha256=str(index) * 64)
        for index in range(2)]
    input_part = inputs[0] / "part_00000000.jsonl"
    input_receipt = inputs[0] / "part_00000000.receipt.json"
    original_prepare = store._prepare

    def tamper_after_input_preflight(
            path: Path, claim: dict[str, object]) -> str:
        result = original_prepare(path, claim)
        changed = input_part if tamper == "payload" else input_receipt
        changed.write_bytes(changed.read_bytes()[:-1])
        return result

    monkeypatch.setattr(store, "_prepare", tamper_after_input_preflight)
    with pytest.raises(cs.CheckpointError,
                       match=r"changed input sort (part|receipt) before merge"):
        store.merge_run("merged", inputs)
    assert not list((store.root / "merged").glob("part_*.receipt.json"))



@pytest.mark.parametrize("metadata", ["RUN.json", "CLAIM.json"])
def test_input_run_metadata_changed_after_preflight_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=4, byte_cap=4096, fanin=4)
    inputs = [store.seal_source_run(
        f"source_{index}", [cs.SortEntry(DIGEST, index, (0, index))],
        source_identity_sha256=str(index) * 64)
        for index in range(2)]
    original_prepare = store._prepare

    def tamper(path: Path, claim: dict[str, object]) -> str:
        result = original_prepare(path, claim)
        target = inputs[0] / metadata
        value = json.loads(target.read_bytes())
        key = ("receipt_chain_sha256" if metadata == "RUN.json"
               else "source_identity_sha256")
        value[key] = "f" * 64
        target.write_bytes(cs._canonical(value))
        return result

    monkeypatch.setattr(store, "_prepare", tamper)
    with pytest.raises(cs.CheckpointError,
                       match=r"changed input sort (run receipt|claim)"):
        store.merge_run("merged", inputs)
    assert not list((store.root / "merged").glob("part_*.receipt.json"))


def test_cached_ancestor_receipt_changed_between_merge_groups_is_refused(
        tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=4, byte_cap=4096, fanin=4)
    inputs = [store.seal_source_run(
        f"source_{index}", [cs.SortEntry(DIGEST, index, (0, index))],
        source_identity_sha256=hashlib.sha256(str(index).encode()).hexdigest())
        for index in range(16)]
    changed = False

    def tamper(run: Path, _: int) -> None:
        nonlocal changed
        if run.name == "merge_000_000001" and not changed:
            receipt = store.root / "merge_000_000000" / "RUN.json"
            value = json.loads(receipt.read_bytes())
            value["receipt_chain_sha256"] = "f" * 64
            receipt.write_bytes(cs._canonical(value))
            changed = True

    with pytest.raises(cs.CheckpointError,
                       match="changed input sort run receipt"):
        store.merge_all(inputs, prefix="merge", after_part_seal=tamper)
    assert changed
    assert not (store.root / "merge_001_000000" / "RUN.json").exists()


def test_verification_session_caps_are_explicit(tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2, byte_cap=256, fanin=2)
    session = cs._VerificationSession(store)
    for index in range(cs.MAX_VERIFIED_RUNS):
        session.record_run(Path(f"run_{index}"), {"parts": 1})
    with pytest.raises(cs.CheckpointError, match="run cap"):
        session.record_run(Path("one_more_run"), {"parts": 1})

    proof = cs._PartProof("a" * 64, "b" * 64, 1, 1,
                          (DIGEST, 0), (DIGEST, 0))
    for index in range(cs.MAX_VERIFIED_PARTS):
        session.record_part(Path("run"), index, proof)
    with pytest.raises(cs.CheckpointError, match="part cap"):
        session.record_part(Path("run"), cs.MAX_VERIFIED_PARTS, proof)


def test_verification_session_binds_store_identity(tmp_path: Path) -> None:
    store = cs.CheckpointedSort(
        tmp_path / "runs", source_sha256=SOURCE, config_sha256=CONFIG,
        kind="digest", row_cap=2, byte_cap=2048, fanin=2)
    run = store.seal_source_run(
        "source_0", [cs.SortEntry(DIGEST, 0, (0, 0))],
        source_identity_sha256="0" * 64)
    session = cs._VerificationSession(store)
    store.verify_run(run, _session=session)
    changed = cs.CheckpointedSort(
        store.root, source_sha256=SOURCE, config_sha256="f" * 64,
        kind="digest", row_cap=2, byte_cap=2048, fanin=2)
    with pytest.raises(cs.CheckpointError, match="session store identity"):
        changed.verify_run(run, _session=session)
    store.code_sha256 = "0" * 64
    with pytest.raises(cs.CheckpointError, match="session store identity"):
        store.verify_run(run, _session=session)


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
