"""The actual builder must not publish rows that its validation rejects."""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.eval.audit_history import MatchedAuditRows
from chess_anti_engine.replay.shard import POLICY_SIZE
from scripts import match_audit_rows as mar


def _inputs(root: Path, *, kind: str = "valid") -> tuple[Path, Path, str]:
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    x = mar.encode_cboard(
        mar.CBoard.from_board(board),
        input_history_encoding=mar.STORED_HISTORY_ENCODING,
        input_extra_features=mar.STORED_EXTRA_FEATURES,
    ).astype(np.float16)
    audit_board = board.copy()
    if kind == "legal":
        audit_board.castling_rights = 0
    elif kind == "current":
        x[:12][x[:12] > 0.5] = np.float16(0.75)
    elif kind == "castling":
        x[104:108][x[104:108] > 0.5] = np.float16(0.75)
    elif kind == "history":
        x[13, 0, 0] = 1
    elif kind != "valid":
        raise AssertionError(kind)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshot"
    snapshot.mkdir(exist_ok=True)
    group = zarr.open_group(str(snapshot / "one.zarr"), mode="w")
    group.create_dataset("x", data=x[None])
    # These declaration-valid labels are never trained/scored. The real lazy
    # replay reader requires their shape; only x/game identity is consumed.
    group.create_dataset("policy_target", data=np.zeros((1, POLICY_SIZE), dtype=np.float32))
    group.create_dataset("wdl_target", data=np.asarray([1], dtype=np.int64))
    group.create_dataset("game_id", data=np.asarray([7], dtype=np.int64))
    group.create_dataset("has_game_id", data=np.asarray([1], dtype=np.uint8))
    group.attrs["input_history_encoding"] = mar.STORED_HISTORY_ENCODING
    key = mar.position_key(board)
    audit = root / "audit.jsonl"
    audit.write_text(json.dumps({
        "key": key, "fen": audit_board.fen(), "phase": 0, "source": 0,
    }) + "\n", encoding="utf-8")
    return audit, snapshot, key


def _run(audit: Path, snapshot: Path, out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "match_audit_rows.py", "--audit-set", str(audit),
        "--snapshot", str(snapshot), "--out", str(out),
    ])
    mar.main()


@pytest.mark.parametrize("kind", ["legal", "current", "castling"])
@pytest.mark.parametrize("existing", [False, True])
def test_rejected_build_never_publishes_or_overwrites_final_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, existing: bool,
) -> None:
    out = tmp_path / "matched.npz"
    report_path = out.with_suffix(".npz.report.json")
    before: dict[Path, bytes] = {}
    if existing:
        audit, snapshot, _key = _inputs(tmp_path / "old")
        _run(audit, snapshot, out, monkeypatch)
        before = {p: p.read_bytes() for p in (out, report_path)}
    audit, snapshot, _key = _inputs(tmp_path / "rejected", kind=kind)
    with pytest.raises(SystemExit):
        _run(audit, snapshot, out, monkeypatch)
    if existing:
        assert {p: p.read_bytes() for p in before} == before
    else:
        assert not out.exists()
        assert not report_path.exists()


@pytest.mark.parametrize("kind", ["valid", "history"])
def test_valid_builder_output_is_accepted_by_real_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    audit, snapshot, key = _inputs(tmp_path, kind=kind)
    out = tmp_path / "matched.npz"
    _run(audit, snapshot, out, monkeypatch)
    rows = MatchedAuditRows(out)
    rows.require_index_layout()
    assert rows.stored_row(key).shape == (175, 8, 8)
    assert rows.game_cluster_id(key) == 0
    report = json.loads(out.with_suffix(".npz.report.json").read_text())
    assert report["canonicalisation_current_frame_identical"]
    assert report["canonicalisation_castling_104_107_identical"]
    assert report["legal_move_sets_identical"]
    assert (rows.stored_row(key)[13, 0, 0] == 1) == (kind == "history")


def _report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rejected_evidence_is_bound_reproducible_and_not_a_consumer_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    import hashlib

    audit, snapshot, key = _inputs(tmp_path, kind="castling")
    audit_bytes = audit.read_bytes()
    out = tmp_path / "matched.npz"
    with pytest.raises(SystemExit, match="castling-plane"):
        _run(audit, snapshot, out, monkeypatch)
    reports = list(tmp_path.glob("matched.npz.rejected.*.json"))
    assert len(reports) == 1
    report = _report(reports[0])
    assert report["validation_status"] == "rejected"
    assert report["publication"] == "not attempted"
    assert report["audit_set_sha256"] == hashlib.sha256(audit_bytes).hexdigest()
    assert report["candidate_output"] == {"path": str(out), "sha256": None}
    assert report["prior_outputs"] == report["observed_final_outputs"]
    assert report["validation_errors"] == ["recovered rows do not preserve castling-plane identity"]
    evidence = report["offending_rows"][0]
    assert evidence["key"] == key
    assert evidence["audit_row"] == evidence["src_row"] == 0
    arrays = {}
    for name, record in evidence["arrays"].items():
        array = np.frombuffer(base64.b64decode(record["base64_c_order"]),
                              dtype=np.dtype(record["dtype"])).reshape(record["shape"])
        assert mar._array_identity(array)["sha256"] == record["sha256"]
        arrays[name] = array
    assert not np.array_equal(arrays["stored"][104:108], arrays["fen_only"][104:108])
    recovered = mar.decode_board_from_planes(
        arrays["stored"], input_history_encoding=mar.STORED_HISTORY_ENCODING,
    )
    assert recovered is not None
    assert mar.position_key(recovered) == key
    shard = report["consumed_shards"][0]
    group = zarr.open_group(str(snapshot / "one.zarr"), mode="r")
    assert "candidate_full_rows" not in shard
    assert shard["candidate_row_decision_inputs"] == mar._candidate_rows_identity(np.asarray(group["x"]), [0])
    assert shard["current_frame_scan"] == mar._array_identity(np.asarray(group["x"][:, :12], dtype=np.float32))
    assert shard["identity_columns"]["has_game_id"] == mar._array_identity(np.asarray([1], dtype=np.uint8))
    assert not list(tmp_path.glob("*.npz"))
    # Later source mutations cannot erase the bounded offending-row witness.
    audit.write_text("source changed")
    group = zarr.open_group(str(snapshot / "one.zarr"), mode="a")
    group["x"][:] = 0
    assert _report(reports[0]) == report
    with pytest.raises(SystemExit, match="does not exist"):
        MatchedAuditRows(out)


def test_repeated_rejection_keeps_distinct_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path, kind="current")
    out = tmp_path / "matched.npz"
    for _ in range(2):
        with pytest.raises(SystemExit):
            _run(audit, snapshot, out, monkeypatch)
    reports = list(tmp_path.glob("*.rejected.*.json"))
    assert len(reports) == 2
    assert len({_report(path)["build_id"] for path in reports}) == 2
    assert not out.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_rejected_evidence_write_failure_cannot_promote_invalid_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    from chess_anti_engine.utils import atomic

    out = tmp_path / "matched.npz"
    normal_report = out.with_suffix(".npz.report.json")
    before = {}
    if existing:
        audit, snapshot, _key = _inputs(tmp_path / "old")
        _run(audit, snapshot, out, monkeypatch)
        before = {p: p.read_bytes() for p in (out, normal_report)}
    audit, snapshot, _key = _inputs(tmp_path / "bad", kind="legal")
    original = atomic._fsync_file

    def fail_rejected_report(path: Path) -> None:
        if ".rejected." in path.name:
            raise OSError("injected rejected evidence sync failure")
        original(path)

    monkeypatch.setattr(atomic, "_fsync_file", fail_rejected_report)
    with pytest.raises(OSError, match="rejected evidence"):
        _run(audit, snapshot, out, monkeypatch)
    assert {p: p.read_bytes() for p in before} == before
    if not existing:
        assert not out.exists()
        assert not normal_report.exists()
    assert not list(tmp_path.glob("*.rejected.*.json"))


@pytest.mark.parametrize("stage", ["serialization", "attempt-report", "file-fsync", "replace", "dir-fsync", "compat-report"])
def test_actual_publication_boundaries_preserve_old_before_commit_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    import hashlib
    from chess_anti_engine.utils import atomic

    out = tmp_path / "matched.npz"
    normal_report = out.with_suffix(".npz.report.json")
    audit, snapshot, key = _inputs(tmp_path / "old")
    _run(audit, snapshot, out, monkeypatch)
    old_npz, old_report = out.read_bytes(), normal_report.read_bytes()
    audit, snapshot, _key = _inputs(tmp_path / "new", kind="history")
    events: list[str] = []
    save, fsync_file, fsync_dir, replace = (
        np.savez_compressed, atomic._fsync_file, atomic._fsync_dir, atomic.os.replace,
    )

    def writing(stream, **arrays) -> None:
        events.append("serialization")
        if stage == "serialization":
            stream.write(b"incomplete")
            raise OSError("injected serialization")
        save(stream, **arrays)

    def sync_file(path: Path) -> None:
        name = path.name
        if ".validated." in name:
            events.append("attempt-report")
            if stage == "attempt-report":
                raise OSError("injected attempt-report")
        elif name.endswith(".npz"):
            events.append("file-fsync")
            if stage == "file-fsync":
                raise OSError("injected file-fsync")
        elif name.startswith(normal_report.name + ".tmp."):
            events.append("compat-report")
            if stage == "compat-report":
                raise OSError("injected compat-report")
        fsync_file(path)

    def replacing(src: str, dst: str) -> None:
        if Path(dst) == out:
            events.append("replace")
            if stage == "replace":
                raise OSError("injected replace")
            replace(src, dst)
            events.append("replaced")
        else:
            replace(src, dst)

    def sync_dir(path: Path) -> None:
        if path == out.parent and "replaced" in events and "dir-fsync" not in events:
            events.append("dir-fsync")
            if stage == "dir-fsync":
                raise OSError("injected dir-fsync")
        fsync_dir(path)

    monkeypatch.setattr(mar.np, "savez_compressed", writing)
    monkeypatch.setattr(atomic, "_fsync_file", sync_file)
    monkeypatch.setattr(atomic, "_fsync_dir", sync_dir)
    monkeypatch.setattr(atomic.os, "replace", replacing)
    with pytest.raises(OSError, match="injected"):
        _run(audit, snapshot, out, monkeypatch)
    postcommit = stage in ("dir-fsync", "compat-report")
    assert (out.read_bytes() != old_npz) == postcommit
    assert normal_report.read_bytes() == old_report
    assert (MatchedAuditRows(out).stored_row(key)[13, 0, 0] == 1) == postcommit
    order = ["serialization", "attempt-report", "file-fsync", "replace", "replaced", "dir-fsync", "compat-report"]
    assert events == order[:order.index(stage) + 1]
    attempts = [_report(p) for p in tmp_path.glob("*.validated.*.json")]
    if stage not in ("serialization", "attempt-report"):
        assert len(attempts) == 2
        new = next(r for r in attempts if r["audit_set"] == str(audit))
        assert new["publication"] == "validated candidate; publication not confirmed"
        if postcommit:
            assert new["candidate_output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_no_matches_is_rejected_with_json_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    audit.write_text(json.dumps({"key": mar.position_key(chess.Board()),
                                 "fen": chess.STARTING_FEN, "phase": 0, "source": 0}) + "\n")
    out = tmp_path / "matched.npz"
    with pytest.raises(SystemExit, match="no audit rows matched"):
        _run(audit, snapshot, out, monkeypatch)
    report = _report(next(tmp_path.glob("*.rejected.*.json")))
    assert report["matched"] == 0
    assert report["history_plane_nonzero_frac_stored"] is None
    assert report["offending_rows"] == []
    assert not out.exists()


def test_array_hash_binds_shape_dtype_and_values_in_c_order() -> None:
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    identity = mar._array_identity(values)
    assert identity == mar._array_identity(np.asfortranarray(values))
    assert identity != mar._array_identity(values.astype(np.float64))
    assert identity != mar._array_identity(values.reshape(20, 5))
    assert mar._array_identity(values[:, ::2]) == mar._array_identity(values[:, ::2].copy())
    values[0, 0] = 99
    assert identity != mar._array_identity(values)


@pytest.mark.parametrize("name", ["matched.npz", "matched.custom", "matched", ".npz", "matched.NPZ"])
def test_legacy_output_suffix_and_report_bind_compressed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    import hashlib

    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / name
    sentinel = b"unrelated custom-name input stays intact"
    if not str(out).endswith(".npz"):
        out.write_bytes(sentinel)
    _run(audit, snapshot, out, monkeypatch)
    report = _report(out.with_suffix(out.suffix + ".report.json"))
    actual = out if str(out).endswith(".npz") else Path(str(out) + ".npz")
    assert report["candidate_output"]["path"] == str(actual)
    assert report["candidate_output"]["sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()
    with np.load(actual, allow_pickle=False) as arrays:
        assert str(arrays["build_id"][0]) == report["build_id"]
    assert _report(Path(report["attempt_report"]))["candidate_output"] == report["candidate_output"]
    if not str(out).endswith(".npz"):
        assert out.read_bytes() == sentinel


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_interrupted_serialization_preserves_prior_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException],
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / "matched.npz"
    report_path = out.with_suffix(".npz.report.json")
    _run(audit, snapshot, out, monkeypatch)
    before = {p: p.read_bytes() for p in (out, report_path)}

    def interrupt(stream, **_arrays) -> None:
        stream.write(b"partial")
        raise failure("injected interruption")

    monkeypatch.setattr(mar.np, "savez_compressed", interrupt)
    with pytest.raises(failure, match="interruption"):
        _run(audit, snapshot, out, monkeypatch)
    assert {p: p.read_bytes() for p in before} == before
    assert not list(tmp_path.glob("*.tmp.*"))


def test_attempt_report_serialization_failure_preserves_prior_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / "matched.npz"
    report_path = out.with_suffix(".npz.report.json")
    _run(audit, snapshot, out, monkeypatch)
    before = {p: p.read_bytes() for p in (out, report_path)}
    write_text = Path.write_text

    def fail_write(path: Path, *args, **kwargs):
        if ".validated." in path.name:
            write_text(path, "partial diagnostic")
            raise OSError("injected report write")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="report write"):
        _run(audit, snapshot, out, monkeypatch)
    assert {p: p.read_bytes() for p in before} == before
    assert not list(tmp_path.glob("*.tmp.*"))


def test_concurrent_complete_publisher_is_not_rolled_back_after_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.utils import atomic

    out = tmp_path / "matched.npz"
    audit_a, snapshot_a, key = _inputs(tmp_path / "a")
    audit_b, snapshot_b, _key = _inputs(tmp_path / "b", kind="history")
    replace, sync_dir = atomic.os.replace, atomic._fsync_dir
    published_b = False
    a_committed = False

    def replacing(src: str, dst: str) -> None:
        nonlocal a_committed
        replace(src, dst)
        if Path(dst) == out and not a_committed:
            a_committed = True

    def syncing(path: Path) -> None:
        nonlocal published_b
        if path == out.parent and a_committed and not published_b:
            published_b = True
            _run(audit_b, snapshot_b, out, monkeypatch)
            raise OSError("injected A postcommit failure")
        sync_dir(path)

    monkeypatch.setattr(atomic.os, "replace", replacing)
    monkeypatch.setattr(atomic, "_fsync_dir", syncing)
    with pytest.raises(OSError, match="postcommit"):
        _run(audit_a, snapshot_a, out, monkeypatch)
    assert MatchedAuditRows(out).stored_row(key)[13, 0, 0] == 1
    report = _report(out.with_suffix(".npz.report.json"))
    assert report["audit_set"] == str(audit_b)


def test_concurrent_publisher_can_leave_detectably_stale_compatibility_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    from chess_anti_engine.utils import atomic

    out = tmp_path / "matched.npz"
    audit_a, snapshot_a, key = _inputs(tmp_path / "a")
    audit_b, snapshot_b, _key = _inputs(tmp_path / "b", kind="history")
    replace = atomic.os.replace
    published_b = False

    def replacing(src: str, dst: str) -> None:
        nonlocal published_b
        replace(src, dst)
        if Path(dst) == out and not published_b:
            published_b = True
            _run(audit_b, snapshot_b, out, monkeypatch)

    monkeypatch.setattr(atomic.os, "replace", replacing)
    _run(audit_a, snapshot_a, out, monkeypatch)
    assert MatchedAuditRows(out).stored_row(key)[13, 0, 0] == 1
    report = _report(out.with_suffix(".npz.report.json"))
    assert report["audit_set"] == str(audit_a)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert report["candidate_output"]["sha256"] != digest
    attempts = [_report(p) for p in tmp_path.glob("*.validated.*.json")]
    matching = [r for r in attempts if r["candidate_output"]["sha256"] == digest]
    assert len(matching) == 1
    assert matching[0]["audit_set"] == str(audit_b)


def test_offending_row_evidence_is_bounded_without_losing_input_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    audit, snapshot, _key = _inputs(tmp_path, kind="current")
    audit.write_bytes(audit.read_bytes() * 9)
    captured = audit.read_bytes()
    out = tmp_path / "matched.npz"
    with pytest.raises(SystemExit):
        _run(audit, snapshot, out, monkeypatch)
    report = _report(next(tmp_path.glob("*.rejected.*.json")))
    assert report["offending_rows_total"] == 9
    assert len(report["offending_rows"]) == report["offending_rows_limit"] == 4
    assert report["audit_set_sha256"] == hashlib.sha256(captured).hexdigest()
    assert [r["audit_row"] for r in report["offending_rows"]] == [0, 1, 2, 3]


def test_array_hash_scratch_chunks_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []
    real = mar.hashlib.sha256

    class ObservedHash:
        def __init__(self, initial: bytes) -> None:
            self.value = real(initial)

        def update(self, data: bytes) -> None:
            sizes.append(len(data))
            self.value.update(data)

        def hexdigest(self) -> str:
            return self.value.hexdigest()

    monkeypatch.setattr(mar.hashlib, "sha256", ObservedHash)
    mar._array_identity(np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024).T)
    assert len(sizes) >= 8
    assert max(sizes) <= 131072 * 4


def test_captured_audit_hash_does_not_reread_mutated_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    audit, snapshot, _key = _inputs(tmp_path, kind="castling")
    captured = audit.read_bytes()
    read = mar.load_shard_arrays

    def mutate_after_audit_capture(path: Path, **kwargs):
        audit.write_text("changed after capture")
        return read(path, **kwargs)

    monkeypatch.setattr(mar, "load_shard_arrays", mutate_after_audit_capture)
    out = tmp_path / "matched.npz"
    with pytest.raises(SystemExit):
        _run(audit, snapshot, out, monkeypatch)
    report = _report(next(tmp_path.glob("*.rejected.*.json")))
    assert report["audit_set_sha256"] == hashlib.sha256(captured).hexdigest()
    assert report["audit_set_sha256"] != hashlib.sha256(audit.read_bytes()).hexdigest()


@pytest.mark.parametrize(("modes", "mask"), [((0o600, 0o600), 0o022), ((0o640, 0o604), 0o077)])
@pytest.mark.parametrize("fail", [False, True])
def test_existing_modes_preserved_from_staging_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modes: tuple[int, int], mask: int, fail: bool,
) -> None:
    import os
    import stat
    from chess_anti_engine.utils import atomic

    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / "matched.npz"
    normal = out.with_suffix(".npz.report.json")
    _run(audit, snapshot, out, monkeypatch)
    for path, mode in zip((out, normal), modes, strict=True):
        path.chmod(mode)
    before = {p: p.read_bytes() for p in (out, normal)}
    original_open, original_sync = Path.open, atomic._fsync_file
    observed: list[tuple[str, int]] = []

    def expected(path: Path) -> int:
        if ".validated." in path.name:
            return 0o600
        return modes[1] if ".report.json" in path.name else modes[0]

    def opening(path: Path, mode: str = "r", *args, **kwargs):
        if ".tmp." in path.name and "w" in mode:
            actual = stat.S_IMODE(path.stat().st_mode)
            observed.append((path.name, actual))
            # A restrictive umask can narrow the initial mode, never widen it.
            assert actual & ~expected(path) == 0
        return original_open(path, mode, *args, **kwargs)

    def syncing(path: Path) -> None:
        assert stat.S_IMODE(path.stat().st_mode) == expected(path)
        if fail and path.suffix == ".npz":
            raise OSError("injected private precommit failure")
        original_sync(path)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(atomic, "_fsync_file", syncing)
    old_mask = os.umask(mask)
    try:
        if fail:
            with pytest.raises(OSError, match="private precommit"):
                _run(audit, snapshot, out, monkeypatch)
            assert {p: p.read_bytes() for p in before} == before
        else:
            _run(audit, snapshot, out, monkeypatch)
    finally:
        os.umask(old_mask)
    assert observed
    assert tuple(stat.S_IMODE(p.stat().st_mode) for p in (out, normal)) == modes
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in tmp_path.glob("*.validated.*.json"))


def test_rejected_diagnostic_is_private_and_raw_rows_are_not_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import os
    import stat

    audit, snapshot, _key = _inputs(tmp_path, kind="castling")
    out = tmp_path / "matched.npz"
    opening = Path.open
    observations: list[int] = []

    def observe(path: Path, mode: str = "r", *args, **kwargs):
        if ".rejected." in path.name and "w" in mode:
            observations.append(stat.S_IMODE(path.stat().st_mode))
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    old_mask = os.umask(0)
    try:
        with pytest.raises(SystemExit, match="castling-plane"):
            _run(audit, snapshot, out, monkeypatch)
    finally:
        os.umask(old_mask)
    report_path = next(tmp_path.glob("*.rejected.*.json"))
    assert observations == [0o600]
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    report = _report(report_path)
    output = capsys.readouterr().out
    assert "castling-plane" in output
    assert str(report_path) in output
    assert "base64_c_order" not in output
    assert report["offending_rows"][0]["arrays"]["stored"]["base64_c_order"] not in output


@pytest.mark.parametrize("which", ["npz", "report"])
@pytest.mark.parametrize("dangling", [False, True])
def test_terminal_output_symlinks_are_refused_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str, dangling: bool,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / "matched.npz"
    normal = out.with_suffix(".npz.report.json")
    target = tmp_path / "target"
    if not dangling:
        target.write_bytes(b"do not replace or follow")
    link = out if which == "npz" else normal
    link.symlink_to(target)
    with pytest.raises(SystemExit, match="symlink output"):
        _run(audit, snapshot, out, monkeypatch)
    assert link.is_symlink()
    assert link.readlink() == target
    assert target.exists() is not dangling
    if not dangling:
        assert target.read_bytes() == b"do not replace or follow"
    assert not list(tmp_path.glob("*.validated.*.json"))


def test_symlink_introduced_during_serialization_refuses_npz_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    out = tmp_path / "matched.npz"
    normal = out.with_suffix(".npz.report.json")
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    save = mar.np.savez_compressed

    def save_and_link(*args, **kwargs):
        save(*args, **kwargs)
        normal.symlink_to(target)

    monkeypatch.setattr(mar.np, "savez_compressed", save_and_link)
    with pytest.raises(SystemExit, match="symlink output"):
        _run(audit, snapshot, out, monkeypatch)
    assert not out.exists()
    assert normal.is_symlink()
    assert target.read_bytes() == b"unchanged"


def test_candidate_decision_digest_binds_indices_order_and_only_selected_rows() -> None:
    x = np.arange(3 * 175 * 8 * 8, dtype=np.float16).reshape(3, 175, 8, 8)
    identity = mar._candidate_rows_identity(x, [0, 2])
    unused = x.copy()
    unused[1, 20, 0, 0] = -1
    assert mar._candidate_rows_identity(unused, [0, 2]) == identity
    used = x.copy()
    used[2, 20, 0, 0] = -1
    assert mar._candidate_rows_identity(used, [0, 2]) != identity
    assert mar._candidate_rows_identity(x, [2, 0]) != identity
    assert mar._candidate_rows_identity(x, [0, 1]) != identity
    assert mar._candidate_rows_identity(x.astype(np.float32), [0, 2]) != identity
    equal_rows = np.zeros_like(x)
    assert mar._candidate_rows_identity(equal_rows, [0]) != mar._candidate_rows_identity(equal_rows, [1])
    assert mar._candidate_rows_identity(x[:, :, :, ::-1][:, :, :, ::-1], [0, 2]) == identity


def test_candidate_decision_digest_streams_row_views_not_advanced_index_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.zeros((64, 175, 8, 8), dtype=np.float16)
    calls = []
    original = mar._array_identity

    def observe(row: np.ndarray):
        assert row.shape == (175, 8, 8)
        assert np.shares_memory(x, row)
        calls.append(row.nbytes)
        return original(row)

    monkeypatch.setattr(mar, "_array_identity", observe)
    mar._candidate_rows_identity(x, [1, 10, 63])
    assert calls == [175 * 8 * 8 * 2] * 3


def test_candidate_collision_is_included_even_when_it_does_not_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, snapshot, _key = _inputs(tmp_path)
    path = snapshot / "one.zarr"
    group = zarr.open_group(str(path), mode="a")
    valid = np.asarray(group["x"])[0]
    collision = valid.copy()
    collision[104:108] = 0  # Same piece fingerprint; different position key.
    unused = valid.copy()
    unused[:12] = 0  # No candidate fingerprint, so full history is unused.
    del group["x"]
    group.create_dataset("x", data=np.stack([valid, collision, unused]))
    for name, value in (("policy_target", np.zeros((3, POLICY_SIZE), dtype=np.float32)),
                        ("wdl_target", np.ones(3, dtype=np.int64)),
                        ("game_id", np.arange(3, dtype=np.int64)),
                        ("has_game_id", np.ones(3, dtype=np.uint8))):
        del group[name]
        group.create_dataset(name, data=value)
    out = tmp_path / "matched.npz"
    _run(audit, snapshot, out, monkeypatch)
    first = _report(out.with_suffix(".npz.report.json"))["consumed_shards"][0]
    assert first["candidate_row_decision_inputs"]["selected_rows"] == 2
    group["x"][2, 20, 0, 0] = 1
    _run(audit, snapshot, out, monkeypatch)
    second = _report(out.with_suffix(".npz.report.json"))["consumed_shards"][0]
    assert first == second
    group["x"][1, 20, 0, 0] = 1
    _run(audit, snapshot, out, monkeypatch)
    third = _report(out.with_suffix(".npz.report.json"))["consumed_shards"][0]
    assert second["current_frame_scan"] == third["current_frame_scan"]
    assert second["candidate_row_decision_inputs"] != third["candidate_row_decision_inputs"]
    group["x"][2, 0, 0, 0] = 0.25  # Still no candidate, but every scan byte is bound.
    _run(audit, snapshot, out, monkeypatch)
    fourth = _report(out.with_suffix(".npz.report.json"))["consumed_shards"][0]
    assert third["current_frame_scan"] != fourth["current_frame_scan"]
    assert third["candidate_row_decision_inputs"] == fourth["candidate_row_decision_inputs"]
    group["game_id"][0] = 71
    _run(audit, snapshot, out, monkeypatch)
    fifth = _report(out.with_suffix(".npz.report.json"))["consumed_shards"][0]
    assert fourth["identity_columns"]["game_id"] != fifth["identity_columns"]["game_id"]


def test_candidate_decision_digest_has_independent_reproducible_oracle() -> None:
    import hashlib

    array = np.arange(60, dtype=np.float16).reshape(3, 4, 5)
    indices = [2, 0]
    header = {"scheme": "ordered-index-and-row-digests-v1", "dtype": "<f2",
              "source_shape": [3, 4, 5], "selected_rows": 2}
    expected = hashlib.sha256(json.dumps(header, sort_keys=True).encode() + b"\0")
    for index in indices:
        row_header = {"dtype": "<f2", "shape": [4, 5]}
        row_digest = hashlib.sha256(json.dumps(row_header, sort_keys=True).encode()
                                    + b"\0" + array[index].tobytes(order="C")).digest()
        expected.update(index.to_bytes(8, "little", signed=True))
        expected.update(row_digest)
    assert mar._candidate_rows_identity(array, indices) == {**header, "sha256": expected.hexdigest()}
