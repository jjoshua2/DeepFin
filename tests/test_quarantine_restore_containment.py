"""Restore uses only synthetic directories; mutation spies guard refusals."""
import json
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.replay.shard import copy_or_link_shard
from scripts import quarantine_desync_shards as qds
from tests.test_quarantine_desync_shards import _apply, _write


def journal(tmp_path: Path, records: Any, reservation: Any = None) -> tuple[Path, Path]:
    shards, quarantine = tmp_path / "shards", tmp_path / "quarantine"
    shards.mkdir()
    quarantine.mkdir()
    (quarantine / qds.MANIFEST_NAME).write_text(json.dumps({
        "shard_dir": str(shards.resolve()), "shards": records,
        "reservation_name": reservation,
    }), encoding="utf-8")
    return shards, quarantine


def forbid_mutations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fail(*_args: Any, **_kwargs: Any) -> None:
        calls.append("mutation")
        raise AssertionError("invalid journal reached a mutation")

    monkeypatch.setattr(qds.shutil, "move", fail)
    monkeypatch.setattr(qds.shutil, "rmtree", fail)
    monkeypatch.setattr(Path, "replace", fail)
    return calls


@pytest.mark.parametrize("name", [
    "../shard_000001.zarr", "/tmp/shard_000001.zarr", "C:\\shard_000001.zarr",
    "\\\\host\\share\\shard_000001.zarr", "nested/shard_000001.zarr",
    "nested\\shard_000001.zarr", "shard_000001.zarr:stream", ".", "..", "",
    "shard_\0.zarr", None, 123, "unrelated-directory",
    # Well-formed prefix and suffix; refused only by the portable-name rule.
    "shard_..\\victim.zarr", "shard_000001:alt.zarr",
])
def test_invalid_late_record_refuses_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: Any,
) -> None:
    shards, quarantine = journal(tmp_path, [{"name": "shard_000000.zarr"}, {"name": name}])
    # The FIRST record is genuinely restorable (present, digest-free), so a
    # restore that validated record-by-record would move it before refusing.
    (quarantine / "shard_000000.zarr").mkdir()
    calls = forbid_mutations(monkeypatch)
    assert qds.do_restore(shards, quarantine) == 2
    assert calls == []
    assert (quarantine / qds.MANIFEST_NAME).exists()


@pytest.mark.parametrize("reservation", ["../victim", "shard_999999.zarr"])
def test_reservation_must_be_a_valid_recorded_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reservation: str,
) -> None:
    shards, quarantine = journal(tmp_path, [{"name": "shard_000000.zarr"}], reservation)
    calls = forbid_mutations(monkeypatch)
    assert qds.do_restore(shards, quarantine) == 2
    assert calls == []


@pytest.mark.parametrize("records", [{}, [None], [{"name": "shard_000001.zarr"}] * 2])
def test_malformed_or_duplicate_records_refuse_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: Any,
) -> None:
    shards, quarantine = journal(tmp_path, records)
    calls = forbid_mutations(monkeypatch)
    assert qds.do_restore(shards, quarantine) == 2
    assert calls == []


def test_symlinked_shard_window_round_trips_through_apply_and_restore(tmp_path: Path) -> None:
    """A window built by ``copy_or_link_shard`` (relative symlinks) must restore.

    ``do_apply`` quarantines a linked shard by renaming the LINK, so restore
    has to accept a symlink entry or the quarantine is one-way. Window
    directories of linked shards exist in the production run's replay tree.

    MUTATION: refuse symlink entries in ``_restore_path`` (the #1110 rule) --
    ``do_restore`` returns 2 and the dirty shards stay stranded in quarantine.
    """
    store, window = tmp_path / "store", tmp_path / "replay_shards"
    store.mkdir()
    for i in (100, 101, 102, 103, 104):
        _write(store, i)
    for i in (105, 106, 107):
        _write(store, i, miss_frac=0.6, attached=False)
    for src in sorted(store.iterdir()):
        copy_or_link_shard(src, window / src.name)
    assert all((window / src.name).is_symlink() for src in store.iterdir())
    before = {p.name: qds.digest(p) for p in qds.iter_shard_paths(window)}

    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    moved = json.loads((quar / qds.MANIFEST_NAME).read_text())["shards"]
    assert moved
    assert all((quar / rec["name"]).is_symlink() for rec in moved)

    assert qds.do_restore(window, quar) == 0
    after = {p.name: qds.digest(p) for p in qds.iter_shard_paths(window)}
    assert after == before
    assert all((window / rec["name"]).is_symlink() for rec in moved)


def test_valid_move_and_digest_checked_idempotence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards, quarantine = journal(tmp_path, [])
    name = "shard_000001.zarr"
    source = quarantine / name
    source.mkdir()
    (source / "synthetic.txt").write_bytes(b"fixture")
    rec = {"name": name, "sha256": qds.digest(source)}
    moved: list[tuple[str, str]] = []
    original = qds.shutil.move

    def spy(src: str, dest: str) -> str:
        moved.append((src, dest))
        return original(src, dest)

    monkeypatch.setattr(qds.shutil, "move", spy)
    assert qds._move_back(rec, shards, quarantine) == "moved"
    assert qds._move_back(rec, shards, quarantine) == "skip"
    assert moved == [(str(source), str(shards / name))]
    (shards / name / "synthetic.txt").write_bytes(b"changed")
    assert qds._move_back(rec, shards, quarantine).startswith("REFUSE")


def test_valid_reservation_restore_preserves_order_and_retires_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, last = "shard_000001.zarr", "shard_000002.zarr"
    records = [{"name": first, "sha256": "first"}, {"name": last, "sha256": "last"}]
    shards, quarantine = journal(tmp_path, records, last)
    (shards / last).mkdir()
    calls: list[str] = []
    monkeypatch.setattr(qds, "shard_positions", lambda _path: 0)
    monkeypatch.setattr(qds, "_move_back", lambda rec, *_args: calls.append(rec["name"]) or "moved")
    original = qds.shutil.rmtree

    def remove(path: Path) -> None:
        calls.append("remove:" + path.name)
        original(path)

    monkeypatch.setattr(qds.shutil, "rmtree", remove)
    assert qds.do_restore(shards, quarantine) == 0
    assert calls == [first, "remove:" + last, last]
    assert (quarantine / qds.RESTORED_MANIFEST_NAME).exists()


def test_direct_move_refuses_unchecked_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = forbid_mutations(monkeypatch)
    assert qds._move_back({"name": "../victim"}, tmp_path, tmp_path).startswith("REFUSE")
    assert calls == []
