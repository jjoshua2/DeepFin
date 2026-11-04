"""Restore uses only synthetic directories; mutation spies guard refusals."""
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import quarantine_desync_shards as qds


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
])
def test_invalid_late_record_refuses_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: Any,
) -> None:
    shards, quarantine = journal(tmp_path, [{"name": "shard_000000.zarr"}, {"name": name}])
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


@pytest.mark.parametrize("side", ["shards", "quarantine"])
def test_symlink_entry_refuses_without_move_or_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, side: str,
) -> None:
    name = "shard_000001.zarr"
    shards, quarantine = journal(tmp_path, [{"name": name}], name)
    target = (shards if side == "shards" else quarantine) / name
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    calls = forbid_mutations(monkeypatch)
    assert qds.do_restore(shards, quarantine) == 2
    assert calls == []


def test_resolved_escape_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "shard_000001.zarr"
    shards, quarantine = journal(tmp_path, [{"name": name}])
    target = quarantine / name
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **kw:
                        tmp_path / "outside" / name if path == target else original(path, *a, **kw))
    calls = forbid_mutations(monkeypatch)
    assert qds.do_restore(shards, quarantine) == 2
    assert calls == []


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
