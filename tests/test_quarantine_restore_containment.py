"""Restore uses only synthetic directories; mutation spies guard refusals."""
import json
import os
import shutil
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


def _window(tmp_path: Path, linked: bool) -> Path:
    """Five clean + three dirty shards; ``linked`` builds it the production way.

    ``copy_or_link_shard`` writes RELATIVE links -- production windows hold
    ones like ``../../../../../data/salvage/.../shard_005121.zarr``.
    """
    store, window = tmp_path / "data" / "salvage" / "store", tmp_path / "runs" / "t" / "replay_shards"
    target = store if linked else window
    target.mkdir(parents=True)
    for i in (100, 101, 102, 103, 104):
        _write(target, i)
    for i in (105, 106, 107):
        _write(target, i, miss_frac=0.6, attached=False)
    if linked:
        for src in sorted(store.iterdir()):
            copy_or_link_shard(src, window / src.name)
        assert all(not os.path.isabs(os.readlink(p)) for p in window.iterdir())
    return window


def _snapshot(window: Path) -> dict[str, tuple[str, str | None]]:
    return {
        p.name: (qds.digest(p), os.readlink(p) if p.is_symlink() else None)
        for p in qds.iter_shard_paths(window)
    }


QUARANTINE_LAYOUTS = {
    "same_depth": ("runs", "t", "q"),
    "deeper": ("runs", "t", "x", "y", "z", "q"),
    "shallow_other_parent": ("q",),
    "other_parent": ("elsewhere", "quarantine"),
}


@pytest.mark.parametrize("layout", sorted(QUARANTINE_LAYOUTS))
@pytest.mark.parametrize("linked", [True, False], ids=["linked", "plain"])
def test_apply_then_restore_round_trips_byte_for_byte(
    tmp_path: Path, layout: str, linked: bool,
) -> None:
    """Restore puts back exactly what apply took: bytes AND link text.

    MUTATION A: move the link verbatim in ``do_apply`` (main's behaviour) --
    every layout but ``same_depth`` refuses with "is missing".
    MUTATION B: restore links by ``shutil.move`` of the quarantined entry --
    the restored link text is the absolute target, not the original.
    """
    window = _window(tmp_path, linked)
    before = _snapshot(window)
    quar = tmp_path.joinpath(*QUARANTINE_LAYOUTS[layout])
    assert _apply(window, quar) == 0
    moved = json.loads((quar / qds.MANIFEST_NAME).read_text())["shards"]
    assert moved
    assert all(("link_text" in rec) == linked for rec in moved)
    # Every quarantined entry is valid where it sits (so --verify can digest it).
    assert all((quar / rec["name"]).exists() for rec in moved)

    assert qds.do_restore(window, quar) == 0
    assert _snapshot(window) == before
    assert not any((quar / rec["name"]).exists() or (quar / rec["name"]).is_symlink()
                   for rec in moved)


def _downgrade_to_old_journal(quar: Path) -> list[dict[str, Any]]:
    """Rewrite a new-format quarantine as the pre-link_text code left it."""
    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    for rec in man["shards"]:
        text = rec.pop("link_text")
        entry = quar / rec["name"]
        entry.unlink()
        os.symlink(text, entry)  # the old code moved the relative text verbatim
    (quar / qds.MANIFEST_NAME).write_text(json.dumps(man))
    return man["shards"]


def test_old_journal_with_dangling_relative_links_restores_against_the_window(
    tmp_path: Path,
) -> None:
    window = _window(tmp_path, linked=True)
    before = _snapshot(window)
    quar = tmp_path / "elsewhere" / "quarantine"
    assert _apply(window, quar) == 0
    records = _downgrade_to_old_journal(quar)
    assert all(not (quar / r["name"]).exists() for r in records), "fixture must dangle"
    assert qds.do_restore(window, quar) == 0
    assert _snapshot(window) == before


def test_old_journal_link_that_resolves_nowhere_refuses_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    window = _window(tmp_path, linked=True)
    quar = tmp_path / "elsewhere" / "quarantine"
    assert _apply(window, quar) == 0
    records = _downgrade_to_old_journal(quar)
    first = quar / records[0]["name"]
    first.unlink()
    os.symlink("../nowhere/" + records[0]["name"], first)
    assert qds.do_restore(window, quar) == 2
    out = capsys.readouterr().out
    assert "journal without link_text" in out
    assert records[0]["original_path"] in out
    assert not (window / records[0]["name"]).is_symlink()


def test_records_without_sha256_restore_with_a_counted_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: drop the warning -- an unverified restore is silent again."""
    window = _window(tmp_path, linked=False)
    quar = tmp_path / "q"
    assert _apply(window, quar) == 0
    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    for rec in man["shards"][:2]:
        rec.pop("sha256")
    (quar / qds.MANIFEST_NAME).write_text(json.dumps(man))
    assert qds.do_restore(window, quar) == 0
    out = capsys.readouterr().out
    for rec in man["shards"][:2]:
        assert f"WARNING: {rec['name']}: journal record has no sha256" in out
    assert f"WARNING: 2 of {len(man['shards'])} restored records had no sha256" in out


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


def _linked_quarantine(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    window = _window(tmp_path, linked=True)
    quar = tmp_path / "elsewhere" / "quarantine"
    assert _apply(window, quar) == 0
    return window, quar, json.loads((quar / qds.MANIFEST_NAME).read_text())["shards"]


def test_a_linked_target_changed_after_apply_refuses_restore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: skip the target digest check -- the changed shard is linked back."""
    window, quar, records = _linked_quarantine(tmp_path)
    target = Path(records[0]["original_path"])
    victim = next(p for p in sorted(target.rglob("*")) if p.is_file())
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))
    assert qds.do_restore(window, quar) == 2
    assert "digest" in capsys.readouterr().out
    assert not (window / records[0]["name"]).is_symlink()


@pytest.mark.parametrize("edit", ["other_shard", "absolute", "escaping"])
def test_an_edited_link_text_is_refused_before_anything_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], edit: str,
) -> None:
    """The journal's text must land where the quarantined link points.

    With sha256 absent nothing else would catch an edited ``link_text``: it
    would plant another shard (or any path) in the window, or hash ``/etc``.
    MUTATION: drop the tie check -- ``other_shard`` is planted and the others
    get hashed.
    """
    window, quar, records = _linked_quarantine(tmp_path)
    man = json.loads((quar / qds.MANIFEST_NAME).read_text())
    other = sorted(Path(r["original_path"]) for r in records)[-1]
    text = {
        "other_shard": os.path.relpath(other, window),
        "absolute": str(tmp_path / "data"),
        "escaping": "../../../data",
    }[edit]
    for rec in man["shards"]:
        rec.pop("sha256", None)
    man["shards"][0]["link_text"] = text
    (quar / qds.MANIFEST_NAME).write_text(json.dumps(man))
    first = man["shards"][0]["name"]
    hashed: list[Path] = []
    real_digest = qds.digest
    monkeypatch.setattr(qds, "digest", lambda p: hashed.append(Path(p)) or real_digest(p))
    assert qds.do_restore(window, quar) == 2
    out = capsys.readouterr().out
    assert "the journal does not describe this entry" in out
    assert not (window / first).is_symlink()
    tampered = qds._link_target(text, window)
    assert all(qds._link_target(str(h), window) != tampered for h in hashed)


def test_a_dangling_link_already_at_the_destination_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: drop the dangling-destination refusal -- os.symlink raises over it."""
    window, quar, records = _linked_quarantine(tmp_path)
    os.symlink("../nowhere/" + records[0]["name"], window / records[0]["name"])
    assert qds.do_restore(window, quar) == 2
    assert "dangling link" in capsys.readouterr().out
    assert (quar / records[0]["name"]).is_symlink(), "the quarantined entry must survive"


def test_a_quarantined_entry_that_is_not_a_link_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The journal says link; the entry is a real directory -- refuse, move nothing."""
    window, quar, records = _linked_quarantine(tmp_path)
    entry = quar / records[0]["name"]
    target = Path(records[0]["original_path"])
    entry.unlink()
    shutil.copytree(target, entry)
    assert qds.do_restore(window, quar) == 2
    assert "should be the quarantined link" in capsys.readouterr().out
    assert entry.is_dir()
    assert not entry.is_symlink()
    assert not (window / records[0]["name"]).exists()
