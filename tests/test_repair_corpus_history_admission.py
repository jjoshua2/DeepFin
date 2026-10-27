"""Admission and inventory checks only: no corpus repair, engine or producer runs."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from scripts import repair_corpus_history as repair


def _progress_fixture(root: Path, *, torn: bool = False) -> None:
    # These are inventory placeholders, not decodable shards; never open their rows.
    name = "w03-00000.jsonl.zst"
    (root / name).write_bytes(b"inventory placeholder")
    (root / "w03-00001.jsonl.zst").write_bytes(b"unlisted placeholder")
    record = {"path": f"/old/host/{name}", "rows": 7, "codec": "zstd", "games": [1]}
    (root / "w03.progress.jsonl").write_text(
        json.dumps(record) + "\n" + ('{"path":' if torn else ""), encoding="utf-8",
    )


@pytest.mark.parametrize("progress", ["absent", "stale", "torn"])
@pytest.mark.parametrize("summary", ['{"run_finished": true}', '{"run_finished": false}', "invalid"])
def test_summary_inventory_refused_before_progress_read(
    tmp_path: Path, progress: str, summary: str,
) -> None:
    (tmp_path / repair.corpus.SUMMARY_NAME).write_text(summary, encoding="utf-8")
    if progress != "absent":
        _progress_fixture(tmp_path, torn=progress == "torn")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with (
        mock.patch.object(
            repair.derive, "read_progress_inventory", wraps=repair.derive.read_progress_inventory,
        ) as reader,
        pytest.raises(repair.RepairError, match=r"summary.json exists;.*legacy"),
    ):
        repair.corpus_inventory(tmp_path)
    reader.assert_not_called()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("torn", [False, True])
def test_legacy_progress_inventory_preserved(tmp_path: Path, torn: bool) -> None:
    _progress_fixture(tmp_path, torn=torn)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    listed, unlisted, inventory = repair.corpus_inventory(tmp_path)
    assert listed == (("w03-00000.jsonl.zst", 7),)
    assert unlisted == ("w03-00001.jsonl.zst",)
    assert inventory.rows_claimed == 7
    assert inventory.progress_files == ("w03.progress.jsonl",)
    assert inventory.torn_tail_files == (("w03.progress.jsonl",) if torn else ())
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_legacy_missing_progress_still_refused(tmp_path: Path) -> None:
    with pytest.raises(repair.RepairError, match="no shard inventory exists"):
        repair.corpus_inventory(tmp_path)


@pytest.mark.parametrize("flags", [
    [], ["--audit-only"], ["--audit-mode"],
    ["--audit-mode", "--audit-only", "--workers", "3", "--shards", "0-0"],
])
def test_cli_summary_refused_before_mutation_or_resources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str],
) -> None:
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / repair.corpus.SUMMARY_NAME).write_text("{}", encoding="utf-8")
    _progress_fixture(in_dir, torn=True)
    before = {p.name: p.read_bytes() for p in in_dir.iterdir()}
    out_dir = tmp_path / "output"
    with ExitStack() as stack:
        guards = [stack.enter_context(mock.patch.object(
            owner, name, side_effect=AssertionError(f"unexpected resource/mutation: {name}"),
        )) for owner, name in [
            (repair.corpus, "apply_history_rep_fix"),
            (repair.corpus, "require_history_rep_fix"),
            (repair, "build_spec"),
            (repair, "verify_book"),
            (repair, "warm_opening_book_cache"),
            (repair, "repair_worker"),
        ]]
        assert repair.main(["--in", str(in_dir), "--out", str(out_dir), *flags]) == 1
        for guard in guards:
            guard.assert_not_called()
    assert "summary.json exists;" in capsys.readouterr().err
    assert not out_dir.exists()
    assert {p.name: p.read_bytes() for p in in_dir.iterdir()} == before


def test_legacy_run_reaches_existing_regime_guard(tmp_path: Path) -> None:
    args = repair.build_parser().parse_args(["--in", str(tmp_path), "--audit-only"])
    with (
        mock.patch.object(repair.corpus, "apply_history_rep_fix") as apply,
        mock.patch.object(
            repair.corpus, "require_history_rep_fix", side_effect=repair.RepairError("regime sentinel"),
        ) as require,
        pytest.raises(repair.RepairError, match="regime sentinel"),
    ):
        repair.run(args)
    apply.assert_called_once_with()
    require.assert_called_once_with()
