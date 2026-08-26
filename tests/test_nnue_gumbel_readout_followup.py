from __future__ import annotations

from pathlib import Path

from scripts import nnue_gumbel_readout as readout


def test_worker_specs_drop_empty_requested_workers() -> None:
    cfg = object.__new__(readout.RunConfig)
    object.__setattr__(cfg, "workers", 8)
    object.__setattr__(cfg, "games", 1)
    specs = readout._build_worker_specs(cfg)
    assert len(specs) == 1
    assert specs[0].game_indices == (0,)


def test_atomic_write_replaces_existing_result(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("old\n", encoding="utf-8")
    readout._atomic_write_text(path, "new\n")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".result.json.*.tmp"))
