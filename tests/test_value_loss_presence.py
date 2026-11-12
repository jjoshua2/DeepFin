"""Per-row metadata masks survive the real shard writer and scorer boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays
from scripts import value_loss_scorer as scorer


def _shard(
    root: Path, *, missing_game: bool = False, missing_population: bool = False,
    missing_ply: bool = False,
) -> Path:
    samples = [
        ReplaySample(
            x=np.zeros((146, 8, 8), np.float32),
            policy_target=np.full(POLICY_SIZE, 1 / POLICY_SIZE, np.float32),
            wdl_target=1, sf_wdl=np.array([0.2, 0.3, 0.5], np.float32),
            game_id=(None if missing_game and i == 1 else i),
            ply_index=(None if missing_ply and i == 1 else i + 10), is_network_turn=True,
            is_selfplay=(None if missing_population and i == 1 else bool(i % 2)),
        )
        for i in range(4)
    ]
    return save_local_shard_arrays(root / "shard_000001.zarr", arrs=samples_to_arrays(samples))


def test_actual_writer_marks_unknown_id_beside_a_valid_zero(tmp_path: Path) -> None:
    path = _shard(tmp_path, missing_game=True)
    g = zarr.open_group(str(path), mode="r")
    assert np.asarray(g["game_id"][:]).tolist() == [0, 0, 2, 3]
    assert np.asarray(g["has_game_id"][:]).tolist() == [1, 0, 1, 1]
    with pytest.raises(SystemExit, match=r"game_id.*missing"):
        scorer.load_rows_from_shards(str(tmp_path))


def test_missing_game_guard_runs_before_checkpoint_load(tmp_path: Path) -> None:
    _shard(tmp_path, missing_game=True)
    with pytest.raises(SystemExit, match=r"game_id.*missing"):
        scorer.main(["--shard-dir", str(tmp_path), "--checkpoint", str(tmp_path / "absent.pt")])


def test_partial_population_is_unknown_and_dump_does_not_launder_it(tmp_path: Path) -> None:
    _shard(tmp_path, missing_population=True)
    rows = scorer.load_rows_from_shards(str(tmp_path))
    assert len(rows) == 4  # Ordinary unstratified scoring still keeps all rows.
    assert rows.population_known is False
    dump = tmp_path / "bank.npz"
    scorer.write_dump(str(dump), rows, rows.p_sf, np.zeros(4, bool), with_x=True, meta={})
    assert scorer.load_rows_from_npz(str(dump)).population_known is False


def test_partial_population_split_refuses_before_checkpoint_load(tmp_path: Path) -> None:
    _shard(tmp_path, missing_population=True)
    with pytest.raises(SystemExit, match="split-selfplay"):
        scorer.main([
            "--shard-dir", str(tmp_path), "--split-selfplay",
            "--checkpoint", str(tmp_path / "absent.pt"),
        ])


@pytest.mark.parametrize("field", ["game_id", "is_selfplay", "ply_index"])
@pytest.mark.parametrize("exclude", ["eligibility", "limit"])
def test_missing_metadata_outside_selected_rows_does_not_block(
    tmp_path: Path, field: str, exclude: str,
) -> None:
    path = _shard(
        tmp_path, missing_game=field == "game_id",
        missing_population=field == "is_selfplay", missing_ply=field == "ply_index",
    )
    if exclude == "eligibility":
        g = zarr.open_group(str(path), mode="a")
        g["has_sf_wdl"][1] = 0
    rows = scorer.load_rows_from_shards(str(tmp_path), max_rows=1 if exclude == "limit" else 0)
    assert len(rows) == (1 if exclude == "limit" else 3)
    assert rows.population_known is True
    assert rows.ply_known is True
    assert rows.game_id[0] == 0  # Zero is a valid observed id, not a sentinel.


def test_fully_measured_writer_rows_keep_identity_and_population(tmp_path: Path) -> None:
    _shard(tmp_path)
    rows = scorer.load_rows_from_shards(str(tmp_path))
    assert rows.game_id.tolist() == [0, 1, 2, 3]
    assert rows.is_selfplay.tolist() == [False, True, False, True]
    assert rows.population_known is True
    assert rows.ply.tolist() == [10, 11, 12, 13]
    assert rows.ply_known is True


@pytest.mark.parametrize("field", ["game_id", "is_selfplay", "ply_index"])
def test_malformed_presence_mask_is_not_broadcast(tmp_path: Path, field: str) -> None:
    path = _shard(tmp_path)
    g = zarr.open_group(str(path), mode="a")
    g.create_dataset(f"has_{field}", data=np.ones((4, 1), np.uint8), overwrite=True)
    with pytest.raises(SystemExit, match=f"has_{field} shape"):
        scorer.load_rows_from_shards(str(tmp_path))


def test_partial_ply_is_unknown_and_dump_omits_it(tmp_path: Path) -> None:
    path = _shard(tmp_path, missing_ply=True)
    g = zarr.open_group(str(path), mode="r")
    # The writer stores a placeholder beside a cleared mask.
    assert np.asarray(g["has_ply_index"][:]).tolist() == [1, 0, 1, 1]
    rows = scorer.load_rows_from_shards(str(tmp_path))
    assert len(rows) == 4  # Ply is reporting metadata; scoring keeps every row.
    assert rows.ply_known is False
    assert rows.take(np.array([0, 2])).ply_known is False
    dump = tmp_path / "bank.npz"
    scorer.write_dump(str(dump), rows, rows.p_sf, np.zeros(4, bool), with_x=True, meta={})
    with np.load(dump) as z:
        assert "ply" not in z.files
        assert "game_id" in z.files
    reloaded = scorer.load_rows_from_npz(str(dump))
    assert reloaded.ply_known is False
    # A second round trip must not launder the reload's zero default into a
    # measured ply column.
    redump = tmp_path / "rebank.npz"
    scorer.write_dump(str(redump), reloaded, reloaded.p_sf, np.zeros(4, bool), with_x=True, meta={})
    with np.load(redump) as z:
        assert "ply" not in z.files


def test_observed_ply_round_trips_through_the_dump(tmp_path: Path) -> None:
    _shard(tmp_path)
    rows = scorer.load_rows_from_shards(str(tmp_path))
    dump = tmp_path / "bank.npz"
    scorer.write_dump(str(dump), rows, rows.p_sf, np.zeros(4, bool), with_x=True, meta={})
    with np.load(dump) as z:
        assert np.asarray(z["ply"]).tolist() == [10, 11, 12, 13]
    reloaded = scorer.load_rows_from_npz(str(dump))
    assert reloaded.ply_known is True
    assert reloaded.ply.tolist() == [10, 11, 12, 13]


def test_maskless_shard_keeps_its_ply_column(tmp_path: Path) -> None:
    """Shards written before the mask existed keep the old contract."""
    path = _shard(tmp_path)
    g = zarr.open_group(str(path), mode="a")
    del g["has_ply_index"]
    assert "has_ply_index" not in set(g.array_keys())
    rows = scorer.load_rows_from_shards(str(tmp_path))
    assert rows.ply_known is True
    assert rows.ply.tolist() == [10, 11, 12, 13]
