from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from scripts import sf_d9_rank_sidecar as tool


def raw_row(*, stm: str = "w") -> dict[str, object]:
    moves = (
        [[1, "e2e4", 20.0, 100], [2, "d2d4", 13.0, 100], [3, "g1f3", 3.0, 100]]
        if stm == "w"
        else [[1, "e7e5", 20.0, 100], [2, "d7d5", 13.0, 100], [3, "g8f6", 3.0, 100]]
    )
    return {
        "game_id": 7,
        "ply": 4,
        "stm": stm,
        "phases": [
            {
                "per_depth": [
                    {"depth": 8, "complete": True, "lines": moves},
                    {"depth": 9, "complete": True, "lines": moves},
                ],
            }
        ],
    }


@pytest.mark.parametrize("stm", ["w", "b"])
def test_rank_observation_banks_top_indices_and_cp_gaps(stm: str) -> None:
    observed = tool.rank_observation(raw_row(stm=stm), top_k=3)
    assert observed.game_id == 7
    assert observed.ply == 4
    assert observed.count == 3
    assert observed.indices.dtype == np.uint16
    assert len(set(map(int, observed.indices))) == 3
    np.testing.assert_array_equal(observed.gaps_cp, [0.0, 7.0, 17.0])


def test_rank_observation_pads_short_width() -> None:
    row: Any = raw_row()
    row["phases"][0]["per_depth"][1]["lines"] = [[1, "e2e4", 20.0, 100]]
    observed = tool.rank_observation(row, top_k=3)
    assert observed.count == 1
    np.testing.assert_array_equal(
        observed.indices[1:],
        np.asarray([tool.INVALID_INDEX, tool.INVALID_INDEX], dtype=np.uint16),
    )
    assert bool(np.all(np.isinf(observed.gaps_cp[1:])))


def test_d9_lines_reads_only_the_full_width_phase_zero_block() -> None:
    row: Any = raw_row()
    expected = row["phases"][0]["per_depth"][1]["lines"]
    row["phases"].append(
        {
            "per_depth": [
                {
                    "depth": 9,
                    "complete": True,
                    "lines": [[1, "a2a3", 99.0, 50]],
                }
            ]
        }
    )

    assert tool.d9_lines(row) == expected


def test_rank_observation_refuses_unknown_side_to_move() -> None:
    with pytest.raises(ValueError, match="side-to-move"):
        tool.rank_observation(raw_row(stm="unknown"), top_k=3)


def test_flush_refuses_row_order_drift(tmp_path: Path) -> None:
    rows = [
        tool.rank_observation(raw_row(), top_k=3),
        tool.RankObservation(
            game_id=8,
            ply=5,
            indices=tool.rank_observation(raw_row(), top_k=3).indices,
            gaps_cp=tool.rank_observation(raw_row(), top_k=3).gaps_cp,
            count=3,
        ),
    ]
    source_path = tmp_path / "shard_000000.zarr"
    source = zarr.open_group(str(source_path), mode="w")
    source.create_dataset("game_id", data=np.asarray([7, 8], dtype=np.int64))
    source.create_dataset("ply_index", data=np.asarray([4, 5], dtype=np.int32))
    legal = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.uint8)
    policy = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float16)
    for index in rows[0].indices:
        legal[:, int(index)] = 1
    policy[:, int(rows[0].indices[0])] = 1.0
    source.create_dataset("legal_mask", data=legal)
    source.create_dataset("policy_target", data=policy)

    destination = tmp_path / "rank.zarr"
    receipt = tool._flush(
        observations=rows,
        order=np.asarray([0, 1]),
        source_path=source_path,
        destination=destination,
        top_k=3,
        source_summary_sha256="a" * 64,
        raw_config_sha256="b" * 64,
    )
    assert receipt["rows"] == 2
    stored = zarr.open_group(str(destination), mode="r")
    np.testing.assert_array_equal(
        np.asarray(stored[tool.GAP_FIELD][:, 1]),
        [7.0, 7.0],
    )

    with pytest.raises(ValueError, match="does not match derived order"):
        tool._flush(
            observations=rows,
            order=np.asarray([1, 0]),
            source_path=source_path,
            destination=tmp_path / "bad.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )

    first = rows[0]
    repeated = tool.RankObservation(
        game_id=first.game_id,
        ply=first.ply,
        indices=np.asarray(
            [first.indices[0], first.indices[0], first.indices[2]],
            dtype=np.uint16,
        ),
        gaps_cp=first.gaps_cp,
        count=3,
    )
    with pytest.raises(ValueError, match="repeat a compact move index"):
        tool._flush(
            observations=[repeated, rows[1]],
            order=np.asarray([0, 1]),
            source_path=source_path,
            destination=tmp_path / "repeated.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )

    decreasing = tool.RankObservation(
        game_id=first.game_id,
        ply=first.ply,
        indices=first.indices,
        gaps_cp=np.asarray([0.0, 17.0, 7.0], dtype=np.float32),
        count=3,
    )
    with pytest.raises(ValueError, match="gaps must be nondecreasing"):
        tool._flush(
            observations=[decreasing, rows[1]],
            order=np.asarray([0, 1]),
            source_path=source_path,
            destination=tmp_path / "decreasing.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )
