import numpy as np
import pytest
from scripts.stored_wdl_ply_coverage import summarize


def count(plies, groups=None, mask=None):
    n = len(plies)
    return summarize(
        np.array(plies, np.int32),
        np.array(mask if mask is not None else [1] * n, np.uint8),
        tuple(groups or ["g"] * n),
    )


def test_unsorted_exact_keys_not_adjacency():
    s = count([2, 0, 1])
    assert s["unique_successor_pairs"] == [[1, 2], [2, 0]]
    assert s["missing_successor_rows"] == 1
    assert not s["all_rows_have_unique_successor_key"]


def test_other_game_not_successor():
    assert count([0, 1], ["a", "b"])["unique_successor_count"] == 0


def test_duplicates_fail_ambiguity():
    s = count([0, 1, 1, 2])
    assert s["duplicate_key_count"] == 1
    assert s["duplicate_parent_rows"] == 2
    assert s["ambiguous_successor_rows"] == 1
    assert s["unique_successor_count"] == 0


def test_missing_and_negative_not_keys():
    s = count([0, 1, -1], mask=[1, 0, 1])
    assert s["valid_rows"] == 1
    assert s["missing_presence"] == 1
    assert s["present_negative_ply"] == 1


def test_int32_limit_no_wrap():
    assert count([2147483647, 0])["unique_successor_count"] == 0


def test_bad_mask():
    with pytest.raises(ValueError, match="binary presence"):
        count([0], mask=[2])


def test_synthetic_reader_and_row_binding(tmp_path):
    import hashlib
    import zarr
    from tests.test_stored_wdl_direct import direct_fixture
    from scripts.stored_wdl_direct import read_direct_shard
    from scripts.stored_wdl_ply_coverage import read_ply_coverage
    from chess_anti_engine.replay.directory_seal import write_directory_producer_seal

    path, kw = direct_fixture(tmp_path)
    g = zarr.open_group(str(path), mode="a")
    for name, a in {
        "ply_index": np.array([0, 1, 2], np.int32),
        "has_ply_index": np.ones(3, np.uint8),
    }.items():
        g.create_dataset(name, data=a, chunks=a.shape, overwrite=True)
    write_directory_producer_seal(g)
    kw["expected_seal_sha256"] = hashlib.sha256(
        (path / "directory_producer_seal.json").read_bytes()
    ).hexdigest()
    accepted = read_direct_shard(path, **kw)
    accepted.pop("x")
    result = read_ply_coverage(path, accepted)
    assert result["summary"]["present"] == 3
    assert result["child_identity_qualified"] is False
    accepted["row_ids"].reverse()
    with pytest.raises(ValueError, match="original ordered"):
        read_ply_coverage(path, accepted)
