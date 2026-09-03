"""Stored-field default invariants for the replay schema.

Three separate silent-corruption sites in the 2026-08-03 train/data audit share
one root: code that synthesizes a MISSING stored field with a bare `np.zeros`,
or that reads as if a field pair were optional when it is not. The schema's own
`zeros_for_storage_field` disagrees with `np.zeros` for a small explicit set,
and for every member zero is a meaningful wrong value rather than an empty one.
"""

from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.replay import disk_buffer as db
from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.replay.shard import (
    _OPTIONAL_FIELD_SPECS,
    _SHARD_FIELDS,
    NONZERO_DEFAULT_STORAGE_FIELDS,
    prune_storage_arrays,
    zeros_for_storage_field,
)

_POLICY_SIZE = 4672


def _arrays(n: int = 4) -> dict[str, np.ndarray]:
    policy = np.zeros((n, _POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float16),
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def test_nonzero_default_storage_fields_is_re_derived_from_the_schema() -> None:
    """The constant must track `zeros_for_storage_field`, not a stale memory of it.

    A new field with a non-zero default that is not listed here would walk
    straight past the `_gather_rows` guard below.
    """
    derived = {
        name
        for name in _SHARD_FIELDS
        if np.any(
            zeros_for_storage_field(name, n=3, x_planes=146, policy_size=_POLICY_SIZE) != 0
        )
    }
    assert derived == set(NONZERO_DEFAULT_STORAGE_FIELDS)


def test_shard_field_order_emits_value_array_before_its_flag() -> None:
    """`_concat_sparse_batches` relied on this ordering, in the wrong direction.

    It carried a branch retaining a uniformly-absent value array "when its flag
    is already in `out`", which the ordering below makes unreachable. The branch
    is gone; if the ordering ever flips, the pair-retention question is live
    again and this test is where that shows up.
    """
    order = {name: i for i, name in enumerate(_SHARD_FIELDS)}
    for spec in _OPTIONAL_FIELD_SPECS:
        assert order[spec.arr] < order[spec.flag], spec.arr


@pytest.mark.parametrize(
    ("field", "silent_damage"),
    [
        # Both required members of NONZERO_DEFAULT_STORAGE_FIELDS, because they fail
        # DIFFERENTLY downstream and a guard that covers one is not a guard.
        ("has_policy", "rows dropped from the policy loss and its accuracy stats"),
        ("priority", "rows made unsamplable under a priority draw"),
    ],
)
def test_gather_rows_refuses_chunks_that_disagree(tmp_path, field, silent_damage) -> None:
    """Union zero-fill on either field changes training rows with nothing counting it."""
    assert field in {"has_policy", "priority"}, silent_damage
    buf = db.DiskReplayBuffer(
        100,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=100,
        shard_size=100,
    )
    buf._append_shuffle_arrays(_arrays(4))
    buf._append_shuffle_arrays(_arrays(4))
    assert len(buf._shuffle_buf) == 2
    del buf._shuffle_buf[1][field]

    spanning = np.array([0, 5], dtype=np.int64)  # one row from each chunk
    with pytest.raises(ValueError, match=f"disagree about '{field}'"):
        buf._gather_rows(spanning)


def test_gather_rows_is_unaffected_when_every_chunk_carries_the_field(tmp_path) -> None:
    """Negative control: the guard must not fire on the production shape."""
    buf = db.DiskReplayBuffer(
        100,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=100,
        shard_size=100,
    )
    buf._append_shuffle_arrays(_arrays(4))
    buf._append_shuffle_arrays(_arrays(4))

    out = buf._gather_rows(np.array([0, 5], dtype=np.int64))

    assert out["has_policy"].shape == (2,)
    assert np.all(out["has_policy"] == 1)
    assert out["priority"].shape == (2,)


def test_gather_rows_preserves_legacy_network_turn_default(tmp_path) -> None:
    buf = db.DiskReplayBuffer(
        100,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=100,
        shard_size=100,
    )
    tagged = _arrays(4)
    tagged["is_network_turn"] = np.zeros(4, dtype=np.uint8)
    tagged["has_is_network_turn"] = np.ones(4, dtype=np.uint8)
    legacy = _arrays(4)
    buf._append_shuffle_arrays(tagged)
    buf._append_shuffle_arrays(legacy)

    out = buf._gather_rows(np.array([0, 5], dtype=np.int64))

    assert out["is_network_turn"].tolist() == [0, 1]


def test_array_gather_preserves_legacy_network_turn_default() -> None:
    buf = ArrayReplayBuffer(100, rng=np.random.default_rng(0))
    tagged = _arrays(4)
    tagged["is_network_turn"] = np.zeros(4, dtype=np.uint8)
    tagged["has_is_network_turn"] = np.ones(4, dtype=np.uint8)
    buf.add_many_arrays(tagged)
    buf.add_many_arrays(_arrays(4))

    out = buf._gather_rows(np.array([0, 5], dtype=np.int64))

    assert out["is_network_turn"].tolist() == [0, 1]


def test_prune_storage_arrays_refuses_a_set_flag_with_no_value_array() -> None:
    """`has_X = 1` with X absent reads downstream as a perfectly-fit head.

    The loss takes the target-absent branch (numerator structurally zero) while
    the head's mask still counts the row.

    ⚠ WHAT THIS DOES AND DOES NOT PIN. The raise comes from `validate_arrays`,
    which `prune_storage_arrays` calls on its input two lines earlier -- NOT
    from this PR's change (making the value assignment unconditional). Restoring
    the old `if value_name in arrs:` conditional leaves this test GREEN, because
    validation has already rejected the only input that could reach it. The
    conditional was misleading, not load-bearing; what is pinned here is the
    end-to-end invariant at the write-side funnel, which is the property worth
    keeping either way.
    """
    arrs = _arrays(4)
    arrs["has_sf_p0"] = np.ones((4,), dtype=np.uint8)  # flag set, target missing
    with pytest.raises(ValueError, match="has_sf_p0 is set but sf_p0_policy_target is missing"):
        prune_storage_arrays(arrs)
