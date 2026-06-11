"""Shared device-cached policy lookup tensors (moves/torch_maps.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
)
from chess_anti_engine.moves.torch_maps import (
    compact_to_full_index,
    compact_to_full_index_for,
    full_to_compact_index,
    policy_index_remap_table,
)


def test_tables_match_numpy_sources_and_cache_by_device():
    dev = torch.device("cpu")
    c2f = compact_to_full_index(dev)
    f2c = full_to_compact_index(dev)
    np.testing.assert_array_equal(c2f.numpy(), COMPACT_TO_FULL_POLICY)
    np.testing.assert_array_equal(f2c.numpy(), FULL_TO_COMPACT_POLICY)
    # Same device -> same cached tensor object, including via the tensor-keyed helper.
    assert compact_to_full_index(dev) is c2f
    assert compact_to_full_index_for(torch.zeros(3)) is c2f


def test_remap_table_widths():
    dev = torch.device("cpu")
    assert policy_index_remap_table(POLICY_SIZE, POLICY_SIZE, dev) is None
    t = policy_index_remap_table(POLICY_SIZE, COMPACT_POLICY_SIZE, dev)
    assert t is not None and t.shape == (POLICY_SIZE,)
    t2 = policy_index_remap_table(COMPACT_POLICY_SIZE, POLICY_SIZE, dev)
    assert t2 is not None and t2.shape == (COMPACT_POLICY_SIZE,)
    with pytest.raises(ValueError, match="incompatible"):
        policy_index_remap_table(123, POLICY_SIZE, dev)
