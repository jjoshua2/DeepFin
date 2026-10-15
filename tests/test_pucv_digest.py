from __future__ import annotations

import hashlib
import tracemalloc

import numpy as np
import pytest

from chess_anti_engine.inference_cache import PucvEvalCache, _encoded_row_digest


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16, np.uint16, np.int8, ">f4"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly"])
def test_digest_preserves_encoded_byte_contract(dtype, layout) -> None:
    row = np.arange(146 * 64).reshape(146, 8, 8).astype(dtype)
    if layout == "fortran":
        row = np.asfortranarray(row)
    elif layout == "strided":
        row = np.stack((row, row), axis=-1)[..., 0]
    elif layout == "readonly":
        row.flags.writeable = False
    original = row.copy()
    canonical = row if row.flags["C_CONTIGUOUS"] else np.ascontiguousarray(row, dtype=np.float32)
    expected = hashlib.blake2b(canonical.tobytes(order="C"), digest_size=16).digest()
    assert _encoded_row_digest(row) == expected
    np.testing.assert_array_equal(row, original)


def test_masked_array_digest_retains_filled_bytes() -> None:
    row = np.ma.array(np.arange(32, dtype=np.float32), mask=np.arange(32) % 3 == 0)
    expected = hashlib.blake2b(row.tobytes(order="C"), digest_size=16).digest()
    assert _encoded_row_digest(row) == expected
    assert expected != hashlib.blake2b(row.data, digest_size=16).digest()


def test_array_subclass_retains_tobytes_error() -> None:
    class SpecialRow(np.ndarray):
        def tobytes(self, order="C") -> bytes:
            raise RuntimeError(f"special encoding bytes order={order}")

    row = np.zeros(32, dtype=np.float32).view(SpecialRow)
    with pytest.raises(RuntimeError, match="special encoding bytes order=C"):
        _encoded_row_digest(row)


def test_contiguous_digest_avoids_row_sized_temporary() -> None:
    row = np.zeros((175, 8, 8), dtype=np.float32)
    _encoded_row_digest(row)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        digest = _encoded_row_digest(row)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if not tracing:
            tracemalloc.stop()
    assert len(digest) == 16
    assert peak < row.nbytes // 4


def test_cache_rejects_changed_owned_row_after_put() -> None:
    cache = PucvEvalCache(max_entries=2)
    row = np.zeros((146, 8, 8), dtype=np.float32)
    key = np.array([13, 17], dtype=np.uint64)
    policy = np.arange(8, dtype=np.float32)
    wdl = np.array([.2, .3, .5], dtype=np.float32)
    cache.put(key, row, policy, wdl)
    row[0, 0, 0] = 1
    assert cache.get(key, row) is None
    row[0, 0, 0] = 0
    hit = cache.get(key, row)
    assert hit is not None
    np.testing.assert_array_equal(hit[0], policy)
    np.testing.assert_array_equal(hit[1], wdl)
