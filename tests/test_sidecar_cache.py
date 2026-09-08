"""On-disk layout compatibility and conservative actual NPY header reservation."""
import io

import numpy as np

from scripts import corpus_row_provenance as provenance
from scripts.sidecar_cache import raw_identity_cache_bytes, rank_identity_dtype


def test_raw_identity_layout_and_numpy_header_fit_existing_reservation():
    legacy = np.dtype([
        ("source", "<u4"), ("row", "<u4"), ("game_id", "<i8"),
        ("ply", "<i4"), ("worker_id", "<i4"),
        ("input_key", "u1", (16,)), ("stored_input_key", "u1", (16,)),
    ])
    assert legacy == provenance.RECORD_DTYPE
    written = 0
    for rows in (1, 9, 8192):
        array = np.zeros(rows, dtype=provenance.RECORD_DTYPE)
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        assert len(stream.getvalue()) <= raw_identity_cache_bytes(rows, 1)
        written += len(stream.getvalue())
    assert written <= raw_identity_cache_bytes(1 + 9 + 8192, 3)


def test_rank_layout_preserves_stored_fields_and_top_k_width():
    for top_k in (1, 3, 5):
        legacy = np.dtype([
            ("game_id", "<i8"), ("ply", "<i4"), ("worker_id", "<i4"),
            ("input_key", "u1", (16,)), ("stored_input_key", "u1", (16,)),
            ("indices", "<u2", (top_k,)), ("gaps", "<f4", (top_k,)),
            ("count", "u1"), ("valid", "u1"),
        ])
        assert rank_identity_dtype(top_k) == legacy
