"""Build a Ceres byte feed from qualified stored student rows.

This is the physical inference path. Source/legal-map validation belongs to
the independent target readback, where it is checked against the same rows.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from chess_anti_engine.encoding.ceres_tpg import stored_x_to_ceres_tpg_bytes

STORED_SHAPE = (175, 8, 8)
STORED_ROW_BYTES = 175 * 8 * 8 * 4
CERES_ROW_BYTES = 64 * 137


def ceres_feed_bytes_from_stored_rows(
    raw_rows: Sequence[bytes], input_sha256: Sequence[str], *,
    input_history_encoding: str, history_rep_fix: bool,
    physical_slots: int | None = None,
) -> bytes:
    """Validate stored-f16 lineage and convert a whole Ceres call at once.

    Each raw row is little-endian float32 representing an exact stored-f16
    value. ``input_sha256`` binds the staged bytes to qualified source rows.
    The caller must pass the qualified source's history profile explicitly;
    raw bytes and a hash alone do not attest that profile.
    Any padded slots repeat the last physical byte record, as the fixed-batch
    Ceres teacher does. The 137 feature bytes are built by the shared TPG
    encoder, including history, repetition, rule50, castling and en passant.
    """
    if input_history_encoding != "lc0_root_legacy_meta" or history_rep_fix is not True:
        raise ValueError("Ceres stored feed requires corrected lc0_root_legacy_meta history")
    count = len(raw_rows)
    if not count or count != len(input_sha256):
        raise ValueError("Ceres rows and source hashes must have equal nonzero length")
    if physical_slots is None:
        physical_slots = count
    if type(physical_slots) is not int or physical_slots < count:
        raise ValueError("physical_slots must be an integer covering every real Ceres row")

    stored_rows = []
    for raw, expected_sha in zip(raw_rows, input_sha256, strict=True):
        if len(raw) != STORED_ROW_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("staged Ceres row length or source hash mismatch")
        original = np.frombuffer(raw, dtype="<f4").reshape(STORED_SHAPE)
        if not np.isfinite(original).all():
            raise ValueError("nonfinite staged Ceres row")
        stored = original.astype("<f2")
        if stored.astype("<f4").tobytes() != raw:
            raise ValueError("Ceres row is not an exact stored-f16 round trip")
        stored_rows.append(stored)

    converted = stored_x_to_ceres_tpg_bytes(
        np.stack(stored_rows), input_history_encoding=input_history_encoding,
        history_rep_fix=history_rep_fix,
    )
    if converted.shape != (count, 64, 137) or converted.dtype != np.uint8:
        raise ValueError("unexpected Ceres TPG feed shape or dtype")
    real = converted.tobytes()
    last = converted[-1].tobytes()
    result = real + last * (physical_slots - count)
    if len(result) != physical_slots * CERES_ROW_BYTES:
        raise ValueError("unexpected Ceres physical feed byte length")
    return result
