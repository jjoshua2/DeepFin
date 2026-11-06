"""Optional exact-root evidence, independent of the latest policy candidates.

The checksum detects accidental field corruption. It is not authentication,
nor proof that an engine evaluated the requested position. Legality is checked
when the producing search result is attached to that position.

Optional int32[8] layout: version, action in this policy encoding, CP, mate,
score kind (1=CP, 2=mate including mate zero), policy width, root CRC31,
policy CRC31. CRCs use canonical little-endian int32 bytes; the second covers
all non-padding raw policy rows in order. Invalid evidence never gets a new
seal during encoding or mirror transforms. WDL and coordinated transplantation
of an entire valid witness/policy pair are not authenticated by these checks.
"""
from __future__ import annotations

from collections.abc import Callable
import zlib

import numpy as np


SF_EVAL_WITNESS_LEN = 8
SF_EVAL_WITNESS_VERSION = 1


def _checksum(values: np.ndarray) -> int:
    return zlib.crc32(np.asarray(values, dtype="<i4").tobytes()) & 0x7FFFFFFF


def _policy_checksum(raw: np.ndarray | None) -> int | None:
    if raw is None:
        return _checksum(np.empty((0, 5), dtype=np.int32))
    rows = np.asarray(raw)
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.issubdtype(rows.dtype, np.integer):
        return None
    if np.any(rows < -(2**31)) or np.any(rows > 2**31 - 1):
        return None
    return _checksum(rows[rows[:, 0] >= 0])


def make_sf_eval_witness(
    action: int, cp: int | None, mate: int | None, raw: np.ndarray | None,
    *, policy_width: int = 4672,
) -> np.ndarray:
    """Pack an observed exact root tuple; action -1 records invalid identity."""
    values = np.array([
        SF_EVAL_WITNESS_VERSION,
        int(action),
        -32768 if cp is None else max(-32000, min(32000, int(cp))),
        0 if mate is None else max(-127, min(127, int(mate))),
        1 if cp is not None and mate is None else 2 if cp is None and mate is not None else 0,
        int(policy_width),
        0,
        -1,
    ], dtype=np.int32)
    values[6] = _checksum(values[:6])
    policy_checksum = _policy_checksum(raw)
    if policy_checksum is not None:
        values[7] = policy_checksum
    return values


def valid_sf_eval_witness(
    witness: np.ndarray, raw: np.ndarray | None, *, policy_width: int | None = None,
) -> bool:
    """Check version, domain, identity and integrity, without blessing a copy."""
    values = np.asarray(witness)
    if values.shape != (SF_EVAL_WITNESS_LEN,) or not np.issubdtype(values.dtype, np.integer):
        return False
    version, action, cp, mate, kind, width, checksum, policy_checksum = (int(v) for v in values)
    if width not in (1858, 4672) or (policy_width is not None and width != int(policy_width)):
        return False
    if version != SF_EVAL_WITNESS_VERSION or not 0 <= action < width:
        return False
    if not (-32000 <= cp <= 32000 or cp == -32768) or not -127 <= mate <= 127:
        return False
    if not ((kind == 1 and cp != -32768 and mate == 0) or (kind == 2 and cp == -32768)):
        return False
    return checksum == _checksum(values[:6]) and policy_checksum == _policy_checksum(raw)


def sf_eval_witness_mismatch(
    witness: np.ndarray, meta: np.ndarray, raw: np.ndarray | None, *, policy_width: int = 4672,
) -> bool:
    """Invalid evidence or a scalar/witness disagreement is not a clean row."""
    if not valid_sf_eval_witness(witness, raw, policy_width=policy_width):
        return True
    values = np.asarray(meta)
    if values.ndim != 1 or values.size < 4 or not np.issubdtype(values.dtype, np.integer):
        return True
    return bool(witness[2] != values[2] or witness[3] != values[3])


def copy_sf_eval_witness(witness: np.ndarray | None, raw: np.ndarray | None) -> np.ndarray | None:
    """Copy before any narrowing cast; malformed evidence stays explicitly invalid."""
    if witness is None:
        return None
    if not valid_sf_eval_witness(witness, raw):
        return np.zeros(SF_EVAL_WITNESS_LEN, dtype=np.int32)
    return np.array(witness, dtype=np.int32, copy=True)


def remap_sf_eval_witness(
    witness: np.ndarray | None, map_action: Callable[[int], int],
    raw_before: np.ndarray | None, raw_after: np.ndarray | None, *, source_width: int = 4672,
    target_width: int | None = None,
) -> np.ndarray | None:
    """Remap only verified evidence; corrupt input must never get a fresh seal."""
    if witness is None:
        return None
    result = np.array(witness, copy=True)
    if not valid_sf_eval_witness(result, raw_before, policy_width=source_width):
        return np.zeros(SF_EVAL_WITNESS_LEN, dtype=np.int32)
    result[1] = int(map_action(int(result[1])))
    result[5] = source_width if target_width is None else int(target_width)
    result[6] = _checksum(result[:6])
    policy_checksum = _policy_checksum(raw_after)
    result[7] = -1 if policy_checksum is None else policy_checksum
    return result
