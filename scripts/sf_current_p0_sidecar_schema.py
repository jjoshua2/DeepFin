"""Numeric carrier contract; masks describe source checks, never training admission."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import struct
from typing import Any

SCHEMA = "CURRENT_P0_SF_SIDECAR_V1"
# Every row is retained. -1 native counts and zero float padding are non-observations.
FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "policy_f16": ("<f2", (1858,)),
    "stored_record_wdl_f16": ("<f2", (3,)),
    "stored_query_wdl_f16": ("<f2", (3,)),
    "native_wdl_permille_i16": ("<i2", (3,)),
    "legal_mask_u8": ("|u1", (1858,)),
    "qualified_policy": ("|u1", ()),
    "qualified_stored_wdl": ("|u1", ()),
    "qualified_native_wdl": ("|u1", ()),
    "predecessor_meta_i32": ("<i4", (6,)),
    "observed_predecessor_meta": ("|u1", ()),
    "predecessor_pv_i16": ("<i2", (48, 5)),
    "observed_predecessor_pv": ("|u1", ()),
}


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def arrays_for_rows(rows: int) -> dict[str, Any]:
    import numpy as np

    require(type(rows) is int and rows > 0, "positive carrier rows")
    result = {name: np.zeros((rows, *tail), dtype=dtype) for name, (dtype, tail) in FIELDS.items()}
    result["native_wdl_permille_i16"][:] = -1
    result["predecessor_pv_i16"][:] = (-1, -32768, 0, -1, -1)
    return result


def decoded_bytes(rows: int) -> int:
    import math
    import numpy as np

    return sum(rows * math.prod(tail) * np.dtype(dtype).itemsize for dtype, tail in FIELDS.values())


def array_digest(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def check_arrays(arrays: dict[str, Any], rows: int, cap: int) -> None:
    import numpy as np

    require(set(arrays) == set(FIELDS) and decoded_bytes(rows) <= cap, "carrier fields/decoded cap")
    for name, (dtype, tail) in FIELDS.items():
        value = arrays[name]
        require(value.dtype == np.dtype(dtype) and value.shape == (rows, *tail), "carrier dtype/shape: " + name)
        require(bool(np.isfinite(value).all()), "carrier nonfinite: " + name)
        if name.startswith(("qualified_", "observed_")) or name == "legal_mask_u8":
            require(bool(np.isin(value, [0, 1]).all()), "carrier flag/mask: " + name)
    require(np.array_equal(arrays["stored_query_wdl_f16"], arrays["stored_record_wdl_f16"][:, ::-1]),
            "stored WDL one-flip identity")
    for i in range(rows):
        if arrays["qualified_policy"][i]:
            policy = arrays["policy_f16"][i]
            require(bool((policy >= 0).all()) and abs(float(policy.astype(np.float64).sum()) - 1) <= .002
                    and not np.any(policy[arrays["legal_mask_u8"][i] == 0]), "qualified carrier policy")
        if arrays["qualified_stored_wdl"][i]:
            value = arrays["stored_query_wdl_f16"][i]
            require(bool((value >= 0).all()) and abs(float(value.astype(np.float64).sum()) - 1) <= .002,
                    "qualified carrier stored WDL")
        if arrays["qualified_native_wdl"][i]:
            value = arrays["native_wdl_permille_i16"][i]
            require(bool((value >= 0).all()) and int(value.sum()) == 1000, "qualified native permille")


def load_npy(path: Path, name: str, rows: int, cap: int) -> Any:
    """Refuse unsafe/oversized headers before NumPy loads the plain numeric file."""
    import numpy as np

    dtype, tail = FIELDS[name]
    expected = rows * np.dtype(dtype).itemsize
    for n in tail:
        expected *= n
    require(expected <= cap, "carrier array cap")
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "not a plain NPY")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00"), "unsupported NPY version")
        length_size = 2 if version == b"\x01\x00" else 4
        raw = stream.read(length_size)
        require(len(raw) == length_size, "short NPY length")
        length = struct.unpack("<H" if length_size == 2 else "<I", raw)[0]
        require(0 < length <= 4096, "NPY header cap")
        header = ast.literal_eval(stream.read(length).decode("latin1"))
        require(set(header) == {"descr", "fortran_order", "shape"} and header["descr"] == dtype
                and header["fortran_order"] is False and header["shape"] == (rows, *tail), "NPY header contract")
        require(path.stat().st_size == stream.tell() + expected, "NPY byte length/trailing payload")
    return np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=4096)
