"""A capped ruler inherits provenance only from its matching complete source."""
from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.eval.era_probe import (
    PROBE_SET_FIELDS,
    probe_set_digest,
    provenance_path,
)
from chess_anti_engine.replay.shard import load_shard_arrays
from tests.test_era_forgetting_probe import _load, _probe_arrays, _write_probe_set


def _legacy_digest(arrays: dict[str, np.ndarray]) -> str:
    """Original on-disk digest encoding, independent of the new buffer path."""
    h = hashlib.sha256()
    for name in PROBE_SET_FIELDS:
        if name not in arrays:
            h.update(f"{name}:absent".encode())
            continue
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}:{a.shape}:{a.dtype.str}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_cap_rejects_stale_full_set_even_with_identical_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cap: int,
) -> None:
    original = _probe_arrays(n=4)
    changed = {k: v.copy() for k, v in original.items()}
    changed["x"][3, -1, 7, 7] = 3.0
    path = _write_probe_set(tmp_path / "probe.npz", changed, digest_of=original)
    probe = _load(path, max_rows=cap)
    assert probe.provenance == {}
    assert probe.digest == _legacy_digest({k: v[:cap] for k, v in original.items()})
    assert probe.n_rows == cap
    for key in PROBE_SET_FIELDS:
        np.testing.assert_array_equal(probe.arrays[key], original[key][:cap])
    output = capsys.readouterr().out
    assert "describes a DIFFERENT set" in output
    assert "desync_screened=unrecorded" in output
    assert "Provenance still applies" not in output


@pytest.mark.parametrize("cap", [1, 2, 0, 4, 8])
def test_matching_full_provenance_preserves_selected_ruler(
    tmp_path: Path, cap: int,
) -> None:
    arrays = _probe_arrays(n=4)
    probe = _load(_write_probe_set(tmp_path / "probe.npz", arrays), max_rows=cap)
    keep = min(cap, 4) if cap else 4
    assert probe.provenance["desync_screened"] is True
    assert probe.provenance["digest"] == _legacy_digest(arrays)
    assert probe.digest == _legacy_digest({k: v[:keep] for k, v in arrays.items()})
    assert (probe.n_rows, probe.n_policy_rows) == (keep, keep)
    for key in PROBE_SET_FIELDS:
        np.testing.assert_array_equal(probe.arrays[key], arrays[key][:keep])


@pytest.mark.parametrize("cap", [0, 2])
@pytest.mark.parametrize("digest", ["<absent>", None, "", 17, False])
def test_unbound_sidecar_cannot_claim_screening(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cap: int, digest: object,
) -> None:
    path = _write_probe_set(tmp_path / "probe.npz", _probe_arrays(n=4))
    data = json.loads(provenance_path(path).read_text())
    if digest == "<absent>":
        data.pop("digest")
    else:
        data["digest"] = digest
    provenance_path(path).write_text(json.dumps(data))
    assert _load(path, max_rows=cap).provenance == {}
    output = capsys.readouterr().out
    assert "no usable digest" in output or "describes a DIFFERENT set" in output
    assert "desync_screened=unrecorded" in output


@pytest.mark.parametrize("cap", [0, 2])
def test_missing_sidecar_keeps_usable_but_unvouched_rows(tmp_path: Path, cap: int) -> None:
    path = _write_probe_set(tmp_path / "probe.npz", _probe_arrays(n=4))
    provenance_path(path).unlink()
    probe = _load(path, max_rows=cap)
    assert probe.provenance == {}
    assert probe.n_rows == (cap or 4)


def test_capped_fortran_npz_binds_full_source(tmp_path: Path) -> None:
    arrays = _probe_arrays(n=4)
    arrays["x"] = np.asfortranarray(arrays["x"])
    arrays["x"][:, -1, 7, 7] = np.arange(4)
    path = _write_probe_set(tmp_path / "probe.npz", arrays)
    loaded, _ = load_shard_arrays(path)
    assert loaded["x"].flags.f_contiguous
    assert not loaded["x"].flags.c_contiguous
    probe = _load(path, max_rows=2)
    assert probe.provenance["digest"] == _legacy_digest(loaded)
    assert probe.digest == _legacy_digest({k: loaded[k][:2] for k in PROBE_SET_FIELDS})
    np.testing.assert_array_equal(probe.arrays["x"], loaded["x"][:2])


def test_cap_checks_provenance_before_policy_eligibility_intersection(tmp_path: Path) -> None:
    arrays = _probe_arrays(n=4, has_mask=np.array([0, 1, 1, 1], dtype=np.uint8))
    probe = _load(_write_probe_set(tmp_path / "probe.npz", arrays), max_rows=2)
    assert probe.provenance["digest"] == _legacy_digest(arrays)
    assert probe.digest == _legacy_digest({k: v[:2] for k, v in arrays.items()})
    assert probe.n_policy_rows == 1
    np.testing.assert_array_equal(probe.arrays["has_sf_p0_regret"], [0, 1])


@pytest.mark.parametrize("array", [
    np.arange(24, dtype=np.float16).reshape(2, 3, 4),
    np.arange(24, dtype=np.dtype(">f4")).reshape(2, 3, 4),
    np.asfortranarray(np.arange(24, dtype=np.int16).reshape(2, 3, 4)),
    np.arange(48, dtype=np.uint8).reshape(4, 3, 4)[::-1, :, ::2],
    np.empty((0, 3, 4), dtype=np.float32),
    np.array(3, dtype=np.int64),
])
def test_buffer_digest_preserves_legacy_bytes(array: np.ndarray) -> None:
    arrays = {"x": array}
    assert probe_set_digest(arrays) == _legacy_digest(arrays)


@pytest.mark.parametrize("fortran", [False, True])
def test_digest_does_not_copy_a_complete_large_array(fortran: bool) -> None:
    array = np.zeros((128, 128, 128), dtype=np.float32, order="F" if fortran else "C")
    arrays = {"x": array}
    expected = _legacy_digest(arrays)
    tracemalloc.start()
    try:
        actual = probe_set_digest(arrays)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    # An 8MiB source is preallocated. Noncontiguous hashing has a 1MiB scratch
    # bound, not a full-source contiguous copy followed by another bytes copy.
    assert peak < array.nbytes // 2
