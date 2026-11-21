"""Synthetic process/resource and artifact-publication contracts."""
from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import stored_wdl_rows_process as process
from tests.test_stored_wdl_rows import fixture


def test_real_child_synthetic_only(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    zarr.open_group(str(paths[0]), mode="a").attrs["history_rep_fix"] = True
    audit = tmp_path / "audit.jsonl"
    audit.write_bytes(data)
    result = process.run_reader(audit, hashlib.sha256(data).hexdigest(), paths,
                                tmp_path / "output", expected_history_rep_fix=True)
    out = Path(result["output_dir"])
    assert (out / "result.json").is_file()
    assert result["candidate_universe_complete"] is False
    assert result["inference_performed"] is False
    assert result["limits"]["address_space_bytes"] == 8 * 1024**3
    assert result["limits"]["cpu_seconds"] == 120
    assert result["limits"]["cuda_visible_devices"] == ""
    assert result["limits"]["omp_threads"] == "1"
    assert result["project_source_sha256"]["scripts/stored_wdl_rows.py"]
    assert any("._lc0_ext" in name for name in result["loaded_imports"])
    with np.load(out / "rows.npz", allow_pickle=False) as arrays:
        assert arrays["x"].shape == (2, 175, 8, 8)
    second = process.run_reader(audit, hashlib.sha256(data).hexdigest(), paths,
                                tmp_path / "output", expected_history_rep_fix=True)
    assert result["output_dir"] != second["output_dir"]


def test_child_failure_has_no_success_marker(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    audit = tmp_path / "audit.jsonl"
    audit.write_bytes(data)
    with pytest.raises(ValueError, match="reader failed"):
        process.run_reader(audit, "0"*64, paths, tmp_path / "output",
                           expected_history_rep_fix=True)
    dirs = list((tmp_path / "output").iterdir())
    assert len(dirs) == 1
    assert not (dirs[0] / "result.json").exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), OSError("interrupted"),
                                    subprocess.TimeoutExpired("owned", 150)])
def test_all_abnormal_exits_kill_and_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException,
) -> None:
    calls = []
    class Fake:
        pid = 987654
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.calls = 0
            assert kwargs["start_new_session"] is True
        def communicate(self, *_args: object, **_kwargs: object) -> tuple[None, None]:
            self.calls += 1
            calls.append("communicate")
            if self.calls == 1:
                raise error
            return None, None
        def poll(self) -> None:
            return None
    monkeypatch.setattr(process.subprocess, "Popen", Fake)
    monkeypatch.setattr(process.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    expected = ValueError if isinstance(error, subprocess.TimeoutExpired) else type(error)
    with pytest.raises(expected):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output", expected_history_rep_fix=True)
    assert calls == ["communicate", (987654, signal.SIGKILL), "communicate"]
    assert not list((tmp_path / "output").glob("*/result.json"))


def test_dirty_nonzero_child_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Fake:
        returncode = 1
        def __init__(self, *_args: object, **kwargs: object) -> None:
            pass
        def communicate(self, request: str, **_kwargs: object) -> tuple[None, None]:
            out = Path(json.loads(request)["output_dir"])
            (out / "validated-candidate.json").write_text('{"status":"COMPLETE_DESCRIPTIVE_PREFIX"}')
            return None, None
    monkeypatch.setattr(process.subprocess, "Popen", Fake)
    with pytest.raises(ValueError, match="reader failed"):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output", expected_history_rep_fix=True)
    assert not list((tmp_path / "output").glob("*/result.json"))


@pytest.mark.parametrize("kwargs", [{"expected_history_rep_fix": 1},
                                     {"expected_history_rep_fix": True, "count": True},
                                     {"expected_history_rep_fix": True, "count": 257}])
def test_bad_request_fails_before_child(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="explicit history"):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output", **kwargs)


def test_oversize_request_does_not_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process, "MAX_REQUEST", 10)
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_a, **_k: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="oversized"):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output", expected_history_rep_fix=True)

@pytest.mark.parametrize("bad_field", ["artifact_sha256", "requested_shard_paths", "expected_history_rep_fix"])
def test_clean_exit_mismatch_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_field: str,
) -> None:
    class Fake:
        returncode = 0
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass
        def communicate(self, request: str, **_kwargs: object) -> tuple[None, None]:
            payload = json.loads(request)
            out = Path(payload["output_dir"])
            (out / "rows.npz").write_bytes(b"synthetic fake child bytes")
            receipt = {"status": "COMPLETE_DESCRIPTIVE_PREFIX",
                       "request_binding": {k: v for k, v in payload.items() if k != "output_dir"},
                       "audit_sha256": payload["audit_sha256"],
                       "requested_shard_paths": payload["shard_paths"],
                       "expected_history_rep_fix": True,
                       "artifact_sha256": process._digest(out / "rows.npz")}
            receipt[bad_field] = False
            (out / "validated-candidate.json").write_text(json.dumps(receipt))
            return None, None
    monkeypatch.setattr(process.subprocess, "Popen", Fake)
    with pytest.raises(ValueError, match="identity mismatch"):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output", expected_history_rep_fix=True)
    assert not list((tmp_path / "output").glob("*/result.json"))

def test_artifact_is_synced_before_candidate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import torch
    from scripts import stored_wdl_rows
    audit = tmp_path / "audit"
    audit.write_bytes(b"synthetic input")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _n: None)
    monkeypatch.setattr(process, "_source_snapshot", lambda _p: {})
    monkeypatch.setattr(stored_wdl_rows, "scan_prefix", lambda *_a, **_k: {
        "x": np.zeros((2, 175, 8, 8), dtype=np.float32),
        "targets": np.full((2, 3), 1/3), "limits": {}, "audit_sha256": "synthetic",
    })
    synced = []
    original = os.fsync
    def observe(fd: int) -> None:
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        original(fd)
    monkeypatch.setattr(process.os, "fsync", observe)
    process._execute({"audit_path": str(audit), "audit_sha256": "synthetic",
                      "shard_paths": [], "expected_history_rep_fix": True,
                      "count": 2, "output_dir": str(out)})
    assert synced == [str(out / "rows.npz"), str(out),
                      str(out / "validated-candidate.json"), str(out)]
    assert not (out / "result.json").exists()


def test_parent_checks_size_before_hash_and_syncs_success_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_observations = []
    class Fake:
        returncode = 0
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass
        def communicate(self, request: str, **_kwargs: object) -> tuple[None, None]:
            payload = json.loads(request)
            out = Path(payload["output_dir"])
            artifact = out / "rows.npz"
            artifact.write_bytes(b"valid synthetic child output")
            receipt = {"status": "COMPLETE_DESCRIPTIVE_PREFIX",
                       "request_binding": {k: v for k, v in payload.items() if k != "output_dir"},
                       "audit_sha256": payload["audit_sha256"],
                       "requested_shard_paths": payload["shard_paths"],
                       "expected_history_rep_fix": True,
                       "artifact_sha256": process._digest(artifact)}
            (out / "validated-candidate.json").write_text(json.dumps(receipt))
            return None, None
    original = process._sync_directory
    def sync(path: Path) -> None:
        sync_observations.append((path, (path / "result.json").exists()))
        original(path)
    monkeypatch.setattr(process.subprocess, "Popen", Fake)
    monkeypatch.setattr(process, "_sync_directory", sync)
    result = process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                                tmp_path / "output", expected_history_rep_fix=True)
    assert sync_observations[-1] == (Path(result["output_dir"]), True)
    monkeypatch.setattr(process, "MAX_FILE", 1)
    with pytest.raises(ValueError, match="artifact size"):
        process.run_reader(tmp_path / "audit", "0"*64, (tmp_path / "one.zarr",),
                           tmp_path / "output2", expected_history_rep_fix=True)
    assert not list((tmp_path / "output2").glob("*/result.json"))

def test_direct_real_child_is_target_free_and_identity_bound(tmp_path: Path) -> None:
    from tests.test_stored_wdl_direct import direct_fixture
    path, kwargs = direct_fixture(tmp_path)
    result = process.run_direct_reader(path, tmp_path / "direct-output", **kwargs)
    out = Path(result["output_dir"])
    assert result["status"] == "COMPLETE_DESCRIPTIVE_DIRECT"
    assert result["mode"] == "descriptive-producer-rows"
    assert result["request_binding"]["expected_rows"] == 3
    assert result["request_binding"]["expected_attrs_sha256"] == kwargs["expected_attrs_sha256"]
    assert result["project_source_sha256"]["scripts/stored_wdl_direct.py"]
    assert "targets" not in result
    with np.load(out / "rows.npz", allow_pickle=False) as arrays:
        assert arrays.files == ["x"]
        assert arrays["x"].shape == (3, 175, 8, 8)
    assert result["limits"]["torch_threads"] == 1
    assert result["limits"]["cuda_visible_devices"] == ""


def test_direct_bad_metadata_never_publishes(tmp_path: Path) -> None:
    from tests.test_stored_wdl_direct import direct_fixture
    path, kwargs = direct_fixture(tmp_path)
    kwargs["expected_attrs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reader failed"):
        process.run_direct_reader(path, tmp_path / "direct-output", **kwargs)
    assert not list((tmp_path / "direct-output").glob("*/result.json"))


@pytest.mark.parametrize("change", [
    {"expected_rows": True}, {"expected_rows": 257},
    {"expected_history_rep_fix": 1}, {"expected_attrs_sha256": None},
    {"expected_seal_sha256": "not-a-hash"},
])
def test_direct_invalid_request_precedes_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict,
) -> None:
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_a, **_k: pytest.fail("spawned"))
    kwargs: dict[str, Any] = {"expected_rows": 3, "expected_history_rep_fix": True,
              "expected_attrs_sha256": "a" * 64, "expected_seal_sha256": "b" * 64, **change}
    with pytest.raises(ValueError, match=r"explicit|exact metadata"):
        process.run_direct_reader(tmp_path / "direct.zarr", tmp_path / "output", **kwargs)
