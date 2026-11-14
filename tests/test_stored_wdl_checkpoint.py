"""Synthetic checkpoints only; no repository training artifacts are opened."""
import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from scripts.stored_wdl_checkpoint import _canonical, qualify_checkpoint


@pytest.fixture
def checkpoint(tmp_path: Path):
    config = ModelConfig(kind="transformer", embed_dim=32, num_layers=1,
                         num_heads=4, use_smolgen=False,
                         input_history_encoding="lc0_root_legacy_meta",
                         input_extra_features="v2_threats", history_rep_fix=False)
    model = build_model(config)
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(config)}
    payload = {"arch": arch, "model": model.state_dict()}
    return tmp_path / "synthetic.pt", payload


def save(path: Path, payload: dict) -> str:
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_transformer_strict_load_no_inference(checkpoint) -> None:
    path, payload = checkpoint
    digest = save(path, payload)
    threads = torch.get_num_threads()
    report = qualify_checkpoint(path, digest, _canonical(payload["arch"]))
    assert report["sha256"] == digest
    assert report["head_declared"] == "wdl"
    assert report["inference_performed"] is False
    assert report["production_qualified"] is False
    assert torch.get_num_threads() == threads
    assert report["limits"] == {"address_space_bytes": 16 * 1024**3,
                                "cpu_seconds": 60, "torch_threads": 1,
                                "cuda_visible_devices": ""}
    assert len(report["source_revision"]) == 40
    assert len(report["loader_sha256"]) == 64


@pytest.mark.parametrize("fault", ["hash", "architecture", "missing_tensor", "nonfinite", "encoding", "dtype"])
def test_rejects_invalid_checkpoint(checkpoint, fault: str) -> None:
    path, payload = checkpoint
    if fault == "missing_tensor":
        payload["model"].pop(next(iter(payload["model"])))
    if fault == "nonfinite":
        value = next(iter(payload["model"].values()))
        value.flatten()[0] = float("nan")
    if fault == "dtype":
        key = next(iter(payload["model"]))
        payload["model"][key] = payload["model"][key].double()
    if fault == "encoding":
        payload["arch"]["input_history_encoding"] = "lc0_legacy"
    digest = save(path, payload)
    arch = dict(payload["arch"])
    if fault == "architecture":
        arch["num_layers"] = 2
    if fault == "hash":
        digest = "0" * 64
    with pytest.raises(ValueError, match="qualification failed"):
        qualify_checkpoint(path, digest, _canonical(arch))


@pytest.mark.parametrize("kind", ["empty", "oversized", "directory"])
def test_file_bounds_fail_before_deserialization(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "bounded"
    if kind == "directory":
        path.mkdir()
    else:
        with path.open("wb") as stream:
            if kind == "oversized":
                stream.truncate(2 * 1024**3 + 1)
    with pytest.raises(ValueError, match=r"bounded regular checkpoint|IsADirectoryError"):
        qualify_checkpoint(path, "0" * 64, "{}")


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "exit", "io"])
def test_timeout_terminates_only_owned_group(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    import subprocess
    import scripts.stored_wdl_checkpoint as module

    killed = []
    class Process:
        pid = 123456789
        returncode = None
        calls = 0
        def poll(self):
            return None
        def communicate(self, _request=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                assert timeout is not None
                if failure == "interrupt":
                    raise KeyboardInterrupt
                if failure == "exit":
                    raise SystemExit
                if failure == "io":
                    raise OSError("synthetic pipe failure")
                raise subprocess.TimeoutExpired("synthetic child", timeout)
            return "", ""
    process = Process()
    def popen(*_args, **kwargs):
        assert kwargs["start_new_session"] is True
        return process
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    exception = {"timeout": ValueError, "interrupt": KeyboardInterrupt,
                 "exit": SystemExit, "io": OSError}[failure]
    with pytest.raises(exception):
        qualify_checkpoint(Path("synthetic"), "0" * 64, "{}")
    assert killed == [(process.pid, module.signal.SIGKILL)]
    assert process.calls == 2


@pytest.mark.parametrize("conflicting", [False, True])
def test_shared_parameter_alias_contract(conflicting: bool) -> None:
    from scripts.stored_wdl_checkpoint import _validate_aliases
    model = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(3))
    model.register_parameter("first", weight)
    model.register_parameter("second", weight)
    state = {"first": torch.ones(3), "second": torch.ones(3)}
    if conflicting:
        state["second"] = torch.zeros(3)
        with pytest.raises(ValueError, match="shared-parameter aliases"):
            _validate_aliases(model, state)
    else:
        _validate_aliases(model, state)


def test_synthetic_transformer_forward_repeat_and_batch_invariance(checkpoint) -> None:
    import numpy as np
    from scripts.stored_wdl_checkpoint import forward_checkpoint
    from scripts.stored_wdl_compare import rows_digest
    path, payload = checkpoint
    digest = save(path, payload)
    planes = np.zeros((4, 175, 8, 8), np.float32)
    planes[:, 17, 0, 0] = [.1, .2, .3, .4]
    ids, groups = ["r1", "r2", "r3", "r4"], ["g1", "g1", "g2", "g2"]
    identity = rows_digest(planes, ids, groups)
    def run(batch):
        return forward_checkpoint(path, digest, _canonical(payload["arch"]), planes,
                                  row_ids=ids, group_ids=groups, expected_rows_sha256=identity,
                                  batch_size=batch)
    first, repeated, batched = run(1), run(1), run(4)
    np.testing.assert_array_equal(first["probabilities"], repeated["probabilities"])
    np.testing.assert_allclose(first["probabilities"], batched["probabilities"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(first["probabilities"]).sum(axis=1), 1, atol=1e-6)
    assert first["inference_performed"] is True
    assert first["controls_performed"] == []
    assert first["output_semantics_measured"] is False
    assert first["production_qualified"] is False


@pytest.mark.parametrize("fault", ["identity", "bool_batch", "large_batch", "many_rows", "dtype"])
def test_forward_input_rejected_before_child(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import numpy as np
    import scripts.stored_wdl_checkpoint as module
    from scripts.stored_wdl_compare import rows_digest
    x = np.zeros((2, 175, 8, 8), np.float32)
    ids = ["r1", "r2"]
    identity = rows_digest(x, ids, ids)
    batch = 1
    if fault == "identity":
        identity = "0" * 64
    elif fault == "bool_batch":
        batch = True
    elif fault == "large_batch":
        batch = 65
    elif fault == "many_rows":
        x = np.zeros((257, 175, 8, 8), np.float32)
    else:
        x = x.astype(np.float64)
    def forbidden(_payload):
        pytest.fail("child launched for invalid input")
    monkeypatch.setattr(module, "_run_child", forbidden)
    with pytest.raises(ValueError, match=r"bounded|identity"):
        module.forward_checkpoint(Path("not-opened"), "a" * 64, "{}", x,
                                  row_ids=ids, group_ids=ids, expected_rows_sha256=identity,
                                  batch_size=batch)


@pytest.mark.parametrize("fault", ["length", "digest", "bool_rows", "bool_batch"])
def test_child_revalidates_wire_identity(fault: str) -> None:
    import base64
    import numpy as np
    from scripts.stored_wdl_checkpoint import _decode_planes
    from scripts.stored_wdl_compare import rows_digest
    x = np.zeros((2, 175, 8, 8), np.float32)
    ids = ["r1", "r2"]
    request = {"rows": 2, "batch_size": 1, "row_ids": ids, "group_ids": ids,
               "rows_sha256": rows_digest(x, ids, ids),
               "planes_base64": base64.b64encode(x.tobytes()).decode("ascii")}
    if fault == "length":
        request["planes_base64"] = ""
    elif fault == "digest":
        request["rows_sha256"] = "0" * 64
    elif fault == "bool_rows":
        request["rows"] = True
    else:
        request["batch_size"] = True
    with pytest.raises(ValueError, match=r"bounded|identity|byte count"):
        _decode_planes(request)


def test_two_strict_loads_share_one_interop_thread(checkpoint):
    import json
    import os
    import subprocess
    import sys

    path, payload = checkpoint
    digest = save(path, payload)
    request = {"path": str(path), "sha256": digest,
               "architecture_json": _canonical(payload["arch"])}
    script = """
import json, sys, torch
from scripts.stored_wdl_checkpoint import _qualify
request = json.loads(sys.argv[1])
def outer(model, receipt):
    def inner(other, second):
        assert model is not other
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        assert all(torch.equal(value, other.state_dict()[name])
                   for name, value in model.state_dict().items())
        return {'paired': True}
    return _qualify(request, verified_model_callback=inner)
report = _qualify(request, verified_model_callback=outer)
assert report['diagnostic']['diagnostic']['paired'] is True
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", script, json.dumps(request)],
                   cwd=Path(__file__).resolve().parents[1], env=env,
                   check=True, timeout=30, capture_output=True, text=True)
