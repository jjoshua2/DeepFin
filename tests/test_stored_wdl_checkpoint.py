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
