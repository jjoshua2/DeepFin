"""An interrupted fixed epoch must produce the same trained state as one pass."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pytest
import torch
import zarr

from chess_anti_engine.replay.shard import local_shard_path, save_local_shard_arrays
from chess_anti_engine.train.trainer import Trainer
from scripts import offline_replay_epoch as driver
from scripts.offline_epoch_resume import candidate_lock


def _source(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "device: cpu\nembed_dim: 16\nnum_layers: 1\nnum_heads: 2\n"
        "no_smolgen: true\nuse_nla: false\nfeature_dropout_p: 0.2\n"
        "resid_channel_dropout: 0.1\nwarmup_steps: 3\nlr: 0.001\n"
    )
    source = tmp_path / "shards"
    rng = np.random.default_rng(81)
    for index in range(2):
        n = 6
        policy = np.zeros((n, 1858), dtype=np.float16)
        policy[np.arange(n), np.arange(n) + 15 + index * n] = 1
        save_local_shard_arrays(
            local_shard_path(source, index),
            arrs={
                "x": rng.normal(0, 0.1, (n, 146, 8, 8)).astype(np.float16),
                "policy_target": policy,
                "wdl_target": np.zeros(n, dtype=np.int8),
                "priority": np.ones(n, dtype=np.float32),
                "has_policy": np.ones(n, dtype=np.uint8),
                "sf_wdl": np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float16), (n, 1)),
                "has_sf_wdl": np.ones(n, dtype=np.uint8),
                "search_wdl": np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float16), (n, 1)),
                "has_search_wdl": np.ones(n, dtype=np.uint8),
                "priority_policy_kl": np.zeros(n, dtype=np.float32),
                "has_priority_policy_kl": np.ones(n, dtype=np.uint8),
                "priority_q_delta": np.zeros(n, dtype=np.float32),
                "has_priority_q_delta": np.ones(n, dtype=np.uint8),
            },
        )
    return config, source


def _run(monkeypatch: pytest.MonkeyPatch, *, config: Path, source: Path,
         out: Path, resume: bool = False, soft_mask: bool = False,
         candidate: str = "adamw", soft_keep_prob: float = 0.8) -> None:
    argv = [
        "offline_replay_epoch.py", "--config", str(config),
        "--replay-dir", str(source), "--out-dir", str(out),
        "--candidates", candidate, "--batch-size", "2", "--epochs", "2",
        "--eval-positions", "2", "--eval-steps", "1", "--seed", "7",
        "--no-amp", "--checkpoint-every-seconds", "0.000001",
    ]
    if resume:
        argv.append("--resume")
    if soft_mask:
        argv.extend(["--vr-mode", "soft", "--vr-q-thresh", "0.1",
                     "--vr-sfe-thresh", "0.1", "--vr-kl-veto", "0.1",
                     "--vr-soft-keep-prob", str(soft_keep_prob),
                     "--sample-weight-signal", "priority"])
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()


def _same_state(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same_state(a, b)
    else:
        assert left == right


def test_candidate_lock_refuses_concurrent_resume(tmp_path: Path) -> None:
    with (candidate_lock(tmp_path / "candidate"),
          pytest.raises(ValueError, match="already active"),
          candidate_lock(tmp_path / "candidate")):
        pass


@pytest.mark.parametrize("interrupt_at", [2, 4, 7, 8])
@pytest.mark.parametrize("soft_mask", [False, True])
def test_resume_restores_real_optimizer_schedule_dropout_and_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    interrupt_at: int, soft_mask: bool,
) -> None:
    config, source = _source(tmp_path)
    uninterrupted = tmp_path / "uninterrupted"
    restarted = tmp_path / "restarted"
    _run(monkeypatch, config=config, source=source, out=uninterrupted,
         soft_mask=soft_mask)

    real_train = Trainer.train_steps
    calls = 0

    def interrupt_after_cursor(self: Trainer, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == interrupt_at:
            raise RuntimeError("injected interruption")
        return real_train(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "train_steps", interrupt_after_cursor)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _run(monkeypatch, config=config, source=source, out=restarted,
             soft_mask=soft_mask)
    monkeypatch.setattr(Trainer, "train_steps", real_train)
    committed = sorted((restarted / "adamw").glob("resume-*.json"))
    assert committed
    assert json.loads(committed[-1].read_text())["cursor"]["steps"] == interrupt_at - 1

    # A fully written weights file without its cursor is an orphan generation.
    (restarted / "adamw" / "resume-trainer-999999999999-orphan.pt").write_bytes(b"orphan")
    _run(monkeypatch, config=config, source=source, out=restarted, resume=True,
         soft_mask=soft_mask)
    a = torch.load(uninterrupted / "adamw" / "trainer.pt", map_location="cpu", weights_only=False)
    b = torch.load(restarted / "adamw" / "trainer.pt", map_location="cpu", weights_only=False)
    for key in ("model", "opt", "scheduler", "step", "peak_lr", "zclip"):
        _same_state(a[key], b[key])
    result = (restarted / "results.jsonl").read_text()
    assert len(result.splitlines()) == 1
    if soft_mask:
        assert json.loads(result)["vr_dropped"] > 0
    _run(monkeypatch, config=config, source=source, out=restarted, resume=True,
         soft_mask=soft_mask)
    assert (restarted / "results.jsonl").read_text() == result
    assert len(list((restarted / "adamw").glob("resume-*.json"))) == 2


def test_aurora_exact_resume_restores_real_optimizer_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source = _source(tmp_path)
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    candidate = "aurora_mlp_only"
    _run(monkeypatch, config=config, source=source, out=full,
         candidate=candidate, soft_mask=True)
    real_train = Trainer.train_steps
    calls = 0

    def interrupt(self: Trainer, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected interruption")
        return real_train(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "train_steps", interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _run(monkeypatch, config=config, source=source, out=partial,
             candidate=candidate, soft_mask=True)
    monkeypatch.setattr(Trainer, "train_steps", real_train)
    _run(monkeypatch, config=config, source=source, out=partial,
         candidate=candidate, soft_mask=True, resume=True)
    a = torch.load(full / candidate / "trainer.pt", map_location="cpu", weights_only=False)
    b = torch.load(partial / candidate / "trainer.pt", map_location="cpu", weights_only=False)
    for key in ("model", "opt", "scheduler", "step", "peak_lr", "zclip"):
        _same_state(a[key], b[key])


def test_all_soft_dropped_shards_advance_cursor_and_bound_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source = _source(tmp_path)
    out = tmp_path / "all_dropped"
    _run(monkeypatch, config=config, source=source, out=out,
         soft_mask=True, soft_keep_prob=0.0)
    result = json.loads((out / "results.jsonl").read_text())
    assert result["steps"] == 0
    assert result["vr_seen"] == result["vr_dropped"] == 24
    commits = list((out / "adamw").glob("resume-*.json"))
    assert len(commits) == 2
    assert len(list((out / "adamw").glob("resume-trainer-*.pt"))) == 2
    _run(monkeypatch, config=config, source=source, out=out,
         soft_mask=True, soft_keep_prob=0.0, resume=True)


def test_resume_refuses_changed_source_and_missing_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source = _source(tmp_path)
    out = tmp_path / "run"
    real_train = Trainer.train_steps

    def interrupt(_self: Trainer, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("interrupt before first batch")

    monkeypatch.setattr(Trainer, "train_steps", interrupt)
    with pytest.raises(RuntimeError, match="interrupt before first batch"):
        _run(monkeypatch, config=config, source=source, out=out)
    monkeypatch.setattr(Trainer, "train_steps", real_train)
    shard = local_shard_path(source, 0)
    group = zarr.open_group(str(shard), mode="r+")
    original = cast(Any, group["x"])[0, 0, 0, 0]
    group["x"][0, 0, 0, 0] = float(original) + 0.125
    with pytest.raises(ValueError, match="resume shard bytes changed"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)
    group["x"][0, 0, 0, 0] = original
    for sidecar in (out / "adamw").glob("resume-*.json"):
        sidecar.unlink()
    with pytest.raises(ValueError, match="existing committed trainer and cursor"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)


def test_resume_refuses_changed_spec_sidecar_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source = _source(tmp_path)
    out = tmp_path / "run"
    real_train = Trainer.train_steps

    def interrupt(_self: Trainer, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("interrupt before first batch")

    monkeypatch.setattr(Trainer, "train_steps", interrupt)
    with pytest.raises(RuntimeError, match="interrupt before first batch"):
        _run(monkeypatch, config=config, source=source, out=out)
    monkeypatch.setattr(Trainer, "train_steps", real_train)

    original_config = config.read_text()
    config.write_text(original_config.replace("lr: 0.001", "lr: 0.002"))
    with pytest.raises(ValueError, match="scientific configuration or source code changed"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)
    config.write_text(original_config)

    sidecar_path = next((out / "adamw").glob("resume-*.json"))
    original_sidecar = sidecar_path.read_bytes()
    sidecar = json.loads(original_sidecar)
    sidecar["science_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="resume cursor identity changed"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)
    sidecar_path.write_bytes(original_sidecar)

    sidecar = json.loads(original_sidecar)
    sidecar["cursor"]["positions"] = 99
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="resume cursor identity changed"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)
    sidecar_path.write_bytes(original_sidecar)

    checkpoint = out / "adamw" / sidecar["checkpoint_name"]
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="trainer checkpoint missing or changed"):
        _run(monkeypatch, config=config, source=source, out=out, resume=True)
