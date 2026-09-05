"""Opt-in runtime upgrade checks; neither mode is part of ordinary pytest."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stage(report: dict[str, Any], name: str) -> None:
    report["stage"] = name
    print(f"stage={name}", flush=True)


def _gpu(report: dict[str, Any], memory_fraction: float) -> None:
    # Set before importing torch so compiler subprocesses inherit these limits.
    for key, value in {
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MAX_JOBS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }.items():
        os.environ[key] = value
    import torch

    from chess_anti_engine.model import ModelConfig, build_model, infer_input_planes

    torch.set_num_threads(2)
    torch.manual_seed(20260905)
    report.update(torch=torch.__version__, cuda=torch.version.cuda, memory_fraction=memory_fraction)
    _check(torch.cuda.is_available(), "CUDA is unavailable")
    report.update(
        device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        compiled_architectures=torch.cuda.get_arch_list(),
    )
    _check(torch.cuda.is_bf16_supported(), "This device does not support BF16")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    config = ModelConfig(kind="transformer", embed_dim=64, num_layers=2, num_heads=4, use_smolgen=True)
    model = build_model(config).cuda().eval()
    x = torch.randn(2, infer_input_planes(config.input_extra_features), 8, 8, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    _stage(report, "bf16_training")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            loss = output["policy_own"].float().square().mean() + output["wdl"].float().square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        _check(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), "Non-finite or missing gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()
        _check(bool(torch.isfinite(loss)), "Non-finite loss")

    _stage(report, "model_and_optimizer_checkpoint")
    with tempfile.TemporaryDirectory(prefix="deepfin-gpu-smoke-") as temporary:
        checkpoint = Path(temporary) / "state.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        restored = build_model(config).cuda().eval()
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, fused=True)
        restored_optimizer.load_state_dict(saved["optimizer"])
        original_state = optimizer.state_dict()
        restored_state = restored_optimizer.state_dict()
        _check(original_state["param_groups"] == restored_state["param_groups"], "Optimizer parameter groups changed")
        _check(original_state["state"].keys() == restored_state["state"].keys(), "Optimizer state keys changed")
        for parameter, values in original_state["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(restored_state["state"][parameter][key], value)
        with torch.inference_mode():
            eager = model(x)
            for name, value in restored(x).items():
                torch.testing.assert_close(value, eager[name])

    _stage(report, "compiled_vs_eager")
    with torch.inference_mode():
        compiled = torch.compile(model, fullgraph=True)
        result = compiled(x)
        torch.cuda.synchronize()
        for name, value in result.items():
            torch.testing.assert_close(value, eager[name], rtol=1e-3, atol=1e-4)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2


def _ray(report: dict[str, Any]) -> None:
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import ray
    from ray import tune

    report["ray"] = ray.__version__
    _check(not ray.is_initialized(), "Run the Ray smoke in a fresh process; an existing Ray connection is active")

    class Counter(tune.Trainable):
        count: int = 0
        restored: bool = False

        def setup(self, config: dict[str, Any]) -> None:
            del config
            self.count = 0
            self.restored = False

        def step(self) -> dict[str, Any]:
            self.count += 1
            return {"count": self.count, "restored": self.restored}

        def save_checkpoint(self, checkpoint_dir: str) -> None:
            Path(checkpoint_dir, "counter.json").write_text(json.dumps({"count": self.count}))

        def load_checkpoint(self, checkpoint: Any) -> None:
            if not isinstance(checkpoint, str):
                raise TypeError("Expected a directory checkpoint")
            self.count = json.loads(Path(checkpoint, "counter.json").read_text())["count"]
            self.restored = True

    with tempfile.TemporaryDirectory(prefix="deepfin-ray-smoke-") as temporary:
        root = Path(temporary)
        _stage(report, "ray_local_cluster")
        try:
            ray.init(
                address="local", num_cpus=1, num_gpus=0, include_dashboard=False,
                object_store_memory=100 * 1024**2, _temp_dir=str(root / "ray"), log_to_driver=False,
            )
            _stage(report, "ray_checkpoint")
            first = tune.Tuner(Counter, run_config=tune.RunConfig(
                name="first", storage_path=str(root / "results"), stop={"training_iteration": 1},
                checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=1), verbose=0,
            )).fit()
            _check(not first.errors, f"Ray trial errors: {first.errors}")
            checkpoint = first[0].checkpoint
            if checkpoint is None:
                raise RuntimeError("Ray trial did not save a checkpoint")
            _stage(report, "ray_tuner_restore")
            restored_tuner = tune.Tuner.restore(str(root / "results" / "first"), trainable=Counter)
            metrics = restored_tuner.get_results()[0].metrics
            _check(metrics is not None and metrics["count"] == 1, "Persisted tuner result changed")
            _stage(report, "ray_checkpoint_continuation")
            second = tune.run(
                Counter, name="continued", storage_path=str(root / "results"), stop={"count": 2},
                checkpoint_freq=1, restore=checkpoint.path, resources_per_trial={"cpu": 1}, verbose=0,
            )
            result = second.trials[0]
            _check(result.status == "TERMINATED", f"Continuation status: {result.status}")
            _check(result.last_result["count"] == 2, f"Continuation result: {result.last_result}")
            _check(result.last_result["restored"] is True, "Continuation did not load the checkpoint")
        finally:
            ray.shutdown()


def _memory_fraction(value: str) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("memory fraction must be greater than 0 and at most 1")
    return fraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gpu", "ray"), help="Explicitly select the runtime to exercise")
    parser.add_argument("--report", type=Path, help="Write the final JSON result, including failures")
    parser.add_argument("--memory-fraction", type=_memory_fraction, default=0.08, help="GPU allocator fraction (default: 0.08); does not reserve memory")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "mode": args.mode, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "pid": os.getpid(), "stage": "imports", "ok": False,
    }
    started = time.monotonic()
    try:
        if args.mode == "gpu":
            _gpu(report, args.memory_fraction)
        else:
            _ray(report)
        report.update(ok=True, stage="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        payload = json.dumps(report, indent=2) + "\n"
        print(payload, end="", flush=True)
        if args.report is not None:
            args.report.write_text(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
