from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_bench_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_checkpoint_inference_ab.py"
    spec = importlib.util.spec_from_file_location("bench_checkpoint_inference_ab_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_batches_uses_default_batch() -> None:
    module = _load_bench_module()

    assert module.parse_batches(680, "") == (680,)


def test_parse_batches_accepts_comma_separated_values() -> None:
    module = _load_bench_module()

    assert module.parse_batches(1, "256, 512,680") == (256, 512, 680)


def test_parse_batches_rejects_non_positive_values() -> None:
    module = _load_bench_module()

    with pytest.raises(ValueError, match="> 0"):
        module.parse_batches(680, "256,0")


def test_speedup_pct_reports_candidate_speedup() -> None:
    module = _load_bench_module()

    assert module.speedup_pct(100.0, 80.0) == pytest.approx(25.0)
    assert module.speedup_pct(100.0, 125.0) == pytest.approx(-20.0)


def test_timing_to_json_preserves_batch_metrics() -> None:
    module = _load_bench_module()
    timing = module.CheckpointTiming(
        label="candidate",
        checkpoint="ckpt.pt",
        params=123,
        batches=(
            module.BatchTiming(
                batch=680,
                wall_median_ms=10.0,
                wall_mean_ms=11.0,
                wall_p95_ms=12.0,
                gpu_median_ms=9.0,
                gpu_mean_ms=9.5,
                gpu_p95_ms=10.5,
                boards_per_s=68000.0,
            ),
        ),
    )

    payload = module.timing_to_json(timing)

    assert payload["label"] == "candidate"
    assert payload["params"] == 123
    assert payload["batches"][0]["batch"] == 680
    assert payload["batches"][0]["boards_per_s"] == 68000.0
