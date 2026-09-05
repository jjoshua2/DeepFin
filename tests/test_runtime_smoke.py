"""Test diagnostic reporting without importing or starting either runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import runtime_smoke as smoke


def test_failure_preserves_stage_and_emits_json_and_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    def fail(report: dict[str, Any]) -> None:
        smoke._stage(report, "ray_checkpoint_continuation")
        raise RuntimeError("checkpoint did not advance")

    monkeypatch.setattr(smoke, "_ray", fail)
    path = tmp_path / "result.json"
    assert smoke.main(["ray", "--report", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["ok"] is False
    assert report["stage"] == "ray_checkpoint_continuation"
    assert report["error"] == "RuntimeError: checkpoint did not advance"
    assert report["elapsed_s"] >= 0
    assert report["python"]
    assert report["executable"]
    output = capsys.readouterr()
    assert '"ok": false' in output.out
    assert "Traceback" in output.err
    assert "checkpoint did not advance" in output.err


def test_gpu_budget_reaches_only_selected_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fractions: list[float] = []

    def gpu(_report: dict[str, Any], fraction: float) -> None:
        fractions.append(fraction)

    def ray(_report: dict[str, Any]) -> None:
        pytest.fail("GPU mode started Ray")

    monkeypatch.setattr(smoke, "_gpu", gpu)
    monkeypatch.setattr(smoke, "_ray", ray)
    path = tmp_path / "result.json"
    assert smoke.main(["gpu", "--memory-fraction", "0.04", "--report", str(path)]) == 0
    assert fractions == [0.04]
    report = json.loads(path.read_text())
    assert report["ok"] is True
    assert report["stage"] == "complete"


@pytest.mark.parametrize("arguments", [[], ["gpu", "--memory-fraction", "0"], ["gpu", "--memory-fraction", "1.01"], ["gpu", "--memory-fraction", "nan"]])
def test_invalid_invocation_never_starts_a_runtime(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str],
) -> None:
    def unexpected(*_args: Any) -> None:
        pytest.fail("Invalid invocation started a runtime")

    monkeypatch.setattr(smoke, "_gpu", unexpected)
    monkeypatch.setattr(smoke, "_ray", unexpected)
    with pytest.raises(SystemExit) as exc:
        smoke.main(arguments)
    assert exc.value.code == 2
