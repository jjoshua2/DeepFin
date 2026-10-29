"""A singleton join cannot support an inferential bootstrap verdict."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.paired_compare import (
    load_dump,
    paired_bootstrap_ci,
    paired_cluster_bootstrap_ci,
    report,
)


def _dump(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("values", [[], [-2.0], [0.0], [2.0]])
def test_paired_bootstrap_refuses_fewer_than_two_positions(values: list[float]) -> None:
    with pytest.raises(ValueError, match="at least two paired positions"):
        paired_bootstrap_ci(np.array(values), n_boot=101)


@pytest.mark.parametrize("delta", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("require_n", [None, 1])
def test_report_refuses_singleton_after_join_and_filtering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], delta: float,
    require_n: int | None,
) -> None:
    a = load_dump(_dump(tmp_path / "a.jsonl", [
        {"fen": "paired", "value": 5.0 + delta},
        {"fen": "only-a", "value": 8.0},
        {"fen": "unusable", "value": None},
    ]))
    b = load_dump(_dump(tmp_path / "b.jsonl", [
        {"fen": "paired", "value": 5.0},
        {"fen": "only-b", "value": 9.0},
        {"fen": "unusable", "value": 4.0},
    ]))
    with pytest.raises(SystemExit, match="only 1 paired position"):
        report(a, b, label_a="A", label_b="B", n_boot=101, require_n=require_n)
    out = capsys.readouterr().out
    assert "95% CI" not in out
    assert "verdict at" not in out


@pytest.mark.parametrize("delta", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("mcnemar", [False, True])
def test_cli_refuses_singleton_without_printing_a_verdict(
    tmp_path: Path, delta: float, mcnemar: bool,
) -> None:
    a = _dump(tmp_path / "a.jsonl", [{"fen": "p", "value": 5.0 + delta}])
    b = _dump(tmp_path / "b.jsonl", [{"fen": "p", "value": 5.0}])
    cmd = [sys.executable, "-m", "scripts.paired_compare", a, b, "--n-boot", "101"]
    if mcnemar:
        cmd += ["--mcnemar-at", "5"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 1
    assert "only 1 paired position" in result.stderr
    assert "sampling uncertainty" in result.stderr
    assert "Traceback" not in result.stderr
    assert "95% CI" not in result.stdout
    assert "verdict at" not in result.stdout


def test_require_n_mismatch_still_precedes_singleton_inference(tmp_path: Path) -> None:
    path = _dump(tmp_path / "a.jsonl", [{"fen": "p", "value": 5.0}])
    dump = load_dump(path)
    with pytest.raises(SystemExit, match="--require-n 2 but the join produced 1"):
        report(dump, dump, label_a="A", label_b="B", n_boot=101, require_n=2)


@pytest.mark.parametrize(("seed", "alpha", "expected"), [
    (0, 0.05, (-4.5875, 7.0)),
    (0, 0.2, (-1.4, 5.650000000000002)),
    (5, 0.05, (-2.25, 6.087499999999997)),
    (5, 0.2, (-1.0, 5.0)),
    (19, 0.05, (-4.25, 7.75)),
    (19, 0.2, (-1.4, 5.75)),
])
def test_valid_sample_retains_seeded_interval(
    seed: int, alpha: float, expected: tuple[float, float],
) -> None:
    # Recorded on the unchanged implementation; pin draws and quantile arithmetic.
    assert paired_bootstrap_ci(
        np.array([-7.0, 1.0, 4.0, 9.0]), n_boot=63, seed=seed, alpha=alpha,
    ) == expected


@pytest.mark.parametrize("delta", [-2.0, 0.0, 2.0])
def test_multiple_constant_deltas_keep_the_empirical_bootstrap_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], delta: float,
) -> None:
    # Two observed units with zero empirical variance are not a missing second unit.
    assert paired_bootstrap_ci(np.array([delta, delta]), n_boot=101) == (delta, delta)
    a = load_dump(_dump(tmp_path / "a.jsonl", [
        {"fen": f"p{i}", "value": 5.0 + delta} for i in range(2)
    ]))
    b = load_dump(_dump(tmp_path / "b.jsonl", [
        {"fen": f"p{i}", "value": 5.0} for i in range(2)
    ]))
    report(a, b, label_a="A", label_b="B", n_boot=101, require_n=2)
    out = capsys.readouterr().out
    verdict = "A better" if delta < 0 else "B better" if delta > 0 else "NOT significant"
    assert "paired positions: 2\n" in out
    assert f"[95% CI {delta:+.2f} .. {delta:+.2f}]" in out
    assert f"verdict at 95%: {verdict}" in out


def test_cluster_unit_policy_is_not_changed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows: list[dict[str, object]] = [
        {"fen": f"p{i}", "value": float(i), "game_id": i // 2} for i in range(4)
    ]
    path = _dump(tmp_path / "clusters.jsonl", rows)
    dump = load_dump(path, cluster_key="game_id")
    report(dump, dump, label_a="A", label_b="B", n_boot=101, cluster_key="game_id")
    assert "paired clusters (game_id): 2" in capsys.readouterr().out
    with pytest.raises(ValueError, match="at least two clusters"):
        paired_cluster_bootstrap_ci(np.array([1.0, 2.0]), np.array(["only", "only"]))


@pytest.mark.parametrize("cluster_count", [1, 2])
def test_cli_keeps_existing_cluster_count_boundary(tmp_path: Path, cluster_count: int) -> None:
    path = _dump(tmp_path / "clusters.jsonl", [
        {"fen": f"p{i}", "value": float(i), "game_id": i % cluster_count}
        for i in range(4)
    ])
    result = subprocess.run(
        [sys.executable, "-m", "scripts.paired_compare", path, path,
         "--cluster-key", "game_id", "--n-boot", "101"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    if cluster_count == 1:
        assert result.returncode == 1
        assert "one cluster" in result.stderr
        assert "verdict at" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "paired clusters (game_id): 2" in result.stdout
        assert "verdict at 95%: NOT significant" in result.stdout
