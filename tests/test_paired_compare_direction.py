"""Direction is part of the metric contract, not the bootstrap arithmetic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.paired_compare import load_dump, report


def _dump(path: Path, field: str, value: bool | float, *, n: int = 4) -> str:
    path.write_text("".join(json.dumps({
        "key": f"position_{i}", "phase": 1, "game_id": i // 2,
        "cand": {"sf_soft": {field: value}},
    }) + "\n" for i in range(n)), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(("field", "a_value", "b_value", "winner", "fraction_a"), [
    ("top1_agree", True, False, "A better", "100.0%"),
    ("top1_agree", False, True, "B better", "0.0%"),
    ("top1_agree", True, True, "NOT significant", "0.0%"),
    ("out_of_top10", False, True, "A better", "100.0%"),
    ("out_of_top10", True, False, "B better", "0.0%"),
    ("out_of_top10", False, False, "NOT significant", "0.0%"),
    ("top1", 2.0, 5.0, "A better", "100.0%"),
    ("top1", 5.0, 2.0, "B better", "0.0%"),
    ("top1", 2.0, 2.0, "NOT significant", "0.0%"),
])
@pytest.mark.parametrize("clustered", [False, True])
def test_cli_producer_metric_direction(
    tmp_path: Path, field: str, a_value: bool | float, b_value: bool | float,
    winner: str, fraction_a: str, clustered: bool,
) -> None:
    a = _dump(tmp_path / "a.jsonl", field, a_value)
    b = _dump(tmp_path / "b.jsonl", field, b_value)
    command = [sys.executable, "-m", "scripts.paired_compare", a, b,
               "--join-key", "key", "--field", f"cand.sf_soft.{field}",
               "--label-a", "A", "--label-b", "B", "--n-boot", "101"]
    if clustered:
        command.extend(["--cluster-key", "game_id"])
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stderr
    delta = float(a_value) - float(b_value)
    assert f"paired delta (A-B): {delta:+.2f}" in result.stdout
    assert f"CI {delta:+.2f} .. {delta:+.2f}]" in result.stdout
    assert f"verdict at 95%: {winner}" in result.stdout
    assert f"(A better {fraction_a} /" in result.stdout


@pytest.mark.parametrize(("a_value", "b_value", "winner"), [
    (True, False, "A better"), (False, True, "B better"),
    (True, True, "NOT significant"),
])
def test_direction_reaches_direct_report_and_phase_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    a_value: bool, b_value: bool, winner: str,
) -> None:
    a = load_dump(_dump(tmp_path / "a.jsonl", "top1_agree", a_value, n=40),
                  join_key="key", field="cand.sf_soft.top1_agree")
    b = load_dump(_dump(tmp_path / "b.jsonl", "top1_agree", b_value, n=40),
                  join_key="key", field="cand.sf_soft.top1_agree")
    report(a, b, label_a="A", label_b="B", n_boot=101)
    out = capsys.readouterr().out
    delta = float(a_value) - float(b_value)
    assert f"verdict at 95%: {winner}" in out
    assert f"delta {delta:+.2f} [{delta:+.2f} .. {delta:+.2f}]" in out
    assert "middlegame" in out


@pytest.mark.parametrize(("field", "value", "direction"), [
    ("top1_agree", True, "higher"), ("top1_agree", 1.0, "higher"),
    ("out_of_top10", False, "lower"), ("out_of_top10", 0.0, "lower"),
    ("top1", 5.0, "lower"), ("custom", 1.0, "lower"),
    ("custom", True, None),
])
def test_load_records_preference_without_refusing_unknown_boolean_data(
    tmp_path: Path, field: str, value: bool | float, direction: str | None,
) -> None:
    dump = load_dump(_dump(tmp_path / "a.jsonl", field, value),
                     join_key="key", field=f"cand.sf_soft.{field}")
    assert dump.metric_direction == direction
    assert dump.rows["position_0"][0] == float(value)


@pytest.mark.parametrize("field", ["top1_agree", "out_of_top10", "custom"])
@pytest.mark.parametrize("direction", ["lower", "higher"])
def test_explicit_preference_overrides_known_defaults_and_resolves_unknown_bool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, direction: str,
) -> None:

    a = load_dump(_dump(tmp_path / "a.jsonl", field, True), join_key="key",
                  field=f"cand.sf_soft.{field}", metric_direction=direction)
    b = load_dump(_dump(tmp_path / "b.jsonl", field, False), join_key="key",
                  field=f"cand.sf_soft.{field}", metric_direction=direction)
    report(a, b, label_a="A", label_b="B", n_boot=101)
    out = capsys.readouterr().out
    assert f"metric direction: {direction} is better" in out
    assert "not playing strength" in out
    assert "paired delta (A-B): +1.00" in out
    assert f"verdict at 95%: {'A' if direction == 'higher' else 'B'} better" in out


def test_unknown_boolean_refuses_only_directional_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    a = load_dump(_dump(tmp_path / "a.jsonl", "custom", True),
                  join_key="key", field="cand.sf_soft.custom")
    with pytest.raises(SystemExit, match="--metric-direction"):
        report(a, a, label_a="A", label_b="B", n_boot=101)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("direction", [None, "higher", "lower"])
def test_cli_unknown_boolean_requires_declaration(tmp_path: Path, direction: str | None) -> None:
    a = _dump(tmp_path / "a.jsonl", "custom", True)
    b = _dump(tmp_path / "b.jsonl", "custom", False)
    command = [sys.executable, "-m", "scripts.paired_compare", a, b,
               "--join-key", "key", "--field", "cand.sf_soft.custom", "--n-boot", "101"]
    if direction is not None:
        command.extend(["--metric-direction", direction])
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    if direction is None:
        assert result.returncode != 0
        assert "--metric-direction" in result.stderr
        assert "verdict" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert f"verdict at 95%: {'A' if direction == 'higher' else 'B'} better" in result.stdout


def test_report_refuses_different_preferences_before_inference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = _dump(tmp_path / "a.jsonl", "top1_agree", True)
    a = load_dump(path, join_key="key", field="cand.sf_soft.top1_agree")
    b = load_dump(path, join_key="key", field="cand.sf_soft.top1_agree", metric_direction="lower")
    with pytest.raises(SystemExit, match="direction disagrees"):
        report(a, b, label_a="A", label_b="B", n_boot=101)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("clustered", [False, True])
def test_preference_does_not_change_raw_statistics_rng_or_phase_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clustered: bool,
) -> None:

    # Non-degenerate draws catch sign transformations and changed RNG calls.
    values = [3.0, -2.0, 0.0, 4.0] * 10
    a_path = tmp_path / "a.jsonl"
    a_path.write_text("".join(json.dumps({
        "key": f"position_{i}", "phase": 1, "game_id": i // 2,
        "cand": {"sf_soft": {"custom": value}},
    }) + "\n" for i, value in enumerate(values)), encoding="utf-8")
    b_path = _dump(tmp_path / "b.jsonl", "custom", 0.0, n=40)
    outputs = []
    directions: tuple[str, ...] = ("lower", "higher", "lower")
    for direction in directions:
        cluster_key = "game_id" if clustered else None
        a = load_dump(str(a_path), join_key="key", field="cand.sf_soft.custom",
                      metric_direction=direction, cluster_key=cluster_key)
        b = load_dump(b_path, join_key="key", field="cand.sf_soft.custom",
                      metric_direction=direction, cluster_key=cluster_key)
        report(a, b, label_a="A", label_b="B", n_boot=137, cluster_key=cluster_key)
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[2]
    for prefix in ("paired delta", "  middlegame", "A =", "B ="):
        assert [line for line in outputs[0].splitlines() if line.startswith(prefix)] == [
            line for line in outputs[1].splitlines() if line.startswith(prefix)]
    assert "A better 25.0% / B better 50.0% / tied 25.0%" in outputs[0]
    assert "A better 50.0% / B better 25.0% / tied 25.0%" in outputs[1]


@pytest.mark.parametrize("direction", ["lower", "higher"])
def test_boolean_mcnemar_still_refused_even_with_explicit_preference(
    tmp_path: Path, direction: str,
) -> None:

    with pytest.raises(SystemExit, match="BOOLEAN"):
        load_dump(_dump(tmp_path / "a.jsonl", "top1_agree", True),
                  join_key="key", field="cand.sf_soft.top1_agree", mcnemar_at=0.0,
                  metric_direction=direction)


@pytest.mark.parametrize(("a_value", "b_value", "favoured"), [
    (2.0, 0.0, "A"), (0.0, 2.0, "B"),
])
def test_higher_numeric_mcnemar_preserves_predicate_and_reverses_preference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    a_value: float, b_value: float, favoured: str,
) -> None:
    a = load_dump(_dump(tmp_path / "a.jsonl", "custom", a_value, n=40),
                  join_key="key", field="cand.sf_soft.custom", metric_direction="higher")
    b = load_dump(_dump(tmp_path / "b.jsonl", "custom", b_value, n=40),
                  join_key="key", field="cand.sf_soft.custom", metric_direction="higher")
    report(a, b, label_a="A", label_b="B", n_boot=101, mcnemar_at=1.0)
    out = capsys.readouterr().out
    assert f"verdict at 95%: {favoured} better" in out
    assert "McNemar on (metric <= 1)" in out
    assert f"point estimate favours {favoured}" in out
    assert "lower = A better — OPPOSITE of the metric delta above" in out
    assert "fewer <= threshold outcomes are preferred" in out
    assert f"paired rate diff (A-B) {'-' if favoured == 'A' else '+'}1.00000" in out


def test_mcnemar_complement_preserves_test_not_preference_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import numpy as np
    from scripts.paired_compare import mcnemar, report_mcnemar

    # Include exact-threshold equality. The complement of <= is >, not >=.
    va = np.array([2.0] * 20 + [1.0] * 4 + [0.0] * 6)
    vb = np.array([1.0] * 20 + [2.0] * 4 + [0.0] * 6)
    table = mcnemar(va, vb, at=1.0)
    complement = mcnemar((va <= 1.0).astype(float), (vb <= 1.0).astype(float), at=0.0)
    assert (table.b, table.c, table.both, table.neither) == (
        complement.c, complement.b, complement.neither, complement.both)
    assert table.p_two_sided == complement.p_two_sided
    assert table.discordance == complement.discordance
    assert table.half_width == complement.half_width
    report_mcnemar(table, at=1.0, label_a="A", label_b="B", metric_direction="higher")
    higher = capsys.readouterr().out
    report_mcnemar(table, at=1.0, label_a="A", label_b="B")
    lower = capsys.readouterr().out
    assert "point estimate favours A" in higher
    assert "point estimate favours B" in lower
    for prefix in ("McNemar", "  both", "  discordant", "  null-case"):
        assert [line for line in higher.splitlines() if line.startswith(prefix)] == [
            line for line in lower.splitlines() if line.startswith(prefix)]
    assert higher.split("paired rate diff")[1].split("   (")[0] == lower.split(
        "paired rate diff")[1].split("   (")[0]


def test_boolean_suffix_outside_producer_schema_is_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_text('\n'.join(json.dumps({"fen": str(i), "top1_agree": True})
                              for i in range(2)), encoding="utf-8")
    dump = load_dump(str(path), field="top1_agree")
    assert dump.metric_direction is None
    with pytest.raises(SystemExit, match="--metric-direction"):
        report(dump, dump, label_a="A", label_b="B", n_boot=101)


def test_invalid_direct_preference_is_rejected_before_io() -> None:
    with pytest.raises(ValueError, match="metric_direction"):
        load_dump("not-opened.jsonl", metric_direction="guess")


def test_higher_mcnemar_retains_tie_and_void(capsys: pytest.CaptureFixture[str]) -> None:
    import numpy as np
    from scripts.paired_compare import mcnemar, report_mcnemar

    a = np.array([0.0, 2.0, 1.0, 1.0])
    b = np.array([2.0, 0.0, 1.0, 1.0])
    report_mcnemar(mcnemar(a, b, at=1.0), at=1.0, label_a="A", label_b="B",
                   metric_direction="higher")
    assert "exactly tied" in capsys.readouterr().out
    report_mcnemar(mcnemar(a, a, at=1.0), at=1.0, label_a="A", label_b="B",
                   metric_direction="higher")
    out = capsys.readouterr().out
    assert "VOID" in out
    assert "favours" not in out


def test_numeric_mean_and_threshold_preference_can_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Higher mean and more > threshold outcomes are different estimands.
    path = tmp_path / "a.jsonl"
    path.write_text("".join(json.dumps({"fen": str(i), "value": value}) + "\n"
                            for i, value in enumerate([100.0, 0.0, 0.0, 0.0] * 10)),
                    encoding="utf-8")
    a = load_dump(str(path), metric_direction="higher")
    path.write_text("".join(json.dumps({"fen": str(i), "value": 1.0}) + "\n"
                            for i in range(40)), encoding="utf-8")
    b = load_dump(str(path), metric_direction="higher")
    report(a, b, label_a="A", label_b="B", n_boot=137, mcnemar_at=0.5)
    out = capsys.readouterr().out
    assert "verdict at 95%: A better" in out
    assert "point estimate favours B" in out
