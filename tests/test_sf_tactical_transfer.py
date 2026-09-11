"""High-confidence SF mass transfer on stored B100 policy targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts import sf_tactical_transfer as tool
from tests.test_sf_tactical_policy import core, fixture as old_fixture
from tests.test_sf_policy_rewrite import CONFIG_SHA


def test_strict_300cp_gate_keeps_exact_boundary_and_changes_above() -> None:
    obs, base = core([500.0, 200.0], [0.1, 0.9])
    exact, exact_diag = tool.tactical_transfer_target(obs, base)
    assert exact.tobytes() == base.tobytes()
    assert exact_diag["ordinary_gate"] is False
    assert exact_diag["ordinary_gap_cp"] == 300.0

    obs.scores[1] = 199.0
    changed, changed_diag = tool.tactical_transfer_target(obs, base)
    assert changed_diag["ordinary_gate"] is True
    assert changed_diag["ordinary_gap_cp"] == 301.0
    np.testing.assert_allclose(
        changed[:2].astype(np.float64),
        [0.55, 0.45],
        atol=5e-4,
    )


def test_transfer_can_restore_zero_mass_sf_winner() -> None:
    obs, base = core([500.0, 0.0, -100.0], [0.0, 0.5, 0.5])
    result, diagnostic = tool.tactical_transfer_target(obs, base)
    np.testing.assert_allclose(
        result[:3].astype(np.float64),
        [0.5, 0.25, 0.25],
        atol=5e-4,
    )
    assert diagnostic["ordinary_gate"] is True
    assert diagnostic["support_gains"] == 1
    assert diagnostic["sf_best_base_mass"] == 0.0
    assert diagnostic["sf_best_ideal_mass"] == pytest.approx(0.5)


def test_tied_sf_best_preserves_b100_odds_inside_best_set() -> None:
    obs, base = core([500.0, 500.0, 0.0], [0.1, 0.2, 0.7])
    result, diagnostic = tool.tactical_transfer_target(obs, base)
    legal = result[:3].astype(np.float64)
    assert diagnostic["ordinary_gate"] is True
    assert legal[0] / legal[1] == pytest.approx(0.5, rel=0.003)
    assert legal[2] == pytest.approx(0.35, abs=5e-4)
    assert legal[:2].sum() == pytest.approx(0.65, abs=8e-4)


def test_winning_mates_receive_strong_categorical_transfer() -> None:
    obs, base = core(
        [99900.0, 50000.0, 100.0, -99900.0],
        [0.01, 0.09, 0.4, 0.5],
    )
    result, diagnostic = tool.tactical_transfer_target(obs, base)
    legal = result[:4].astype(np.float64)
    assert diagnostic["category"] == "winning_mate_available"
    assert diagnostic["ordinary_gate"] is False
    assert diagnostic["transferred_mate_mass"] == pytest.approx(0.675)
    assert legal[:2].sum() == pytest.approx(0.775, abs=8e-4)
    assert legal[0] / legal[1] == pytest.approx(1 / 9, rel=0.01)
    assert legal[2:].sum() == pytest.approx(0.225, abs=8e-4)


def test_losing_mate_and_large_nonmate_gap_apply_disjoint_transfers() -> None:
    obs, base = core([500.0, 100.0, -99900.0], [0.1, 0.4, 0.5])
    result, diagnostic = tool.tactical_transfer_target(obs, base)
    np.testing.assert_allclose(
        result[:3].astype(np.float64),
        [0.675, 0.2, 0.125],
        atol=8e-4,
    )
    assert diagnostic["category"] == "losing_mate_alternatives"
    assert diagnostic["ordinary_gate"] is True
    assert diagnostic["transferred_mate_mass"] == pytest.approx(0.375)
    assert diagnostic["transferred_ordinary_mass"] == pytest.approx(0.2)


def test_all_forced_losses_remain_byte_identical() -> None:
    obs, base = core(
        [-50000.0, -80000.0, -99900.0, -100000.0],
        [0.4, 0.3, 0.2, 0.1],
    )
    result, diagnostic = tool.tactical_transfer_target(obs, base)
    assert result.tobytes() == base.tobytes()
    assert diagnostic["category"] == "all_forced_losses"
    assert diagnostic["transferred_total_mass"] == 0.0


@pytest.mark.parametrize(
    "score",
    [32001.0, -32001.0, 49999.0, 99900.1, 100001.0, float("nan")],
)
def test_ambiguous_score_domain_refused(score: float) -> None:
    obs, base = core([score, 0.0])
    with pytest.raises(ValueError, match=r"scores|domain"):
        tool.tactical_transfer_target(obs, base)


def _args_from_old_fixture(old: Any, out: Path) -> Any:
    return tool.build_parser().parse_args(
        [
            "--raw",
            str(old.raw),
            "--source",
            str(old.source),
            "--b100-source",
            str(old.tactical_bt4_source),
            "--out",
            str(out),
            "--expected-source-summary-sha256",
            str(old.expected_source_summary_sha256),
            "--expected-b100-summary-sha256",
            str(old.expected_bt4_summary_sha256),
            "--expected-b100-mix-sha256",
            str(old.expected_bt4_mix_sha256),
            "--minimum-free-gib",
            "0",
        ]
    )


def test_real_shuffled_transfer_rewrite_preserves_all_nonpolicy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, rows = old_fixture(tmp_path)
    out = tmp_path / "transfer"
    invocation = _args_from_old_fixture(old, out)

    def no_reencode(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("inherited history must not be re-encoded")

    monkeypatch.setattr(tool.base.derive.TargetDeriver, "_encode", no_reencode)
    result = tool.rewrite(invocation)
    assert result["kind"] == "bt4_sf_tactical_policy_mass_transfer"
    assert result["algorithm"] == tool.ALGORITHM
    assert result["recipe"]["gap_cp"] == 300.0
    assert result["recipe"]["ordinary_transfer_fraction"] == 0.5
    assert result["recipe"]["mate_transfer_fraction"] == 0.75
    assert (result["rows"], result["shards"], result["raw_limit"]) == (6, 2, 7)
    assert result["rows_dropped_no_result"] == 1
    assert result["changed_rows"] > 0
    assert result["stored_support_gains"] >= 0
    assert result["stored_mass_error_max"] <= 2**-10

    summary = json.loads((out / tool.base.derive.SUMMARY_NAME).read_text())
    compact = json.loads((out / tool.SUMMARY).read_text())
    assert compact["algorithm"] == tool.ALGORITHM
    assert summary["policy_target_postprocess"] == {
        key: value for key, value in compact.items() if key != "outputs"
    }

    for spec in result["outputs"]:
        parent = Path(invocation.b100_source) / spec["path"]
        original, _ = load_shard_arrays(parent)
        actual, metadata = load_shard_arrays(out / spec["path"])
        assert metadata["policy_target_rewrite"]["algorithm"] == tool.ALGORITHM
        assert not any(key.startswith("policy_target_mix_") for key in metadata)
        for key in tool.base.ARRAYS - {"policy_target"}:
            np.testing.assert_array_equal(actual[key], original[key])
            for path in (parent / key).iterdir():
                assert path.read_bytes() == (
                    out / spec["path"] / key / path.name
                ).read_bytes()

        for row_index, game in enumerate(actual["game_id"]):
            obs = tool.base.observation(rows[int(game)], CONFIG_SHA)
            expected, _ = tool.tactical_transfer_target(
                obs, original["policy_target"][row_index]
            )
            np.testing.assert_array_equal(actual["policy_target"][row_index], expected)


def test_missing_or_wrong_b100_provenance_refuses(tmp_path: Path) -> None:
    old, _ = old_fixture(tmp_path)
    invocation = _args_from_old_fixture(old, tmp_path / "transfer")
    invocation.expected_b100_mix_sha256 = "0" * 64
    with pytest.raises(ValueError, match="pin differs"):
        tool.rewrite(invocation)
    assert not Path(invocation.out).exists()


def test_output_is_fresh_and_partial_is_not_reused(tmp_path: Path) -> None:
    old, _ = old_fixture(tmp_path)
    invocation = _args_from_old_fixture(old, tmp_path / "transfer")
    partial = Path(str(invocation.out) + ".writing")
    partial.mkdir()
    with pytest.raises(ValueError, match="partial already exists"):
        tool.rewrite(invocation)
