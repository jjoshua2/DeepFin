from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import bt4_joint_readout as tool


def bank(
    root: Path, arm: str, sims: int, *, candidate: str | None = None,
    opening_suffix: str = "", hoist: str = "4096",
) -> dict:
    reference = root / "S0.pt"
    settings = {
        "candidate": str(root / (candidate or f"{arm}.pt")), "reference": str(reference),
        "games": 1000, "mode": "matched_sims", "sims_candidate": sims,
        "sims_reference": sims, "seed": 42, "opening_plies": 16,
        "openings_kind": "book", "openings": "frozen-book.zip", "max_plies": 300,
        "temperature": 0.1, "gumbel_add_noise": True,
        "search_candidate": {"shape": "training"},
        "search_reference": {"shape": "training"},
    }
    records: list[dict[str, Any]] = [{"kind": "header", "driver": "arena_standard", "version": 1,
                "settings": settings, "fingerprint": tool.settings_fingerprint(settings)}]
    for pair in range(tool.PAIRS):
        total = (pair % 4) * 0.5  # Mean score .375: both arms lose, yet comparison is required.
        for half, score in enumerate((min(total, 1.0), max(0.0, total - 1.0))):
            white_score = score if half == 0 else 1 - score
            records.append({
                "kind": "game", "pair_id": pair, "half": half,
                "a_is_white": half == 0, "opening_index": pair,
                "opening_fen": f"opening{pair}{opening_suffix}",
                "start_fen": f"opening{pair}{opening_suffix}",
                "result": {0.0: "0-1", 0.5: "1/2-1/2", 1.0: "1-0"}[white_score],
                "score_candidate": score, "seed": 42, "loop": "chunked",
                "compile": "on", "eval_hoist": hoist,
            })
    path = root / f"{arm}.{sims}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return tool.read_arm(path, reference=reference, seed=42, sims=sims)


def test_identical_correlated_curves_have_zero_interaction_interval(tmp_path: Path) -> None:
    cells = {(arm, sims): bank(tmp_path, arm, sims)
             for arm in tool.ARMS for sims in tool.BUDGETS}
    report = tool.build_report(cells)
    assert report["screen_complete"]
    assert len(report["cells"]) == 9
    for result in report["cells"].values():
        assert result["strength_interpretation"] == "LOSS"
    for contrast in [*report["search_interactions_25_to_400"].values(),
                     *report["global_arm_comparisons_by_budget"].values()]:
        assert contrast["status"] == "READ"
        assert contrast["score_advantage"] == 0.0
        assert contrast["ci95"] == [0.0, 0.0]
        assert contrast["pair_outcome_covariance"] > 0
    assert report["promotion"].startswith("NONE")


@pytest.mark.parametrize("mismatch", ["opening", "candidate", "execution", "shared_candidate"])
def test_cross_cell_identity_mismatches_are_refused(tmp_path: Path, mismatch: str) -> None:
    left = bank(tmp_path, "G20T1", 25)
    arm = "G20T05" if mismatch == "shared_candidate" else "G20T1"
    right = bank(tmp_path, arm, 400,
        candidate="changed.pt" if mismatch == "candidate" else
                  "G20T1.pt" if mismatch == "shared_candidate" else None,
        opening_suffix="changed" if mismatch == "opening" else "",
        hoist="2048" if mismatch == "execution" else "4096")
    reason = {"opening": "same opening sequence", "candidate": "differs across budgets",
              "execution": "execution differs", "shared_candidate": "same candidate"}[mismatch]
    with pytest.raises(ValueError, match=reason):
        tool.build_report({("G20T1", 25): left, (arm, 400): right})


def test_missing_cells_remain_unread(tmp_path: Path) -> None:
    report = tool.build_report({("G20T1", 25): bank(tmp_path, "G20T1", 25)})
    assert not report["screen_complete"]
    assert len(report["missing_cells"]) == 8
    assert report["cells"]["G20T05:100"] == {"status": "UNREAD"}
    assert report["search_interactions_25_to_400"]["G20T1"] == {
        "status": "UNREAD", "missing_cells": ["G20T1:400"],
    }
    assert all(result["status"] == "UNREAD"
               for result in report["global_arm_comparisons_by_budget"].values())
    assert len(tool.build_report({})["missing_cells"]) == 9


def test_paired_interaction_retains_a_constant_advantage() -> None:
    baseline = np.tile([0.0, 0.25, 0.5, 0.75], 125)
    result = tool.compare(baseline + 0.25, baseline, seed=20260903, samples=10000)
    assert result["score_advantage"] == 0.25
    assert result["ci95"] == [0.25, 0.25]
