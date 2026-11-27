"""Synthetic descriptive-mode controls, not real-model qualification."""
from typing import Any

import numpy as np
import pytest

from scripts.stored_wdl_compare import LAYOUT, compare_descriptive, rows_digest, targets_digest


def fixture():
    x = np.zeros((4, 175, 8, 8), np.float32)
    x[:, 17, 0, 0] = [.1, .2, .4, .6]
    target = np.array([[.7, .2, .1], [.6, .3, .1], [.2, .5, .3], [.1, .2, .7]])
    def donor(a):
        w = a[:, 17, 0, 0].astype(float)
        return np.column_stack((w, np.full(len(a), .2), .8 - w))
    kw: dict[str, Any] = {"row_ids": ["r1", "r2", "r3", "r4"],
                             "group_ids": ["c1", "c1", "c2", "c2"],
                             "group_kind": "connected_component", "donor": donor, "candidate": donor,
                             "donor_sha256": "a" * 64, "candidate_sha256": "a" * 64,
                             "layout": LAYOUT, "target_description": "synthetic SF soft labels STM WDL",
                             "bootstraps": 20}
    bind(x, target, kw)
    return x, target, kw


def bind(x, target, kw):
    kw["expected_rows_sha256"] = rows_digest(x, kw["row_ids"], kw["group_ids"])
    kw["expected_targets_sha256"] = targets_digest(
        target, kw["row_ids"], kw["group_ids"], rows_sha256=kw["expected_rows_sha256"],
        target_description=kw["target_description"])


def test_donor_identity_and_honest_labels():
    x, target, kw = fixture()
    result = compare_descriptive(x, target, **kw)
    assert result["mode"] == "descriptive"
    assert result["training_exposure"] == "unknown"
    assert result["controls_performed"] == []
    assert "source_games" not in result
    assert "per_game" not in result
    assert all(v["delta"] == 0 for v in result["metrics"].values())


@pytest.mark.parametrize("fault", ["targets", "semantics", "rows", "cap", "group"])
def test_identity_failures_precede_callbacks(fault):
    x, target, kw = fixture()
    if fault == "targets":
        target = target[::-1]
    elif fault == "semantics":
        kw["target_description"] = "different labels"
    elif fault == "rows":
        x = x[::-1]
    elif fault == "cap":
        x = np.zeros((257, 175, 8, 8), np.float32)
    else:
        kw["group_kind"] = "unknown"
    def forbidden(_a):
        pytest.fail("callback reached after invalid identity")
    kw.update(donor=forbidden, candidate=forbidden)
    with pytest.raises(ValueError, match=r"identity|digest|256"):
        compare_descriptive(x, target, **kw)


def test_synthetic_batch_and_pov_controls():
    x, target, kw = fixture()
    donor = kw["donor"]
    kw["candidate"] = lambda a: np.concatenate([donor(a[i:i + 1]) for i in range(len(a))])
    baseline = compare_descriptive(x, target, **kw)
    assert all(v["delta"] == 0 for v in baseline["metrics"].values())
    # Consistently reverse W/L in targets and both predictions. This is a
    # mathematical control, not verification of any real encoder's board POV.
    kw.update(donor=lambda a: donor(a)[:, ::-1], candidate=lambda a: donor(a)[:, ::-1])
    target = target[:, ::-1]
    bind(x, target, kw)
    swapped = compare_descriptive(x, target, **kw)
    for key in baseline["metrics"]:
        assert swapped["metrics"][key]["donor"] == pytest.approx(baseline["metrics"][key]["donor"])
