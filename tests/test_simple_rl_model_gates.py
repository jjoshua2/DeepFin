"""scripts/simple_rl_model_gates.py on a tiny CPU run (final model and a snapshot)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import simple_rl_loop
from scripts import simple_rl_model_gates as mg
from tests.test_simple_rl_resume import _setup
from tests.test_simple_rl_snapshots_arms import _argv


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    assert simple_rl_loop.main(_argv(run_dir, openings, model_config, "--snapshot-updates", "1")) == 0
    return run_dir


def _report(tmp_path: Path, *args: str) -> tuple[int, dict]:
    out = tmp_path / "r.json"
    code = mg.main([*args, "--out", str(out)])
    return code, json.loads(out.read_text())


def test_heldout_ce_at_t1_is_the_train_stage_ce(run: Path, tmp_path: Path) -> None:
    model = run / "gen_0001" / "train" / "model.pt"
    summary = json.loads((run / "gen_0001" / "train" / "metrics.json").read_text())["summary"]
    _, r = _report(tmp_path, "--model", str(model), "--collect", str(run), "--gen", "1")
    h = r["heldout"]
    assert h["holdout_frac"] == 0.5
    assert h["rows"] == summary["heldout_after"]["rows"] > 0
    for k in ("policy_ce", "policy_entropy", "value_nll"):
        assert h[k] == pytest.approx(summary["heldout_after"][k], rel=1e-9, abs=1e-12), k
    # Re-tempered target: the CE moves, the model-only entropy does not.
    _, r2 = _report(tmp_path, "--model", str(model), "--collect", str(run), "--gen", "1", "--train-target-temp", "2.0")
    assert r2["heldout"]["policy_ce"] != pytest.approx(h["policy_ce"], rel=1e-6)
    assert r2["heldout"]["policy_entropy"] == pytest.approx(h["policy_entropy"])
    # A snapshot's held-out numbers equal what the train stage measured on it.
    snap = run / "gen_0001" / "train" / "snapshots" / "step_1"
    _, r3 = _report(tmp_path, "--model", str(snap / "model.pt"), "--collect", str(run), "--gen", "1")
    want = json.loads((snap / "metrics.json").read_text())["heldout"]
    assert r3["heldout"]["policy_ce"] == pytest.approx(want["policy_ce"], rel=1e-9)


def test_audit_confidence_and_thresholds(run: Path, tmp_path: Path) -> None:
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    with np.load(run / "probe.npz") as z:
        x = z["x"]
    audit = tmp_path / "audit.npz"
    np.savez(audit, x=x)
    model = run / "gen_0001" / "train" / "model.pt"
    code, r = _report(tmp_path, "--model", str(model), "--audit-rows", str(audit))
    assert code == 0
    with torch.no_grad():
        p = torch.softmax(load_model_from_checkpoint(model)(torch.from_numpy(x).float())["wdl"].float(), -1).double()
    assert r["audit"]["rows"] == len(x)
    assert r["audit"]["mean_abs_q"] == pytest.approx(float((p[:, 0] - p[:, 2]).abs().mean()))
    assert r["audit"]["mean_p_draw"] == pytest.approx(float(p[:, 1].mean()))
    th = tmp_path / "t.json"
    th.write_text(json.dumps({"audit.mean_abs_q": {"min": 0.0, "max": 1.0}, "audit.mean_p_draw": {"min": 0.0}}))
    assert _report(tmp_path, "--model", str(model), "--audit-rows", str(audit), "--thresholds", str(th))[0] == 0
    th.write_text(json.dumps({"audit.mean_abs_q": {"min": 2.0}}))
    assert _report(tmp_path, "--model", str(model), "--audit-rows", str(audit), "--thresholds", str(th))[0] == 3
    th.write_text(json.dumps({"heldout.policy_ce": {"max": 99.0}}))  # not measured: missing fails
    assert _report(tmp_path, "--model", str(model), "--audit-rows", str(audit), "--thresholds", str(th))[0] == 3
