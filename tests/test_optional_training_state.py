"""Optional training sidecars are atomic, best-effort startup inputs."""
import json
import math

import pytest

from chess_anti_engine.tune.promotion_gate import (
    GateConfig,
    GateHoldController,
    PromotionGate,
)
from chess_anti_engine.tune.trainable import _load_best_state, _startup_gate_state
from chess_anti_engine.tune.trial_config import RestoreResult


@pytest.mark.parametrize("key", ["best_loss", "opp_strength_ema"])
@pytest.mark.parametrize("value", [None, {}, [], "bad", 10**400])
def test_bad_best_scalar_uses_whole_default(tmp_path, capsys, key, value):
    path = tmp_path / "best.json"
    path.write_text(json.dumps({key: value, "source": "test_loss"}))
    assert _load_best_state(path) == (float("inf"), 0.0, "train_loss")
    assert "using defaults" in capsys.readouterr().out


@pytest.mark.parametrize("state", [
    {"matches": None}, {"matches": {}}, {"matches": []},
    {"matches": float("inf")}, {"holds": float("nan")},
    {"samples": [{"cur_w": float("inf")}]},
    {"samples": [{"cur_wdl_regret": 10**400}]},
    {"hold": {"anchor_refresh_failures": 4, "fallback_missing": float("inf")}},
])
def test_corrupt_own_gate_never_borrows_pool(tmp_path, capsys, state):
    path = tmp_path / "gate_state.json"
    path.write_text(json.dumps(state))
    restore = RestoreResult(restored_gate_state={"matches": 900, "holds": 3})
    assert _startup_gate_state(gate_state_path=path, restore=restore) == {}
    assert "EMPTY own window" in capsys.readouterr().out


def test_direct_gate_late_error_does_not_partially_install():
    gate = PromotionGate()
    gate.holds = 2
    gate.hold_active = False
    before = gate.state_dict()
    with pytest.raises((ValueError, OverflowError)):
        gate.load_state_dict({
            "holds": 8, "hold_active": True,
            "samples": [{"cur_w": 2}, {"prev_l": float("inf")}],
        })
    assert gate.state_dict() == before


def test_direct_hold_late_error_does_not_partially_install():
    hold = GateHoldController(gate=PromotionGate())
    hold.anchor_refresh_failures = 2
    hold.fallback_missing = 3
    with pytest.raises(OverflowError):
        hold.load_state_dict({"anchor_refresh_failures": 4, "fallback_missing": float("inf")})
    assert hold.state_dict() == {"anchor_refresh_failures": 2, "fallback_missing": 3}


def test_legacy_coercions_and_unknown_confound_roundtrip():
    gate = PromotionGate(cfg=GateConfig())
    gate.load_state_dict({
        "holds": True, "hold_active": "yes",
        "samples": [None, {"iteration": 2.9, "cur_w": True, "cur_l": "ignored"}],
    })
    assert gate.holds == 1
    assert gate.hold_active
    assert gate.samples[0].iteration == 2
    assert gate.samples[0].cur_w == 1
    assert gate.samples[0].cur_l == 0
    assert math.isnan(gate.samples[0].regret_fit_slope)
    encoded = json.loads(json.dumps(gate.state_dict(), allow_nan=False))
    restored = PromotionGate(cfg=gate.cfg)
    restored.load_state_dict(encoded)
    assert restored.state_dict() == gate.state_dict()
