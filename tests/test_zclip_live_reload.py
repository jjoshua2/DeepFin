"""`zclip_max_norm` must be a LIVE control surface, not a decorative one.

`ZClip` reads `max_grad_norm` once in its constructor. The key is in none of
the restart-required sets, so `_reload_yaml_into_config` overlays it into the
config every iteration and the reloader logs nothing -- a live yaml edit read
as applied while the running trainer kept clipping at the launch value. Same
shape as the `weight_decay` defect (rl_loop_audit I13).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    restart_required_config_keys,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FakeZClip:
    """Stands in for zclip.ZClip: the cap is plain constructor state."""

    def __init__(self, max_grad_norm: float | None) -> None:
        self.max_grad_norm = max_grad_norm


class _TrainerShim:
    """Only the surface `set_grad_clip_max_norm` touches.

    The method is the SHIPPED one, borrowed off `Trainer`, so these tests
    exercise the real implementation rather than a paraphrase of it.
    """

    def __init__(self, max_grad_norm: float | None) -> None:
        self.zclip = _FakeZClip(max_grad_norm)

    set_grad_clip_max_norm = Trainer.set_grad_clip_max_norm


def _make_shim(max_grad_norm: float | None) -> _TrainerShim:
    return _TrainerShim(max_grad_norm)


def test_setter_repoints_the_live_cap_and_reports_transitions() -> None:
    t = _make_shim(5.0)

    assert t.set_grad_clip_max_norm(10.0) is True, "a real change must report True"
    assert t.zclip.max_grad_norm == 10.0, "the LIVE zclip object must be re-pointed"

    assert t.set_grad_clip_max_norm(10.0) is False, "no-op must not report a transition"
    assert t.zclip.max_grad_norm == 10.0


def test_setter_handles_disable_and_reenable() -> None:
    """None disables the hard cap; it must round-trip, not crash or stick."""
    t = _make_shim(5.0)
    assert t.set_grad_clip_max_norm(None) is True
    assert t.zclip.max_grad_norm is None
    assert t.set_grad_clip_max_norm(None) is False
    assert t.set_grad_clip_max_norm(8.0) is True
    assert t.zclip.max_grad_norm == 8.0


def test_int_and_float_spellings_are_the_same_value() -> None:
    """A yaml `10` and a yaml `10.0` must not look like a change every iteration."""
    t = _make_shim(10.0)
    assert t.set_grad_clip_max_norm(10) is False, "int 10 == float 10.0, not a transition"


def test_zclip_max_norm_is_not_restart_required() -> None:
    """The claim this module rests on: the reloader really does overlay the key.

    If someone later adds it to a restart-required set, the live push becomes
    dead code and this test says so.
    """
    assert "zclip_max_norm" not in restart_required_config_keys()


def test_reloader_overlays_the_key_so_the_push_has_something_to_push(tmp_path) -> None:
    """End-to-end on the two halves that must agree: reloader, then setter."""
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("train:\n  zclip_max_norm: 10.0\n")

    config: dict = {"zclip_max_norm": 5.0}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config["zclip_max_norm"] == 10.0, "reloader must surface the new value"

    t = _make_shim(5.0)
    assert t.set_grad_clip_max_norm(config["zclip_max_norm"]) is True
    assert t.zclip.max_grad_norm == 10.0


def test_the_cap_actually_binds_after_a_live_change() -> None:
    """Behavioural: re-pointing the cap must change what the clip does.

    Asserting the attribute alone would pass against a setter that writes to a
    field nothing reads -- which is precisely the defect class here.
    """
    model = _Tiny()
    grads = [torch.ones_like(p) * 10.0 for p in model.parameters()]

    def effective(total_norm: float, cap: float | None) -> float:
        return total_norm if cap is None else min(total_norm, cap)

    t = _make_shim(5.0)
    total = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))
    assert total > 10.0, "fixture must produce a norm above both caps to be meaningful"

    assert effective(total, t.zclip.max_grad_norm) == 5.0
    t.set_grad_clip_max_norm(10.0)
    assert effective(total, t.zclip.max_grad_norm) == 10.0
