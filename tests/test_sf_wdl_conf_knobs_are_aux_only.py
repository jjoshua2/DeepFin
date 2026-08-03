"""`sf_wdl_conf_power` / `sf_wdl_draw_scale` shape ONLY the aux `sf_eval` head.

Both knobs carry the `sf_wdl` prefix and the production yaml used to describe
them as "scale SF-WDL", which reads as the SF component of the WDL VALUE target
-- the load-bearing blend. They cannot touch it: the mask they build
(`_compute_sf_wdl_mask`) has exactly one consumer, `m_sf_eval`, while the blend
weights its SF component by `sf_available * keep` and `keep` carries only the
`sf_search_dampen_sf_*` terms.

This is the falsifier named in the 2026-08-03 train/data audit (C1) and in the
correction to the 2026-08-02 config audit: flip either knob and `wdl_ce` must
move if they are blend knobs. It is bit-identical, and `sf_eval_ce` moves --
which is what pins the classification. If someone later wires confidence
damping into the value blend (a training-affecting change that needs a ledger
entry), this test fails and says so.
"""

from __future__ import annotations

import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.losses import compute_loss


def _batch(b: int) -> dict[str, torch.Tensor]:
    policy = torch.ones((b, POLICY_SIZE))
    policy = policy / policy.sum(dim=-1, keepdim=True)
    # Mixed draw probabilities and mixed game results so BOTH knobs have rows
    # to act on: conf_power keys off sf_wdl[:, 1], draw_scale off wdl_t == 1.
    sf_wdl = torch.tensor([[0.2, 0.7, 0.1], [0.45, 0.1, 0.45]] * (b // 2))
    return {
        "x": torch.randn((b, 146, 8, 8)),
        "policy_t": policy,
        "wdl_t": torch.tensor([1, 0] * (b // 2)),
        "has_policy": torch.ones((b,)),
        "is_network_turn": torch.ones((b,)),
        "has_sf_wdl": torch.ones((b,)),
        "sf_wdl": sf_wdl,
    }


def _outputs(b: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "policy_own": torch.randn((b, POLICY_SIZE)),
        "wdl": torch.randn((b, 3)),
        "sf_eval": torch.randn((b, 3)),
    }


def _losses(*, conf_power: float, draw_scale: float) -> dict[str, torch.Tensor]:
    b = 8
    return compute_loss(
        _outputs(b),
        _batch(b),
        # A non-zero SF share of the value blend, so the blend's SF component is
        # live: if the knobs reached it at all, they would move `wdl_ce` here.
        sf_wdl_frac=0.45,
        w_sf_eval=0.15,
        sf_wdl_conf_power=conf_power,
        sf_wdl_draw_scale=draw_scale,
    )


def test_conf_knobs_leave_the_wdl_value_target_bit_identical() -> None:
    off = _losses(conf_power=0.0, draw_scale=1.0)
    on = _losses(conf_power=1.0, draw_scale=0.55)

    assert torch.equal(off["wdl_ce"], on["wdl_ce"]), (
        "sf_wdl_conf_power/sf_wdl_draw_scale moved the WDL value loss. They are "
        "documented (docs/model_heads.md, losses._compute_sf_wdl_mask) as aux-only. "
        "Wiring them into the blend is a training-affecting change: ledger entry first."
    )
    # `wdl_ce` and `blended_wdl_ce` are the same tensor; pin both so a future
    # split of the two cannot let one drift unnoticed.
    assert torch.equal(off["blended_wdl_ce"], on["blended_wdl_ce"])
    assert torch.equal(off["wdl_onehot_ce"], on["wdl_onehot_ce"])


def test_conf_knobs_do_move_the_aux_sf_eval_head() -> None:
    """Negative control: the knobs are not simply inert everywhere."""
    off = _losses(conf_power=0.0, draw_scale=1.0)
    on = _losses(conf_power=1.0, draw_scale=0.55)

    assert not torch.equal(off["sf_eval_ce"], on["sf_eval_ce"])
    assert not torch.equal(off["total"], on["total"])
