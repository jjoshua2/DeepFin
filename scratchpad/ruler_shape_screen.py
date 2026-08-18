"""Screen: can a POSITIVE->POSITIVE change move the loss but NOT the ruler id?

Peer review 2026-08-18. `_EFFECTIVE_WEIGHT` fixed off/on normalisation, but
`TRAINER_WEIGHT_KEYS` is not a list of pure scalar multipliers -- several
entries redefine target distributions, masks and thresholds. For those, a
magnitude change within the active range changes the OBJECTIVE while membership
(and therefore the ruler id) sits still: the false-negative class the machinery
exists to prevent.

For every key, at two POSITIVE values:
    objective moved?  = any per-term loss differs
    ruler moved?      = eval_ruler_id_for differs
    FAILURE           = objective moved AND ruler did not
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/josh/projects/chess-armf")

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import Trainer
from tests.test_sf_policy_floor import _tiny_batch

PROBES: dict[str, tuple[float, float]] = {
    "sf_wdl_temperature": (1.0, 2.0),
    "soft_policy_min_tv": (0.1, 0.3),
    "sf_wdl_conf_power": (0.5, 1.0),
    "sf_wdl_draw_scale": (0.5, 1.5),
    "sf_wdl_frac": (0.3, 0.7),
    "search_wdl_frac": (0.3, 0.7),
    "sf_search_dampen_sf_low": (0.2, 0.6),
    "sf_search_dampen_sf_high": (0.2, 0.6),
}
DEFAULT_PROBE = (0.5, 1.5)
ON = {
    "w_policy": 1.0, "w_soft": 1.0, "w_future": 1.0, "w_sf_own": 1.0,
    "w_sf_own_regret": 1.0, "w_wdl": 1.0, "w_sf_move": 1.0, "w_sf_eval": 1.0,
    "w_categorical": 1.0, "w_volatility": 1.0, "w_sf_volatility": 1.0,
    "w_moves_left": 1.0,
    # ⚑ WITHOUT THESE THE WHOLE SF/SEARCH BLEND IS MULTIPLIED BY ZERO.
    # `target += sf_wdl_frac_f * sf_component`, and both fracs DEFAULT TO 0.0
    # (losses.py:1179-1180). Every knob that acts through `sf_component` --
    # the dampen pair, conf_power, draw_scale, temperature -- then reads
    # "not exercised" while `keep`/`sf_effective` demonstrably move.
    # Measured 2026-08-18: keep 1.0->0.1 with sf_frac_f == 0.0.
    "sf_wdl_frac": 0.4, "search_wdl_frac": 0.4,
}


def dampen_variant():
    """A fixture that CONTAINS both SF/search disagreement directions.

    `keep = 1 - (dampen_low * dis_sf_low + dampen_high * dis_sf_high)` and
    `sf_effective = sf_available * keep`, so the dampen knobs can only move the
    blended WDL target on rows where the corresponding disagreement EXISTS.
    A random fixture has ~none, which is why they read "not exercised" -- the
    same vacuity that hid `soft_policy_min_tv`.

    sf_sig = sf_wdl[win] - sf_wdl[loss]; likewise for search.
      row 0: SF losing / search winning  -> dis_sf_low
      row 1: SF winning / search losing  -> dis_sf_high
    """
    outputs, batch = _tiny_batch()
    n = int(batch["wdl_t"].shape[0])
    outputs["sf_eval"] = torch.randn(n, 3)
    sf = torch.full((n, 3), 1.0 / 3.0)
    sr = torch.full((n, 3), 1.0 / 3.0)
    sf[0] = torch.tensor([0.10, 0.20, 0.70])   # SF: STM losing
    sr[0] = torch.tensor([0.70, 0.20, 0.10])   # search: STM winning
    sf[1] = torch.tensor([0.70, 0.20, 0.10])   # SF: STM winning
    sr[1] = torch.tensor([0.10, 0.20, 0.70])   # search: STM losing
    batch["sf_wdl"] = sf
    batch["has_sf_wdl"] = torch.ones(n)
    batch["search_wdl"] = sr
    batch["has_search_wdl"] = torch.ones(n)
    a = torch.linspace(0.0, 1.0, n).unsqueeze(1)
    r = torch.softmax(torch.randn(n, int(batch["policy_t"].shape[1])), dim=-1)
    batch["policy_soft_t"] = (1.0 - a) * batch["policy_t"] + a * r
    batch["has_policy_soft"] = torch.ones(n)
    return outputs, batch


def variants():
    yield dampen_variant()
    for search_wdl in (False, True):
        for sf_wdl in (True, False):
            outputs, batch = _tiny_batch()
            n = int(batch["wdl_t"].shape[0])
            outputs["sf_eval"] = torch.randn(n, 3)
            if sf_wdl:
                batch["sf_wdl"] = torch.rand(n, 3)
                batch["has_sf_wdl"] = torch.ones(n)
            if search_wdl:
                batch["search_wdl"] = torch.rand(n, 3)
                batch["has_search_wdl"] = torch.ones(n)
            # ⚑ soft target must be a graded PERTURBATION of the hard target, or
            # every row's TV sits far above the probe and `soft_policy_min_tv`
            # is inert -- "not exercised" masquerading as "no defect".
            _alpha = torch.linspace(0.0, 1.0, n).unsqueeze(1)
            _rand = torch.softmax(torch.randn(n, int(batch["policy_t"].shape[1])), dim=-1)
            batch["policy_soft_t"] = (1.0 - _alpha) * batch["policy_t"] + _alpha * _rand
            batch["has_policy_soft"] = torch.ones(n)
            yield outputs, batch


def scalars(res) -> dict[str, float]:
    out = {}
    for k, v in res.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k] = float(v.detach().item())
    return out


def ruler_for(weights: dict[str, float]) -> str:
    return Trainer.eval_ruler_id_for(
        batch_size=512, steps=0, mirror_prob=0.0, full_pass=True,
        loss_weights=weights, loss_shape={},
    )


def main() -> None:
    base_w = {k: 1.0 for k in TRAINER_WEIGHT_KEYS}
    rows = []
    for key in TRAINER_WEIGHT_KEYS:
        lo, hi = PROBES.get(key, DEFAULT_PROBE)
        moved = False
        for outputs, batch in variants():
            try:
                a = scalars(compute_loss(outputs, batch, **{**ON, key: lo}))
                b = scalars(compute_loss(outputs, batch, **{**ON, key: hi}))
            except TypeError:
                a = b = {}
            # ⚑ EXCLUDE `total`. A pure multiplier moves `total` and leaves every
            # per-head loss untouched -- that is the ACCEPTED magnitude exclusion,
            # not a defect. A SHAPE knob moves the per-head loss itself, because it
            # redefines the target/mask that loss is computed against. Only the
            # second is a ruler false negative.
            per_head = [k for k in a if k != "total"]
            if per_head and any(a[k] != b.get(k) for k in per_head):
                moved = True
                break
        r_lo = ruler_for({**base_w, key: lo})
        r_hi = ruler_for({**base_w, key: hi})
        rows.append((key, lo, hi, moved, r_lo != r_hi))

    print(f"{'key':28s} {'probe':>12s} {'per-head moved':>16s} {'ruler moved':>12s}  verdict")
    fails = []
    for key, lo, hi, moved, rmoved in rows:
        verdict = "ok"
        if moved and not rmoved:
            verdict = "⚑ FALSE NEGATIVE"
            fails.append(key)
        elif not moved:
            verdict = "(not exercised)"
        print(f"{key:28s} {lo:5.2f}->{hi:<5.2f} {str(moved):>16s} {str(rmoved):>12s}  {verdict}")
    print(f"\nFALSE NEGATIVES ({len(fails)}): {fails}")


if __name__ == "__main__":
    main()
