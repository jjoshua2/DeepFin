"""Terminal-proximal outcome share of the WDL value target.

The blend under test lives at ONE site — the `wdl` target built inside
`train/losses.py::compute_loss`, which is the tensor every value gradient comes
from. These tests read the blend back out of `compute_loss` rather than
re-deriving it from a copy of the formula, so a knob that is accepted and then
silently ignored fails here.

HOW THE TARGET IS RECOVERED. `compute_loss` returns losses, not targets, so the
blend is solved for: the value loss is a soft CE, ``ce = -sum_c t_c *
log_softmax(logits)_c``. Run the SAME batch through two different `wdl` logit
vectors and add ``sum_c t_c == 1`` and the 3-vector is determined exactly by a
3x3 solve. The batch is then built so the three components are the three basis
vectors — SF label = pure W, search label = pure D, game outcome = pure L — so
the recovered vector IS ``(w_sf, w_search, w_game + w_outcome)``, no
disentangling needed. `game_frac` is 0 in every case here (the fracs sum to 1,
as in production), so the third entry is the outcome's weight alone.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.train.losses import (
    _get_mask,
    _normalize_sf_wdl_probs,
    compute_loss,
    masked_mean,
    soft_cross_entropy,
    terminal_outcome_transfer_taper,
)

# Production ply cap (`selfplay.max_plies` in configs/pbt2_small.yaml). The
# `moves_left` field is plies-remaining DIVIDED BY THIS, so it is also the
# multiplier that recovers a ply distance.
MAX_PLIES = 450

# Planned production values.
PLIES = 7
FULL_PLIES = 2

# Realized production fracs at the time of writing: sf_wdl_frac sits on its
# 0.69 floor (PID-recomputed each iteration), search takes the remaining 0.31,
# game_frac 0.
SF_FRAC = 0.69
SEARCH_FRAC = 0.31

_PROBE_LOGITS = (
    torch.tensor([[0.0, 1.0, 3.0]]),
    torch.tensor([[2.0, -1.0, 0.5]]),
)


def _moves_left_for(d: int, *, max_plies: int = MAX_PLIES) -> float:
    """The stored `moves_left` a row at ply distance ``d`` would carry.

    Deliberately goes through float16: that is the shard dtype
    (`replay/shard.py`), and its quantization is why the decoder rounds.
    """
    return float(np.float16(float(d) / float(max_plies)))


def _batch(
    d: int | None,
    *,
    outcome: int = 2,
    has_search: float = 1.0,
    has_sf: float = 1.0,
    max_plies: int = MAX_PLIES,
) -> dict[str, torch.Tensor]:
    """One row whose three blend components are the three basis vectors.

    ``d is None`` omits `moves_left` entirely (a legacy shard); a row can also
    carry the field with ``has_moves_left = 0``, which `_unmasked_batch` below
    builds.
    """
    batch: dict[str, torch.Tensor] = {
        "x": torch.zeros((1, 1, 1, 1)),
        "policy_t": torch.tensor([[1.0, 0.0, 0.0]]),
        "wdl_t": torch.tensor([outcome], dtype=torch.long),
        "has_policy": torch.ones((1,)),
        "is_network_turn": torch.ones((1,)),
        "sf_wdl": torch.tensor([[1.0, 0.0, 0.0]]),
        "has_sf_wdl": torch.full((1,), has_sf),
        "search_wdl": torch.tensor([[0.0, 1.0, 0.0]]),
        "has_search_wdl": torch.full((1,), has_search),
    }
    if d is not None:
        batch["moves_left"] = torch.tensor([_moves_left_for(d, max_plies=max_plies)])
        batch["has_moves_left"] = torch.ones((1,))
    return batch


def _losses(batch: dict[str, torch.Tensor], logits: torch.Tensor, **knobs) -> dict:
    return compute_loss(
        {"policy": torch.zeros((1, 3)), "wdl": logits},
        batch,
        sf_wdl_frac=SF_FRAC,
        search_wdl_frac=SEARCH_FRAC,
        w_policy=0.0, w_sf_move=0.0, w_sf_eval=0.0,
        w_categorical=0.0, w_volatility=0.0, w_moves_left=0.0,
        moves_left_max_plies=float(MAX_PLIES),
        **knobs,
    )


def _recover_target(batch: dict[str, torch.Tensor], **knobs) -> np.ndarray:
    """The 3-vector WDL target `compute_loss` actually trained on (see module doc)."""
    rows = [np.ones(3, dtype=np.float64)]
    rhs = [1.0]
    for logits in _PROBE_LOGITS:
        ce = float(_losses(batch, logits.clone(), **knobs)["wdl_ce"].item())
        rows.append(-torch.log_softmax(logits, dim=-1)[0].double().numpy())
        rhs.append(ce)
    return np.linalg.solve(np.stack(rows), np.asarray(rhs, dtype=np.float64))


def _reference_untouched_blend(
    batch: dict[str, torch.Tensor], *, sf_wdl_frac: float, search_wdl_frac: float,
) -> torch.Tensor:
    """The pre-feature blend, transcribed from `main`'s expressions verbatim.

    Structurally a duplicate, and that is the point: it is the exact
    arithmetic, in the exact order, that the feature-off path must still
    perform. Anything less than `torch.equal` against this is a behaviour
    change on the production value target.
    """
    game_oh = torch.nn.functional.one_hot(batch["wdl_t"].to(torch.int64), 3).float()
    sf_probs = _normalize_sf_wdl_probs(batch.get("sf_wdl"))
    search_probs = _normalize_sf_wdl_probs(batch.get("search_wdl"))
    sf_available_b = _get_mask(batch, "has_sf_wdl").float().unsqueeze(1)
    search_available_b = _get_mask(batch, "has_search_wdl").float().unsqueeze(1)
    game_frac = 1.0 - (sf_wdl_frac + search_wdl_frac)
    target = game_frac * game_oh
    if sf_probs is not None:
        target += sf_wdl_frac * (
            sf_available_b * sf_probs + (1.0 - sf_available_b) * game_oh
        )
    else:
        target += sf_wdl_frac * game_oh
    if search_probs is not None:
        target += search_wdl_frac * (
            search_available_b * search_probs + (1.0 - search_available_b) * game_oh
        )
    else:
        target += search_wdl_frac * game_oh
    return target


def _reference_wdl_ce(batch: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
    target = _reference_untouched_blend(
        batch, sf_wdl_frac=SF_FRAC, search_wdl_frac=SEARCH_FRAC,
    )
    return masked_mean(
        soft_cross_entropy(logits, target.detach()), _get_mask(batch, "is_network_turn"),
    )


def _expected_taper(d: int, *, plies: int = PLIES, full_plies: int = FULL_PLIES) -> float:
    return float(np.clip((plies - d) / (plies - full_plies), 0.0, 1.0))


# --------------------------------------------------------------------------
# (a) taper math and the moves_left round trip
# --------------------------------------------------------------------------

def test_taper_at_the_boundary_distances() -> None:
    """plies=7, full=2 => span 5. d<=2 takes the whole share, d>=7 takes none."""
    expected = {1: 1.0, 2: 1.0, 6: 0.2, 7: 0.0, 8: 0.0}
    for d, want in expected.items():
        taper = terminal_outcome_transfer_taper(
            _batch(d), plies=PLIES, full_plies=FULL_PLIES, max_plies=float(MAX_PLIES),
        )
        assert taper is not None
        assert taper.shape == (1, 1)
        assert abs(float(taper.item()) - want) < 1e-6, f"d={d}"


def test_taper_is_off_by_default_and_at_zero_plies() -> None:
    """0 is the default and means OFF — not "a taper of width zero"."""
    assert terminal_outcome_transfer_taper(
        _batch(1), plies=0, full_plies=FULL_PLIES, max_plies=float(MAX_PLIES),
    ) is None


def test_taper_degenerates_to_a_step_when_full_plies_reaches_plies() -> None:
    """span <= 0 is the limit of the ramp, not a divide-by-zero."""
    for full_plies in (PLIES, PLIES + 5):
        for d, want in ((0, 1.0), (6, 1.0), (7, 0.0), (9, 0.0)):
            taper = terminal_outcome_transfer_taper(
                _batch(d), plies=PLIES, full_plies=full_plies, max_plies=float(MAX_PLIES),
            )
            assert taper is not None
            assert float(taper.item()) == want, f"full={full_plies} d={d}"


def test_moves_left_round_trips_through_float16_at_max_plies_450() -> None:
    """Every ply distance a production row can carry decodes back exactly.

    float16 has 11 bits of mantissa and `d / 450` is not representable, so the
    reconstruction lands just off the integer — 7 comes back as 7.0003. Nearest
    rounding is what makes the decode exact; truncation would read 6.
    """
    for d in range(MAX_PLIES + 1):
        stored = _moves_left_for(d)
        recovered = float(np.round(stored * MAX_PLIES))
        assert recovered == float(d), f"d={d} stored={stored!r} -> {recovered}"


def test_float16_quantization_actually_moves_the_value() -> None:
    """The round-trip test above is not vacuous — the stored value is inexact."""
    inexact = [d for d in range(1, 20) if _moves_left_for(d) * MAX_PLIES != float(d)]
    assert len(inexact) >= 15


# --------------------------------------------------------------------------
# (b) default-off bit identity
# --------------------------------------------------------------------------

def test_default_off_is_bit_identical_to_the_untouched_blend() -> None:
    """The knob's default must not perturb the production value target at all.

    Compared with `torch.equal` against a verbatim transcription of the
    pre-feature expressions — the loss is a deterministic function of the
    target, so bit-equal losses over rows spanning the whole taper window mean
    bit-equal targets.
    """
    for d in (0, 1, 2, 3, 6, 7, 8, 40, 200):
        batch = _batch(d)
        logits = _PROBE_LOGITS[0].clone()
        got = _losses(batch, logits)
        assert torch.equal(got["wdl_ce"], _reference_wdl_ce(batch, logits)), f"d={d}"
        assert float(got["wdl_terminal_outcome_weight_sum"].item()) == 0.0
        assert float(got["wdl_terminal_outcome_rows"].item()) == 0.0


def test_default_off_ignores_moves_left_entirely() -> None:
    """Feature off => the target cannot be a function of `moves_left` at all."""
    near = _losses(_batch(1), _PROBE_LOGITS[0].clone())["wdl_ce"]
    far = _losses(_batch(300), _PROBE_LOGITS[0].clone())["wdl_ce"]
    absent = _losses(_batch(None), _PROBE_LOGITS[0].clone())["wdl_ce"]
    assert torch.equal(near, far)
    assert torch.equal(near, absent)


def test_sf_frac_default_leaves_the_sf_share_bit_identical() -> None:
    """`wdl_terminal_outcome_sf_frac` defaults to 0.0 = the SF share is untouched.

    Only the SEARCH share moves, so at d <= full_plies the whole target is the
    SF component plus the outcome — exactly the untouched blend with the search
    weight relabelled onto the outcome. That is asserted bit-for-bit, because
    the SF component is load-bearing supervision.
    """
    d = 1
    logits = _PROBE_LOGITS[0].clone()
    on = _losses(
        _batch(d), logits,
        wdl_terminal_outcome_plies=PLIES, wdl_terminal_outcome_full_plies=FULL_PLIES,
    )["wdl_ce"]
  # `search_wdl` set to the outcome one-hot reproduces "the search share went
  # to the outcome" through the UNTOUCHED code path.
    equivalent = _batch(d)
    equivalent["search_wdl"] = torch.nn.functional.one_hot(
        equivalent["wdl_t"].to(torch.int64), 3,
    ).float()
    assert torch.equal(on, _reference_wdl_ce(equivalent, logits))


# --------------------------------------------------------------------------
# (c) the missing-moves_left guard
# --------------------------------------------------------------------------

def test_rows_without_moves_left_keep_the_unchanged_blend() -> None:
    """`has_moves_left = 0` (or the field absent) => no transfer.

    The failure this guards is specific: `moves_left` defaults to 0.0 in the
    collated batch, which decodes as d = 0 — the MAXIMUM transfer. A row whose
    shard never carried the field must not be read as terminal.
    """
    knobs = {
        "wdl_terminal_outcome_plies": PLIES,
        "wdl_terminal_outcome_full_plies": FULL_PLIES,
    }
    logits = _PROBE_LOGITS[0].clone()

    masked = _batch(0)
    masked["has_moves_left"] = torch.zeros((1,))
    got = _losses(masked, logits, **knobs)
    assert torch.equal(got["wdl_ce"], _reference_wdl_ce(masked, logits))
    assert float(got["wdl_terminal_outcome_rows"].item()) == 0.0

    absent = _batch(None)
    got_absent = _losses(absent, logits, **knobs)
    assert torch.equal(got_absent["wdl_ce"], _reference_wdl_ce(absent, logits))
    assert float(got_absent["wdl_terminal_outcome_rows"].item()) == 0.0


def test_rows_without_search_wdl_are_unchanged_by_the_transfer() -> None:
    """No new path for a missing search label: those rows already fall back to
    the outcome, so moving the search share onto the outcome is a no-op."""
    batch = _batch(1, has_search=0.0)
    logits = _PROBE_LOGITS[0].clone()
    got = _losses(
        batch, logits,
        wdl_terminal_outcome_plies=PLIES, wdl_terminal_outcome_full_plies=FULL_PLIES,
    )
    assert torch.equal(got["wdl_ce"], _reference_wdl_ce(batch, logits))


# --------------------------------------------------------------------------
# (d) the blend itself, at every d
# --------------------------------------------------------------------------

def test_search_share_moves_to_the_outcome_on_the_taper() -> None:
    for d in (1, 2, 6, 7, 8):
        w_sf, w_search, w_outcome = _recover_target(
            _batch(d),
            wdl_terminal_outcome_plies=PLIES,
            wdl_terminal_outcome_full_plies=FULL_PLIES,
        )
        taper = _expected_taper(d)
        assert abs(w_sf - SF_FRAC) < 1e-5, f"d={d}"
        assert abs(w_search - SEARCH_FRAC * (1.0 - taper)) < 1e-5, f"d={d}"
        assert abs(w_outcome - SEARCH_FRAC * taper) < 1e-5, f"d={d}"
        assert abs((w_sf + w_search + w_outcome) - 1.0) < 1e-6, f"d={d}"


def test_the_sf_share_is_invariant_at_every_d_with_the_default_sf_frac() -> None:
    """The whole taper window, not just its ends: the SF weight never moves."""
    for d in range(12):
        w_sf, _, _ = _recover_target(
            _batch(d),
            wdl_terminal_outcome_plies=PLIES,
            wdl_terminal_outcome_full_plies=FULL_PLIES,
        )
        assert abs(w_sf - SF_FRAC) < 1e-5, f"d={d}"


def test_the_transfer_rides_the_realized_search_frac_not_a_constant() -> None:
    """The fracs are PID-recomputed every iteration; a hard-coded 0.31 would
    survive the test above and fail this one."""
    for search_frac in (0.10, 0.31, 0.50):
        sf_frac = 1.0 - search_frac
        got = compute_loss(
            {"policy": torch.zeros((1, 3)), "wdl": _PROBE_LOGITS[0].clone()},
            _batch(6),
            sf_wdl_frac=sf_frac, search_wdl_frac=search_frac,
            w_policy=0.0, w_sf_move=0.0, w_sf_eval=0.0,
            w_categorical=0.0, w_volatility=0.0, w_moves_left=0.0,
            moves_left_max_plies=float(MAX_PLIES),
            wdl_terminal_outcome_plies=PLIES,
            wdl_terminal_outcome_full_plies=FULL_PLIES,
        )
        want = search_frac * _expected_taper(6)
        assert abs(float(got["wdl_terminal_outcome_weight_sum"].item()) - want) < 1e-6


def test_sf_frac_arm_matches_the_closed_form() -> None:
    """The aggressive (offline-screen) arm: sf_frac=0.5 on the same taper.

    Pre-committed numbers from the spec, at the production floor sf=0.69 /
    search=0.31 — d<=2 gives SF 0.345 / search 0.0 / outcome 0.655, and d=6
    (taper 0.2) gives SF 0.621 / search 0.248 / outcome 0.131.
    """
    cases = {
        1: (0.345, 0.0, 0.655),
        2: (0.345, 0.0, 0.655),
        6: (0.621, 0.248, 0.131),
        7: (SF_FRAC, SEARCH_FRAC, 0.0),
    }
    for d, want in cases.items():
        got = _recover_target(
            _batch(d),
            wdl_terminal_outcome_plies=PLIES,
            wdl_terminal_outcome_full_plies=FULL_PLIES,
            wdl_terminal_outcome_sf_frac=0.5,
        )
        assert np.allclose(got, np.asarray(want), atol=1e-5), f"d={d}: {got} != {want}"
        assert abs(float(np.sum(got)) - 1.0) < 1e-6, f"d={d}"


def test_sf_frac_arm_is_reported_in_the_transferred_weight() -> None:
    """Both transfers land in the observability column, not just the search one."""
    got = _losses(
        _batch(1), _PROBE_LOGITS[0].clone(),
        wdl_terminal_outcome_plies=PLIES,
        wdl_terminal_outcome_full_plies=FULL_PLIES,
        wdl_terminal_outcome_sf_frac=0.5,
    )
    assert abs(
        float(got["wdl_terminal_outcome_weight_sum"].item()) - (SEARCH_FRAC + 0.5 * SF_FRAC)
    ) < 1e-6
    assert float(got["wdl_terminal_outcome_rows"].item()) == 1.0


# --------------------------------------------------------------------------
# (e) the yaml keys reach the loss call the training step makes
# --------------------------------------------------------------------------

class _TinyModel(torch.nn.Module):
    """Smallest model a Trainer will build an optimizer over."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32),
        }


def test_the_yaml_keys_reach_the_loss_call(tmp_path) -> None:
    """config -> Trainer -> `_loss_kwargs` -> `compute_loss`.

    `_loss_kwargs` is the dict the training step splats into `compute_loss`
    (`trainer.py`, both call sites), so a key present here is a key the trained
    target sees. `moves_left_max_plies` is sourced from the SELFPLAY `max_plies`
    the rows were written under, not from a knob of its own.
    """
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    config = {
        "device": "cpu",
        "lr": 1e-3,
        "no_amp": True,
        "max_plies": MAX_PLIES,
        "wdl_terminal_outcome_plies": PLIES,
        "wdl_terminal_outcome_full_plies": FULL_PLIES,
        "wdl_terminal_outcome_sf_frac": 0.5,
    }
    ctor = trainer_kwargs_from_config(config, log_dir=tmp_path)
    assert ctor["wdl_terminal_outcome_plies"] == PLIES
    assert ctor["wdl_terminal_outcome_full_plies"] == FULL_PLIES
    assert ctor["wdl_terminal_outcome_sf_frac"] == 0.5
    assert ctor["moves_left_max_plies"] == MAX_PLIES

    trainer = Trainer(_TinyModel(), prefetch_batches=False, **ctor)
    kwargs = trainer._loss_kwargs
    assert kwargs["wdl_terminal_outcome_plies"] == PLIES
    assert kwargs["wdl_terminal_outcome_full_plies"] == FULL_PLIES
    assert kwargs["wdl_terminal_outcome_sf_frac"] == 0.5
    assert kwargs["moves_left_max_plies"] == float(MAX_PLIES)

  # And the dict as a whole is accepted by the function it is splatted into,
  # with the fracs the trainer would carry at that moment.
    trainer.sf_wdl_frac = SF_FRAC
    trainer.search_wdl_frac = SEARCH_FRAC
    got = compute_loss(
        {"policy": torch.zeros((1, 3)), "wdl": _PROBE_LOGITS[0].clone()},
        _batch(1),
        **trainer._loss_kwargs,
    )
    assert float(got["wdl_terminal_outcome_rows"].item()) == 1.0


def test_the_defaults_are_off_end_to_end(tmp_path) -> None:
    """An empty config must not enable the transfer on the production path."""
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    ctor = trainer_kwargs_from_config(
        {"device": "cpu", "lr": 1e-3, "no_amp": True}, log_dir=tmp_path,
    )
    assert ctor["wdl_terminal_outcome_plies"] == 0
    assert ctor["wdl_terminal_outcome_sf_frac"] == 0.0
    trainer = Trainer(_TinyModel(), prefetch_batches=False, **ctor)
    assert trainer._loss_kwargs["wdl_terminal_outcome_plies"] == 0


def test_a_live_yaml_edit_pushes_the_knobs_onto_the_trainer(tmp_path) -> None:
    """The per-iteration push, so enabling does not wait for a second restart.

    `moves_left_max_plies` deliberately does NOT ride along: it re-interprets
    rows already in the replay window, so it stays construction-time.
    """
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    ctor = trainer_kwargs_from_config(
        {"device": "cpu", "lr": 1e-3, "no_amp": True, "max_plies": MAX_PLIES},
        log_dir=tmp_path,
    )
    trainer = Trainer(_TinyModel(), prefetch_batches=False, **ctor)
    assert trainer._loss_kwargs["wdl_terminal_outcome_plies"] == 0

    _apply_lr_gamma_weights(
        trainer,
        {
            "wdl_terminal_outcome_plies": PLIES,
            "wdl_terminal_outcome_full_plies": FULL_PLIES,
            "wdl_terminal_outcome_sf_frac": 0.25,
        },
        rescale_current_lr=True,
    )
    kwargs = trainer._loss_kwargs
    assert kwargs["wdl_terminal_outcome_plies"] == PLIES
    assert kwargs["wdl_terminal_outcome_full_plies"] == FULL_PLIES
    assert kwargs["wdl_terminal_outcome_sf_frac"] == 0.25
    assert kwargs["moves_left_max_plies"] == float(MAX_PLIES)


def test_the_live_yaml_validator_accepts_the_new_keys() -> None:
    """An unknown key rejects the WHOLE reload, so the keys must be registered."""
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({
        "train": {
            "wdl_terminal_outcome_plies": PLIES,
            "wdl_terminal_outcome_full_plies": FULL_PLIES,
            "wdl_terminal_outcome_sf_frac": 0.0,
        },
        "selfplay": {"max_plies": MAX_PLIES},
    })
    assert flat["wdl_terminal_outcome_plies"] == PLIES
    assert flat["wdl_terminal_outcome_full_plies"] == FULL_PLIES
    assert flat["wdl_terminal_outcome_sf_frac"] == 0.0
    assert flat["max_plies"] == MAX_PLIES


def test_enabled_without_the_ply_cap_raises() -> None:
    """A missing divisor mis-decodes every distance — it must not default quietly."""
    with pytest.raises(ValueError, match="moves_left_max_plies"):
        terminal_outcome_transfer_taper(
            _batch(1), plies=PLIES, full_plies=FULL_PLIES, max_plies=0.0,
        )
