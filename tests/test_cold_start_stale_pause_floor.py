"""The AUTO-derived stale-pause target must not sit below iteration 1's need.

A LATENT bug. It did NOT fire on the 2026-08-04 cold start -- see the ledger
CORRECTION of that night's entry: production sets
``distributed_stale_pause_target_games: 1870`` explicitly, the auto formula at
``distributed_runtime.py:493-497`` is behind ``if < 0`` and never ran, and the
cap sat 4.25x ABOVE the 440-game need rather than below it. This file is about
the path a config that does NOT set the key takes.

The mechanism, for a config relying on the default:

* the trainer's iteration-1 ingest waits for ``_games_per_iter_for_iteration(tc,
  1)`` MATCHING games (``trainable_phases.py:930,1002``);
* matching = the accepted SHA set, which from cold is exactly ONE sha (there is
  no previous publish, so ``prev_published_model_sha`` is None);
* the server pauses the fleet once that one sha has ``stale_pause_target_games``
  games queued (``server/app.py:_apply_dynamic_stale_pause``), and workers
  resume only on a NEW published sha;
* the auto target is ``ceil(games_per_iter * distributed_prev_model_max_fraction)``
  = ``ceil(440 * 0.6)`` = 264, which is BELOW the 440 the trainer is waiting for.

So the fleet stops producing at 264 while the trainer needs 440, and only a new
sha -- which requires the iteration to finish -- can release it. A RESUME cannot
hit this: prev-sha games count as matching, which is why the whole class is
cold-start-only.

The fix floors the auto value at iteration 1's need, and is gated three ways:

* **only when ``trainer_step <= 0``** -- the cold start. No training step taken
  means no second sha can have been published, so the accepted set has exactly
  one member. It self-clears the moment training advances the step, and a
  resume never enters it.
* **only the auto path** -- an explicit target is a recorded operator decision
  (production's 1870) and is published verbatim.
* **it only ever RAISES** -- never caps.

⚑ The first revision of this fix had only the last two gates and asserted that
steady state was unchanged. That was a hand-wave, and it was wrong: at
games_per_iter=1000 / prev_frac=0.5 the steady-state target moved 500 -> 1000.
Three pre-existing cases in ``test_distributed_selfplay_backpressure`` that
publish at ``trainer_step=123`` caught it in CI. ``test_the_floor_does_not_fire_
once_training_has_stepped`` below now enforces the claim next to the fix instead
of relying on a suite that happened to cover it.

⚑ Expected values here are HARDCODED rather than computed from the helper under
test. A test that re-derives the number from the same function it is checking
cannot fail when that function is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.tune.distributed_runtime import _publish_distributed_trial_state


class _FakeTrainer:
    def export_swa(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}}, str(path))


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_mult=2,
        use_smolgen=False,
        use_nla=False,
        use_qk_rmsnorm=False,
        use_gradient_checkpointing=False,
    )


def _publish(tmp_path: Path, config: dict, *, trainer_step: int = 0) -> dict:
    """Publish once and return the realized manifest's backpressure block.

    ``trainer_step`` defaults to 0 -- the COLD START. It is the signal the fix
    gates on: no training step taken means no second sha can exist, so the
    trainer's accepted set has exactly one member. Pass a non-zero value for a
    steady-state publish.
    """
    base = {
        "selfplay_batch": 16,
        "max_plies": 240,
        "distributed_prev_model_max_fraction": 0.60,
    }
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(),
        config={**base, **config},
        model_cfg=_model_cfg(),
        server_root=tmp_path / "server",
        trial_id="t0",
        training_iteration=0 if trainer_step <= 0 else 7,
        trainer_step=trainer_step,
        sf_nodes=1000,
        mcts_simulations=32,
    )
    manifest = json.loads(
        (tmp_path / "server" / "trials" / "t0" / "publish" / "manifest.json").read_text(
            encoding="utf-8",
        )
    )
    return dict(manifest["backpressure"])


def test_auto_target_is_floored_at_the_first_iteration_need(tmp_path: Path) -> None:
    """Production's shape with the key UNSET: 440 games/iter, ramp disabled.

    RED on origin/main, which publishes ceil(440 * 0.60) = 264 -- below the 440
    matching games iteration 1 waits for, which is the deadlock.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 440,
            "games_per_iter_start": 440,
            "games_per_iter_ramp_iters": 0,
        },
    )

    assert bp["stale_pause_target_games"] == 440, (
        "the auto target must be floored at iteration 1's 440-game need; "
        "ceil(440*0.60)=264 pauses the fleet before the trainer can ever "
        "reach its target, and only a NEW sha resumes it"
    )


def test_the_floor_uses_the_RAMPED_need_not_games_per_iter(tmp_path: Path) -> None:
    """Under a ramp, iteration 1 needs ``games_per_iter_start``, not the target.

    ``_games_per_iter_for_iteration(tc, 1)`` returns ``start`` whenever
    ``ramp_iters >= 2``. Here start=400 and games_per_iter=440, so:

    * origin/main publishes 264 (RED -- below the 400 actually needed);
    * flooring at ``games_per_iter`` would publish 440 (also wrong: it over-caps
      by 40 games and, more importantly, means the floor is not reading the
      quantity the trainer actually waits on);
    * the fix publishes exactly 400.

    The exact-equality assertion is what separates the second case from the
    third -- a ``>= need`` assertion passes for both.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 440,
            "games_per_iter_start": 400,
            "games_per_iter_ramp_iters": 10,
        },
    )

    assert bp["stale_pause_target_games"] == 400, (
        "iteration 1 under a ramp needs games_per_iter_start (400), so the "
        "floor must be 400 -- not 264 (unfloored) and not 440 (floored at the "
        "ramp TARGET instead of iteration 1's value)"
    )


def test_a_ramp_shorter_than_two_iterations_needs_the_full_target(tmp_path: Path) -> None:
    """``ramp_iters <= 1`` means iteration 1 is already at the full target.

    Pins the boundary in ``_games_per_iter_for_iteration``'s
    ``iteration_idx >= ramp_iters`` branch: with ramp_iters=1, iteration 1 needs
    440 even though ``games_per_iter_start`` is 100.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 440,
            "games_per_iter_start": 100,
            "games_per_iter_ramp_iters": 1,
        },
    )

    assert bp["stale_pause_target_games"] == 440, (
        "at ramp_iters=1 iteration 1 is already at the full target"
    )


def test_the_floor_does_not_lower_a_larger_auto_value(tmp_path: Path) -> None:
    """The floor RAISES; it must never cap.

    With a ramp start of 10, iteration 1 needs 10 while the frac-based value is
    264. The published target must stay 264 -- a ``min`` here would hand the
    fleet a far tighter brake than the steady-state design intends.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 440,
            "games_per_iter_start": 10,
            "games_per_iter_ramp_iters": 10,
        },
    )

    assert bp["stale_pause_target_games"] == 264, (
        "ceil(440*0.60)=264 already exceeds iteration 1's need of 10; the "
        "floor must leave it alone"
    )


def test_an_explicit_key_is_left_exactly_as_configured(tmp_path: Path) -> None:
    """The explicit path is NOT floored, deliberately.

    Production's 1870 is a deliberate, ledger'd value carried by the minimal
    core, and it is 4.25x above the need -- nothing to fix. This test pins the
    other direction too: an operator who explicitly asks for a target BELOW the
    need still gets exactly what they asked for. Widening the fix to the
    explicit path would silently override a recorded decision, and the ledger
    CORRECTION is explicit that the explicit value was never the bug.
    """
    for explicit in (1870, 100):
        bp = _publish(
            tmp_path / f"explicit_{explicit}",
            {
                "games_per_iter": 440,
                "games_per_iter_start": 440,
                "games_per_iter_ramp_iters": 0,
                "distributed_stale_pause_target_games": explicit,
            },
        )
        assert bp["stale_pause_target_games"] == explicit, (
            f"explicit {explicit} must be published verbatim; the floor applies "
            "only to the auto-derived path"
        )


def test_the_floor_does_not_fire_once_training_has_stepped(tmp_path: Path) -> None:
    """STEADY STATE MUST BE UNTOUCHED -- and this is the assertion that makes
    that claim true rather than a hand-wave.

    ⚑ The first revision of this fix applied the floor at EVERY auto publish
    and asserted in its own ledger entry that "steady state is unchanged". It
    was not: with games_per_iter=1000 and prev_frac=0.5 the steady-state target
    moved 500 -> 1000, silently changing when backpressure engages in a regime
    where no deadlock is possible. CI caught it through three pre-existing
    cases in ``test_distributed_selfplay_backpressure`` that publish at
    ``trainer_step=123``. The claim is now enforced here as well, next to the
    fix, rather than living only in a suite that happens to cover it.

    Once training has stepped, a second sha exists, prev-sha games count as
    matching, and the cold-start deadlock is unreachable -- so the floor has no
    business firing.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 1000,
            "games_per_iter_start": 1000,
            "games_per_iter_ramp_iters": 0,
            "distributed_prev_model_max_fraction": 0.5,
        },
        trainer_step=123,
    )

    assert bp["stale_pause_target_games"] == 500, (
        "at trainer_step=123 the fleet already has a previous sha to draw "
        "matching games from; the target must stay the frac-based 500"
    )


def test_the_floor_still_fires_on_the_bootstrap_publish(tmp_path: Path) -> None:
    """The paired positive control for the gate above.

    Same config, cold. Without both halves a mutation that disables the floor
    entirely and one that applies it everywhere are indistinguishable.
    """
    bp = _publish(
        tmp_path,
        {
            "games_per_iter": 1000,
            "games_per_iter_start": 1000,
            "games_per_iter_ramp_iters": 0,
            "distributed_prev_model_max_fraction": 0.5,
        },
        trainer_step=0,
    )

    assert bp["stale_pause_target_games"] == 1000, (
        "from cold there is only one sha, so the target must reach iteration "
        "1's full need"
    )
