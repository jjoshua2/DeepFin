"""Pin the lc0 positive control to the REPLAY BUFFER production is running.

⚑⚑ THE THIRD AXIS OF "OUR EXACT STACK", AND IT WAS THE ONE NOBODY WAS LOOKING
AT. ``lc0_control_arch`` pins the network, ``lc0_control_trainer`` pins
``trainer_kwargs_from_config``. Neither can see the buffer: ``DiskReplayBuffer``
is constructed by ``tune/trainable_init.py`` from ``TrialConfig`` fields that
``trainer_kwargs_from_config`` does not read, so a control driver that omits
them gets ``disk_buffer.py``'s CONSTRUCTOR DEFAULTS and both existing pins pass.

Measured 2026-08-17 against the LIVE ``configs/pbt2_small.yaml``, the driver's
buffer deviated from production's on **seven** axes, not the three the review
named and not the two its own comment claimed::

    shuffle_cap                 20000 -> 100000
    shard_size                   1000 -> 2000
    refresh_interval                5 -> 4
    refresh_shards                  3 -> 5
    diff_focus_pol_scale          0.0 -> 3.5
    diff_focus_q_weight           0.0 -> 6.0
    input_planes                 None -> 175

The arm's hypothesis is *H_stack* — "the plateau is the training stack". A
plateau produced by 5x less hot-pool diversity is a plateau of the RIG, and it
is indistinguishable in the held-out slope from the plateau the arm is looking
for. That is not a tolerance question; it is the confound that makes the whole
readout unattributable.

⚑ WHY A THIRD PIN RATHER THAN A WIDER ``LIVE_TRAINER_PIN``. That pin's contract
is exactly ``trainer_kwargs_from_config``: ``trainer_kwargs_signature`` calls
it, ``scripts/lc0_control_arch_pin.py --axis trainer`` regenerates it from it,
and its docstring's whole argument for completeness ("any key this function
reads is part of the answer") stops being true the moment a key that function
does not read is pasted in. A pin whose contents no single function produces is
a hand-maintained list, which is the thing the arch/trainer pins were written to
stop being.

⚑⚑ AND THE COVERAGE CHECK IS THE PART THAT MAKES THIS NOT DRIFT AGAIN. Listing
twelve buffer kwargs by hand has exactly the defect it is closing: the
thirteenth, added later, is silently absent and the pin still passes.
``assert_buffer_kwargs_are_classified`` reads ``DiskReplayBuffer.__init__``'s
own signature and refuses unless EVERY parameter is in one of the four
classes below. A new buffer knob is then a loud failure at the next control
launch, not a silent deviation.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chess_anti_engine.eval.lc0_control_arch import LIVE_FILE_UNREAD
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

# ── the four classes every DiskReplayBuffer kwarg must fall into ─────────────

# (1) Per-run, not per-config: the window, the directory, the generator, and the
# read/write intent. Production and the control differ on all four BY DESIGN
# (the control opens a fixed corpus read-only through a symlink farm), so they
# are excluded from the comparison rather than allowed to differ inside it.
PER_RUN_KWARGS: dict[str, str] = {
    "capacity": (
        "The control's window is its whole frozen corpus, sized by --rows on "
        "the driver, while production's is a rolling 1.5M-row window the PID "
        "and the ingest rate move every iteration. There is no value both can "
        "hold, so this is excluded from the comparison rather than allowed to "
        "differ inside it."
    ),
    "shard_dir": (
        "The control opens a symlink farm over a frozen shard list; production "
        "opens the live session's selfplay_shards directory, which grows and "
        "is pruned under it. Comparing the two paths would compare two "
        "machines' directory layouts, not two configurations."
    ),
    "rng": (
        "Seeded per run from --init-seed so an arm is reproducible. Production "
        "seeds from the trial. Equality here would mean the control could not "
        "be re-run, which is the opposite of what a control is for."
    ),
    "read_only": (
        "The control NEVER ingests -- the corpus is frozen and its purity "
        "receipt is the whole basis of the arm. Production ingests every "
        "iteration. This is the one asymmetry the rig exists to guarantee, so "
        "it is a class rather than a deviation."
    ),
}

# (2) Read straight off `TrialConfig`, exactly as `tune/trainable_init.py` does.
# ⚑ Through `TrialConfig`, not off the flat dict: `from_dict`'s defaults are what
# production realizes for a key its yaml omits, and a `.get(key, <my guess>)`
# here would compare the control against a default production does not have.
CONFIG_KWARGS: dict[str, str] = {
    "shuffle_cap": "shuffle_buffer_size",
    "shard_size": "shard_size",
    "refresh_interval": "shuffle_refresh_interval",
    "refresh_shards": "shuffle_refresh_shards",
    "shard_recency_exponent": "replay_shard_recency_exponent",
    "draw_cap_frac": "shuffle_draw_cap_frac",
    "wl_max_ratio": "shuffle_wl_max_ratio",
    "sf_gap_priority_weight": "replay_sf_gap_priority_weight",
    "fast_low_surprise_priority": "replay_fast_low_surprise_priority",
    "diff_focus_pol_scale": "diff_focus_pol_scale",
    "diff_focus_q_weight": "diff_focus_q_weight",
    "upgrade_v1_planes": "replay_upgrade_v1_planes",
}

# (3) Production DERIVES it rather than reading a field of the same name.
DERIVED_KWARGS: dict[str, str] = {
    "input_planes": (
        "Production derives it from the encoding version rather than reading a "
        "field of the same name, so there is no TrialConfig key to map. ⚑ NOT "
        "cosmetic: _pad_x_planes returns early on None, which also disables "
        "_upgrade_x_planes, _upgrade_target_version and the `stored > target` "
        "'refusing to truncate encoded inputs' ValueError -- at None the whole "
        "plane-width safety is off, so the control must set it explicitly."
    ),
}

# (4) Production passes nothing, so the constructor default IS production's
# realized value. Pinned at that default so the comparison still covers them: a
# default that moves is a production change, and the control must follow it or
# declare a deviation.
PRODUCTION_DEFAULTED_KWARGS: dict[str, str] = {
    "sf_gap_priority_signed": (
        "tune/trainable_init.py passes nothing for it, so DiskReplayBuffer's "
        "own default IS production's realized value and pinning at that default "
        "keeps the key inside the comparison. A default that moves is a "
        "production change the control must follow or declare."
    ),
    "deterministic_refresh": (
        "Same shape: production never passes it, so the constructor default is "
        "what production runs. Pinned rather than excluded, because 'production "
        "does not set it' is a statement about today's call site, not a "
        "permanent property of the knob."
    ),
}


class ControlReplayDrift(RuntimeError):
    """The control would sample from a buffer production is not running."""


def assert_buffer_kwargs_are_classified() -> None:
    """Every ``DiskReplayBuffer.__init__`` parameter is in exactly one class.

    ⚑ This is the anti-drift half of the module, and it is the half that has to
    run. Without it, a knob added to the buffer next month is absent from
    ``CONFIG_KWARGS``, absent from the signature, absent from the pin, and the
    control silently takes its default while every gate reports green — the
    same shape as the defect this module closes.
    """
    parameters = set(inspect.signature(DiskReplayBuffer.__init__).parameters) - {"self"}
    classified = (
        set(CONFIG_KWARGS) | set(DERIVED_KWARGS)
        | set(PRODUCTION_DEFAULTED_KWARGS) | set(PER_RUN_KWARGS)
    )
    unclassified = sorted(parameters - classified)
    stale = sorted(classified - parameters)
    problems: list[str] = []
    if unclassified:
        problems.append(
            "DiskReplayBuffer gained parameter(s) no class in "
            f"lc0_control_replay covers: {', '.join(unclassified)}. Until they "
            "are classified the control takes the constructor default for each "
            "one and no pin can see it.",
        )
    if stale:
        problems.append(
            f"lc0_control_replay classifies parameter(s) DiskReplayBuffer no "
            f"longer has: {', '.join(stale)}.",
        )
    if problems:
        raise ControlReplayDrift(" ".join(problems))


def _buffer_defaults() -> dict[str, Any]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(
            DiskReplayBuffer.__init__,
        ).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def replay_kwargs_signature(flat_config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``DiskReplayBuffer`` kwargs PRODUCTION would construct from a config.

    Excludes ``PER_RUN_KWARGS``. Mirrors ``tune/trainable_init.py``'s call site,
    which is the only definition of "production's buffer" there is.
    """
    assert_buffer_kwargs_are_classified()
    from chess_anti_engine.encoding import input_plane_count
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict(dict(flat_config))
    defaults = _buffer_defaults()
    signature: dict[str, Any] = {
        kwarg: getattr(tc, attribute) for kwarg, attribute in CONFIG_KWARGS.items()
    }
    signature["input_planes"] = input_plane_count(tc.input_extra_features)
    for kwarg in PRODUCTION_DEFAULTED_KWARGS:
        signature[kwarg] = defaults[kwarg]
    return dict(sorted(signature.items()))


# ── the recorded live replay buffer ──────────────────────────────────────────
#
# Measured 2026-08-17 by running `replay_kwargs_signature` over the LIVE working
# tree's `configs/pbt2_small.yaml`.
# ⚑ Regenerate with `scripts/lc0_control_arch_pin.py --axis replay`; do NOT
# hand-edit an entry to make a test pass.
LIVE_REPLAY_PIN: dict[str, Any] = {
    "recorded": "2026-08-17",
    "live_branch": "ops/live-20260725",
    "live_commit": "3eb0448a3",
    "provenance": (
        "shuffle_buffer_size 100000 / refresh 4x5 (raised from 25000 / 1 on "
        "2026-07-27 for the 1073 ms -> 10 ms per-batch disk cost); shard_size "
        "2000; diff_focus 3.5/6.0; recency 1.0; v2_threats -> 175 planes"
    ),
    "kwargs": {
        "deterministic_refresh": False,
        "diff_focus_pol_scale": 3.5,
        "diff_focus_q_weight": 6.0,
        "draw_cap_frac": 0.9,
        "fast_low_surprise_priority": 1.0,
        "input_planes": 175,
        "refresh_interval": 4,
        "refresh_shards": 5,
        "sf_gap_priority_signed": False,
        "sf_gap_priority_weight": 0.0,
        "shard_recency_exponent": 1.0,
        "shard_size": 2000,
        "shuffle_cap": 100000,
        "upgrade_v1_planes": False,
        "wl_max_ratio": 1.5,
    },
}

# ── the ONLY buffer kwargs the control may differ on, and WHY ────────────────
#
# ⚑ A mapping, not a set — same rule as `LC0_TRAINER_DEVIATIONS`: an allow-list
# entry without a stated reason is how a diff test decays into a rubber stamp.
#
# ⚑⚑ BOTH ENTRIES ARE ABOUT THE CORPUS BEING FIXED, NOT ABOUT THE STACK. Neither
# touches how much data is in the hot pool, how often it is refreshed, or how it
# is prioritised — the axes the arm's hypothesis is actually about. That is the
# test a new entry here has to pass.
LC0_REPLAY_DEVIATIONS: dict[str, str] = {
    "shard_recency_exponent": (
        "LIVE 1.0 -> 0.0, i.e. a UNIFORM shard draw. Production weights "
        "newest-first because fresh selfplay is the scarce thing; this arm's "
        "corpus is a fixed set of converted lc0 hours where every shard is "
        "equally old, so a recency weight would over-train whichever hour was "
        "converted last and call it an epoch."
    ),
    "deterministic_refresh": (
        "LIVE False -> True. Production's default lets a background prefetch "
        "thread race the synchronous path, and the two consume different "
        "generators, so the same seed gives different data. An offline ruler "
        "whose deciding statistic is a paired difference between two "
        "checkpoints needs the draw to be a pure function of the seed."
    ),
}


def replay_kwargs_drift(
    control: Mapping[str, Any], reference: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Every kwarg where the two signatures disagree, as ``{key: (ctrl, live)}``."""
    return {
        key: (control.get(key, "<absent>"), reference.get(key, "<absent>"))
        for key in sorted(set(control) | set(reference))
        if control.get(key, "<absent>") != reference.get(key, "<absent>")
    }


def _reference_signature(
    live_config: Path | None, pinned: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """The buffer signature to judge against, and where it came from."""
    if live_config is None:
        origin = (
            "the recorded replay pin"
            if pinned is LIVE_REPLAY_PIN else "a CALLER-SUPPLIED replay pin"
        )
        return dict(pinned["kwargs"]), (
            f"{origin} ({pinned.get('recorded', '<undated>')}, "
            f"{pinned.get('live_branch', '<no branch>')} "
            f"{pinned.get('live_commit', '<no commit>')}) — no live config was "
            f"given, so {LIVE_FILE_UNREAD}"
        )
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    flat = flatten_run_config_defaults(load_yaml_file(str(live_config)))
    return replay_kwargs_signature(flat), f"the LIVE file {live_config}"


def assert_control_matches_live_replay(
    control_flat_config: Mapping[str, Any],
    *,
    live_config: Path | None = None,
    pin: Mapping[str, Any] | None = None,
    allowed: Mapping[str, str] = LC0_REPLAY_DEVIATIONS,
    context: str = "",
) -> str:
    """Raise unless the control samples from PRODUCTION'S buffer, minus ``allowed``.

    Returns the provenance string of whatever it judged against.

    ⚑⚑ THIS FUNCTION DOES **NOT** IMPLEMENT THE TRAINER GUARD'S SECOND HALF,
    AND IT MUST NOT. ``assert_control_matches_live_trainer`` additionally fails
    an ALLOWED key that does not differ, because there a deviation production
    has since adopted is a stale waiver. Here that check would fail EVERY
    legitimate run: the deviations are applied by the DRIVER *after* this call,
    so what is compared is the CONFIG's buffer signature against live's, and
    the two declared deviations are supposed to read identical at this point —
    the control config carries production's value for them precisely so the
    driver's override is visible as an override.

    ⚑ This paragraph used to claim the opposite ("also a failure, exactly as in
    assert_control_matches_live_trainer"), directly contradicted by the
    paragraph beneath it, in a module whose subject is values that are accepted
    and then silently ignored. Independent review of #438 demonstrated the
    divergence by execution: the same manipulation passes the replay guard
    silently and raises in the trainer sibling. The sentence is corrected
    rather than the code, because the code is right and the asymmetry is real.

    The stale-waiver direction IS covered, one level over and at the right
    granularity: ``test_the_driver_deviations_are_exactly_the_declared_ones``
    asserts that the set ``apply_control_deviations`` actually CHANGES equals
    ``LC0_REPLAY_DEVIATIONS`` exactly — both directions, against the pin, on
    the post-override signature where "does not differ" is meaningful.
    """
    pinned = dict(pin) if pin is not None else LIVE_REPLAY_PIN
    reference, provenance = _reference_signature(live_config, pinned)
    if not reference:
        raise ControlReplayDrift(
            f"the reference replay signature is EMPTY (judged against "
            f"{provenance}){f' [{context}]' if context else ''}. Every "
            "comparison against it is vacuous.",
        )
    control = replay_kwargs_signature(control_flat_config)
    drift = replay_kwargs_drift(control, reference)
    problems = [
        f"{key}: control={ctrl!r} live={live!r}"
        for key, (ctrl, live) in drift.items()
        if key not in allowed
    ]
    if not problems:
        return provenance
    where = f" [{context}]" if context else ""
    raise ControlReplayDrift(
        f"the lc0 control's REPLAY BUFFER is NOT production's{where}. Judged "
        f"against {provenance}. Problems:\n  " + "\n  ".join(problems) + "\n"
        "⚑ The arm's hypothesis is H_stack — 'the plateau is the training "
        "stack'. A hot shuffle pool 5x smaller than production's, or a refresh "
        "cadence that is not production's, produces a plateau of the RIG that "
        "is indistinguishable in the held-out slope from the plateau the arm "
        "is looking for. Deviations belong in `LC0_REPLAY_DEVIATIONS` WITH A "
        "REASON. If production moved, regenerate the pin with "
        "`scripts/lc0_control_arch_pin.py --axis replay` and decide whether "
        "the arm follows.",
    )


def apply_control_deviations(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Production's buffer signature with this arm's two declared overrides.

    ⚑ ONE SOURCE FOR THE GUARD AND THE CONSTRUCTION. Review F6 on this arm was
    exactly "guard 0c certifies `kwargs`, and THEN the driver overwrites two of
    them from the CLI": a certified-then-changed recipe with nothing in the
    artifact saying so. So the driver does not hand-write buffer kwargs next to
    a check of different ones — it calls this, and every override it applies is
    a key of ``LC0_REPLAY_DEVIATIONS``, which is the mapping the guard reads.
    """
    overrides: dict[str, Any] = {
        "shard_recency_exponent": 0.0,
        "deterministic_refresh": True,
    }
    unexplained = sorted(set(overrides) - set(LC0_REPLAY_DEVIATIONS))
    if unexplained:
        raise ControlReplayDrift(
            f"the driver would override buffer kwarg(s) {', '.join(unexplained)} "
            "with no entry in LC0_REPLAY_DEVIATIONS. An override the guard does "
            "not know about is a silent deviation.",
        )
    return {**dict(signature), **overrides}
