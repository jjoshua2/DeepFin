"""Pin the lc0 positive control to the architecture PRODUCTION IS RUNNING.

⚑⚑ THE ARM'S PREMISE IS "TEST THE STACK WE RUN". A CONFIG TEST THAT DIFFS
AGAINST THE IN-TREE PRODUCTION YAML CANNOT SEE THAT PREMISE BREAK.

``tests/test_lc0_control_config.py`` pins the control's *diff* rather than
pinning absolute numbers, which is the right design. What it may NOT diff the
ARCHITECTURE against is the in-tree ``configs/pbt2_small.yaml``: that file is
whatever the branch happens to carry, and the live run reads the yaml **in the
live working tree**, on the live branch, which moves independently. Measured
2026-08-16, and STILL TRUE after the bt4heads promotion reached ``main``:

    this tree's configs/pbt2_small.yaml    63,084,128 trainable (unique storage)
    the LIVE tree's pbt2_small.yaml        61,444,448 trainable (unique storage)

The live file carries the bt4heads bundle promoted 2026-08-15 —
``aux_policy_head_dim: 128``, ``categorical_head_coupled: true``,
``policy_embedding_mode: linear`` — and ``main``'s copy of the production yaml
carries none of them, because ``main`` has committed nothing to that file since
the live branch diverged. Two of the three touch the POLICY HEAD, the arm's
only yardstick. A control judged against the in-tree copy would be a network
production does not run.

⚑ WHAT CHANGED ON 2026-08-16, AND WHAT DID NOT. PR #439 put all three keys into
``main``'s SCHEMA (``utils/config_yaml.py``), so this tree can now BUILD the
live network — before that it could not, and an unknown ``model:`` key is
CLAUDE.md category (a), fatal at launch. ``configs/lc0_positive_control.yaml``
now carries the three keys and the check below PASSES against the live file.
What did NOT change is ``main``'s in-tree production YAML: it is still stale by
exactly those three keys, so it is still the wrong reference, and every gate
here still judges against the live file or the pin. ``LIVE_ARCH_PIN`` was
RE-MEASURED in this tree from the live file after the merge (it agreed to the
parameter), and ``tests/test_lc0_control_arch.py`` now BUILDS the pinned
architecture rather than asserting that it cannot.

So the architecture claim gets its own instrument, and the instrument prefers
the LIVE FILE over any in-tree copy:

1. ``live_production_config_path()`` reads ``$CHESS_LIVE_PRODUCTION_CONFIG``.
   The repo is public, so the live path is never committed — it is supplied by
   whoever launches the arm. ⚑ The DRIVERS call it and pass the result in;
   ``assert_control_matches_live_architecture`` never reads the environment
   itself, so its verdict does not change with the caller's shell.
2. When that env var names a readable yaml, the check compares the control
   against **that file's** ``model:`` section. This is the only mode in which
   the check has actually seen production.
3. When it does not, the check falls back to ``LIVE_ARCH_PIN`` below — a
   recorded measurement of the live architecture, with the date and the branch
   it was taken on. A stale pin is still infinitely better than the in-tree
   copy, because the pin at least *knows* it is describing a different file.
4. The pinned COUNT is checked against the BUILT net whenever a caller passes
   ``model=``. That is the only check that can see "same ``model:`` keys,
   different code" — which is precisely the state this branch is in.

The parameter count is counted by UNIQUE ``untyped_storage().data_ptr()``.
``sum(v.numel() for v in state_dict().values())`` reads 77,173,088 on the live
net because the 16 ``layer_smolgens.N.gen_weight.weight`` keys are ONE shared
tensor counted 16 times (CLAUDE.md). ``unique_storage_param_count`` is the
counter; ``tests/test_lc0_control_arch.py`` mutates it against a deliberately
tied model so the dedup cannot silently stop happening.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

# ── the recorded live architecture ────────────────────────────────────────────
#
# Measured 2026-08-16 by building the LIVE working tree's
# `configs/pbt2_small.yaml` (`ops/live-20260725`) and counting unique storages.
# ⚑ RE-MEASURED the same day, from THIS tree's code, once PR #439 gave `main`
# the schema to build it — `scripts/lc0_control_arch_pin.py --live-config <live>`
# reproduced every field and both counts exactly. Before that merge the numbers
# could only be taken in the live tree; they are now falsifiable here, and
# `tests/test_lc0_control_arch.py::test_this_tree_builds_the_pinned_live_architecture`
# rebuilds them on every run.
# ⚑ Regenerate with `scripts/lc0_control_arch_pin.py` whenever production's
# architecture moves; do NOT hand-edit a field to make a test pass.
LIVE_ARCH_PIN: dict[str, Any] = {
    "recorded": "2026-08-16",
    "live_branch": "ops/live-20260725",
    "live_commit": "3f3ff00e6",
    "provenance": "bt4heads bundle promoted 2026-08-15, +13.9 Elo [+5.7, +22.1]",
    "trainable_params_unique_storage": 61_444_448,
    "state_dict_numel_sum": 77_173_088,
    "model": {
        "aux_policy_head_dim": 128,
        "categorical_head_coupled": True,
        "embed_dim": 512,
        "ffn_mult": 1.5,
        "ffn_mult_by_layer": [
            1.5, 1.5, 1.5, 1.5, 1.5, 1.555556, 1.65, 1.772222,
            1.894444, 1.666667, 1.638889, 1.905556, 1.783333, 1.794444,
            1.744444, 1.5,
        ],
        "gradient_checkpointing": False,
        "input_extra_features": "v2_threats",
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_pos_encoding": "arc_adapter",
        "kind": "transformer",
        "num_heads": 16,
        "num_layers": 16,
        "policy_embedding_mode": "linear",
        "policy_encoding": "lc0_1858",
        "qkv_projection": "split",
        "smolgen_bias_norm": "center_rms",
        "smolgen_bias_scale": "layer_head",
        "smolgen_mode": "per_layer",
        "smolgen_relation_basis": True,
        "smolgen_relation_coeff_norm": "none",
        "smolgen_relation_norm": "none",
        "smolgen_relation_scale": "none",
        "use_deepnorm": True,
        "use_smolgen": True,
    },
}

LIVE_CONFIG_ENV = "CHESS_LIVE_PRODUCTION_CONFIG"


class ControlArchitectureDrift(RuntimeError):
    """The control would train a network production is not running."""


def live_production_config_path() -> Path | None:
    """The LIVE production yaml, or ``None`` if the operator did not name one.

    ⚑ Never defaults to an in-tree path. Falling back to
    ``configs/pbt2_small.yaml`` is exactly the blindness this module exists to
    remove: the check would pass by comparing a file to itself.
    """
    raw = os.environ.get(LIVE_CONFIG_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ControlArchitectureDrift(
            f"${LIVE_CONFIG_ENV}={raw!r} does not name a readable file. Point it "
            "at the live tree's configs/pbt2_small.yaml, or unset it to fall "
            "back to the recorded pin.",
        )
    return path


def unique_storage_param_count(model: torch.nn.Module) -> int:
    """Trainable parameter count, deduped by UNIQUE STORAGE.

    ⚑ Not ``sum(v.numel() for v in state_dict().values())``. Weight tying gives
    several state_dict keys one storage, and summing the keys counts the shared
    tensor once per key — on the live net that is 77,173,088 against a true
    61,444,448 (CLAUDE.md).
    """
    by_storage: dict[int, int] = {}
    for param in model.parameters():
        if param.requires_grad:
            by_storage.setdefault(param.untyped_storage().data_ptr(), param.numel())
    return sum(by_storage.values())


def model_section(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The raw (un-flattened) ``model:`` section of a run config."""
    section = raw_config.get("model")
    if not isinstance(section, Mapping):
        return {}
    return dict(section)


def model_section_drift(
    control: Mapping[str, Any], reference: Mapping[str, Any],
) -> list[str]:
    """Human-readable drift lines between two raw ``model:`` sections."""
    drift: list[str] = []
    for key in sorted(set(control) | set(reference)):
        left = control.get(key, "<absent>")
        right = reference.get(key, "<absent>")
        if left != right:
            drift.append(f"{key}: control={left!r} live={right!r}")
    return drift


def _reference_model_section(
    live_config: Path | None, pinned: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """The ``model:`` section to judge against, and where it came from.

    ⚑ ``pinned`` is a PARAMETER, not the module global. It was the global until
    2026-08-16, which made ``assert_control_matches_live_architecture``'s
    ``pin=`` argument accepted and then silently ignored — this repo's
    signature defect, inside the module written to close an instance of it (PR
    #438 review F4). The provenance string it returns says which pin it used.
    """
    if live_config is None:
        origin = (
            "the recorded pin"
            if pinned is LIVE_ARCH_PIN else "a CALLER-SUPPLIED pin"
        )
        return dict(pinned["model"]), (
            f"{origin} ({pinned.get('recorded', '<undated>')}, "
            f"{pinned.get('live_branch', '<no branch>')} "
            f"{pinned.get('live_commit', '<no commit>')}) — no live config was "
            "given, so THE LIVE FILE WAS NOT READ"
        )
    from chess_anti_engine.utils import load_yaml_file

    return model_section(load_yaml_file(str(live_config))), f"the LIVE file {live_config}"


def assert_control_matches_live_architecture(
    control_raw_config: Mapping[str, Any],
    *,
    model: torch.nn.Module | None = None,
    live_config: Path | None = None,
    pin: Mapping[str, Any] | None = None,
    context: str = "",
) -> str:
    """Raise unless the control would build PRODUCTION'S architecture.

    ``control_raw_config`` is the RAW yaml mapping (not flattened), because the
    drift that matters here is a key the flattener would reject outright.
    Returns the provenance string of whatever it judged against, so a caller
    can print which instrument fired.

    ⚑ ``live_config=None`` means "judge against ``pin``", and NOTHING here
    reads the environment. It used to call ``live_production_config_path()``
    itself, which overloaded ``None`` into two different meanings — "no live
    config" at the call site and "go read $CHESS_LIVE_PRODUCTION_CONFIG" inside
    — so the same call returned different verdicts in the operator's shell than
    in a test (review F6: one arch test failed whenever the env var was set,
    i.e. in exactly the mode the rig's own docs prescribe). Resolve the env at
    the CALL SITE with ``live_production_config_path()``; both drivers do.

    ⚑ ``model`` is the BUILT net, and it is the only argument that can catch
    "same yaml keys, different code": the pinned parameter count is a property
    of the live TREE's builder, not of the ``model:`` section alone. It is
    checked whenever the reference's own count is known — against the pin, and
    against a live file whose ``model:`` section the pin describes exactly.
    """
    pinned = dict(pin) if pin is not None else LIVE_ARCH_PIN
    reference, provenance = _reference_model_section(live_config, pinned)
    if not reference:
        raise ControlArchitectureDrift(
            f"the reference architecture is EMPTY (judged against {provenance}"
            f"){f' [{context}]' if context else ''}. Every drift comparison "
            "against it is vacuous: an absent `model:` section agrees with any "
            "control, and the parameter crosscheck is off in live-file mode. "
            "Point the check at a config that HAS a `model:` section.",
        )
  # The pin's parameter count describes the pin's own `model:` section. It
  # applies to a live file only when that file's section is byte-identical —
  # otherwise the live tree is building something the pin never measured, and
  # a count taken from the pin would be an invented number.
    expected_params: int | None = None
    if live_config is None or reference == dict(pinned["model"]):
        expected_params = int(pinned["trainable_params_unique_storage"])

    problems = model_section_drift(model_section(control_raw_config), reference)
    if model is not None and expected_params is not None:
        realized = unique_storage_param_count(model)
        if realized != expected_params:
            problems.append(
                f"trainable params (unique storage): control={realized} "
                f"live={expected_params}",
            )
    if not problems:
        return provenance
    where = f" [{context}]" if context else ""
    raise ControlArchitectureDrift(
        f"the lc0 control's architecture is NOT production's{where}. Judged "
        f"against {provenance}. Drift:\n  " + "\n  ".join(problems) + "\n"
        "⚑ The arm's premise is 'our EXACT net and trainer on someone else's "
        "data'. A policy-head difference makes the top-1 yardstick a "
        "measurement of a network we do not run, and no number from such a run "
        "may be quoted as 'our stack'. Fix by re-pointing "
        "configs/lc0_positive_control.yaml at the live model: section and "
        "regenerating the pin with scripts/lc0_control_arch_pin.py. ⚑ A live "
        "model: key this tree's utils/config_yaml.py schema does not know is "
        "CLAUDE.md category (a) and is FATAL AT LAUNCH, so such a key must "
        "reach the schema BEFORE the control config can carry it — that is "
        "what PR #439 did for the bt4heads bundle on 2026-08-16. "
        "--allow-arch-drift downgrades this to a banner for PLUMBING SMOKE "
        "RUNS ONLY.",
    )
