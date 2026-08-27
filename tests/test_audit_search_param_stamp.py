"""`audit_targets.py` must record the SEARCH its numbers came from.

`--vloss-weight`, `--vloss-mode` and `--target-batch` have no `--config`
source: the script never reads `gumbel_vloss_weight` / `gumbel_target_batch`
from the yaml, so nothing outside the command line pins them. Until 2026-08-15
neither the report header nor the per-position dump recorded any of the three,
which left every banked report from after `21c21fc4f` (`gumbel_vloss_weight: 1`,
2026-07-29) untraceable to the search that produced it — not known-wrong, but
UNKNOWN, which is the one state a ruler must never be in.

These tests pin the halves that can fail:

* the stamp TRACKS THE ARGUMENTS — every field moves when, and only when, its
  own CLI argument moves. A test that merely asserted the keys were present
  would pass against a hardcoded stamp, which is the defect one level down;
* the sim / topk / temperature entries are the REALIZED values, so the `0`
  sentinel of `--rl-sims`, the PLAY default behind `--gumbel-topk`, and a
  `--gumbel k=v` rewrite of the built config can none of them make the stamp
  disagree with the search;
* both consumers read the SAME stamp object, so the header and the dump cannot
  drift apart or from the run;
* the added dump keys do not break the tools that read these dumps —
  `paired_compare.py`, `audit_compare_buckets.py` and `tail_stats.py` — and are
  deliberately not ruler fields in `paired_compare`. The first and third are
  exercised here rather than enumerated in prose: an enumeration is a claim, and
  this one was already reported complete while `tail_stats` was missing from it.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest
from scripts import audit_targets as at
from scripts import paired_compare as pc
from scripts import tail_stats

AUDIT_TARGETS_SRC = Path(at.__file__)

# Distinctive values: no two fields share one, so a stamp that read the wrong
# argument produces a wrong VALUE rather than an accidentally-right one.
CLI_VALUES: dict[str, float] = {
    "vloss_weight": 1,
    "vloss_mode": 0,
    "target_batch": 1024,
    "batch_size": 7,
    "sims": 33,
    "policy_temp": 1.5,
}

# The value each flag is MOVED to when the fixture checks that its own stamp
# field follows it. ⚑ Not `value + 41`: that put `policy_temp` at 42.5, which
# `apply_policy_temp` silently swallows as out-of-band — so the fixture asserted
# the stamp recorded a temperature the search would never have applied. A
# fixture that pins a fiction is worse than no fixture, and `--policy-temp` is
# now refused outside the band, so the moved value has to be IN band.
MOVED_VALUES: dict[str, float] = {
    "vloss_weight": 42,
  # `main()` refuses mode 1 before any run, so this combination is unreachable
  # end-to-end. The stamp is still required to be a faithful mapping, and
  # dropping the case would let a hardcoded `"vloss_mode": 0` survive.
    "vloss_mode": 1,
    "target_batch": 1065,
    "batch_size": 48,
    "sims": 74,
    "policy_temp": 2.2,       # inside [POLICY_TEMP_MIN, POLICY_TEMP_MAX]
}

# Distinct from every CLI value above, so a profile column and a flag can never
# be confused for one another.
FLAT_SIMS = 100
FLAT: dict[str, object] = {
    "gumbel_c_scale": 0.1,
    "gumbel_topk": 16,
    "mcts_simulations": FLAT_SIMS,
    "fast_simulations": 24,
}

# What `_stamp(_args())` must produce. Spelled out rather than derived, so a
# change to the stamp has to be restated here deliberately.
DEFAULT_STAMP: dict[str, float | bool] = {
  # Both profiles carry the EXPLICIT flag value here, because `CLI_VALUES` sets
  # one. They diverge only on the None ("inherit production") path, which
  # `test_the_stamp_resolves_the_inherit_production_default` covers — and which
  # this fixture, by always passing a number, structurally cannot reach.
    "play_vloss_weight": 1,
    "rl_vloss_weight": 1,
    "vloss_mode": 0,
    "play_target_batch": 1024,
    "rl_target_batch": 1024,
    "batch_size": 7,
    "sims": 33,                  # --sims
    "rl_sims": 100,              # FLAT mcts_simulations, via the --rl-sims sentinel
    "fast_sims": 24,             # FLAT fast_simulations
    "play_topk": 32,             # PLAY_SEARCH_DEFAULTS, via --gumbel-topk unset
    "rl_topk": 16,               # FLAT gumbel_topk
    "play_policy_temp": 1.5,     # --policy-temp
    "rl_policy_temp": 1.5,       # --policy-temp, untouched by a PLAY-only override
    "gumbel_training_rows": False,
}


def _args(
  # `float | None`: None is the SHIPPED default of --vloss-weight /
  # --target-batch ("inherit production"), so a fixture that cannot express it
  # cannot reach the ordinary invocation.
    overrides: dict[str, float | None] | None = None,
    *,
    gumbel: list[str] | None = None,
    gumbel_topk: int | None = None,
    gumbel_training_rows: bool = False,
    rl_sims: int = 0,
) -> argparse.Namespace:
    """A stub `args` carrying what the stamp and the profile builder read."""
    values: dict[str, float | None] = dict(CLI_VALUES)
    values.update(overrides or {})
    return argparse.Namespace(
        **values,
        rl_sims=rl_sims,
        gumbel=gumbel,
        gumbel_topk=gumbel_topk,
        gumbel_training_rows=gumbel_training_rows,
    )


def _stamp(args: argparse.Namespace) -> dict[str, float | bool]:
    """Stamp built the way `main()` builds it: off the real profiles."""
    profiles, _ = at.profiles_for_audit(args, FLAT)
    return at.search_param_stamp(args, profiles=profiles)


# --------------------------------------------------------------------------
# the stamp tracks the arguments
# --------------------------------------------------------------------------


def test_stamp_records_every_declared_field() -> None:
    stamp = _stamp(_args())
    assert tuple(stamp) == at.SEARCH_PARAM_FIELDS
    assert stamp == DEFAULT_STAMP


def test_the_stamp_resolves_the_inherit_production_default() -> None:
    """REGRESSION: `--vloss-weight` / `--target-batch` unset is the NORMAL case.

    ⚑ Every other fixture in this module passes an explicit number for both,
    so none of them can reach the shipped default — which is `None`, meaning
    "inherit whatever the resolved production config sets". A stamp built with
    `int(args.vloss_weight)` therefore raised `TypeError` on every ordinary
    invocation while the whole suite stayed green: the fixture, not the
    assertion, was what failed to discriminate.

    And it is not enough for the stamp to merely BUILD. `build_search_profiles`
    resolves the None asymmetrically on purpose — RL rows inherit production,
    the PLAY row stays at 0 because row (b) is a standing ruler — so this pins
    the two columns to DIFFERENT values. Coercing the None to 0 would build
    fine, satisfy a key-presence check, and stamp 0 onto RL rows that searched
    production's value.
    """
  # ⚑ Its OWN flat, not the shared `FLAT`. That stub declares no
  # `gumbel_vloss_weight` / `gumbel_target_batch`, so production resolves both
  # to 0 -- and against a 0 production value a per-profile stamp is
  # indistinguishable from a single shared one. Values chosen distinct from 0
  # AND from each other so a column that read the wrong one is a wrong VALUE.
    flat = dict(FLAT) | {"gumbel_vloss_weight": 1, "gumbel_target_batch": 96}
    args = _args({"vloss_weight": None, "target_batch": None})
    profiles, _ = at.profiles_for_audit(args, flat)
    stamp = at.search_param_stamp(args, profiles=profiles)

    assert tuple(stamp) == at.SEARCH_PARAM_FIELDS

  # Production's own builder, called the way `build_search_profiles`'s `_rl`
  # calls it -- `production_search_shape`, not the profile the stamp was made
  # from, or the assertion would be checking the stamp against itself.
    from chess_anti_engine.eval.production_shape import production_search_shape

    prod = production_search_shape(flat, simulations=FLAT_SIMS)
    assert stamp["rl_vloss_weight"] == int(prod.vloss_weight) == 1
    assert stamp["rl_target_batch"] == int(prod.target_batch) == 96
    assert stamp["play_vloss_weight"] == 0
    assert stamp["play_target_batch"] == 0

    # Non-vacuity: the split is only a real check while the two disagree.
    assert stamp["rl_vloss_weight"] != stamp["play_vloss_weight"], (
        "production's gumbel_vloss_weight is 0, so this test cannot tell a "
        "per-profile stamp from a single shared one -- do not read it as a pass"
    )


# Which stamp field(s) each CLI flag is allowed to move. `--policy-temp` feeds
# BOTH profiles, because it is applied to every built GumbelConfig; a
# `--gumbel policy_temp=` override is the PLAY-only one.
MOVES: dict[str, tuple[str, ...]] = {
  # An EXPLICIT --vloss-weight / --target-batch feeds both profiles, so both
  # columns follow it. Only the unset (None) default splits them.
    "vloss_weight": ("play_vloss_weight", "rl_vloss_weight"),
    "vloss_mode": ("vloss_mode",),
    "target_batch": ("play_target_batch", "rl_target_batch"),
    "batch_size": ("batch_size",),
    "sims": ("sims",),
    "policy_temp": ("play_policy_temp", "rl_policy_temp"),
}


@pytest.mark.parametrize("field", sorted(CLI_VALUES))
def test_each_field_follows_its_own_cli_argument(field: str) -> None:
    """Moving one argument moves its own stamp field(s) and NOTHING else.

    Presence is not the property under test: a stamp built from constants, or
    one that read `args.batch_size` where it meant `args.sims`, would satisfy a
    key-presence assertion and fail here.
    """
    moved = MOVED_VALUES[field]
    assert moved != CLI_VALUES[field], "a vacuous move proves nothing"
    stamp = _stamp(_args({field: moved}))
    baseline = _stamp(_args())
    for target in MOVES[field]:
        assert stamp[target] == moved
    for other in at.SEARCH_PARAM_FIELDS:
        if other not in MOVES[field]:
            assert stamp[other] == baseline[other], (
                f"moving --{field.replace('_', '-')} also moved {other}"
            )


# --------------------------------------------------------------------------
# REALIZED, not requested: the three ways a flag differs from the search
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rl_sims_arg", "expected"),
    [(0, 100), (400, 400)],
    ids=["sentinel-uses-the-config", "override-wins"],
)
def test_stamped_rl_sims_equals_the_profile_the_training_rows_are_searched_at(
    rl_sims_arg: int, expected: int,
) -> None:
    """`--rl-sims 0` means "use the config"; stamping 0 would be false provenance."""
    args = _args(rl_sims=rl_sims_arg)
    profiles, _ = at.profiles_for_audit(args, FLAT)
    assert profiles["train"].sims == expected
    assert at.search_param_stamp(args, profiles=profiles)["rl_sims"] == expected


def test_stamped_play_topk_is_the_resolved_play_default_when_the_flag_is_absent() -> None:
    """`--gumbel-topk` defaults to the PLAY table, not to a literal."""
    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS

    args = _args()
    profiles, _ = at.profiles_for_audit(args, FLAT)
    stamp = at.search_param_stamp(args, profiles=profiles)
    assert stamp["play_topk"] == int(PLAY_SEARCH_DEFAULTS["topk"])
    assert _stamp(_args(gumbel_topk=9))["play_topk"] == 9
    # and the TRAINING topk is the config's, unmoved by the PLAY flag
    assert stamp["rl_topk"] == FLAT["gumbel_topk"]
    assert _stamp(_args(gumbel_topk=9))["rl_topk"] == FLAT["gumbel_topk"]


# (spec, PLAY-side stamp key, RL-side stamp key, value). Every knob a
# `--gumbel` override can move has BOTH, because the override is PLAY-only
# unless `--gumbel-training-rows` is passed — one unqualified column would be
# false for whichever rows the override missed.
OVERRIDE_ARMS = [
    ("simulations=300", "sims", "rl_sims", 300.0),
    ("topk=8", "play_topk", "rl_topk", 8.0),
    ("policy_temp=2.2", "play_policy_temp", "rl_policy_temp", 2.2),
]


@pytest.mark.parametrize(
    ("spec", "play_key", "expected"),
    [(spec, play_key, value) for spec, play_key, _rl, value in OVERRIDE_ARMS],
    ids=["simulations", "topk", "policy_temp"],
)
def test_a_gumbel_override_rewrites_the_stamp_not_just_the_search(
    spec: str, play_key: str, expected: float,
) -> None:
    """`--gumbel k=v` is applied by `dataclasses.replace` on the BUILT config.

    So it lands AFTER the profile's own columns: `--gumbel simulations=300`
    searches at 300 while `profile.sims` still reads 33. A stamp echoing the
    column would print a number nothing was searched at — provenance that is
    false, which is worse than provenance that is missing.
    """
    stamp = _stamp(_args(gumbel=[spec]))
    assert stamp[play_key] == pytest.approx(expected)
    # the flag it overrides is genuinely different, or this proves nothing
    assert _stamp(_args())[play_key] != pytest.approx(expected)


@pytest.mark.parametrize(
    ("spec", "play_key", "rl_key", "expected"),
    OVERRIDE_ARMS,
    ids=["simulations", "topk", "policy_temp"],
)
def test_a_play_only_override_must_not_move_the_training_side_of_the_stamp(
    spec: str, play_key: str, rl_key: str, expected: float,
) -> None:
    """⚑ THE FINDING THIS SPLIT EXISTS FOR — B1, PR #434.

    `policy_temp` shipped as ONE unqualified column read off the PLAY profile.
    Two 6-position audits differing only in `--gumbel-training-rows` then came
    out with BYTE-IDENTICAL provenance in every field of both the report and the
    dump, while `cand.train` — the production training-target row — genuinely
    differed: `paired_compare --field cand.train.exp` read −9.66 cp
    [−18.07, −3.00]. The stamp said "same search" about a significant delta.

    Worse, the first version of the test above PINNED that behaviour: mutating
    the stamp to read the `train` profile was KILLED by it. So this is the
    assertion that has to exist alongside it — the PLAY side moves, the RL side
    does NOT, and no single column can satisfy both.
    """
    play_only = _stamp(_args(gumbel=[spec]))
    baseline = _stamp(_args())
    assert play_only[play_key] == pytest.approx(expected)
    assert play_only[rl_key] == pytest.approx(baseline[rl_key])
    assert play_only[rl_key] != pytest.approx(expected)

    both = _stamp(_args(gumbel=[spec], gumbel_training_rows=True))
    assert both[play_key] == pytest.approx(expected)
    assert both[rl_key] == pytest.approx(expected)


def test_two_runs_differing_only_in_the_override_scope_get_different_stamps() -> None:
    """The scope flag closes the hole for overrides with NO dedicated column.

    `sims` / `topk` / `policy_temp` escape it only because they happen to have
    `play_*`/`rl_*` pairs. `--gumbel halving_div=4` does not: with and without
    `--gumbel-training-rows` it serializes the same `gumbel_overrides` and
    reaches different rows. A stamp that cannot tell those apart is a gate that
    cannot fail.
    """
    spec = ["halving_div=4"]
    play_only = _stamp(_args(gumbel=spec))
    both = _stamp(_args(gumbel=spec, gumbel_training_rows=True))
    assert play_only != both, (
        "two materially different searches carry identical provenance"
    )
    assert play_only["gumbel_training_rows"] is False
    assert both["gumbel_training_rows"] is True
    # and it is the SCOPE that separates them, not a coincidental column
    assert {
        k for k in at.SEARCH_PARAM_FIELDS if play_only[k] != both[k]
    } == {"gumbel_training_rows"}


def test_last_write_wins_matches_how_build_resolves_a_repeated_key() -> None:
    """`_build` collects the overrides into a dict comprehension."""
    assert _stamp(_args(gumbel=["simulations=300", "simulations=500"]))["sims"] == 500


def test_row_e_realized_sims_is_stamped_separately_from_row_d() -> None:
    """`train_fast` is a third profile and the report quotes its sim count.

    Under `--gumbel simulations=17 --gumbel-training-rows` all three rows search
    at 17; without `fast_sims` the header's "+ 32 fast" had no correction
    anywhere in the artifact.
    """
    assert _stamp(_args())["fast_sims"] == FLAT["fast_simulations"]
    both = _stamp(_args(gumbel=["simulations=17"], gumbel_training_rows=True))
    assert (both["sims"], both["rl_sims"], both["fast_sims"]) == (17, 17, 17)
    play_only = _stamp(_args(gumbel=["simulations=17"]))
    assert play_only["fast_sims"] == FLAT["fast_simulations"]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_header_line_carries_every_field_and_its_value() -> None:
    stamp = _stamp(_args())
    line = at.format_search_params(stamp)
    for key, value in stamp.items():
        assert f"{key}={value}" in line
    # one token per field, so the line stays greppable
    assert len(line.split()) == len(at.SEARCH_PARAM_FIELDS)


def test_header_line_carries_the_free_form_gumbel_overrides_too() -> None:
    """The stamp is the FIXED set; anything else `--gumbel` reached rides here."""
    args = _args(gumbel=["halving_div=4"])
    _, overrides = at.profiles_for_audit(args, FLAT)
    line = at.format_search_params(_stamp(args), gumbel_overrides=overrides)
    assert "gumbel_overrides=halving_div=4.0" in line
    assert "gumbel_overrides" not in at.format_search_params(_stamp(args))


# --------------------------------------------------------------------------
# WIRING: both consumers read the one stamp
# --------------------------------------------------------------------------
#
# The report header and the dump record are built deep inside `main()`, which
# needs a checkpoint, an audit set and an hour of Stockfish. The property that
# actually fails when the wiring rots is structural — "is the stamp the ONLY
# source of these values in main()" — so it is checked structurally, the same
# way this repo proves gate reachability by AST rather than by hope.


def _main_body() -> ast.FunctionDef:
    tree = ast.parse(AUDIT_TARGETS_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("scripts/audit_targets.py has no main()")


def test_the_stamp_is_built_exactly_once_in_main() -> None:
    calls = [
        n for n in ast.walk(_main_body())
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "search_param_stamp"
    ]
    assert len(calls) == 1, (
        "a second stamp is a second chance for the header and the dump to "
        "disagree about the same run"
    )


def test_the_per_position_dump_unpacks_the_stamp_and_spells_out_no_field() -> None:
    """The dump must take the values from the stamp, not re-read `args`.

    `batch_size` used to be spelled out here. Leaving it spelled out while the
    header reads the stamp is precisely how the two start disagreeing.
    """
    dumps = [
        n for n in ast.walk(_main_body())
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "per_pos_dump"
    ]
    assert len(dumps) == 1
    record = dumps[0].args[0]
    assert isinstance(record, ast.Dict)
    unpacked = [
        ast.unparse(v) for k, v in zip(record.keys, record.values, strict=True)
        if k is None
    ]
    assert "search_params" in unpacked, (
        "the per-position dump does not unpack the search-parameter stamp"
    )
    literal_keys = {
        k.value for k in record.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert not literal_keys & set(at.SEARCH_PARAM_FIELDS), (
        "a stamped field is also spelled out literally in the dump record; "
        "the two can drift"
    )


def test_the_report_header_renders_the_stamp() -> None:
    """Scoped to the `report` string ITSELF, not to `main()` at large.

    `main()` also prints the stamp to the console, and the console line is not
    what gets banked. A test that searched the whole function body would be
    satisfied by that print while the written report carried nothing — measured:
    that exact mutant SURVIVED the first version of this assertion.
    """
    assigns = [
        n for n in ast.walk(_main_body())
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "report" for t in n.targets)
    ]
    assert len(assigns) == 1, "expected exactly one `report = ...` in main()"
    rendered = ast.unparse(assigns[0].value)
    assert "format_search_params(search_params" in rendered, (
        "the written report's header does not render the search-parameter stamp"
    )
    assert "gumbel_overrides=gumbel_overrides" in rendered, (
        "the written report's header drops the free-form --gumbel overrides"
    )


# --------------------------------------------------------------------------
# the existing reader keeps parsing
# --------------------------------------------------------------------------


def _dump_row(**stamp: float) -> dict[str, object]:
    return {
        "key": "pos1",
        "phase": 0,
        "input_encoding": {"raw": "fen_only", "sf_soft": None},
        # `exp` is what paired_compare joins on here; `top1` is what
        # tail_stats reads (`cand.raw.top1`). Both, so one fixture serves both
        # readers and neither check silently stops covering its tool.
        "cand": {"raw": {"exp": 12.0, "top1": 30.0}},
        **stamp,
    }


def test_paired_compare_still_parses_a_dump_carrying_the_new_keys(
    tmp_path: Path,
) -> None:
    """Extra keys must be inert for the one tool that reads these dumps."""
    path = tmp_path / "dump_new_keys.jsonl"
    path.write_text(json.dumps(_dump_row(**_stamp(_args()))) + "\n", encoding="utf-8")
    dump = pc.load_dump(str(path), join_key="key", field="cand.raw.exp")
    assert dump.rows == {"pos1": (12.0, "endgame")}
    assert dump.unusable == 0


def test_search_params_are_not_ruler_fields_so_two_arms_still_join(
    tmp_path: Path,
) -> None:
    """DELIBERATE: `--target-batch 1024` vs `0` is a banked control.

    Promoting the search params to `RULER_FIELDS` would make
    `require_same_ruler` REFUSE exactly the paired comparison the stamp exists
    to make legible.
    """
  # ⚑ EXACT, not a sample. The first version listed 3 of the 9 fields and the
  # sibling comment in paired_compare.py listed 5 — both went short within one
  # commit of being written. `batch_size` is the one deliberate overlap: it is a
  # genuine ruler (raw policy regret moves ~0.8 cp between 64 and 256).
    assert set(at.SEARCH_PARAM_FIELDS) & set(pc.RULER_FIELDS) == {"batch_size"}
    paths = []
    for label, target_batch in (("a", 0), ("b", 1024)):
        p = tmp_path / f"dump_{label}.jsonl"
        stamp = _stamp(_args({"target_batch": target_batch}))
        p.write_text(json.dumps(_dump_row(**stamp)) + "\n", encoding="utf-8")
        paths.append(p)
    dumps = [
        pc.load_dump(str(p), join_key="key", field="cand.raw.exp") for p in paths
    ]
    pc.require_same_ruler(dumps[0], dumps[1], label_a="a", label_b="b")


# --------------------------------------------------------------------------
# guards that keep the STAMPED value the value the search ran
# --------------------------------------------------------------------------


def _refuse_policy_temp(temp: float) -> None:
    """Drive the band guard the way `main()`'s `--policy-temp` path drives it.

    The per-key `_refuse_dead_override(name, value, where=...)` this test used
    to call is gone: the guard now validates the ASSEMBLED `GumbelConfig`
    (`_refuse_dead_search_cfg`), which is the object `_net_candidates` will
    `dataclasses.replace` the override onto, and which shares
    `validate_gumbel_config`'s bands with every other caller instead of keeping
    a second copy of `0.05 <= T <= 20.0` in the script. Same criterion, same
    band, same `where=` label — only the call shape moved.
    """
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import GumbelConfig

    at._refuse_dead_search_cfg(
        _dc.replace(GumbelConfig(), policy_temp=temp), where="--policy-temp"
    )


@pytest.mark.parametrize("temp", [0.0, 0.01, 42.5, 1e300])
def test_an_out_of_band_policy_temp_is_refused_rather_than_stamped(
    temp: float,
) -> None:
    """`apply_policy_temp` swallows these, so the search runs the DEFAULT prior.

    Before the stamp existed the value was merely missing from the artifact;
    stamping it would print a temperature the audit never applied. The message
    must name `--policy-temp`, not the `--gumbel` override that shares the band,
    or a test could pass on the wrong guard.
    """
    with pytest.raises(SystemExit, match=r"--policy-temp: policy_temp=") as excinfo:
        _refuse_policy_temp(temp)
    assert "--gumbel" not in str(excinfo.value)


def test_the_untempered_prior_and_an_in_band_value_are_both_allowed() -> None:
    """1.0 is an explicit "run the untempered prior", not a dead value."""
    _refuse_policy_temp(1.0)
    _refuse_policy_temp(2.2)


def test_main_refuses_a_negative_vloss_weight() -> None:
    """Reachability: the guard must be CALLED, not merely defined.

    A grep by name cannot show that, so this reads `main()`'s AST — a guard that
    exists and is never invoked is this repo's signature defect.

    ⚑ THE `--policy-temp` HALF OF THIS TEST DELIBERATELY LIVES ELSEWHERE, and it
    is not a coverage loss: `tests/test_gumbel_config_validation.py` drives the
    refusal BY EXECUTION through the real builder, with a positive control and
    on BOTH of its exits (`..._policy_temp_flag_is_refused_like_its_sibling` and
    `..._training_row_guard_is_wired_not_just_present`). That is strictly the
    stronger instrument: `--policy-temp` reaches a `GumbelConfig` only inside
    `build_profile_search_shape`, which was made module-level and public for
    exactly this reason — per its own docstring, "an AST test cannot tell a
    called guard from a dead one". An AST assertion here would restate a weaker
    version of that and would go red on any refactor that moved the call without
    breaking it.

    The `--vloss-weight` half stays AST because that guard sits directly in
    `main()`'s body and refuses before any machinery a test could drive.
    """
    body = ast.unparse(_main_body())
    assert "args.vloss_weight) < 0" in body, (
        "main() never refuses a negative --vloss-weight, which the C runner drops"
    )
  # ⚑ And it must be None-SAFE. `--vloss-weight` defaults to None ("inherit the
  # resolved production config's `gumbel_vloss_weight`"), so a bare
  # `int(args.vloss_weight)` raises TypeError on every ordinary invocation --
  # a guard that crashes the script it protects, which is how this landed.
    assert "args.vloss_weight is not None" in body, (
        "main()'s vloss guard is not None-safe; --vloss-weight defaults to None "
        "and int(None) raises before the refusal can be reached"
    )


def test_the_legacy_search_header_line_is_rendered_from_the_stamp() -> None:
    """B2: the report must not contradict itself one line apart.

    The `- search: PLAY N sims / RL train ...` line predates the stamp and read
    `args.sims` and the PRE-OVERRIDE profile columns, so at
    `--sims 8 --gumbel simulations=17 --gumbel-training-rows` it printed
    "PLAY 8 sims / RL train 8 full + 32 fast" directly above a stamp reading 17
    — and it is the line an operator greps.
    """
    body = ast.unparse(_main_body())
    assert "play_sims_note = search_params['sims']" in body
    assert "rl_sims = search_params['rl_sims']" in body
    assert "rl_fast_sims = search_params['fast_sims']" in body
    assigns = [
        n for n in ast.walk(_main_body())
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "report" for t in n.targets)
    ]
    rendered = ast.unparse(assigns[0].value)
    assert "- search: PLAY {play_sims_note} sims" in rendered
    assert "args.sims" not in rendered, (
        "the report still renders the pre-override --sims somewhere"
    )


def test_tail_stats_still_parses_a_dump_carrying_the_new_keys(
    tmp_path: Path,
) -> None:
    """The third reader of these dumps, checked rather than asserted.

    `scripts/tail_stats.py --raw-top1` reads `cand.raw.top1` off exactly these
    files. It was reported as checked-and-inert while being absent from the
    change entirely — the same defect this PR is about, one level up. A test
    cannot be reported done without being run.
    """
    path = tmp_path / "dump_tail_stats.jsonl"
    path.write_text(json.dumps(_dump_row(**_stamp(_args()))) + "\n", encoding="utf-8")
    rows = tail_stats.load(str(path), "cand.raw.top1")
    assert rows == {"pos1": (30.0, "endgame")}


# The knobs `search_param_stamp`'s docstring declares as KNOWN GAPS. Verified by
# execution while writing them down: two profile sets differing only in the
# config's `gumbel_c_scale` (0.1 -> 0.4) or `volatility_q_scale` (0.0 -> 0.5)
# produce a genuinely different RL search and a byte-identical stamp.
DOCUMENTED_GAPS = (
    "gumbel_c_scale",
    "volatility_q_scale",
    "volatility_fpu",
    "volatility_anchor",
    "syzygy_in_search",
    "syzygy_path",
)


@pytest.mark.parametrize("knob", DOCUMENTED_GAPS)
def test_the_docstring_names_every_knob_it_does_not_cover(knob: str) -> None:
    """A doc that over-claims is how the next person gets fooled.

    An earlier revision opened with "EVERY PROFILE-VARYING KNOB IS STAMPED PER
    PROFILE", which was false — `gumbel_c_scale` and the `volatility_*` family
    are profile-varying, config-sourced and unstamped. Shipping that sentence in
    THIS file would have undone the point of the change, since the defect being
    fixed is a claim in this file that outran the code.

    Pinned in BOTH directions, so it cannot rot either way: the knob must be
    named in the docstring AND must be absent from the stamp. Stamping one
    without deleting it from the gap list fails here, and so does quietly
    dropping it from the list.
    """
    doc = at.search_param_stamp.__doc__ or ""
    assert "KNOWN GAPS" in doc
    assert knob in doc, f"the docstring stopped naming the unstamped {knob}"
    assert knob not in at.SEARCH_PARAM_FIELDS, (
        f"{knob} is stamped now — remove it from the docstring's KNOWN GAPS "
        "list, or the doc is claiming a gap that no longer exists"
    )
