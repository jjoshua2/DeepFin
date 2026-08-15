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
* the added dump keys do not break `paired_compare.py`, the one tool that reads
  these dumps for provenance, and are deliberately not ruler fields there.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest
from scripts import audit_targets as at
from scripts import paired_compare as pc

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

# Distinct from every CLI value above, so a profile column and a flag can never
# be confused for one another.
FLAT: dict[str, object] = {
    "gumbel_c_scale": 0.1,
    "gumbel_topk": 16,
    "mcts_simulations": 100,
    "fast_simulations": 32,
}


def _args(
    overrides: dict[str, float] | None = None,
    *,
    gumbel: list[str] | None = None,
    gumbel_topk: int | None = None,
    gumbel_training_rows: bool = False,
    rl_sims: int = 0,
) -> argparse.Namespace:
    """A stub `args` carrying what the stamp and the profile builder read."""
    values = dict(CLI_VALUES)
    values.update(overrides or {})
    return argparse.Namespace(
        **values,
        rl_sims=rl_sims,
        gumbel=gumbel,
        gumbel_topk=gumbel_topk,
        gumbel_training_rows=gumbel_training_rows,
    )


def _stamp(args: argparse.Namespace) -> dict[str, float]:
    """Stamp built the way `main()` builds it: off the real profiles."""
    profiles, _ = at.profiles_for_audit(args, FLAT)
    return at.search_param_stamp(args, profiles=profiles)


# --------------------------------------------------------------------------
# the stamp tracks the arguments
# --------------------------------------------------------------------------


def test_stamp_records_every_declared_field() -> None:
    stamp = _stamp(_args())
    assert tuple(stamp) == at.SEARCH_PARAM_FIELDS
    assert stamp == {
        **CLI_VALUES,
        "rl_sims": 100,       # FLAT's mcts_simulations, via the --rl-sims sentinel
        "play_topk": 32,      # PLAY_SEARCH_DEFAULTS, via --gumbel-topk unset
        "rl_topk": 16,        # FLAT's gumbel_topk
    }


@pytest.mark.parametrize("field", sorted(CLI_VALUES))
def test_each_field_follows_its_own_cli_argument(field: str) -> None:
    """Moving one argument moves that field and NOTHING else.

    Presence is not the property under test: a stamp built from constants, or
    one that read `args.batch_size` where it meant `args.sims`, would satisfy a
    key-presence assertion and fail here.
    """
    moved = CLI_VALUES[field] + 41
    stamp = _stamp(_args({field: moved}))
    baseline = _stamp(_args())
    assert stamp[field] == moved
    for other in at.SEARCH_PARAM_FIELDS:
        if other != field:
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


@pytest.mark.parametrize(
    ("spec", "key", "expected"),
    [
        ("simulations=300", "sims", 300),
        ("topk=8", "play_topk", 8),
        ("policy_temp=2.2", "policy_temp", 2.2),
    ],
    ids=["simulations", "topk", "policy_temp"],
)
def test_a_gumbel_override_rewrites_the_stamp_not_just_the_search(
    spec: str, key: str, expected: float,
) -> None:
    """`--gumbel k=v` is applied by `dataclasses.replace` on the BUILT config.

    So it lands AFTER the profile's own columns: `--gumbel simulations=300`
    searches at 300 while `profile.sims` still reads 33. A stamp echoing the
    column would print a number nothing was searched at — provenance that is
    false, which is worse than provenance that is missing.
    """
    stamp = _stamp(_args(gumbel=[spec]))
    assert stamp[key] == pytest.approx(expected)
    # the flag it overrides is genuinely different, or this proves nothing
    assert _stamp(_args())[key] != pytest.approx(expected)


def test_a_gumbel_override_reaches_the_training_rows_only_under_the_flag() -> None:
    """`--gumbel` is PLAY-only unless `--gumbel-training-rows` is passed."""
    play_only = _stamp(_args(gumbel=["simulations=300"]))
    assert play_only["sims"] == 300
    assert play_only["rl_sims"] == 100

    both = _stamp(_args(gumbel=["simulations=300"], gumbel_training_rows=True))
    assert both["sims"] == 300
    assert both["rl_sims"] == 300


def test_last_write_wins_matches_how_build_resolves_a_repeated_key() -> None:
    """`_build` collects the overrides into a dict comprehension."""
    assert _stamp(_args(gumbel=["simulations=300", "simulations=500"]))["sims"] == 500


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
        "cand": {"raw": {"exp": 12.0}},
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
    assert not {"vloss_weight", "vloss_mode", "target_batch"} & set(pc.RULER_FIELDS)
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
