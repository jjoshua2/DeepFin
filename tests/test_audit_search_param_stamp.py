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
* `rl_sims` is the RESOLVED sim count, never the `0` sentinel;
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
CLI_VALUES: dict[str, int] = {
    "vloss_weight": 1,
    "vloss_mode": 0,
    "target_batch": 1024,
    "batch_size": 7,
    "sims": 33,
}
RESOLVED_RL_SIMS = 111


def _args(**overrides: int) -> argparse.Namespace:
    """A stub `args` carrying only what the stamp reads."""
    values = dict(CLI_VALUES)
    values.update(overrides)
    return argparse.Namespace(**values)


# --------------------------------------------------------------------------
# the stamp tracks the arguments
# --------------------------------------------------------------------------


def test_stamp_records_every_declared_field() -> None:
    stamp = at.search_param_stamp(_args(), rl_sims=RESOLVED_RL_SIMS)
    assert tuple(stamp) == at.SEARCH_PARAM_FIELDS
    assert stamp == {**CLI_VALUES, "rl_sims": RESOLVED_RL_SIMS}


@pytest.mark.parametrize("field", sorted(CLI_VALUES))
def test_each_field_follows_its_own_cli_argument(field: str) -> None:
    """Moving one argument moves that field and NOTHING else.

    Presence is not the property under test: a stamp built from constants, or
    one that read `args.batch_size` where it meant `args.sims`, would satisfy a
    key-presence assertion and fail here.
    """
    moved = CLI_VALUES[field] + 41
    stamp = at.search_param_stamp(_args(**{field: moved}), rl_sims=RESOLVED_RL_SIMS)
    baseline = at.search_param_stamp(_args(), rl_sims=RESOLVED_RL_SIMS)
    assert stamp[field] == moved
    for other in at.SEARCH_PARAM_FIELDS:
        if other != field:
            assert stamp[other] == baseline[other], (
                f"moving --{field.replace('_', '-')} also moved {other}"
            )


def test_rl_sims_is_the_resolved_budget_not_the_argument() -> None:
    """`--rl-sims 0` means "use the config"; stamping 0 would be false provenance."""
    stamp = at.search_param_stamp(_args(), rl_sims=RESOLVED_RL_SIMS)
    assert stamp["rl_sims"] == RESOLVED_RL_SIMS


# --------------------------------------------------------------------------
# the resolved value the stamp is handed is the one the search ran
# --------------------------------------------------------------------------


def _profile_args(rl_sims: int) -> argparse.Namespace:
    return argparse.Namespace(
        **CLI_VALUES,
        rl_sims=rl_sims,
        gumbel=None,
        gumbel_topk=None,
        gumbel_training_rows=False,
    )


FLAT: dict[str, object] = {
    "gumbel_c_scale": 0.1,
    "gumbel_topk": 16,
    "mcts_simulations": 100,
    "fast_simulations": 32,
}


@pytest.mark.parametrize(
    ("rl_sims_arg", "expected"),
    [(0, 100), (400, 400)],
    ids=["sentinel-uses-the-config", "override-wins"],
)
def test_stamped_rl_sims_equals_the_profile_the_training_rows_are_searched_at(
    rl_sims_arg: int, expected: int,
) -> None:
    """The stamp must equal `profiles["train"].sims`, which IS the search."""
    args = _profile_args(rl_sims_arg)
    profiles, _ = at.profiles_for_audit(args, FLAT)
    stamp = at.search_param_stamp(args, rl_sims=profiles["train"].sims)
    assert profiles["train"].sims == expected
    assert stamp["rl_sims"] == expected


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_header_line_carries_every_field_and_its_value() -> None:
    stamp = at.search_param_stamp(_args(), rl_sims=RESOLVED_RL_SIMS)
    line = at.format_search_params(stamp)
    for key, value in stamp.items():
        assert f"{key}={value}" in line
    # one token per field, so the line stays greppable
    assert len(line.split()) == len(at.SEARCH_PARAM_FIELDS)


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
    assert "format_search_params(search_params)" in ast.unparse(assigns[0].value), (
        "the written report's header does not render the search-parameter stamp"
    )


# --------------------------------------------------------------------------
# the existing reader keeps parsing
# --------------------------------------------------------------------------


def _dump_row(**stamp: int) -> dict[str, object]:
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
    row = _dump_row(**at.search_param_stamp(_args(), rl_sims=RESOLVED_RL_SIMS))
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
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
    assert not set(at.SEARCH_PARAM_FIELDS[:3]) & set(pc.RULER_FIELDS)
    paths = []
    for label, target_batch in (("a", 0), ("b", 1024)):
        p = tmp_path / f"dump_{label}.jsonl"
        stamp = at.search_param_stamp(
            _args(target_batch=target_batch), rl_sims=RESOLVED_RL_SIMS,
        )
        p.write_text(json.dumps(_dump_row(**stamp)) + "\n", encoding="utf-8")
        paths.append(p)
    dumps = [
        pc.load_dump(str(p), join_key="key", field="cand.raw.exp") for p in paths
    ]
    pc.require_same_ruler(dumps[0], dumps[1], label_a="a", label_b="b")
