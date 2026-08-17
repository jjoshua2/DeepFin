"""The instruments must prove they measured PRODUCTION, and fail when they did not.

Every test here is written against a MUTANT: the guard is only interesting if
some concrete input makes it go red, so each case constructs that input rather
than asserting the happy path. A guard no mutant kills is a finding, not a
feature.

The mutants used, and what each stands for:

* a live yaml that sets a search key the instrument does not carry — the #227
  defect exactly, and the one that shipped;
* an instrument config whose search key VALUE disagrees with the live yaml —
  the stale-in-tree-config case, which a presence check cannot see;
* a soft target built at a temperature other than the live one — shards
  outliving a config edit;
* a missing live config — which must FAIL CLOSED. This bullet used to read
  "the degradation path, which must stay loud and non-fatal", and that was the
  hole: ``$CHESS_ANTI_ENGINE_LIVE_CONFIG`` is unset by default, so the
  degradation path was the DEFAULT path, and on it the audit reproduced the
  very defect it fixes while printing that the config matched "live".
  ``--allow-stale-config`` is the deliberate escape, and it stamps the dump.
* a guard whose call site is deleted while the guard itself is left intact —
  the mutant that survived 35/35 tests, because every case drove the guard
  directly.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pytest
import yaml

from chess_anti_engine.eval.production_shape import (
    LIVE_CONFIG_ENV,
    RUNNER_ARG_FIELDS,
    FieldDiff,
    assert_matches_production,
    compare_config_values,
    load_live_config,
    production_search_shape,
    production_selfplay_gumbel_config,
    resolve_live_config_path,
    shape_field_diff,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parent.parent
_IN_TREE_CONFIG = _REPO / "configs" / "pbt2_small.yaml"


def _flat() -> dict[str, Any]:
    """The in-tree config, with the three live-only search keys filled in.

    The in-tree ``configs/pbt2_small.yaml`` does not carry
    ``gumbel_policy_temp``, ``gumbel_target_max_visit_cap`` or
    ``gumbel_target_untempered_prior`` at all, while the live yaml sets all
    three. Tests that need a production-SHAPED config therefore have to supply
    them, and a test written against the bare in-tree file would be vacuous:
    the two target knobs would sit at the GumbelConfig defaults, so mutating
    them to those same defaults would produce no diff and the guard would look
    like it passed.
    """
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(_IN_TREE_CONFIG))))
    flat["gumbel_policy_temp"] = 1.5
    flat["gumbel_target_max_visit_cap"] = 5
    flat["gumbel_target_untempered_prior"] = True
    return flat


def _write_config(tmp_path: Path, overrides: dict, *, name: str = "mutant.yaml") -> Path:
    """A copy of the in-tree production config with `overrides` applied.

    Nested keys are applied by searching the mapping for the section that
    already holds them, so a typo'd key name cannot silently become a new
    top-level key that nothing reads — the failure this whole module is about.
    """
    raw = yaml.safe_load(_IN_TREE_CONFIG.read_text(encoding="utf-8"))

    def _apply(node: object, key: str, value: object) -> bool:
        if not isinstance(node, dict):
            return False
        if key in node:
            node[key] = value
            return True
        return any(_apply(v, key, value) for v in node.values())

    for key, value in overrides.items():
        if not _apply(raw, key, value):
          # ⚑ Not an error: several production search keys are ABSENT from the
          # in-tree config precisely because the in-tree copy is stale — the
          # live yaml on the training host sets `gumbel_policy_temp: 1.5` and
          # `gumbel_target_max_visit_cap: 5`, and this file has neither. That
          # is the premise of this whole module, so the helper inserts the key
          # into the section production reads it from rather than refusing.
            raw.setdefault("selfplay", {})[key] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The production shape comes from production's own builder
# ---------------------------------------------------------------------------


def test_production_builder_carries_the_three_target_shaping_keys() -> None:
    """The keys #227 found dropped must reach the GumbelConfig.

    This is the NEGATIVE-CONTROL half of the fix: if these three ever stop
    flowing from the yaml into production's own builder, the audit's new
    "derive it from production" strategy would faithfully derive the wrong
    thing and every guard below would still pass.
    """
    flat = _flat()
    flat["gumbel_policy_temp"] = 1.75
    flat["gumbel_target_max_visit_cap"] = 7
    flat["gumbel_target_untempered_prior"] = True
    cfg = production_selfplay_gumbel_config(flat, simulations=64)
    assert cfg.policy_temp == pytest.approx(1.75)
    assert cfg.target_max_visit_cap == 7
    assert cfg.target_untempered_prior is True
    assert cfg.simulations == 64


def test_the_three_keys_are_not_gumbel_defaults() -> None:
    """Mutating them must MOVE the config, not coincide with the default.

    Without this, the assertions above would pass against a builder that
    ignored the yaml entirely and returned defaults that happened to match.
    """
    base = GumbelConfig()
    assert base.policy_temp != 1.75
    assert base.target_max_visit_cap != 7
    assert base.target_untempered_prior is False


# ---------------------------------------------------------------------------
# MUTANT: a production search key the instrument does not carry
# ---------------------------------------------------------------------------


def test_drifted_field_is_caught() -> None:
    prod = production_search_shape(_flat(), simulations=32)
    realized = dataclasses.replace(
        prod, cfg=dataclasses.replace(prod.cfg, target_max_visit_cap=0),
    )
    diffs = shape_field_diff(realized, prod, exempt={})
    assert [d.field for d in diffs] == ["target_max_visit_cap"]
    with pytest.raises(SystemExit, match="target_max_visit_cap"):
        assert_matches_production(realized, prod, exempt={}, where="unit")


def test_exempting_a_field_suppresses_it_but_only_that_field() -> None:
    """An exemption must be narrow — otherwise it is a gate that cannot fail."""
    prod = production_search_shape(_flat(), simulations=32)
    realized = dataclasses.replace(
        prod,
        cfg=dataclasses.replace(prod.cfg, target_max_visit_cap=0, topk=3),
    )
    diffs = shape_field_diff(
        realized, prod, exempt={"target_max_visit_cap": "unit-test reason"},
    )
    assert [d.field for d in diffs] == ["topk"]


def test_identical_configs_produce_no_diff() -> None:
    """The guard must be capable of PASSING, or it is noise rather than a check."""
    prod = production_search_shape(_flat(), simulations=32)
    assert shape_field_diff(prod, prod, exempt={}) == []
    assert_matches_production(prod, prod, exempt={}, where="unit")


# ---------------------------------------------------------------------------
# MUTANT: a config whose VALUE disagrees with the live one
# ---------------------------------------------------------------------------


def test_value_compare_catches_a_differing_value() -> None:
    got = compare_config_values(
        {"gumbel_c_scale": 0.025}, {"gumbel_c_scale": 0.1}, ("gumbel_c_scale",),
    )
    assert got == [FieldDiff("gumbel_c_scale", 0.025, 0.1)]


def test_value_compare_treats_absence_as_a_difference() -> None:
    """⚑ A presence check is not a value read — and neither is its inverse.

    A key production sets and the instrument's config omits is the drift being
    hunted; skipping it because it is "not there to compare" is how the
    original defect stayed invisible.
    """
    got = compare_config_values({}, {"gumbel_topk": 16}, ("gumbel_topk",))
    assert [d.field for d in got] == ["gumbel_topk"]
    assert got[0].realized == "<absent>"


def test_value_compare_passes_on_equal_values() -> None:
    assert compare_config_values(
        {"gumbel_topk": 16}, {"gumbel_topk": 16}, ("gumbel_topk",),
    ) == []


# ---------------------------------------------------------------------------
# audit_targets: the training rows follow production, the PLAY row does not
# ---------------------------------------------------------------------------


def test_audit_training_rows_track_a_mutated_production_config(tmp_path: Path) -> None:
    """MUTANT: move the three keys in the yaml, watch the profiles move.

    Before the fix these were hardcoded to GumbelConfig's defaults, so this
    test fails on `main` — which is what makes it a test rather than a
    restatement of the code.
    """
    from scripts.audit_targets import build_search_profiles

    path = _write_config(tmp_path, {
        "gumbel_policy_temp": 1.25,
        "gumbel_target_max_visit_cap": 9,
        "gumbel_target_untempered_prior": True,
    })
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(path))))
    profiles = build_search_profiles(flat, play_sims=256, play_topk=None)
    for name in ("train", "train_fast"):
        p = profiles[name]
        assert p.policy_temp == pytest.approx(1.25), name
        assert p.target_max_visit_cap == 9, name
        assert p.target_untempered_prior is True, name
        assert p.production_base is not None, name


def test_audit_training_rows_report_productions_topk_and_c_scale(tmp_path: Path) -> None:
    """The HEADER fields must track production too, not only the search fields.

    ⚑ Added because a mutant survived: replacing ``topk=int(prod.topk)`` with a
    ``flat.get("gumbel_topk", 16)`` hand-list and ``c_scale`` with the literal
    0.1 changed nothing that any test looked at. The built GumbelConfig stayed
    correct (it comes from ``production_base``), so the SEARCH was fine — but
    the profile is what prints the report header and the JSON, so the mutant
    produced a run whose numbers were right and whose stated shape was wrong.
    That is precisely "a number that does not mean what its name says", so the
    reported shape gets a guard of its own.
    """
    from scripts.audit_targets import build_search_profiles

    path = _write_config(tmp_path, {"gumbel_topk": 23, "gumbel_c_scale": 0.077})
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(path))))
    profiles = build_search_profiles(flat, play_sims=256, play_topk=None)
    for name in ("train", "train_fast"):
        assert profiles[name].topk == 23, name
        assert profiles[name].c_scale == pytest.approx(0.077), name


def test_audit_load_config_checks_as_well_as_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading and checking are ONE call, so the check cannot be dropped alone.

    ⚑ Added because a mutant survived: deleting the
    ``_assert_config_is_production(...)`` line from ``main()`` left the load in
    place and no test noticed. The two are now fused in ``load_audit_config``.
    """
    from scripts.audit_targets import load_audit_config

    live = _write_config(tmp_path, {"gumbel_c_scale": 0.1}, name="live.yaml")
    stale = _write_config(tmp_path, {"gumbel_c_scale": 0.025}, name="stale.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    with pytest.raises(SystemExit, match="c_scale"):
        load_audit_config(str(stale), allow_stale=False)
    # And it really does return a usable config, and its VERDICT, on the happy
    # path. The verdict is returned rather than only printed because it has to
    # reach the dump: `--allow-stale-config` stamps every row.
    flat, authority = load_audit_config(str(live), allow_stale=False)
    assert flat["gumbel_c_scale"] == pytest.approx(0.1)
    assert authority.authoritative is True
    assert authority.stamp()["authoritative"] is True


def test_audit_play_row_is_deliberately_not_the_training_shape(tmp_path: Path) -> None:
    """The PLAY row must NOT inherit the target-shaping knobs.

    Guards against over-correcting: making everything match production would
    destroy the PLAY/TRAIN distinction the script exists to draw.
    """
    from scripts.audit_targets import build_search_profiles

    path = _write_config(tmp_path, {"gumbel_target_max_visit_cap": 9})
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(path))))
    play = build_search_profiles(flat, play_sims=256, play_topk=None)["search"]
    assert play.production_base is None
    assert play.target_max_visit_cap == 0


def test_audit_refuses_a_config_that_is_not_the_live_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTANT: audit config X while the live config says Y."""
    from scripts.audit_targets import _assert_config_is_production

    live = _write_config(tmp_path, {"gumbel_c_scale": 0.1}, name="live.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    stale_path = _write_config(tmp_path, {"gumbel_c_scale": 0.025}, name="stale.yaml")
    stale = dict(flatten_run_config_defaults(load_yaml_file(str(stale_path))))
    with pytest.raises(SystemExit, match="c_scale"):
        _assert_config_is_production("mutant", stale, allow_stale=False)
    # ...and the escape hatch must NOT be silent, and must stamp the dump.
    authority = _assert_config_is_production("mutant", stale, allow_stale=True)
    assert "WARNING (--allow-stale-config)" in capsys.readouterr().out
    assert authority.authoritative is False
    assert authority.stamp()["authoritative"] is False
    assert "c_scale" in str(authority.stamp()["reason"])


def test_audit_accepts_the_live_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.audit_targets import _assert_config_is_production

    live = _write_config(tmp_path, {"gumbel_c_scale": 0.1})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    authority = _assert_config_is_production(str(live), flat, allow_stale=False)
    out = capsys.readouterr().out
    assert "matches the LIVE config" in out
    assert "[LIVE]" in out
    assert authority.authoritative is True


class _BuildKwargs(TypedDict):
    """Exactly the checkpoint-derived arguments `build_profile_search_shape` takes.

    A TypedDict rather than `dict[str, Any]` so that unpacking it at a call site
    keeps each field's type. A plain dict collapses them to the union of the
    value types, which then fails to match the typed parameters — and silencing
    that with `Any` would mean this file's own call sites were no longer checked
    against the signature they exist to pin.
    """

    hist: str
    extra: str
    pol_enc: str
    use_rel: bool
    play_policy_temp: float


def _build_kwargs(*, play_policy_temp: float = 1.0) -> _BuildKwargs:
    """The checkpoint-derived arguments, pinned to production's own values.

    Using production's encoding here is deliberate: these four fields are
    EXEMPT from the shape check, so passing foreign values would make the
    exempt map do the work and hide whether the non-exempt comparison fires.
    """
    from chess_anti_engine.eval.production_shape import production_input_encoding

    enc = production_input_encoding(_flat())
    return _BuildKwargs(
        hist=enc["input_history_encoding"],
        extra=enc["input_extra_features"],
        pol_enc="lc0_1858",
        use_rel=False,
        play_policy_temp=play_policy_temp,
    )


def test_built_training_config_is_productions() -> None:
    """The object handed to the SEARCH — not the profile — must be production's.

    The profile could carry the right numbers while `_build` dropped them on
    the floor; that gap is exactly why this builder was lifted out of its
    closure.
    """
    from scripts.audit_targets import build_profile_search_shape, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_search_shape(
        "train", profiles["train"], **_build_kwargs(),
    ).cfg
    assert cfg.policy_temp == pytest.approx(1.5)
    assert cfg.target_max_visit_cap == 5
    assert cfg.target_untempered_prior is True
    assert cfg.add_noise is False  # the one documented deviation


def test_build_actually_compares_the_non_exempt_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: shrink the exempt map and watch a real deviation surface.

    ``build_profile_search_shape`` sets ``add_noise=False`` against
    production's ``True``. That deviation is silent only because "add_noise" is
    in ``TRAIN_SHAPE_DEVIATIONS``; removing the entry must make the guard fire.
    This is the test that proves the guard is CALLED and that it is the exempt
    map — not a vacuous comparison — doing the suppressing.

    ⚑ An earlier draft of this test tampered with ``production_base`` instead.
    That could never fail: the builder derives the realized config FROM
    ``production_base`` and then compares against the same object, so the
    mutation moved both sides of the comparison equally. It is recorded here
    because it is the exact shape of a gate that cannot fail.
    """
    import scripts.audit_targets as at

    profiles = at.build_search_profiles(_flat(), play_sims=256, play_topk=None)
    # Sanity: with the real map it passes.
    at.build_profile_search_shape("train", profiles["train"], **_build_kwargs())
    trimmed = {k: v for k, v in at.TRAIN_SHAPE_DEVIATIONS.items() if k != "add_noise"}
    monkeypatch.setattr(at, "TRAIN_SHAPE_DEVIATIONS", trimmed)
    with pytest.raises(SystemExit, match="add_noise"):
        at.build_profile_search_shape("train", profiles["train"], **_build_kwargs())


def test_the_override_is_applied_BEFORE_the_assertion_and_the_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTANT (R3): move the override block back below the assert and print.

    ``build_profile_search_shape``'s docstring says "⚑ ORDER IS THE POINT" and
    nothing tested it: moving the override block back to where it used to be
    survived 91 tests. The consequence is not loud — the returned shape carries
    3.0 either way, because the override still lands before the return — it is
    that the PRINTED table and (through the caller) the RULER STAMP describe
    the pre-override object. And the table does not print a wrong value, it
    prints NOTHING: with `policy_temp` in the exempt map and got == want == 1.5
    the exempt branch takes its `continue`. So the row that says "this run did
    not score production's target here" silently disappears, on precisely the
    run where it is load-bearing.

    ⚑ BOTH BRANCHES, asserted on the OUTPUT rather than on the object: the
    overridden field prints with the operator's value AND the DELIBERATE
    reason, and an un-overridden neighbour prints production's value with no
    such marking.
    """
    import scripts.audit_targets as at

    profiles = at.build_search_profiles(
        _flat(), play_sims=256, play_topk=None,
        gumbel_overrides=(("policy_temp", 3.0),), override_training_rows=True,
    )
    capsys.readouterr()
    shape = at.build_profile_search_shape(
        "train", profiles["train"], **_build_kwargs(),
    )
    out = capsys.readouterr().out

    assert shape.cfg.policy_temp == pytest.approx(3.0), (
        "the override never reached the runner — this test would be vacuous"
    )
    temp_lines = [ln for ln in out.splitlines() if "policy_temp" in ln]
    assert temp_lines, (
        "the shape table printed NO policy_temp row for a run that overrode it. "
        "The override is being applied AFTER the table is built, so the table "
        "describes a config this run did not search with and the operator's "
        "deviation is invisible. ORDER IS THE POINT."
    )
    assert any("3.0" in ln and "DELIBERATE" in ln for ln in temp_lines), (
        f"policy_temp printed as {temp_lines!r}; expected the operator's 3.0 "
        f"marked DELIBERATE against production's 1.5"
    )
    assert any("--gumbel policy_temp=3.0" in ln for ln in temp_lines)
  # The un-overridden neighbour is the control: it must still print
  # production's value, unmarked, or the table is marking everything.
    cap_lines = [ln for ln in out.splitlines() if "target_max_visit_cap" in ln]
    assert cap_lines, "the table printed no target_max_visit_cap row at all"
    assert "DELIBERATE" not in cap_lines[0]
    assert "5" in cap_lines[0]


def test_build_play_row_still_honours_policy_temp_flag() -> None:
    """--policy-temp must keep working on the PLAY row after the split."""
    from scripts.audit_targets import build_profile_search_shape, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_search_shape(
        "search", profiles["search"], **_build_kwargs(play_policy_temp=2.2),
    ).cfg
    assert cfg.policy_temp == pytest.approx(2.2)


def test_policy_temp_flag_does_not_reach_the_training_rows() -> None:
    """MUTANT-BY-CONSTRUCTION: a PLAY flag leaking into a TARGET row.

    Scoring the "production training target" at an operator's --policy-temp is
    the mislabeling this whole change is about, in miniature.
    """
    from scripts.audit_targets import build_profile_search_shape, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_search_shape(
        "train", profiles["train"], **_build_kwargs(play_policy_temp=2.2),
    ).cfg
    assert cfg.policy_temp == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The degradation path must be loud, and must not be fatal
# ---------------------------------------------------------------------------


def test_missing_live_config_REFUSES_rather_than_degrading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """⚑ THE FAIL-CLOSED CASE, and the one that shipped disarmed.

    The first revision of this guard WARNED here and carried on comparing
    ``--config`` against the in-tree fallback. On this machine
    ``$CHESS_ANTI_ENGINE_LIVE_CONFIG`` is unset and exported nowhere in the
    repo, and ``origin/main``'s ``configs/pbt2_small.yaml`` carries 0 of the 3
    keys the finding is about — so from the worktree CLAUDE.md mandates for
    branch work, the FIXED script reproduced the exact defect it fixes and
    printed "all 10 production search keys match the live config by VALUE".

    A guard disarmed by its own default environment is not a guard, so the
    contract is now: no authoritative reference, no training-target row.
    """
    from scripts.audit_targets import _assert_config_is_production

    monkeypatch.setenv(LIVE_CONFIG_ENV, str(tmp_path / "does-not-exist.yaml"))
    assert load_live_config() is None
    with pytest.raises(SystemExit, match="REFUSING to score a production"):
        _assert_config_is_production("whatever", _flat(), allow_stale=False)


def test_unset_env_REFUSES_even_though_the_in_tree_config_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worktree case: a config resolves, it is just not the LIVE one.

    Distinct from the test above because the failure mode is the opposite of a
    crash — everything loads, everything compares, and the comparison is
    against a file that is stale by construction.
    """
    from scripts.audit_targets import _assert_config_is_production

    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    assert load_live_config() is not None, "the in-tree fallback should load"
    with pytest.raises(SystemExit, match="REFUSING to score a production"):
        _assert_config_is_production(str(_IN_TREE_CONFIG), _flat(), allow_stale=False)


def test_no_live_config_never_says_LIVE_in_the_affirmative(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The reassuring line a reader greps for must not be printable here.

    The shipped wording was "all 10 production search keys match the live
    config by VALUE" on a branch where the code had ALREADY decided the
    reference was not live. Grepping for a phrase is how these reports are
    read, so the phrase itself has to be unreachable on this path.
    """
    from scripts.audit_targets import _assert_config_is_production

    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    authority = _assert_config_is_production(
        str(_IN_TREE_CONFIG), _flat(), allow_stale=True,
    )
    out = capsys.readouterr().out
    assert "WARNING (--allow-stale-config)" in out
    assert "match" not in out.lower().replace("mismatch", "")
    assert "LIVE config" not in out
    assert authority.authoritative is False


def test_allow_stale_config_is_still_a_supported_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Off-host use (foreign nets, historical configs) must remain possible.

    Refusing everywhere would be its own regression — the point is that the
    escape is EXPLICIT and lands on the artifact, not that it is unavailable.
    """
    import scripts.audit_targets as at
    from scripts.audit_targets import _assert_config_is_production

    monkeypatch.setenv(LIVE_CONFIG_ENV, str(tmp_path / "does-not-exist.yaml"))
    authority = _assert_config_is_production(
        "whatever", _flat(), allow_stale=True,
    )
    assert "WARNING (--allow-stale-config)" in capsys.readouterr().out
    assert authority.authoritative is False
    stamp = authority.stamp()
    assert stamp["authoritative"] is False
    assert stamp["reference"] == "<none>"
    assert stamp["reason"] == authority.reason
  # ⚑ The SCOPE of the boolean, banked beside it. A flag whose coverage is
  # not stated is read at the coverage the reader needs — including the LIMIT
  # of the AST scan that keeps `config_keys` complete.
    covers = cast("dict[str, Any]", stamp["covers"])
    assert covers["search_shape"] == "complete selfplay runner argument set"
    assert covers["config_keys"] == list(at.AUDIT_DIRECT_CONFIG_KEYS)
    assert "alias" in covers["scan"], (
        "the stamp must state the scan's limit; a scope claim with no stated "
        "boundary is read at the boundary the reader needs"
    )
  # ⚑ And the VALUES ride even on the degraded path — that is the run most
  # likely to have been built from a non-production target config, so dropping
  # them here would leave paired_compare comparing {} to {}, i.e. EQUAL.
    assert set(authority.config_values()) == set(at.AUDIT_DIRECT_CONFIG_KEYS)
    assert "does not exist" in authority.reason


def test_unset_env_falls_back_but_reports_non_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    path, provenance, authoritative = resolve_live_config_path()
    assert path == _IN_TREE_CONFIG
    assert authoritative is False
    assert LIVE_CONFIG_ENV in provenance


def test_env_set_is_authoritative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(tmp_path / "live.yaml"))
    _, _, authoritative = resolve_live_config_path()
    assert authoritative is True


# ---------------------------------------------------------------------------
# probe_policy_targets: recover the temperature from the DATA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temp", [1.5, 2.0, 3.0])
def test_soft_policy_temp_is_recovered_from_stored_arrays(temp: float) -> None:
    """Recovered against PRODUCTION's own `apply_policy_temperature`.

    A re-implementation of the tempering here would only prove this file is
    self-consistent — the guard must share the criterion's instrument.
    """
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature
    from scripts.probe_policy_targets import _recover_soft_policy_temp

    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.ones(12) * 0.4, size=256).astype(np.float32)
    q = np.stack([apply_policy_temperature(row, temp) for row in p]).astype(np.float32)
    hat = _recover_soft_policy_temp(p, q)
    finite = hat[np.isfinite(hat)]
    assert finite.size == 256
    assert float(np.median(finite)) == pytest.approx(temp, rel=1e-4)


def test_recovered_temp_mismatch_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: shards at T=3.0 while the live config says 2.0."""
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature
    from scripts.probe_policy_targets import (
        _check_soft_policy_temp,
        _recover_soft_policy_temp,
    )

    live = _write_config(tmp_path, {"soft_policy_temp": 2.0})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(12) * 0.4, size=128).astype(np.float32)
    q = np.stack([apply_policy_temperature(row, 3.0) for row in p]).astype(np.float32)
    line = _check_soft_policy_temp(_recover_soft_policy_temp(p, q))
    assert "DOES NOT MATCH" in line
    assert "do NOT come from the live temperature" in line


def test_recovered_temp_match_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same check must be able to PASS, or it certifies nothing."""
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature
    from scripts.probe_policy_targets import (
        _check_soft_policy_temp,
        _recover_soft_policy_temp,
    )

    live = _write_config(tmp_path, {"soft_policy_temp": 2.0})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(12) * 0.4, size=128).astype(np.float32)
    q = np.stack([apply_policy_temperature(row, 2.0) for row in p]).astype(np.float32)
    line = _check_soft_policy_temp(_recover_soft_policy_temp(p, q))
    assert "MATCHES" in line
    assert "DOES NOT MATCH" not in line


def test_near_one_hot_rows_still_recover_the_true_temperature() -> None:
    """Near-degenerate rows must give the right T or none, never a wild one.

    ⚑ This is the honest version of a test that was written to kill a mutant
    and could not. Removing the ``den > 1e-6`` floor survives every test, and
    investigation showed WHY: the floor is inert. The upstream mask drops every
    entry below 1e-6, so a surviving off-reference entry always contributes a
    large log-ratio — measured ``den`` on near-one-hot rows is 130.8 / 364.5 /
    0.0 / 0.0 for entries of 1e-3 / 1e-5 / 1e-7 / 1e-9. There is no
    small-but-positive regime for the floor to catch, so ``den > 0.0`` and
    ``den > 1e-6`` select identically.

    Reported as an equivalent mutant rather than papered over with a test
    contrived to fail. What this test DOES pin is the property that matters:
    on rows this degenerate the estimate is either absent or correct.
    """
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature
    from scripts.probe_policy_targets import _recover_soft_policy_temp

    rows = []
    for eps in (1e-3, 1e-5, 1e-7, 1e-9):
        row = np.full(12, eps, dtype=np.float64)
        row[5] = 1.0 - 11 * eps
        rows.append(row)
    p = np.stack(rows).astype(np.float32)
    q = np.stack([apply_policy_temperature(r, 2.0) for r in p]).astype(np.float32)
    hat = _recover_soft_policy_temp(p, q)
    finite = hat[np.isfinite(hat)]
    assert finite.size == 2, "the 1e-7/1e-9 rows must drop out, the others must not"
    assert np.allclose(finite, 2.0, rtol=1e-6)


def test_one_hot_rows_recover_nothing_and_say_so() -> None:
    """One-hot rows are fixed points of every T; claiming a T from them would
    be a number that does not mean what its name says."""
    from scripts.probe_policy_targets import (
        _check_soft_policy_temp,
        _recover_soft_policy_temp,
    )

    onehot = np.zeros((16, 12), dtype=np.float32)
    onehot[:, 5] = 1.0
    hat = _recover_soft_policy_temp(onehot, onehot)
    assert not np.isfinite(hat).any()
    assert "NOT CHECKED" in _check_soft_policy_temp(hat)


# ---------------------------------------------------------------------------
# arena_standard: the training shape must reproduce production's PLAY behaviour
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# value_regret: the input layout is its only production-shape dependence
# ---------------------------------------------------------------------------


def test_value_regret_warns_on_a_foreign_input_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTANT: score an lc0-layout net while production is v2_threats.

    A WARNING rather than a refusal is deliberate — scoring foreign nets on the
    frozen audit set is a supported, documented use. What must not happen is
    the mismatch going unsaid.
    """
    from scripts.value_regret import _report_encoding_vs_production

    monkeypatch.setenv(LIVE_CONFIG_ENV, str(_write_config(tmp_path, {})))
    _report_encoding_vs_production(
        {"input_history_encoding": "lc0_root", "input_extra_features": "v1"},
        input_encoding="fen_only",
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "input_extra_features" in out
    assert "must not" in out


def test_value_regret_confirms_a_matching_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The same reporter must be able to say MATCHES, or it is pure noise."""
    from chess_anti_engine.eval.production_shape import production_input_encoding
    from scripts.value_regret import _report_encoding_vs_production

    live = _write_config(tmp_path, {})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    _report_encoding_vs_production(
        production_input_encoding(flat), input_encoding="stored",
    )
    out = capsys.readouterr().out
    assert "MATCHES production" in out
    assert "WARNING" not in out


def test_value_regret_never_says_MATCHES_PRODUCTION_off_the_live_tree(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTANT (R4 / Codex P2): the affirmative line on a NON-AUTHORITATIVE ref.

    ⚑ `authoritative`, NOT `live is None`. With $CHESS_ANTI_ENGINE_LIVE_CONFIG
    unset — the DEFAULT in every worktree, which is where CLAUDE.md mandates
    branch work happens — `load_live_config_or_reason` returns the IN-TREE
    config with `authoritative=False`. Branching on None alone therefore
    printed "checkpoint input layout MATCHES production" about a file the
    resolver had already decided was not production. The `[NOT-LIVE]` header
    does not repair it: the RESULT line is what gets copied into a table, and
    it travels without its header.

    Same defect the sibling probe fixed in this same change, left standing in
    `value_regret.py`.
    """
    from chess_anti_engine.eval.production_shape import (
        load_live_config,
        production_input_encoding,
    )
    from scripts.value_regret import _report_encoding_vs_production

    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    live = load_live_config()
    assert live is not None, "the in-tree fallback did not resolve at all"
    assert not live.authoritative, (
        "the in-tree fallback resolved as AUTHORITATIVE — this test would be "
        "vacuous"
    )
    _report_encoding_vs_production(
        production_input_encoding(live.flat), input_encoding="stored",
    )
    out = capsys.readouterr().out
    assert "MATCHES production" not in out, (
        "value_regret claims a PRODUCTION match against the in-tree config, "
        "which is stale by construction outside the live working tree"
    )
    assert "NON-AUTHORITATIVE" in out
    assert "NOT a production check" in out
    assert LIVE_CONFIG_ENV in out


def test_value_regret_marks_a_MISMATCH_non_authoritative_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half: the negative verdict is a claim about production too.

    "This checkpoint's layout is NOT production's" read off the in-tree config
    is exactly as unfounded as the affirmative, and it is the line that would
    make someone re-export a net.
    """
    from scripts.value_regret import _report_encoding_vs_production

    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    _report_encoding_vs_production(
        {"input_history_encoding": "lc0_root", "input_extra_features": "v1"},
        input_encoding="fen_only",
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "NON-AUTHORITATIVE" in out
    assert "NOT a production check" in out


# ---------------------------------------------------------------------------
# arena_standard: the training shape must reproduce production's PLAY behaviour
# ---------------------------------------------------------------------------


def test_arena_training_shape_deviations_are_target_only() -> None:
    """The arena omits exactly the two TARGET-shaping knobs, and says why.

    Pins the claim the code comment makes. If someone adds a move-affecting
    field to the exempt map to silence a failure, this goes red.
    """
    from scripts.arena_standard import ARENA_SHAPE_DEVIATIONS

    assert set(ARENA_SHAPE_DEVIATIONS) == {
        "simulations", "temperature", "add_noise", "gumbel_scale",
        "target_max_visit_cap", "target_untempered_prior",
        "input_history_encoding", "input_extra_features",
        "policy_encoding", "compute_relations",
    }
    assert all(reason.strip() for reason in ARENA_SHAPE_DEVIATIONS.values())


def test_arena_training_shape_carries_move_affecting_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: raise gumbel_policy_temp in the live yaml; the arena must follow."""
    from scripts.arena_standard import _assert_training_shape_is_production

    live = _write_config(tmp_path, {"gumbel_policy_temp": 1.9})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    prod = production_search_shape(flat, simulations=1)
    # The shape the arena would build if it had NOT been updated: stale temp.
    # Every OTHER field is production's, so the refusal names policy_temp and
    # only policy_temp — a test that let several fields diverge would pass on
    # any one of them and could not tell which guard fired.
    with pytest.raises(SystemExit, match="policy_temp") as excinfo:
        _assert_training_shape_is_production({
            "c_scale": float(prod.cfg.c_scale),
            "topk": int(prod.cfg.topk),
            "policy_temp": 1.0,
            "volatility_q_scale": float(prod.cfg.volatility_q_scale),
            "volatility_fpu": float(prod.cfg.volatility_fpu),
            "volatility_anchor": float(prod.cfg.volatility_anchor),
        }, flat, vloss_weight=int(prod.vloss_weight),
           target_batch=int(prod.target_batch))
    assert "vloss_weight" not in str(excinfo.value)


def test_arena_returns_the_dict_it_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: guard one dict, ship another.

    The first draft of this change did exactly that — it built a checked
    ``gumbel`` dict and then returned a stale literal beside it, so the guard
    passed while the arena searched at the old shape. Nothing in the suite
    noticed, because every test drove the guard directly. This drives
    ``resolve_search_shape`` end to end instead.
    """
    from scripts.arena_standard import resolve_search_shape

    live = _write_config(tmp_path, {
        "gumbel_policy_temp": 1.9,
        "gumbel_topk": 24,
      # ⚑ Must differ from GumbelConfig's own default (0.1). A mutant that
      # dropped `c_scale` from the shape dict entirely survived the first
      # version of this test, because the config under test happened to carry
      # the default value — so realized and production agreed by coincidence
      # and the guard had nothing to report. A test config that matches the
      # defaults cannot detect a field going missing.
        "gumbel_c_scale": 0.0375,
    })
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    side = resolve_search_shape("training")
    assert side.gumbel["policy_temp"] == pytest.approx(1.9)
    assert side.gumbel["topk"] == 24
    assert side.gumbel["c_scale"] == pytest.approx(0.0375)

  # ⚑ AND THE GUARD IS ACTUALLY CALLED FROM HERE. Every other case in this
  # file drives `_assert_training_shape_is_production` directly, so replacing
  # its call site in `resolve_search_shape` with `pass` — the guard function
  # left fully intact — kept 35/35 tests green. A guard whose INVOCATION no
  # test covers is indistinguishable from a deleted one, which is this
  # codebase's signature defect applied to its own safety net.
  #
  # Emptying the deviation map makes the guard's verdict change (the arena
  # legitimately differs from selfplay on `temperature`, `add_noise` and the
  # two target-only knobs), so a wired guard MUST raise and an unwired one
  # cannot. Nothing else in `resolve_search_shape` reads this map.
    import scripts.arena_standard as arena_mod

    monkeypatch.setattr(arena_mod, "ARENA_SHAPE_DEVIATIONS", {})
    with pytest.raises(SystemExit, match="NOT the search production runs"):
        resolve_search_shape("training")


def test_arena_guard_fires_on_a_key_omitted_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key DROPPED from the shape dict, not merely stale, must also be caught.

    ⚑ This test exists because of what a mutation run taught, and the lesson is
    not the obvious one. The original mutant dropped the three VOLATILITY keys
    and survived — but that mutant is EQUIVALENT, not evidence of a gap:
    `config_yaml` hard-refuses a non-zero `volatility_q_scale`/`volatility_fpu`
    on the production path (no distributed evaluator implements
    `evaluate_encoded_with_volatility`), so production can only ever ship the
    values that equal GumbelConfig's defaults. There is no live config on which
    omitting them changes anything.

    "This check cannot fire" and "this check cannot fire on any REACHABLE
    config" are different claims, and only the first is a defect. So the
    omission case is exercised on `c_scale`, which production does vary.
    """
    from scripts.arena_standard import _assert_training_shape_is_production

    live = _write_config(tmp_path, {"gumbel_c_scale": 0.0375})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    prod = production_search_shape(flat, simulations=1)
    assert prod.cfg.c_scale == pytest.approx(0.0375), "the mutation did not take"
    with pytest.raises(SystemExit, match="c_scale"):
        _assert_training_shape_is_production({
            "topk": int(prod.cfg.topk),
            "policy_temp": float(prod.cfg.policy_temp),
        }, flat, vloss_weight=int(prod.vloss_weight),
           target_batch=int(prod.target_batch))


def test_arena_training_shape_passes_when_it_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.arena_standard import _assert_training_shape_is_production

    live = _write_config(tmp_path, {"gumbel_policy_temp": 1.9})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    prod = production_search_shape(flat, simulations=1)
    _assert_training_shape_is_production({
        "c_scale": float(prod.cfg.c_scale),
        "topk": int(prod.cfg.topk),
        "policy_temp": float(prod.cfg.policy_temp),
        "volatility_q_scale": float(prod.cfg.volatility_q_scale),
        "volatility_fpu": float(prod.cfg.volatility_fpu),
        "volatility_anchor": float(prod.cfg.volatility_anchor),
    }, flat, vloss_weight=int(prod.vloss_weight),
       target_batch=int(prod.target_batch))


def test_arena_guard_sees_a_runner_argument_no_GumbelConfig_FIELD_CARRIES(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT (F1): the arena passes the WRONG `vloss_weight`.

    THE case the pre-fix guard could not fail on. `gumbel_vloss_weight` reaches
    the C runner as a keyword argument and has no `GumbelConfig` field, so a
    comparison that iterated `dataclasses.fields(GumbelConfig)` returned an
    empty diff for it BY CONSTRUCTION and printed the affirmative line. Both
    branches are exercised: production's value passes, one off it refuses.
    """
    from scripts.arena_standard import _assert_training_shape_is_production

    live = _write_config(tmp_path, {"gumbel_vloss_weight": 1})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    prod = production_search_shape(flat, simulations=1)
    assert prod.vloss_weight == 1, "the mutation did not take"
    assert not any(
        f.name == "vloss_weight" for f in dataclasses.fields(GumbelConfig)
    ), "vloss_weight became a GumbelConfig field; this test no longer proves anything"
    shape = {
        "c_scale": float(prod.cfg.c_scale),
        "topk": int(prod.cfg.topk),
        "policy_temp": float(prod.cfg.policy_temp),
        "volatility_q_scale": float(prod.cfg.volatility_q_scale),
        "volatility_fpu": float(prod.cfg.volatility_fpu),
        "volatility_anchor": float(prod.cfg.volatility_anchor),
    }
    # PASSES on production's value...
    _assert_training_shape_is_production(
        shape, flat, vloss_weight=1, target_batch=int(prod.target_batch),
    )
    # ...and REFUSES one off it.
    with pytest.raises(SystemExit, match="vloss_weight"):
        _assert_training_shape_is_production(
            shape, flat, vloss_weight=0, target_batch=int(prod.target_batch),
        )


# ---------------------------------------------------------------------------
# The --config check is FIELD-COMPLETE, not another hand-list
# ---------------------------------------------------------------------------


def test_the_config_key_hand_list_is_gone() -> None:
    """`PRODUCTION_SEARCH_KEYS` was #227 one level up. It must not come back.

    A list of ten key NAMES cannot see a knob added to production later, and
    the two guards downstream of it are both built from the SAME stale flat, so
    they agree with each other and certify nothing. Pinned as a NAME check
    because that is what a reintroduction would look like.
    """
    import scripts.audit_targets as at

    assert not hasattr(at, "PRODUCTION_SEARCH_KEYS")
    # What replaced it: an exempt map with reasons, and the two keys the audit
    # reads straight from the flat instead of through production's builder.
    assert set(at.CONFIG_COMPARE_EXEMPT) == {"simulations"}
    assert all(r.strip() for r in at.CONFIG_COMPARE_EXEMPT.values())
    assert set(at.AUDIT_DIRECT_CONFIG_KEYS) >= {
        "mcts_simulations", "fast_simulations",
    }


def _flat_alias_names(tree: Any) -> set[str]:
    """``{"flat"}`` plus every local name bound directly from it, transitively.

    ⚑ R2. Without this the completeness scan below is bound to the IDENTIFIER
    ``flat``, so ``config_authority.covers`` is complete BY CONVENTION rather
    than by check: measured, a one-line alias

        _cfgmap = flat
        _cfgmap.get("soft_policy_temp", 2.0)

    survives 91 tests, reads a production value straight out of the config, and
    is never compared against the live yaml — while the stamp keeps claiming
    "every config value this script consumes was proved equal to production's".
    A scanner that a rename defeats is a scanner whose result is a statement
    about spelling.

    LIMIT, stated because it is real and cannot be closed by an AST walk: this
    follows NAME BINDINGS only. A dict handed to a helper that names its
    parameter something other than ``flat`` (``_read(flat)`` where ``_read``
    takes ``cfg``) is not followed, and neither is a copy through a call
    (``dict(flat)``). The stamp says so — see ``ConfigAuthority.stamp``'s
    ``covers.scan`` — so the claim is bounded rather than overstated.
    """
    import ast

    aliases = {"flat"}
    for _ in range(8):  # transitive closure; the depth here is 1 in practice
        before = len(aliases)
        for node in ast.walk(tree):
            pairs: list[tuple[Any, Any]] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                        pairs.extend(zip(target.elts, node.value.elts, strict=False))
                    else:
                        pairs.append((target, node.value))
            elif (
                isinstance(node, ast.AnnAssign) and node.value is not None
            ) or isinstance(node, ast.NamedExpr):
                assert node.value is not None
                pairs.append((node.target, node.value))
            for target, value in pairs:
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Name)
                    and value.id in aliases
                ):
                    aliases.add(target.id)
        if len(aliases) == before:
            break
    return aliases


def _direct_flat_reads(source: str) -> set[str]:
    """Every string key read straight off the flattened config.

    ⚑ BOTH ACCESS FORMS. A `.get`-only scan misses `flat["k"]`, and a
    subscript-only scan misses `flat.get("k", d)`. CLAUDE.md records exactly
    this asymmetry producing a false negative on the single most consequential
    key in a `TrialConfig` audit: `from_dict` reads `lr` by SUBSCRIPT, so a
    `.get`-only scan reported it as unread.

    ``audit_targets`` happens to use only the `.get` form today, which would
    make the subscript branch unreachable-by-construction and therefore
    untestable against the real module — the shape of an equivalent mutant.
    ``test_the_flat_read_scanner_sees_both_access_forms`` drives it against a
    synthetic source that uses both, so neither branch is dead.

    ⚑ AND EVERY ALIAS OF IT, not only the identifier ``flat`` — see
    ``_flat_alias_names`` for the mutant that motivated it and for the limit
    that remains.
    """
    import ast

    tree = ast.parse(source)
    names = _flat_alias_names(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            found.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
    return found


def test_the_flat_read_scanner_sees_both_access_forms() -> None:
    """The scanner's own negative control — it is a guard, so it needs one.

    Without this the subscript branch is dead code against the real module
    (which uses only `.get` today), so deleting it would change nothing and the
    completeness test below would silently stop covering the form that bit the
    `TrialConfig` audit.
    """
    src = (
        "def f(flat, other):\n"
        "    a = flat['by_subscript']\n"
        "    b = flat.get('by_get', 3)\n"
        "    c = other.get('not_flat', 1)\n"
        "    d = other['also_not_flat']\n"
        "    e = flat.get(dynamic_key, 1)\n"
        "    return a, b, c, d, e\n"
    )
    assert _direct_flat_reads(src) == {"by_subscript", "by_get"}


def test_the_flat_read_scanner_follows_an_alias() -> None:
    """R2's mutant, as a unit: renaming the local must not empty the scan.

    ⚑ BOTH BRANCHES. The aliased read must be FOUND (or the completeness test
    it feeds is defeated by one line), and a same-shaped read off an unrelated
    dict must NOT be found (or the scanner over-reports and the check starts
    refusing configs over keys nothing consumes).
    """
    src = (
        "def f(flat, other):\n"
        "    _cfgmap = flat\n"
        "    _again = _cfgmap\n"
        "    a = _cfgmap.get('via_alias', 2.0)\n"
        "    b = _again['via_alias_twice']\n"
        "    c, d = flat, other\n"
        "    e = c.get('via_tuple_alias', 1)\n"
        "    g = d.get('not_flat_at_all', 1)\n"
        "    return a, b, e, g\n"
    )
    assert _direct_flat_reads(src) == {
        "via_alias", "via_alias_twice", "via_tuple_alias",
    }


def test_every_direct_config_read_is_checked() -> None:
    """MUTANT (F4): a `flat[...]` read the authority stamp does not cover.

    ``config_authority.authoritative`` claims "every config value this script
    consumes was proved equal to the live config's". That claim is only true
    while ``AUDIT_DIRECT_CONFIG_KEYS`` is COMPLETE, and completeness is exactly
    the property a hand-list loses first: before this test the list held two
    keys while the module read fifteen, so a ``--config`` differing from live
    on ``sf_policy_temp`` was stamped as proved.

    So the list is REGENERATED from the source rather than trusted. Adding a
    direct read without adding the key fails here, which is the only reason the
    stamp's scope line can be believed.
    """
    import scripts.audit_targets as at

    found = _direct_flat_reads(Path(at.__file__).read_text(encoding="utf-8"))
    assert found, "the AST scan found no `flat` reads at all — it is vacuous"
    missing = found - set(at.AUDIT_DIRECT_CONFIG_KEYS)
    assert not missing, (
        f"audit_targets reads {sorted(missing)} straight out of the config, but "
        "AUDIT_DIRECT_CONFIG_KEYS does not list them — so config_authority "
        "would be stamped True over values that were never compared to the "
        "live config. Add them to the list, or route the read through "
        "production's builder."
    )
    unused = set(at.AUDIT_DIRECT_CONFIG_KEYS) - found
    assert not unused, (
        f"AUDIT_DIRECT_CONFIG_KEYS lists {sorted(unused)}, which this module no "
        "longer reads directly. A list longer than the reads makes the check "
        "refuse configs over keys that cannot affect the output."
    )


def test_config_check_catches_a_key_no_name_list_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: a production key the deleted hand-list had no entry for.

    ``input_extra_features`` reaches the search through
    ``build_selfplay_gumbel_config``'s ``game`` half, and it is REACHABLE
    rather than hypothetical — production migrated v1 (146 planes) ->
    v2_threats (175 planes). ``PRODUCTION_SEARCH_KEYS`` listed ten
    ``gumbel_*``/``volatility_*``/sim names and none of the game ones, so this
    exact disagreement between the audited config and the live one used to pass
    every guard and print "production training target" over the result.
    """
    from scripts.audit_targets import _assert_config_is_production

    live = _write_config(tmp_path, {}, name="live.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    stale_path = _write_config(
        tmp_path, {"input_extra_features": "v1"}, name="stale.yaml",
    )
    stale = dict(flatten_run_config_defaults(load_yaml_file(str(stale_path))))
    live_flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    assert live_flat["input_extra_features"] != stale["input_extra_features"], (
        "the mutation did not take — this test would be vacuous"
    )
    with pytest.raises(SystemExit, match="input_extra_features"):
        _assert_config_is_production("mutant", stale, allow_stale=False)


def test_config_check_still_value_checks_the_sim_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two keys a field-complete GumbelConfig diff structurally cannot see.

    ``simulations`` is pinned to 1 on both sides of that diff so the SHAPE is
    compared independently of the budget, which means `mcts_simulations` and
    `fast_simulations` — which the audit reads straight out of the flat — need
    their own value check. Without it the exempt entry would be a hole rather
    than a documented deviation.
    """
    from scripts.audit_targets import _assert_config_is_production

    live = _write_config(tmp_path, {"mcts_simulations": 256}, name="live.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    stale_path = _write_config(
        tmp_path, {"mcts_simulations": 64}, name="stale.yaml",
    )
    stale = dict(flatten_run_config_defaults(load_yaml_file(str(stale_path))))
    with pytest.raises(SystemExit, match="mcts_simulations"):
        _assert_config_is_production("mutant", stale, allow_stale=False)


# ---------------------------------------------------------------------------
# The guards' INVOCATIONS, not just the guards
# ---------------------------------------------------------------------------


def test_audit_main_loads_its_config_through_the_checking_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: `main()` calls `flatten_run_config_defaults` directly.

    ``load_audit_config`` fuses the load with the check so the check cannot be
    deleted on its own — but nothing proved ``main()`` calls THAT loader rather
    than flattening the yaml itself, which is the same "guard exists, guard is
    not wired" gap one level up. This drives the real ``main()`` with the
    loader replaced by a tripwire; the run dies at the tripwire or the mutant
    is live.
    """
    import scripts.audit_targets as at

    called: list[str] = []

    def _tripwire(config_path: str, *, allow_stale: bool):
        del allow_stale
        called.append(str(config_path))
        raise SystemExit("TRIPWIRE: load_audit_config was reached")

    monkeypatch.setattr(at, "load_audit_config", _tripwire)
    monkeypatch.setattr(
        "sys.argv",
        ["audit_targets.py", "--checkpoint", str(tmp_path / "nonexistent.pt"),
         "--device", "cpu", "--audit-set", str(tmp_path / "nonexistent.jsonl")],
    )
    with pytest.raises(SystemExit, match="TRIPWIRE"):
        at.main()
    assert called, "main() never reached the checking loader"


def test_audit_main_default_config_is_the_resolved_production_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--config`'s default must be the file it is CHECKED against.

    It used to default to the CWD-relative literal `configs/pbt2_small.yaml`
    while the reference is resolved module-relative, so running from any
    directory but the repo root audited one file and named another.
    """
    import scripts.audit_targets as at

    live = _write_config(tmp_path, {}, name="live.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    seen: list[str] = []

    def _tripwire(config_path: str, *, allow_stale: bool):
        del allow_stale
        seen.append(str(config_path))
        raise SystemExit("TRIPWIRE")

    monkeypatch.setattr(at, "load_audit_config", _tripwire)
    monkeypatch.setattr(
        "sys.argv",
        ["audit_targets.py", "--checkpoint", str(tmp_path / "nonexistent.pt"),
         "--device", "cpu"],
    )
    with pytest.raises(SystemExit, match="TRIPWIRE"):
        at.main()
    assert seen == [str(live)]


# ---------------------------------------------------------------------------
# The dump's ruler stamp (rows (d)/(e) MOVED)
# ---------------------------------------------------------------------------


def test_train_shape_stamp_tracks_the_shape_the_runner_was_handed() -> None:
    """MUTANT (F2): a stamp read off the PRE-override profile.

    The stamp is a ruler declaration. Reading it off the ``_SearchProfile``
    made it lie precisely when a ruler difference existed — measured, a run
    with ``--gumbel-training-rows --gumbel policy_temp=3.0,
    target_max_visit_cap=99`` banked ``{1.5, 5, True}`` and searched
    ``3.0 / 99``, so two dumps from a sweep stamped identically and
    ``require_same_ruler`` joined them.

    BOTH branches, because a stamp that always reports the override would be
    just as broken: without overrides it must report production's shape.
    """
    from scripts.audit_targets import (
        build_profile_search_shape,
        build_search_profiles,
        train_shape_stamp,
    )

    kw = _build_kwargs()
    plain = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    stamp = train_shape_stamp(
        build_profile_search_shape("train", plain["train"], **kw),
    )
    assert stamp["policy_temp"] == pytest.approx(1.5)
    assert stamp["target_max_visit_cap"] == 5
    assert stamp["target_untempered_prior"] is True

    swept = build_search_profiles(
        _flat(), play_sims=256, play_topk=None,
        gumbel_overrides=(("policy_temp", 3.0), ("target_max_visit_cap", 99.0)),
        override_training_rows=True,
    )
    swept_stamp = train_shape_stamp(
        build_profile_search_shape("train", swept["train"], **kw),
    )
    assert swept_stamp["policy_temp"] == pytest.approx(3.0)
    assert swept_stamp["target_max_visit_cap"] == 99
    assert swept_stamp != stamp, (
        "the stamp did not move with the override — paired_compare would join "
        "two arms of a sweep as the same ruler"
    )


def test_the_stamp_is_complete_over_the_runner_argument_set() -> None:
    """MUTANT (Codex P1): stamp only the three fields that were wrong.

    A three-field stamp fixes the ruler change that already happened and
    nothing after it: move ``topk`` or the sim budget and both dumps pass their
    own live-config checks while emitting an identical stamp. The stamp's field
    set is therefore DERIVED from the dataclasses, not written down.
    """
    from scripts.audit_targets import (
        _CHECKPOINT_DERIVED_FIELDS,
        build_profile_search_shape,
        build_search_profiles,
        train_shape_stamp,
        train_shape_stamp_fields,
    )

    want = {
        f.name for f in dataclasses.fields(GumbelConfig)
    } - set(_CHECKPOINT_DERIVED_FIELDS) | set(RUNNER_ARG_FIELDS)
    assert set(train_shape_stamp_fields()) == want
    for name in ("topk", "c_scale", "simulations", "vloss_weight", "target_batch"):
        assert name in want, f"{name} must be part of the ruler"
    for name in _CHECKPOINT_DERIVED_FIELDS:
        assert name not in want, (
            f"{name} comes off the checkpoint, not the config; stamping it "
            "would refuse legitimate cross-net joins that `input_encoding` "
            "already governs"
        )

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    stamp = train_shape_stamp(
        build_profile_search_shape("train", profiles["train"], **_build_kwargs()),
    )
    assert set(stamp) == want


def test_an_unstamped_dump_is_unknown_not_a_guessed_policy_temp() -> None:
    """MUTANT (F5 / Codex P1): infer `policy_temp=1.0` for a legacy dump.

    Pre-fix, `--policy-temp` was an ordinary operator-settable flag that fed
    EVERY profile including the training rows, so a legacy dump made at
    `--policy-temp 2.2` is not deducible as 1.0. The sentinel keeps both
    behaviours the inference was there for — legacy-vs-legacy joins,
    legacy-vs-post-fix is refused — without the guess.
    """
    import json

    from scripts.paired_compare import INFERRED_WHEN_ABSENT, UNSTAMPED_LEGACY

    assert INFERRED_WHEN_ABSENT["search_shape"] == UNSTAMPED_LEGACY
    decoded = json.loads(UNSTAMPED_LEGACY)
    assert isinstance(decoded, str), (
        "the legacy value must not be a shape dict — any dict is a guess about "
        "what the legacy run searched with"
    )
    assert "unstamped" in decoded


def _realized_shapes_fixture(*, fast_sims: int = 32) -> dict[str, Any]:
    """The ``{row: SelfplaySearchShape}`` map ``_net_candidates`` returns.

    Built through the REAL profile builder rather than hand-listing two rows,
    because the key set is load-bearing: ``_net_candidates`` returns a shape
    for every profile it built, the PLAY row included, and a fixture that
    quietly covered only the training rows would make the stamp tests describe
    a dump shape production never writes.
    """
    import scripts.audit_targets as at

    flat = dict(_flat())
    flat["fast_simulations"] = int(fast_sims)
    profiles = at.build_search_profiles(flat, play_sims=64, play_topk=None)
    return {
        name: at.build_profile_search_shape(name, p, **_build_kwargs())
        for name, p in profiles.items()
    }


def test_the_dump_stamps_track_BOTH_training_rows() -> None:
    """MUTANT (Codex): stamp only the full-sims row.

    ``fast_simulations`` is the playout-capped row's entire budget, and it is a
    live-editable key. With only ``realized_shapes["train"]`` banked, two dumps
    made either side of a ``fast_simulations`` edit carry BYTE-IDENTICAL
    provenance while ``cand.train_fast.*`` came from different searches — so
    ``paired_compare`` joins them and charges the ruler change to the
    checkpoints, which is the one failure the stamp exists to stop.

    ⚑ Driven by EXECUTION, not by grepping the source for a line. The grep
    version of this test went red for a correct refactor and would have stayed
    green for a stamp built from the wrong object.
    """
    import scripts.audit_targets as at

    authority = at.ConfigAuthority(
        authoritative=True, reference="x", allow_stale=False, reason="",
        values={"fast_simulations": 32},
    )
    before = at.dump_ruler_stamps(_realized_shapes_fixture(fast_sims=32), authority)
    after = at.dump_ruler_stamps(_realized_shapes_fixture(fast_sims=13), authority)

    shape_before = cast("dict[str, Any]", before["search_shape"])
    shape_after = cast("dict[str, Any]", after["search_shape"])
  # ⚑ EVERY row `_net_candidates` realizes, not a hand-picked pair — the PLAY
  # row runs a search too, and its `policy_temp` is operator-settable.
    assert set(shape_before) == set(at.build_search_profiles(
        _flat(), play_sims=64, play_topk=None,
    ))
    assert shape_before["train"] == shape_after["train"], (
        "the full-sims row did not move — the fixture is not isolating "
        "fast_simulations"
    )
    assert shape_before["train_fast"] != shape_after["train_fast"], (
        "a fast_simulations change leaves the banked provenance identical, so "
        "two dumps whose cand.train_fast.* rows came from different search "
        "budgets are joinable. Stamp every realized row, not just `train`."
    )


def test_the_dump_stamps_carry_config_VALUES_not_only_key_names() -> None:
    """MUTANT (Codex): the authority stamp banks names, so drift is invisible.

    ``authoritative`` is a SAME-RUN verdict. Two audits weeks apart under
    different ``sf_policy_temp`` each pass their contemporaneous live check and
    each stamp ``authoritative: true`` — and their row-(c) numbers are built
    from different targets. Banking the key NAMES cannot see that: the names
    are identical in both dumps by construction.

    ⚑ BOTH BRANCHES: equal configs must still produce equal stamps, or the
    field would refuse every join instead of the wrong ones.
    """
    import scripts.audit_targets as at

    shapes = _realized_shapes_fixture()
    same = at.ConfigAuthority(
        authoritative=True, reference="x", allow_stale=False, reason="",
        values={"sf_policy_temp": 2.0},
    )
    drifted = at.ConfigAuthority(
        authoritative=True, reference="x", allow_stale=False, reason="",
        values={"sf_policy_temp": 3.0},
    )
    a = at.dump_ruler_stamps(shapes, same)
    b = at.dump_ruler_stamps(shapes, drifted)
    assert a["config_authority"] == b["config_authority"], (
        "the authority verdict must NOT move on a config value change — if it "
        "did, this would be a same-run flag masquerading as a ruler"
    )
    assert a["target_config"] != b["target_config"], (
        "the banked target config is identical across a sf_policy_temp change, "
        "so paired_compare cannot see temporal config drift. Bank the VALUES."
    )
    assert at.dump_ruler_stamps(shapes, same)["target_config"] == a["target_config"]


def test_the_banked_values_come_from_the_config_that_was_AUDITED(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT (Codex, the PRODUCER half): bank the key names, not the values.

    ⚑ THE TEST ABOVE CANNOT CATCH THIS AND THAT IS THE POINT. It constructs a
    ``ConfigAuthority`` with values it chose itself, so it proves the STAMP
    formats what it is given — it never runs the code that decides what to
    give it. Replacing ``_realized_config_values``' body with
    ``dict.fromkeys(AUDIT_DIRECT_CONFIG_KEYS, "<name-only>")`` survived it, and
    survived the coverage assertion in
    ``test_allow_stale_config_is_still_a_supported_escape`` too, because that
    one compares the KEY SET. Certifying and dispatching are two events; this
    is the one that watches the producer.

    So drive ``_assert_config_is_production`` for real against two live configs
    that differ on one target key and assert the banked value FOLLOWS. Both
    branches: it tracks the change, and it reports the config that was audited
    rather than a constant.
    """
    from scripts.audit_targets import _assert_config_is_production

    def _authority_for(temp: float) -> Any:
        live = _write_config(
            tmp_path, {"sf_policy_temp": temp}, name=f"live_{temp}.yaml",
        )
        monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
        flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
        return _assert_config_is_production(str(live), flat, allow_stale=False)

    low = _authority_for(2.0)
    high = _authority_for(3.0)
    assert low.authoritative, "the low fixture did not certify"
    assert high.authoritative, "the high fixture did not certify"
    assert low.config_values()["sf_policy_temp"] == pytest.approx(2.0), (
        "the banked target config does not carry the value the audited config "
        "actually held — it is a list of key names wearing a value's name, so "
        "paired_compare cannot see temporal drift with it"
    )
    assert high.config_values()["sf_policy_temp"] == pytest.approx(3.0)
    assert low.config_values() != high.config_values()
  # ...and the un-mutated neighbours still agree, so the difference above is
  # the key that moved and not the whole dict being rebuilt.
    assert low.config_values()["mcts_simulations"] == (
        high.config_values()["mcts_simulations"]
    )


def test_dump_row_carries_every_ruler_field_paired_compare_reads() -> None:
    """The dump row itself, not just the helper, must carry the stamps.

    Half AST (the row really does unpack the builder) and half execution (what
    that builder produces), so neither half can be satisfied by a name. It is
    deliberately paired with the two tests above (the VALUES are right) and
    with `tests/test_paired_compare.py` (the gate READS them), so no single one
    of them is load-bearing on its own.

    ⚑ Closes the loop against `paired_compare.RULER_FIELDS`: a field added to
    the gate that no producer writes is a check that can never fail, and this
    repo has shipped exactly that.
    """
    import ast
    import inspect

    import scripts.audit_targets as at
    from scripts.paired_compare import RULER_FIELDS

    tree = ast.parse(inspect.getsource(at.main))
    appends = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "per_pos_dump"
    ]
    assert len(appends) == 1, (
        f"{len(appends)} per_pos_dump.append(...) sites in main(); the dump row "
        "moved — re-point this test, do not delete it"
    )
    row = appends[0].args[0]
    assert isinstance(row, ast.Dict)
    literal_keys = {
        k.value for k in row.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    unpacks = [
        ast.unparse(v) for k, v in zip(row.keys, row.values, strict=True) if k is None
    ]
    assert "dump_ruler_stamps(realized_shapes, config_authority)" in unpacks, (
        f"the dump row no longer unpacks the stamp builder (unpacks: {unpacks}); "
        "the stamps may still be there, but nothing can check them by execution"
    )
    produced = set(at.dump_ruler_stamps(
        _realized_shapes_fixture(),
        at.ConfigAuthority(
            authoritative=True, reference="x", allow_stale=False, reason="",
        ),
    ))
    assert "config_authority" in produced
    missing = set(RULER_FIELDS) - literal_keys - produced
    assert not missing, (
        f"paired_compare gates on {sorted(missing)}, which audit_targets' dump "
        f"row never writes. A ruler field no producer stamps is a gate that "
        f"cannot fail — every join takes the 'not declared' warn path."
    )


# ---------------------------------------------------------------------------
# probe_policy_targets: the `den` floor is NOT an equivalent mutant
# ---------------------------------------------------------------------------


def _fp16_near_tie_row(ulps: int, temp: float) -> tuple[np.ndarray, np.ndarray]:
    """A two-entry row whose top moves differ by `ulps` fp16 steps near 0.5.

    Both arrays round-trip through float16 because that is how they are stored
    (``replay/buffer.py:199`` writes ``policy_target`` as float16), which is
    what makes a near-tie the common case rather than a contrived one.
    """
    top = np.float16(0.5)
    second = top
    for _ in range(ulps):
        second = np.nextafter(second, np.float16(0.0))
    p16 = np.zeros((1, 8), dtype=np.float16)
    p16[0, 0] = top
    p16[0, 1] = second
    p = p16.astype(np.float32)
    pn = p / p.sum(axis=1, keepdims=True)
    q = pn ** (1.0 / temp)
    q = q / q.sum(axis=1, keepdims=True)
    return p, q.astype(np.float16).astype(np.float32)


def test_fp16_near_tie_needs_the_den_floor() -> None:
    """MUTANT: `den > 0.0` instead of `den > 1e-6`. It is NOT equivalent.

    ⚑ This mutant was RECORDED AS EQUIVALENT by the commit under review, on the
    argument that "the mask already drops every entry below 1e-6, so any
    surviving off-reference entry contributes a LARGE log-ratio". The argument
    is about the wrong quantity: ``den`` accumulates squared log-RATIOS to the
    reference entry, so two entries that both clear the mask and are nearly
    EQUAL give a near-zero contribution, not a large one.

    Measured, on the row below (3 fp16 ulps below 0.5, written at T = 2.0):
    ``den = 2.385350e-07``, and with the floor removed the row reports
    ``T_hat = 3.0015``. The floor is load-bearing in the OPPOSITE direction to
    the claim — it filters fp16 near-ties, not degenerate zeros.

    Practical impact on the reported statistic is nil (4000-row Monte-Carlo:
    median 2.0000 in both fp32 and fp16), so BEHAVIOUR IS UNCHANGED; only the
    comment was wrong. This test exists so the mutant is no longer recorded as
    equivalent.
    """
    from scripts.probe_policy_targets import _recover_soft_policy_temp

    p, q = _fp16_near_tie_row(ulps=3, temp=2.0)
    # The construction is what it claims: one off-reference entry, a single
    # fp16 step apart after the temperature was applied and re-rounded.
    assert q[0, 0] != q[0, 1], "q collapsed to an exact tie; wrong ulp count"
    assert abs(float(q[0, 0]) - float(q[0, 1])) < 1e-3

    got = _recover_soft_policy_temp(p, q)
    assert np.isnan(got[0]), (
        "a row whose soft entries are one fp16 ulp apart carries no recoverable "
        "temperature; without the floor it reports T~=3.0 for a T=2.0 shard"
    )

    # And the floor is not simply rejecting everything: widen the gap and the
    # same machinery recovers a temperature again. Without this the test would
    # pass against `ok = False`.
    p_wide, q_wide = _fp16_near_tie_row(ulps=32, temp=2.0)
    wide = _recover_soft_policy_temp(p_wide, q_wide)
    assert np.isfinite(wide[0])
    assert wide[0] == pytest.approx(2.0, abs=0.05)


def test_absent_soft_policy_temp_is_NOT_CHECKED_not_a_silent_2_point_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT: `live.flat.get("soft_policy_temp", 2.0)`.

    This whole check exists because a hard-coded `3.0` in the module docstring
    went stale. A hard-coded `2.0` in the comparison is the same bug with a
    fresher number: against a config that does not carry the key it would print
    MATCHES, which is an affirmative verdict about a value nothing set.

    ⚑ REACHABILITY, stated rather than assumed — the reviewer's premise was
    right in form and wrong about the state.
    `test_the_schema_default_is_why_that_fallback_never_fired` measures that a
    flattened production config ALWAYS carries the key, so this branch is a
    backstop against a schema change (retiring the key, renaming it), not a
    path any config takes today. It is driven here with a stubbed config
    because that is the only way to reach it — which is exactly why the
    fallback survived review.
    """
    import scripts.probe_policy_targets as probe
    from chess_anti_engine.eval.production_shape import LiveConfig
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature

    stub = LiveConfig(
        path=tmp_path / "live.yaml", flat={}, sha256="0" * 64,
        provenance="stub", authoritative=True,
    )
    monkeypatch.setattr(
        "chess_anti_engine.eval.production_shape.load_live_config_or_reason",
        lambda: (stub, ""),
    )
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(12) * 0.4, size=64).astype(np.float32)
    q = np.stack([apply_policy_temperature(r, 2.0) for r in p]).astype(np.float32)
    line = probe._check_soft_policy_temp(probe._recover_soft_policy_temp(p, q))
    assert "NOT CHECKED (key absent from config" in line
    assert "MATCHES" not in line


def test_probe_marks_a_NON_AUTHORITATIVE_reference_as_such(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT (F3): branch on `live is None` and never read `live.authoritative`.

    ``load_live_config_or_reason`` returns the IN-TREE fallback with
    ``authoritative=False`` — NOT ``None`` — when
    ``$CHESS_ANTI_ENGINE_LIVE_CONFIG`` is unset, which is the default in every
    worktree. Branching on ``None`` alone therefore printed "live config says
    2.0 -> MATCHES" about a file the resolver had already decided was not
    live. Same defect ``audit_targets`` fixed in the same change, left standing
    in a sibling instrument.

    BOTH branches are asserted: the authoritative case must still say "live
    config" and must NOT carry the warning, or the fix would be a gate that
    cannot pass.
    """
    import scripts.probe_policy_targets as probe
    from chess_anti_engine.eval.production_shape import LiveConfig
    from chess_anti_engine.selfplay.temperature import apply_policy_temperature

    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(12) * 0.4, size=64).astype(np.float32)
    q = np.stack([apply_policy_temperature(r, 2.0) for r in p]).astype(np.float32)
    recovered = probe._recover_soft_policy_temp(p, q)

    def _stub(authoritative: bool) -> None:
        cfg = LiveConfig(
            path=tmp_path / "pbt2_small.yaml", flat={"soft_policy_temp": 2.0},
            sha256="0" * 64,
            provenance="stub" if authoritative else "in-tree fallback",
            authoritative=authoritative,
        )
        monkeypatch.setattr(
            "chess_anti_engine.eval.production_shape.load_live_config_or_reason",
            lambda: (cfg, ""),
        )

    _stub(False)
    stale = probe._check_soft_policy_temp(recovered)
    assert "NOT-LIVE" in stale, "the [NOT-LIVE] provenance mark never printed"
    assert "NOT a production check" in stale
    assert "live config says" not in stale, (
        "the probe called a non-authoritative reference the live config"
    )
    assert "MATCHES" in stale  # the numeric verdict is still reported

    _stub(True)
    live = probe._check_soft_policy_temp(recovered)
    assert "[LIVE]" in live
    assert "live config says 2.0 -> MATCHES" in live
    assert "NOT a production check" not in live


def test_the_schema_default_is_why_that_fallback_never_fired() -> None:
    """The measurement behind the reachability note above.

    ``flatten_run_config_defaults`` fills every schema key, so `.get(key, 2.0)`
    could not fire on any config that flattens at all. Pinned so that if
    ``soft_policy_temp`` is ever retired from the schema the NOT-CHECKED branch
    becomes live and this test says why.
    """
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(_IN_TREE_CONFIG))))
    assert "soft_policy_temp" in flat


# ---------------------------------------------------------------------------
# The four states a bare `None` used to collapse into
# ---------------------------------------------------------------------------


def test_unavailable_live_config_names_which_of_four_states_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT: `except Exception: return None`, discarding the exception text.

    Unset, missing-file, unreadable-file and fails-to-flatten call for four
    different operator actions — export the variable, fix the path, fix the
    permissions, rebase onto the branch whose schema defines the live yaml's
    new key. A message announcing the union of them sends the reader to the
    wrong one most of the time.
    """
    from chess_anti_engine.eval.production_shape import load_live_config_or_reason

    missing = tmp_path / "nope.yaml"
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(missing))
    cfg, reason = load_live_config_or_reason()
    assert cfg is None
    assert "does not exist" in reason
    assert str(missing) in reason

    unflattenable = tmp_path / "bad.yaml"
    unflattenable.write_text(
        yaml.safe_dump({"selfplay": {"a_key_no_schema_defines": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(unflattenable))
    cfg, reason = load_live_config_or_reason()
    assert cfg is None
    assert "does not flatten" in reason
    # The exception's own TYPE and TEXT, which is what names the offending key.
    assert "a_key_no_schema_defines" in reason

    # ...and the success path carries no reason at all, so a caller cannot
    # print a diagnosis for a config that loaded.
    good = _write_config(tmp_path, {}, name="good.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(good))
    cfg, reason = load_live_config_or_reason()
    assert cfg is not None
    assert reason == ""


def test_an_unreadable_config_is_REPORTED_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT (Codex P2): `read_bytes()` outside the try.

    ``load_config_file``'s contract is that it NEVER RAISES — every caller
    prints a degradation warning and carries on, and these instruments are run
    off the training host often enough that a hard requirement would be a
    regression. With the read above the guard, a path that exists but cannot be
    read (permissions, an I/O error, or removal in the window between
    ``is_file()`` and the read) killed probe / value-regret / search-shape
    before any of them could emit the warning the module exists to emit.

    ⚑ BOTH BRANCHES: the unreadable file is reported with its own reason, and
    the readable one still loads — a "fix" that reported everything as
    unreadable would be a gate that cannot pass.
    """
    import os

    from chess_anti_engine.eval.production_shape import load_live_config_or_reason

    if os.geteuid() == 0:
        pytest.skip("running as root — chmod 000 is not enforced")
    unreadable = _write_config(tmp_path, {}, name="unreadable.yaml")
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(unreadable))
    # Sanity: it loads while it is readable, so the mutation below is what moves it.
    assert load_live_config_or_reason()[0] is not None
    unreadable.chmod(0o000)
    try:
        cfg, reason = load_live_config_or_reason()
    finally:
        unreadable.chmod(0o644)
    assert cfg is None, "an unreadable config must not be reported as loaded"
    assert "cannot be read" in reason
    assert "PermissionError" in reason
    assert str(unreadable) in reason


# ---------------------------------------------------------------------------
# The remaining INVOCATION gaps, closed with tripwires
# ---------------------------------------------------------------------------
#
# ⚑ Each of these replaces a real guard with a function that raises, drives the
# production entry point, and asserts the run dies at the tripwire. That is the
# only construction that distinguishes "the guard exists" from "the guard is
# called" — an AST test cannot, and every other case in this file drives the
# guard directly, which is exactly how the 35/35-green mutant survived.


def _stub_net(monkeypatch: pytest.MonkeyPatch):
    """A REAL `NetSource` whose `load` returns a model-shaped namespace.

    A real one rather than a duck-typed stand-in so the call sites below are
    still type-checked against the signature they are exercising — a stub that
    only happens to satisfy the runtime is how a test stops noticing that the
    function it drives changed shape.

    Enough for the two rulers to reach their guard: both call `net.load(...)`
    and then read the encoding attributes off the result before doing any
    inference. Nothing here touches torch beyond `LocalModelEvaluator`'s
    attribute-only `__init__`.
    """
    import types

    from scripts.net_source import NetSource

    def _load(self: NetSource, **kwargs: Any) -> Any:
        del self, kwargs
        return types.SimpleNamespace(
            input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats",
            policy_encoding="lc0_1858",
            use_dynamic_relations=False,
        )

    monkeypatch.setattr(NetSource, "load", _load)
    return NetSource(checkpoint="stub-net.pt")


def test_audit_net_candidates_builds_its_configs_through_the_guarded_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: `_net_candidates` assembles a GumbelConfig itself.

    `build_profile_search_shape` carries the training rows'
    `assert_matches_production` call. Building the config any other way inside
    `_net_candidates` would leave the guard intact, tested, and unreachable.
    """
    import chess

    import scripts.audit_targets as at

    def _tripwire(*args: Any, **kwargs: Any):
        del args, kwargs
        raise SystemExit("TRIPWIRE: build_profile_search_shape was reached")

    monkeypatch.setattr(at, "build_profile_search_shape", _tripwire)
    profiles = at.build_search_profiles(_flat(), play_sims=8, play_topk=None)
    with pytest.raises(SystemExit, match="TRIPWIRE"):
        at._net_candidates(
            [chess.Board()], net=_stub_net(monkeypatch), device="cpu",
            batch_size=1, seed=0,
            profiles=profiles, requested_gumbel_overrides=(),
        )


def test_value_regret_reports_the_input_layout_before_it_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: the layout report is defined and never called.

    `value_regret` WARNS rather than refuses (foreign nets are a supported
    use), which makes the invocation the whole guard: an uncalled warning is
    indistinguishable from no warning, and the cp figure then sits in a table
    with production-layout readings under nothing at all.
    """
    import scripts.value_regret as vr

    def _tripwire(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise SystemExit("TRIPWIRE: _report_encoding_vs_production was reached")

    monkeypatch.setattr(vr, "_report_encoding_vs_production", _tripwire)
    with pytest.raises(SystemExit, match="TRIPWIRE"):
        vr.value_1ply_regret(
            net=_stub_net(monkeypatch), positions=[], device="cpu",
            batch_size=1, pos_chunk=1,
        )


def test_probe_main_runs_the_temperature_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT: `main()` prints the table without the temperature line.

    The line decides whether the table below it describes the live setting or a
    superseded one, so a `main()` that stops calling it turns every number in
    the report into an unattributed one.
    """
    import scripts.probe_policy_targets as probe

    n = 4
    fake = {
        "tv": np.zeros(n), "kl_pq": np.zeros(n), "kl_qp": np.zeros(n),
        "argmax_agree": np.ones(n), "source": np.zeros(n, dtype=np.int64),
        "phase": np.zeros(n, dtype=np.int64), "temp_hat": np.full(n, 2.0),
        "_used_shards": np.array(1),
    }
    monkeypatch.setattr(probe, "_collect", lambda *a, **k: fake)

    def _tripwire(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise SystemExit("TRIPWIRE: _check_soft_policy_temp was reached")

    monkeypatch.setattr(probe, "_check_soft_policy_temp", _tripwire)
    monkeypatch.setattr(
        "sys.argv",
        ["probe_policy_targets.py", "--replay-dir", str(tmp_path),
         "--out", str(tmp_path / "out.json")],
    )
    with pytest.raises(SystemExit, match="TRIPWIRE"):
        probe.main()


# ---------------------------------------------------------------------------
# F1: the compared object is the CONSUMER's argument set, not a dataclass
# ---------------------------------------------------------------------------


# Every argument production's C-runner call site passes OUTSIDE the
# `**search_shape.runner_kwargs()` unpack, pinned to its exact expression and
# to the reason it is not a config value the shape should carry.
#
# ⚑ A WHITELIST, and that is the fix. The first revision of this pin was a
# BLACKLIST — it refused any expression starting `search.` — which is a guard
# that cannot see the bypass it exists to stop. Measured: replacing
# `allow_terminal_root_shortcuts=True` with
# `allow_terminal_root_shortcuts=bool(state.game.syzygy_in_search)` routes a
# CONFIG value to the runner past the shape, and survived 120 tests, because
# the config arrived through `state.game` rather than through `search.`. The
# line directly above it (`tb_probe=... if state.game.syzygy_in_search ...`) is
# the standing precedent for that exact route, so the bypass is not a
# hypothetical spelling — it is a copy of the neighbouring line.
#
# So the assertion is now "this set, exactly": a new argument, a changed
# expression, or a deleted one all fail and force a decision here. An argument
# that IS config-derived belongs on `SelfplaySearchShape`; one that genuinely
# is not gets an entry with the reason.
_CALL_SITE_ARGS: dict[str, str] = {
    "<positional 0>=state.model": "the loaded net — runtime object",
    "<positional 1>=sub_cboards": "the boards being searched — runtime",
    "device=state.device": "the trainer/worker's device — runtime",
    "rng=rng": "the selfplay RNG — runtime",
    "evaluator=eval_impl": "the broker/evaluator handle — runtime",
    "pre_pol_logits=sub_pol": "already-computed root logits — runtime",
    "pre_wdl_logits=sub_wdl": "already-computed root WDL — runtime",
    "per_game_simulations=sub_sims": (
        "per-game budgets from the playout-cap DRAW; the config half "
        "(`simulations`) is on the shape as `cfg.simulations`"
    ),
    "per_game_add_noise=sub_noise": (
        "the per-PLY root-noise schedule — a function of the move number, not "
        "a scalar. This is the coverage gap SHAPE_COVERAGE_NOTE names out loud "
        "rather than implying away"
    ),
    "per_game_gumbel_scale=sub_scale": (
        "same per-PLY schedule (`_scheduled_gumbel_scale`); see above"
    ),
    "cboards=sub_cboards": "the boards being searched — runtime",
    "tree=state.mcts_tree": "the reused search tree — runtime",
    "root_node_ids=sub_root_ids": "tree-reuse root ids — runtime",
    "tb_probe=state.tb_probe if state.game.syzygy_in_search else None": (
        "⚑ THE ONE CONFIG-GATED ARGUMENT. The prober itself is a runtime "
        "object, but its GATE (`game.syzygy_in_search`) is a config value that "
        "reaches the runner without passing through SelfplaySearchShape, so a "
        "shape diff structurally cannot see it. It is covered by a DIFFERENT "
        "instrument instead — `audit_targets.AUDIT_DIRECT_CONFIG_KEYS` "
        "value-compares `syzygy_in_search` and `syzygy_path` against the live "
        "yaml — and SHAPE_COVERAGE_NOTE says so, so a clean shape table is not "
        "read as covering it."
    ),
    "allow_terminal_root_shortcuts=True": (
        "a literal: selfplay always allows them. Not config-derived, and it "
        "must not BECOME config-derived without landing on the shape"
    ),
    "return_diagnostics=True": (
        "a literal: selfplay always wants the per-move diagnostics"
    ),
}


def _call_site_arg_texts(call: Any) -> dict[str, str]:
    """``{"name=expr": expr}`` for every non-splat argument of ``call``.

    Positionals are keyed by index, because a config value can arrive
    positionally just as easily as by keyword and a keyword-only scan would not
    see it.
    """
    import ast

    out: dict[str, str] = {}
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Starred):
            out[f"<starred positional {i}>=" + ast.unparse(arg.value)] = ast.unparse(arg)
            continue
        out[f"<positional {i}>=" + ast.unparse(arg)] = ast.unparse(arg)
    for kw in call.keywords:
        if kw.arg is None:
            continue
        out[f"{kw.arg}=" + ast.unparse(kw.value)] = ast.unparse(kw.value)
    return out


def test_the_shape_is_exactly_what_the_C_runner_is_handed() -> None:
    """MUTANT (F1/R1): a config-derived runner argument added BESIDE the shape.

    "Field-complete over <schema>" is only as complete as <schema>. The
    previous revision diffed ``GumbelConfig`` and called that exhaustive while
    ``gumbel_vloss_weight`` / ``gumbel_target_batch`` reached the C runner as
    keyword arguments with no such field. Widening to a roomier dataclass would
    have been the same defect with a later expiry date, so the boundary is the
    runner's own argument set — and this test is what keeps that true.

    It parses production's call site and asserts that the config-derived
    arguments come from ``SelfplaySearchShape.runner_kwargs()`` and that
    EVERYTHING else at that call site is one of the pinned, reasoned entries in
    ``_CALL_SITE_ARGS``. Adding ``vloss_weight=int(search.gumbel_vloss_weight)``
    back beside the unpack fails here — and so does routing a config value in
    through ``state.game``, which the blacklist this replaced could not see.
    """
    import ast
    import inspect

    from chess_anti_engine.selfplay import network_turn

    src = inspect.getsource(network_turn)
    tree = ast.parse(src)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gumbel_c_fn"
    ]
    assert len(calls) == 1, (
        f"expected exactly one C-runner call site, found {len(calls)}; a second "
        "one is a second place for a config-derived argument to hide"
    )
    call = calls[0]
    unpacked = [kw for kw in call.keywords if kw.arg is None]
    assert len(unpacked) == 1, "the shape is no longer unpacked at the call site"
    assert ast.unparse(unpacked[0].value) == "search_shape.runner_kwargs()"

    found = set(_call_site_arg_texts(call))
    added = sorted(found - set(_CALL_SITE_ARGS))
    assert not added, (
        f"the C-runner call site passes {added} outside "
        f"**search_shape.runner_kwargs(). If any of them is config-derived it "
        f"is invisible to every instrument that compares search shapes — put "
        f"it on SelfplaySearchShape. If it is genuinely runtime state, add it "
        f"to _CALL_SITE_ARGS with the reason. ⚑ 'It does not start with "
        f"`search.`' is not a reason: config reaches this call through "
        f"`state.game` too, one line above."
    )
    gone = sorted(set(_CALL_SITE_ARGS) - found)
    assert not gone, (
        f"_CALL_SITE_ARGS pins {gone}, which the call site no longer passes (or "
        f"passes with a changed expression). Re-point the entry — a stale "
        f"whitelist silently stops covering the argument it names."
    )
    assert all(r.strip() for r in _CALL_SITE_ARGS.values())


def test_runner_kwargs_covers_every_non_cfg_field_of_the_shape() -> None:
    """The one remaining hand-list is regenerated from the dataclass.

    ``RUNNER_ARG_FIELDS`` names the shape's non-``cfg`` fields. Adding a field
    to ``SelfplaySearchShape`` without adding it here would leave it out of
    every diff, every printed table and the ruler stamp — the drift mechanism
    #227 was about, in the one place a list survives.
    """
    declared = {
        f.name for f in dataclasses.fields(SelfplaySearchShape) if f.name != "cfg"
    }
    assert declared == set(RUNNER_ARG_FIELDS)
    shape = production_search_shape(_flat(), simulations=8)
    assert set(shape.runner_kwargs()) == {"cfg"} | declared


def test_audit_defaults_the_runner_args_to_production_not_to_zero() -> None:
    """MUTANT (F1, the unconditional half): CLI default 0 vs production's 1.

    The reviewer's sharper case: even with ``--config`` EQUAL to the live yaml,
    ``_net_candidates`` received ``vloss_weight=0`` from the CLI while
    production runs 1, so rows (d)/(e) were certified and printed under
    "production training target" while searching the duplicate-leaf shape.

    BOTH branches: unset follows production, and an explicit value is carried
    AND declared as a deviation (so the shape table prints it rather than
    refusing or hiding it).
    """
    import scripts.audit_targets as at

    flat = _flat()
    flat["gumbel_vloss_weight"] = 1
    prod = production_search_shape(flat, simulations=8)
    assert prod.vloss_weight == 1, "the fixture is not exercising the case"

    default = at.build_search_profiles(flat, play_sims=64, play_topk=None)
    assert default["train"].vloss_weight == 1
    assert default["train_fast"].vloss_weight == 1
    assert at.profile_shape_deviations(default["train"]).get("vloss_weight") is None

    explicit = at.build_search_profiles(
        flat, play_sims=64, play_topk=None, vloss_weight=0,
    )
    assert explicit["train"].vloss_weight == 0
    reason = at.profile_shape_deviations(explicit["train"])["vloss_weight"]
    assert "--vloss-weight" in reason
    assert "production runs 1" in reason
  # ...and the declared deviation is what lets the build SUCCEED rather than
  # refuse: an operator arm must stay runnable, just not called production's.
    shape = at.build_profile_search_shape(
        "train", explicit["train"], **_build_kwargs(),
    )
    assert shape.vloss_weight == 0
