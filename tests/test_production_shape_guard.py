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
* a missing live config — the degradation path, which must stay loud and
  non-fatal.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pytest
import yaml

from chess_anti_engine.eval.production_shape import (
    LIVE_CONFIG_ENV,
    FieldDiff,
    assert_matches_production,
    compare_config_values,
    gumbel_field_diff,
    load_live_config,
    production_selfplay_gumbel_config,
    resolve_live_config_path,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig
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
    prod = production_selfplay_gumbel_config(_flat(), simulations=32)
    realized = dataclasses.replace(prod, target_max_visit_cap=0)
    diffs = gumbel_field_diff(realized, prod, exempt={})
    assert [d.field for d in diffs] == ["target_max_visit_cap"]
    with pytest.raises(SystemExit, match="target_max_visit_cap"):
        assert_matches_production(realized, prod, exempt={}, where="unit")


def test_exempting_a_field_suppresses_it_but_only_that_field() -> None:
    """An exemption must be narrow — otherwise it is a gate that cannot fail."""
    prod = production_selfplay_gumbel_config(_flat(), simulations=32)
    realized = dataclasses.replace(prod, target_max_visit_cap=0, topk=3)
    diffs = gumbel_field_diff(
        realized, prod, exempt={"target_max_visit_cap": "unit-test reason"},
    )
    assert [d.field for d in diffs] == ["topk"]


def test_identical_configs_produce_no_diff() -> None:
    """The guard must be capable of PASSING, or it is noise rather than a check."""
    prod = production_selfplay_gumbel_config(_flat(), simulations=32)
    assert gumbel_field_diff(prod, prod, exempt={}) == []
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
    with pytest.raises(SystemExit, match="gumbel_c_scale"):
        load_audit_config(str(stale), allow_stale=False)
    # And it really does return a usable config on the happy path.
    flat = load_audit_config(str(live), allow_stale=False)
    assert flat["gumbel_c_scale"] == pytest.approx(0.1)


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
    with pytest.raises(SystemExit, match="gumbel_c_scale"):
        _assert_config_is_production("mutant", stale, allow_stale=False)
    # ...and the escape hatch must NOT be silent.
    _assert_config_is_production("mutant", stale, allow_stale=True)
    assert "WARNING (--allow-stale-config)" in capsys.readouterr().out


def test_audit_accepts_the_live_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.audit_targets import _assert_config_is_production

    live = _write_config(tmp_path, {"gumbel_c_scale": 0.1})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    _assert_config_is_production(str(live), flat, allow_stale=False)
    out = capsys.readouterr().out
    assert "production search keys match the live config by VALUE" in out
    assert "[LIVE]" in out


class _BuildKwargs(TypedDict):
    """Exactly the checkpoint-derived arguments `build_profile_gumbel_config` takes.

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
    from scripts.audit_targets import build_profile_gumbel_config, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_gumbel_config(
        "train", profiles["train"], **_build_kwargs(),
    )
    assert cfg.policy_temp == pytest.approx(1.5)
    assert cfg.target_max_visit_cap == 5
    assert cfg.target_untempered_prior is True
    assert cfg.add_noise is False  # the one documented deviation


def test_build_actually_compares_the_non_exempt_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT: shrink the exempt map and watch a real deviation surface.

    ``build_profile_gumbel_config`` sets ``add_noise=False`` against
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
    at.build_profile_gumbel_config("train", profiles["train"], **_build_kwargs())
    trimmed = {k: v for k, v in at.TRAIN_SHAPE_DEVIATIONS.items() if k != "add_noise"}
    monkeypatch.setattr(at, "TRAIN_SHAPE_DEVIATIONS", trimmed)
    with pytest.raises(SystemExit, match="add_noise"):
        at.build_profile_gumbel_config("train", profiles["train"], **_build_kwargs())


def test_build_play_row_still_honours_policy_temp_flag() -> None:
    """--policy-temp must keep working on the PLAY row after the split."""
    from scripts.audit_targets import build_profile_gumbel_config, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_gumbel_config(
        "search", profiles["search"], **_build_kwargs(play_policy_temp=2.2),
    )
    assert cfg.policy_temp == pytest.approx(2.2)


def test_policy_temp_flag_does_not_reach_the_training_rows() -> None:
    """MUTANT-BY-CONSTRUCTION: a PLAY flag leaking into a TARGET row.

    Scoring the "production training target" at an operator's --policy-temp is
    the mislabeling this whole change is about, in miniature.
    """
    from scripts.audit_targets import build_profile_gumbel_config, build_search_profiles

    profiles = build_search_profiles(_flat(), play_sims=256, play_topk=None)
    cfg = build_profile_gumbel_config(
        "train", profiles["train"], **_build_kwargs(play_policy_temp=2.2),
    )
    assert cfg.policy_temp == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The degradation path must be loud, and must not be fatal
# ---------------------------------------------------------------------------


def test_missing_live_config_degrades_loudly_not_fatally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.audit_targets import _assert_config_is_production

    monkeypatch.setenv(LIVE_CONFIG_ENV, str(tmp_path / "does-not-exist.yaml"))
    assert load_live_config() is None
    _assert_config_is_production("whatever", _flat(), allow_stale=False)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "has NOT been shown to match" in out


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
    prod = production_selfplay_gumbel_config(flat, simulations=1)
    # The shape the arena would build if it had NOT been updated: stale temp.
    with pytest.raises(SystemExit, match="policy_temp"):
        _assert_training_shape_is_production({
            "c_scale": float(prod.c_scale),
            "topk": int(prod.topk),
            "policy_temp": 1.0,
            "volatility_q_scale": float(prod.volatility_q_scale),
            "volatility_fpu": float(prod.volatility_fpu),
            "volatility_anchor": float(prod.volatility_anchor),
        }, flat)


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
    prod = production_selfplay_gumbel_config(flat, simulations=1)
    assert prod.c_scale == pytest.approx(0.0375), "the mutation did not take"
    with pytest.raises(SystemExit, match="c_scale"):
        _assert_training_shape_is_production({
            "topk": int(prod.topk),
            "policy_temp": float(prod.policy_temp),
        }, flat)


def test_arena_training_shape_passes_when_it_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.arena_standard import _assert_training_shape_is_production

    live = _write_config(tmp_path, {"gumbel_policy_temp": 1.9})
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    flat = dict(flatten_run_config_defaults(load_yaml_file(str(live))))
    prod = production_selfplay_gumbel_config(flat, simulations=1)
    _assert_training_shape_is_production({
        "c_scale": float(prod.c_scale),
        "topk": int(prod.topk),
        "policy_temp": float(prod.policy_temp),
        "volatility_q_scale": float(prod.volatility_q_scale),
        "volatility_fpu": float(prod.volatility_fpu),
        "volatility_anchor": float(prod.volatility_anchor),
    }, flat)
