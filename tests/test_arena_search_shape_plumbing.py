"""The arena must state, and actually run, the search it claims to measure.

Two defects of the house pattern — a value accepted and then silently ignored —
both of which invalidated published Elo (docs/experiment_ledger.md 2026-07-28):

1. ``selfplay/match.py::pick_moves_for_boards`` passed neither ``vloss_weight``
   nor ``target_batch`` to the C search, so both sat at the function default 0
   while production selfplay ran ``gumbel_vloss_weight: 1``. They are function
   ARGUMENTS, not ``GumbelConfig`` fields, so ``--cand-gumbel`` (which works by
   ``dataclasses.replace``) could not reach them either. Measured cost: the
   arena played a different move from production on 10% of positions at 32 sims
   and 27.5% at 256 — production's own budget.
2. ``scripts/arena_standard.py`` seeded every run from ``PLAY_SEARCH_DEFAULTS``
   even with no flag, so every arena silently measured the UCI/play shape
   (c_scale 0.025, topk 32, LOG root) instead of the training shape, which is
   read from the production yaml and uses the LINEAR root.

Do NOT re-quote the training shape's numbers anywhere in this file. They move
with the config: ``ed9de8ee9`` (2026-08-06) took production selfplay from
``gumbel_topk: 16`` to 32, which falsified two literal ``== 16`` expectations
PR #286 had left behind. Those failures did not show up in CI, because CI runs
against the *committed* ``configs/pbt2_small.yaml`` and the live production yaml
is edited in place and lags behind it — so a literal here fails first against
the run that matters and only later against the gate. Assert relationships
(base vs override, training vs play) or read the value back from
``production_selfplay_search_config()``.

The shape is now a required choice with no default, and the realized values are
printed and stored in the JSONL record.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import sys
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    PLAY_SEARCH_TARGET_BATCH,
    PLAY_SEARCH_VLOSS_WEIGHT,
    GumbelConfig,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay import match as match_mod
from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
from scripts.arena_standard import (
    SEARCH_SHAPES,
    SideSearch,
    add_common_args,
    apply_search_overrides,
    build_result_record,
    pentanomial_counts,
    play_paired_games_matched_sims,
    production_selfplay_search_config,
    resolve_search_shape,
    run_arena,
    summarize_pentanomial,
)

# GumbelConfig fields the ARENA owns rather than the training config: the sim
# budget and the move-selection/noise policy come from arena flags. Everything
# else production sets from config has to be carried by the training shape.
_ARENA_OWNED_GUMBEL_FIELDS = {"simulations", "temperature", "add_noise", "gumbel_scale"}

# Sentinel values handed to the production SearchConfig -> GumbelConfig mapping.
# Every GumbelConfig default is <= 99 apart from cpuct_base (38739), so nothing
# in the 701+ band can match by coincidence; ``_sentinel_search_config`` asserts
# that rather than trusting it.
_SENTINEL_BASE = 701
# Passed as ``simulations=``, which the arena owns; must not be a sentinel.
_PROBE_SIMULATIONS = 37


def _sentinel_search_config() -> tuple[SearchConfig, dict[float, str]]:
    """A ``SearchConfig`` whose every numeric field carries a unique sentinel.

    Returns the config and the sentinel -> attribute-name inverse map, so a
    ``GumbelConfig`` field holding a sentinel identifies the ``search`` knob
    that fed it.

    ⚑ Numeric fields only — a bool has no room for a sentinel. All six knobs the
    mapping carries today are numeric, but a future BOOL ``SearchConfig`` knob
    wired into ``GumbelConfig`` would land in ``_search_attrs_the_mapping_mentions``
    and NOT here, and fail the equality spuriously. Give it a sentinel by another
    means rather than deleting the assertion.
    """
    base = SearchConfig()
    replacements: dict[str, Any] = {}
    by_sentinel: dict[float, str] = {}
    for offset, field in enumerate(dataclasses.fields(SearchConfig)):
        current = getattr(base, field.name)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            continue
        sentinel = _SENTINEL_BASE + offset
        replacements[field.name] = type(current)(sentinel)
        by_sentinel[float(sentinel)] = field.name

    assert len(by_sentinel) == len(replacements), "the sentinels must be unique"
    defaults = {
        float(getattr(GumbelConfig(), f.name))
        for f in dataclasses.fields(GumbelConfig)
        if isinstance(getattr(GumbelConfig(), f.name), (int, float))
        and not isinstance(getattr(GumbelConfig(), f.name), bool)
    }
    assert not defaults & set(by_sentinel), "a sentinel collides with a GumbelConfig default"
    assert float(_PROBE_SIMULATIONS) not in by_sentinel
    return dataclasses.replace(base, **replacements), by_sentinel


def _selfplay_gumbel_fields_driven_by_search() -> dict[str, str]:
    """GumbelConfig field -> SearchConfig attribute, proved by CALLING the mapping.

    This used to parse ``network_turn.run_network_turn`` for an inline
    ``GumbelConfig(...)``. That construction is now the module-level, public
    ``build_selfplay_gumbel_config``, made addressable precisely so a test could
    invoke it — and invoking it is strictly stronger than parsing it: an AST
    scan cannot tell a field wired to ``search.x`` from one wired to a constant,
    and "a value accepted and then silently ignored" is this repo's signature
    defect. A field that comes back carrying its knob's sentinel has been
    observed to flow, at runtime, through the production mapping.
    """
    from chess_anti_engine.selfplay import network_turn

    search, by_sentinel = _sentinel_search_config()
    cfg = network_turn.build_selfplay_gumbel_config(
        search=search, game=GameConfig(), simulations=_PROBE_SIMULATIONS,
    )
    driven: dict[str, str] = {}
    for field in dataclasses.fields(GumbelConfig):
        value = getattr(cfg, field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        attr = by_sentinel.get(float(value))
        if attr is not None:
            driven[field.name] = attr
    return driven


def _search_attrs_the_mapping_mentions() -> set[str]:
    """Every ``search.<attr>`` the production mapping's own source NAMES.

    The counterpart to the call above: the call proves what flows, this proves
    what was meant to. A knob named here but missing from the call's result is
    exactly the accepted-then-ignored shape — mangled en route, or overwritten
    by a constant.
    """
    from chess_anti_engine.selfplay import network_turn

    src = inspect.getsource(network_turn.build_selfplay_gumbel_config)
    mentioned: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "GumbelConfig":
            continue
        for kw in node.keywords:
            if not kw.arg:
                continue
            match = re.search(r"\bsearch\.(\w+)", ast.unparse(kw.value))
            if match:
                mentioned.add(match.group(1))
    return mentioned


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = int(x.shape[0])
        return {
            "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.zeros((bs, 3), dtype=torch.float32),
        }


def _capture_c_search(monkeypatch: Any) -> list[dict[str, Any]]:
    """Record every kwarg set the C gumbel entry point is called with."""
    seen: list[dict[str, Any]] = []

    def fake_gumbel(_model, boards, **kwargs: Any):
        seen.append(kwargs)
        n = len(boards)
        return (
            [np.zeros(POLICY_SIZE, dtype=np.float32)] * n,
            [0] * n,
            [0.0] * n,
            [np.ones(POLICY_SIZE, dtype=bool)] * n,
        )

    monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", True)
    monkeypatch.setattr(match_mod, "_run_gumbel_root_many_c", fake_gumbel)
    return seen


def _side(**kwargs: Any) -> SideSearch:
    base = {
        "shape": "test",
        "source": "test",
        "gumbel": {},
        "vloss_weight": 0,
        "target_batch": 0,
    }
    base.update(kwargs)
    return SideSearch(**base)  # pyright: ignore[reportArgumentType]


def _play_one_arena(monkeypatch: Any, *, cand: SideSearch, ref: SideSearch) -> list[dict]:
    seen = _capture_c_search(monkeypatch)
    model = _DummyModel().eval()
    board = chess.Board()
    board.push_uci("e2e4")
    play_paired_games_matched_sims(
        model, model, [board],
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=2, temperature=0.0, gumbel_add_noise=False,
        search_candidate=cand, search_reference=ref,
    )
    assert seen, "the C search was never invoked"
    return seen


# ---------------------------------------------------------------------------
# Defect 1: the two C-path controls reach the search
# ---------------------------------------------------------------------------


def test_both_c_path_controls_reach_the_arena_search(monkeypatch: Any) -> None:
    """The plumbing proper. Absent kwargs = the C defaults of 0 = pre-C17."""
    side = _side(vloss_weight=5, target_batch=9)
    for kwargs in _play_one_arena(monkeypatch, cand=side, ref=side):
        assert kwargs.get("vloss_weight") == 5, "arena drops vloss_weight again"
        assert kwargs.get("target_batch") == 9, "arena drops target_batch again"


def test_the_two_sides_can_carry_different_virtual_loss(monkeypatch: Any) -> None:
    """The A/B this unblocks: same net, vloss on vs off, as a paired arena.

    Also catches a single shared value being threaded to both sides, which
    would look correct in the test above.
    """
    seen = _play_one_arena(
        monkeypatch,
        cand=_side(vloss_weight=1),
        ref=_side(vloss_weight=0),
    )
    assert {kw.get("vloss_weight") for kw in seen} == {0, 1}


def test_pick_moves_defaults_are_todays_behaviour_for_other_callers() -> None:
    """The in-loop gate and match_checkpoints must not change under this PR."""
    params = inspect.signature(match_mod.pick_moves_for_boards).parameters
    assert params["gumbel_vloss_weight"].default == 0
    assert params["gumbel_target_batch"].default == 0


def test_the_python_search_refuses_to_silently_drop_them() -> None:
    """The Python reference has no equivalent, so it must fail, not ignore.

    Quietly running without a requested virtual loss is the whole defect; on
    the fallback path (no C extension, or volatility search) the honest answer
    is an error.
    """
    with pytest.raises(ValueError, match="C-path only"):
        match_mod.pick_moves_for_boards(
            _DummyModel().eval(), [chess.Board()],
            device="cpu", rng=np.random.default_rng(0),
            mcts_type="gumbel", mcts_simulations=2, temperature=0.0, c_puct=2.5,
            gumbel_add_noise=False,
            volatility_q_scale=0.5,  # forces the Python path
            gumbel_vloss_weight=1,
        )


# ---------------------------------------------------------------------------
# Defect 2: the shape is explicit, and it is the shape it says it is
# ---------------------------------------------------------------------------


def test_the_shape_flag_has_no_default() -> None:
    """A default is how this became silent; ``None`` forces the choice."""
    import argparse

    p = argparse.ArgumentParser()
    add_common_args(p)

    assert p.parse_args([]).search_shape is None
    assert set(SEARCH_SHAPES) == {"play", "training"}


def test_matched_sims_refuses_to_run_without_an_explicit_shape() -> None:
    """run_arena is the single funnel, so the refusal belongs there.

    ``scripts/elo_vs_sims.py`` calls it too and had NO shape surface at all
    before this change — it silently ran a third shape (bare GumbelConfig at
    vloss 0).
    """
    with pytest.raises(SystemExit, match="--search-shape"):
        run_arena(
            candidate="c.pt", reference="r.pt", games=2,
            openings_path=None, openings_fen=None, opening_plies=16,
            mode="matched_sims", sims_candidate=2, sims_reference=2,
            ms_per_move=0, max_plies=2, temperature=0.0, gumbel_add_noise=False,
            device="cpu", seed=0, out_path=None,
        )


def test_matched_time_rejects_a_shape_it_cannot_apply() -> None:
    """UCI subprocesses carry their own shape; accepting the flag would lie."""
    with pytest.raises(SystemExit, match="matched_time"):
        run_arena(
            candidate="c.pt", reference="r.pt", games=2,
            openings_path=None, openings_fen=None, opening_plies=16,
            mode="matched_time", sims_candidate=2, sims_reference=2,
            ms_per_move=10, max_plies=2, temperature=0.0, gumbel_add_noise=False,
            device="cpu", seed=0, out_path=None,
            search_candidate=resolve_search_shape("play"),
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--search-shape", "play"),
        ("--cand-gumbel", "c_scale=0.5"),
        ("--ref-gumbel", "c_scale=0.5"),
        ("--cand-vloss-weight", "5"),
        ("--ref-vloss-weight", "5"),
        ("--cand-target-batch", "7"),
        ("--ref-target-batch", "7"),
    ],
)
def test_matched_time_rejects_every_in_process_search_flag(
    monkeypatch: Any, flag: str, value: str,
) -> None:
    """matched_time runs UCI subprocesses that build their own search.

    Accepting an in-process search flag there and running something else is the
    accepted-then-ignored defect this whole PR exists to remove — and four of
    these flags are NEW here, so leaving them inert would have introduced the
    defect while fixing it. ``--cand-gumbel`` had the hole on main already.
    """
    from scripts import arena_standard as arena_mod

    monkeypatch.setattr(sys, "argv", [
        "arena_standard.py",
        "--candidate", "c.pt", "--reference", "r.pt",
        "--mode", "matched_time", "--ms-per-move", "10", "--games", "2",
        flag, value,
    ])

    with pytest.raises(SystemExit, match=re.escape(flag)):
        arena_mod.main()


def test_matched_time_without_those_flags_still_gets_past_validation(
    monkeypatch: Any,
) -> None:
    """The rejection above must not pass by matched_time always exiting.

    Stops at the engine launch, which is as far as a test can go without two
    real checkpoints — the point is that validation let it through.
    """
    from scripts import arena_standard as arena_mod

    monkeypatch.setattr(sys, "argv", [
        "arena_standard.py",
        "--candidate", "c.pt", "--reference", "r.pt",
        "--mode", "matched_time", "--ms-per-move", "10", "--games", "2",
    ])
    reached: list[str] = []

    def fake_run_arena(**kwargs: Any) -> dict:
        reached.append(str(kwargs["mode"]))
        return {}

    monkeypatch.setattr(arena_mod, "run_arena", fake_run_arena)
    arena_mod.main()

    assert reached == ["matched_time"]


def test_the_play_shape_is_the_engine_play_shape() -> None:
    side = resolve_search_shape("play")

    for key, want in PLAY_SEARCH_DEFAULTS.items():
        assert side.realized_gumbel()[key] == want, key
    assert side.vloss_weight == PLAY_SEARCH_VLOSS_WEIGHT
    assert side.target_batch == PLAY_SEARCH_TARGET_BATCH


def test_the_training_shape_uses_the_linear_root_not_the_play_log_root() -> None:
    """The root TRANSFORM is what still separates the two shapes.

    This test used to also assert ``training["topk"] != play["topk"]``, on the
    theory that the shapes must differ on every knob. They need not, and today
    they do not: ``ed9de8ee9`` (2026-08-06) moved production selfplay to
    ``gumbel_topk: 32`` — the same breadth the play shape uses — deliberately.
    A shape difference is not the invariant; carrying production's actual value
    is, and ``test_the_training_shape_is_what_the_worker_would_actually_run``
    (equality against ``production_selfplay_search_config()``) plus
    ``test_a_config_value_actually_reaches_the_arenas_training_shape`` (the
    non-vacuous yaml->arena discriminator) already cover that. This one keeps
    only the separation it uniquely owns.

    On the old docstring's sims-ladder worry: ``topk`` is capped in
    ``mcts/gumbel.py::_select_top_m_with_gumbel`` (called by
    ``_init_board_search_state``) by ``m_cap = max(2, (sim_budget + 1) // 2)``,
    so the realized breadth is ``m = min(topk, m_cap, legal moves)``:

        ``topk`` BINDS iff ``sim_budget > 2 * topk``.
        ``topk`` is INERT iff ``sim_budget <= 2 * topk`` (m_cap binds).

    Stated as an inequality on purpose — the prose form ("changes nothing below
    2x the budget") was written backwards twice, and it is the one sentence a
    ladder designer acts on. At the ratchet's 32 sims, ``topk`` 16 and 32 both
    give ``m = 16`` (32 <= 2*16, inert either way); at 256 sims ``topk`` 32
    gives ``m = 32`` (256 > 64, binds). So a sims ladder only varies breadth
    across rungs where ``m_cap`` stops binding — read breadth off the
    ``min(...)`` above per rung, not off ``topk``.
    """
    training = resolve_search_shape("training").realized_gumbel()
    play = resolve_search_shape("play").realized_gumbel()

    # Linear root, i.e. the sentinels — NOT the play shape's log root.
    assert training["c_scale_root"] == GumbelConfig().c_scale_root
    assert training["q_visit_exp_root"] == GumbelConfig().q_visit_exp_root
    assert training["c_visit_root"] == GumbelConfig().c_visit_root
    # ...and `play` really does transform the root differently, so the three
    # asserts above separate two shapes rather than restating a dataclass
    # default against nothing.
    assert (play["c_scale_root"], play["q_visit_exp_root"], play["c_visit_root"]) != (
        training["c_scale_root"],
        training["q_visit_exp_root"],
        training["c_visit_root"],
    )


def test_the_training_shape_is_what_the_worker_would_actually_run() -> None:
    """End to end against the real publish channel, not against constants.

    Rebuilt here independently of the script helper: production yaml ->
    validator -> ``build_recommended_worker`` -> the worker's own
    ``_build_selfplay_configs``. If a knob is not published to workers, both
    sides land on the worker default and the arena is still right.
    """
    search = production_selfplay_search_config()
    side = resolve_search_shape("training")

    assert side.realized_gumbel()["c_scale"] == pytest.approx(search.gumbel_c_scale)
    assert side.realized_gumbel()["topk"] == search.gumbel_topk
    assert side.vloss_weight == search.gumbel_vloss_weight
    assert side.target_batch == search.gumbel_target_batch


def test_a_config_value_actually_reaches_the_arenas_training_shape(
    monkeypatch: Any, tmp_path: Any,
) -> None:
    """The discriminating test: change the yaml, the arena must follow.

    The comparison above is only as strong as the config it runs against: any
    knob whose production value happens to equal the ``GumbelConfig`` /
    ``SearchConfig`` default is checked by a comparison that would also pass if
    the yaml never reached the arena at all, and which knobs those are moves
    with every config edit. A check that can silently stop being a check is not
    a check — this one runs the whole channel against values chosen to differ
    from every default in sight, and asserts that at the bottom.
    """
    import yaml as _yaml

    from chess_anti_engine.utils.config_yaml import load_yaml_file
    from scripts import arena_standard as arena_mod

    raw = load_yaml_file(str(arena_mod.PRODUCTION_CONFIG))
    raw["selfplay"]["gumbel_vloss_weight"] = 2
    raw["selfplay"]["gumbel_topk"] = 24
    raw["selfplay"]["gumbel_c_scale"] = 0.077
    raw["selfplay"]["gumbel_policy_temp"] = 1.7
    patched = tmp_path / "patched.yaml"
    patched.write_text(_yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(arena_mod, "PRODUCTION_CONFIG", patched)

    side = arena_mod.resolve_search_shape("training")

    assert side.vloss_weight == 2, "the published vloss_weight never reached the arena"
    assert side.realized_gumbel()["topk"] == 24
    assert side.realized_gumbel()["c_scale"] == pytest.approx(0.077)
    # ⚑ policy_temp must be checked BY VALUE here, not only for key presence.
    # `test_the_realized_view_is_not_just_the_override_dict` pins the key SET, so
    # it catches a deleted `policy_temp` and NOT a hard-coded one; and the drift
    # guard's `pinned` is also `set(...gumbel)`, so a constant leaves the key
    # present and the "provably inert today" branch unreachable. With production
    # about to run T=1.5, a collapsed-to-1.0 line here would make every
    # `--search-shape training` arena — the Cheese-tail yardstick included —
    # measure a sharper prior than the live run trains on, with CI green.
    assert side.realized_gumbel()["policy_temp"] == pytest.approx(1.7), (
        "the published gumbel_policy_temp never reached the arena's training "
        "shape: the arena would search at a temperature production does not run"
    )
    # None of the four is a default, so none can pass by accident.
    assert GumbelConfig().topk != 24
    assert GumbelConfig().c_scale != 0.077
    assert GumbelConfig().policy_temp != 1.7
    assert SearchConfig().gumbel_vloss_weight != 2


def test_every_config_driven_knob_reaches_the_arena_or_is_provably_inert() -> None:
    """Drift guard: a NEW yaml-driven search knob must reach the arena too.

    Builds the ``GumbelConfig`` production selfplay searches with — by CALLING
    ``network_turn.build_selfplay_gumbel_config``, the real mapping — and takes
    every field it fills from ``search.*``: not just ``search.gumbel_*``, since
    the same call is also fed ``search.volatility_q_scale/_fpu/_anchor`` and a
    narrower scan would call itself a coverage guard while missing three knobs.

    A field the training shape does not carry passes only if production's value
    for it equals the ``GumbelConfig`` default, i.e. omitting it is provably a
    no-op TODAY. Turning such a knob on in the yaml fails this test, which is
    the intent: the arena would otherwise keep measuring the old shape.
    """
    config_driven = _selfplay_gumbel_fields_driven_by_search()

    # PER-FIELD pins, not a count. A `len(...) >= N` floor lets any ONE knob be
    # silently hard-coded as long as another is added, which is how the first
    # version of this re-pointing ended up WEAKER than the AST test it replaced:
    # the mapping carries six fields, the floor said five, so hard-coding
    # `c_scale` passed here while still failing on `main`. Every entry below is
    # a knob whose value must be OBSERVED to flow (see the helper: sentinel in,
    # sentinel out), so hard-coding any one of them fails this line by name.
    assert set(config_driven.values()) == {
        "gumbel_topk",
        "gumbel_policy_temp",
        "gumbel_c_scale",
        "volatility_q_scale",
        "volatility_fpu",
        "volatility_anchor",
    }, (
        f"the selfplay search is driven by {sorted(set(config_driven.values()))}; "
        "if that is a deliberate change, update this set and check the arena's "
        "training shape still carries every knob production sets"
    )
    # Separately: what the mapping NAMES must equal what it CARRIES.
    #
    # ⚑ This does NOT catch a hard-coded field. Both sides derive from the same
    # source, so deleting `search.x` from the call deletes it from `mentioned`
    # too and the equality passes vacuously — the set assertion above is what
    # catches that. What this DOES catch is the named-but-mangled case
    # (`float(search.gumbel_policy_temp) * 0.0 + 1.0`), where the source still
    # reads as wired and the value still does not arrive. That case is invisible
    # to any source-reading test, which is the whole reason the mapping was made
    # callable.
    mentioned = _search_attrs_the_mapping_mentions()
    assert mentioned == set(config_driven.values()), (
        f"build_selfplay_gumbel_config names {sorted(mentioned)} but only "
        f"{sorted(set(config_driven.values()))} reach the returned GumbelConfig"
    )
    search = production_selfplay_search_config()
    pinned = set(resolve_search_shape("training").gumbel) | _ARENA_OWNED_GUMBEL_FIELDS
    live_and_unpinned = {
        field: (getattr(search, attr), getattr(GumbelConfig(), field))
        for field, attr in config_driven.items()
        if field not in pinned and getattr(search, attr) != getattr(GumbelConfig(), field)
    }
    assert not live_and_unpinned, (
        f"production selfplay configures {sorted(live_and_unpinned)} away from the "
        f"GumbelConfig default {live_and_unpinned}, but the arena's training shape "
        "does not carry it -- --search-shape training would measure a search "
        "production does not run"
    )


def test_an_override_layers_on_top_of_the_shape_and_is_recorded() -> None:
    """``--cand-gumbel`` still works, and the provenance string says so."""
    base = resolve_search_shape("training")
    side = apply_search_overrides(base, spec="c_scale=0.05", vloss_weight=4)

    assert side.realized_gumbel()["c_scale"] == 0.05
    # Untouched by the override — compared against the BASE shape, not against
    # a literal: the literal here was production's topk, and it went stale.
    assert side.realized_gumbel()["topk"] == base.realized_gumbel()["topk"]
    assert side.vloss_weight == 4
    assert "c_scale=0.05" in side.source
    assert "vloss_weight=4" in side.source


def test_an_unlayered_shape_reports_no_cli_provenance() -> None:
    """The test above must not pass by every source string containing 'CLI'."""
    assert "CLI" not in apply_search_overrides(resolve_search_shape("training")).source


def test_a_malformed_override_is_rejected() -> None:
    with pytest.raises(SystemExit, match="expected k=v pairs"):
        apply_search_overrides(resolve_search_shape("play"), spec="c_scale")


def test_an_override_naming_a_nonexistent_knob_is_rejected_at_parse_time() -> None:
    """A typo must not survive until ``dataclasses.replace`` blows up.

    That happens minutes in, after both checkpoints have loaded and compiled.
    """
    with pytest.raises(SystemExit, match="not a GumbelConfig field"):
        apply_search_overrides(resolve_search_shape("play"), spec="c_scal=0.05")


def test_the_record_stores_the_search_that_produced_the_elo() -> None:
    """A result row must be re-interpretable years later without its argv."""
    summary = summarize_pentanomial(pentanomial_counts([2.0, 1.0]))
    training = resolve_search_shape("training")
    record = build_result_record(
        summary,
        mode="matched_sims", candidate="c.pt", reference="r.pt",
        openings_path="book.pgn.zip", opening_plies=16,
        sims_candidate=32, sims_reference=32, ms_per_move=None,
        temperature=0.1, gumbel_add_noise=True, max_plies=10,
        seed=0, device="cpu", duration_s=0.0,
        search_candidate=training,
        search_reference=resolve_search_shape("play"),
    )

    assert record["search_candidate"]["shape"] == "training"
    assert record["search_reference"]["shape"] == "play"
    assert record["search_reference"]["vloss_weight"] == PLAY_SEARCH_VLOSS_WEIGHT
    # The REALIZED knobs, not just the ones that were overridden: a reader must
    # be able to see c_scale on a row that never passed --cand-gumbel. Checked
    # against the shape that was handed in (and, for play, its constant) rather
    # than against production's number of the day, which goes stale.
    assert record["search_candidate"]["gumbel"]["c_scale"] == pytest.approx(
        training.realized_gumbel()["c_scale"]
    )
    assert record["search_reference"]["gumbel"]["c_scale"] == pytest.approx(
        PLAY_SEARCH_DEFAULTS["c_scale"]
    )
    assert record["search_candidate"]["source"]


def test_the_realized_view_is_not_just_the_override_dict() -> None:
    """A sparse override dict is what made the old runs unreadable."""
    side = resolve_search_shape("training")

    assert set(side.gumbel) == {"c_scale", "topk", "policy_temp"}
    assert set(side.realized_gumbel()) >= set(PLAY_SEARCH_DEFAULTS)
    # A knob the training shape never overrode still resolves to its realized
    # value (the GumbelConfig default), which is the point of the view.
    assert side.realized_gumbel()["c_visit"] == GumbelConfig().c_visit


# --- F1: cold-vs-warm tree, play-path audit 2026-08-03 --------------------


def test_every_arena_side_records_that_it_searches_a_cold_tree() -> None:
    """Production selfplay carries the tree across plies; no arena does.

    ``selfplay/network_turn.py:799-801`` passes ``tree``/``root_node_ids`` and
    advances the root with ``find_child`` after each ply;
    ``selfplay/match.pick_moves_for_boards`` — the ONLY search entry point this
    script, ``chess_anti_engine/arena.py`` and the match scripts use — passes
    neither. Measured (audit repro, 256 nominal sims): cold roots see 256
    visits / max_visit 60 on every ply, warm ones 315-363 / 77-108, so the root
    value transform ``q_scale = c_scale*(c_visit + max_visit)`` is up to +44%
    sharper in production. ``matched_sims`` matches the nominal budget, not the
    visit counts the returned policy is built from.

    This PR records the divergence rather than closing it: a tree carry in
    ``pick_moves_for_boards`` changes arena behaviour and needs its own
    pre-registered readout.
    """
    for shape in SEARCH_SHAPES:
        side = resolve_search_shape(shape)
        assert side.tree_reuse == "cold", shape
        assert "tree_reuse=cold" in side.describe(), shape
        assert side.as_record()["tree_reuse"] == "cold", shape


def test_pick_moves_for_boards_really_does_pass_no_tree() -> None:
    """The claim the `tree_reuse=cold` label rests on, asserted at the source.

    If someone gives ``pick_moves_for_boards`` a tree carry, this fails and the
    label has to be re-derived rather than silently becoming a lie.
    """
    src = inspect.getsource(match_mod.pick_moves_for_boards)
    assert "tree=" not in src
    assert "root_node_ids" not in src


def test_an_override_carries_the_tree_reuse_label_through() -> None:
    base = resolve_search_shape("play")
    side = apply_search_overrides(base, spec="topk=8", vloss_weight=1)
    assert side.tree_reuse == base.tree_reuse
    assert "tree_reuse=cold" in side.describe()
