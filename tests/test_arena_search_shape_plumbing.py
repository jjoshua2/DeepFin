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
    ARENA_OWNED_GUMBEL_FIELDS,
    CHECKPOINT_OWNED_GUMBEL_FIELDS,
    SEARCH_SHAPES,
    SideSearch,
    add_common_args,
    apply_search_overrides,
    build_result_record,
    pentanomial_counts,
    play_paired_games_matched_sims,
    production_selfplay_gumbel_config,
    production_selfplay_search_config,
    resolve_search_shape,
    run_arena,
    summarize_pentanomial,
    training_shape_carried_fields,
)

# GumbelConfig fields the ARENA owns rather than the training config: the sim
# budget and the move-selection/noise policy come from arena flags. Everything
# else production sets from config has to be carried by the training shape.
_ARENA_OWNED_GUMBEL_FIELDS = set(ARENA_OWNED_GUMBEL_FIELDS)

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

    ⚑ Numeric fields only — a bool has no room for a sentinel. Bool knobs are
    covered instead by ``_bool_gumbel_fields_driven_by_search`` below, which
    flips ONE at a time and watches which ``GumbelConfig`` field follows: the
    same "observed to flow" standard, by differencing rather than by a marker
    value. Both results are merged in
    ``_selfplay_gumbel_fields_driven_by_search``, so a bool knob wired into the
    mapping is proved to arrive rather than being skipped as unsentinelable.
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


def _bool_gumbel_fields_driven_by_search() -> dict[str, str]:
    """GumbelConfig BOOL field -> SearchConfig bool attribute, by differencing.

    A bool cannot carry a sentinel, so flip exactly one ``SearchConfig`` bool at
    a time and record which ``GumbelConfig`` bool changes with it. That is the
    same standard the numeric probe holds itself to — the value is OBSERVED to
    arrive through the real mapping — and it is what keeps a target-only flag
    like ``gumbel_target_untempered_prior`` inside the coverage set instead of
    silently exempt because of its type.
    """
    from chess_anti_engine.selfplay import network_turn

    def _mapped(search: SearchConfig) -> dict[str, bool]:
        cfg = network_turn.build_selfplay_gumbel_config(
            search=search, game=GameConfig(), simulations=_PROBE_SIMULATIONS,
        )
        return {
            f.name: bool(getattr(cfg, f.name))
            for f in dataclasses.fields(GumbelConfig)
            if isinstance(getattr(cfg, f.name), bool)
        }

    base = SearchConfig()
    baseline = _mapped(base)
    driven: dict[str, str] = {}
    for field in dataclasses.fields(SearchConfig):
        if not isinstance(getattr(base, field.name), bool):
            continue
        flipped = _mapped(
            dataclasses.replace(base, **{field.name: not getattr(base, field.name)}),
        )
        for name, value in flipped.items():
            if value == baseline[name]:
                continue
            assert name not in driven, (
                f"GumbelConfig.{name} follows two different SearchConfig bools "
                f"({driven[name]} and {field.name}); the mapping is ambiguous"
            )
            driven[name] = field.name
    return driven


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
    driven.update(_bool_gumbel_fields_driven_by_search())
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
        # TARGET-side only. It changes the STORED improved policy and leaves the
        # played move on the uncapped sigma, so an arena -- which plays games and
        # stores no training rows -- cannot observe it. It is carried into the
        # arena's training shape anyway (below), because "provably inert here"
        # must be re-proved by the `live_and_unpinned` clause rather than
        # asserted once in a comment.
        "gumbel_target_max_visit_cap",
        # Same shape, other term of the same softmax: it undoes
        # policy_temp on the STORED target's log_prior and leaves the
        # played move on the tempered arm, so an arena cannot observe it
        # either -- and is carried into the training shape anyway, for
        # the `live_and_unpinned` clause below to re-prove.
        "gumbel_target_untempered_prior",
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
    # ⚑ HONEST NOTE ON THIS LAST CLAUSE. Since the training shape became the
    # DERIVED complement, `pinned` covers every carried field by construction, so
    # `live_and_unpinned` can no longer be non-empty and this clause is a
    # tautology. It is kept because its FIRST half (the `config_driven` set
    # equality above) still discriminates -- it is what proves each knob is
    # OBSERVED to flow through the real mapping -- and because deleting the
    # clause would delete the message that finally named the defect. Do NOT
    # read a green here as evidence that the arena carries production's shape:
    # `test_the_training_shape_equals_productions_gumbel_config_field_by_field`
    # and `test_the_three_ownership_sets_partition_gumbelconfig` are the checks
    # that can actually fail, and the mutation test above proves they do.
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


# ---------------------------------------------------------------------------
# Defect 3: the training shape is DERIVED from production, not restated
#
# `c62eb8ff2` (2026-08-10 23:28) promoted `gumbel_target_max_visit_cap: 5` and
# `gumbel_target_untempered_prior: true` into the production yaml. The training
# shape restated three knobs by hand, the hand-written list did not grow, and
# `--search-shape training` kept announcing itself as production's search while
# building a GumbelConfig production does not build. The guard above detected it
# immediately and the failure went unread for five days.
#
# The fix is structural: the carried set is now the COMPLEMENT of two justified
# exclusion sets, so a promoted knob needs no edit. What is left to test is that
# the complement really PARTITIONS the dataclass (a new field cannot fall
# between the sets), that the derived shape really equals production's config
# field by field, and — the part that keeps this from being a gate that cannot
# fail — that dropping any single live knob makes the equality FAIL.
# ---------------------------------------------------------------------------


# The exact carried set, pinned BY NAME. A `len(...)` floor would let one knob be
# dropped as long as another arrived; that is the failure mode the guard above
# already documents. A new GumbelConfig field fails this line by name, which is
# the intended forcing function: classify it as arena-owned, checkpoint-owned, or
# carried, and say which in the PR.
_EXPECTED_CARRIED_FIELDS = {
    "topk",
    "policy_temp",
    "c_visit",
    "c_scale",
    "c_puct",
    "cpuct_factor",
    "cpuct_base",
    "fpu_reduction",
    "q_visit_exp",
    "q_global_scale",
    "q_visit_floor",
    "target_max_visit_cap",
    "target_untempered_prior",
    # target-construction property like the two above: a TRAINING-shape fact
    # the arena carries on both sides, owned by neither arm (2026-08-20 probe).
    "target_q_rescale",
    "halving_div",
    "c_visit_root",
    "c_scale_root",
    "q_visit_exp_root",
    "full_tree",
    "volatility_q_scale",
    "volatility_fpu",
    "volatility_anchor",
    "volatility_factor_clip",
}


def test_the_three_ownership_sets_partition_gumbelconfig() -> None:
    """Every GumbelConfig field is arena-owned, checkpoint-owned, or carried.

    ⚑ This is the acceptance bar in one line. A NEW field lands in `carried` by
    construction (it is the complement), so the arena starts carrying it with no
    edit — and this test fails by name so the addition is still looked at rather
    than absorbed silently. A field can never fall between the sets, which is how
    `target_max_visit_cap` fell between them before.
    """
    all_fields = {f.name for f in dataclasses.fields(GumbelConfig)}
    carried = set(training_shape_carried_fields())

    # ⚑ All THREE sets are pinned by name, not just `carried`. `carried` is the
    # complement, so widening an exclusion set silently narrows it: dropping a
    # newly promoted knob into ARENA_OWNED would restore exactly the drift this
    # fix removes, and would leave `carried` looking untouched. Pinning the
    # exclusions is what makes the misclassification escape hatch fail too.
    assert set(ARENA_OWNED_GUMBEL_FIELDS) == {
        "simulations", "temperature", "add_noise", "gumbel_scale",
    }, (
        "a GumbelConfig field was declared ARENA-owned. That EXCLUDES it from "
        "production-parity, so it must be a property of the MATCH (budget, "
        "move-selection, root noise) and not of the training config. Justify it "
        "in the PR before updating this pin."
    )
    assert set(CHECKPOINT_OWNED_GUMBEL_FIELDS) == {
        "input_history_encoding", "input_extra_features",
        "policy_encoding", "compute_relations",
    }, (
        "a GumbelConfig field was declared CHECKPOINT-owned. That EXCLUDES it "
        "from production-parity, so it must be something `pick_moves_for_boards` "
        "genuinely reads off the loaded model. Justify it in the PR."
    )
    assert carried == _EXPECTED_CARRIED_FIELDS, (
        "GumbelConfig's field set moved. The training shape carries the "
        "complement of ARENA_OWNED_GUMBEL_FIELDS | CHECKPOINT_OWNED_GUMBEL_FIELDS, "
        "so a new field is carried automatically -- confirm that is right for it "
        "(is it a MATCH property? a CHECKPOINT property?) and update this pin."
    )
    # Disjoint, and together exhaustive.
    assert not (ARENA_OWNED_GUMBEL_FIELDS & CHECKPOINT_OWNED_GUMBEL_FIELDS)
    assert not (carried & ARENA_OWNED_GUMBEL_FIELDS)
    assert not (carried & CHECKPOINT_OWNED_GUMBEL_FIELDS)
    assert (
        carried | set(ARENA_OWNED_GUMBEL_FIELDS) | set(CHECKPOINT_OWNED_GUMBEL_FIELDS)
    ) == all_fields
    # Not a gate that cannot fail: it examines a real, non-trivial number of
    # fields, and the exclusions are a small minority of them.
    assert len(carried) >= 20
    assert len(carried) > len(ARENA_OWNED_GUMBEL_FIELDS | CHECKPOINT_OWNED_GUMBEL_FIELDS)


def _arena_training_gumbel_config() -> GumbelConfig:
    """The GumbelConfig an arena move ACTUALLY searches with, training shape.

    Rebuilt the way ``selfplay/match.pick_moves_for_boards`` builds it — base
    dataclass, then ``dataclasses.replace`` with the shape's dict — rather than
    inspecting the dict directly. A knob present in the dict but rejected by
    ``replace`` would pass a dict-only check and still never reach the search.
    """
    side = resolve_search_shape("training")
    return dataclasses.replace(GumbelConfig(), **side.gumbel)


def test_the_training_shape_equals_productions_gumbel_config_field_by_field() -> None:
    """Exhaustive, not three knobs: every carried field, compared by value."""
    prod = production_selfplay_gumbel_config()
    arena = _arena_training_gumbel_config()

    mismatched = {
        name: (getattr(prod, name), getattr(arena, name))
        for name in training_shape_carried_fields()
        if getattr(prod, name) != getattr(arena, name)
    }
    assert not mismatched, (
        f"the arena's training shape differs from production selfplay on "
        f"{sorted(mismatched)} (production, arena) = {mismatched}; "
        "--search-shape training would measure a search production does not run"
    )


def _fields_where_production_differs_from_the_dataclass_default() -> list[str]:
    """Carried fields whose production value is NOT the GumbelConfig default.

    These are the only fields on which the equality test above can discriminate:
    for a field sitting at its default, dropping it from the carried set changes
    nothing and the test would pass either way. Naming them explicitly is what
    turns "the test passes" into "the test could have failed".
    """
    prod = production_selfplay_gumbel_config()
    base = GumbelConfig()
    return [
        name
        for name in training_shape_carried_fields()
        if getattr(prod, name) != getattr(base, name)
    ]


def test_the_equality_check_is_not_vacuous_and_catches_a_dropped_knob() -> None:
    """MUTATION, in-suite: drop each live knob and watch the equality go RED.

    ⚑ Without this the test above is satisfied by a training shape that carries
    NOTHING on any day production happens to sit at every dataclass default. The
    repo's standing rule is that a new test is vacuous until mutated, and the
    fixture — not the assertion — is usually what fails to discriminate. So:
    enumerate the fields that actually discriminate today, assert the list is
    non-empty and contains the two knobs whose drift motivated this, and then
    prove that removing ANY ONE of them from the carried dict makes the
    comparison fail.
    """
    live = _fields_where_production_differs_from_the_dataclass_default()

    assert live, (
        "production sits at the GumbelConfig default on EVERY carried field, so "
        "the equality test cannot discriminate a carried shape from an empty "
        "one. This is a gate that cannot fail -- do not read it as a pass."
    )
    # The two knobs this whole fix exists for. Pinned so that a yaml revert
    # turns this into a visible decision rather than silently defusing the test.
    assert {"target_max_visit_cap", "target_untempered_prior"} <= set(live), (
        f"production no longer sets the two knobs the drift was found on; live "
        f"discriminating fields are {live}. If the yaml genuinely reverted them, "
        "update this pin -- but re-check that some other knob still discriminates."
    )

    prod = production_selfplay_gumbel_config()
    full = dict(resolve_search_shape("training").gumbel)
    for name in live:
        dropped = {k: v for k, v in full.items() if k != name}
        mutated = dataclasses.replace(GumbelConfig(), **dropped)
        assert getattr(mutated, name) != getattr(prod, name), (
            f"dropping {name!r} from the training shape left the arena config "
            "unchanged, so the equality test cannot see that knob at all"
        )


def test_a_newly_promoted_knob_is_carried_without_editing_the_arena(
    monkeypatch: Any, tmp_path: Any,
) -> None:
    """The structural claim, exercised: promote a knob, the arena follows.

    Uses a knob the production yaml leaves at its default today, so the value
    that arrives cannot be the one that was already there. This is the property
    the old hand-written three-knob list did not have and could not have.
    """
    import yaml as _yaml

    from chess_anti_engine.utils.config_yaml import load_yaml_file
    from scripts import arena_standard as arena_mod

    assert production_selfplay_gumbel_config().target_max_visit_cap == 5
    before = resolve_search_shape("training").gumbel["target_max_visit_cap"]
    assert before == 5

    raw = load_yaml_file(str(arena_mod.PRODUCTION_CONFIG))
    raw["selfplay"]["gumbel_target_max_visit_cap"] = 11
    patched = tmp_path / "promoted.yaml"
    patched.write_text(_yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(arena_mod, "PRODUCTION_CONFIG", patched)

    side = arena_mod.resolve_search_shape("training")
    assert side.gumbel["target_max_visit_cap"] == 11
    # ...and it survives the `dataclasses.replace` the real search path uses.
    assert dataclasses.replace(GumbelConfig(), **side.gumbel).target_max_visit_cap == 11
    assert GumbelConfig().target_max_visit_cap != 11


def test_the_runtime_check_refuses_a_training_shape_that_drifted(
    monkeypatch: Any,
) -> None:
    """The live-tree gate: `resolve_search_shape` refuses to hand back a lie.

    CI reads the COMMITTED yaml; an arena reads whichever yaml is in its tree,
    and the live one is edited in place and lags. So the check that matters runs
    at arena start-up, not only in pytest. Mutated here by making the derivation
    drop a knob — the same shape as the hand-written list that caused this.
    """
    from scripts import arena_standard as arena_mod

    full = tuple(arena_mod.training_shape_carried_fields())
    assert "target_max_visit_cap" in full
    monkeypatch.setattr(
        arena_mod,
        "training_shape_carried_fields",
        lambda: tuple(f for f in full if f != "target_max_visit_cap"),
    )

    with pytest.raises(SystemExit, match="target_max_visit_cap"):
        arena_mod.resolve_search_shape("training")


def test_the_runtime_check_is_silent_when_the_shape_is_right() -> None:
    """Negative control: the refusal above must not fire on every call."""
    assert resolve_search_shape("training").shape == "training"


def test_an_explicit_volatility_request_survives_the_training_shape() -> None:
    """REGRESSION (caught in review of this PR, not by me).

    Making the training shape exhaustive made it carry `volatility_q_scale` /
    `volatility_fpu` / `volatility_anchor` at production's values (0.0 today).
    `pick_moves_for_boards` applies `gumbel_overrides` AFTER the dedicated
    volatility arguments, so the shape silently reset an explicit
    `--volatility-q-scale` back to zero, kept the run on the C path, and would
    have reported a volatility arena that ran no volatility search -- the exact
    accepted-then-ignored defect this module exists to stop, reintroduced by the
    fix for another instance of it.

    ⚑ Asserted on the MERGED dict the search actually receives, not on the flags,
    because the flags were always correct; it was the precedence that was not.
    """
    from scripts.arena_standard import overrides_with_volatility

    side = resolve_search_shape("training")
    # The shape genuinely carries these -- otherwise this test guards nothing.
    assert side.gumbel["volatility_q_scale"] == 0.0
    assert side.gumbel["volatility_fpu"] == 0.0

    merged = overrides_with_volatility(
        side, {"volatility_q_scale": 0.5, "volatility_fpu": 0.25,
               "volatility_anchor": 0.07},
    )
    assert merged["volatility_q_scale"] == 0.5, "the shape clobbered the request"
    assert merged["volatility_fpu"] == 0.25
    assert merged["volatility_anchor"] == 0.07
    # ...and it does not quietly drop the rest of the shape while doing so.
    assert merged["target_max_visit_cap"] == side.gumbel["target_max_visit_cap"]
    assert merged["c_scale"] == side.gumbel["c_scale"]

    # No request => the shape's own values, unchanged.
    assert overrides_with_volatility(side, None) == side.gumbel
    assert overrides_with_volatility(side, {}) == side.gumbel


def test_volatility_reaches_the_search_through_the_merged_overrides(
    monkeypatch: Any,
) -> None:
    """End of the wire, not the helper: what does the SEARCH receive?

    `overrides_with_volatility` returning the right dict proves nothing if a
    play loop still passes `side.gumbel` straight down. This drives the real
    matched_sims loop and reads the config off the search entry point.
    """
    seen = _capture_c_search(monkeypatch)
    captured: list[Any] = []

    real = match_mod.pick_moves_for_boards

    def spy(*args: Any, **kwargs: Any):
        captured.append(dict(kwargs.get("gumbel_overrides") or {}))
        return real(*args, **kwargs)

    # The play loop imports it from `selfplay.match` at CALL time, so patching
    # the module attribute is what the loop will actually pick up.
    monkeypatch.setattr(match_mod, "pick_moves_for_boards", spy)
    from scripts import arena_standard as arena_mod

    model = _DummyModel().eval()
    board = chess.Board()
    board.push_uci("e2e4")
    side = _side(gumbel={"c_scale": 0.1, "volatility_q_scale": 0.0})
    arena_mod.play_paired_games_matched_sims(
        model, model, [board],
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=2, temperature=0.0, gumbel_add_noise=False,
        search_candidate=side, search_reference=side,
        volatility_candidate={"volatility_q_scale": 0.5},
    )

    assert seen, "the search was never invoked"
    assert captured, "pick_moves_for_boards was never called"
    # The CANDIDATE side must carry the request; at least one call has it.
    assert any(c.get("volatility_q_scale") == 0.5 for c in captured), (
        f"no call carried the explicit volatility request; saw {captured}"
    )


def test_the_record_shows_the_target_only_knobs_it_ran() -> None:
    """A banked row must be re-readable as to WHICH regime it measured.

    The six rows banked between 2026-08-10 and this fix cannot be: their
    ``gumbel`` dict lists seven knobs and neither target-side one is among them,
    so nothing in the record distinguishes production's shape from the arena's.
    """
    realized = resolve_search_shape("training").realized_gumbel()

    assert "target_max_visit_cap" in realized
    assert "target_untempered_prior" in realized
    assert realized["target_max_visit_cap"] == 5
    assert realized["target_untempered_prior"] is True


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

    # The training shape carries the DERIVED complement, so this is the full
    # carried set rather than the three knobs it used to restate by hand.
    assert set(side.gumbel) == set(training_shape_carried_fields())
    assert set(side.realized_gumbel()) >= set(PLAY_SEARCH_DEFAULTS)
    # A knob the shape never overrode still resolves to its realized value (the
    # GumbelConfig default), which is the point of the view. Both real shapes now
    # set every PLAY_SEARCH_DEFAULTS key explicitly — `play` from that dict,
    # `training` from the derived complement — so the fallback branch has to be
    # exercised on a shape that omits one, or this clause asserts nothing.
    sparse = _side(gumbel={"c_scale": 0.05})
    assert "c_visit" not in sparse.gumbel
    assert sparse.realized_gumbel()["c_visit"] == GumbelConfig().c_visit
    assert sparse.realized_gumbel()["c_scale"] == 0.05


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
