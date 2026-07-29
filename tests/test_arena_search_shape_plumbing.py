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
   (c_scale 0.025, topk 32, log root) instead of training's (0.1, 16, linear).

The shape is now a required choice with no default, and the realized values are
printed and stored in the JSONL record.
"""
from __future__ import annotations

import ast
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
from chess_anti_engine.selfplay.config import SearchConfig
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


def test_the_training_shape_is_not_the_play_shape() -> None:
    """The regression that invalidated the sims ladder.

    Root breadth (topk) and the root transform are the two knobs a sims ladder
    is most sensitive to: under the play shape the root candidate set doubles
    between the 32- and 256-sim rungs, so the ladder varies breadth AND depth.
    """
    training = resolve_search_shape("training").realized_gumbel()
    play = resolve_search_shape("play").realized_gumbel()

    assert training["c_scale"] != play["c_scale"]
    assert training["topk"] != play["topk"]
    # Linear root, i.e. the sentinels — NOT the play shape's log root.
    assert training["c_scale_root"] == GumbelConfig().c_scale_root
    assert training["q_visit_exp_root"] == GumbelConfig().q_visit_exp_root
    assert training["c_visit_root"] == GumbelConfig().c_visit_root


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

    The comparison above cannot fail on today's shipped config, because every
    value it checks happens to equal the dataclass default (``gumbel_topk: 16``
    is ``GumbelConfig.topk``, and ``main``'s yaml does not set
    ``gumbel_vloss_weight`` at all). A check that cannot fail is not a check —
    this one runs the whole channel against values chosen to differ from every
    default in sight.
    """
    import yaml as _yaml

    from chess_anti_engine.utils.config_yaml import load_yaml_file
    from scripts import arena_standard as arena_mod

    raw = load_yaml_file(str(arena_mod.PRODUCTION_CONFIG))
    raw["selfplay"]["gumbel_vloss_weight"] = 2
    raw["selfplay"]["gumbel_topk"] = 24
    raw["selfplay"]["gumbel_c_scale"] = 0.077
    patched = tmp_path / "patched.yaml"
    patched.write_text(_yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(arena_mod, "PRODUCTION_CONFIG", patched)

    side = arena_mod.resolve_search_shape("training")

    assert side.vloss_weight == 2, "the published vloss_weight never reached the arena"
    assert side.realized_gumbel()["topk"] == 24
    assert side.realized_gumbel()["c_scale"] == pytest.approx(0.077)
    # None of the three is a default, so none can pass by accident.
    assert GumbelConfig().topk != 24
    assert GumbelConfig().c_scale != 0.077
    assert SearchConfig().gumbel_vloss_weight != 2


def test_every_config_driven_knob_reaches_the_arena_or_is_provably_inert() -> None:
    """Drift guard: a NEW yaml-driven search knob must reach the arena too.

    Reads the ``GumbelConfig(...)`` production selfplay builds
    (``network_turn.py``) and takes every field it fills from ``search.*`` —
    not just ``search.gumbel_*``: the same call is also fed
    ``search.volatility_q_scale/_fpu/_anchor``, and a narrower scan would call
    itself a coverage guard while missing three knobs.

    A field the training shape does not carry passes only if production's value
    for it equals the ``GumbelConfig`` default, i.e. omitting it is provably a
    no-op TODAY. Turning such a knob on in the yaml fails this test, which is
    the intent: the arena would otherwise keep measuring the old shape.
    """
    from chess_anti_engine.selfplay import network_turn

    src = inspect.getsource(network_turn.run_network_turn)
    config_driven: dict[str, str] = {}
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
                config_driven[kw.arg] = match.group(1)

    assert len(config_driven) >= 5, (
        f"only {sorted(config_driven)} found in network_turn's GumbelConfig -- "
        "the selfplay search construction moved; re-point this test"
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
    side = apply_search_overrides(
        resolve_search_shape("training"), spec="c_scale=0.05", vloss_weight=4,
    )

    assert side.realized_gumbel()["c_scale"] == 0.05
    assert side.realized_gumbel()["topk"] == 16  # untouched by the override
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
    record = build_result_record(
        summary,
        mode="matched_sims", candidate="c.pt", reference="r.pt",
        openings_path="book.pgn.zip", opening_plies=16,
        sims_candidate=32, sims_reference=32, ms_per_move=None,
        temperature=0.1, gumbel_add_noise=True, max_plies=10,
        seed=0, device="cpu", duration_s=0.0,
        search_candidate=resolve_search_shape("training"),
        search_reference=resolve_search_shape("play"),
    )

    assert record["search_candidate"]["shape"] == "training"
    assert record["search_reference"]["shape"] == "play"
    assert record["search_reference"]["vloss_weight"] == PLAY_SEARCH_VLOSS_WEIGHT
    # The REALIZED knobs, not just the ones that were overridden: a reader must
    # be able to see c_scale on a row that never passed --cand-gumbel.
    assert record["search_candidate"]["gumbel"]["c_scale"] == pytest.approx(0.1)
    assert record["search_reference"]["gumbel"]["c_scale"] == pytest.approx(0.025)
    assert record["search_candidate"]["source"]


def test_the_realized_view_is_not_just_the_override_dict() -> None:
    """A sparse override dict is what made the old runs unreadable."""
    side = resolve_search_shape("training")

    assert set(side.gumbel) == {"c_scale", "topk"}
    assert set(side.realized_gumbel()) >= set(PLAY_SEARCH_DEFAULTS)
    assert side.realized_gumbel()["fpu_reduction"] == GumbelConfig().fpu_reduction
