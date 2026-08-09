"""A Python-only search knob must never be accepted by a caller that runs the C tree.

The mirror of ``test_inert_gumbel_knobs.py``. There the failure was a knob that
could not change ANY search; here it is a knob that changes the Python search and
is silently dropped by the C one -- which is the search production and every arena
actually run (``selfplay/match.py`` dispatches to ``_run_gumbel_root_many_c``
whenever ``mcts._mcts_tree`` imports, and it always does).

The observable damage is identical, and worse for being plausible: the knob is
accepted on the command line, echoed back by ``SideSearch.realized_gumbel()``,
written into the arena JSONL as realized configuration, and then discarded. A
Swiss over it returns a flawless null that reads as a measurement, and the entry
in the ledger says the idea was tested.

The C-only direction of this was already guarded (``gumbel_vloss_weight`` /
``gumbel_target_batch`` raise rather than run the Python path without them, with
the comment "silently dropping these is the exact defect this plumbing fixes").
There was no symmetric guard. These tests fail on the pre-fix tree.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    INERT_GUMBEL_KNOBS,
    PLAY_SEARCH_DEFAULTS,
    PY_ONLY_GUMBEL_KNOBS,
    GumbelConfig,
    py_only_knobs_set,
)
from scripts.arena_standard import apply_search_overrides, resolve_search_shape

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_list_names_real_gumbel_fields() -> None:
    fields = {f.name for f in dataclasses.fields(GumbelConfig)}
    assert PY_ONLY_GUMBEL_KNOBS
    assert fields >= PY_ONLY_GUMBEL_KNOBS


def test_a_knob_is_never_both_inert_and_python_only() -> None:
    """The two denylists mean opposite things; overlap would be a contradiction."""
    assert not (PY_ONLY_GUMBEL_KNOBS & INERT_GUMBEL_KNOBS)


def test_no_python_only_knob_is_published_as_a_production_search_default() -> None:
    """``PLAY_SEARCH_DEFAULTS`` is consumed by callers that take the C path."""
    assert not (set(PLAY_SEARCH_DEFAULTS) & PY_ONLY_GUMBEL_KNOBS)


def test_the_c_sources_really_do_not_implement_these() -> None:
    """The membership claim itself, checked against the C tree rather than assumed.

    If someone implements the knob in C and forgets to remove it from the list,
    this fails and points at the list -- the failure mode is a refused-but-working
    knob, which is loud. The reverse (removing it from the list without the C
    implementation) is caught by nothing else, which is why this test reads the
    sources rather than trusting the constant.
    """
    sources = [
        REPO_ROOT / "chess_anti_engine/mcts/_mcts_tree.c",
        REPO_ROOT / "chess_anti_engine/mcts/gumbel_c.py",
    ]
    present = [p for p in sources if p.exists()]
    assert present, "neither C-path source found; the guard's premise is unchecked"
    blob = "\n".join(p.read_text(errors="replace") for p in present)
    for knob in sorted(PY_ONLY_GUMBEL_KNOBS):
        assert knob not in blob, (
            f"{knob!r} appears in the C path ({[p.name for p in present]}); if it is "
            "implemented there, drop it from PY_ONLY_GUMBEL_KNOBS -- the guard is "
            "now refusing a knob that would work"
        )


def test_py_only_knobs_set_is_quiet_on_defaults_and_loud_off_them() -> None:
    assert py_only_knobs_set(GumbelConfig()) == []
    for knob in sorted(PY_ONLY_GUMBEL_KNOBS):
        default = getattr(GumbelConfig(), knob)
        other = (not default) if isinstance(default, bool) else type(default)(default) + 1
        cfg = dataclasses.replace(GumbelConfig(), **{knob: other})
        assert py_only_knobs_set(cfg) == [knob]


def test_the_arena_refuses_a_python_only_override_instead_of_measuring_a_null() -> None:
    base = resolve_search_shape("play")
    for knob in sorted(PY_ONLY_GUMBEL_KNOBS):
        with pytest.raises(SystemExit) as excinfo:
            apply_search_overrides(base, spec=f"{knob}=1")
        assert knob in str(excinfo.value)
        assert "C" in str(excinfo.value)


def test_a_live_knob_is_still_accepted() -> None:
    """Negative control: the refusal must be specific, not a blanket rejection."""
    out = apply_search_overrides(resolve_search_shape("play"), spec="topk=8")
    assert int(out.gumbel["topk"]) == 8


def test_dispatching_to_the_c_tree_with_the_knob_set_raises() -> None:
    """The guard at the dispatch itself, not just at the CLI.

    ``arena_standard`` is one caller. Anything building a ``GumbelConfig`` in
    process and handing it to ``pick_moves_for_boards`` bypasses that refusal, so
    the load-bearing check is here.
    """
    match_mod = pytest.importorskip(
        "chess_anti_engine.selfplay.match",
        reason="selfplay.match needs the compiled encoding extensions",
    )
    if not match_mod._HAS_GUMBEL_C:
        pytest.skip("C gumbel extension not built in this tree")

    class _Model:
        input_history_encoding = "legacy"
        input_extra_features = "v2_threats"
        policy_encoding = "lc0_1858"
        use_dynamic_relations = False
        history_rep_fix = False

    knob = sorted(PY_ONLY_GUMBEL_KNOBS)[0]
    with pytest.raises(ValueError, match=knob):
        match_mod.pick_moves_for_boards(
            _Model(), [chess.Board()],
            device="cpu", rng=np.random.default_rng(0),
            mcts_type="gumbel", mcts_simulations=8, temperature=1.0,
            c_puct=1.75, gumbel_add_noise=False,
            gumbel_overrides={knob: True},
        )


def test_the_guard_does_not_fire_on_a_default_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the dispatch guard: reachable AND quiet.

    A guard that raised unconditionally would pass the test above while breaking
    every search. The C search itself is stubbed out so this measures the guard
    and nothing else: with a live knob the dispatch must be REACHED, with the
    python-only one it must never be.
    """
    match_mod = pytest.importorskip(
        "chess_anti_engine.selfplay.match",
        reason="selfplay.match needs the compiled encoding extensions",
    )
    if not match_mod._HAS_GUMBEL_C:
        pytest.skip("C gumbel extension not built in this tree")

    from chess_anti_engine.moves import POLICY_SIZE

    class _Model:
        input_history_encoding = "legacy"
        input_extra_features = "v2_threats"
        policy_encoding = "lc0_1858"
        use_dynamic_relations = False
        history_rep_fix = False

    reached: list[GumbelConfig] = []

    def _stub(_model, boards, **kwargs):
        reached.append(kwargs["cfg"])
        n = len(boards)
        return (
            [np.zeros(POLICY_SIZE, dtype=np.float32)] * n,
            [int(next(iter(boards[0].legal_moves)).from_square)] * n,
            [0.0] * n,
            [np.zeros(POLICY_SIZE, dtype=np.bool_)] * n,
        )

    monkeypatch.setattr(match_mod, "_run_gumbel_root_many_c", _stub)
    common = {
        "device": "cpu", "rng": np.random.default_rng(0), "mcts_type": "gumbel",
        "mcts_simulations": 8, "temperature": 1.0, "c_puct": 1.75,
        "gumbel_add_noise": False,
    }
    match_mod.pick_moves_for_boards(
        _Model(), [chess.Board()], gumbel_overrides={"topk": 8}, **common,
    )
    assert len(reached) == 1, "a live knob must still reach the C dispatch"
    assert int(reached[0].topk) == 8

    knob = sorted(PY_ONLY_GUMBEL_KNOBS)[0]
    with pytest.raises(ValueError, match=knob):
        match_mod.pick_moves_for_boards(
            _Model(), [chess.Board()], gumbel_overrides={knob: True}, **common,
        )
    assert len(reached) == 1, (
        "the python-only knob reached the C search anyway -- the guard is after "
        "the dispatch, or is not on this path at all"
    )
