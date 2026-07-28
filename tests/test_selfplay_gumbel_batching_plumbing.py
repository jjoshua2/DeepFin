"""Pins that the C17 batching knobs actually REACH the selfplay search call.

This project's recurring defect is not a wrong value, it is a value that never
arrives: `diff_focus` never reached the worker, `soft_policy_temp` was never
published, the sf_p0 teacher was dead for 15.8 days. A knob that is accepted and
silently ignored is worse than no knob, because it produces a confident verdict
on an experiment that never ran.

`selfplay/network_turn.py` is the ONLY production selfplay search call site
(`selfplay/match.py` is the arena/match path). Before this plumbing it passed
neither `target_batch` nor `vloss_weight`, so both took their 0 defaults and
production ran the 56%-duplicate arm with no way to change it from config.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from chess_anti_engine.selfplay import network_turn
from chess_anti_engine.selfplay.config import SearchConfig


def test_the_knobs_default_to_todays_production_behaviour() -> None:
    """0/0 must stay byte-identical to the behaviour before the fields existed.

    The whole point of a default-off research knob is that merging it changes
    nothing until someone opts in.
    """
    cfg = SearchConfig()
    assert cfg.gumbel_target_batch == 0
    assert cfg.gumbel_vloss_weight == 0


def _gumbel_c_call_kwargs() -> set[str]:
    """Keyword names passed to the C gumbel runner in run_network_turn."""
    # `run_network_turn` is module-level, so its source is already at column 0.
    # Do NOT cleandoc it -- that strips the body's indentation and the parse
    # fails on the docstring.
    src = inspect.getsource(network_turn.run_network_turn)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "gumbel_c_fn":
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(
        "no gumbel_c_fn(...) call found in run_network_turn -- the selfplay "
        "search call site moved; this test must be re-pointed, not deleted",
    )


def test_both_knobs_are_passed_to_the_c_gumbel_runner() -> None:
    """The plumbing test proper: the kwargs are present at the call site.

    Asserted on the AST rather than by running a search, because reaching this
    call needs a model, a broker and a live game state. An AST check cannot
    prove the VALUE is right, so `test_the_call_reads_them_from_search_config`
    below pins where the value comes from.
    """
    kwargs = _gumbel_c_call_kwargs()
    missing = {"target_batch", "vloss_weight"} - kwargs
    assert not missing, f"selfplay drops {sorted(missing)} on the floor again"


def test_the_call_reads_them_from_search_config() -> None:
    """Guards the failure where a knob is passed but wired to a constant.

    Passing `target_batch=0` literally would satisfy the test above while making
    the config field dead -- which is exactly how `matrix_weight_decay` became
    decorative.
    """
    src = inspect.getsource(network_turn.run_network_turn)
    for name in ("gumbel_target_batch", "gumbel_vloss_weight"):
        assert f"search.{name}" in src, (
            f"{name} is not read from the search config at the call site"
        )


def test_virtual_mean_is_not_reachable_from_selfplay_config() -> None:
    """VIRTUAL_MEAN is deliberately NOT offered to selfplay.

    It is measured-dominated: it removes duplicates but collapses the batch to
    ~26 rows/call, the same 8x round-trip cost as `target_batch=1`, and returns
    nothing for it. The pessimism in LEGACY is what spreads walkers onto
    distinct leaves; VIRTUAL_MEAN's whole selling point -- leaving Q untouched --
    is why it cannot fill a batch. Keeping it out of SearchConfig means nobody
    can select it here by reading the C enum and assuming higher is better.
    """
    assert not hasattr(SearchConfig(), "gumbel_vloss_mode")


def test_the_recurring_defect_is_documented_where_someone_will_look() -> None:
    """The field comment must carry the measurement, not just the mechanism.

    A bare `gumbel_vloss_weight: int = 0` invites someone to 'clean up' an
    apparently unused knob. The measured duplicate rate quoted WITH its board
    count is what makes the field's existence self-justifying.
    """
    src = Path(inspect.getfile(SearchConfig)).read_text()
    idx = src.index("gumbel_target_batch")
    preamble = src[max(0, idx - 1800):idx]
    assert "C17" in preamble
    # The rate is meaningless without the board count it was measured at.
    assert "1024/boards-per-call" in preamble
