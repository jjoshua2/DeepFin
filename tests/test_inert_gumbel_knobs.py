"""No knob that a Gumbel search cannot act on may be published as a Gumbel knob.

Play-path code audit 2026-08-03, F2. ``c_puct`` / ``fpu_reduction`` /
``cpuct_factor`` / ``cpuct_base`` drive the PUCT descent (``tree_select_child``,
``_mcts_tree.c:3160-3169``), which is reached only when ``sel->full_tree`` is
false. ``GumbelConfig.full_tree`` defaults True and NOTHING sets it false, and
``cpuct_factor``/``cpuct_base`` are not even arguments of ``start_gumbel_sims``
(the C reads them off the tree via ``set_cpuct_scaling``, which only the PUCT
entry points call). The audit's repro measured L1 0.000000 over the returned
policy and 0/4 moves changed for each of the four, against positive controls
(``topk``, ``halving_div``, ``policy_temp``) that move it.

They were nevertheless listed in ``PLAY_SEARCH_DEFAULTS`` (documented as "the
production PLAY/EVAL Gumbel search settings"), advertised in
``arena_standard.py --cand-gumbel``'s help as knobs for "a pure search-config
Swiss", and printed at arena startup and written into the JSONL record by
``SideSearch.realized_gumbel()`` as REALIZED search configuration. A Swiss over
``c_puct`` would have returned a flat, perfectly reproducible null and read as a
measurement.

These tests fail on the pre-fix tree. They do NOT re-derive inertness (that is
the audit's repro, and a C-level fact); they assert that the config SURFACE no
longer claims otherwise.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chess_anti_engine.mcts.gumbel import (
    INERT_GUMBEL_KNOBS,
    PLAY_PUCT_DEFAULTS,
    PLAY_SEARCH_DEFAULTS,
    GumbelConfig,
)
from scripts.arena_standard import apply_search_overrides, resolve_search_shape

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_denylist_names_exactly_the_puct_descent_knobs() -> None:
    assert set(INERT_GUMBEL_KNOBS) == {
        "c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction",
    }


def test_play_search_defaults_publishes_no_inert_knob() -> None:
    """The deciding assertion. Fails on pre-fix `main` (all four were present)."""
    leaked = sorted(set(PLAY_SEARCH_DEFAULTS) & INERT_GUMBEL_KNOBS)
    assert not leaked, (
        f"PLAY_SEARCH_DEFAULTS publishes {leaked} as Gumbel search settings, but a "
        "Gumbel search cannot act on them (audit 2026-08-03 F2). PUCT defaults belong "
        "in PLAY_PUCT_DEFAULTS."
    )


def test_every_remaining_play_default_is_a_real_gumbel_field() -> None:
    fields = set(GumbelConfig.__dataclass_fields__)
    assert set(PLAY_SEARCH_DEFAULTS) <= fields


def test_the_puct_defaults_kept_their_values_in_their_new_home() -> None:
    """Moving them must not have retuned the UCI PUCT walkers."""
    assert PLAY_PUCT_DEFAULTS == {
        "c_puct": 1.75,
        "cpuct_factor": 3.89,
        "cpuct_base": 38739.0,
        "fpu_reduction": 0.33,
    }
    assert set(PLAY_PUCT_DEFAULTS) == set(INERT_GUMBEL_KNOBS)


def test_the_inert_fields_stay_on_gumbelconfig_for_the_puct_callers() -> None:
    """uci/search.py and uci/walker_pool.py read these OFF the shared config.

    Deleting the fields would break real PUCT descents; only the Gumbel
    *surface* was the lie.
    """
    fields = set(GumbelConfig.__dataclass_fields__)
    assert fields >= INERT_GUMBEL_KNOBS


def test_the_python_gumbel_descent_has_no_puct_arm_left() -> None:
    """F12: `_collect_forced_leaf`'s PUCT else-arm was unreachable dead code."""
    import inspect

    from chess_anti_engine.mcts import gumbel as gumbel_mod

    src = inspect.getsource(gumbel_mod._collect_forced_leaf)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "_select_child(" not in code
    assert "cfg.c_puct" not in code
    assert "cfg.fpu_reduction" not in code
    assert "cfg.full_tree" not in code


def test_realized_gumbel_never_reports_an_inert_knob() -> None:
    """Fails on pre-fix `main`: realized_gumbel() iterated PLAY_SEARCH_DEFAULTS."""
    side = resolve_search_shape("play")
    realized = side.realized_gumbel()
    assert not (set(realized) & INERT_GUMBEL_KNOBS)
    described = side.describe()
    for knob in INERT_GUMBEL_KNOBS:
        assert knob not in described
    assert not (set(side.as_record()["gumbel"]) & INERT_GUMBEL_KNOBS)


@pytest.mark.parametrize("knob", sorted(INERT_GUMBEL_KNOBS))
def test_the_arena_refuses_an_inert_override_instead_of_measuring_a_null(
    knob: str,
) -> None:
    """Pre-fix this was accepted and silently produced a flat, reproducible null."""
    base = resolve_search_shape("play")
    with pytest.raises(SystemExit) as exc:
        apply_search_overrides(base, spec=f"{knob}=1.234")
    assert knob in str(exc.value)
    assert "F2" in str(exc.value)


def test_a_live_knob_is_still_accepted() -> None:
    """Negative control: the denylist must not reject knobs that DO take effect."""
    base = resolve_search_shape("play")
    side = apply_search_overrides(base, spec="topk=8,c_scale=0.077")
    assert side.realized_gumbel()["topk"] == 8
    assert side.realized_gumbel()["c_scale"] == pytest.approx(0.077)


def test_the_cand_gumbel_help_no_longer_advertises_the_inert_knobs() -> None:
    """The help must not SUGGEST a knob the parser will refuse.

    Asserting only that the word "REJECTED" appears would pass with all four
    names still listed as suggested overrides -- a test that cannot fail for
    the reason it is named. So: locate the "REJECTED" sentence, and require
    every inert name to appear ONLY at or after it.
    """
    text = (REPO_ROOT / "scripts" / "arena_standard.py").read_text()
    start = text.index('"--cand-gumbel"')
    end = text.index("p.add_argument", start + 1)
    help_block = text[start:end]

    assert "REJECTED" in help_block, "the help no longer says the knobs are refused"
    rejected_at = help_block.index("REJECTED")
    # The sentence is written "c_puct/fpu_reduction/cpuct_factor/cpuct_base are
    # REJECTED", so the names sit just before the word; take the sentence start.
    sentence_at = help_block.rindex(". ", 0, rejected_at) if ". " in help_block[:rejected_at] else 0

    for knob in sorted(INERT_GUMBEL_KNOBS):
        assert knob in help_block, f"the help does not name {knob} as refused"
        assert help_block.index(knob) >= sentence_at, (
            f"--cand-gumbel's help still advertises {knob} as a usable override "
            "before the sentence that says it is refused"
        )

    # ...and it must still advertise knobs that DO work.
    for live in ("c_scale", "topk", "halving_div"):
        assert help_block.index(live) < sentence_at, f"{live} dropped from the help"


def test_nothing_sets_full_tree_false_anywhere_in_the_tree() -> None:
    """The premise of the whole finding, asserted so a future edit re-reads it."""
    hits: list[str] = []
    for root in ("chess_anti_engine", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if "full_tree" not in line or line.strip().startswith("#"):
                    continue
                squashed = line.replace(" ", "")
                if "full_tree=False" in squashed or "full_tree:bool=False" in squashed:
                    hits.append(f"{path}:{lineno}")
    assert not hits, (
        "something now sets GumbelConfig.full_tree=False, which makes the PUCT "
        f"descent reachable from a Gumbel search: {hits}. Re-decide F2 before "
        "shipping it -- the four knobs stop being inert."
    )
    assert GumbelConfig().full_tree is True


def test_the_play_config_built_from_the_defaults_still_replaces_cleanly() -> None:
    cfg = dataclasses.replace(
        GumbelConfig(simulations=256, topk=32, temperature=0.0, add_noise=False),
        **PLAY_SEARCH_DEFAULTS,
    )
    assert cfg.c_scale == 0.025
    assert cfg.topk == 32
    # Untouched: the PUCT fields keep the GumbelConfig library defaults, which
    # is honest -- the play shape does not set them because it cannot use them.
    assert cfg.c_puct == GumbelConfig().c_puct
