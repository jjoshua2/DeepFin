"""A C-only search knob must be REFUSED on the Python path, never ignored.

The mirror of ``tests/test_uci_search_options.py::test_the_c_path_refuses_a_
python_only_knob``, and the residual of narrowed task #264.

``GumbelConfig.full_tree`` is read by the C fast path and by nothing else. The
C takes ``bool(cfg.full_tree)`` in ``gumbel_c.run_gumbel_root_many_c``, hands it
to ``start_gumbel_sims`` as ``g->sel.full_tree``, and
``tree_gumbel_collect_leaf`` branches on it to run ``tree_gumbel_select_child``
(true) or ``tree_select_child``, the PUCT descent (false). The Python reference
search descends through ``_select_full_gumbel_child`` unconditionally --
``cfg.full_tree`` does not occur in ``mcts/gumbel.py`` at all.

So before this file existed the value was accepted by
``arena_standard --cand-gumbel full_tree=0`` (it is not in
``INERT_GUMBEL_KNOBS``), banked by ``realized_gumbel()`` as that side's realized
search, honoured if the C ran, and silently dropped if the extension was
missing or volatility forced the Python path. Which of the two searches the
number described was decided after the record was written.

MEASURED here, not argued: ``test_the_c_path_really_reads_it`` and
``test_the_python_path_really_does_not`` run the same deterministic search on
both paths at both values. L1 over the returned policy, ``full_tree`` True vs
False, everything else fixed:

    sims    C topk=4    C topk=8    Python (either)
      64    0.000000    0.000000    0.000000
     128    0.000000    0.000000    0.000000
     256    0.000000    0.000000    0.000000
     512    0.000000    0.000000    0.000000
     768    0.000000    0.000000    0.000000
     896    0.000000    0.096116    0.000000
    1024    1.996486    0.063902    0.000000
    1152    1.916876    0.043043    0.000000
    1280    0.000001    0.028351    0.000000
    1536    0.000000    1.980454    0.000000
    2048    0.000000    0.000036    0.000000
    3072    2.000000    0.000000    0.000000   <- played ACTION changes
    4096    2.000000    0.000000    0.000000   <- played ACTION changes

⚑⚑ THE C COLUMN IS NOT MONOTONE AND THERE IS NO DEPTH THRESHOLD. An earlier
revision of this docstring said the difference "needs a deep enough subtree to
reach the branch", which predicts a threshold the data does not have: topk=4
diverges at 1024-1152, returns to 0.0 at 1536-2048, and diverges again at
3072-4096. Sequential halving frequently eliminates to the SAME root policy
even when the descent below the root genuinely differed, so the branch being
reached is necessary and nowhere near sufficient. **The sim counts below are an
empirically found working point, not a derived one.** If this test ever fails,
do not go hunting for the mechanism that sets the threshold -- there isn't one.
Re-run the sweep (``_SWEEP`` drives it) and move to a point that still
diverges.

What the table does establish, at all 26 measured points: the Python column is
EXACTLY 0.0 everywhere, and the C column is non-zero at 8 of 26. That asymmetry
is the whole finding. The tests below re-assert it at the DISCRIMINATING points
-- every point where the C diverges, where a Python 0.0 is a statement about
the path rather than about the sim count -- and not at all 26, because a Python
0.0 where the C is also 0.0 discriminates nothing and the full re-sweep costs
~98s of reference search. A probe that sampled only 128/256/512 would have read
0.0 on both paths and closed the question the wrong way.

The mutation matrix (9 mutants, each applied, run, reverted) is in the PR body.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel as gumbel_mod
from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import (
    C_ONLY_GUMBEL_KNOBS,
    PY_ONLY_GUMBEL_KNOBS,
    GumbelConfig,
    assert_python_path_can_run,
    c_only_knobs_set,
    run_gumbel_root_many,
    validate_gumbel_config,
)
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import POLICY_ENCODING_AZ_4672, POLICY_SIZE


class _HashEvaluator:
    """A deterministic pure function of the ENCODED position.

    Not a random generator: two searches over the same tree must see the same
    logits for the same node, or the paths would differ for a reason that has
    nothing to do with the knob under test. Keyed on the encoded planes, so a
    transposition also evaluates identically.
    """

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        xs = np.asarray(x, dtype=np.float32)
        n = int(xs.shape[0])
        pol = np.empty((n, 1858), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            digest = hashlib.blake2b(xs[i].tobytes(), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            pol[i] = rng.standard_normal(1858, dtype=np.float32) * 2.0
            wdl[i] = rng.standard_normal(3, dtype=np.float32) * 2.0
        return pol, wdl


def _attribute_reads(module, attr: str) -> set[str]:
    """Every ``<expr>.attr`` LOAD in ``module``'s source, as receiver strings.

    Parsed, not grepped. A text search over the module answers a different
    question: this file's own prose names ``cfg.full_tree`` while explaining
    that nothing reads it, and a docstring is not a read. The AST sees code.
    """
    tree = ast.parse(inspect.getsource(module))
    return {
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.ctx, ast.Load)
    }


def _cfg(**kw: object) -> GumbelConfig:
  # `replace` rather than a `**kw` spread into the constructor: the spread
  # erases every field's declared type into the dict's value type, which is the
  # same defect `scripts/audit_targets.py` records at its own PLAY row.
    base = GumbelConfig(simulations=32, topk=4, add_noise=False, temperature=0.0)
    return dataclasses.replace(base, **kw)


def _search(fn, cfg: GumbelConfig) -> tuple[np.ndarray, int]:
    """One board, one fixed RNG stream, one evaluator: only ``cfg`` varies."""
    out = fn(
        None, [chess.Board()], device="cpu",
        rng=np.random.default_rng(7), cfg=cfg, evaluator=_HashEvaluator(),
    )
    return np.asarray(out[0][0]), int(out[1][0])


# --- the set, and how membership was decided ---------------------------------


def test_the_set_holds_real_fields_and_does_not_overlap_its_mirror() -> None:
    fields = set(GumbelConfig.__dataclass_fields__)
    assert fields >= C_ONLY_GUMBEL_KNOBS
    assert not (C_ONLY_GUMBEL_KNOBS & PY_ONLY_GUMBEL_KNOBS)
    assert sorted(C_ONLY_GUMBEL_KNOBS) == ["full_tree"]


def test_membership_is_read_off_the_two_paths_source_not_off_the_name() -> None:
    """The classification method the declaration's comment claims, executed.

    A comment saying "the C reads it, the Python does not" is the kind of claim
    that rots into a comment about a field that grew a consumer. This asserts
    both halves against the shipped source, so wiring ``full_tree`` into
    ``mcts/gumbel.py`` fails HERE and forces the set to be re-derived.
    """
    assert "cfg" in _attribute_reads(gumbel_c_mod, "full_tree")
  # The Python module's only `.full_tree` reads are the guard's own generic
  # `getattr(base/cfg, name)`, which the AST records under no receiver at all.
    assert _attribute_reads(gumbel_mod, "full_tree") == set()


# --- MEASURED: which path acts on it -----------------------------------------


# The C points from the module docstring's table that diverge, chosen for
# DISTANCE from a zero rather than for size. ``(1024, 4)`` is the headline
# number but sits one step from ``(896, 4)`` = 0.0; the topk=8 run 896-1536 is
# a contiguous non-zero band, so a knife-edge moving by one step cannot empty
# this list. Not a threshold -- see the docstring.
_DIVERGING = [(1024, 4), (1152, 4), (896, 8), (1024, 8), (1152, 8), (1536, 8)]

# Where the descent swap reaches the PLAYED MOVE, not just the stored target.
_ACTION_CHANGING = [(3072, 4), (4096, 4)]

# Every point the module docstring tabulates. Not a test parametrisation --
# re-running the whole thing on the PYTHON path costs ~98s, and at a point
# where the C is 0.0 too a Python 0.0 discriminates nothing. It is here so the
# sweep behind the table can be reproduced without reconstructing the list:
#   [(s, k, l1) for s, k in _SWEEP]  against `_search(run_gumbel_root_many_c,...)`
_SWEEP = [
    (s, k)
    for k in (4, 8)
    for s in (64, 128, 256, 512, 768, 896, 1024, 1152, 1280, 1536, 2048, 3072, 4096)
]

# The DISCRIMINATING points for the Python-side control: every point where the
# C really does diverge, so "Python is 0.0 here" is a statement about the path
# and not about the sim count -- plus two from the shallow regime where neither
# path moves. ``(4096, 4)`` is left out on cost alone (29s of Python reference
# search for a point ``(3072, 4)`` already covers).
_PY_CONTROL = [(128, 4), (256, 4), *_DIVERGING, (3072, 4)]


@pytest.mark.parametrize(("sims", "topk"), _DIVERGING)
def test_the_c_path_really_reads_it(sims: int, topk: int) -> None:
    """POSITIVE CONTROL. Without this the guard could be guarding a null.

    ``full_tree=False`` swaps the C descent to PUCT, which changes the returned
    policy -- the stored TRAINING TARGET -- even where it leaves the played
    action alone, so "the move did not change" is not evidence the knob is
    inert. (It does reach the move too; see the next test.)
    """
    on = _search(run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk))
    off = _search(
        run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk, full_tree=False),
    )
    assert np.abs(on[0] - off[0]).sum() > 1e-3


@pytest.mark.parametrize(("sims", "topk"), _ACTION_CHANGING)
def test_the_c_path_reads_it_all_the_way_to_the_played_move(
    sims: int, topk: int,
) -> None:
    """The strongest form of the positive control: a DIFFERENT MOVE.

    At most diverging points only the returned policy moves, which is already
    consequential (it is the stored training target). At these two the played
    action changes as well, so the knob is not merely reshaping a target the
    search then ignores.
    """
    on = _search(run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk))
    off = _search(
        run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk, full_tree=False),
    )
    assert on[1] != off[1]


@pytest.mark.parametrize(("sims", "topk"), _PY_CONTROL)
def test_the_python_path_really_does_not(
    sims: int, topk: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WHY the guard has to raise instead of warn: the search is bit-identical.

    The guard is patched out for exactly this measurement -- otherwise the
    ignoring cannot be observed, and a test that only sees the refusal cannot
    tell a knob that is dropped from one that is honoured.
    """
    monkeypatch.setattr(
        gumbel_mod, "assert_python_path_can_run", lambda cfg, *, where: None,
    )
    on = _search(run_gumbel_root_many, _cfg(simulations=sims, topk=topk))
    off = _search(
        run_gumbel_root_many, _cfg(simulations=sims, topk=topk, full_tree=False),
    )
    assert np.array_equal(on[0], off[0])
    assert on[1] == off[1]


# --- the dispatch guard ------------------------------------------------------


def test_the_python_path_refuses_a_c_only_knob() -> None:
    with pytest.raises(ValueError, match="full_tree") as exc:
        _search(run_gumbel_root_many, _cfg(full_tree=False))
    message = str(exc.value)
    assert "run_gumbel_root_many" in message
    assert "run_gumbel_root_many_c" in message


def test_the_refusal_fires_before_the_empty_batch_shortcut() -> None:
    """A caller must not be able to step around the guard with ``[]``."""
    with pytest.raises(ValueError, match="full_tree"):
        run_gumbel_root_many(
            None, [], device="cpu", rng=np.random.default_rng(0),
            cfg=_cfg(full_tree=False), evaluator=_HashEvaluator(),
        )


def test_the_volatility_reroute_hits_the_same_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route that fires on a machine where the C extension IS built.

    ``pick_moves_for_boards`` sends a volatility-enabled config to the Python
    path regardless of ``_HAS_GUMBEL_C``, so this is the dispatch that turns a
    working C-path arm into a silently different search mid-experiment. Pinned
    with ``_HAS_GUMBEL_C`` forced True so the test cannot pass merely because
    the extension was absent.
    """
    import torch

    from chess_anti_engine.encoding import rep_fix
    from chess_anti_engine.selfplay import match as match_mod

    class _Model(torch.nn.Module):
        def forward(self, x):
            bs = int(x.shape[0])
            return {
                "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
                "wdl": torch.zeros((bs, 3), dtype=torch.float32),
            }

  # ⚑ `pick_moves_for_boards` calls `rep_fix.apply` on the way in, and that
  # flag is PROCESS-GLOBAL in the compiled encoders -- it changes how every
  # later board in this interpreter encodes its repetition planes. A test that
  # leaves it flipped is a test that silently re-encodes the rest of the
  # session, which is the same accept-then-affect-something-else shape this
  # whole file is about. Snapshot and restore.
    _rep_fix_before = rep_fix.current()
    try:
        monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", True)
        with pytest.raises(ValueError, match="full_tree"):
            match_mod.pick_moves_for_boards(
                _Model().eval(), [chess.Board()], device="cpu",
                rng=np.random.default_rng(0), mcts_type="gumbel",
                mcts_simulations=4, temperature=0.0, c_puct=2.5,
                gumbel_add_noise=False,
                volatility_q_scale=1.0,
                gumbel_overrides={"full_tree": 0.0},
            )
    finally:
        if rep_fix.current() != _rep_fix_before:
            # `boards_discarded=True` holds: every board this test made is gone
            # by here (the guard raised before any CBoard was built).
            rep_fix.apply(bool(_rep_fix_before), boards_discarded=True)
            if _rep_fix_before is None:
              # `apply` cannot express "never set". The extensions are now on
              # their documented default (off), which is the state a
              # never-set flag leaves them in, so restoring the module's own
              # sentinel restores the pair.
                rep_fix._current = None  # noqa: SLF001


def test_production_selfplay_has_no_route_to_the_knob_at_all() -> None:
    """WHY there is no ``network_turn`` route test, stated as an assertion.

    Production selfplay dispatches at ``run_network_turn`` (``_HAS_GUMBEL_C and
    not volatility_search_enabled``), which the
    ``pick_moves_for_boards`` test above does not cover. A route test there
    would have to FABRICATE a config production cannot produce, so it would be
    testing the callee-side placement -- which mutant M1 already covers from
    ``run_gumbel_root_many`` -- rather than a reachable path.

    The stronger and cheaper statement is this one: the knob has no config
    surface anywhere. ``build_selfplay_gumbel_config`` is the whole
    ``SearchConfig -> GumbelConfig`` mapping and it never passes ``full_tree``,
    ``SearchConfig`` has no such field, and the yaml schema has no key for it.
    So the production path is protected structurally, and this test is what
    fails the day somebody adds ``gumbel_full_tree`` to the yaml and wires it
    through without re-reading ``C_ONLY_GUMBEL_KNOBS``.
    """
    from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    cfg = build_selfplay_gumbel_config(
        search=SearchConfig(), game=GameConfig(), simulations=8,
    )
    assert cfg.full_tree is True
    assert c_only_knobs_set(cfg) == ()
    assert not [f for f in SearchConfig.__dataclass_fields__ if "full_tree" in f]
    assert not [k for k in flatten_run_config_defaults({}) if "full_tree" in k]


def test_the_c_path_still_accepts_it() -> None:
    """The C capability is NOT withdrawn -- only the instruments refuse it.

    A caller that constructs the config itself and knows which path it is on
    keeps the PUCT descent. Deleting this test's counterpart in the code
    (making ``run_gumbel_root_many_c`` refuse too) would turn the guard into a
    feature removal nobody asked for.
    """
    probs, action = _search(run_gumbel_root_many_c, _cfg(full_tree=False))
    assert probs.shape == (POLICY_SIZE,)
    assert 0 <= action < POLICY_SIZE


def test_a_value_the_c_would_silently_coerce_is_refused_too() -> None:
    """``full_tree=2`` runs the DEFAULT search under a record naming 2.

    ``bool(2.0)`` is True, so the C reads it as the shipped descent while
    ``realized_gumbel()`` banks ``2.0``. Accepted, silently coerced, recorded as
    something nothing ran -- so the comparison is strict rather than sharing the
    consumer's ``bool()``.
    """
    from scripts.arena_standard import apply_search_overrides

    assert c_only_knobs_set(_cfg(full_tree=2)) == ("full_tree",)
    with pytest.raises(SystemExit, match="full_tree"):
        apply_search_overrides(_play_side(), spec="full_tree=2")


def test_a_string_typed_knob_at_its_default_is_not_an_offender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plain ``==`` ahead of the float coercion, exercised on a real field.

    Both shipped sets hold numeric fields, so nothing today reaches the
    equality arm. ``policy_encoding`` is a genuine ``str`` field of the same
    dataclass, so pointing the set at it is the cheapest faithful stand-in for
    the future C-only knob that is an enum. Without the early return
    ``float("lc0_1858")`` raises, the fallback calls it an offender, and EVERY
    config in the repo -- including the default -- fails the guard at a value
    nobody changed.
    """
    monkeypatch.setattr(
        gumbel_mod, "C_ONLY_GUMBEL_KNOBS", frozenset({"policy_encoding"}),
    )
    assert c_only_knobs_set(GumbelConfig()) == ()
    validate_gumbel_config(GumbelConfig(), where="test")
  # ...and it still fires when the string genuinely differs.
    assert c_only_knobs_set(_cfg(policy_encoding=POLICY_ENCODING_AZ_4672)) == (
        "policy_encoding",
    )


def test_an_int_too_wide_for_a_float_is_refused_not_propagated() -> None:
    """``float(10**400)`` raises ``OverflowError``, not ``ValueError``.

    Both catch sites matter and neither used to name it: the shared comparison
    would have propagated it out of ``assert_python_path_can_run``, and
    ``validate_gumbel_config``'s own finiteness loop would have propagated it
    out of a function whose CLI callers wrap ``ValueError`` alone -- so an
    operator would have got a raw traceback from inside the guard against raw
    tracebacks.
    """
    huge = 10 ** 400
    assert c_only_knobs_set(_cfg(full_tree=huge)) == ("full_tree",)
    with pytest.raises(ValueError, match="full_tree"):
        validate_gumbel_config(_cfg(full_tree=huge), where="test")
  # The finiteness loop's own site: a numeric field, so it reaches that loop
  # rather than the C-only comparison.
    with pytest.raises(ValueError, match="c_scale|halving_div"):
        validate_gumbel_config(_cfg(c_scale=huge, halving_div=1), where="test")


def test_a_value_float_cannot_read_is_an_offender_on_both_knob_sets() -> None:
    """The comparison is now SHARED with ``python_only_knobs_set``.

    Before this PR that helper called ``float()`` bare, so a non-numeric knob
    left the guard as a ``TypeError`` naming ``float`` and not the knob -- and
    ``validate_gumbel_config``'s callers only wrap ``ValueError``, so at an eval
    boundary it would have surfaced as a raw traceback. Sharing one comparison
    is the point; this pins that the sharing did not lose the PY_ONLY side.
    """
    from chess_anti_engine.mcts.gumbel import python_only_knobs_set

    assert c_only_knobs_set(_cfg(full_tree="yes")) == ("full_tree",)
    assert python_only_knobs_set(_cfg(volatility_q_scale="yes")) == (
        "volatility_q_scale",
    )
    with pytest.raises(ValueError, match="full_tree"):
        validate_gumbel_config(_cfg(full_tree="yes"), where="test")


@pytest.mark.parametrize("value", [True, 1, 1.0])
def test_the_default_passes_both_paths(value: object) -> None:
    """NEGATIVE CONTROL, including the float the CLI parser produces.

    ``--cand-gumbel full_tree=1`` arrives as ``1.0``, not ``True``, and both
    float to the same number as the default -- so an explicit request for the
    shipped search is accepted however it is spelled, rather than refused on a
    type.
    """
    cfg = _cfg(full_tree=value)
    assert c_only_knobs_set(cfg) == ()
    assert_python_path_can_run(cfg, where="test")
    validate_gumbel_config(cfg, where="test")
    for fn in (run_gumbel_root_many, run_gumbel_root_many_c):
        probs, action = _search(fn, cfg)
        assert probs.shape == (POLICY_SIZE,)
        assert 0 <= action < POLICY_SIZE


# --- the eval boundaries that BANK a realized shape --------------------------


def _play_side():
    from scripts.arena_standard import resolve_search_shape

    return resolve_search_shape("play")


def test_the_arena_override_path_refuses_it() -> None:
    """THE finding: ``--cand-gumbel full_tree=0``.

    Driven through ``apply_search_overrides``, the function ``main()`` parses
    ``--cand-gumbel`` / ``--ref-gumbel`` with, before any checkpoint is loaded.
    """
    from scripts.arena_standard import apply_search_overrides

    with pytest.raises(SystemExit) as exc:
        apply_search_overrides(_play_side(), spec="full_tree=0")
    message = str(exc.value)
    assert "full_tree" in message
    assert "C-path only" in message
  # The per-knob tail from `_C_ONLY_KNOB_DETAIL`, which is what tells the
  # operator their arm would run an UNTUNABLE PUCT descent rather than just
  # "a knob two paths disagree about". Pinned so moving it into the table did
  # not quietly drop it from the message.
    assert "PUCT" in message
    assert "INERT_GUMBEL_KNOBS" in message


def test_the_check_is_on_the_side_constructor_not_the_override_parser() -> None:
    """Placement, pinned -- same reason as the ``policy_temp`` sibling.

    ``apply_search_overrides`` would have covered the two call sites in
    ``main()`` and let a resolved shape or a programmatic caller
    (``scripts/elo_vs_sims.py``) walk past.
    """
    from scripts.arena_standard import SideSearch

    with pytest.raises(SystemExit, match="full_tree"):
        SideSearch(
            shape="training", source="not the CLI",
            gumbel={"full_tree": 0.0}, vloss_weight=1, target_batch=0,
        )


def test_the_audit_targets_override_path_refuses_it() -> None:
    from scripts.audit_targets import parse_gumbel_overrides

    with pytest.raises(SystemExit, match="full_tree"):
        parse_gumbel_overrides(["full_tree=0"])


def test_the_default_spelled_explicitly_is_accepted_and_banked() -> None:
    """NEGATIVE CONTROL for the three refusals above.

    ``full_tree=1`` describes the search BOTH paths run, so banking it is
    truthful and refusing it would be the guard over-firing.
    """
    from scripts.arena_standard import apply_search_overrides
    from scripts.audit_targets import parse_gumbel_overrides

    side = apply_search_overrides(_play_side(), spec="full_tree=1")
    assert side.realized_gumbel()["full_tree"] == 1.0
    assert parse_gumbel_overrides(["full_tree=1"]) == (("full_tree", 1.0),)


def test_what_the_record_would_have_said_without_the_refusal() -> None:
    """WHY refusing beats letting the dispatch sort it out.

    Built the way ``apply_search_overrides`` used to, bypassing the constructor
    check: ``realized_gumbel()`` -- the dict ``as_record()`` banks into the
    JSONL and ``describe()`` prints at startup -- reports ``full_tree: 0.0`` as
    this side's REALIZED search. The record is written at shape-resolution
    time, before ``_HAS_GUMBEL_C`` or ``volatility_search_enabled`` has decided
    which of the two searches it names.
    """
    from scripts.arena_standard import SideSearch

    side = SideSearch.__new__(SideSearch)
    for key, value in {
        "shape": "play", "source": "counterfactual", "gumbel": {"full_tree": 0.0},
        "vloss_weight": 3, "target_batch": 0, "tree_reuse": "cold",
    }.items():
        object.__setattr__(side, key, value)

    assert side.realized_gumbel()["full_tree"] == 0.0
    assert side.as_record()["gumbel"]["full_tree"] == 0.0


def test_the_inert_set_refusal_this_one_backs_up_is_still_there() -> None:
    """``full_tree=False`` un-inerts the knobs the same parser refuses.

    ``INERT_GUMBEL_KNOBS`` is justified by ``full_tree=True`` making the PUCT
    descent unreachable, and the refusal message says so. An accepted
    ``full_tree=0`` would falsify that message two branches away in the same
    parser -- and hand the operator a PUCT descent pinned at ``GumbelConfig``'s
    own untuned defaults, since the tuned ``PLAY_PUCT_DEFAULTS`` are not what a
    bare ``GumbelConfig`` carries.
    """
    from chess_anti_engine.mcts.gumbel import (
        INERT_GUMBEL_KNOBS,
        PLAY_PUCT_DEFAULTS,
    )
    from scripts.arena_standard import apply_search_overrides

    for knob in sorted(INERT_GUMBEL_KNOBS):
        with pytest.raises(SystemExit, match="full_tree"):
            apply_search_overrides(_play_side(), spec=f"{knob}=1.0")

    base = GumbelConfig()
    assert base.c_puct != PLAY_PUCT_DEFAULTS["c_puct"]
    assert base.fpu_reduction != PLAY_PUCT_DEFAULTS["fpu_reduction"]


# --- policy_encoding: a carried field, not a knob ----------------------------


def test_neither_search_path_reads_policy_encoding() -> None:
    """The declaration's comment, executed on both paths.

    ``GumbelConfig.policy_encoding`` is threaded from the checkpoint by
    ``pick_moves_for_boards`` and ``uci.__main__._build_engine`` and is read
    back only by ``eval.puzzles``, as the fallback source for writing the same
    field onto a ``replace``. The policy WIDTH is inferred from the array by
    ``policy_batch_to_full_if_needed``, which takes no encoding argument, so
    moving this field cannot move a search.

    Behavioural, not a grep: a grep would pass the day somebody reads it
    through ``getattr(cfg, "policy_encoding")``.
    """
    for fn in (run_gumbel_root_many, run_gumbel_root_many_c):
        default = _search(fn, _cfg())
        other = _search(fn, _cfg(policy_encoding=POLICY_ENCODING_AZ_4672))
        assert np.array_equal(default[0], other[0])
        assert default[1] == other[1]


def test_the_only_production_reader_of_policy_encoding_is_still_the_one_named() -> None:
    """Pins the map the declaration's comment records.

    Not "nothing reads it" -- ``eval.puzzles.run_puzzle_eval`` does, and that
    read is why the field is documented rather than deleted.

    The scan covers the FIVE modules the comment's map names: the two search
    paths, plus the two writers and the one reader. An earlier revision scanned
    only the two search modules while claiming "if a second consumer appears,
    this fails" -- which was false for a consumer appearing in any of the other
    three, i.e. in exactly the modules the map is about.

    ``uci/__main__.py`` and ``selfplay/match.py`` come back with NO attribute
    reads at all: they read the checkpoint's encoding through
    ``getattr(model, "policy_encoding", ...)``, which is a call and not an
    attribute node, and then write it onto the config. That is the map's
    "writers" row, asserted below by the keyword they pass.
    """
    from chess_anti_engine.eval import puzzles as puzzles_mod
    from chess_anti_engine.selfplay import match as match_mod
    from chess_anti_engine.uci import __main__ as uci_main_mod

    assert "gumbel_cfg.policy_encoding" in inspect.getsource(
        puzzles_mod.run_puzzle_eval,
    )
    assert "policy_encoding=policy_encoding" in inspect.getsource(
        match_mod.pick_moves_for_boards,
    )
  # `gumbel_cfg` is the SOLE receiver, and only in `puzzles`. Anything else --
  # a `cfg.policy_encoding` in either search path, or a second consumer in a
  # writer module -- lands here as a new receiver and fails.
    expected = {
        gumbel_mod: set(),
        gumbel_c_mod: set(),
        match_mod: set(),
        uci_main_mod: set(),
        puzzles_mod: {"gumbel_cfg"},
    }
    for module, receivers in expected.items():
        assert _attribute_reads(module, "policy_encoding") == receivers, module.__name__
        assert 'getattr(cfg, "policy_encoding"' not in inspect.getsource(module)
