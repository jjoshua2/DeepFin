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
both paths at both values. Positive control on the C (L1 ~= 2.0 over the
returned policy at 1024 sims / topk 4), exact 0.0 on the Python at every sim
count tried. The C difference needs a deep enough subtree to reach the branch
at all -- at 128/256/512 sims both C arms are bit-identical too, which is why a
probe that stopped at "L1 = 0" would have reported the knob inert on BOTH paths
and closed the question the wrong way.

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


@pytest.mark.parametrize(("sims", "topk"), [(1024, 4), (1024, 8)])
def test_the_c_path_really_reads_it(sims: int, topk: int) -> None:
    """POSITIVE CONTROL. Without this the guard could be guarding a null.

    ``full_tree=False`` swaps the C descent to PUCT, which changes the returned
    policy -- the stored TRAINING TARGET -- even where it leaves the played
    action alone, so "the move did not change" is not evidence the knob is
    inert.
    """
    on = _search(run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk))
    off = _search(
        run_gumbel_root_many_c, _cfg(simulations=sims, topk=topk, full_tree=False),
    )
    assert np.abs(on[0] - off[0]).sum() > 1e-3


@pytest.mark.parametrize(("sims", "topk"), [(128, 4), (256, 4), (1024, 4), (1024, 8)])
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

    from chess_anti_engine.selfplay import match as match_mod

    class _Model(torch.nn.Module):
        def forward(self, x):
            bs = int(x.shape[0])
            return {
                "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
                "wdl": torch.zeros((bs, 3), dtype=torch.float32),
            }

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
    read is why the field is documented rather than deleted. If a second
    consumer appears, this fails and the comment gets rewritten instead of
    quietly becoming false.
    """
    from chess_anti_engine.eval import puzzles as puzzles_mod
    from chess_anti_engine.selfplay import match as match_mod

    assert "gumbel_cfg.policy_encoding" in inspect.getsource(
        puzzles_mod.run_puzzle_eval,
    )
    assert "policy_encoding=policy_encoding" in inspect.getsource(
        match_mod.pick_moves_for_boards,
    )
    for module in (gumbel_mod, gumbel_c_mod):
        assert _attribute_reads(module, "policy_encoding") == set()
        assert 'getattr(cfg, "policy_encoding"' not in inspect.getsource(module)
