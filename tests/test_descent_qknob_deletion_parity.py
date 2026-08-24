"""The deleted descent q-knobs must leave the search exactly where it was.

``q_visit_exp`` / ``q_global_scale`` / ``q_visit_floor`` were three DESCENT
value-transform knobs on ``GumbelConfig``. Nothing in this repo ever set one:
they are absent from every config, from ``PLAY_SEARCH_DEFAULTS`` and from the
production selfplay call site. Deleting a knob pinned at its default is
behaviour-identical BY CONSTRUCTION -- and this repo's standing lesson is that
"by construction" is a claim, so these tests are the observation.

⚑ What the deletion actually had to preserve is NOT "the Python path ignored
them". That premise was FALSE and the measurement caught it. The Python DESCENT
(``_sigma_scale``) genuinely never read any of the three, but
``_root_sigma_scale`` read TWO of them: ``q_visit_exp`` as the
``q_visit_exp_root >= 90`` fallback, and ``q_visit_floor`` for its additive
branch. Measured on ``origin/main`` before the deletion, over a fixed-seed
4-position x 2-shape x 2-path battery: perturbing ``q_visit_exp`` to 0.5 moved
the chosen action / policy digest on 4/4 positions on BOTH the C and the Python
path in the linear-root shape; ``q_visit_floor=25`` moved 4/4 on the C path in
both shapes and 4/4 on the Python path in the linear-root shape;
``q_global_scale=True`` moved the C path only. So the knobs were live, and what
makes the deletion safe is that their defaults are pinned into both paths -- not
that they were inert.

The cross-commit battery itself is not pinned here: its digests depend on torch's
initialisation RNG and on the compiled extension's floating point, so a pinned
golden would be a machine fingerprint rather than a property. It was run on both
commits with one identical binary and reported in the PR. These tests are the
portable half -- each one fails if the pinning drifts.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    GumbelConfig,
    _root_sigma_scale,
)
from chess_anti_engine.mcts.gumbel_c import (
    _DELETED_Q_GLOBAL_SCALE,
    _DELETED_Q_VISIT_EXP,
    _DELETED_Q_VISIT_FLOOR,
    run_gumbel_root_many_c,
)
from chess_anti_engine.mcts.search_options import SEARCH_OPTIONS

DELETED = ("q_visit_exp", "q_global_scale", "q_visit_floor")


def _pre_deletion_root_sigma_scale(
    *, max_visit: int, cfg: GumbelConfig,
    q_visit_exp: float = 1.0, q_visit_floor: float = -1.0,
) -> float:
    """``_root_sigma_scale`` as it stood on ``origin/main`` at 85e603bbb.

    Transcribed from the pre-deletion source with the two deleted knobs restored
    as arguments. Keeping the old algebra here is the whole point: the shipped
    function must equal it at the deleted defaults, and must be free to differ
    from it nowhere else that matters.
    """
    cvr = float(cfg.c_visit_root) if float(cfg.c_visit_root) >= 0.0 else float(cfg.c_visit)
    csr = float(cfg.c_scale_root) if float(cfg.c_scale_root) >= 0.0 else float(cfg.c_scale)
    qer = (
        float(cfg.q_visit_exp_root)
        if float(cfg.q_visit_exp_root) < 90.0
        else float(q_visit_exp)
    )
    mv = float(max_visit)
    if qer < 0.0:
        return csr * math.log1p(cvr + mv)
    mv_term = mv if qer == 1.0 else mv**qer
    if float(q_visit_floor) >= 0.0:
        return float(q_visit_floor) + csr * mv_term
    return csr * (cvr + mv_term)


@pytest.mark.parametrize("c_scale", [0.025, 0.1, 7.0])
@pytest.mark.parametrize("c_visit_root", [-1.0, 0.0, 900.0])
@pytest.mark.parametrize("c_scale_root", [-1.0, 7.0])
@pytest.mark.parametrize("q_visit_exp_root", [-1.0, 0.5, 1.0, 99.0])
@pytest.mark.parametrize("max_visit", [0, 1, 60, 256, 1_000_000])
def test_root_sigma_scale_equals_the_pre_deletion_formula_at_the_old_defaults(
    c_scale: float, c_visit_root: float, c_scale_root: float,
    q_visit_exp_root: float, max_visit: int,
) -> None:
    """The shipped root transform must be the old one with the knobs at default.

    Swept rather than spot-checked because the two arms that read the deleted
    knobs are BRANCHES: ``q_visit_exp`` was reached only at
    ``q_visit_exp_root >= 90``, and ``q_visit_floor`` only on the non-log arm. A
    grid that never sets ``q_visit_exp_root=99`` would agree no matter what the
    fallback resolved to.
    """
    cfg = GumbelConfig(
        c_scale=c_scale, c_visit=50.0, c_visit_root=c_visit_root,
        c_scale_root=c_scale_root, q_visit_exp_root=q_visit_exp_root,
    )
    assert _root_sigma_scale(max_visit=max_visit, cfg=cfg) == (
        _pre_deletion_root_sigma_scale(max_visit=max_visit, cfg=cfg)
    ), "the root transform moved when the deleted knobs were at their defaults"


def test_the_linear_root_sentinel_really_resolves_to_exponent_one() -> None:
    """The `>= 90` arm is the one the deletion hard-coded, so name it.

    A differential against a reference that made the SAME substitution would
    pass even if both said 2.0, so this asserts the value independently: at the
    sentinel the root must be linear, i.e. `csr * (cvr + max_visit)`.
    """
    cfg = GumbelConfig(c_scale=0.1, c_visit=50.0, q_visit_exp_root=99.0)
    for mv in (0, 7, 256):
        assert _root_sigma_scale(max_visit=mv, cfg=cfg) == pytest.approx(
            0.1 * (50.0 + mv)
        )
    # ...and the shipped PLAY shape is still the LOG root, not this arm.
    assert float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"]) < 0.0


def test_the_wrapper_passes_the_deleted_defaults_into_the_c_search() -> None:
    """Observed from the CONSUMER's own call, not from the constants' values.

    ``_DELETED_Q_*`` being right is worth nothing if the wrapper stopped passing
    them, or passed them in the wrong positional slot -- the arguments that
    follow (``halving_div`` / ``c_visit_root`` / ``c_scale_root`` /
    ``q_visit_exp_root`` / ``vloss_mode``) are all live, so a dropped argument
    would silently shift every one of them by a slot. This runs a real search
    and reads the arguments the C entry point actually received.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree
    from chess_anti_engine.model import ModelConfig, build_model

    seen: list[tuple] = []

    class _SpyTree:
        """Proxy that records ``start_gumbel_sims`` args, then forwards them."""

        def __init__(self) -> None:
            self._real = MCTSTree()

        def start_gumbel_sims(self, *args):
            seen.append(args)
            return self._real.start_gumbel_sims(*args)

        def __getattr__(self, name: str):
            return getattr(self._real, name)

    import torch

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            input_extra_features="v1", embed_dim=32, num_layers=1,
            num_heads=2, use_smolgen=False,
        )
    ).eval()
    cfg = GumbelConfig(
        simulations=16, add_noise=False, temperature=0.0,
        input_extra_features="v1",
    )
    run_gumbel_root_many_c(
        model, [chess.Board()], device="cpu",
        rng=np.random.default_rng(0), cfg=cfg,
        tree=_SpyTree(),  # pyright: ignore[reportArgumentType]
    )

    assert seen, "start_gumbel_sims was never called; the spy proved nothing"
    # Positional layout after the required args (see _mcts_tree.c
    # MCTSTree_start_gumbel_sims): ... enc_buf, vloss_weight, target_batch,
    # input_history_lc0_root, rel_buf, q_visit_exp, q_global_scale,
    # q_visit_floor, halving_div, c_visit_root, c_scale_root, q_visit_exp_root,
    # vloss_mode. Indexing from the END keeps this pinned to the three slots
    # under test even if a new REQUIRED argument is added at the front.
    for args in seen:
        q_visit_exp, q_global_scale, q_visit_floor = args[-8:-5]
        assert q_visit_exp == 1.0, args
        assert q_global_scale == 0, args
        assert q_visit_floor == -1.0, args
        # and the live knobs that FOLLOW them still line up
        assert args[-5] == cfg.halving_div, "argument slots shifted"
        assert args[-4] == cfg.c_visit_root, "argument slots shifted"
        assert args[-3] == cfg.c_scale_root, "argument slots shifted"
        assert args[-2] == cfg.q_visit_exp_root, "argument slots shifted"


def test_the_pinned_literals_are_the_c_sources_own_defaults() -> None:
    """The .c was deliberately NOT edited, so its defaults are the contract.

    Editing ``_mcts_tree.c`` would make the next
    ``scripts/build_production_extensions.py`` a silent deploy, so the deletion
    stayed on the Python side and pins the C's own declared defaults instead.
    If someone changes those declarations, the pinned literals stop meaning
    "unchanged search" and this fires.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "chess_anti_engine" / "mcts" / "_mcts_tree.c"
    ).read_text(encoding="utf-8")

    def _decl(kind: str, name: str) -> str:
        m = re.search(rf"^\s*{kind} {name} = ([^;]+);", src, re.MULTILINE)
        assert m is not None, f"no default declaration for {name} in _mcts_tree.c"
        return m.group(1).strip()

    assert float(_decl("double", "q_visit_exp")) == _DELETED_Q_VISIT_EXP
    assert int(_decl("int", "q_global_scale")) == _DELETED_Q_GLOBAL_SCALE
    assert float(_decl("double", "q_visit_floor")) == _DELETED_Q_VISIT_FLOOR


def test_the_knobs_are_gone_from_every_surface() -> None:
    """Deleted means deleted: no field, no UCI option, no config key.

    The failure this guards is the repo's signature defect in reverse -- a knob
    half-removed, still accepted somewhere and ignored everywhere.
    """
    fields = set(GumbelConfig.__dataclass_fields__)
    assert fields.isdisjoint(DELETED), (
        f"{sorted(fields & set(DELETED))} came back onto GumbelConfig"
    )
    option_fields = {o.field for o in SEARCH_OPTIONS}
    assert option_fields.isdisjoint(DELETED), (
        f"{sorted(option_fields & set(DELETED))} came back onto the UCI registry"
    )
    configs = Path(__file__).resolve().parents[1] / "configs"
    for name in DELETED:
        hits = [
            p.name for p in sorted(configs.glob("*.yaml"))
            if name in p.read_text(encoding="utf-8")
        ]
        assert hits == [], f"configs set the deleted knob {name!r}: {hits}"
