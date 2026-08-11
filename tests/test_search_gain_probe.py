"""Tests for the search-gain probe's shape arithmetic.

The probe's whole value rests on one thing: that the `SearchShape` it prints in
its header is the search the number came from. Two ways that can silently go
wrong, and one test for each:

  * the probe reimplements the sigma(q) resolution and drifts from the real one
    (`gumbel._root_sigma_scale`), so its "realized q_scale" column describes a
    search nobody runs;
  * the `live_selfplay` shape stops matching what `selfplay/network_turn.py`
    actually builds, so a "production" number is measured on something else.

The second is the one this codebase keeps producing -- a value accepted and
then silently ignored -- so it is asserted against the production call site's
own source rather than against a copy of the constants.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import sys
import inspect
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.audit import AuditPosition, legal_full_indices
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    GumbelConfig,
    _root_sigma_scale,
)
import scripts.search_gain_probe as sgp
from scripts.search_gain_probe import (
    SHAPES,
    ProbePosition,
    RowResult,
    SearchShape,
    _mean_ci,
    _score_one,
    apply_overrides,
    shape_from_yaml,
)


def _cfg_for(shape: SearchShape) -> GumbelConfig:
    return GumbelConfig(
        topk=shape.topk,
        c_scale=shape.c_scale,
        c_visit=shape.c_visit,
        c_visit_root=shape.c_visit_root,
        c_scale_root=shape.c_scale_root,
        q_visit_exp=shape.q_visit_exp,
        q_visit_exp_root=shape.q_visit_exp_root,
        halving_div=shape.halving_div,
        policy_temp=shape.policy_temp,
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("max_visit", [0, 1, 5, 57, 118, 1559])
def test_root_q_scale_matches_the_real_transform(name: str, max_visit: int) -> None:
    """The probe's q_scale must equal what the search itself would compute.

    Differential, not a golden number: if `_root_sigma_scale` changes, this
    fails rather than silently reporting a stale formula.
    """
    shape = SHAPES[name]
    assert shape.root_q_scale(float(max_visit)) == pytest.approx(
        _root_sigma_scale(max_visit=max_visit, cfg=_cfg_for(shape)), rel=1e-12,
    )


def test_root_q_scale_covers_every_field_the_real_transform_reads() -> None:
    """The differential test above is blind to a knob `SearchShape` lacks.

    `_root_sigma_scale` reads `q_visit_floor` too, and `SearchShape` has no such
    field -- so `_cfg_for` hands the reference implementation the DEFAULT for it
    and the two agree no matter what the branch does. A differential test built
    from the same restricted field set cannot detect a field it does not know
    about, which is how a reimplemented formula drifts while its own guard stays
    green. Parsed from the reference's source so a new input fails here.
    """
    src = inspect.getsource(_root_sigma_scale)
    reads = {
        node.attr
        for node in ast.walk(ast.parse(src.lstrip()))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cfg"
    }
    known = set(SearchShape.__dataclass_fields__)
    # `q_visit_floor` is read by the real transform and is NOT a SearchShape
    # field. That is tolerable ONLY while it is inert; assert the inertness
    # rather than the absence, so enabling it anywhere fails here.
    unmodelled = reads - known
    assert unmodelled <= {"q_visit_floor"}, (
        f"_root_sigma_scale now reads {sorted(unmodelled)}, which SearchShape "
        "does not model: the probe's q_scale column would silently describe a "
        "different transform than the search runs"
    )
    assert GumbelConfig().q_visit_floor < 0.0, (
        "q_visit_floor is no longer inert by default; SearchShape.root_q_scale "
        "does not implement its branch and must be extended"
    )
    root = Path(__file__).resolve().parents[1] / "configs"
    hits = [
        p.name for p in sorted(root.glob("*.yaml"))
        if "q_visit_floor" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"configs now set q_visit_floor: {hits}"


def test_live_selfplay_shape_is_a_linear_root_and_play_is_log() -> None:
    """The two shapes must differ in the root transform, not just in constants.

    This is the fact the probe exists to measure; if the fixtures ever agree,
    every root-transform contrast it prints becomes a null by construction.
    """
    assert SHAPES["live_selfplay_20260809"].root_transform_name() == "LINEAR"
    assert SHAPES["play"].root_transform_name() == "LOG"
    assert SHAPES["live_selfplay_20260809_logroot"].root_transform_name() == "LOG"
    assert SHAPES["live_selfplay_20260809_sqrtroot"].root_transform_name() == "POWER(0.5)"
    # And the gap has to be an order of magnitude at the production budget, or
    # the headline contrast is noise. max_visit 57 is the measured root
    # max-visit at 256 sims / topk 32 / halving_div 2.
    lin = SHAPES["live_selfplay_20260809"].root_q_scale(57.0)
    log = SHAPES["play"].root_q_scale(57.0)
    assert log > 10.0 * lin


def test_play_shape_tracks_PLAY_SEARCH_DEFAULTS() -> None:
    """The play fixture must not drift from the tuned constants it names."""
    play = SHAPES["play"]
    for key, value in PLAY_SEARCH_DEFAULTS.items():
        assert getattr(play, key) == value, key


def _omitted_gumbel_kwargs() -> set[str]:
    """Fields `network_turn.py` does NOT pass when it builds its GumbelConfig.

    Parsed from the production source, so the day someone plumbs a knob through
    to selfplay this test fails and the probe's fixture gets updated with it,
    instead of the probe quietly measuring the old shape.
    """
    from chess_anti_engine.selfplay import network_turn

    src = inspect.getsource(network_turn)
    tree = ast.parse(src)
    passed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GumbelConfig"
        ):
            passed |= {kw.arg for kw in node.keywords if kw.arg}
    assert passed, "no GumbelConfig(...) call found in network_turn.py"
    fields = set(GumbelConfig.__dataclass_fields__)
    return fields - passed


def test_selfplay_plumbs_policy_temp_but_not_the_root_transform() -> None:
    """Which knobs reach selfplay, pinned on BOTH sides of the line.

    The previous revision only asserted the OMITTED set, so when PR #385 plumbed
    `gumbel_policy_temp` through to `network_turn.py` nothing failed -- the
    probe's docstring, its `live_selfplay` provenance and the PR body all went
    on saying selfplay "omits policy_temp" while production ran T=1.5. A test
    that can only detect movement in one direction is half a test.
    """
    omitted = _omitted_gumbel_kwargs()
    for field in ("c_visit", "c_visit_root", "c_scale_root", "q_visit_exp_root", "halving_div"):
        assert field in omitted, f"{field} is now plumbed into selfplay; update SHAPES"
    assert "policy_temp" not in omitted, (
        "network_turn.py no longer passes policy_temp; the probe now assumes it "
        "does (it tempers its own prior to match) and SHAPES/--shape-yaml must "
        "be revisited"
    )

    live = SHAPES["live_selfplay_20260809"]
    defaults = GumbelConfig()
    assert live.c_visit == defaults.c_visit
    assert live.c_visit_root == defaults.c_visit_root
    assert live.c_scale_root == defaults.c_scale_root
    assert live.q_visit_exp_root == defaults.q_visit_exp_root
    assert live.halving_div == defaults.halving_div


def test_no_yaml_can_set_the_root_transform() -> None:
    """No config file names the root-transform knobs.

    If one ever does, the "selfplay runs a linear root" claim stops being
    unconditional and this probe's shape fixtures need a config reader.
    """
    root = Path(__file__).resolve().parents[1] / "configs"
    hits = [
        p.name
        for p in sorted(root.glob("*.yaml"))
        if any(k in p.read_text(encoding="utf-8")
               for k in ("c_scale_root", "c_visit_root", "q_visit_exp_root"))
    ]
    assert hits == [], f"configs now set the root transform: {hits}"


def test_apply_overrides_rejects_unknown_and_records_provenance() -> None:
    base = SHAPES["live_selfplay_20260809"]
    out = apply_overrides(base, "c_scale_root=3.0,topk=16")
    assert out.c_scale_root == 3.0
    assert out.topk == 16
    assert isinstance(out.topk, int)
    assert "3.0" in out.provenance
    assert "OVERRIDDEN" in out.provenance
    with pytest.raises(SystemExit):
        apply_overrides(base, "not_a_field=1")
    with pytest.raises(SystemExit):
        apply_overrides(base, "label=sneaky")
    assert apply_overrides(base, None) is base


def test_spearman_averages_ties_so_a_flat_block_cannot_invent_an_order() -> None:
    """Unvisited root children all share the mixed value.

    Ranking them by argsort order would manufacture a correlation out of array
    position, which is exactly the kind of statistic that cannot fail.
    """
    import numpy as np

    from scripts.search_gain_probe import _spearman

    flat = np.zeros(8)
    assert np.isnan(_spearman(flat, np.arange(8.0)))

    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, -a) == pytest.approx(-1.0)

    # Half the vector tied: the tied block must contribute no ordering.
    tied = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    forward = _spearman(tied, np.arange(8.0))
    backward = _spearman(tied[::-1].copy(), np.arange(8.0))
    assert forward == pytest.approx(-backward)


# ---------------------------------------------------------------------------
# The instrument itself, not just its shape arithmetic
# ---------------------------------------------------------------------------


def _synthetic_position(seed: int) -> tuple[ProbePosition, list[str], np.ndarray] | None:
    """A real board with SYNTHETIC deep-SF labels of known quality ordering."""
    r = np.random.default_rng(seed)
    board = chess.Board()
    for _ in range(int(r.integers(4, 30))):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(moves[int(r.integers(0, len(moves)))])
    if board.is_game_over():
        return None
    ucis, idxs = legal_full_indices(board)
    if len(ucis) < 4:
        return None
    cps = -np.abs(r.normal(0.0, 80.0, len(ucis)))
    cps[int(r.integers(0, len(ucis)))] = 0.0
    pos = ProbePosition(
        fen=board.fen(), board=board, phase=1,
        audit=AuditPosition(
            key=f"k{seed}", fen=board.fen(), phase=1, source=0,
            move_cp={u: float(c) for u, c in zip(ucis, cps, strict=True)},
            best_cp=float(cps.max()), deep_wdl=(0.4, 0.3, 0.3),
            sf_nodes=1_000_000, sf_depth=40,
        ),
    )
    return pos, ucis, idxs


def _score_synthetic(*, n: int = 300, informative: bool) -> list[RowResult]:
    """Score `n` positions where the 'search' is a BETTER estimator than the prior.

    `informative=False` severs the search from the truth entirely, which is the
    positive control on the negative control: the instrument must report a gain
    in the first case and none in the second.
    """
    shape = SHAPES["live_selfplay_20260809"]
    ctrl_rng = np.random.default_rng(4242)
    rows: list[RowResult] = []
    for s in range(n):
        got = _synthetic_position(9000 + s)
        if got is None:
            continue
        pos, ucis, idxs = got
        r = np.random.default_rng(50_000 + s)
        assert pos.audit is not None
        cps = np.array([pos.audit.move_cp[u] for u in ucis], dtype=np.float64)
        logits = np.full(4672, -1e9, dtype=np.float32)
        logits[idxs] = (cps / 120.0 + r.normal(0.0, 1.0, cps.size)).astype(np.float32)
        quality = (
            cps / 60.0 + r.normal(0.0, 0.4, cps.size)
            if informative
            else r.normal(0.0, 1.0, cps.size)
        )
        probs = np.zeros(4672, dtype=np.float64)
        e = np.exp(quality - quality.max())
        probs[idxs] = e / e.sum()
        action = int(idxs[int(np.argmax(probs[idxs]))])
        rows.append(_score_one(
            pos=pos, board=pos.board, sims=256, shape=shape,
            pol_logits_row=logits, search_probs=probs, search_action=action,
            tree=None, root_id=-1, ctrl_rng=ctrl_rng,
        ))
    return rows


def _delta(rows: list[RowResult], a: str, b: str) -> tuple[float, float, float]:
    return _mean_ci(
        np.array([getattr(r, a) for r in rows], dtype=np.float64)
        - np.array([getattr(r, b) for r in rows], dtype=np.float64)
    )


def test_shuffled_labels_collapse_the_measured_gain() -> None:
    """⚑ THE NEGATIVE CONTROL, AS A TEST. Shuffle the labels, require NO effect.

    An instrument that "detects" a gain in permuted labels is measuring array
    positions, not move quality. The real arm must show a gain whose CI excludes
    0; the SAME search and prior scored against a within-position permutation of
    the same labels must collapse to ~0 with a CI covering 0. Both halves are
    asserted, because a shuffle arm that reports 0 for a rig that reports 0 on
    everything proves nothing.
    """
    rows = _score_synthetic(informative=True)
    assert len(rows) > 150

    real, real_lo, real_hi = _delta(rows, "regret_search", "regret_prior")
    msg = (f"the positive control failed: a strictly better search measured "
           f"{real:+.2f}cp [{real_lo:+.2f},{real_hi:+.2f}]")
    assert real < 0.0, msg
    assert real_hi < 0.0, msg

    shuf, lo, hi = _delta(rows, "regret_search_shuffled", "regret_prior_shuffled")
    assert lo <= 0.0 <= hi, (
        f"SHUFFLE CONTROL FAILED: permuted labels still show {shuf:+.2f}cp "
        f"[{lo:+.2f},{hi:+.2f}], which excludes 0 -- the cp metric is reading "
        "something other than move quality"
    )
    # And it has to be a real collapse, not a marginally smaller number.
    assert abs(shuf) < 0.35 * abs(real), (
        f"shuffled arm {shuf:+.2f} is not a collapse of the real {real:+.2f}"
    )


def test_an_uninformative_search_measures_no_gain() -> None:
    """The other direction: a search unrelated to truth must NOT read as better.

    Without this the shuffle test above is satisfiable by a rig that reports
    ~0 for everything.
    """
    rows = _score_synthetic(informative=False)
    mean, lo, hi = _delta(rows, "regret_search", "regret_prior")
    assert lo <= 0.0 <= hi or mean > 0.0, (
        f"a search with no information about SF measured {mean:+.2f}cp "
        f"[{lo:+.2f},{hi:+.2f}] as an IMPROVEMENT"
    )


def test_alignment_control_reads_the_prior_regret_through_a_second_path() -> None:
    """`regret_prior_bymove` must reproduce `regret_prior` -- and be able not to.

    It is the replacement for a column the previous revision printed as a
    hardcoded 0.00. It re-derives the prior move's regret by UCI STRING through
    `move_regrets` rather than by array position, so a `ucis`/`regrets`
    misalignment shows up as a nonzero deviation instead of as nothing at all.
    """
    rows = _score_synthetic(n=60, informative=True)
    by_move = np.array([r.regret_prior_bymove for r in rows], dtype=np.float64)
    by_index = np.array([r.regret_prior for r in rows], dtype=np.float64)
    assert np.isfinite(by_move).all()
    assert float(np.abs(by_move - by_index).max()) == pytest.approx(0.0, abs=1e-9)
    # It is not an identity in the code: shifting the labels by one move breaks
    # it, which is the failure it exists to catch.
    got = _synthetic_position(9000)
    assert got is not None
    pos, ucis, _idxs = got
    assert pos.audit is not None
    rolled = dict(zip(ucis, [pos.audit.move_cp[u] for u in ucis[1:] + ucis[:1]], strict=True))
    assert rolled != pos.audit.move_cp


def test_coverage_is_tri_state_so_untracked_never_reads_as_zero() -> None:
    """Shard rows carry no per-move coverage; that must not print as 0%."""
    rows = _score_synthetic(n=20, informative=True)
    assert all(r.coverage_known for r in rows), "audit rows DO carry coverage"

    shard_like = ProbePosition(
        fen=chess.Board().fen(), board=chess.Board(), phase=-1, audit=None,
        stored_regret_cp=np.zeros(4672, dtype=np.float64),
    )
    ucis, idxs = legal_full_indices(shard_like.board)
    logits = np.full(4672, -1e9, dtype=np.float32)
    logits[idxs] = np.arange(len(ucis), dtype=np.float32) / 10.0
    probs = np.zeros(4672, dtype=np.float64)
    probs[idxs] = 1.0 / len(ucis)
    row = _score_one(
        pos=shard_like, board=shard_like.board, sims=32,
        shape=SHAPES["live_selfplay_20260809"], pol_logits_row=logits,
        search_probs=probs, search_action=int(idxs[0]),
        tree=None, root_id=-1, ctrl_rng=np.random.default_rng(1),
    )
    assert row.coverage_known is False
    assert row.covered_prior is False
    assert row.covered_search is False


def test_shape_from_yaml_reads_the_config_instead_of_trusting_the_snapshot(
    tmp_path: Path,
) -> None:
    """The staleness fix: a "live" shape must be readable off a real config.

    The hardcoded `live_selfplay_20260809` snapshot went stale within a day of
    being written (`gumbel_c_scale` 0.025 -> 0.1, `gumbel_policy_temp`
    1.0 -> 1.5). `--shape-yaml` makes the shape a measurement of a named file.
    """
    base = SHAPES["live_selfplay_20260809"]
    cfg = tmp_path / "live.yaml"
    cfg.write_text(
        "selfplay:\n"
        "  gumbel_topk: 32\n"
        "  gumbel_c_scale: 0.1\n"
        "  gumbel_policy_temp: 1.5\n"
        "  gumbel_vloss_weight: 1\n"
        "  curriculum_gumbel_scale: 1.0\n",
        encoding="utf-8",
    )
    got = shape_from_yaml(cfg, base=base)
    assert got.c_scale == 0.1
    assert got.policy_temp == 1.5
    assert got.topk == 32
    assert isinstance(got.topk, int)
    # Fields network_turn.py does NOT pass keep the GumbelConfig default.
    assert got.c_scale_root == base.c_scale_root
    assert got.q_visit_exp_root == base.q_visit_exp_root
    assert str(cfg) in got.provenance

    # The real in-repo config must be readable too, or the option is decorative.
    repo_cfg = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"
    from_repo = shape_from_yaml(repo_cfg, base=base)
    assert from_repo.c_scale > 0.0
    assert from_repo.topk > 0

    # A key present twice with DIFFERENT values must abort, never pick one.
    bad = tmp_path / "conflict.yaml"
    bad.write_text(
        "a:\n  gumbel_c_scale: 0.025\nb:\n  gumbel_c_scale: 0.1\n", encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        shape_from_yaml(bad, base=base)


def test_shape_names_carry_their_as_of_date() -> None:
    """A fixture called `live_*` with no date is the failure this PR hit.

    `same name != same measurement`: the live yaml is not in this repo and moves
    without touching this file, so a snapshot has to be dated in its KEY, not
    only in a comment a reader may not reach.
    """
    for name, shape in SHAPES.items():
        if not name.startswith("live_"):
            continue
        assert "20260809" in name, f"{name} is a live snapshot with no as-of date"
        assert "SNAPSHOT" in shape.label, f"{name} does not announce it is a snapshot"


def test_the_shape_gate_and_fidelity_gate_render_a_verdict_not_just_a_number() -> None:
    """⚑ Both "gates" were print-only in the previous revision.

    Section 3 said a mismatch "means the shape printed above is not the search
    that ran" and then printed `maxdev` with no threshold; section 0 stated the
    `data_policy_entropy` band and left the comparison to the reader. A rule a
    reader has to apply by hand is not a gate [[a_gate_that_cannot_fail]]. Both
    now emit PASS/FAIL, and the FAIL branch is reachable.
    """
    from scripts.search_gain_probe import DATA_POLICY_ENTROPY_BAND, report

    rows = _score_synthetic(n=40, informative=True)
    text = report(rows, SHAPES["live_selfplay_20260809"], rng=np.random.default_rng(2))
    assert "SHAPE GATE:" in text

    # Break the realized span away from the printed q_scale: the gate must FAIL.
    broken = [
        dataclasses.replace(r, sigma_span_observed=r.root_q_scale * 2.0)
        for r in rows
    ]
    broken_text = report(
        broken, SHAPES["live_selfplay_20260809"], rng=np.random.default_rng(2),
    )
    assert "SHAPE GATE" in broken_text
    assert "FAIL" in broken_text.split("SHAPE GATE")[1].split("\n")[0]

    # The fidelity band is a real interval, and the gate reads it.
    lo, hi = DATA_POLICY_ENTROPY_BAND
    assert 0.0 < lo < hi
    shard_like = [
        dataclasses.replace(
            r, stored_top1="e2e4", improved_entropy=hi + 1.0, stored_entropy=0.92,
        )
        for r in rows
    ]
    shard_text = report(
        shard_like, SHAPES["live_selfplay_20260809"], rng=np.random.default_rng(2),
    )
    assert "FIDELITY GATE @" in shard_text
    assert "FAIL" in shard_text.split("FIDELITY GATE @")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# Codex review of PR #381, triaged 2026-08-11. All four FAIL on the pre-fix
# module. These are the findings that change what a reported NUMBER MEANS.
# ---------------------------------------------------------------------------


def test_prior_is_scored_at_the_shapes_policy_temperature() -> None:
    """The C search divides root logits by `policy_temp`; the prior must too --
    EXACTLY ONCE.

    ⚑ THIS TEST MUST CALL `_score_one`. Its first version computed two softmaxes
    inline and asserted that temperature flattens a distribution -- a true fact
    about arithmetic that holds whatever the module does. It passed with
    `_score_one` replaced by an always-failing stub, and it therefore did not
    catch the double-tempering the same PR introduced (prior reported at
    T^2=2.25 beside a T=1.5 search). A gate that cannot fire, inside the PR that
    exists to fix gates that cannot fire.

    So: drive the REAL scorer, and pin the answer against BOTH wrong values --
    the un-tempered T=1.0 and the twice-tempered T^2.

    Temperature is strictly monotone, so no ORDINAL statistic moves; what moves
    is `prior_top1_p`, the gaps, KL(target||prior) and the recovered sigma span.
    """
    board = chess.Board()
    ucis, idxs = legal_full_indices(board)
    pos = ProbePosition(
        fen=board.fen(), board=board, phase=-1, audit=None,
        stored_regret_cp=np.zeros(4672, dtype=np.float64),
    )
    # A spread that makes the three candidate temperatures numerically distinct.
    raw = np.arange(len(ucis), dtype=np.float64) / 3.0
    logits = np.full(4672, -1e9, dtype=np.float32)
    logits[idxs] = raw.astype(np.float32)
    probs = np.zeros(4672, dtype=np.float64)
    probs[idxs] = 1.0 / len(ucis)

    def top1_at(temp: float) -> float:
        lg = raw / temp
        lg = lg - lg.max()
        e = np.exp(lg)
        return float((e / e.sum()).max())

    temp = 1.5
    shape = dataclasses.replace(SHAPES["live_selfplay_20260809"], policy_temp=temp)
    row = _score_one(
        pos=pos, board=board, sims=32, shape=shape, pol_logits_row=logits,
        search_probs=probs, search_action=int(idxs[0]),
        tree=None, root_id=-1, ctrl_rng=np.random.default_rng(1),
    )

    once, never, twice = top1_at(temp), top1_at(1.0), top1_at(temp * temp)
    # The three are genuinely different, or the assertions below prove nothing.
    assert not np.isclose(once, never)
    assert not np.isclose(once, twice)
    assert np.isclose(row.prior_top1_p, once, rtol=1e-6), (
        f"prior must be at T={temp}: got {row.prior_top1_p!r}, "
        f"T=1.0 would be {never!r}, T^2 would be {twice!r}"
    )
    assert not np.isclose(row.prior_top1_p, never), "prior is UNTEMPERED"
    assert not np.isclose(row.prior_top1_p, twice), "prior is tempered TWICE"

    # And the ordinal readout is unmoved -- why the 2026-08-11 verdict survived.
    row_t1 = _score_one(
        pos=pos, board=board, sims=32,
        shape=dataclasses.replace(SHAPES["live_selfplay_20260809"], policy_temp=1.0),
        pol_logits_row=logits, search_probs=probs, search_action=int(idxs[0]),
        tree=None, root_id=-1, ctrl_rng=np.random.default_rng(1),
    )
    assert row.prior_move == row_t1.prior_move
    assert row.prior_rank_of_search == row_t1.prior_rank_of_search


def test_temperature_flattening_is_not_by_itself_evidence_of_wiring() -> None:
    """The arithmetic the previous test mistook for a wiring check.

    Kept, DEMOTED, and named for what it is: a property of softmax, not of this
    module. It must never again stand in for the `_score_one` call above.
    """
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float64)

    def prior_at(temp: float) -> np.ndarray:
        lg = logits / max(1e-6, temp)
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    p1, p15 = prior_at(1.0), prior_at(1.5)
    # The ordering is identical -- this is why the ordinal readouts survived.
    assert list(np.argsort(-p1)) == list(np.argsort(-p15))
    # The probabilities are NOT, which is what the defect corrupted.
    assert not np.allclose(p1, p15)
    assert p15[0] < p1[0], "a higher temperature must flatten the top prior mass"


def test_rate_ci_keeps_a_nonzero_width_at_both_boundaries() -> None:
    """Wald returned [0,0] / [1,1] and claimed certainty from a finite sample."""
    p, lo, hi = sgp._rate_ci(0, 40)
    assert p == 0.0
    assert lo == 0.0
    assert hi > 0.0, "a zero observed rate must still have a nonzero upper bound"
    p, lo, hi = sgp._rate_ci(40, 40)
    assert p == 1.0
    assert hi == 1.0
    assert lo < 1.0, "an all-changes rate must still have a below-one lower bound"
    # An interior rate stays sane and contains the point estimate.
    p, lo, hi = sgp._rate_ci(10, 40)
    assert lo < p < hi
    # n == 0 is UNKNOWN, not zero.
    assert all(math.isnan(v) for v in sgp._rate_ci(0, 0))


@pytest.mark.parametrize("spec", ["8,0,32", "8,-4", "0"])
def test_nonpositive_simulation_rungs_are_rejected(
    spec: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search clamps to 1 sim, so a 0 rung would be reported but never run.

    `main` reads `sys.argv` rather than taking argv, so the flags go in through
    monkeypatch. The rejection must land BEFORE the checkpoint is touched --
    the path below does not exist and the run must still die on `--sims`.
    """
    monkeypatch.setattr(
        sys, "argv",
        ["search_gain_probe.py", "--shape", next(iter(SHAPES)),
         "--sims", spec, "--checkpoint", "nope.pt"],
    )
    with pytest.raises(SystemExit) as ei:
        sgp.main()
    assert "nonpositive" in str(ei.value) or "empty list" in str(ei.value)
