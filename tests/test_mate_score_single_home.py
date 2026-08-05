"""One mate -> score mapping, pinned at every site that folds a mate.

The defect (audit N1): the codebase carried TWO mate->cp formulas.
`stockfish.wdl.mate_to_effective_cp` mapped mates into +-1500..2480 — INSIDE
the reachable cp band, since the shard clamps cp at +-32000 and Stockfish
routinely reports |cp| ~ 20000 in decisive endgames — while
`selfplay.finalize._sf_move_score` and its C twin used +-100000. On 1.34% of
live scored rows (40 shards of the 0f888 lineage, 75,356 positions with >= 2
scored PVs) the two named a DIFFERENT best move, and both feed `policy_own`:
the cp-mode policy target from the first, `sf_p0_regret` from the second at 7x
the loss weight. See the worked example pinned in
`test_worked_example_mate_beats_cp_decliners`.

These tests pin the properties that make the mapping single-homed:
  * the mate band cannot overlap the cp band (the defect itself),
  * distinct mates stay ordered, in both signs,
  * every twin (numpy, torch, the C mirror, `_sf_move_score`) equals the scalar
    definition rather than re-deriving it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.selfplay.finalize import _sf_move_score
from chess_anti_engine.stockfish.wdl import (
    _MATE_BASE_CP,
    _MATE_DEPTH_STEP_CP,
    _MATE_MAX_PLIES,
    SF_CP_CLAMP_CP,
    mate_to_effective_cp,
    mate_to_effective_cp_array,
)
from chess_anti_engine.train.sparse_sf_ce import _mate_to_effective_cp_torch
from chess_anti_engine.train.target_builder import (
    SfTargetParams,
    rebuild_sf_policy_target,
)

# Every mate SF can emit, plus the clamp region past it.
_MATES = [1, 2, 3, 5, 9, 10, 30, 49, 50, 51, 127, 499, 500, 501, 5000, 32767]


def test_mate_band_cannot_overlap_the_cp_band() -> None:
    """THE defect. The weakest mate must still outrank the strongest cp.

    Reverting `_MATE_BASE_CP` to its historical 1500 fails here first.
    """
    weakest = min(abs(mate_to_effective_cp(m)) for m in _MATES)
    assert weakest > SF_CP_CLAMP_CP, (
        f"weakest mate scores {weakest} cp, inside the +-{SF_CP_CLAMP_CP} cp "
        f"band — a large non-mating cp line now outranks a forced mate"
    )
    # the floor is what guarantees it for arbitrarily long mates
    assert _MATE_BASE_CP - _MATE_MAX_PLIES * _MATE_DEPTH_STEP_CP > SF_CP_CLAMP_CP


def test_distinct_mates_stay_ordered_in_both_signs() -> None:
    # winning: quicker mate is better
    assert mate_to_effective_cp(2) > mate_to_effective_cp(9)
    assert mate_to_effective_cp(1) > mate_to_effective_cp(2)
    # losing: being mated LATER is better (less negative)
    assert mate_to_effective_cp(-9) > mate_to_effective_cp(-2)
    # and every winning mate beats every losing one
    assert min(mate_to_effective_cp(m) for m in _MATES) > max(
        mate_to_effective_cp(-m) for m in _MATES
    )


def test_long_mates_clamp_instead_of_wrapping_into_the_cp_band() -> None:
    """Without the ply floor a long enough mate walks down through cp and
    eventually flips sign. int16 holds mate up to 32767, so this is
    representable in a stored row even though SF never emits it."""
    for m in (500, 501, 5000, 32767):
        assert mate_to_effective_cp(m) == mate_to_effective_cp(500)
        assert mate_to_effective_cp(-m) == mate_to_effective_cp(-500)
    assert mate_to_effective_cp(32767) > SF_CP_CLAMP_CP


@pytest.mark.parametrize("mate", _MATES)
def test_every_twin_equals_the_scalar_definition(mate: int) -> None:
    """numpy / torch / `_sf_move_score` are twins, not second formulas."""
    for m in (mate, -mate):
        want = mate_to_effective_cp(m)
        assert float(mate_to_effective_cp_array(np.array([m]))[0]) == want
        assert float(
            _mate_to_effective_cp_torch(torch.tensor([m], dtype=torch.int32))[0]
        ) == want
        # mate takes precedence over any cp, including the sentinel
        assert _sf_move_score(0, m) == want
        assert _sf_move_score(1234, m) == want
        assert _sf_move_score(SF_CP_SENTINEL, m) == want


def test_c_mirror_literals_match_the_python_constants() -> None:
    """The C twin hardcodes the band; pin its LITERALS to the Python home.

    A behavioural pin cannot do this job. `_build_sf_p0_regret_vector` — the
    only consumer of the C scorer — emits capped DIFFERENCES
    (`SF_OWN_REGRET_CAP_CP` = 1000cp), and a shared offset cancels in every
    difference: moving the C base from 100000 to 90000 leaves every regret in
    `test_native_sf_finalize.py` bit-identical, because mate-vs-cp gaps
    saturate to 1.0 either way and mate-vs-mate gaps depend only on the step.
    (Verified: that mutation SURVIVED the parity test before this test
    existed.) The step and the ply floor are pinned behaviourally over there;
    the base can only be pinned here.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / (
        "chess_anti_engine/encoding/_lc0_ext.c"
    )
    text = src.read_text()
    start = text.index("static inline int sf_multipv_row_score")
    body = text[start:text.index("\n}", start)]
    mate_branch = body[body.index("if (mate != 0)"):]
    # Strip C comments first — the branch's own comment cites these constants,
    # and a literal quoted in prose must not satisfy (or break) the pin.
    code_only = re.sub(r"/\*.*?\*/", "", mate_branch, flags=re.DOTALL)
    literals = [float(x) for x in re.findall(r"(\d+\.\d+)", code_only)]
    assert sorted(set(literals)) == sorted(
        {_MATE_BASE_CP, _MATE_DEPTH_STEP_CP, _MATE_MAX_PLIES, 1.0}
    ), (
        f"the C mate branch's literals {sorted(set(literals))} no longer match "
        f"the Python home (base {_MATE_BASE_CP}, step {_MATE_DEPTH_STEP_CP}, "
        f"floor {_MATE_MAX_PLIES}) — the two mappings have split again"
    )


def test_sf_move_score_non_mate_rows_are_untouched() -> None:
    assert _sf_move_score(250, 0) == 250.0
    assert _sf_move_score(-40, 0) == -40.0
    assert _sf_move_score(SF_CP_SENTINEL, 0) is None


def test_worked_example_mate_beats_cp_decliners() -> None:
    """The live row that proves the split, replayed through the production
    cp-mode target build.

    `runs/pbt2_small/replay/train_trial_0f888_.../shard_001187.zarr` index 642:
    a forced mate in 3 alongside four cp-19996/19998 moves that decline it.
    Under the old band the mate scored 2440, took 0.000000 of the target mass
    and the decliners split ~1.0 — while `sf_p0_regret`, built from the OTHER
    formula at 7x the weight, gave those same decliners the maximum 1.0 regret
    and the mate 0.0. Reverting `_MATE_BASE_CP` to 1500 fails this test.
    """
    rows = np.full((5, 5), -1, dtype=np.int16)
    rows[0] = (1420, SF_CP_SENTINEL, 3, -1, -1)   # the forced mate
    rows[1] = (1634, 19998, 0, -1, -1)
    rows[2] = (1647, 19996, 0, -1, -1)
    rows[3] = (1654, 19996, 0, -1, -1)
    rows[4] = (1635, 19996, 0, -1, -1)
    legal = np.array([1420, 1634, 1647, 1654, 1635], dtype=np.int64)
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_policy_score_mode="cp", sf_policy_cp_temp=16.2,
    )

    target = rebuild_sf_policy_target(
        rows, legal_indices=legal, policy_size=4672, params=params,
    )
    assert target is not None
    mate_mass = float(target[1420])
    assert mate_mass > 0.98, (
        f"the forced mate holds {mate_mass:.6f} of the cp-mode policy target; "
        f"the mate band is back inside the cp band"
    )

    # ...and the regret target, built by the other path, must AGREE about the
    # best move rather than contradict it.
    scores = [_sf_move_score(int(cp), int(m)) for _i, cp, m, _w, _d in rows.tolist()]
    assert scores[0] == max(s for s in scores if s is not None)
