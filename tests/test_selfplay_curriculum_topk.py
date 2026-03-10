from __future__ import annotations

from chess_anti_engine.selfplay.manager import _effective_curriculum_topk


def test_effective_curriculum_topk_shrinks_as_random_stage_burns_off() -> None:
    assert _effective_curriculum_topk(random_move_prob=0.8, stage_end=0.5, topk_max=12) == 12
    assert _effective_curriculum_topk(random_move_prob=0.5, stage_end=0.5, topk_max=12) == 12
    assert _effective_curriculum_topk(random_move_prob=0.4, stage_end=0.5, topk_max=12) == 10
    assert _effective_curriculum_topk(random_move_prob=0.2, stage_end=0.5, topk_max=12) == 5
    assert _effective_curriculum_topk(random_move_prob=0.0, stage_end=0.5, topk_max=12) == 1
