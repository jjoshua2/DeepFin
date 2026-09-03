from __future__ import annotations

import numpy as np

from chess_anti_engine.inference_cache import EncodedEvalCache


class _RelationsRequiredEvaluator:
    input_extra_features = "v1"
    input_history_encoding = "legacy"
    policy_encoding = "lc0_1858"
    use_dynamic_relations = True

    def __init__(self) -> None:
        self.seen_relations: np.ndarray | None = None

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is None:
            raise AssertionError("relations were dropped by the cache")
        self.seen_relations = relations
        n = int(x.shape[0])
        return (
            np.zeros((n, 4), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
        )


def test_empty_dense_batch_forwards_relations_to_inner_evaluator() -> None:
    inner = _RelationsRequiredEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    batch = np.zeros((0, 146, 8, 8), dtype=np.float32)
    relations = np.zeros((0, 2, 4, 4), dtype=np.uint8)

    policy, wdl = cache.evaluate_encoded(batch, relations=relations)

    assert inner.seen_relations is relations
    assert policy.shape == (0, 4)
    assert wdl.shape == (0, 3)
    stats = cache.stats()
    assert stats.requests == 0
    assert stats.inner_calls == 0
    assert stats.rows_submitted == 0
