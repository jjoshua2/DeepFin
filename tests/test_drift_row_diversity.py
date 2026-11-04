"""Numeric-only qualification: extract exact pure helpers, never import replay.

Run directly with unittest; no repository conftest or runtime imports.
"""
from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
import unittest

import numpy as np


@dataclass
class Metrics:
    data_policy_entropy: float = 0.0
    data_unique_positions: float = 0.0
    data_wdl_balance: float = 0.0


class Arrays:
    def __init__(self, x: np.ndarray, size: int | None = None) -> None:
        self.x = x
        self.size = len(x) if size is None else size

    def __len__(self) -> int:
        return self.size

    def sample_batch_arrays(self, n: int, *, wdl_balance: bool) -> dict[str, np.ndarray]:
        del n, wdl_balance
        return {"x": self.x, "wdl_target": np.ones(len(self.x)),
                "policy_target": np.ones((len(self.x), 2))}


def load_metrics() -> Callable[..., Metrics]:
    source = Path(__file__).resolve().parents[1] / "chess_anti_engine/tune/trainable_metrics.py"
    tree = ast.parse(source.read_text())
    names = {"_sample_drift_arrays", "_mean_entropy", "_wdl_hist", "_compute_drift_metrics"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(body) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body], type_ignores=[])
    scope = {"np": np, "DriftMetrics": Metrics}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return cast(Callable[..., Metrics], scope["_compute_drift_metrics"])


class RowDiversityTest(unittest.TestCase):
    def metric(self, x: np.ndarray) -> float:
        fn = load_metrics()
        result = fn(buf=Arrays(x), holdout_buf=Arrays(x[:0]), drift_sample_size=len(x))
        return result.data_unique_positions

    def test_equal_sum_distinct_and_duplicate_rows(self) -> None:
        rows = np.zeros((3, 146, 8, 8), dtype=np.float32)
        rows[0, 0, 0, 0] = 1
        rows[1, 0, 0, 1] = 1
        rows[2] = rows[0]
        assert self.metric(rows) == 2 / 3

    def test_identical_and_singleton(self) -> None:
        assert self.metric(np.ones((4, 2, 8, 8), dtype=np.float32)) == 1 / 4
        assert self.metric(np.ones((1, 2, 8, 8), dtype=np.float32)) == 1

    def test_sampler_casts_float_dtype_and_handles_strides(self) -> None:
        for dtype in (np.float16, np.float32, np.float64):
            rows = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=dtype)
            with self.subTest(dtype=dtype):
                assert self.metric(rows[:, ::2]) == 2 / 3

    def test_bit_identity_is_not_numeric_equality(self) -> None:
        rows = np.array([[0], [0x80000000], [0x7FC00001], [0x7FC00002]], dtype=np.uint32).view(np.float32)
        assert self.metric(rows) == 1

    def test_embedded_and_trailing_zero_bytes(self) -> None:
        bits = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint32)
        assert self.metric(bits.view(np.float32)) == 2 / 3

    def test_empty_buffer_short_circuits(self) -> None:
        fn = load_metrics()
        empty = Arrays(np.empty((0, 146, 8, 8), dtype=np.float32))
        assert fn(buf=empty, holdout_buf=empty, drift_sample_size=256).data_unique_positions == 0

    def test_empty_sample_preserves_existing_error_policy(self) -> None:
        fn = load_metrics()
        empty = Arrays(np.empty((0, 146, 8, 8), dtype=np.float32), size=1)
        try:
            fn(buf=empty, holdout_buf=empty, drift_sample_size=1)
        except ValueError:
            pass
        else:
            raise AssertionError("empty sample must retain baseline ValueError")


if __name__ == "__main__":
    unittest.main()
