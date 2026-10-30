"""Exercise checkpoint scalar validation without importing the training runtime."""

from __future__ import annotations

import ast
import copy
import logging
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _inert_trainer() -> Any:
    path = Path(__file__).resolve().parents[1] / "chess_anti_engine/train/trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer"
    )
    names = {"load_zclip_state", "_restore_zclip_stats"}
    methods = [
        n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {method.name for method in methods} == names
    inert = ast.ClassDef(
        name="InertTrainer", bases=[], keywords=[], body=[*methods], decorator_list=[]
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "logging": logging,
        "__name__": __name__,
    }
    module = ast.fix_missing_locations(ast.Module(body=[inert], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["InertTrainer"]()


class TestZclipVarianceDomain(unittest.TestCase):
    trainer: Any = None

    def setUp(self) -> None:
        self.trainer = _inert_trainer()
        self.trainer.zclip = SimpleNamespace(
            initialized=True,
            mean=5.0,
            var=2.0,
            buffer=[3.0],
            warmup_steps=25,
        )

    def test_negative_variance_refused_without_mutating_fresh_or_current_state(
        self,
    ) -> None:
        for current_initialized in (True, False):
            for incoming_initialized in (True, False):
                for variance in (-1.0, "-0.25", -5e-324):
                    with self.subTest(
                        current=current_initialized,
                        incoming=incoming_initialized,
                        var=variance,
                    ):
                        self.trainer.zclip = SimpleNamespace(
                            initialized=current_initialized,
                            mean=5.0 if current_initialized else None,
                            var=2.0 if current_initialized else None,
                            buffer=[3.0] if current_initialized else [],
                            warmup_steps=25,
                        )
                        before = copy.deepcopy(vars(self.trainer.zclip))
                        payload = {
                            "initialized": incoming_initialized,
                            "mean": 1,
                            "var": variance,
                            "buffer": [2],
                        }
                        with self.assertLogs(__name__, level="WARNING") as logs:
                            assert not self.trainer.load_zclip_state(payload)
                        assert "negative" in " ".join(logs.output)
                        assert vars(self.trainer.zclip) == before

    def test_zero_positive_and_partial_warmup_remain_valid(self) -> None:
        for initialized, mean, variance in (
            (True, 1, 0),
            (True, 1, -0.0),
            (True, 1, 4),
            (False, None, None),
            (False, 1, 0),
            (False, 1, 4),
        ):
            with self.subTest(initialized=initialized, var=variance):
                payload = {
                    "initialized": initialized,
                    "mean": mean,
                    "var": variance,
                    "buffer": [2, 3],
                }
                assert self.trainer.load_zclip_state(payload)
                assert self.trainer.zclip.initialized == initialized
                assert self.trainer.zclip.mean == mean
                assert self.trainer.zclip.var == variance
                assert self.trainer.zclip.buffer == [2, 3]

    def test_existing_malformed_and_nonfinite_refusal_preserves_state(self) -> None:
        payloads: list[Any] = [
            None,
            [],
            {},
            {"initialized": True, "mean": None, "var": 1},
        ]
        for field in ("mean", "var", "buffer"):
            for value in (float("nan"), float("inf"), float("-inf"), "bad"):
                payload: dict[str, Any] = {
                    "initialized": True,
                    "mean": 1,
                    "var": 1,
                    "buffer": [],
                }
                payload[field] = [value] if field == "buffer" else value
                payloads.append(payload)
        for payload in payloads:
            with self.subTest(payload=payload):
                before = copy.deepcopy(vars(self.trainer.zclip))
                with self.assertLogs(__name__, level="INFO"):
                    assert not self.trainer.load_zclip_state(payload)
                assert vars(self.trainer.zclip) == before


if __name__ == "__main__":
    unittest.main()
