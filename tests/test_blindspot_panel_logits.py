"""Numerical contract tests without importing the model/CLI runtime.

Execute the panel's exact AST normalization block and severity expression, not
an independent softmax implementation. Expected values below are analytic.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np


def _panel_values(logits: list[list[float]]) -> list[float]:
    source = Path("scripts/blindspot_panel.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(
        i for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign) and ast.unparse(n).startswith("wdl = np.asarray(")
    )
    end = next(
        i for i in range(start + 1, len(main.body))
        if isinstance(main.body[i], ast.Expr)
    )
    block = ast.Module(body=main.body[start:end], type_ignores=[])
    # Reject any accidental expansion into runtime calls before executing.
    allowed = {"np.asarray", "np.allclose", "np.exp", "wdl.sum", "wdl.max", "e.sum"}
    for node in ast.walk(block):
        assert isinstance(node, (ast.Assign, ast.Name, ast.Load, ast.Store,
                                 ast.Call, ast.Attribute, ast.keyword, ast.Constant,
                                 ast.If, ast.UnaryOp, ast.Not, ast.BinOp, ast.Sub,
                                 ast.Div, ast.Module)), type(node).__name__
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) in allowed
    namespace: dict[str, Any] = {"np": np, "wdl": logits, "__builtins__": {"__import__": __import__}}
    exec(compile(block, str(source), "exec"), namespace)
    assignment = next(
        n for loop in main.body if isinstance(loop, ast.For)
        for n in loop.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "net_after" for t in n.targets)
    )
    expression = ast.Expression(body=assignment.value)
    assert ast.unparse(expression) == "float(w[2] - w[0])"
    severity = compile(expression, str(source), "eval")
    return [eval(severity, {"float": float, "w": row, "__builtins__": {}})
            for row in namespace["wdl"]]


def _close(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    assert all(abs(a - b) < 1e-12 for a, b in zip(actual, expected, strict=True)), actual


def test_sum_one_logits_use_probabilities_and_correct_awareness() -> None:
    expected = (1.0 - math.e) / (math.e + 2.0)
    actual = _panel_values([[1.0, 0.0, 0.0]])
    _close(actual, [expected])
    assert not actual[0] < -0.4  # the old heuristic reports -1, incorrectly AWARE
    assert actual[0] < -0.2


def test_multiple_sum_one_logits_and_class_orientation() -> None:
    expected = (math.e - 1.0) / (math.e + 2.0)
    _close(_panel_values([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
           [-expected, 0.0, expected])


def test_additive_shifts_do_not_change_severity() -> None:
    expected = (1.0 - math.e) / (math.e + 2.0)
    for shift in (-1000.0, -1.0, 0.0, 1.0, 1000.0):
        _close(_panel_values([[1.0 + shift, shift, shift]]), [expected])


def test_batch_composition_and_order_do_not_change_a_row() -> None:
    expected = (1.0 - math.e) / (math.e + 2.0)
    _close(_panel_values([[1.0, 0.0, 0.0]]), [expected])
    _close(_panel_values([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), [expected, 0.0])
    _close(_panel_values([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), [0.0, expected])


def test_ordinary_logits_and_uniform_control() -> None:
    expected = (1.0 - math.e ** 2) / (math.e ** 2 + math.e + 1.0)
    _close(_panel_values([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), [expected, 0.0])
