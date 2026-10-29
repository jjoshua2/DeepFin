"""Stage comparisons must not inherit Python's bounded int-from-string limit."""
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chess_anti_engine.utils.versioning import parse_version, version_lt


@pytest.fixture
def inert_manifest_check() -> Callable[[object, dict[str, object]], SimpleNamespace]:
    """Execute the real method body without importing/starting any worker code."""
    source = Path(__file__).resolve().parents[1] / "chess_anti_engine" / "worker.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "WorkerSession")
    method = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_check_manifest_compat"
    )
    unit = ast.Module(body=[method], type_ignores=[])
    namespace: dict[str, Any] = {
        "_ManifestCompat": SimpleNamespace, "version_lt": version_lt, "PACKAGE_VERSION": "0.0.3",
    }
    exec(compile(unit, str(source), "exec"), namespace)
    return cast(Callable[[object, dict[str, object]], SimpleNamespace], namespace[method.name])


@pytest.mark.parametrize("size", [4300, 4301])
@pytest.mark.parametrize("template", [
    "0.0.2a{}.dev1", "0.0.2.post{}.dev1", "0.0.2a1.post{}",
])
def test_parent_stage_limit_cannot_crash_newer_worker(
    inert_manifest_check: Callable[[object, dict[str, object]], SimpleNamespace],
    size: int, template: str,
) -> None:
    limit_before = sys.get_int_max_str_digits()
    minimum = template.format("9" * size)
    result = inert_manifest_check(None, {"min_worker_version": minimum})
    assert result.version_too_old is False
    assert result.min_worker_version == minimum
    assert sys.get_int_max_str_digits() == limit_before


@pytest.mark.parametrize("stage", ["a", "b", "rc", ".post", ".dev"])
@pytest.mark.parametrize("digits", ["9" * 4300, "9" * 4301, "9" * 10000],
                         ids=["4300-digit", "4301-digit", "10000-digit"])
def test_large_stage_orders_by_numeric_value_without_integer_conversion(
    stage: str, digits: str,
) -> None:
    limit_before = sys.get_int_max_str_digits()
    lower = f"1.0{stage}{digits}"
    higher = f"1.0{stage}1{'0' * len(digits)}"
    assert version_lt(lower, higher) is True
    assert version_lt(higher, lower) is False
    assert version_lt(lower, lower) is False
    assert parse_version(lower) == (1, 0, 0)
    assert sys.get_int_max_str_digits() == limit_before


@pytest.mark.parametrize("template", ["1.0a{}", "1.0rc1.post{}", "1.0.post2.dev{}"])
@pytest.mark.parametrize("digits", ["0" * 5000, "0" * 5000 + "12", "\u0660" * 5000 + "\u0661\u0662"],
                         ids=["all-zero", "ascii-leading-zero", "unicode-leading-zero"])
def test_long_leading_zeroes_and_unicode_digits_keep_existing_numeric_meaning(
    template: str, digits: str,
) -> None:
    normalized = "0" if set(digits) == {"0"} else "12"
    a, b = template.format(digits), template.format(normalized)
    assert version_lt(a, b) is False
    assert version_lt(b, a) is False


@pytest.mark.parametrize("zero", ["0", "\u0660", "\uff10", "\U0001d7ce"])
def test_already_accepted_unicode_decimal_zero_is_normalized(zero: str) -> None:
    for stage in ("a", ".post", ".dev"):
        a, b = f"1{stage}{zero}", f"1{stage}0"
        assert version_lt(a, b) is False
        assert version_lt(b, a) is False


def test_large_parent_stages_retain_composite_hierarchy() -> None:
    n = "9" * 4301
    ladder = [
        f"1.0a{n}.dev1", f"1.0a{n}", f"1.0a{n}.post1", "1.0b0.dev0",
        f"1.0rc{n}.post{n}.dev1", f"1.0rc{n}.post{n}", "1.0",
        f"1.0.post{n}.dev1", f"1.0.post{n}", "1.0.1",
    ]
    for i, a in enumerate(ladder):
        for j, b in enumerate(ladder):
            assert version_lt(a, b) is (i < j)


@pytest.mark.parametrize(("minimum", "too_old"), [
    ("garbage9.9.9", True), ("garbage", False), ("0.0.3", False),
    ("0.0.3.post1.dev1", True), ("0.0.3rc1", False),
])
def test_inert_caller_preserves_plain_and_malformed_requirement_policy(
    inert_manifest_check: Callable[[object, dict[str, object]], SimpleNamespace],
    minimum: str, too_old: bool,
) -> None:
    result = inert_manifest_check(None, {"min_worker_version": minimum})
    assert result.version_too_old is too_old
