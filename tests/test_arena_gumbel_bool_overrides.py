"""``--*-gumbel`` overrides of a BOOL GumbelConfig field are strict.

The number parser used to accept ``target_untempered_prior=0.5``: banked as 0.5,
read as True by every consumer's ``bool(...)`` -- accepted and then silently
coerced. A bool field now takes 0/1/true/false only, stored as a real bool.
"""
from __future__ import annotations

import pytest

from scripts.arena_standard import apply_search_overrides, resolve_search_shape


@pytest.mark.parametrize(("spelling", "value"), [("1", True), ("true", True), ("TRUE", True),
                                                 ("0", False), ("false", False)])
def test_a_bool_field_takes_a_real_bool(spelling: str, value: bool) -> None:
    side = apply_search_overrides(resolve_search_shape("play"), spec=f"target_untempered_prior={spelling}")
    got = side.gumbel["target_untempered_prior"]
    assert got is value


@pytest.mark.parametrize("spelling", ["0.5", "2", "yes", "", "1.0"])
def test_a_bool_field_refuses_anything_else(spelling: str) -> None:
    with pytest.raises(SystemExit, match="target_untempered_prior"):
        apply_search_overrides(resolve_search_shape("play"), spec=f"target_untempered_prior={spelling}")


def test_numeric_fields_are_unchanged() -> None:
    side = apply_search_overrides(resolve_search_shape("play"), spec="c_scale=0.25,topk=8")
    assert side.gumbel["c_scale"] == 0.25
    assert side.gumbel["topk"] == 8
