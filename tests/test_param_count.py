"""Architecture counts and tied-parameter accounting, independent of guide wording.

Config changes should deliberately remeasure these expectations. State-dict keys
are references and can count the same shared Smolgen storage more than once.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parents[1]

# Measured 2026-07-26 by rebuilding each config and deduping by storage.
_PRODUCTION_PARAMS = 63_084_128
_REFERENCE_PARAMS = 73_700_885
# 16 per-layer Smolgen slots, one shared (gen_sz=256) -> (64*64) generator.
_TIED_GEN_WEIGHT_NUMEL = 1_048_576
_TIED_GEN_WEIGHT_REFS = 16


def _count_distinct(source: nn.Module | Mapping[str, torch.Tensor]) -> int:
    """Parameter elements that actually exist, deduped by storage.

    ``nn.Module.parameters()`` already dedupes (``remove_duplicate=True`` is its
    default), so the module branch is the easy case. The mapping branch is the
    one that matters: it is what you are forced to use when all you have is a
    checkpoint on disk, and it is where the 78.8M came from.
    """
    if isinstance(source, nn.Module):
        return sum(p.numel() for p in source.parameters())
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in source.values():
        # int() around storage_offset: it is SymInt-typed for tracing, and a
        # suppression here would be a lie the day torch narrows the annotation.
        key = (tensor.untyped_storage().data_ptr(), int(tensor.storage_offset()))
        if key in seen:
            continue
        seen.add(key)
        total += int(tensor.numel())
    return total


def _build(config_name: str) -> nn.Module:
    flat = flatten_run_config_defaults(load_yaml_file(str(_REPO / "configs" / config_name)))
    return build_model(model_config_from_flat_config(flat))


@pytest.fixture(scope="module")
def production_model() -> nn.Module:
    return _build("pbt2_small.yaml")


def test_production_param_count(production_model: nn.Module) -> None:
    assert _count_distinct(production_model) == _PRODUCTION_PARAMS
    trainable = sum(p.numel() for p in production_model.parameters() if p.requires_grad)
    assert trainable == _PRODUCTION_PARAMS, "every production parameter should train"


def test_reference_config_param_count() -> None:
    assert _count_distinct(_build("default.yaml")) == _REFERENCE_PARAMS


def test_state_dict_sum_double_counts_the_tied_smolgen(production_model: nn.Module) -> None:
    """Pin the trap itself, not just the right answer.

    If the shared generator is ever untied, this goes red -- which is the point:
    the architecture count would change and must be re-measured rather than
    quietly drifting into agreement with a naive sum.
    """
    sd = production_model.state_dict()
    gen_keys = [k for k in sd if k.endswith("gen_weight.weight")]
    assert len(gen_keys) == _TIED_GEN_WEIGHT_REFS
    storages = {sd[k].untyped_storage().data_ptr() for k in gen_keys}
    assert len(storages) == 1, "the per-layer Smolgen generator is supposed to be shared"

    naive = sum(int(v.numel()) for v in sd.values())
    assert _count_distinct(sd) == _PRODUCTION_PARAMS
    assert naive - _PRODUCTION_PARAMS == (_TIED_GEN_WEIGHT_REFS - 1) * _TIED_GEN_WEIGHT_NUMEL
    assert naive == 78_812_768, "the naive sum includes repeated shared weights"


def test_dedupe_helper_can_actually_see_tying() -> None:
    """The instrument must fail when the thing it measures is wrong.

    Without this, ``_count_distinct`` returning the naive sum would still make
    every assertion above pass on an untied model and quietly stop testing
    anything on a tied one.
    """
    shared = nn.Linear(4, 4, bias=False)
    tied = nn.ModuleList([shared, shared])
    sd = tied.state_dict()
    assert sum(int(v.numel()) for v in sd.values()) == 32
    assert _count_distinct(sd) == 16
    assert _count_distinct(tied) == 16


def test_production_smolgen_share(production_model: nn.Module) -> None:
    """Keep the architecture's tied Smolgen allocation visible independently of prose."""
    smolgen = sum(
        p.numel() for n, p in production_model.named_parameters()
        if n.startswith("layer_smolgens.")
    )
    total = _count_distinct(production_model)
    assert f"{smolgen / 1e6:.1f}M" == "26.7M"
    assert f"{100.0 * smolgen / total:.1f}%" == "42.3%"
