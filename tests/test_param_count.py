"""The documented parameter count must be the number of parameters.

The defect this file exists to prevent (`docs/rl_loop_audit.md` A6/J11): the
production net's size was "corrected" from ~63M *up* to 78.8M on 2026-07-26 and
the correction was the error. 77,173,088 is ``sum(v.numel())`` over the
``state_dict`` entries, and a ``state_dict`` lists a tied parameter once per
*reference*. The 16 ``layer_smolgens.N.gen_weight.weight`` keys are one shared
``nn.Linear`` (built once in ``ChessNet.__init__`` and passed to every
``Smolgen``), so 15 x 1,048,576 = 15,728,640 params get counted that were never
allocated: 61,444,448 + 15,728,640 = 77,173,088 exactly.

Why it is worth a test rather than a careful docstring. The count is the
denominator for params-per-FLOP reasoning, for the ``matrix_optimizer_scope``
coverage figure (I12), and for every scale comparison against BT4 -- and the
number sits in prose, where nothing re-derives it. Both of the two wrong
figures found here (78.8M for production, "~105M" for the reference config)
were written by someone reasoning about the model rather than measuring it.

What these tests can NOT express, stated so nobody reads more into a green run:
they check the count the code produces against the count the docs claim. They
say nothing about whether the architecture is the intended one. A config change
that alters the model will fail them, and the correct response is to re-measure
and edit the docs -- not to update the constants to whatever came out.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parents[1]

# Measured 2026-07-26 by rebuilding each config and deduping by storage.
_PRODUCTION_PARAMS = 61_444_448
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
    the documented count would then change and the docs must be re-measured
    rather than quietly drifting into agreement with a naive sum.
    """
    sd = production_model.state_dict()
    gen_keys = [k for k in sd if k.endswith("gen_weight.weight")]
    assert len(gen_keys) == _TIED_GEN_WEIGHT_REFS
    storages = {sd[k].untyped_storage().data_ptr() for k in gen_keys}
    assert len(storages) == 1, "the per-layer Smolgen generator is supposed to be shared"

    naive = sum(int(v.numel()) for v in sd.values())
    assert _count_distinct(sd) == _PRODUCTION_PARAMS
    assert naive - _PRODUCTION_PARAMS == (_TIED_GEN_WEIGHT_REFS - 1) * _TIED_GEN_WEIGHT_NUMEL
    assert naive == 77_173_088, "the wrong number CLAUDE.md warns about"


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


@pytest.mark.parametrize(
    ("doc", "claims"),
    [
        ("CLAUDE.md", ("61,444,448", "61.44M", "73,700,885")),
        ("tcec.md", ("61,444,448",)),
    ],
)
def test_docs_quote_the_measured_count(doc: str, claims: tuple[str, ...]) -> None:
    """Every doc that states a count must state the measured one.

    Scoped to the files that describe the CURRENT model on purpose. Historical
    experiment write-ups (``docs/experiments/``, ``docs/threaded_dispatcher_results.md``,
    ``docs/REVIEW_BUG_HUNT.md``) name the model they were run against and must
    keep their own numbers; rewriting those would falsify the record. Same for
    ``docs/rl_loop_audit.md`` and ``docs/experiment_ledger.md``, which quote the
    wrong figures deliberately.

    ``AGENTS.md`` used to be listed here. It is now a pointer at ``CLAUDE.md``
    and states no count at all, which is why
    ``test_agents_md_stays_a_pointer_at_claude_md`` guards it instead: a doc that
    quotes nothing cannot quote the wrong number, but it CAN quietly grow a
    second copy of the rules, or vanish.
    """
    text = (_REPO / doc).read_text(encoding="utf-8")
    for claim in claims:
        assert claim in text, f"{doc} no longer states {claim}"


def test_agents_md_stays_a_pointer_at_claude_md() -> None:
    """``AGENTS.md`` must exist, stay short, and name ``CLAUDE.md``.

    Two opposite regressions, one gate. Deleting the file is not a
    consolidation: it is the file Codex loads, so the rules would stay written
    down in ``CLAUDE.md`` and silently stop being delivered to one of the agents
    that has to follow them. Letting it grow back into content re-creates the
    duplicate that drifts -- the version this repo shipped for months called the
    production net "384-dim, 12-layer, ~46M params".
    """
    agents = _REPO / "AGENTS.md"
    assert agents.is_file(), "AGENTS.md is what Codex loads; keep it as a pointer"
    text = agents.read_text(encoding="utf-8")
    assert "CLAUDE.md" in text, "AGENTS.md must point at CLAUDE.md"
    assert len(text) < 600, "AGENTS.md is a pointer, not a second copy of the rules"


def test_claude_md_syzygy_pair_matches_the_production_config() -> None:
    """The documented tablebase pair must be the pair the config actually uses.

    Same defect as the param count, in path form. ``CLAUDE.md`` carried
    ``.../syzygy_3-4-5:/mnt/e/chess/syzygy_6_dtz`` for a month after commit
    ``6a02200f2`` (2026-07-14) moved production to the local
    ``data/syzygy_6``, and the sentence *instructs an action* -- "pass the full
    pair as ``SyzygyPath`` to BOTH engines" -- so following the stale doc aimed
    an engine at an 82G external-drive copy while production read a 151G local
    one. Prose next to a config is not pinned by anything, which is why it drifts.

    Two assertions, and the second is the one that earns its keep. The first
    fails when the config moves and the doc does not. The second fails when the
    doc grows a SECOND pair -- the realistic regression here, since the stale
    path is still live in 15 research configs (14 ``configs/exp_*.yaml`` plus
    ``configs/bt4_aurora_asha.yaml``) and gets copied back in good faith. A bare
    "is the right pair mentioned" check passes happily on a file that also
    states the wrong one.
    """
    flat = flatten_run_config_defaults(load_yaml_file(str(_REPO / "configs" / "pbt2_small.yaml")))
    pair = str(flat["syzygy_path"])
    assert ":" in pair, "production syzygy_path is supposed to be a colon-separated pair"

    text = (_REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert f"`{pair}`" in text, f"CLAUDE.md must quote the production syzygy pair {pair}"

    quoted = set(re.findall(r"/[^\s`]*syzygy[^\s`]*:/[^\s`]*", text))
    assert quoted == {pair}, f"CLAUDE.md states a non-production syzygy pair: {quoted - {pair}}"


def test_claude_md_smolgen_share_matches_the_model(production_model: nn.Module) -> None:
    """CLAUDE.md's '26.7M of 61.44M (43.5%)' is a live claim, so measure it."""
    smolgen = sum(
        p.numel() for n, p in production_model.named_parameters()
        if n.startswith("layer_smolgens.")
    )
    total = _count_distinct(production_model)
    assert f"{smolgen / 1e6:.1f}M" == "26.7M"
    assert f"{100.0 * smolgen / total:.1f}%" == "43.5%"
    text = (_REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`layer_smolgens` **26.7M of 61.44M (43.5%)**" in text
