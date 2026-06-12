"""Smoke-run the batch-encoder differential fuzzer (scripts/fuzz_batch_encode_diff.py).

The _mcts_tree batch encoders are the production inference encode path; each
.so carries its own buffer-layout loops and module globals (including the
history_rep_fix flag), so batch/single parity is a contract to assert, not an
identity. A few games here catch gross regressions in CI; the real budgets
live in scripts/fuzz/run_fuzz.sh (mode: batch).
"""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from tests.script_loading import load_script_module

_MOD = load_script_module("fuzz_batch_encode_diff.py")


@pytest.fixture(autouse=True)
def _reset_flag():
    rep_fix.apply(False)
    yield
    rep_fix.apply(False)


@pytest.mark.parametrize("history_rep_fix", [False, True])
def test_batch_encoders_match_single_board_oracle(history_rep_fix: bool) -> None:
    failure = _MOD.run(
        games=4, seed=99, max_plies=120, check_every=6, batch_cap=24,
        history_rep_fix=history_rep_fix,
    )
    assert failure is None, str(failure)


def test_bf16_reference_matches_c_rounding() -> None:
    """The numpy RNE reference must reproduce C float_to_bf16_bits exactly,
    including ties — otherwise the bf16 oracle would chase phantom diffs."""
    vals = np.array(
        [0.0, -0.0, 1.0, -1.0, 0.5, 1.0 / 3.0, 100.0, 1e-30, 3.14159265,
         np.float32(1.0039062)],  # mantissa tie territory
        dtype=np.float32,
    )
    bits = _MOD._f32_to_bf16_bits(vals)
    # Reconstruct and check round-trip error is within 1 bf16 ulp. (Values
    # near float32 max are excluded: RNE correctly overflows them to inf.)
    back = (bits.astype(np.uint32) << 16).view(np.float32)
    rel = np.abs(back - vals) / np.maximum(np.abs(vals), 1e-38)
    assert float(rel.max()) <= 2.0 ** -8
