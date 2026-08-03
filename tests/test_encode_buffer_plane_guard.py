"""Plane-count guards for buffers the C encoders read their layout from.

Encoding audit E4: every C batch encoder infers the feature-plane count from
the OUTPUT BUFFER (``n_extra = PyArray_DIM(out, 1) - 112``) and never sees
``input_extra_features``; the C shape check accepts either 146 or 175 and then
trusts the buffer. So a buffer sized from one source of truth and a config
saying something else encode a different plane layout rather than raising. The
two branches of ``run_mcts_many_c`` disagreed about which source of truth wins
(evaluator buffer when ``n_boards <= _max_batch``, config otherwise), which
would make the SAME run encode differently depending on batch size.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import check_encode_buffer_planes, input_plane_count
from chess_anti_engine.inference import require_model_planes


def test_matching_buffer_passes() -> None:
    buf = np.zeros((4, 175, 8, 8), dtype=np.float32)
    check_encode_buffer_planes(buf, "v2_threats", where="unit")
    buf146 = np.zeros((4, 146, 8, 8), dtype=np.float32)
    check_encode_buffer_planes(buf146, "v1", where="unit")


def test_v1_buffer_under_v2_config_raises() -> None:
    """The live shape: a 146-wide pinned slot while the config says v2_threats.

    The C encoder accepts this and writes v1 features into 146 of the planes.
    Nothing downstream can tell, because the *count* is a legal count.
    """
    buf = np.zeros((4, input_plane_count("v1"), 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"146 planes but .*v2_threats.* implies 175"):
        check_encode_buffer_planes(buf, "v2_threats", where="unit")


def test_v2_buffer_under_v1_config_raises() -> None:
    buf = np.zeros((4, input_plane_count("v2_threats"), 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"175 planes but .*v1.* implies 146"):
        check_encode_buffer_planes(buf, "v1", where="unit")


def test_message_names_the_call_site() -> None:
    buf = np.zeros((4, 146, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="selfplay root inplace"):
        check_encode_buffer_planes(
            buf, "v2_threats", where="selfplay root inplace (fp32)",
        )


def test_bf16_bits_buffer_is_checked_the_same_way() -> None:
    """The bf16 slot path carries the same layout in uint16 bits."""
    buf = np.zeros((4, 146, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="implies 175"):
        check_encode_buffer_planes(buf, "v2_threats", where="unit")


class _Model(torch.nn.Module):
    def __init__(self, extra: str, history: str = "lc0_root_legacy_meta") -> None:
        super().__init__()
        self.input_extra_features = extra
        self.input_history_encoding = history


def test_require_model_planes_accepts_a_matching_slot() -> None:
    require_model_planes(_Model("v2_threats"), 175, where="unit")
    require_model_planes(_Model("v1"), 146, where="unit")


def test_require_model_planes_rejects_the_146_default_under_a_175_model() -> None:
    """The exact production hazard: --input-planes left at its v1 default.

    146 is the argparse default on BOTH sides of the broker/worker shared-memory
    contract (inference.py --input-planes, worker.py
    --inference-slot-input-planes) while production is 175.
    """
    with pytest.raises(ValueError, match=r"146 input planes.*v2_threats.* = 175"):
        require_model_planes(_Model("v2_threats"), 146, where="unit")


def test_require_model_planes_refuses_an_undeclared_model() -> None:
    """A model that does not declare its encoding must not be guessed at."""
    model = _Model("v2_threats")
    del model.input_history_encoding
    with pytest.raises(ValueError, match="input_history_encoding"):
        require_model_planes(model, 175, where="unit")


# ---------------------------------------------------------------------------
# On the production search path, not just in the helper
# ---------------------------------------------------------------------------


class _NarrowSlotEvaluator:
    """An evaluator whose pinned slot is 146 planes wide (the v1 default).

    Shaped like ``DirectGPUEvaluator``/``AOTEvaluator`` for the in-place branch:
    ``get_input_buffer`` + ``evaluate_inplace`` + ``n_slots`` + ``_max_batch``.
    """

    n_slots = 1
    _max_batch = 64

    def __init__(self, planes: int) -> None:
        self._buf = np.zeros((self._max_batch, planes, 8, 8), dtype=np.float32)

    def get_input_buffer(self, n: int, slot: int = 0) -> np.ndarray:
        assert slot == 0
        return self._buf[:n]

    def evaluate_inplace(
        self, n: int, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError(
            f"evaluate_inplace(n={n}, slot={slot}) must not be reached on a "
            "plane-count mismatch"
        )

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError(
            f"evaluate_encoded(shape={x.shape}, relations={relations is not None}) "
            "must not be reached on a plane-count mismatch"
        )


def test_puct_c_inplace_branch_rejects_a_narrow_slot() -> None:
    """``run_mcts_many_c``'s in-place branch takes its plane count from the
    evaluator and its fallback branch from the config — so a disagreement used
    to change the encoding with the BATCH SIZE. It must raise instead."""
    import chess

    from chess_anti_engine.mcts.puct import MCTSConfig
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c

    cfg = MCTSConfig(
        simulations=2, input_extra_features="v2_threats",
        input_history_encoding="lc0_root_legacy_meta",
    )
    with pytest.raises(ValueError, match="run_mcts_many_c root inplace"):
        run_mcts_many_c(
            None, [chess.Board()], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg,
            evaluator=cast("Any", _NarrowSlotEvaluator(input_plane_count("v1"))),
        )
