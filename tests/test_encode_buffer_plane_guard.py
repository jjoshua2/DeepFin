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

    n_slots = 2
    _max_batch = 64
    supports_input_bf16_bits = False
    supports_legal_bf16 = False

    def __init__(self, planes: int) -> None:
        self._buf = np.zeros((self._max_batch, planes, 8, 8), dtype=np.float32)

    # gumbel_c's in-place gate is supports_inplace_api(), which needs all three
    # slot methods present; puct_c/network_turn need only get_input_buffer.
    def get_input_buffer(self, n: int, slot: int = 0) -> np.ndarray:
        assert 0 <= slot < self.n_slots
        return self._buf[:n]

    def get_input_buffer_bf16_bits(self, n: int, slot: int = 0) -> np.ndarray:
        raise AssertionError(
            f"bf16 slot requested (n={n}, slot={slot}); this stub declares "
            "supports_input_bf16_bits=False and drives the fp32 branch"
        )

    def evaluate_inplace_async(
        self, n: int, slot: int = 0, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        raise AssertionError(
            f"evaluate_inplace_async(n={n}, slot={slot}, "
            f"relations={relations is not None}) reached after a 146-plane "
            "encode under a v2_threats config"
        )

    def evaluate_inplace(
        self, n: int, slot: int = 0, *, copy_out: bool = True,
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Deliberately permissive signature: with the guard removed the C
        # encoder writes 146 planes into this buffer WITHOUT COMPLAINT and
        # execution arrives here, which is the finding. The stub must fail on
        # that fact, not on an incidental TypeError.
        raise AssertionError(
            f"evaluate_inplace(n={n}, slot={slot}, copy_out={copy_out}, "
            f"relations={relations is not None}) reached after a 146-plane "
            "encode under a v2_threats config"
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


def test_gumbel_c_root_inplace_branch_rejects_a_narrow_slot() -> None:
    """The production search path (``mcts_type: gumbel``), same defect.

    Review F3: only ``puct_c`` was pinned; ``gumbel_c`` is what production runs.
    """
    import chess

    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    cfg = GumbelConfig(
        simulations=2, input_extra_features="v2_threats",
        input_history_encoding="lc0_root_legacy_meta",
    )
    with pytest.raises(ValueError, match="run_gumbel_many_c root inplace"):
        run_gumbel_root_many_c(
            None, [chess.Board()], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg,
            evaluator=cast("Any", _NarrowSlotEvaluator(input_plane_count("v1"))),
        )


def test_selfplay_root_inplace_branch_rejects_a_narrow_slot() -> None:
    """``_evaluate_root_batch`` — the path EVERY TRAINING ROW is written from.

    ``selfplay/network_turn.py:499/:624`` store the batch-encode output as the
    row's ``x``, so a plane layout that is silently wrong here is written to the
    shards and poisons training, search and every ruler at once. Drives the real
    function with a game config declaring v2_threats and an evaluator whose
    pinned slot is the v1 146.
    """
    import chess
    from types import SimpleNamespace

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import (
        batch_encode_146_lc0_root_legacy_meta,
    )
    from chess_anti_engine.selfplay.network_turn import _evaluate_root_batch

    state = SimpleNamespace(
        evaluator=_NarrowSlotEvaluator(input_plane_count("v1")),
        cboards=[CBoard.from_board(chess.Board())],
        has_c_ply=False,
        game=SimpleNamespace(
            input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats",
            record_relations=False,
        ),
        batch_enc_146_lc0_root_legacy_meta=batch_encode_146_lc0_root_legacy_meta,
        batch_enc_146_lc0_root_legacy_meta_bf16=None,
        batch_enc_146=None,
        batch_enc_146_bf16=None,
        batch_enc_146_lc0_root=None,
        batch_enc_146_lc0_root_bf16=None,
    )
    with pytest.raises(ValueError, match=r"selfplay root inplace \(fp32\)"):
        _evaluate_root_batch(cast("Any", state), [0])
