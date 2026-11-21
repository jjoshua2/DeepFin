from typing import cast
from chess_anti_engine.model.transformer import ChessNet
import chess
import numpy as np
import pytest
import torch
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves.encode import (
    legal_move_mask_for_encoding,
    COMPACT_TO_FULL_POLICY,
)
from scripts.stored_policy_probe import policy_probe
from scripts.stored_wdl_compare import rows_digest
from tests.test_stored_wdl_transition import encoded


def fixture():
    torch.manual_seed(7)
    model = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            use_smolgen=False,
        )
    ).eval()
    boards = [
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ]
    x = np.stack([encoded(b) for b in boards])
    ids = ["a", "b", "c"]
    groups = ["g"] * 3
    return model, x, ids, groups, boards


def test_repeat_batch_dense_legal_api_and_masks():
    model, x, ids, groups, boards = fixture()
    model = cast(ChessNet, model)
    digest = rows_digest(x, ids, groups)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = policy_probe(model, x, ids, groups, expected_rows_sha256=digest, batch_size=3)
    b = policy_probe(model, x, ids, groups, expected_rows_sha256=digest, batch_size=1)
    repeat = policy_probe(
        model, x, ids, groups, expected_rows_sha256=digest, batch_size=3
    )
    assert np.allclose(a["probabilities"], b["probabilities"], atol=1e-6, rtol=1e-5)
    assert np.array_equal(a["logits"], repeat["logits"])
    assert np.all(a["probabilities"][~a["legal_mask"]] == 0)
    assert np.allclose(a["probabilities"].sum(1), 1)
    for i, board in enumerate(boards):
        assert np.array_equal(a["legal_mask"][i], legal_move_mask_for_encoding(board))
        legal = np.flatnonzero(a["legal_mask"][i])
        with torch.inference_mode():
            sparse = model.forward_legal_policy(
                torch.from_numpy(x[i : i + 1]),
                torch.tensor(COMPACT_TO_FULL_POLICY[legal]),
                torch.tensor([len(legal)]),
            )["policy_own"].numpy()
        assert np.allclose(sparse, a["logits"][i, legal], atol=1e-6, rtol=1e-5)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("batch", [True, 0, 65])
def test_invalid_batch(batch):
    m, x, i, g, _ = fixture()
    with pytest.raises(ValueError, match="batch"):
        policy_probe(
            m, x, i, g, expected_rows_sha256=rows_digest(x, i, g), batch_size=batch
        )


def test_identity_fails_before_forward():
    m, x, i, g, _ = fixture()
    with pytest.raises(ValueError, match="identity"):
        policy_probe(m, x, i, g, expected_rows_sha256="wrong")


def test_saturated_metadata_not_repaired():
    m, x, i, g, _ = fixture()
    x[0, 109] = 1
    with pytest.raises(ValueError, match="saturated"):
        policy_probe(m, x, i, g, expected_rows_sha256=rows_digest(x, i, g))
