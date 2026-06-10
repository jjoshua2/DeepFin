"""Dynamic board-relation matrices + attention bias: parity, goldens,
orientation, zero-init equivalence, transport.

Relation indices (row = from-square, col = to-square, side-to-move oriented):
R0 attacks, R1 defends, R2 pinned_by, R3 shares_open_line, R4 pawn_tension.
See features.relation_matrices / _features_impl.h compute_relations.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.features import (
    RELATION_COUNT,
    relation_matrices,
    relation_matrices_c,
)
from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant

R_ATTACKS, R_DEFENDS, R_PINNED_BY, R_OPEN_LINE, R_TENSION = range(5)


def _random_board(rng: random.Random, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return b


def _rel(matrices: np.ndarray, k: int, from_sq: str, to_sq: str) -> int:
    """Relation value for a WHITE-to-move position (no orientation flip)."""
    return int(matrices[k, chess.parse_square(from_sq), chess.parse_square(to_sq)])


def test_relation_count_constant():
    assert RELATION_COUNT == 5
    r = relation_matrices(chess.Board())
    assert r.shape == (5, 64, 64)
    assert r.dtype == np.uint8


def test_c_python_parity_random_positions():
    rng = random.Random(4242)
    for i in range(220):
        b = _random_board(rng, rng.randint(0, 100))
        py = relation_matrices(b)
        cc = relation_matrices_c(b)
        np.testing.assert_array_equal(
            py, cc, err_msg=f"relation parity mismatch at position {i}: {b.fen()}",
        )


def test_cboard_and_batch_parity():
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
    from chess_anti_engine.mcts._mcts_tree import batch_compute_relations

    rng = random.Random(7)
    boards = [_random_board(rng, rng.randint(0, 60)) for _ in range(8)]
    cbs = [cboard_from_board_fast(b) for b in boards]
    out = np.empty((len(cbs), 5, 64, 64), dtype=np.uint8)
    batch_compute_relations(cbs, out)
    for i, b in enumerate(boards):
        np.testing.assert_array_equal(out[i], relation_matrices(b), err_msg=b.fen())
        np.testing.assert_array_equal(cbs[i].compute_relations(), relation_matrices(b))


def test_orientation_mirror_invariance():
    rng = random.Random(99)
    checked = 0
    for _ in range(60):
        b = _random_board(rng, rng.randint(0, 60))
        if b.is_game_over():
            continue
        m = b.mirror()
        for fn in (relation_matrices, relation_matrices_c):
            np.testing.assert_array_equal(
                fn(b), fn(m), err_msg=f"orientation mismatch ({fn.__name__}): {b.fen()}",
            )
        checked += 1
    assert checked >= 40


@pytest.mark.parametrize("impl", [relation_matrices, relation_matrices_c])
class TestGoldenRelations:
    def test_attacks_and_defends_startpos(self, impl):
        r = impl(chess.Board())
        # Knight b1 attacks a3/c3 (empty) and d2 (own pawn -> also defends).
        assert _rel(r, R_ATTACKS, "b1", "a3") == 1
        assert _rel(r, R_ATTACKS, "b1", "c3") == 1
        assert _rel(r, R_ATTACKS, "b1", "d2") == 1
        assert _rel(r, R_DEFENDS, "b1", "d2") == 1
        assert _rel(r, R_DEFENDS, "b1", "a3") == 0   # empty square: no defense
        # Pawn attack maps are capture squares only: e2 attacks d3/f3, not e3.
        assert _rel(r, R_ATTACKS, "e2", "d3") == 1
        assert _rel(r, R_ATTACKS, "e2", "e3") == 0

    def test_pin(self, impl):
        # Black bishop b4 pins the white pawn d2 against the king e1.
        b = chess.Board("4k3/8/8/8/1b6/8/3P4/4K3 w - - 0 1")
        r = impl(b)
        assert _rel(r, R_PINNED_BY, "d2", "b4") == 1
        assert int(r[R_PINNED_BY].sum()) == 1  # nothing else is pinned

    def test_open_line_battery(self, impl):
        # Rooks d2/d5 + black queen d8 on the open d-file: adjacent pairs
        # see each other, the blocked d2<->d8 pair does not. Ranks excluded.
        b = chess.Board("3qk3/8/8/3R4/8/8/3R4/4K3 w - - 0 1")
        r = impl(b)
        assert _rel(r, R_OPEN_LINE, "d2", "d5") == 1
        assert _rel(r, R_OPEN_LINE, "d5", "d2") == 1
        assert _rel(r, R_OPEN_LINE, "d5", "d8") == 1
        assert _rel(r, R_OPEN_LINE, "d2", "d8") == 0   # blocked by d5
        # e1 king and e8 king share the open e-file... files count:
        assert _rel(r, R_OPEN_LINE, "e1", "e8") == 1
        # d8 queen and e8 king are rank-adjacent: ranks are excluded.
        assert _rel(r, R_OPEN_LINE, "d8", "e8") == 0

    def test_pawn_tension(self, impl):
        b = chess.Board("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1")
        r = impl(b)
        assert _rel(r, R_TENSION, "d4", "e5") == 1
        assert _rel(r, R_TENSION, "e5", "d4") == 1
        assert int(r[R_TENSION].sum()) == 2


def _fill_param(mod: torch.nn.Module, name: str, value: float) -> None:
    param = getattr(mod, name)
    assert param is not None
    with torch.no_grad():
        param.fill_(value)


def _model(use_relations: bool, policy_relations: bool = False) -> torch.nn.Module:
    return build_model(ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
        input_extra_features="v2_threats",
        use_dynamic_relations=use_relations,
        policy_dynamic_relations=policy_relations,
    )).eval()


def _board_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    b = chess.Board()
    b.push_san("e4")
    b.push_san("c5")
    x = torch.from_numpy(
        encode_position(b, input_extra_features="v2_threats")
    ).unsqueeze(0)
    rel = torch.from_numpy(relation_matrices(b)).unsqueeze(0)
    return x, rel


def test_zero_init_bias_is_identity():
    torch.manual_seed(0)
    m = _model(use_relations=True, policy_relations=True)
    x, rel = _board_inputs()
    with torch.no_grad():
        o_off = m(x)
        o_on = m(x, relations=rel)
    for k in o_off:
        assert torch.equal(o_off[k], o_on[k]), f"head {k} changed under zero-init bias"


def test_warm_start_from_v2_threats_checkpoint_bit_identical():
    torch.manual_seed(1)
    base = _model(use_relations=False)
    dyn = _model(use_relations=True, policy_relations=True)
    load_state_dict_tolerant(dyn, base.state_dict(), label="test-dynrel-warmstart")
    x, rel = _board_inputs()
    with torch.no_grad():
        o_base = base(x)
        o_dyn = dyn(x, relations=rel)
    for k in o_base:
        assert torch.equal(o_base[k], o_dyn[k]), f"head {k} not bit-identical"


def test_nonzero_bias_changes_outputs_and_respects_absence():
    torch.manual_seed(2)
    m = _model(use_relations=True, policy_relations=True)
    _fill_param(m, "dynamic_relation_weight", 0.1)
    _fill_param(m, "policy_relation_weight", 0.1)
    x, rel = _board_inputs()
    with torch.no_grad():
        o_off = m(x)               # relations absent: bias term skipped
        o_off2 = m(x)
        o_on = m(x, relations=rel)
    assert torch.equal(o_off["policy_own"], o_off2["policy_own"])
    assert not torch.equal(o_off["policy_own"], o_on["policy_own"])
    assert not torch.equal(o_off["wdl"], o_on["wdl"])


def test_shard_roundtrip_preserves_relations():
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import (
        arrays_to_samples,
        samples_to_arrays,
        validate_array_declarations,
    )

    rng = np.random.default_rng(0)
    pol = np.zeros(4672, dtype=np.float32)
    pol[0] = 1.0
    rel = (rng.random((5, 64, 64)) < 0.05).astype(np.uint8)
    samples = [
        ReplaySample(
            x=rng.random((175, 8, 8)).astype(np.float32),
            policy_target=pol, wdl_target=1, relations=rel,
        ),
        ReplaySample(  # second sample WITHOUT relations: flag must stay 0
            x=rng.random((175, 8, 8)).astype(np.float32),
            policy_target=pol, wdl_target=0,
        ),
    ]
    arrs = samples_to_arrays(samples)
    validate_array_declarations(arrs)
    assert arrs["relations"].shape == (2, 5, 64, 64)
    assert arrs["has_relations"].tolist() == [1, 0]
    back = arrays_to_samples(arrs)
    assert back[0].relations is not None
    np.testing.assert_array_equal(back[0].relations, rel)
    assert back[1].relations is None


def test_collate_passes_relations_and_model_consumes_them():
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.dataset import collate

    b = chess.Board()
    b.push_san("d4")
    pol = np.zeros(4672, dtype=np.float32)
    pol[0] = 1.0
    x = encode_position(b, input_extra_features="v2_threats")
    samples = [
        ReplaySample(x=x, policy_target=pol, wdl_target=1, relations=relation_matrices(b)),
        ReplaySample(x=x, policy_target=pol, wdl_target=2),  # mixed batch
    ]
    batch = collate(samples, device="cpu")
    assert "relations" in batch
    assert batch["relations"].shape == (2, 5, 64, 64)
    assert batch["relations"].dtype == torch.uint8

    torch.manual_seed(4)
    m = _model(use_relations=True)
    _fill_param(m, "dynamic_relation_weight", 0.1)
    with torch.no_grad():
        out = m(batch["x"], relations=batch["relations"])
        out_no = m(batch["x"])
    # Row 0 has real relations -> biased; row 1 is zero matrices -> identical.
    assert not torch.equal(out["policy_own"][0], out_no["policy_own"][0])
    assert torch.equal(out["policy_own"][1], out_no["policy_own"][1])


def test_e2e_selfplay_gumbel_c_with_relations():
    """In-process gumbel-C selfplay (root + leaf evals) with a live bias."""
    from chess_anti_engine.selfplay.match import play_match_batch

    torch.manual_seed(5)
    m = build_model(ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
        input_extra_features="v2_threats", policy_encoding="lc0_1858",
        use_dynamic_relations=True,
    )).eval()
    _fill_param(m, "dynamic_relation_weight", 0.05)
    stats = play_match_batch(
        m, m, device="cpu", rng=np.random.default_rng(0), games=2, max_plies=8,
        a_plays_white=[True, False], mcts_type="gumbel",
        mcts_simulations=4, temperature=0.5,
    )
    assert stats.games == 2


def test_batch_process_ply_returns_relations():
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
    from chess_anti_engine.mcts._mcts_tree import batch_process_ply

    b = chess.Board()
    b.push_san("e4")
    cbs = [cboard_from_board_fast(b)]
    legal = cbs[0].legal_move_indices()
    action = int(legal[0])
    pol = np.zeros((1, 4672), dtype=np.float32)
    wdl = np.zeros((1, 3), dtype=np.float32)
    probs = np.zeros((1, 4672), dtype=np.float32)
    probs[0, action] = 1.0
    expected = relation_matrices(b)

    result = batch_process_ply(
        cbs, pol, wdl,
        np.array([action], dtype=np.int32), np.zeros(1, dtype=np.float64), probs,
        0, 0.0, 0.0, 1.0, 0.0,
        0, 63, 1,   # full history, v2_threats, with_relations
    )
    assert len(result) == 13
    rel = result[12]
    assert rel.shape == (1, 5, 64, 64)
    np.testing.assert_array_equal(rel[0], expected)
