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


# ---------------------------------------------------------------------------
# Review follow-ups: mirroring, optimizer continuation, fail-loud paths
# ---------------------------------------------------------------------------

def test_mirror_relations_matches_flipped_board():
    from chess_anti_engine.replay.augment import mirror_relations

    rng = random.Random(11)
    checked = 0
    for _ in range(30):
        b = chess.Board("r2q1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2P1PN2/PP3PPP/RN1Q1RK1 w - - 0 9")
        for _ in range(rng.randint(0, 30)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over():
            continue
        m = b.transform(chess.flip_horizontal)
        np.testing.assert_array_equal(
            mirror_relations(relation_matrices(b)), relation_matrices(m),
            err_msg=b.fen(),
        )
        checked += 1
    assert checked >= 20


def test_batch_mirror_keeps_relations_consistent_with_x():
    """maybe_mirror_batch_arrays must mirror relations alongside x — mirrored
    rows' relations equal the flipped board's relations."""
    from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays

    b = chess.Board()
    b.push_san("e4")
    b.push_san("c5")
    pol = np.zeros((2, 4672), np.float16)
    pol[:, 0] = 1.0
    rel = relation_matrices(b)
    arrs = {
        "x": np.stack([encode_position(b, input_extra_features="v2_threats")] * 2).astype(np.float16),
        "policy_target": pol,
        "wdl_target": np.zeros((2,), np.int8),
        "relations": np.stack([rel] * 2),
        "has_relations": np.ones((2,), np.uint8),
    }
    out = maybe_mirror_batch_arrays(
        arrs, rng=np.random.default_rng(0), prob=1.0,
        input_history_encoding="lc0_root_legacy_meta",
    )
    expected = relation_matrices(b.transform(chess.flip_horizontal))
    np.testing.assert_array_equal(out["relations"][0], expected)
    np.testing.assert_array_equal(out["relations"][1], expected)
    # x really was mirrored too (changed) — relations stayed in sync with it
    assert not np.array_equal(out["x"], arrs["x"])


def test_sample_mirror_preserves_relations():
    from chess_anti_engine.replay.augment import mirror_relations, mirror_sample
    from chess_anti_engine.replay.buffer import ReplaySample

    b = chess.Board()
    b.push_san("d4")
    pol = np.zeros(4672, np.float32)
    pol[0] = 1.0
    s = ReplaySample(
        x=encode_position(b, input_extra_features="v2_threats"),
        policy_target=pol, wdl_target=1, relations=relation_matrices(b),
    )
    m = mirror_sample(s, input_history_encoding="lc0_root_legacy_meta")
    assert m.relations is not None  # previously silently dropped
    assert s.relations is not None
    np.testing.assert_array_equal(m.relations, mirror_relations(s.relations))


def test_optimizer_state_splices_fresh_relation_params(tmp_path):
    """Warm-starting from a v2_threats trainer checkpoint must preserve the
    donor optimizer moments + scheduler instead of reinitializing them."""
    from chess_anti_engine.train.trainer import Trainer

    def cfg(dyn: bool) -> ModelConfig:
        return ModelConfig(
            embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
            input_extra_features="v2_threats",
            use_dynamic_relations=dyn, policy_dynamic_relations=dyn,
        )

    torch.manual_seed(0)
    base = build_model(cfg(False))
    t1 = Trainer(base, device="cpu", lr=1e-3, model_config=cfg(False))
    b = chess.Board()
    b.push_san("e4")
    x = torch.from_numpy(
        encode_position(b, input_extra_features="v2_threats")
    ).unsqueeze(0)
    out = base(x)
    torch.stack([v.float().sum() for v in out.values()]).sum().backward()
    t1.opt.step()
    t1.step = 7
    ckpt = tmp_path / "trainer.pt"
    t1.save(ckpt)

    dyn_model = build_model(cfg(True))
    t2 = Trainer(dyn_model, device="cpu", lr=1e-3, model_config=cfg(True))
    t2.load(ckpt)
    assert t2.step == 7
    named = dict(dyn_model.named_parameters())
    donor_state = t2.opt.state.get(named["embed.weight"])
    assert donor_state
    assert float(donor_state["exp_avg"].abs().sum()) > 0
    assert not t2.opt.state.get(named["dynamic_relation_weight"])  # fresh slot
    out2 = dyn_model(x)
    torch.stack([v.float().sum() for v in out2.values()]).sum().backward()
    t2.opt.step()  # must not shape-crash


def test_puct_paths_raise_with_relations():
    from chess_anti_engine.mcts.puct import MCTSConfig, run_mcts_many
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c

    m = _model(use_relations=True)
    cfg = MCTSConfig(simulations=1, compute_relations=True)
    for fn in (run_mcts_many, run_mcts_many_c):
        with pytest.raises(NotImplementedError, match="PUCT"):
            fn(m, [chess.Board()], device="cpu", rng=np.random.default_rng(0), cfg=cfg)


def test_python_gumbel_fallback_transports_relations():
    """The no-C gumbel fallback must pass relations to the evaluator."""
    from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many

    torch.manual_seed(6)
    m = build_model(ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
        input_extra_features="v2_threats", use_dynamic_relations=True,
    )).eval()
    _fill_param(m, "dynamic_relation_weight", 0.05)
    b = chess.Board()
    cfg = GumbelConfig(
        simulations=4, temperature=0.5, add_noise=False,
        input_extra_features="v2_threats", compute_relations=True,
    )
    probs, actions, _values, _masks = run_gumbel_root_many(
        m, [b], device="cpu", rng=np.random.default_rng(0), cfg=cfg,
    )[:4]
    del _values, _masks
    assert len(actions) == 1

    cfg_off = GumbelConfig(
        simulations=4, temperature=0.5, add_noise=False,
        input_extra_features="v2_threats", compute_relations=False,
    )
    probs_off, *_ = run_gumbel_root_many(
        m, [b], device="cpu", rng=np.random.default_rng(0), cfg=cfg_off,
    )
    # nonzero bias + relations transported => different root policy
    assert not np.allclose(probs[0], probs_off[0])


def test_dynamic_relation_count_validated():
    with pytest.raises(ValueError, match="dynamic_relation_count"):
        build_model(ModelConfig(
            embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
            use_dynamic_relations=True, dynamic_relation_count=7,
        ))


def test_transport_check_rejects_puct_and_aot():
    from chess_anti_engine.tune.distributed_runtime import check_dynamic_relations_transport

    base = {"use_dynamic_relations": True, "mcts": "gumbel"}
    check_dynamic_relations_transport(dict(base))  # ok
    with pytest.raises(ValueError, match="mcts"):
        check_dynamic_relations_transport({**base, "mcts": "puct"})
    with pytest.raises(ValueError, match="aot"):
        check_dynamic_relations_transport({**base, "distributed_worker_aot_dir": "/x"})
    with pytest.raises(ValueError, match="broker"):
        check_dynamic_relations_transport(
            {**base, "distributed_inference_broker_enabled": True})
    # record_relations alone (no model flag) is guarded too
    with pytest.raises(ValueError):
        check_dynamic_relations_transport(
            {"record_relations": True, "mcts": "gumbel",
             "distributed_worker_threaded": True})


# ---------------------------------------------------------------------------
# UCI search-path transport (PucvChunker / WalkerPool / MultiGpuPucvPool /
# SearchWorker root eval)
# ---------------------------------------------------------------------------


class _RelRecordingInplaceEvaluator:
    """Inplace-async fake that records the relations passed per submit."""

    n_slots = 2

    def __init__(self, max_batch: int = 64) -> None:
        self.relations_seen: list[np.ndarray | None] = []
        self._bufs = [
            np.zeros((max_batch, 146, 8, 8), dtype=np.float32),
            np.zeros((max_batch, 146, 8, 8), dtype=np.float32),
        ]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:bsz]

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        del slot
        self.relations_seen.append(None if relations is None else relations.copy())
        pol = np.zeros((bsz, 4672), dtype=np.float32)
        wdl = np.zeros((bsz, 3), dtype=np.float32)
        return pol, wdl, None


class _RelRecordingEncodedEvaluator:
    """evaluate_encoded fake that records the relations kwarg."""

    def __init__(self) -> None:
        self.relations_seen: list[np.ndarray | None] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.relations_seen.append(None if relations is None else relations.copy())
        n = int(np.asarray(x).shape[0])
        return (
            np.zeros((n, 4672), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
        )


def _seeded_tree():
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.reserve(1024, 8192)
    cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)
    return tree, rid, cb


def test_pucv_chunker_transports_relations():
    from chess_anti_engine.mcts.puct_vl import PucvChunker

    ev = _RelRecordingInplaceEvaluator()
    tree, rid, cb = _seeded_tree()
    chunker = PucvChunker(ev, gather=8, compute_relations=True)
    n = chunker.run(tree, rid, cb, 16)
    assert n == 16
    assert ev.relations_seen, "no NN submits recorded"
    root_rel = cb.compute_relations()
    saw_root_row = False
    for rel in ev.relations_seen:
        assert rel is not None, "submit without relations on relations-enabled chunker"
        assert rel.shape[1:] == (RELATION_COUNT, 64, 64)
        assert rel.dtype == np.uint8
        # Depth-1 leaves of the startpos root: every leaf is one ply in, and
        # the first batch's first descent lands on a child of the root, so at
        # least one row must be a real (nonzero) relations matrix.
        if any(np.asarray(r).any() for r in rel):
            saw_root_row = True
    assert saw_root_row
    del root_rel


def test_walker_pool_transports_relations():
    import threading

    from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

    ev = _RelRecordingEncodedEvaluator()
    tree, rid, cb = _seeded_tree()
    pool = WalkerPool(
        WalkerPoolConfig(
            n_walkers=2, c_puct=1.4, fpu_at_root=0.0, fpu_reduction=0.2,
            gather=4, compute_relations=True,
        ),
        ev,
    )
    try:
        pool.run(
            tree=tree, root_id=rid, root_cboard=cb,
            target_sims=16, stop_event=threading.Event(),
        )
    finally:
        pool.close()
    assert ev.relations_seen
    for rel in ev.relations_seen:
        assert rel is not None
        assert rel.shape[1:] == (RELATION_COUNT, 64, 64)
        assert rel.dtype == np.uint8
        assert np.asarray(rel).any()  # depth>=1 startpos leaves all have relations


def test_multi_gpu_pucv_pool_transports_relations():
    import threading

    from chess_anti_engine.uci.multi_gpu_pucv_pool import (
        MultiGpuPucvConfig,
        MultiGpuPucvPool,
    )

    evs = [_RelRecordingInplaceEvaluator(), _RelRecordingInplaceEvaluator()]
    tree, rid, cb = _seeded_tree()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, compute_relations=True),
        evaluators=evs,
    )
    try:
        pool.run(
            tree=tree, root_id=rid, root_cboard=cb,
            target_sims=32, stop_event=threading.Event(),
        )
    finally:
        pool.close()
    seen = [rel for ev in evs for rel in ev.relations_seen]
    assert seen
    for rel in seen:
        assert rel is not None
        assert rel.shape[1:] == (RELATION_COUNT, 64, 64)
        assert rel.dtype == np.uint8


def test_search_worker_root_eval_passes_relations():
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.uci.search import SearchWorker

    ev = _RelRecordingEncodedEvaluator()
    worker = SearchWorker(
        ev,  # type: ignore[arg-type] — duck-typed fake
        device="cpu",
        gumbel_cfg=GumbelConfig(
            simulations=4, add_noise=False, temperature=0.0,
            compute_relations=True,
        ),
    )
    board = chess.Board()
    worker._ensure_root_eval_cached(board, None)
    assert len(ev.relations_seen) == 1
    rel = ev.relations_seen[0]
    assert rel is not None
    assert rel.shape == (1, RELATION_COUNT, 64, 64)
    np.testing.assert_array_equal(
        rel[0], CBoard.from_board(board).compute_relations(),
    )
