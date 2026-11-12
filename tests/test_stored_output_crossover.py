import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from scripts.stored_output_crossover import OutputCrossover, search_one
from tests.test_stored_wdl_transition import encoded


def models():
    result = []
    for seed in (21, 23):
        torch.manual_seed(seed)
        model = build_model(
            ModelConfig(
                kind="transformer",
                embed_dim=16,
                num_layers=1,
                num_heads=2,
                use_smolgen=False,
            )
        ).eval()
        model.requires_grad_(False)
        result.append(model)
    return result


def inputs():
    b = chess.Board()
    b.push_uci("g1f3")
    return np.stack([encoded(chess.Board()), encoded(b)])


def test_actual_graph_crosses_outputs_not_weights_with_repeat_batch_identity():
    a, b = models()
    x = inputs()
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (a, b)]
    ap, av = LocalModelEvaluator(a, device="cpu", use_amp=False).evaluate_encoded(x)
    bp, bv = LocalModelEvaluator(b, device="cpu", use_amp=False).evaluate_encoded(x)
    for pm, vm, ep, ev in (
        (a, a, ap, av),
        (b, a, bp, av),
        (a, b, ap, bv),
        (b, b, bp, bv),
    ):
        mixed = OutputCrossover(pm, vm, batch_size=2)
        p, v = mixed.evaluate_encoded(x)
        assert np.array_equal(p, ep)
        assert np.array_equal(v, ev)
        p2, v2 = mixed.evaluate_encoded(x)
        assert np.array_equal(p, p2)
        assert np.array_equal(v, v2)
        p1, v1 = OutputCrossover(pm, vm, batch_size=1).evaluate_encoded(x)
        assert np.allclose(p, p1, atol=1e-6, rtol=1e-5)
        assert np.allclose(v, v1, atol=1e-6, rtol=1e-5)
    for m, saved in zip((a, b), before, strict=True):
        assert all(torch.equal(v, saved[k]) for k, v in m.state_dict().items())


def test_real_python_search_uses_same_mixed_evaluator_for_root_and_leaves():
    a, b = models()
    evaluator = OutputCrossover(a, b)
    cfg = GumbelConfig(
        simulations=8,
        topk=2,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    result = search_one(chess.Board(), evaluator, cfg=cfg, seed=37)
    assert len(evaluator.calls) > 1
    assert evaluator.evaluated_rows > 1
    root_hash = evaluator.calls[0]["input_sha256"]
    assert any(c["input_sha256"] != root_hash for c in evaluator.calls[1:])
    repeat = search_one(chess.Board(), OutputCrossover(a, b), cfg=cfg, seed=37)
    for left, right in zip(result, repeat, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))


def test_value_pov_is_passed_unchanged_not_root_flipped():
    a, b = models()
    x = inputs()
    _, v = OutputCrossover(a, b).evaluate_encoded(x)
    _, expected = LocalModelEvaluator(b, device="cpu", use_amp=False).evaluate_encoded(
        x
    )
    assert np.array_equal(v, expected)
    assert not np.allclose(v, expected[:, ::-1])


def test_invalid_paths_and_finite_budget_fail_closed():
    a, b = models()
    x = inputs()
    with pytest.raises(ValueError, match="batch_size"):
        OutputCrossover(a, b, batch_size=True)
    e = OutputCrossover(a, b, max_rows=1)
    with pytest.raises(ValueError, match="budget"):
        e.evaluate_encoded(x)
    with pytest.raises(ValueError, match="float32"):
        OutputCrossover(a, b).evaluate_encoded(x.astype(np.float16))
    with pytest.raises(ValueError, match="relations"):
        OutputCrossover(a, b).evaluate_encoded(x, relations=np.zeros(1))
    bad = x.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        OutputCrossover(a, b).evaluate_encoded(bad)
    with pytest.raises(ValueError, match="unsupported"):
        search_one(chess.Board(), OutputCrossover(a, b), cfg=GumbelConfig(), seed=0)
    a.train()
    with pytest.raises(ValueError, match="eval"):
        OutputCrossover(a, b)
