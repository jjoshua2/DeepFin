"""Policy-only relation checkpoints must request the same input as trunk bias."""
from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.features import relation_matrices
from chess_anti_engine.inference import LocalModelEvaluator, _policy_output_full
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


@pytest.fixture
def policy_only_checkpoint(tmp_path):
    torch.manual_seed(17)
    cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
        input_extra_features="v2_threats", use_dynamic_relations=False,
        policy_dynamic_relations=True,
    )
    model = build_model(cfg).eval()
    relation_weight = getattr(model, "policy_relation_weight")
    assert isinstance(relation_weight, torch.nn.Parameter)
    with torch.no_grad():
        relation_weight.fill_(0.75)
    checkpoint = tmp_path / "trainer.pt"
    torch.save({
        "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)},
        "model": model.state_dict(),
    }, checkpoint)
    return load_model_from_checkpoint(checkpoint, device="cpu")


class _RecordingEvaluator(LocalModelEvaluator):
    def __init__(self, model):
        super().__init__(model, device="cpu", use_amp=False)
        self.calls = []

    def evaluate_encoded(self, x, relations=None):
        result = super().evaluate_encoded(x, relations)
        self.calls.append((None if relations is None else relations.copy(), result[0].copy()))
        return result

    def evaluate_encoded_async(self, x, relations=None):
        policy, wdl = self.evaluate_encoded(x, relations)
        return torch.from_numpy(policy), torch.from_numpy(wdl), None


def test_policy_only_checkpoint_match_uses_learned_bias(policy_only_checkpoint):
    from chess_anti_engine.selfplay.match import pick_moves_for_boards

    model = policy_only_checkpoint
    assert model.dynamic_relation_weight is None
    board = chess.Board()
    board.push_san("e4")
    board.push_san("c5")
    x = torch.from_numpy(encode_position(board, input_extra_features="v2_threats"))[None]
    relations = torch.from_numpy(relation_matrices(board))[None]
    with torch.no_grad():
        expected = _policy_output_full(model(x, relations=relations)).numpy()
        omitted = _policy_output_full(model(x)).numpy()
    assert not np.allclose(expected, omitted), "nonzero learned policy bias must be observable"
    evaluator = _RecordingEvaluator(model)
    actions = pick_moves_for_boards(
        model, [board], device="cpu", rng=np.random.default_rng(0),
        mcts_type="gumbel", mcts_simulations=4, temperature=0.0,
        c_puct=1.5, gumbel_add_noise=False, evaluator=evaluator,
    )
    assert len(actions) == 1
    root_relations, root_policy = evaluator.calls[0]
    assert root_relations is not None
    np.testing.assert_array_equal(root_relations, relations.numpy())
    np.testing.assert_allclose(root_policy, expected, rtol=1e-5, atol=1e-5)
    assert all(rel is not None for rel, _ in evaluator.calls)


def test_policy_only_checkpoint_uci_warmup_requests_relations(policy_only_checkpoint, monkeypatch):
    from chess_anti_engine.uci import __main__ as uci

    warmups = []
    monkeypatch.setattr(uci, "DirectGPUEvaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(uci, "ThreadSafeGPUDispatcher", lambda evaluator: evaluator)
    monkeypatch.setattr(uci, "_warmup_evaluator", lambda evaluator, **kwargs: warmups.append(kwargs))
    factory = uci._make_evaluator_factory(
        [policy_only_checkpoint], ["cpu"], coalesce=False, n_walkers=1, walker_gather=1,
    )
    factory(4)
    assert warmups[0]["compute_relations"] is True


@pytest.mark.parametrize(("trunk", "policy"), [(False, False), (True, False), (False, True), (True, True)])
def test_relation_manifest_requests_either_bias(tmp_path, trunk, policy):
    from chess_anti_engine.tune.distributed_runtime import (
        _publish_distributed_trial_state,
        build_recommended_worker,
    )

    cfg = ModelConfig(use_dynamic_relations=trunk, policy_dynamic_relations=policy)
    trainer = SimpleNamespace(export_swa=lambda path: path.write_bytes(b"test model"))
    _publish_distributed_trial_state(
        trainer=trainer, config={}, model_cfg=cfg, server_root=tmp_path,
        trial_id="test", training_iteration=1, trainer_step=1,
        sf_nodes=1, mcts_simulations=1,
    )
    manifest = json.loads((tmp_path / "trials" / "test" / "publish" / "manifest.json").read_text())
    expected = trunk or policy
    assert manifest["encoding"]["use_dynamic_relations"] is expected
    assert manifest["recommended_worker"]["use_dynamic_relations"] is expected
    assert manifest["recommended_worker"]["record_relations"] is expected
    assert manifest["model_config"]["use_dynamic_relations"] is trunk
    assert manifest["model_config"]["policy_dynamic_relations"] is policy
    if expected:
        with pytest.raises(ValueError, match="record_relations must be enabled"):
            build_recommended_worker(
                config={"record_relations": False}, model_cfg=cfg,
                sf_nodes=1, mcts_simulations=1,
            )
    else:
        reco = build_recommended_worker(
            config={"record_relations": False}, model_cfg=cfg,
            sf_nodes=1, mcts_simulations=1,
        )
        assert reco["use_dynamic_relations"] is False
        assert reco["record_relations"] is False


@pytest.mark.parametrize(("trunk", "policy"), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("recording", [
    pytest.param({}, id="unset"),
    pytest.param({"record_relations": True}, id="true"),
    pytest.param({"record_relations": False}, id="false"),
    pytest.param({"record_relations": None}, id="null"),
    pytest.param({"record_relations": 0}, id="zero"),
])
def test_relation_bias_requires_worker_recording(trunk, policy, recording):
    from chess_anti_engine.tune.distributed_runtime import (
        build_recommended_worker,
        check_dynamic_relations_transport,
    )

    cfg = ModelConfig(use_dynamic_relations=trunk, policy_dynamic_relations=policy)
    run_config = {"use_dynamic_relations": trunk, "policy_dynamic_relations": policy,
                  "mcts": "gumbel", **recording}
    # The publication guard must honor the actual architecture, even when the
    # raw config's relation flags do not match the model being published.
    publish_config = {"use_dynamic_relations": False, "policy_dynamic_relations": False,
                      **recording}
    disabled = "record_relations" in recording and not bool(recording["record_relations"])
    if (trunk or policy) and disabled:
        with pytest.raises(ValueError, match="record_relations must be enabled"):
            check_dynamic_relations_transport(run_config)
        with pytest.raises(ValueError, match="record_relations must be enabled"):
            build_recommended_worker(
                config=publish_config, model_cfg=cfg, sf_nodes=1, mcts_simulations=1,
            )
    else:
        check_dynamic_relations_transport(run_config)
        reco = build_recommended_worker(
            config=publish_config, model_cfg=cfg, sf_nodes=1, mcts_simulations=1,
        )
        assert reco["use_dynamic_relations"] is (trunk or policy)
        assert reco["record_relations"] is bool(recording.get("record_relations", trunk or policy))


def test_policy_only_era_probe_requires_relation_rows(monkeypatch):
    from chess_anti_engine.tune import trainable_init
    from chess_anti_engine.tune.trial_config import TrialConfig

    seen = []
    monkeypatch.setattr(trainable_init, "load_probe_set", lambda *args, **kwargs: seen.append(kwargs))
    tc = TrialConfig(
        input_extra_features="v2_threats", policy_encoding="lc0_1858",
        era_probe_path="/unused/era.npz", era_probe_inwindow_path="/unused/inwindow.npz",
        era_probe_rows=1, use_dynamic_relations=False, policy_dynamic_relations=True,
    )
    assert trainable_init._init_era_probes(tc=tc) == {}
    assert len(seen) == 2
    assert all(probe["expects_relations"] for probe in seen)


@pytest.mark.parametrize("unsupported", [
    {"mcts": "puct"},
    {"distributed_worker_aot_dir": "/unused"},
    {"distributed_inference_broker_enabled": True},
    {"distributed_inference_shared_broker": True},
    {"distributed_worker_threaded": True},
])
def test_policy_only_relations_reject_unsupported_worker_transport(unsupported):
    from chess_anti_engine.tune.distributed_runtime import check_dynamic_relations_transport

    base = {"use_dynamic_relations": False, "policy_dynamic_relations": True, "mcts": "gumbel"}
    check_dynamic_relations_transport(base)
    with pytest.raises(ValueError, match="dynamic relations"):
        check_dynamic_relations_transport({**base, **unsupported})
