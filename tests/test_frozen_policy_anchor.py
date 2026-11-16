"""Synthetic actual-loss/Trainer controls; no real bank/model/study evidence."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model, infer_input_planes
from chess_anti_engine.policy_anchor_spec import DIRECTION, MASK, SCHEMA, validate_spec
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import samples_to_arrays
from chess_anti_engine.train import Trainer
from chess_anti_engine.train import frozen_policy_anchor as anchor
from scripts import core_finite_learner as core
from scripts.bootstrap_checkpoint_resume import require_equal


def disabled() -> dict[str, Any]:
    return {"schema": SCHEMA, "direction": DIRECTION, "mask": MASK, "coefficient": 0, "reference": None}


def reference(tmp_path: Path, cfg: ModelConfig, *, coefficient: float = 0.7) -> dict[str, Any]:
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    model = build_model(cfg)
    # Deliberately uniform reference vs nonuniform real student, not an identical
    # fresh pair whose first KL gradient would be zero even if unattached.
    state = model.state_dict()
    for tensor in state.values():
        tensor.zero_()
    path = tmp_path / "reference.pt"
    torch.save({"arch": arch, "model": state, "step": 17}, path)
    path.chmod(0o444)
    return validate_spec({"schema": SCHEMA, "direction": DIRECTION, "mask": MASK, "coefficient": coefficient,
            "reference": {"checkpoint": {"path": str(path), "sha256": core.file_sha(path)}, "architecture": arch,
                          "checkpoint_step": 17, "historical_source_revision": None}}, arch)


def trainer(cfg: ModelConfig, state: dict[str, Any]) -> Trainer:
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    return Trainer(model, model_config=cfg, device="cpu", use_amp=False, lr=1e-4,
        mirror_prob=1.0, **{**core.CORE_KWARGS, "feature_dropout_p": 0.5})


class FixedRows:
    """Deterministic batch fixture; actual Trainer mirror/collate/dropout remain."""

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(22)
        x = np.arange(infer_input_planes(cfg.input_extra_features) * 64, dtype=np.float32).reshape(-1, 8, 8) / 10000
        policy = np.zeros(1858, dtype=np.float32)
        policy[0] = 1
        legal = np.zeros(1858, dtype=np.uint8)
        legal[:3] = 1
        self.arrays = samples_to_arrays([ReplaySample(x=x, policy_target=policy, wdl_target=1,
            has_policy=True, is_network_turn=network, is_selfplay=True, legal_mask=legal,
            input_history_encoding=cfg.input_history_encoding, history_rep_fix=True) for network in (True, False)])

    def sample_batch_arrays(self, count: int) -> dict[str, np.ndarray]:
        assert count == 2
        return {key: value.copy() for key, value in self.arrays.items()}


def numerical_state(value: Trainer) -> dict[str, Any]:
    return {"model": copy.deepcopy(value.model.state_dict()), "opt": copy.deepcopy(value.opt.state_dict()),
            "scheduler": copy.deepcopy(value._scheduler.state_dict()), "zclip": copy.deepcopy(value.zclip_state_dict()), "step": value.step}


def test_analytic_forward_kl_gradient_masks_and_detached_reference():
    student = torch.tensor([[0.2, 1.1, -0.4]], requires_grad=True)
    reference_logits = torch.tensor([[1.4, -0.2, 9.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    loss = 0.7 * anchor.forward_kl(student, reference_logits, legal).mean()
    loss.backward()
    p = torch.softmax(student.detach().masked_fill(~legal, -torch.inf), -1)
    q = torch.softmax(reference_logits.detach().masked_fill(~legal, -torch.inf), -1)
    torch.testing.assert_close(student.grad, 0.7 * (p - q))
    assert student.grad is not None
    assert student.grad[0, 2] == 0
    assert reference_logits.grad is None
    assert float(loss) > 0


@pytest.mark.parametrize("fault", ["missing", "unflagged", "empty", "width", "nan", "nonbinary", "eligibility"])
def test_active_mask_refusals_are_not_unrestricted_fallbacks(fault: str):
    batch = {"is_network_turn": torch.ones(2), "has_policy": torch.ones(2),
             "has_legal_mask": torch.ones(2), "legal_mask": torch.ones(2, 3)}
    if fault == "missing":
        del batch["legal_mask"]
    elif fault == "unflagged":
        batch["has_legal_mask"][0] = 0
    elif fault == "empty":
        batch["legal_mask"][0] = 0
    elif fault == "width":
        batch["legal_mask"] = torch.ones(2, 4)
    elif fault == "nan":
        batch["legal_mask"][0, 0] = torch.nan
    elif fault == "nonbinary":
        batch["legal_mask"][0, 0] = 0.5
    else:
        del batch["is_network_turn"]
    with pytest.raises(ValueError, match=r"mask|eligibility"):
        anchor.eligible_legal(torch.zeros(2, 3), batch)


@pytest.mark.parametrize("fault", ["direction", "weight", "zero_ref", "architecture", "historical"])
def test_explicit_identity_is_strict_without_historical_invention(tmp_path: Path, fault: str):
    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    arch = spec["reference"]["architecture"]
    if fault == "direction":
        spec["direction"] = "student_to_reference"
    elif fault == "weight":
        spec["coefficient"] = True
    elif fault == "zero_ref":
        spec["coefficient"] = 0
    elif fault == "architecture":
        arch = {**arch, "kind": "transformer"}
    else:
        spec["reference"]["historical_source_revision"] = "made-up-provenance"
    with pytest.raises(ValueError, match=r"reference|anchor|provenance"):
        validate_spec(spec, arch)


def test_actual_trainer_zero_is_exact_with_no_reference_work(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    initial = copy.deepcopy(build_model(cfg).state_dict())
    base, zero = trainer(cfg, initial), trainer(cfg, initial)
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("zero anchor touched reference setup/forward/custody/RNG")
    monkeypatch.setattr(anchor, "FrozenPolicyAnchor", forbidden)
    monkeypatch.setattr(anchor, "_checkpoint", forbidden)
    zero.configure_frozen_policy_anchor(disabled())
    assert zero.frozen_policy_anchor_state() == disabled()
    with anchor.preserve_rng():
        start = anchor._rng()
        a = base.train_steps(cast(Any, FixedRows(cfg)), batch_size=2, steps=2)
        final = anchor._rng()
        expected = numerical_state(base)
    assert anchor._same_rng(start, anchor._rng())
    b = zero.train_steps(cast(Any, FixedRows(cfg)), batch_size=2, steps=2)
    require_equal(expected, numerical_state(zero), "zero actual state")
    assert anchor._same_rng(final, anchor._rng())
    for field in ("loss", "policy_loss", "wdl_loss", "blended_wdl_loss", "wdl_onehot_loss", "grad_nonfinite_skip_rate", "train_steps_done", "train_samples_seen"):
        assert getattr(a, field) == getattr(b, field)
    assert zero.frozen_policy_anchor_diagnostics() is None
    base.writer.close()
    zero.writer.close()


def test_actual_backward_delta_observes_mirror_dropout_and_reference_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    initial = copy.deepcopy(build_model(cfg).state_dict())
    base, positive = trainer(cfg, initial), trainer(cfg, initial)
    before_setup = anchor._rng()
    positive.configure_frozen_policy_anchor(spec)
    assert anchor._same_rng(before_setup, anchor._rng())
    frozen = positive._frozen_policy_anchor
    assert frozen is not None
    reference_before = copy.deepcopy(frozen.model.state_dict())
    optimized = {id(p) for group in positive.opt.param_groups for p in group["params"]}
    assert not any(id(p) in optimized for p in frozen.model.parameters())
    observations: list[dict[str, Any]] = []
    teacher_inputs = []
    handle = frozen.model.register_forward_pre_hook(lambda _m, args: teacher_inputs.append((args[0].clone(), torch.is_grad_enabled())))
    original = Trainer._host_batch_to_tensors
    def tensors(self: Trainer, batch: Any) -> dict[str, torch.Tensor]:
        result = original(self, batch)
        observations.append({"trainer": self, "batch": result})
        return result
    monkeypatch.setattr(Trainer, "_host_batch_to_tensors", tensors)
    logits: list[torch.Tensor] = []
    inputs = []
    def capture(_m: Any, args: Any, out: dict[str, torch.Tensor]) -> None:
        out["policy"].retain_grad()
        logits.append(out["policy"])
        inputs.append(args[0].clone())
    hooks = [base.model.register_forward_hook(capture), positive.model.register_forward_hook(capture)]
    with anchor.preserve_rng():
        base.train_steps(cast(Any, FixedRows(cfg)), batch_size=2, steps=1)
    positive.train_steps(cast(Any, FixedRows(cfg)), batch_size=2, steps=1)
    for hook in hooks:
        hook.remove()
    handle.remove()
    assert len(logits) == 2
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
    batch = observations[1]["batch"]
    eligible, legal = anchor.eligible_legal(logits[1], batch)
    assert eligible.tolist() == [True, False]
    assert len(teacher_inputs) == 1
    torch.testing.assert_close(teacher_inputs[0][0], inputs[1][eligible], rtol=0, atol=0)
    assert teacher_inputs[0][1] is False
    # Actual mirrored output differs from the source-side legal mask. No second
    # mirroring/augmentation is used to supply the reference or the oracle.
    assert not torch.equal(batch["legal_mask"].cpu(), torch.from_numpy(FixedRows(cfg).arrays["legal_mask"]))
    p = torch.softmax(logits[1].detach()[eligible].float().masked_fill(~legal, -torch.inf), -1)
    q = legal.float() / legal.sum(-1, keepdim=True)
    expected = torch.zeros_like(logits[1])
    expected[eligible] = spec["coefficient"] * (p - q) / int(eligible.sum())
    assert float(expected.abs().max()) > 0
    assert logits[0].grad is not None
    assert logits[1].grad is not None
    torch.testing.assert_close(logits[1].grad - logits[0].grad, expected, rtol=2e-5, atol=2e-7)
    assert bool((logits[1].grad - logits[0].grad)[~eligible].eq(0).all())
    require_equal(reference_before, frozen.model.state_dict(), "reference immutable parameters/buffers")
    assert not frozen.model.training
    assert all(p.grad is None and not p.requires_grad for p in frozen.model.parameters())
    diagnostics = positive.frozen_policy_anchor_diagnostics()
    assert diagnostics is not None
    assert diagnostics["eligible_rows"] == diagnostics["reference_forwards"] == 1
    assert diagnostics["mean_kl"] > 0
    assert diagnostics["forward_host_seconds"] >= 0
    base.writer.close()
    positive.writer.close()


@pytest.mark.parametrize("fault", ["hash", "step", "writable", "changed", "state"])
def test_reference_actual_loader_and_final_custody_refuse_mismatch(tmp_path: Path, fault: str):
    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    value = trainer(cfg, copy.deepcopy(build_model(cfg).state_dict()))
    if fault == "hash":
        spec["reference"]["checkpoint"]["sha256"] = "0" * 64
    elif fault == "step":
        spec["reference"]["checkpoint_step"] = 18
    elif fault == "writable":
        Path(spec["reference"]["checkpoint"]["path"]).chmod(0o644)
    if fault in {"hash", "step", "writable"}:
        with pytest.raises(ValueError, match=r"reference|checkpoint"):
            value.configure_frozen_policy_anchor(spec)
    else:
        value.configure_frozen_policy_anchor(spec)
        if fault == "changed":
            path = Path(spec["reference"]["checkpoint"]["path"])
            path.chmod(0o644)
        else:
            assert value._frozen_policy_anchor is not None
            with torch.no_grad():
                next(value._frozen_policy_anchor.model.parameters()).add_(1)
        with pytest.raises(ValueError, match=r"reference|checkpoint"):
            value.frozen_policy_anchor_state()
    value.writer.close()


def test_empty_eligible_batch_skips_reference_forward_and_addition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    value = trainer(cfg, copy.deepcopy(build_model(cfg).state_dict()))
    value.configure_frozen_policy_anchor(spec)
    frozen = value._frozen_policy_anchor
    assert frozen is not None
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("empty eligible batch called reference")
    monkeypatch.setattr(frozen.model, "forward", forbidden)
    before = anchor._rng()
    result = frozen.penalty({"policy": torch.zeros(2, 3, requires_grad=True)},
        {"is_network_turn": torch.zeros(2), "has_policy": torch.ones(2)})
    assert result is None
    assert anchor._same_rng(before, anchor._rng())
    assert frozen.diagnostics()["eligible_rows"] == 0
    assert frozen.diagnostics()["reference_forwards"] == 0
    assert frozen.diagnostics()["mean_kl"] is None
    value.writer.close()


def test_reference_alias_is_rejected_before_loading_or_freezing_student(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from chess_anti_engine.uci import model_loader

    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    value = trainer(cfg, copy.deepcopy(build_model(cfg).state_dict()))
    before = copy.deepcopy(value.model.state_dict())
    monkeypatch.setattr(model_loader, "build_model", lambda _cfg: value.model)
    with pytest.raises(ValueError, match="aliases student"):
        value.configure_frozen_policy_anchor(spec)
    assert all(p.requires_grad for p in value.model.parameters())
    require_equal(before, value.model.state_dict(), "alias refusal leaves student unchanged")
    value.writer.close()


def test_actual_transformer_reference_receives_same_post_mirror_relations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(kind="transformer", embed_dim=16, num_heads=2, num_layers=1, use_smolgen=False,
        use_dynamic_relations=True, history_rep_fix=True)
    spec = reference(tmp_path, cfg)
    value = trainer(cfg, copy.deepcopy(build_model(cfg).state_dict()))
    value.configure_frozen_policy_anchor(spec)
    frozen = value._frozen_policy_anchor
    assert frozen is not None
    replay = FixedRows(cfg)
    relations = np.zeros((2, 5, 64, 64), dtype=np.uint8)
    relations[:, 0, 1, 2] = 1
    replay.arrays["relations"] = relations
    observed: list[dict[str, torch.Tensor]] = []
    original = value._host_batch_to_tensors
    def tensors(batch: Any) -> dict[str, torch.Tensor]:
        result = original(batch)
        observed.append(result)
        return result
    monkeypatch.setattr(value, "_host_batch_to_tensors", tensors)
    reference_inputs = []
    def receive(_module: Any, args: Any, kwargs: Any) -> None:
        reference_inputs.append((args[0].clone(), kwargs["relations"].clone()))
    handle = frozen.model.register_forward_pre_hook(receive, with_kwargs=True)
    value.train_steps(cast(Any, replay), batch_size=2, steps=1)
    handle.remove()
    assert len(reference_inputs) == 1
    batch = observed[0]
    eligible = batch["is_network_turn"].bool() & batch["has_policy"].bool()
    torch.testing.assert_close(reference_inputs[0][0], batch["x"][eligible], rtol=0, atol=0)
    torch.testing.assert_close(reference_inputs[0][1], batch["relations"][eligible], rtol=0, atol=0)
    value.writer.close()
