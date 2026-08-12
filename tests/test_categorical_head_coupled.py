"""The coupled categorical value head (lc0's value-embedding topology).

Background: the standalone `value_categorical` head was measured as a
two-instrument NULL even after its TARGET was repaired to the SF blend
(Tier-10 arena 2026-08-06, Tier-11 paired `value_regret` 2026-08-12). The
remaining structural difference from lc0 is topology: lc0 hangs the
distributional aux off the SAME value embedding the scalar output reads, so the
aux gradient supervises that representation directly. Ours hung it off an
independent `ValueHead`, where the aux gradient reaches the scalar head only by
diffusing back through the whole 63M trunk.

The load-bearing test here is `test_coupled_categorical_gradient_reaches_wdl_head`
and its standalone counterpart: they assert the MECHANISM (where the gradient
lands), not merely that the config key is plumbed. A knob that is accepted and
then has no effect is this codebase's signature defect.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
import torch
from torch import nn

from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
    ModelConfig,
    build_model,
    model_config_from_manifest_dict,
    model_config_to_manifest_dict,
    resume_model_config_from_arch,
)
from chess_anti_engine.model.transformer import (
    CATEGORICAL_HEAD_BINS,
    ChessNet,
    ValueHead,
)
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable import _build_trial_model_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


def _cfg(*, coupled: bool) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        categorical_head_coupled=coupled,
    )


def _build(*, coupled: bool) -> ChessNet:
    model = build_model(_cfg(coupled=coupled))
    assert isinstance(model, ChessNet)
    return model.eval()


def _planes(batch: int = 3) -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(batch, 175, 8, 8)


def test_head_bin_count_is_pinned_to_the_target_builder() -> None:
    """`model/` cannot import `train/`, so the two constants are pinned by test."""
    assert CATEGORICAL_HEAD_BINS == DEFAULT_CATEGORICAL_BINS


def test_hidden_split_is_bit_identical_to_forward() -> None:
    """`hidden()` + `head_from_hidden()` must reproduce `forward()` exactly.

    If this drifts, the coupled model's WDL output silently stops matching the
    standalone model's and every cross-topology comparison becomes invalid.
    """
    torch.manual_seed(0)
    head = ValueHead(64, 3)
    x = torch.randn(5, 64, 64)
    assert torch.equal(head(x), head.head_from_hidden(head.hidden(x)))


def test_net_layout_guard_fires_when_the_sequential_is_restructured() -> None:
    """Negative control: the positional indexing guard must be ABLE to fail."""
    head = ValueHead(64, 3)
    head.net = nn.Sequential(nn.Linear(8192, 3))  # a plausible future "simplification"
    with pytest.raises(RuntimeError, match=r"ValueHead\.net layout changed"):
        head.hidden(torch.randn(2, 64, 64))


def test_coupled_head_is_built_instead_of_the_standalone_one() -> None:
    std, cpl = _build(coupled=False), _build(coupled=True)
    assert std.value_categorical is not None
    assert std.value_categorical_coupled is None
    assert cpl.value_categorical is None
    assert cpl.value_categorical_coupled is not None
    for model in (std, cpl):
        assert model(_planes())["categorical"].shape == (3, CATEGORICAL_HEAD_BINS)


def test_coupled_head_costs_two_orders_of_magnitude_fewer_params() -> None:
    std, cpl = _build(coupled=False), _build(coupled=True)
    n_std = sum(p.numel() for p in std.parameters())
    n_cpl = sum(p.numel() for p in cpl.parameters())
    standalone_cost = sum(p.numel() for p in ValueHead(64, CATEGORICAL_HEAD_BINS).parameters())
    coupled_cost = cpl.value_wdl.hidden_dim * CATEGORICAL_HEAD_BINS + CATEGORICAL_HEAD_BINS
    assert n_std - n_cpl == standalone_cost - coupled_cost
    assert coupled_cost < standalone_cost


def test_warm_start_from_a_standalone_checkpoint_leaves_wdl_bit_identical() -> None:
    """The whole point of this topology: iter138 loads with every shared tensor
    unchanged, so the coupled arm starts from exactly the banked net."""
    std, cpl = _build(coupled=False), _build(coupled=True)
    missing, unexpected = cpl.load_state_dict(std.state_dict(), strict=False)
    assert set(missing) == {
        "value_categorical_coupled.weight",
        "value_categorical_coupled.bias",
    }
    assert all(k.startswith("value_categorical.") for k in unexpected)

    x = _planes()
    with torch.no_grad():
        assert torch.equal(std(x)["wdl"], cpl(x)["wdl"])


def test_coupled_categorical_logits_start_near_zero() -> None:
    """Small init, so enabling the aux head does not shock a warm-started net."""
    cpl = _build(coupled=True)
    with torch.no_grad():
        assert float(cpl(_planes())["categorical"].abs().max()) < 0.05


def _wdl_first_projection(model: ChessNet) -> torch.Tensor:
    """`value_wdl`'s first projection weight — the parameter the coupled aux head
    is supposed to supervise and the standalone one provably cannot reach.

    Indexed through `nn.Sequential`, whose `__getitem__` is typed `Tensor | Module`,
    so the isinstance narrowing is the version-proof way to get at `.weight`.
    """
    layer = model.value_wdl.net[0]
    assert isinstance(layer, nn.Linear)
    return layer.weight


def _categorical_grad_on(model: ChessNet, param: torch.Tensor) -> float:
    """Backprop a loss on `categorical` ALONE and report |grad| landing on `param`."""
    model.zero_grad(set_to_none=True)
    model(_planes())["categorical"].pow(2).sum().backward()
    return 0.0 if param.grad is None else float(param.grad.abs().sum())


def test_coupled_categorical_gradient_reaches_the_wdl_head_itself() -> None:
    """⚑ THE MECHANISM. Coupled, the aux loss must supervise the very projection
    the WDL logits are read from — not just the shared trunk."""
    cpl = _build(coupled=True)
    assert _categorical_grad_on(cpl, _wdl_first_projection(cpl)) > 0.0


def test_standalone_categorical_gradient_never_touches_the_wdl_head() -> None:
    """The counterpart that makes the test above meaningful: with the legacy
    topology the aux gradient reaches `value_wdl`'s own parameters NOT AT ALL,
    which is precisely the representation supervision lc0 gets and we did not."""
    std = _build(coupled=False)
    assert _categorical_grad_on(std, _wdl_first_projection(std)) == 0.0


def _wdl_grad_on(model: ChessNet, param: torch.Tensor) -> float:
    """Backprop a loss on `wdl` ALONE and report |grad| landing on `param`."""
    model.zero_grad(set_to_none=True)
    model(_planes())["wdl"].pow(2).sum().backward()
    return 0.0 if param.grad is None else float(param.grad.abs().sum())


@pytest.mark.parametrize("coupled", [True, False])
def test_wdl_loss_still_trains_the_wdl_head_under_either_topology(coupled: bool) -> None:
    """⚑ The coupling must be a shared READ of the hidden activation, never a
    rewiring of the WDL head's own backward pass.

    Sharing the hidden invites a `.detach()` -- on the WDL branch it looks like a
    harmless "stop the aux head perturbing value" guard, and it silently zeroes
    the WDL loss's gradient on the head's entire body while every other test in
    this file still passes: outputs, params, warm-start and the aux-gradient
    mechanism are all unchanged. The primary loss would go quiet, which is a far
    worse failure than the aux head not working.
    """
    model = _build(coupled=coupled)
    assert _wdl_grad_on(model, _wdl_first_projection(model)) > 0.0
    assert _wdl_grad_on(model, model.value_wdl.token_proj.weight) > 0.0


def test_flag_survives_the_manifest_round_trip() -> None:
    """All five plumbing sites: a flag that does not round-trip through the
    manifest is a flag that silently reverts on resume.

    Asserted in BOTH directions and read back through
    `model_config_from_manifest_dict`: checking only the write side lets a reader
    hardcoded to `False` pass, and the reader is the half that runs on resume.
    """
    for want in (True, False):
        manifest = model_config_to_manifest_dict(_cfg(coupled=want))
        assert manifest["categorical_head_coupled"] is want
        assert model_config_from_manifest_dict(manifest).categorical_head_coupled is want


def _production_model_config(coupled: bool) -> ModelConfig:
    """The ModelConfig production actually builds: yaml -> TrialConfig -> ModelConfig.

    `model_config_from_flat_config` is NOT this path -- it is only reached from
    `scripts/`. `tune/trainable.py` builds the trial's model from `TrialConfig`,
    so a key that stops at the flat dict never reaches training at all.
    """
    flat = flatten_run_config_defaults(
        {"model": {"categorical_head_coupled": coupled}, "train": {}}
    )
    return _build_trial_model_config(TrialConfig.from_dict(flat))


@pytest.mark.parametrize("coupled", [True, False])
def test_flag_reaches_the_model_config_production_builds(coupled: bool) -> None:
    """⚑ The live yaml's value must survive every hop to the trial's ModelConfig."""
    assert _production_model_config(coupled).categorical_head_coupled is coupled


@pytest.mark.parametrize(("checkpoint_value", "config_value"), [(False, True), (True, False)])
def test_topology_migration_survives_resume(checkpoint_value: bool, config_value: bool) -> None:
    """⚑ Resume takes topology from the checkpoint's `arch`, so an unlisted flag
    reverts to the donor's layout on EVERY resume and the experiment can never
    run -- silently, because the two heads share no state_dict keys and the
    tolerant loader is happy either way. This is the config-owned migration the
    `_RESUME_CONFIG_OWNED_ENCODING_KEYS` docstring exists for.
    """
    donor = dataclasses.replace(_cfg(coupled=False), categorical_head_coupled=checkpoint_value)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, categorical_head_coupled=config_value)
    resumed = resume_model_config_from_arch(arch, run_cfg)
    assert resumed.categorical_head_coupled is config_value


def _trainer(model: ChessNet, log_dir: Path) -> Trainer:
    return Trainer(
        model, device="cpu", lr=1e-3, optimizer="adamw", warmup_steps=10,
        warmup_lr_start=1e-5, use_amp=False, log_dir=log_dir,
        tb_log_interval=1000, prefetch_batches=False,
    )


def test_enabling_the_flag_warm_starts_weights_but_RESETS_the_optimizer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """⚑ EXPERIMENT-VALIDITY PIN, not a fix: switching topology on an existing
    checkpoint keeps the WEIGHTS and throws away every optimizer moment.

    `_remap_optimizer_state_for_new_params` can only SPLICE fresh slots in, and
    only when the donor is SHORTER. Here the donor is LONGER (the standalone
    `ValueHead`'s six tensors leave, two arrive), so no splice is possible --
    and it could not be made positional anyway, since optimizer state is indexed
    by position with no parameter names to realign on.

    ⇒ Both arms of the coupled-head experiment must be warm-started the SAME
    way. Comparing a flag-flip against a run that kept its moments measures the
    optimizer reset, not the head. Asserted rather than fixed, and asserted to be
    LOUD: a silent moment reset is exactly this codebase's signature defect.
    """
    std, cpl = _build(coupled=False), _build(coupled=True)
    src = _trainer(std, tmp_path / "src")
    src.model(_planes())["wdl"].pow(2).sum().backward()
    src.opt.step()
    assert any(bool(s) for s in src.opt.state.values()), "donor must carry real moments"
    ckpt = tmp_path / "standalone.pt"
    src.save(ckpt)

    dst = _trainer(cpl, tmp_path / "dst")
    with caplog.at_level(logging.WARNING):
        dst.load(ckpt)

    assert not any(bool(s) for s in dst.opt.state.values()), "moments must be gone"
    assert any("reinitialising optimizer" in r.message for r in caplog.records)
    with torch.no_grad():  # ...while the weights themselves DID warm-start
        assert torch.equal(std(_planes())["wdl"], cpl(_planes())["wdl"])


def test_the_arena_loader_rebuilds_the_coupled_topology_from_a_saved_checkpoint(
    tmp_path: Path,
) -> None:
    """⚑ THE VERDICT PATH. `scripts/arena_standard.py` and UCI both go through
    `load_model_from_checkpoint`, which builds topology from the checkpoint's
    embedded `arch` and NOT from any yaml. If the flag failed to round-trip
    there, the arena would silently score a DEFAULTED (standalone) model while
    reporting it as the coupled arm -- a wrong verdict rather than a crash.

    Written against a real `Trainer.save` payload, because the claim is about
    what production writes, not about what this test can construct.
    """
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    cfg = _cfg(coupled=True)
    trainer = Trainer(
        build_model(cfg), device="cpu", lr=1e-3, optimizer="adamw", warmup_steps=10,
        warmup_lr_start=1e-5, use_amp=False, log_dir=tmp_path / "log",
        tb_log_interval=1000, prefetch_batches=False, model_config=cfg,
    )
    ckpt = tmp_path / "coupled.pt"
    trainer.save(ckpt)

    loaded = load_model_from_checkpoint(ckpt, device="cpu")
    assert isinstance(loaded, ChessNet)
    assert loaded.value_categorical_coupled is not None
    assert loaded.value_categorical is None


def test_resume_still_takes_real_topology_from_the_checkpoint() -> None:
    """Negative control for the test above: widening the config-owned list must
    not let a SHAPE key escape the checkpoint, which is the silent partial-load
    (and optimizer-moment crash) that `arch` ownership exists to prevent."""
    donor = dataclasses.replace(_cfg(coupled=True), num_layers=1)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, num_layers=2)
    assert resume_model_config_from_arch(arch, run_cfg).num_layers == 1


def test_flag_is_accepted_by_the_yaml_schema() -> None:
    """Category-(a) trap, tested BEHAVIOURALLY: a live-yaml key absent from the
    schema makes `flatten_run_config_defaults` raise, and it is called before the
    argument parser and outside any try -- so the process would not boot at all."""
    flat = flatten_run_config_defaults(
        {"model": {"categorical_head_coupled": True}, "train": {}}
    )
    assert flat["categorical_head_coupled"] is True

    with pytest.raises(ValueError, match="categorical_head_coupled_typo"):
        flatten_run_config_defaults(
            {"model": {"categorical_head_coupled_typo": True}, "train": {}}
        )
