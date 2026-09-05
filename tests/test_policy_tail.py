"""Mechanism fixtures, bank CLI, and a real-checkpoint/replay collector smoke."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.policy_tail import (
    TailCohort,
    compare_policies,
    freeze_cohort,
    legal_log_probs,
    normalized_target,
    tail_event_loss,
)
from scripts.policy_tail_diagnostic import (
    main,
    primary_logits,
    read_bank,
    select_labeled_rows,
    summarize_bank,
    validate_bank,
)


def fixture_bank() -> tuple[dict[str, np.ndarray], dict]:
    # CE improves although the frozen rare move loses 50x probability.
    base = torch.tensor([[0.60, 0.395, 0.005]], dtype=torch.float64).log()
    candidate = torch.tensor([[0.85, 0.1499, 0.0001]], dtype=torch.float64).log()
    target = torch.tensor([[0.95, 0.03, 0.02]], dtype=torch.float64)
    legal = torch.ones_like(base, dtype=torch.bool)
    spec = TailCohort()
    bank = {
        "reference_logits": base.numpy(), "candidate_logits": candidate.numpy(),
        "target": target.numpy(), "legal": legal.numpy(),
        "cohort": freeze_cohort(base, target, legal, spec).numpy(),
        "row_ids": np.asarray(["fixture#row=0"]), "group_ids": np.asarray(["fixture#game=1"]),
    }
    return bank, {"schema_version": 1, "cohort_spec": asdict(spec), "synthetic": True}


def test_ce_can_improve_while_rare_move_collapses():
    bank, manifest = fixture_bank()
    report, _ = summarize_bank(bank, manifest)
    assert report["target_ce_delta"] < 0
    assert report["rare_rows"] == report["rare_actions"] == 1
    assert report["rare_log_mass_delta"] == pytest.approx(math.log(1 / 50))
    assert report["rare_action_drop_10x_fraction"] == 1
    assert report["rare_action_drop_100x_fraction"] == 0
    curve = report["iid_any_cohort_move_probability_not_mcts"]
    assert curve["128"]["reference"] == pytest.approx(1 - 0.995 ** 128)
    assert curve["128"]["candidate"] < curve["128"]["reference"]
    assert "not verified" in report["interpretation"]


def test_union_mass_cannot_hide_individual_move_collapse():
    base = torch.tensor([[0.006, 0.004, 0.99]], dtype=torch.float64).log()
    cand = torch.tensor([[0.00999, 0.00001, 0.99]], dtype=torch.float64).log()
    q = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
    legal = torch.ones_like(base, dtype=torch.bool)
    report, _ = compare_policies(base, cand, q, legal)
    assert report["rare_actions"] == 2
    assert report["rare_log_mass_delta"] == pytest.approx(0, abs=1e-12)
    assert report["rare_action_drop_100x_fraction"] == 0.5


def test_fixed_cohort_keeps_moves_after_they_stop_being_rare():
    bank, manifest = fixture_bank()
    bank["candidate_logits"] = np.log([[0.8, 0.1, 0.1]])
    report, _ = summarize_bank(bank, manifest)
    assert report["rare_actions"] == 1  # NOT reselected with candidate prior > 0.01
    assert report["rare_log_mass_delta"] == pytest.approx(math.log(20))


def test_equal_mean_reward_different_tail_likelihood():
    compact = torch.tensor([[0.0001, 0.9998, 0.0001]], dtype=torch.float64)
    tail = torch.tensor([[0.005, 0.99, 0.005]], dtype=torch.float64)
    reward = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)
    assert (compact * reward).sum() == pytest.approx((tail * reward).sum())
    legal = torch.ones_like(compact, dtype=torch.bool)
    event = (reward > 0.9)[:, None, :]
    assert tail_event_loss(tail.log(), legal, event) < tail_event_loss(compact.log(), legal, event)


def test_rare_tail_keeps_nonzero_gradient_even_below_exp_underflow():
    logits = torch.tensor([[0.0, -1000.0, 99999.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    event = torch.tensor([[[False, True, False]]])
    loss = tail_event_loss(logits, legal, event)
    assert loss.item() == pytest.approx(1000)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0].tolist() == pytest.approx([1, -1, 0])
    with torch.no_grad():
        updated = logits - 0.1 * logits.grad
    assert tail_event_loss(updated, legal, event) < loss


def test_empty_events_and_rows_do_not_create_nan_gradients_or_dilute_loss():
    logits = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    legal = torch.ones_like(logits, dtype=torch.bool)
    events = torch.tensor([[[True, False, False], [False, False, False]],
                           [[False, False, False], [False, False, False]]])
    loss = tail_event_loss(logits, legal, events)
    assert loss.item() == pytest.approx(math.log(3))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1], torch.zeros(3, dtype=torch.float64))


def test_all_empty_batch_is_connected_exact_zero():
    logits = torch.randn(3, 4, requires_grad=True)
    legal = torch.ones_like(logits, dtype=torch.bool)
    loss = tail_event_loss(logits, legal, torch.zeros((3, 2, 4), dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_tail_averaging_is_within_position_not_weighted_by_event_count():
    logits = torch.zeros(2, 4, dtype=torch.float64)
    legal = torch.ones_like(logits, dtype=torch.bool)
    events = torch.tensor([[[True, False, False, False], [True, True, False, False]],
                           [[True, True, True, True], [False, False, False, False]]])
    expected = ((math.log(4) + math.log(2)) / 2 + 0) / 2
    assert tail_event_loss(logits, legal, events).item() == pytest.approx(expected)


def test_tail_loss_gradcheck():
    torch.manual_seed(7)
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    legal = torch.ones_like(logits, dtype=torch.bool)
    events = torch.tensor([[[True, True, False, False]], [[False, False, True, False]]])
    assert torch.autograd.gradcheck(lambda x: tail_event_loss(x, legal, events), (logits,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_precision_and_illegal_logits(dtype):
    logits = torch.tensor([[0, -20, float("nan")]], dtype=dtype, requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    events = torch.tensor([[[False, True, False]]])
    loss = tail_event_loss(logits, legal, events)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert loss.item() == pytest.approx(20, abs=1e-5)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


@pytest.mark.parametrize("kwargs", [
    {"max_prior": 0}, {"max_prior": 1}, {"max_prior": float("nan")},
    {"min_target": 0}, {"min_target": 1.1}, {"min_target": float("nan")},
    {"min_boost": 1}, {"min_boost": float("inf")},
])
def test_bad_cohort_config(kwargs):
    with pytest.raises(ValueError):
        TailCohort(**kwargs)


@pytest.mark.parametrize("kind", ["all_illegal", "legal_nan", "shape", "mask_dtype", "logit_dtype"])
def test_invalid_policy_inputs(kind):
    x = torch.zeros(1, 3)
    legal = torch.ones_like(x, dtype=torch.bool)
    if kind == "all_illegal":
        legal[:] = False
    elif kind == "legal_nan":
        x[0, 1] = float("nan")
    elif kind == "shape":
        legal = legal[:, :2]
    elif kind == "mask_dtype":
        legal = legal.float()
    else:
        x = x.long()
    with pytest.raises(ValueError):
        legal_log_probs(x, legal)


@pytest.mark.parametrize("values", [[0, 0, 0], [1, -0.1, 0], [0.8, 0.1, 0.1], [1, float("nan"), 0]])
def test_missing_or_corrupt_target_is_not_good_evidence(values):
    legal = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError):
        normalized_target(torch.tensor([values], dtype=torch.float64), legal)


def test_illegal_events_are_rejected():
    with pytest.raises(ValueError, match="illegal"):
        tail_event_loss(torch.zeros(1, 3), torch.tensor([[True, True, False]]),
                        torch.tensor([[[False, False, True]]]))


def test_empty_cohort_reports_null_not_zero_effect():
    logits = torch.zeros(2, 3)
    legal = torch.ones_like(logits, dtype=torch.bool)
    report, _ = compare_policies(logits, logits, torch.ones_like(logits), legal)
    assert report["rare_rows"] == 0
    assert report["rare_log_mass_delta"] is None
    assert report["rare_action_drop_10x_fraction"] is None
    json.dumps(report, allow_nan=False)


def test_positive_iid_budgets_only():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="budgets"):
        compare_policies(x, x, x + 1, x.bool() | True, budgets=(0,))


def test_target_roundoff_is_normalized_and_detached():
    q = torch.tensor([[0.3, 0.699, 0]], requires_grad=True)
    normalized = normalized_target(q, torch.tensor([[True, True, False]]))
    assert normalized.sum() == pytest.approx(1)
    assert not normalized.requires_grad


def test_primary_head_not_opponent_reply():
    own, alias, sf = torch.ones(1, 3), torch.ones(1, 3) * 2, torch.zeros(1, 3)
    assert primary_logits({"policy_own": own, "policy": alias, "policy_sf": sf}) is own
    assert primary_logits({"policy": alias, "policy_sf": sf}) is alias
    with pytest.raises(ValueError, match="primary"):
        primary_logits({"policy_sf": sf})


def test_presence_flags_authoritative():
    batch = {"has_policy": np.array([1, 0, 1]), "has_legal_mask": np.array([1, 1, 0]),
             "policy_target": np.ones((3, 2)), "legal_mask": np.ones((3, 2))}
    np.testing.assert_array_equal(select_labeled_rows(batch), [True, False, False])
    del batch["has_legal_mask"]
    assert not select_labeled_rows(batch).any()
    batch["has_legal_mask"] = np.ones(3)
    del batch["legal_mask"]
    with pytest.raises(ValueError, match="missing"):
        select_labeled_rows(batch)


def test_duplicate_bank_rows_rejected():
    bank, manifest = fixture_bank()
    bank = {key: np.concatenate([value, value]) for key, value in bank.items()}
    with pytest.raises(ValueError, match="duplicate"):
        validate_bank(bank, manifest)


def test_bank_roundtrip_cli_and_no_overwrite(tmp_path: Path):
    bank, manifest = fixture_bank()
    source = tmp_path / "bank.npz"
    np.savez_compressed(source, **bank, manifest_json=np.asarray(json.dumps(manifest)))
    original = source.read_bytes()
    reread, provenance = read_bank(source)
    np.testing.assert_array_equal(reread["cohort"], bank["cohort"])
    assert provenance == manifest
    output = tmp_path / "readout"
    command = ["--bank", str(source), "--output-dir", str(output)]
    assert main(command) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["rare_action_drop_10x_fraction"] == 1
    assert (output / "per_row.npz").is_file()
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(command)


def test_bank_disallows_changing_frozen_ruler(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--bank", "unused.npz", "--max-prior", "0.5", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_nonbinary_presence_flags_rejected():
    with pytest.raises(ValueError, match="0/1"):
        select_labeled_rows({"has_policy": np.array([float("nan")])})


def test_scalar_bank_field_rejected():
    bank, manifest = fixture_bank()
    bank["candidate_logits"] = np.array(1.0)
    with pytest.raises(ValueError, match="row dimension"):
        validate_bank(bank, manifest)


def test_repository_collector_real_checkpoint_and_lazy_shard(tmp_path: Path):
    # Requires an installed repository/native build, not only these added files.
    from chess_anti_engine.encoding import input_plane_count
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
    from chess_anti_engine.replay.sample import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays

    config = ModelConfig(kind="tiny")
    model = build_model(config)
    checkpoint = tmp_path / "snapshot.pt"
    torch.save({"model": model.state_dict(), "arch": {**asdict(config), "_schema_version": ARCH_SCHEMA_VERSION}}, checkpoint)
    before = checkpoint.read_bytes()
    target = np.zeros(1858, dtype=np.float32)
    target[:3] = [0.95, 0.03, 0.02]
    legal = target > 0
    sample = ReplaySample(
        x=np.zeros((input_plane_count(config.input_extra_features), 8, 8), dtype=np.float32),
        policy_target=target, wdl_target=1, legal_mask=legal,
        input_history_encoding=config.input_history_encoding, game_id=9, ply_index=0,
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    save_local_shard_arrays(replay / "fixture.zarr", arrs=samples_to_arrays([sample]))
    output = tmp_path / "collected"
    assert main(["--replay-dir", str(replay), "--reference", str(checkpoint),
                 "--candidate", str(checkpoint), "--output-dir", str(output),
                 "--max-positions", "1", "--batch-size", "1"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["rows"] == 1
    assert report["target_ce_delta"] == pytest.approx(0)
    assert report["provenance"]["selected_rows"] == 1
    assert checkpoint.read_bytes() == before
