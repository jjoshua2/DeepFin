from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import bt4_recipe_readout as reader
from scripts import bt4_recipe_screen as launcher
from tests.test_bt4_recipe_readout import make as make, panel as panel, put
from tests.test_bt4_recipe_screen import package as package


def completed_training(tmp_path, checkpoint):
    """Tiny metadata shaped like the actual original-corpus completion receipt."""
    role = checkpoint["role"]
    run = Path(checkpoint["path"]).parent
    run.mkdir(exist_ok=True)
    source = str(tmp_path / "data/nnue_derived/armB/qtemp_0.0005_hist_20m")
    corpus = str(tmp_path / "data" / role)
    pins = {
        str(tmp_path / name): digest for name, digest in reader.TRAINING_PINS.items()
    }
    physical = "e" * 64
    sampling = {
        "mode": "game_epoch",
        "complete": True,
        "seed": 0,
        "batch_size": 512,
        "rows_planned": 18910484,
        "rows_realized": 18910484,
        "batches_planned": 36935,
        "batches_realized": 36935,
        "shards": 2309,
        "games": 97968,
        "plan_workers": 16,
        "load_workers": 16,
        "same_game_repeats_max": 0,
        "decoded_rows_resident": 0,
        "plan_sha256": physical,
        "realized_sha256": physical,
    }
    windows = [
        {
            "window_index": i,
            "steps_requested": min(88, 36935 - (i - 1) * 88),
            "train_steps_done": min(88, 36935 - (i - 1) * 88),
            "steps_cumulative": min(i * 88, 36935),
            "grad_nonfinite_skip_rate": 0.0,
            "transient_cuda_retry_batches": 0,
            "loss": 1.0,
            "grad_norm_mean": 1.0,
            "train_samples_seen": 0,
        }
        for i in range(1, 421)
    ]
    windows[-1]["train_samples_seen"] = 18910484
    summary = {
        "seed": 0,
        "warmup_steps": 1000,
        "batch_size": 512,
        "train_window_steps": 88,
        "steps_realized": 36935,
        "compute_loss_calls": 36935,
        "corpus": {"shard_dirs": [corpus]},
        "sampling": sampling,
        "checkpoints": [
            {"role": "last", "path": checkpoint["path"], "sha256": checkpoint["sha256"]}
        ],
        "train_window_metrics": windows,
        "train_windows": 420,
        "valid_control": False,
        "validity_problems": ["historical sampler intervention"],
    }
    summary_pin = put(run / "summary.json", summary)
    arm = {
        "canonical_plan_sha256": reader.CANONICAL_EPOCH,
        "corpus": corpus,
        "metadata_matches_source": True,
        "training_completion_verified": True,
        "staging": "verified actual",
        "physical_plan_sha256": physical,
        "summary_sha256": summary_pin["sha256"],
    }
    schedule = {
        "arms": {role: arm},
        "seed": 0,
        "batch_size": 512,
        "source": source,
        "pins": pins,
        "source_plan": {
            "plan_sha256": reader.CANONICAL_EPOCH,
            "rows_planned": 18910484,
            "batches_planned": 36935,
            "seed": 0,
            "batch_size": 512,
        },
        "verifier_sha256": reader.SCHEDULE_VERIFIER,
        "runtime": {
            "python": "3.10.12 fixture",
            "torch": "2.11.0+cu128",
            "numpy": "1.26.2",
        },
    }
    return put(
        run / "training.complete.json",
        {
            "role": role,
            "run": str(run),
            "checkpoint": checkpoint,
            "complete": True,
            "input_pins": pins,
            "summary_sha256": summary_pin["sha256"],
            "schedule": put(run / "schedule.json", schedule),
            "canonical_plan_sha256": reader.CANONICAL_EPOCH,
            "physical_plan_sha256": physical,
            "historical_valid_control": False,
            "historical_validity_problems": summary["validity_problems"],
            "training_charge_seconds": 100.0,
        },
    )


def qualify_manifest(m, tmp_path, roles):
    launch = json.loads(Path(m["launch"]["path"]).read_text())
    evidence = {}
    for side, role in zip(("candidate", "reference"), roles):
        evidence[side] = {**launch["identities"][side], "role": role}
        evidence[side + "_training"] = completed_training(tmp_path, evidence[side])
        launch["identities"][side + "_training"] = evidence[side + "_training"]
    launch.update(
        profile=reader.MATCHED_PROFILE,
        training=evidence,
        candidate_role=roles[0],
        reference_role=roles[1],
    )
    m["profile"] = reader.MATCHED_PROFILE
    m["launch"] = put(Path(m["launch"]["path"]), launch)
    return evidence


def test_new_roles_bind_low_high_and_direction_without_fixed_h20_hash(make, tmp_path):
    roles = ("SoftSF10", "G50")
    low = make(values=[0.0, 2.0] * 80, roles=roles)
    evidence = qualify_manifest(low, tmp_path, roles)
    high = make(name="high", low=False, roles=roles)
    qualify_manifest(high, tmp_path, roles)
    high["low_manifest"] = put(tmp_path / "low.json", low)
    report = reader.read_cell(high)
    assert (report["candidate_role"], report["reference_role"]) == roles
    assert report["profile"] == reader.MATCHED_PROFILE
    assert report["fixed_core_cross_budget"]["score_advantage_400_minus_100"] == 0
    assert reader.matched_training_pair(evidence) == roles
    cmd = launcher.command(
        evidence | {"book": {"path": "book.zip"}},
        {"executable": "python"},
        "low",
        tmp_path / "arena",
    )
    assert cmd[cmd.index("--candidate") + 1] == evidence["candidate"]["path"]
    assert cmd[cmd.index("--reference") + 1] == evidence["reference"]["path"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(complete=False),
        lambda r: r.update(role="unqualified"),
        lambda r: r["checkpoint"].update(sha256="f" * 64),
        lambda r: r.update(canonical_plan_sha256="f" * 64),
        lambda r: r["input_pins"].clear(),
    ],
)
def test_failed_wrong_role_checkpoint_schedule_or_training_pins_refused(
    make, tmp_path, change
):
    m = make(roles=("G50", "H20"))
    evidence = qualify_manifest(m, tmp_path, ("G50", "H20"))
    receipt = reader.read_json(evidence["candidate_training"])
    change(receipt)
    evidence["candidate_training"] = put(
        Path(evidence["candidate_training"]["path"]), receipt
    )
    with pytest.raises(reader.InvalidCell):
        reader.matched_training_pair(evidence)


@pytest.mark.parametrize(
    ("member", "change"),
    [
        ("summary", lambda x: x["sampling"].update(batches_realized=36934)),
        ("summary", lambda x: x.update(steps_realized=73870)),
        ("summary", lambda x: x.update(warmup_steps=2000)),
        ("summary", lambda x: x["train_window_metrics"][1].update(window_index=1)),
        (
            "summary",
            lambda x: x["train_window_metrics"][0].update(grad_nonfinite_skip_rate=1.0),
        ),
        ("schedule", lambda x: x["runtime"].update(numpy="2.2.6")),
        ("schedule", lambda x: x.update(source="/different/corpus")),
        ("schedule", lambda x: x["arms"]["G50"].update(metadata_matches_source=False)),
    ],
)
def test_no_new_horizon_corpus_runtime_or_incomplete_epoch(
    make, tmp_path, member, change
):
    m = make(roles=("G50", "H20"))
    evidence = qualify_manifest(m, tmp_path, ("G50", "H20"))
    receipt = reader.read_json(evidence["candidate_training"])
    schedule = reader.read_json(receipt["schedule"])
    if member == "summary":
        path = Path(receipt["run"]) / "summary.json"
        obj = json.loads(path.read_text())
        change(obj)
        receipt["summary_sha256"] = put(path, obj)["sha256"]
        schedule["arms"]["G50"]["summary_sha256"] = receipt["summary_sha256"]
    else:
        change(schedule)
    receipt["schedule"] = put(Path(receipt["schedule"]["path"]), schedule)
    evidence["candidate_training"] = put(
        Path(evidence["candidate_training"]["path"]), receipt
    )
    with pytest.raises(reader.InvalidCell):
        reader.matched_training_pair(evidence)


def test_new_profile_rejects_launch_role_and_training_pin_substitution(make, tmp_path):
    m = make(roles=("G50", "H20"))
    qualify_manifest(m, tmp_path, ("G50", "H20"))
    launch = reader.read_json(m["launch"])
    for changed in ("role", "pin"):
        bad = copy.deepcopy(launch)
        if changed == "role":
            bad["candidate_role"] = "H20"
        else:
            bad["identities"]["candidate_training"] = bad["identities"][
                "reference_training"
            ]
        m["launch"] = put(Path(m["launch"]["path"]), bad)
        with pytest.raises(reader.InvalidCell):
            reader.read_cell(m)


@pytest.mark.parametrize("package", [("SoftSF10", "G50")], indirect=True)
def test_owned_low_high_dispatch_preserves_new_role_and_receipt_bindings(
    package, tmp_path
):
    m, _ = package
    m["profile"] = reader.MATCHED_PROFILE
    for side in ("candidate", "reference"):
        m[side + "_training"] = completed_training(tmp_path, m[side])
    proof = reader.read_json(m["preparation"])
    proof["profile"] = m["profile"]
    proof["inputs"].update(
        {k: m[k] for k in ("candidate_training", "reference_training")}
    )
    m["preparation"] = put(Path(m["preparation"]["path"]), proof)
    launcher.execute(m)
    out = Path(m["output"])
    for stage in ("low", "high"):
        report = json.loads((out / stage / "readout.stdout.json").read_text())
        launch = json.loads((out / f"{stage}.launch.json").read_text())
        assert report["candidate_role"] == "SoftSF10"
        assert report["reference_role"] == "G50"
        assert launch["training"]["candidate_training"] == m["candidate_training"]
        assert launch["training"]["reference_training"] == m["reference_training"]
    complete = json.loads((out / "complete.json").read_text())
    assert complete["profile"] == reader.MATCHED_PROFILE
    assert complete["candidate_role"] == "SoftSF10"
    assert complete["reference_role"] == "G50"


def test_incomplete_receipt_refused_by_launcher_before_runtime_or_arena(
    make, tmp_path, monkeypatch
):
    cell = make(roles=("G50", "H20"))
    evidence = qualify_manifest(cell, tmp_path, ("G50", "H20"))
    receipt = reader.read_json(evidence["candidate_training"])
    receipt["complete"] = False
    evidence["candidate_training"] = put(
        Path(evidence["candidate_training"]["path"]), receipt
    )
    m = evidence | {
        "schema": 1,
        "profile": reader.MATCHED_PROFILE,
        "launcher_sha256": "a" * 64,
        "reader_sha256": "b" * 64,
        "supervisor_sha256": "c" * 64,
        "preregistration": put(tmp_path / "preregistration.json", {}),
        "book": {"path": str(tmp_path / "book"), "sha256": launcher.owned.BOOK_SHA},
    }
    monkeypatch.setattr(launcher.owned, "pin", lambda *args: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("runtime reached"),
    )
    with pytest.raises(reader.InvalidCell, match="training incomplete"):
        launcher.inputs(m)
