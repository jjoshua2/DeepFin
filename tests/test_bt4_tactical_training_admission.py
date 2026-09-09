"""Genuine tactical-summary contract and unchanged historical trainer wiring.

Metadata fixtures only; no operational qualification or training is performed.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import bt4_direct_screen as arena
from scripts import bt4_one_epoch_screen as epoch
from tests.test_bt4_one_epoch_screen import training_fixture, training_only_manifest
from tests.test_bt4_value_training_admission import prepared as value_prepared


SUMMARY = "bt4_sf_tactical_policy_summary.json"


def prepared(tmp_path, monkeypatch):
    m, files, _value, _derived, pins = value_prepared(tmp_path, monkeypatch)
    corpus = epoch.corpus_for(m)
    profile = epoch.TACTICAL_PROFILE
    monkeypatch.setitem(epoch.CORPORA, profile, corpus)
    m["profile"] = profile
    m["input_pins"][str(corpus / SUMMARY)] = m["input_pins"].pop(
        str(corpus / "bt4_value_rewrite_summary.json")
    )
    sf = files[str(epoch.SOURCE / "derive_targets_summary.json")]
    source_sha = epoch.COMMON_PINS[str(epoch.SOURCE / "derive_targets_summary.json")]
    parent = epoch.CORPORA["B100"]
    recipe: dict[str, Any] = {
        "schema": 1,
        "status": "COMPLETE",
        "kind": "bt4_sf_tactical_policy_attenuation",
        "algorithm": "stored-b100-sf-gap100-decay100-floor0.1-categorical-mates-v1",
        "source_dir": str(parent),
        "sf_source_dir": str(epoch.SOURCE),
        "raw_dir": str(epoch.SOFTSF_RAW),
        "raw_limit": 20000000,
        "rows": 18910484,
        "shards": 2309,
        "rows_dropped_no_result": 1089516,
        "mutated_arrays": ["policy_target"],
        "nonpolicy_arrays_copied": 16,
        "raw_manifest_present": False,
        "source_derive_summary_sha256": epoch.B100_PARENT_PINS[
            "derive_targets_summary.json"
        ],
        "source_policy_summary_sha256": epoch.B100_PARENT_PINS[
            "bt4_policy_mix_summary.json"
        ],
        "sf_derive_summary_sha256": source_sha,
        "recipe": {
            "gap_cp": 100.0,
            "decay_cp": 100.0,
            "relative_floor": 0.1,
            "mate_handling": "categorical-v1",
            "cp_domain": [-32000, 32000],
            "base": "normalized stored B100 float16 policy",
            "storage": "float64 attenuation -> float32 -> float16; all-one weights preserve bytes",
        },
        "changed_rows": 5000,
        "stored_mass_error_max": 0.0001,
        "categories": {
            "no_mate": 18910000,
            "losing_mate_alternatives": 200,
            "winning_mate_available": 200,
            "all_forced_losses": 84,
        },
        "winning_mate_zero_base_mass_rows": 10,
        "stored_support_losses": 20,
        "stored_relative_error_max": 1.0,
        "stored_TV_error_max": 0.0001,
        "producer_sha256": {
            str(tmp_path / k): v for k, v in epoch.TACTICAL_PRODUCER_PINS.items()
        },
        "metadata_sha256": {
            str(epoch.SOURCE / "derive_targets_summary.json"): source_sha,
            str(epoch.SOFTSF_RAW / "summary.json"): epoch.SOFTSF_RAW_SUMMARY_SHA,
            **{str(parent / k): v for k, v in epoch.B100_PARENT_PINS.items()},
        },
        "outputs": [
            {
                **shard,
                "source_policy_sha256": "a" * 64,
                "policy_sha256": "b" * 64,
                "source_storage_identity": "c" * 64,
                "original_sf_storage_identity": "d" * 64,
            }
            for shard in sf["shards"]
        ],
    }
    derived = {
        **sf,
        "policy_target_postprocess": {
            k: v for k, v in recipe.items() if k != "outputs"
        },
    }
    files[str(corpus / SUMMARY)] = recipe
    files[str(corpus / "derive_targets_summary.json")] = derived
    qualification = files[m["data_qualification"]["path"]]
    qualification["profile"] = profile
    qualification["rewrite_summary"]["path"] = str(corpus / SUMMARY)
    return m, files, recipe, derived, pins


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "gap",
        "decay",
        "floor",
        "mates",
        "ideal_base",
        "value",
        "history",
        "incomplete",
        "producer",
        "parent",
        "raw",
        "coverage",
        "hash",
        "fake_mix",
        "categories",
        "zero_mates",
        "support",
        "nan",
        "inert",
        "scope",
        "writing",
        "recipe_kind",
        "parent_mix",
    ],
)
def test_tactical_recipe_admission_and_exact_original_command(
    tmp_path, monkeypatch, defect
):
    m, files, recipe, derived, pins = prepared(tmp_path, monkeypatch)
    if defect in {"gap", "decay", "floor"}:
        recipe["recipe"][
            {"gap": "gap_cp", "decay": "decay_cp", "floor": "relative_floor"}[defect]
        ] *= 2
    elif defect == "mates":
        recipe["recipe"]["mate_handling"] = "cp-gaps"
    elif defect == "ideal_base":
        recipe["recipe"]["base"] = "unrounded BT4 logits"
    elif defect == "value":
        derived["value_scheme"] = {"name": "sf-bt4-native-alpha=0.5"}
    elif defect == "history":
        derived["input"] = {"input_history_encoding": "zeros"}
    elif defect == "incomplete":
        recipe["status"] = "FAILED"
    elif defect == "producer":
        recipe["producer_sha256"][str(tmp_path / "scripts/sf_policy_rewrite.py")] = (
            "f" * 64
        )
    elif defect == "parent":
        recipe["source_policy_summary_sha256"] = "f" * 64
    elif defect == "raw":
        recipe["metadata_sha256"][str(epoch.SOFTSF_RAW / "summary.json")] = "f" * 64
    elif defect == "coverage":
        recipe["outputs"] = recipe["outputs"][:-1]
    elif defect == "hash":
        del recipe["outputs"][0]["source_policy_sha256"]
    elif defect == "fake_mix":
        qualification = files[m["data_qualification"]["path"]]
        qualification["mix_summary"] = qualification.pop("rewrite_summary")
    elif defect == "categories":
        recipe["categories"]["no_mate"] -= 1
    elif defect == "zero_mates":
        recipe["winning_mate_zero_base_mass_rows"] = 201
    elif defect == "support":
        recipe["stored_support_losses"] = -1
    elif defect == "nan":
        recipe["stored_TV_error_max"] = float("nan")
    elif defect == "inert":
        recipe["changed_rows"] = 0
    elif defect == "scope":
        recipe["mutated_arrays"] = ["policy_target", "search_wdl"]
    elif defect == "writing":
        corpus = epoch.corpus_for(m)
        corpus.with_name(corpus.name + ".writing").mkdir()
    elif defect == "recipe_kind":
        recipe["kind"] = "global"
    elif defect == "parent_mix":
        parent = epoch.CORPORA["B100"]
        files[str(parent / "bt4_policy_mix_summary.json")]["alpha"] = 0.5
    derived["policy_target_postprocess"] = {
        k: v for k, v in recipe.items() if k != "outputs"
    }
    epoch.validate(m)
    if defect != "none":
        with pytest.raises(ValueError, match=r"tactical|data qualification"):
            epoch.check_pins(m)
        return
    assert epoch.check_pins(m) == {"same_training_runtime": True}
    assert epoch.comparisons(m) == ()
    assert set(m["input_pins"]) <= {p for p, _ in pins}
    assert {
        (str(epoch.CORPORA["B100"] / k), v) for k, v in epoch.B100_PARENT_PINS.items()
    } <= set(pins)
    files[m["runtime_manifest"]["path"]] = {"runtime": {"executable": sys.executable}}
    command = epoch.train_command(m)
    ordinary = epoch.train_command(training_only_manifest(tmp_path))
    command[command.index("--shards") + 1] = ordinary[ordinary.index("--shards") + 1]
    assert command == ordinary


@pytest.mark.parametrize("defect", ["schema2", "wrong_summary", "runtime", "budget"])
def test_tactical_manifest_never_dispatches_legacy_arenas_or_changes_runtime(
    tmp_path, monkeypatch, defect
):
    m, _files, _recipe, _derived, _pins = prepared(tmp_path, monkeypatch)
    if defect == "schema2":
        m["schema"] = 2
    elif defect == "wrong_summary":
        corpus = epoch.corpus_for(m)
        m["input_pins"][str(corpus / "bt4_policy_mix_summary.json")] = m[
            "input_pins"
        ].pop(str(corpus / SUMMARY))
    elif defect == "runtime":
        m["input_pins"][str(arena.RUNTIME / "scripts/lc0_control_train.py")] = "f" * 64
    else:
        m["training_seconds"] = 32400
    with pytest.raises(
        ValueError, match=r"requires schema3|pins differ|budget differs"
    ):
        epoch.validate(m)


@pytest.mark.parametrize("mismatch", [False, True])
def test_actual_summary_qualification_preserves_role_and_schedule(
    tmp_path, monkeypatch, mismatch
):
    m, _files, _recipe, _derived, _pins = prepared(tmp_path, monkeypatch)
    # Use real local JSON/checkpoint files for the completion consumer; the corpus
    # metadata setup above is synthetic and no trainer is launched.
    monkeypatch.undo()
    run, summary, report = training_fixture(tmp_path)
    corpus = m["input_pins"]
    selected = next(Path(p).parent for p in corpus if p.endswith(SUMMARY))
    monkeypatch.setitem(epoch.CORPORA, epoch.TACTICAL_PROFILE, selected)
    summary["corpus"]["shard_dirs"] = [str(selected)]
    arm = report["arms"].pop("E0T05")
    report["arms"][epoch.TACTICAL_PROFILE] = arm
    arm["corpus"] = str(selected)
    (run / "summary.json").write_text(json.dumps(summary))
    arm["summary_sha256"] = arena.sha(run / "summary.json")
    if mismatch:
        arm["canonical_plan_sha256"] = "different"
    path = tmp_path / "schedule_actual.json"
    path.write_text(json.dumps(report))
    if mismatch:
        with pytest.raises(ValueError, match="schedule"):
            epoch.completed_training(m, path)
        return
    result: dict[str, Any] = epoch.completed_training(m, path)
    assert result["role"] == result["checkpoint"]["role"] == epoch.TACTICAL_PROFILE
    assert result["complete"] is True
    assert result["historical_valid_control"] is False
    assert result["historical_validity_problems"] == ["historical purity limitation"]
