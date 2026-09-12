"""Downside completion admission keeps the original trainer and SF-value lineage."""

import sys

import pytest

from scripts import bt4_one_epoch_screen as epoch
from scripts.sf_policy_rewrite import recipe_for_summary
from tests.test_bt4_one_epoch_screen import training_only_manifest
from tests.test_bt4_tactical_training_admission import prepared as tactical_prepared


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "pilot",
        "partial",
        "dose",
        "gap",
        "mates",
        "producer",
        "value",
        "coverage",
        "legacy",
        "schema2",
    ],
)
def test_downside_registered_training_only_contract(tmp_path, monkeypatch, defect):
    m, files, recipe, derived, _ = tactical_prepared(tmp_path, monkeypatch)
    corpus = epoch.corpus_for(m)
    monkeypatch.setitem(epoch.CORPORA, epoch.DOWNSIDE_PROFILE, corpus)
    m["profile"] = epoch.DOWNSIDE_PROFILE
    old = str(corpus / "bt4_sf_tactical_policy_summary.json")
    new = str(corpus / "bt4_sf_downside_policy_summary.json")
    m["input_pins"][new] = m["input_pins"].pop(old)
    files[new] = files.pop(old)
    qualification = files[m["data_qualification"]["path"]]
    qualification["profile"] = m["profile"]
    qualification["rewrite_summary"]["path"] = new
    recipe.update(
        kind="bt4_sf_allmove_downside",
        algorithm="stored-b100-allmove-sf-gapgt300-weight0.5-ordinary-v1",
        recipe=recipe_for_summary(downside=True),
        categories={"ordinary_downside": 18000000, "mate_domain_unchanged": 910484},
        producer_sha256={
            str(tmp_path / k): v for k, v in epoch.DOWNSIDE_PRODUCER_PINS.items()
        },
    )
    if defect == "pilot":
        recipe["pilot_only"] = True
    elif defect == "partial":
        recipe["status"] = "PILOT_COMPLETE_NOT_TRAINING"
    elif defect == "dose":
        recipe["recipe"]["flagged_relative_weight"] = 0.25
    elif defect == "gap":
        recipe["recipe"]["gap_cp_strictly_greater_than"] = 100.0
    elif defect == "mates":
        recipe["recipe"]["mate_handling"] = "categorical-v1"
    elif defect == "producer":
        recipe["producer_sha256"][str(tmp_path / "scripts/sf_policy_rewrite.py")] = (
            "f" * 64
        )
    elif defect == "value":
        derived["value_scheme"] = {"name": "modified"}
    elif defect == "coverage":
        recipe["outputs"] = recipe["outputs"][:-1]
    elif defect == "legacy":
        recipe["recipe"] = recipe_for_summary()
    elif defect == "schema2":
        m["schema"] = 2
    derived["policy_target_postprocess"] = {
        k: v for k, v in recipe.items() if k != "outputs"
    }
    if defect == "schema2":
        with pytest.raises(ValueError, match="requires schema3"):
            epoch.validate(m)
        return
    epoch.validate(m)
    if defect != "none":
        with pytest.raises(ValueError, match=r"downside|tactical|data qualification"):
            epoch.check_pins(m)
        return
    assert epoch.check_pins(m) == {"same_training_runtime": True}
    assert epoch.comparisons(m) == ()
    files[m["runtime_manifest"]["path"]] = {"runtime": {"executable": sys.executable}}
    actual = epoch.train_command(m)
    original = epoch.train_command(training_only_manifest(tmp_path))
    actual[actual.index("--shards") + 1] = original[original.index("--shards") + 1]
    assert actual == original
