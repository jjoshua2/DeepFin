"""Typed metadata admission only; actual target/gradient proof lives beside producer."""

import sys
from typing import Any

import pytest

from scripts import bt4_direct_screen as arena
from scripts import bt4_one_epoch_screen as epoch
from scripts import bt4_value_rewrite as rewrite
from tests.test_bt4_one_epoch_screen import training_only_manifest


def prepared(tmp_path, monkeypatch, profile="B100V10", modern=False):
    alpha = epoch.VALUE_ALPHAS[profile]
    m = training_only_manifest(tmp_path)
    parent = epoch.CORPORA["B100"]
    corpus = tmp_path / "candidate"
    corpus.mkdir()
    monkeypatch.setitem(epoch.CORPORA, profile, corpus)
    m["profile"] = profile
    for name in ("derive_targets_summary.json", "bt4_policy_mix_summary.json"):
        del m["input_pins"][str(parent / name)]
    m["input_pins"].update(
        {
            str(corpus / rewrite.SUMMARY): "a" * 64,
            str(corpus / rewrite.DERIVE_SUMMARY): "b" * 64,
        }
    )
    sf = {
        "scheme": {"canonical": "uniform-d9"},
        "value_scheme": {"name": "search"},
        "input": {"input_history_encoding": "lc0_root_legacy_meta"},
        "shards": [
            {"path": f"shard_{i:06d}.zarr", "rows": 8192 if i < 2308 else 3348}
            for i in range(2309)
        ],
    }
    sf_sha = epoch.COMMON_PINS[str(epoch.SOURCE / rewrite.DERIVE_SUMMARY)]
    policy = {
        "kind": "global",
        "algorithm": "legal-normalized-global-arithmetic-v1",
        "alpha": 1.0,
        "bt4_temperature": 0.5,
        "rows": 18910484,
        "expected_shards": 2309,
        "source_dir": str(epoch.SOURCE),
        "source_derive_summary_sha256": sf_sha,
        "mutated_arrays": ["policy_target"],
    }
    original = {**sf, "policy_target_postprocess": policy}
    recipe: dict[str, Any] = {
        "schema": 1,
        "status": "COMPLETE",
        "kind": "bt4_value_rewrite",
        "algorithm": rewrite.algorithm(alpha),
        "sf_weight": 1 - alpha,
        "bt4_weight": alpha,
        "wdl_order": "WDL",
        "wdl_pov": "side_to_move",
        "wdl_kind": "probabilities",
        "wdl_output": "/output/wdl",
        "onnx_sha256": "1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0",
        "rows": 18910484,
        "shards": 2309,
        "source_dir": str(parent),
        "sf_source_dir": str(epoch.SOURCE),
        "source_derive_summary_sha256": epoch.B100_PARENT_PINS[
            "derive_targets_summary.json"
        ],
        "source_policy_summary_sha256": epoch.B100_PARENT_PINS[
            "bt4_policy_mix_summary.json"
        ],
        "sf_derive_summary_sha256": sf_sha,
        "mutated_arrays": ["search_wdl"],
        "unchanged_arrays": sorted(rewrite.ARRAYS - {"search_wdl"}),
        "value_scheme": rewrite.value_scheme(alpha),
        "value_source": rewrite.value_source(
            "1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0",
            "/output/wdl",
            alpha,
        ),
        "changed_rows": 1000,
        "stored_mass_error_max": 0.0001,
        "outputs": sf["shards"],
        "producer_sha256": {
            str(tmp_path / k): v for k, v in epoch.VALUE_PRODUCER_PINS.items()
        },
    }
    if modern or profile == "B100V50":
        recipe["producer_sha256"][str(tmp_path / "scripts/bt4_value_rewrite.py")] = (
            epoch.VALUE_ALPHA_PRODUCER_SHA
        )
    derived = {
        **original,
        "value_scheme": {
            "name": rewrite.value_scheme(alpha),
            "source": rewrite.value_source(
                "1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0",
                "/output/wdl",
                alpha,
            ),
        },
        "value_target_postprocess": {k: v for k, v in recipe.items() if k != "outputs"},
    }
    qualification = {
        "schema": 1,
        "status": "PASS_REGISTERED_CORPUS_QUALIFICATION",
        "profile": profile,
        "corpus": str(corpus),
        "rows": 18910484,
        "shards": 2309,
        "source": {"path": str(epoch.SOURCE), "derive_sha256": sf_sha},
        "derive_summary": {
            "path": str(corpus / rewrite.DERIVE_SUMMARY),
            "sha256": "b" * 64,
        },
        "rewrite_summary": {"path": str(corpus / rewrite.SUMMARY), "sha256": "a" * 64},
    }
    files: dict[str, Any] = {
        str(epoch.SOURCE / rewrite.DERIVE_SUMMARY): sf,
        str(parent / rewrite.DERIVE_SUMMARY): original,
        str(parent / rewrite.POLICY_SUMMARY): policy,
        str(corpus / rewrite.DERIVE_SUMMARY): derived,
        str(corpus / rewrite.SUMMARY): recipe,
        m["data_qualification"]["path"]: qualification,
    }
    monkeypatch.setattr(arena, "read", lambda path: files[str(path)])
    monkeypatch.setattr(
        arena,
        "sha",
        lambda _path: epoch.B100_PARENT_PINS["bt4_policy_mix_summary.json"],
    )
    pins = []
    monkeypatch.setattr(
        arena, "pin", lambda path, digest: pins.append((str(path), digest))
    )
    monkeypatch.setattr(
        arena, "runtime_identity", lambda _ref: {"same_training_runtime": True}
    )
    return m, files, recipe, derived, pins


@pytest.mark.parametrize("profile", ["B100V10", "B100V50"])
@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "weight",
        "logits",
        "teacher",
        "policy",
        "lineage",
        "coverage",
        "fake_mix",
        "inert",
        "nan",
        "wrong_scope",
        "producer",
        "parent",
        "dose_identity",
        "algorithm",
        "source_dose",
    ],
)
def test_value_training_requires_exact_recipe_and_preserves_runtime(
    tmp_path, monkeypatch, defect, profile
):
    m, files, recipe, derived, pins = prepared(tmp_path, monkeypatch, profile)
    if defect == "dose_identity":
        recipe["value_scheme"] = rewrite.value_scheme(
            0.5 if profile == "B100V10" else 0.1
        )
    elif defect == "source_dose":
        recipe["value_source"] = rewrite.value_source(
            recipe["onnx_sha256"],
            recipe["wdl_output"],
            0.5 if profile == "B100V10" else 0.1,
        )
    elif defect == "algorithm":
        recipe["algorithm"] = "different"
    elif defect == "weight":
        recipe["bt4_weight"] = 0.2
    elif defect == "logits":
        recipe["wdl_kind"] = "logits"
    elif defect == "teacher":
        recipe["onnx_sha256"] = "0" * 64
    elif defect == "policy":
        derived["policy_target_postprocess"] = {"kind": "different"}
    elif defect == "lineage":
        derived["input"] = {"input_history_encoding": "zeros"}
    elif defect == "coverage":
        recipe["outputs"] = recipe["outputs"][:-1]
    elif defect == "fake_mix":
        q = files[m["data_qualification"]["path"]]
        q["mix_summary"] = q.pop("rewrite_summary")
    elif defect == "inert":
        recipe["changed_rows"] = 0
    elif defect == "nan":
        recipe["stored_mass_error_max"] = float("nan")
    elif defect == "producer":
        recipe["producer_sha256"][str(tmp_path / "scripts/bt4_value_rewrite.py")] = (
            "f" * 64
        )
    elif defect == "parent":
        recipe["source_derive_summary_sha256"] = "f" * 64
    elif defect == "wrong_scope":
        recipe["mutated_arrays"] = ["search_wdl", "policy_target"]
    derived["value_target_postprocess"] = {
        k: v for k, v in recipe.items() if k != "outputs"
    }
    epoch.validate(m)
    if defect != "none":
        with pytest.raises(ValueError, match=r"value|data qualification"):
            epoch.check_pins(m)
        return
    assert epoch.check_pins(m) == {"same_training_runtime": True}
    assert epoch.comparisons(m) == ()
    assert set(m["input_pins"]) <= {p for p, _ in pins}
    files[m["runtime_manifest"]["path"]] = {"runtime": {"executable": sys.executable}}
    actual = epoch.train_command(m)
    baseline = epoch.train_command(training_only_manifest(tmp_path))
    actual[actual.index("--shards") + 1] = baseline[baseline.index("--shards") + 1]
    assert actual == baseline


@pytest.mark.parametrize("profile", ["B100V10", "B100V50"])
def test_value_profile_cannot_start_legacy_arenas(tmp_path, monkeypatch, profile):
    m, _, _, _, _ = prepared(tmp_path, monkeypatch, profile)
    m["schema"] = 2
    with pytest.raises(ValueError, match=f"{profile} requires schema3"):
        epoch.validate(m)


@pytest.mark.parametrize("profile", ["B100V10", "B100V50"])
def test_new_producer_and_profile_specific_proof(tmp_path, monkeypatch, profile):
    m, files, recipe, derived, _ = prepared(tmp_path, monkeypatch, profile, modern=True)
    epoch.validate(m)
    assert epoch.check_pins(m) == {"same_training_runtime": True}
    recipe["producer_sha256"][str(tmp_path / "scripts/bt4_value_rewrite.py")] = (
        epoch.VALUE_PRODUCER_PINS["scripts/bt4_value_rewrite.py"]
    )
    derived["value_target_postprocess"] = {
        k: v for k, v in recipe.items() if k != "outputs"
    }
    if profile == "B100V50":
        with pytest.raises(ValueError, match="producer"):
            epoch.check_pins(m)
    else:
        assert epoch.check_pins(m) == {"same_training_runtime": True}
    files[m["data_qualification"]["path"]]["profile"] = (
        "B100V10" if profile == "B100V50" else "B100V50"
    )
    with pytest.raises(ValueError, match="data qualification"):
        epoch.check_pins(m)
