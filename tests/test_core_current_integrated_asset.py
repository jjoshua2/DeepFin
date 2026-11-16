"""Fictional current-I metadata graphs; no checkpoint/model/native reads.

Receipt shapes follow accepted current-I aggregate fields. These tests do not
issue authority, qualify the represented model, or reopen historical source.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_holdout_ruler as ruler

A39 = "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0"
STATE = "9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229"
HISTORICAL_ASSET_SOURCE = "b" * 40
ARCHITECTURE: dict[str, Any] = {'_schema_version': 20, 'arc_attention_bias': 'none', 'aux_policy_head_dim': 128, 'categorical_head_coupled': True, 'dynamic_relation_count': 5, 'embed_dim': 512, 'embed_dim_by_layer': None, 'enable_policy_sf_head': True, 'ffn_mult': 1.5, 'ffn_mult_by_layer': [1.5, 1.5, 1.5, 1.5, 1.5, 1.555556, 1.65, 1.772222, 1.894444, 1.666667, 1.638889, 1.905556, 1.783333, 1.794444, 1.744444, 1.5], 'history_rep_fix': True, 'input_extra_features': 'v2_threats', 'input_global_embedding': 'none', 'input_global_embedding_channels': 0, 'input_history_encoding': 'lc0_root_legacy_meta', 'input_pos_encoding': 'arc_adapter', 'input_square_embedding': 'none', 'kind': 'transformer', 'num_heads': 16, 'num_layers': 16, 'phase_output_adapter': False, 'phase_output_adapter_dim': 64, 'phase_piece_thresholds': [13, 22], 'phase_smolgen': False, 'policy_dynamic_relations': False, 'policy_embedding_mode': 'linear', 'policy_encoding': 'lc0_1858', 'qkv_projection': 'split', 'smolgen_bias_norm': 'center_rms', 'smolgen_bias_scale': 'layer_head', 'smolgen_gen_sz': 256, 'smolgen_hidden_channels': 32, 'smolgen_hidden_sz': 256, 'smolgen_mode': 'per_layer', 'smolgen_pooling': 'flatten', 'smolgen_relation_basis': True, 'smolgen_relation_coeff_norm': 'none', 'smolgen_relation_norm': 'none', 'smolgen_relation_scale': 'none', 'use_deepnorm': True, 'use_dynamic_relations': False, 'use_gradient_checkpointing': False, 'use_nla': False, 'use_qk_rmsnorm': False, 'use_smolgen': True}


def seal(path: Path, body: Any) -> dict[str, str]:
    return ruler.write(path, body)


def current_integrated_graph(tmp_path: Path, fault: str = "") -> dict[str, Any]:
    protocol = seal(tmp_path / "asset-protocol.json", {"fictional": True})
    terminal = seal(tmp_path / "asset-terminal.json", {"fictional_closed_owner": True})
    pins = seal(tmp_path / "asset-source-pins.json", {"fictional_historical_source": HISTORICAL_ASSET_SOURCE})
    other = seal(tmp_path / "other-reference.json", {"fictional_other": True})
    source: dict[str, Any] = {"revision": HISTORICAL_ASSET_SOURCE, "checkout": "/fiction/historical-current-I", "source_pins": pins}
    checkpoint = {"path": str(tmp_path / "never-created-original-model.pt"), "sha256": A39}
    observed: dict[str, Any] = {"actual_CPU_model_only_load": True, "path": checkpoint["path"], "sha256": A39,
        "architecture": copy.deepcopy(ARCHITECTURE), "embedded_step": 680276, "state_sha256": STATE,
        "state_tensors": 494, "optimizer_restore_qualified": False,
        "counts": {"input_planes": 175, "state_entries": 494, "unique_parameters": 61444448, "naive_state_elements": 77173088}}
    if fault == "counts":
        observed["counts"]["input_planes"] = 174
    elif fault == "state_tensors":
        observed["state_tensors"] = 493
    elif fault == "optimizer":
        observed["optimizer_restore_qualified"] = True
    elif fault == "CPU_load":
        observed["actual_CPU_model_only_load"] = False
    elif fault == "all_state_hashes":
        observed["state_sha256"] = "e" * 64
    common: dict[str, Any] = {"protocol": protocol, "source_revision": HISTORICAL_ASSET_SOURCE,
                            "source_checkout": source["checkout"], "source_pins": pins}
    readout: dict[str, Any] = {**common, "status": "COMPLETE_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_QUALIFICATION_CHILD_ROOT_CLOSURE_REQUIRED_V1",
                              "donor_model_only": copy.deepcopy(observed)}
    if fault == "readout_status":
        readout["status"] = "COMPLETE_COLLECTOR_CURRENT_H_CPU_ASSET_QUALIFICATION_CHILD_ROOT_CLOSURE_REQUIRED_V1"
    readout_ref = seal(tmp_path / "asset-readout.json", readout)
    root: dict[str, Any] = {**common, "status": "ROOT_ACCEPTS_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_V1",
                           "readout": readout_ref, "terminal": terminal, "donor_model_only": copy.deepcopy(observed)}
    if fault == "root_status":
        root["status"] = "ROOT_ACCEPTS_COLLECTOR_CURRENT_H_CPU_ASSET_READINESS_V1"
    elif fault == "root_source_pins":
        root["source_pins"] = other
    elif fault == "root_observation":
        root["donor_model_only"]["state_tensors"] = 493
    root_ref = seal(tmp_path / "asset-root.json", root)
    review: dict[str, Any] = {**common, "status": "ACCEPT_CLOSED_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_NO_FINDINGS_V1",
                             "root_acceptance": root_ref, "readout": readout_ref, "terminal": terminal, "findings": []}
    if fault == "ind_status":
        review["status"] = "ACCEPT_CLOSED_COLLECTOR_CURRENT_H_CPU_ASSET_READINESS_NO_FINDINGS_V1"
    elif fault in {"ind_root", "ind_readout", "ind_protocol", "ind_terminal"}:
        review[{"ind_root": "root_acceptance", "ind_readout": "readout", "ind_protocol": "protocol", "ind_terminal": "terminal"}[fault]] = other
    elif fault == "findings":
        review["findings"] = ["fictional unresolved finding"]
    review_ref = seal(tmp_path / "asset-review.json", review)
    donor: dict[str, Any] = {"schema": "core_a39_current_integrated_asset_v1", "checkpoint": checkpoint,
        "architecture": copy.deepcopy(ARCHITECTURE), "step": 680276, "model_state_sha256": STATE,
        "root": root_ref, "independent": review_ref, "asset_source": copy.deepcopy(source)}
    if fault == "versionless":
        del donor["schema"]
    elif fault == "unknown_schema":
        donor["schema"] = "core_a39_unknown_asset_v1"
    elif fault == "extra_key":
        donor["relax_legacy"] = True
    elif fault == "source_revision":
        donor["asset_source"]["revision"] = "c" * 40
    elif fault == "source_checkout":
        donor["asset_source"]["checkout"] = "/fiction/new-consumer"
    elif fault == "source_pins":
        donor["asset_source"]["source_pins"] = other
    elif fault == "checkpoint_hash":
        donor["checkpoint"]["sha256"] = "e" * 64
    elif fault == "step":
        donor["step"] = 680275
    elif fault == "float_step":
        donor["step"] = 680276.0
    elif fault == "actor_path":
        donor["checkpoint"]["path"] = str(tmp_path / "substituted-model.pt")
    elif fault == "architecture":
        donor["architecture"]["input_extra_features"] = "v1"
    elif fault in {"model_state", "all_state_hashes"}:
        donor["model_state_sha256"] = "e" * 64
    return donor


def test_current_integrated_A39_metadata_preserves_historical_source_without_payload_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    donor = current_integrated_graph(tmp_path)
    original = copy.deepcopy(donor)
    original_open = Path.open
    def no_checkpoint_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert path.suffix != ".pt", "metadata API attempted a checkpoint read"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_checkpoint_open)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    identity = ruler.asset_identity(metadata, donor)
    metadata.finish()
    assert identity == {**donor, "label": "A39", "tag": None}
    assert donor == original
    assert identity["asset_source"]["revision"] == HISTORICAL_ASSET_SOURCE
    assert identity["asset_source"]["checkout"] == "/fiction/historical-current-I"
    assert not Path(donor["checkpoint"]["path"]).exists()


@pytest.mark.parametrize("fault", ["versionless", "unknown_schema", "extra_key", "root_status", "ind_status", "readout_status",
    "source_revision", "source_checkout", "source_pins", "root_source_pins", "ind_root", "ind_readout", "ind_protocol",
    "ind_terminal", "findings", "root_observation", "checkpoint_hash", "step", "float_step", "actor_path", "architecture",
    "model_state", "all_state_hashes", "counts", "state_tensors", "optimizer", "CPU_load"])
def test_rehashed_current_integrated_A39_metadata_refusals(tmp_path: Path, fault: str):
    donor = current_integrated_graph(tmp_path, fault)
    if fault == "versionless":
        expected = "current-H A39 asset acceptance differs"
    elif fault in {"unknown_schema", "extra_key"}:
        expected = "invalid current-integrated A39 binding"
    elif fault in {"root_status", "ind_status", "ind_root", "ind_readout", "ind_protocol", "ind_terminal", "findings"}:
        expected = "current-integrated A39 asset acceptance differs"
    elif fault in {"readout_status", "source_revision", "source_checkout", "source_pins", "root_source_pins"}:
        expected = "current-integrated asset source/readout differs"
    elif fault in {"root_observation", "CPU_load"}:
        expected = "current-integrated model-only observation differs"
    elif fault in {"counts", "state_tensors", "optimizer"}:
        expected = "complete current-integrated A39 observation differs"
    elif fault == "checkpoint_hash":
        expected = "not original A39 donor"
    else:
        expected = "current-integrated A39 observed identity differs"
    with pytest.raises(ValueError, match=expected):
        ruler.asset_identity(ruler.Metadata(1 << 20, 8 << 20), donor)
