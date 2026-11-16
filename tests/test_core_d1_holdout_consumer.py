"""Fictional D1 authority graphs and real tiny scoring; no study execution.

Existing accepted-bank authority is an inert upstream token in selection tests;
all NEW D1 delivery/owner/prefix/profile/DEV/winner predicates run unmocked.
The fixture design hash is the caller-trusted synthetic issuer anchor.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_holdout_metrics as metrics
from scripts import core_generation_holdout_read as reader
from scripts import core_generation_holdout_ruler as ruler
from scripts import core_generation_holdout_score as score
from tests.test_core_same_gpu_payload_delivery import gpu_graph, seal


def owner(tmp: Path, name: str, protocol: dict[str, str], helper: dict[str, str], review: dict[str, str], resource: str, seconds: int, role: bool = False) -> dict[str, Any]:
    suffix = [] if role else ["-m", "scripts.fictional_d1_gpu" if resource == "gpu" else "scripts.fictional_d1_cpu"]
    plan_key, plan_flag = ("protocol", "--protocol") if role else ("development_plan", "--development-plan")
    path = tmp / (name + "-admission.json")
    command = ["/fiction/python", "-I", "-B", helper["path"], plan_flag, protocol["path"],
               plan_flag + "-sha256", protocol["sha256"], "--admission", str(path),
               "--admission-sha256", "SELF_ADMISSION_SHA256", *suffix]
    status = "ROOT_ADMITS_KL_DEVELOPMENT_FIRST_D1_PREFIX_CPU_V1" if role else "ROOT_ADMITS_D1_DEVELOPMENT_GPU_V1" if resource == "gpu" else "ROOT_ADMITS_D1_DEVELOPMENT_READER_CPU_V1"
    admission = seal(path, {"status": status, "execution_authorized": True, "normalized_child": copy.deepcopy(command),
        plan_key: protocol, "helper": helper, "source_review": review, "resource": resource,
        "seconds": seconds, "source_pins": {"path": "/fiction/role-pins.json", "sha256": "a" * 64}, "owner": {"sha256": "e" * 64}})
    command[command.index("--admission-sha256") + 1] = admission["sha256"]
    terminal = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": resource, "seconds": seconds, "wall_seconds": 1.0, "command": command,
        "source_revision": "f" * 40, "cwd": "/fiction/owner", "supervisor_sha256": "e" * 64,
        "supervisor_pid": 2147483645, "known_processes": {"2147483646": 1}}
    return {"terminal": seal(tmp / (name + "-terminal.json"), terminal), "admission": admission, "command": command,
        "source_revision": "f" * 40, "cwd": "/fiction/owner", "closure": {"supported_eof": True, "exit_code": 0,
        "supervisor_pid": 2147483645, "supervisor_closed": True, "owned_births_closed": True, "known_processes": terminal["known_processes"]}}


def original_role_graph(tmp: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    original_books = {name: {"path": str(tmp / (name + ".epd")), "sha256": "1" * 64}
                      for name in ("SCREEN_C_A", "SCREEN_C_B", "SCREEN_C_A39", "CONFIRM_C_A", "CONFIRM_C_B", "CONFIRM_C_A39")}
    original_roles = {name: {"path": str(tmp / (name + ".full-role.json")), "sha256": "2" * 64} for name in original_books}
    original = seal(tmp / "original-roles.json", {"status": "ROOT_ACCEPTS_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_V1", "books": original_books, "roles": original_roles})
    independent = seal(tmp / "original-roles-review.json", {"status": "ACCEPT_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_NO_FINDINGS_V1", "root_acceptance": original, "findings": []})
    return original_books, original_roles, original, independent


def role_graph(tmp: Path, design: dict[str, Any], dref: dict[str, str], _source: dict[str, Any], fault: str) -> dict[str, str]:
    original, original_ind = design["roles"]["root"], design["roles"]["independent"]
    old = json.loads(Path(original["path"]).read_bytes())
    original_books, original_roles = old["books"], old["roles"]
    prefixes: dict[str, Any] = {}
    for key, role in (("vs_A39", "SCREEN_C_A"), ("vs_K0", "SCREEN_C_B")):
        design["DEV"]["panel_" + key] = {"source_role": role, "source_book": original_books[role]}
        prefixes[role] = {"source_role": role, "original400_book": original_books[role], "original_full_role": original_roles[role],
            "book": {"path": str(tmp / (key + "-80.epd")), "sha256": "3" * 64},
            "parent_map": {"path": str(tmp / (key + "-80.json")), "sha256": "4" * 64},
            "selection": "exact original ordinals0..79; no rng/subsample/reorder",
            "prefix_start": 1 if fault == "prefix" else 0, "pairs": 80, "complete_parents": 80,
            "source_count": 400, "actual_public_loader_count": 80, "no_skips_dedup_shrink": True,
            "parent_order_sha256": "5" * 64, "history_order_sha256": "6" * 64}
    helper = {"path": "/fiction/role-helper.py", "sha256": "8" * 64}
    pins = {"path": "/fiction/role-pins.json", "sha256": "a" * 64}
    controls = {"path": "/fiction/role-controls.py", "sha256": "b" * 64}
    sr = seal(tmp / "role-source-review.json", {"status": "ACCEPT_KL_DEVELOPMENT_FIRST_D1_PREFIX_SOURCE_ONLY_NO_FINDINGS_V1", "helper": helper, "controls": controls, "source_pins": pins, "findings": []})
    protocol = seal(tmp / "role-protocol.json", {"schema": "D1_PREFIX_METADATA_CPU_PROTOCOL_V1", "status": "BOUND_D1_PREFIX_METADATA_ONLY_V1", "design": dref,
        "study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469", "role_root": original, "role_independent": original_ind,
        "limits": {"outer_seconds": 300, "inner_seconds": 270}, "helper": helper, "controls": controls, "source_review": sr, "source_pins": pins})
    consumed = [{"book": book, "pairs": count, "actual_public_reads": 1, "full16_history_order_EPD_exact": True}
                for book, count in [(ref, 400) for ref in original_books.values()] + [(value["book"], 80) for value in prefixes.values()]]
    readout = seal(tmp / "role-readout.json", {"status": "D1_PREFIX_METADATA_CANDIDATE_PENDING_ROOT_CLOSURE_V1", "protocol": protocol, "design": dref,
        "role_root": original, "role_independent": original_ind, "prefixes": prefixes, "original_six400_books": original_books,
        "all8role_domains_disjoint": {"group": 2656, "parent": 2657, "history": 2656, "epd": 2656, "history_sha": 2656, "input": 5312}, "complete_parent_descriptors_preserved": True,
        "controls_passed": 12, "public_consumption": consumed})
    root = {"study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469", "design": dref, "source_roles": design["roles"],
        "prefixes": prefixes, "original_six400_books": original_books, "status": "ROOT_ACCEPTS_CLOSED_KL_DEVELOPMENT_FIRST_D1_PREFIX_METADATA_V1",
        "protocol": protocol, "readout": readout, "owner": owner(tmp, "roles", protocol, helper, sr, "cpu", 300, role=True)}
    rref = seal(tmp / "roles-root.json", root)
    return {"root": rref, "independent": seal(tmp / "roles-review.json", {**root, "status": "ACCEPT_CLOSED_KL_DEVELOPMENT_FIRST_D1_PREFIX_METADATA_NO_FINDINGS_V1", "root_acceptance": rref, "findings": []})}


def edge(tmp: Path, label: str, reference: str, model_refs: dict[str, Any], states: dict[str, str], dref: dict[str, str],
         source: dict[str, Any], profile: dict[str, str], roles: dict[str, str], fault: str, wins: int) -> dict[str, str]:
    name = label + "_vs_" + reference
    helper = {"path": "/fiction/dev-helper.py", "sha256": "8" * 64}
    sr = seal(tmp / (name + "-gpu-source.json"), {"status": "ACCEPT_D1_DEVELOPMENT_GPU_SOURCE_ONLY_NO_FINDINGS_V1", "helper": helper, "source": source, "findings": []})
    cr = seal(tmp / (name + "-cpu-source.json"), {"status": "ACCEPT_D1_DEVELOPMENT_READER_SOURCE_ONLY_NO_FINDINGS_V1", "helper": helper, "source": source, "findings": []})
    role_root = json.loads(Path(roles["root"]["path"]).read_bytes())
    panel = "vs_A39" if reference == "A39" else "vs_K0"
    active = json.loads(Path(profile["path"]).read_bytes())["active_playing_projection"]
    common = {"study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469", "design": dref, "candidate": label, "reference": reference,
        "models": {"candidate": model_refs[label], "reference": model_refs[reference]}, "profile": profile, "role_use": roles, "source": source}
    protocol = seal(tmp / (name + "-protocol.json"), {**common, "schema": "CORE_POLICY_ANCHOR_D1_DEVELOPMENT_EDGE_V1", "panel": panel,
        "seed": 2026111674 if reference == "A39" else 2026111675, "opening_book": role_root["prefixes"]["SCREEN_C_A" if panel == "vs_A39" else "SCREEN_C_B"]["book"],
        "outer_seconds": 900, "inner_seconds": 870, "active_playing_projection": copy.deepcopy(active), "public_suffix": ["-m", "scripts.fictional_d1_gpu"], "helper": helper, "source_review": sr})
    counts = [80 - wins, 0, 0, 0, wins]
    evidence: dict[str, Any] = {**common, "pairs_completed": 80, "games_completed": 160, "incomplete_pairs": 0, "illegal_or_crashed_games": 0,
        "reciprocal_colours_complete": True, "paired_counts": counts, "paired_score": wins / 80, "wdl_counts": {"wins": wins * 2, "draws": 0, "losses": (80 - wins) * 2},
        "effective_settings": ruler.D1_MATCH_SETTINGS, "served_states": {side: {"pre": states[arm], "post": states[arm]} for side, arm in (("candidate", label), ("reference", reference))}}
    if fault == "served" and label == "K001" and reference == "A39":
        evidence["served_states"]["candidate"]["post"] = "0" * 64
    active = json.loads(Path(profile["path"]).read_bytes())["active_playing_projection"]
    evidence["observed_playing_projection"] = {"candidate": active, "reference": copy.deepcopy(active)}
    evidence["actual_search_objects_checked"] = evidence["actual_runner_arguments_checked"] = True
    if fault == "profile" and label == "K001" and reference == "A39":
        evidence["observed_playing_projection"]["reference"]["gumbel"]["gumbel_scale"] = 1.0
    readout = seal(tmp / (name + "-readout.json"), {**evidence, "status": "D1_DEVELOPMENT_EDGE_PENDING_ROOT_CLOSURE_V1"})
    cpu = seal(tmp / (name + "-reader-protocol.json"), {"schema": "CORE_POLICY_ANCHOR_D1_DEVELOPMENT_READER_V1", "protocol": protocol,
        "readout": readout, "outer_seconds": 300, "inner_seconds": 270, "public_suffix": ["-m", "scripts.fictional_d1_cpu"], "helper": helper, "source_review": cr})
    summary = {**evidence, "status": "D1_DEVELOPMENT_EDGE_RECONCILED_PENDING_ROOT_CLOSURE_V1", "protocol": protocol, "readout": readout,
        "pair_order_verified": True, "all_game_rows_reconciled": True, "pair_order_sha256": "5" * 64, "history_order_sha256": "6" * 64,
        "game_log": {"path": str(tmp / (name + ".jsonl")), "sha256": "a" * 64}}
    if fault == "point" and label == "K001" and reference == "A39":
        summary["paired_score"] = 0.99
    root = {**common, "status": ruler.D1_EDGE_ROOT, "protocol": protocol, "reader_protocol": cpu, "readout": readout,
        "summary": seal(tmp / (name + "-summary.json"), summary), "gpu_owner": owner(tmp, name + "gpu", protocol, helper, sr, "gpu", 900),
        "reader_owner": owner(tmp, name + "cpu", cpu, helper, cr, "cpu", 300)}
    rref = seal(tmp / (name + "-root.json"), root)
    return {"root": rref, "independent": seal(tmp / (name + "-review.json"), {**root, "status": ruler.D1_EDGE_REVIEW, "root_acceptance": rref, "findings": []})}


def selection_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str = "", baseline_wins: int = 48, holdout_refs: dict[str, Any] | None = None) -> tuple[ruler.Metadata, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source: dict[str, Any] = {"revision": "d" * 40, "checkout": "/fiction/dev-source", "files": {"scripts/arena_standard.py": "a" * 64}}
    original_books, _original_roles, original, original_ind = original_role_graph(tmp_path)
    design: dict[str, Any] = {"study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469", "design_id": "D1",
        "foundations": {"TRAIN_bank": {"root": {"path": "/fiction/bank-root.json", "sha256": "1" * 64}, "independent": {"path": "/fiction/bank-review.json", "sha256": "2" * 64}},
            "HOLD_bank": holdout_refs if holdout_refs is not None else {"root": {"path": "/fiction/HOLD-root.json", "sha256": "3" * 64}, "independent": {"path": "/fiction/HOLD-ind.json", "sha256": "4" * 64}}},
        "roles": {"root": original, "independent": original_ind},
        "DEV": {"panel_" + key: {"source_role": role, "source_book": original_books[role]} for key, role in (("vs_A39", "SCREEN_C_A"), ("vs_K0", "SCREEN_C_B"))},
        "final": {key: {"source_role": role, "source_book": original_books[role]} for key, role in (("vs_A39", "CONFIRM_C_A39"), ("vs_K0_if_positive_KL", "CONFIRM_C_A"))}}
    dref = seal(tmp_path / "design-anchor.json", design)
    monkeypatch.setattr(ruler, "D1_DESIGN_SHA256", dref["sha256"])
    roles = role_graph(tmp_path, design, dref, source, fault)
    arms: dict[str, Any] = {}
    trained: dict[str, Any] = {}
    states: dict[str, str] = {}
    model_refs: dict[str, Any] = {}
    donor: dict[str, Any] | None = None
    for label in ruler.D1_COEFFICIENTS:
        folder = tmp_path / label
        folder.mkdir()
        arms[label], train = gpu_graph(folder, arm=label, d1=True, design_ref=dref)
        train = ruler.bank.AcceptedGeneration(identity=train.identity, games=train.games, evidence={"eligible_rows": 17568})
        trained[label] = train
        root = json.loads(Path(arms[label]["root"]["path"]).read_bytes())
        contract = json.loads(Path(root["contract"]["path"]).read_bytes())
        model_refs[label], states[label] = root["checkpoint"], "3" * 64
        donor = {"checkpoint": contract["initial_model"], "step": 680276, "architecture": contract["architecture"], "model_state_sha256": "9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229"}
    assert donor is not None
    model_refs["A39"], states["A39"] = donor["checkpoint"], donor["model_state_sha256"]
    # Isolate the already-tested caller-trusted bank authority; new proof and D1 predicates remain real.
    monkeypatch.setattr(ruler, "training_population", lambda _m, model: trained[model["label"]])
    # Explicit fictional descriptor trust anchors exercise the real projection predicate.
    full_value = {'configs': {'game': {'categorical_blend_frac': 0.0,
                          'categorical_search_blend_frac': 0.0,
                          'history_rep_fix': True,
                          'input_extra_features': 'v2_threats',
                          'input_history_encoding': 'lc0_root_legacy_meta',
                          'max_plies': 450,
                          'policy_encoding': 'lc0_1858',
                          'record_relations': False,
                          'search_wdl_draw_mode': 'net_raw',
                          'syzygy_adjudicate': True,
                          'syzygy_adjudicate_fraction': 1.0,
                          'syzygy_in_search': True,
                          'syzygy_outcome_mode': 'rule50_match_v1',
                          'syzygy_path': '/fiction/tablebases',
                          'syzygy_rescore_policy': True},
                 'search': {'gumbel_c_scale': 0.1,
                            'gumbel_policy_temp': 1.0,
                            'gumbel_scale': 0.75,
                            'gumbel_scale_after': 0.75,
                            'gumbel_target_batch': 0,
                            'gumbel_target_max_visit_cap': 0,
                            'gumbel_target_untempered_prior': False,
                            'gumbel_topk': 16,
                            'gumbel_vloss_weight': 1,
                            'simulations': 64,
                            'volatility_anchor': 0.05,
                            'volatility_fpu': 0.0,
                            'volatility_q_scale': 0.0},
                 'temp': {'after': 0.1,
                          'endgame': 0.1,
                          'selfplay_endgame': 0.1,
                          'selfplay_temperature': 0.1,
                          'temperature': 0.1}}}
    full_profile = seal(tmp_path / "full-profile.json", full_value)
    selected_ref = seal(tmp_path / "profile-selection.json", {"status": "ROOT_SELECTS_FINAL_PROSPECTIVE_COLLECTOR_PROFILE_AND_ROLES_SOURCE_ONLY_V1", "selected_profile": full_profile})
    monkeypatch.setattr(ruler, "D1_PROFILE_SHA256", full_profile["sha256"])
    monkeypatch.setattr(ruler, "D1_PROFILE_SELECTION_SHA256", selected_ref["sha256"])
    profile = seal(tmp_path / "profile.json", {"schema": "D1_MATCH_PROFILE_V1", "design": dref, "effective_settings": ruler.D1_MATCH_SETTINGS,
        "full_profile": full_profile, "selection_authority": selected_ref, "active_playing_projection": ruler.d1_playing_projection(full_value),
        "source_only_not_applicable": ruler.d1_collector_only_profile(full_value)})
    edges = {}
    for label in ruler.D1_COEFFICIENTS:
        wins = baseline_wins if label == "K0" else 52 if label in ("K001", "K01") else 44
        if fault == "none":
            wins = 40
        edges[label + "_vs_A39"] = edge(tmp_path, label, "A39", model_refs, states, dref, source, profile, roles, fault, wins)
    for label in ("K001", "K01", "K1"):
        edges[label + "_vs_K0"] = edge(tmp_path, label, "K0", model_refs, states, dref, source, profile, roles, fault, 48 if fault != "none" else 40)
    winner = "K0" if baseline_wins > 52 else "K001"
    if fault == "missing":
        del edges["K1_vs_K0"]
    if fault == "winner":
        winner = "K1"
    common = {"study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469", "design": dref, "arms": arms, "dev_edges": edges,
              "selected_arm": winner, "profile": profile, "role_use": roles, "dev_source": source}
    root = seal(tmp_path / "selection-root.json", {**common, "status": ruler.D1_SELECTION_ROOT})
    review = seal(tmp_path / "selection-review.json", {**common, "status": ruler.D1_SELECTION_REVIEW, "root_acceptance": root, "findings": []})
    return ruler.Metadata(1 << 20, 64 << 20), {"study_uuid": common["study_uuid"], "design": dref, "selected_arm": winner, "d1": {"root": root, "independent": review}}, donor, design


@pytest.mark.parametrize("baseline_wins", [48, 56])
def test_D1_complete_DEV_selects_one_and_ties_choose_lower_coefficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_wins: int):
    metadata, protocol, donor, design = selection_graph(tmp_path, monkeypatch, baseline_wins=baseline_wins)
    selected, models = ruler.d1_selection(metadata, protocol, donor, design)
    assert selected == ("K001" if baseline_wins == 48 else "K0")
    assert set(models) == {"A39", "K0", "K001", "K01", "K1"}
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [("missing", "D1 all seven DEV edges required"), ("winner", "D1 winner differs"),
    ("none", "D1 no eligible DEV arm"), ("served", "D1 DEV served state differs"), ("point", "D1 DEV paired score differs"),
    ("prefix", "D1 exact prefix materialization differs"), ("profile", "D1 DEV active playing projection differs")])
def test_D1_rehashed_selection_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str):
    metadata, protocol, donor, design = selection_graph(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match=message):
        ruler.d1_selection(metadata, protocol, donor, design)


@pytest.mark.parametrize("selected", ["K0", "K001", "K01", "K1"])
def test_D1_real_tiny_public_loading_scoring_and_streaming_only_selected_labels(tmp_path: Path, selected: str):
    from tests.test_core_generation_holdout_ruler import synthetic_models, synthetic_games, LIMITS

    original, architecture = synthetic_models(tmp_path)
    view = synthetic_games(tmp_path, architecture)
    labels = metrics.d1_labels(selected)
    models = {"A39": original["A39"], "K0": original["A"]}
    if selected != "K0":
        models[selected] = original["C"]
    games = score.load_games(view, LIMITS)
    parents = {game["physical_game_id"]: "parent-" + str(i) for i, (game, _) in enumerate(games)}
    output = tmp_path / "D1-observations"
    output.mkdir()
    result = score.score_games(games, parents, models, output, device="cpu", autocast=False, limits=LIMITS, labels=labels)
    result.update(eligible_games=len(games), eligible_rows=sum(game["rows"] for game, _ in games))
    assert set(result["models"]) == set(labels)
    assert reader.reconcile_rows(result, view, parents, LIMITS["row_log_bytes"], labels=labels) == result["games"]


@pytest.mark.parametrize("selected", ["K0", "K001", "K01", "K1"])
def test_D1_equal_parent_bootstrap_uses_selected_contrast_and_local_rng(selected: str):
    labels = metrics.d1_labels(selected)
    games: list[dict[str, Any]] = []
    for i in range(48):
        rows = 40 if i % 2 == 0 else 60
        difference = 0.2 if i % 2 == 0 else 0.4
        games.append({"physical_game_id": "g" + str(i), "parent_id": "p" + str(i), "rows": rows, "row_identity_sha256": "a" * 64,
            "models": {label: {metric: rows * (0.1 + difference if label == selected else 0.1) for metric in metrics.METRICS}
                       for label in labels}})
    health = metrics.parent_health(games, labels=labels, selected_label=selected)
    assert health[selected + "_minus_A39_equal_parent_WDL_CE"]["point"] == pytest.approx(0.3)
    assert health["severe_damage_veto"] is True
    assert health["bootstrap"]["degenerate"] is False
    assert health["aggregate"]["models"][selected]["row_weighted"]["wdl_ce"] == pytest.approx(0.42)


@pytest.mark.parametrize("arm", ["K0", "K001", "K01", "K1"])
def test_D1_exact3600_proof_accepts_fixed4096_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str):
    dref = seal(tmp_path / "D1-design.json", {"fictional_D1_trust_anchor": True})
    monkeypatch.setattr(ruler, "D1_DESIGN_SHA256", dref["sha256"])
    binding, train = gpu_graph(tmp_path, arm=arm, d1=True, design_ref=dref)
    metadata = ruler.Metadata(1 << 20, 16 << 20)
    model = ruler.model_delivery(metadata, binding, arm, _d1=True)
    ruler.same_gpu_game_coverage(metadata, binding, model, train, _d1=True)
    assert model["step"] == 4096
    assert model["boundary"]["drawn_rows"] == 131072
    assert sum(item[2] for item in model["boundary"]["membership"]) == 17568
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [("cap", "owner cap/wall differs"), ("helper", "unsupported same-GPU prover"),
    ("source_review", "same-GPU prover source acceptance differs"), ("N", "same-GPU draw totals differ"),
    ("checkpoint", "same-GPU saved-state proof incomplete"), ("rng", "same-GPU RNG/membership/backend differs"),
    ("thread_intra", "same-GPU thread-2 evidence differs"), ("thread_backend_interop", "same-GPU thread-2 evidence differs")])
def test_D1_rehashed_proof_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str):
    dref = seal(tmp_path / "D1-design.json", {"fictional_D1_trust_anchor": True})
    monkeypatch.setattr(ruler, "D1_DESIGN_SHA256", dref["sha256"])
    binding, _ = gpu_graph(tmp_path, fault, arm="K0", d1=True, design_ref=dref)
    with pytest.raises(ValueError, match=message):
        ruler.model_delivery(ruler.Metadata(1 << 20, 16 << 20), binding, "K0", _d1=True)


@pytest.mark.parametrize("threaded", [False, True])
def test_D1_cannot_adopt_legacy900_proof(tmp_path: Path, threaded: bool):
    binding, _ = gpu_graph(tmp_path, threaded=threaded)
    with pytest.raises(ValueError, match="D1 requires its versioned same-owner proof"):
        ruler.model_delivery(ruler.Metadata(1 << 20, 8 << 20), binding, "K0", _d1=True)


@pytest.mark.parametrize("labels", [("A39", "K1", "K0"), ("A39", "K0", "K001", "K01"), ("A39", "K01")])
def test_D1_rejects_unselected_or_wrong_model_order(labels: tuple[str, ...]):
    with pytest.raises(ValueError, match="unsupported HOLD model order"):
        metrics.validate_labels(labels)


@pytest.mark.parametrize("fault", ["list", "missing", "extra", "bool", "float", "negative", "parent_small", "input"])
def test_D1_actual_prefix_domain_dictionary_refusals(tmp_path: Path, fault: str) -> None:
    books, _roles, original, independent = original_role_graph(tmp_path)
    design: dict[str, Any] = {"roles": {"root": original, "independent": independent},
        "DEV": {"panel_" + key: {"source_role": role, "source_book": books[role]} for key, role in (("vs_A39", "SCREEN_C_A"), ("vs_K0", "SCREEN_C_B"))},
        "final": {key: {"source_role": role, "source_book": books[role]} for key, role in (("vs_A39", "CONFIRM_C_A39"), ("vs_K0_if_positive_KL", "CONFIRM_C_A"))}}
    dref = seal(tmp_path / "domain-design.json", design)
    binding = role_graph(tmp_path, design, dref, {}, "")
    root = json.loads(Path(binding["root"]["path"]).read_bytes())
    report = json.loads(Path(root["readout"]["path"]).read_bytes())
    domains = report["all8role_domains_disjoint"]
    if fault == "list":
        report["all8role_domains_disjoint"] = list(domains)
    elif fault == "missing":
        del domains["parent"]
    elif fault == "extra":
        domains["unknown"] = 1
    else:
        key, value = {"bool": ("parent", True), "float": ("parent", 2657.0), "negative": ("parent", -1),
                      "parent_small": ("parent", 2655), "input": ("input", 5311)}[fault]
        domains[key] = value
    # Rehash the complete changed graph; errors must reach the domain guard.
    root["readout"] = seal(tmp_path / "faulty-domain-readout.json", report)
    root_ref = seal(tmp_path / "faulty-domain-root.json", root)
    review = seal(tmp_path / "faulty-domain-review.json", {**root, "status": "ACCEPT_CLOSED_KL_DEVELOPMENT_FIRST_D1_PREFIX_METADATA_NO_FINDINGS_V1", "root_acceptance": root_ref, "findings": []})
    with pytest.raises(ValueError, match=r"^D1 actual prefix domain counts differ$"):
        ruler._d1_role_panels(ruler.Metadata(1 << 20, 16 << 20), {"root": root_ref, "independent": review},
            {"study_uuid": root["study_uuid"], "design": dref}, design)


@pytest.mark.parametrize("baseline_wins", [48, 56])
@pytest.mark.parametrize("extra_model", [False, True])
def test_D1_public_binding_dispatch_keeps_real_selection_and_prefix_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_wins: int, extra_model: bool) -> None:
    # Only the established HOLD token and donor-asset API are inert upstream
    # authorities. Actual parent map, D1 bindings dispatch, saved proof, coverage,
    # prefix/profile, all seven DEV guards and selection execute directly.
    collection = "77777777-7777-4777-8777-777777777777"
    planned = [{"physical_game_id": collection + ":" + str(i), "bank": "holdout"} for i in range(64)]
    roster = seal(tmp_path / "public-HOLD-roster.json", planned)
    producer = seal(tmp_path / "public-HOLD-producer.json", {"roster": roster})
    audit = seal(tmp_path / "public-HOLD-audit.json", {"producer_protocol": producer})
    hroot = seal(tmp_path / "public-HOLD-root.json", {"audit_protocol": audit})
    hind = seal(tmp_path / "public-HOLD-independent.json", {"fictional_upstream_token": True})
    hbinding = {"root": hroot, "independent": hind}
    metadata, protocol, donor, _design = selection_graph(tmp_path, monkeypatch, baseline_wins=baseline_wins, holdout_refs=hbinding)
    games = tuple({"physical_game_id": row["physical_game_id"], "rows": 93 if i < 55 else 92} for i, row in enumerate(planned))
    view = ruler.hold.AcceptedHoldout(identity={"bank": "holdout", "architecture": donor["architecture"], "collection_uuid": collection}, games=games, evidence={})
    monkeypatch.setattr(ruler.hold, "validate_accepted_holdout", lambda *_args, **_kwargs: view)
    monkeypatch.setattr(ruler, "asset_identity", lambda _m, binding: {**binding, "label": "A39", "tag": None})
    mapping = seal(tmp_path / "public-HOLD-parent-map.json", {"study_uuid": protocol["study_uuid"], "collection_uuid": collection,
        "games": [{"physical_game_id": row["physical_game_id"], "parent_id": "parent-" + str(i), "parent_member_ids": ["member-" + str(i)]} for i, row in enumerate(planned)]})
    parent_common = {"study_uuid": protocol["study_uuid"], "design": protocol["design"], "producer_protocol": producer, "producer_roster": roster, "parent_map": mapping}
    parents = seal(tmp_path / "public-HOLD-parents-root.json", {**parent_common, "status": ruler.ROSTER_ROOT})
    parent_review = seal(tmp_path / "public-HOLD-parents-review.json", {**parent_common, "status": ruler.ROSTER_REVIEW, "root_acceptance": parents, "findings": []})
    selected_root = json.loads(Path(protocol["d1"]["root"]["path"]).read_bytes())
    labels = metrics.d1_labels(protocol["selected_arm"])
    protocol.update(schema=ruler.D1_SCORE_SCHEMA, holdout=hbinding, parents={"root": parents, "independent": parent_review},
        models={"A39": {**donor, "schema": "core_a39_current_integrated_asset_v1"}, **{label: selected_root["arms"][label] for label in labels if label != "A39"}},
        limits={"metadata_file_bytes": 1 << 20, "metadata_total_bytes": 64 << 20})
    if extra_model:
        protocol["models"]["K1"] = selected_root["arms"]["K1"]
        with pytest.raises(ValueError, match=r"^D1 only selected/control/donor models allowed$"):
            ruler.bindings(metadata, protocol)
    else:
        returned, parent_ids, delivered = ruler.bindings(metadata, protocol)
        assert returned is view
        assert len(parent_ids) == 64
        assert set(delivered) == set(labels)
        assert sum(game["rows"] for game in returned.games) == 5943
        metadata.finish()


def test_D1_projection_separates_source_only_training_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _metadata, protocol, _donor, _design = selection_graph(tmp_path, monkeypatch)
    registry = json.loads(Path(protocol["d1"]["root"]["path"]).read_bytes())
    profile = json.loads(Path(registry["profile"]["path"]).read_bytes())
    source = json.loads(Path(profile["full_profile"]["path"]).read_bytes())
    active = ruler.d1_playing_projection(source)
    keys = ("categorical_blend_frac", "categorical_search_blend_frac", "syzygy_rescore_policy")
    assert not set(keys) & set(active["game"])
    classified = ruler.d1_collector_only_profile(source)
    assert classified == profile["source_only_not_applicable"]
    for key in keys:
        assert classified["configs/game/" + key] == {"selected_source_value": source["configs"]["game"][key],
            "playing_applicability": "NOT_APPLICABLE_LABEL_ONLY"}
    # Source-only perturbation is a classification control, not a new recipe:
    # these finalize-only label fields cannot change the arena projection.
    changed = copy.deepcopy(source)
    changed["configs"]["game"].update(categorical_blend_frac=0.7, categorical_search_blend_frac=0.8, syzygy_rescore_policy=False)
    assert ruler.d1_playing_projection(changed) == active
    assert ruler.d1_collector_only_profile(changed) != classified
