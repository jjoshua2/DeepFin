"""CPU-only boundary and native-encoding controls; fake owned TB, no model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import core_finite_admit as admit
from scripts import core_generation_data_admit as data_admit


def frozen(path: Path, value: Any) -> dict[str, str]:
    admit.learning.write_json(path, value)
    path.chmod(0o444)
    return admit.ref(path)


def test_closed_producer_owner_requires_exact_command_and_source(tmp_path: Path):
    protocol = {"path": str(tmp_path / "protocol.json"), "sha256": "a" * 64}
    admission = frozen(tmp_path / "admission.json", {"status": "ROOT_ADMITS_CORE_GENERATION_V1",
                        "protocol_sha256": protocol["sha256"]})
    command = ["python", "-m", "scripts.core_finite_collect", "--execute", "--protocol", protocol["path"],
               "--protocol-sha256", protocol["sha256"], "--admission", admission["path"], "--admission-sha256", admission["sha256"]]
    producer = {"owner": {"sha256": "b" * 64, "seconds": 120}}
    value = {"status": "PASS", "returncode": 0, "owned_processes_closed": True,
             "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "b" * 64,
             "command": command, "resource": "gpu", "seconds": 120,
             "known_processes": {"2147483647": 1},
             "source_revision": "wrapper-revision-is-separate-from-child"}
    binding = {"terminal": frozen(tmp_path / "terminal.json", value), "owner_sha256": "b" * 64, "command": command}
    assert admit.check_owner(binding, protocol, producer)["status"] == "ROOT_ADMITS_CORE_GENERATION_V1"
    value["owned_processes_closed"] = False
    binding["terminal"] = frozen(tmp_path / "unclosed.json", value)
    with pytest.raises(ValueError, match="closed PASS"):
        admit.check_owner(binding, protocol, producer)


@pytest.mark.parametrize("fault", ["games", "rows", "attempts"])
def test_gates_do_not_select_or_replace_games(fault):
    p = {"gates": {"attempts": 3, "min_resolved_games": 2, "min_total_rows": 4}}
    c = {"games_per_generation": 2, "max_games_per_generation": 3}
    producer = {"attempts": 3}
    eligible = [{"rows": 2}, {"rows": 2}, {"rows": 3}]
    admit.gates(p, c, producer, eligible)
    assert len(eligible) == 3  # Keep all eligible, not exactly the minimum2.
    if fault == "games":
        eligible = eligible[:1]
    elif fault == "rows":
        p["gates"]["min_total_rows"] = 8
    else:
        producer["attempts"] = 4
    with pytest.raises(ValueError, match=r"insufficient|contract mismatch"):
        admit.gates(p, c, producer, eligible)


def synthetic_game(tmp_path: Path, monkeypatch, corruption=None, diagnostics="inactive", *, generation: int = 1,
                   actor_sha: str = "a" * 64) -> SimpleNamespace:
    import chess
    import numpy as np
    from chess_anti_engine.core_generation_data import capture_raw_targets
    from chess_anti_engine.encoding import encode_cboard_for_model, rep_fix
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves import move_to_index, move_to_index_for_encoding
    from chess_anti_engine.replay.augment import _mirror_x_batch
    from chess_anti_engine.replay.sample import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays
    from chess_anti_engine.selfplay.config import GameConfig
    from chess_anti_engine.selfplay.finalize import _result_to_wdl, _stable_game_id
    from chess_anti_engine.selfplay.provenance import GameProvenance, GameStartProvenance, NetRecordProvenance
    from chess_anti_engine.selfplay import strict_tb
    from chess_anti_engine.verified_outcome import VerifiedOutcomeVerifier
    from scripts import core_finite_collect as collect

    rep_fix.apply(True)
    asset_sha = admit.digest([])

    owners = []

    class FakeOwner:
        def __init__(self, _path):
            self.asset_inventory_sha256 = asset_sha
            self.tablebase = SimpleNamespace(wdl={}, dtz={})
            self.closed = False
            owners.append(self)

        def resolve(self, board, *, move_offset):
            natural = board.outcome(claim_draw=True)
            assert natural is not None
            return strict_tb.TablebaseWitness(mode="rule50_match_v1", origin="NATURAL",
                root_fen=board.root().fen(), moves=tuple(move.uci() for move in board.move_stack),
                final_fen=board.fen(), move_offset=move_offset, turn=board.turn,
                halfmove_clock=board.halfmove_clock, wdl=None, dtz=None, status=-1,
                result=natural.result(), max_pieces=6, asset_inventory_sha256=asset_sha)

        def policy_move(self, board):
            move = chess.Move.from_uci("g6g7")
            assert move in board.legal_moves
            return move

        def close(self):
            self.closed = True

    monkeypatch.setattr(strict_tb, "StrictSelfplayTablebase", FakeOwner)
    run = "11111111-1111-4111-8111-111111111111"
    collection = "22222222-2222-4222-8222-222222222222"
    actor, served, source = actor_sha, "b" * 64, "c" * 40
    verifier = VerifiedOutcomeVerifier(path="/synthetic-tables", asset_inventory_sha256=asset_sha,
        collection_uuid=collection, run_uuid=run, generation=generation, source_revision=source,
        concurrent_slots=1, actor_sha256=actor, served_state_sha256=served)
    root = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    move = chess.Move.from_uci("g6g7")
    assert root.is_valid()
    assert move in root.legal_moves
    final = root.copy(stack=True)
    final.push(move)
    assert final.is_checkmate()
    game = GameConfig(selfplay_fraction=1, input_extra_features="v2_threats",
        input_history_encoding="lc0_root_legacy_meta", policy_encoding="lc0_1858", history_rep_fix=True,
        syzygy_outcome_mode="rule50_match_v1", syzygy_path="/synthetic-tables", stockfish_syzygy_path="/synthetic-tables",
        syzygy_in_search=True, syzygy_adjudicate=True, syzygy_adjudicate_fraction=1, syzygy_rescore_policy=True,
        policy_row_retention="all_eligible")
    planned: dict[str, Any] = {"start_ordinal": 0, "physical_game_id": collection + ":0", "bank": "train",
               "root_fen": root.fen(), "opening_moves": []}
    action = move_to_index(move, root)
    gid = _stable_game_id(start_fen=root.fen(), opening_source="test", move_trace=str(action), result="1-0", total_plies_played=1)
    audit = GameProvenance(start=GameStartProvenance(0, 0, root.fen(), (), root.fen(), 0, "test", True, True),
        game_id=gid, played_move_indices=(action,), final_fen=final.fen(),
        records=(NetRecordProvenance(0, 0, True, True, False),), result="1-0", resolved_result="1-0",
        natural_result="1-0", resolution="natural", tablebase_rescore_result=None, cap_reached=False,
        total_plies=1, retained_samples=1, native_result=str(CBoard.from_board(final).result()),
        tablebase_witness=verifier.owner.resolve(final, move_offset=1), tablebase_policy_applied_plies=(0,))
    proof = verifier.verify(asdict(audit), planned, game)
    adapter = SimpleNamespace(input_history_encoding=game.input_history_encoding, input_extra_features=game.input_extra_features)
    x = encode_cboard_for_model(adapter, CBoard.from_board(root))
    legal, policy = np.zeros(1858, dtype=np.uint8), np.zeros(1858, dtype=np.float32)
    for candidate in root.legal_moves:
        legal[move_to_index_for_encoding(candidate, root, policy_encoding="lc0_1858")] = 1
    policy[move_to_index_for_encoding(move, root, policy_encoding="lc0_1858")] = 1
    sample = ReplaySample(x=x, policy_target=policy, wdl_target=_result_to_wdl("1-0", pov_white=True),
        game_id=gid, ply_index=0, legal_mask=legal, is_network_turn=True, is_selfplay=True,
        input_history_encoding=game.input_history_encoding, history_rep_fix=True)
    if diagnostics == "active":
        sample.search_wdl = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
        sample.categorical_target = np.zeros(game.categorical_bins, dtype=np.float32)
        sample.categorical_target[0] = 1
    row_id = planned["physical_game_id"] + ":0"
    raw, raw_meta = capture_raw_targets([sample], audit, [row_id], proof=proof, game=game)
    arrays = samples_to_arrays([sample])
    if corruption == "input":
        arrays["x"][0, 0, 0, 0] += 1
    elif corruption == "policy":
        other = next(i for i in np.flatnonzero(legal) if policy[i] == 0)
        raw["policy_target_raw"][0] = 0
        raw["policy_target_raw"][0, other] = 1
        arrays["policy_target"][0] = raw["policy_target_raw"][0].astype(np.float16)
    elif corruption == "outcome":
        raw["outcome_wdl"][0] = 1
        arrays["wdl_target"][0] = 1
    directory = tmp_path / "game-0"
    directory.mkdir()
    save_local_shard_arrays(directory / "shard", arrs=arrays, meta={"games": 1, "positions": 1})
    shard = {"path": str(directory / "shard"), "files": collect.inventory(directory / "shard")}
    common = {"run_uuid": run, "collection_uuid": collection, "generation": generation, "source_revision": source,
              "actor_sha256": actor, "served_state_sha256": served, "physical_game_id": planned["physical_game_id"],
              "bank": "train", "game_id": gid, "rows": 1, "scientific_class": "NATURAL"}
    row = {"physical_row_id": row_id, "ply": 0, "move_offset": 0,
           "history_sha256": admit.digest([root.fen(), []]),
           "stored_input_float16_sha256": hashlib_sha(np.asarray(x, dtype="<f2").tobytes()),
           "stored_input_float16_mirror_sha256": hashlib_sha(_mirror_x_batch(np.asarray([x], dtype=np.float16),
               np.ones(1, dtype=bool), input_history_encoding=game.input_history_encoding)[0].astype("<f2").tobytes())}
    validation = {"schema": "VERIFIED_CORE_GAME_METADATA_V1", **common, "full_history_verified": True,
        "resolved": True, "raw_targets_verified": True, "usable_shard": True, "rows_detail": [row],
        "verified_outcome_sha256": proof.proof_sha256, "shard_inventory_sha256": admit.digest(shard["files"])}
    original: dict[str, Any] = {"schema": "CORE_PRODUCING_OUTCOME_PROOF_V1", **common,
                "verification": proof.metadata(), "proof_sha256": proof.proof_sha256}
    if corruption == "proof":
        original["verification"]["actor_sha256"] = "f" * 64
        original["proof_sha256"] = admit.digest(original["verification"])
    evidence = {"provenance": frozen(directory / "provenance.json", {"schema": "CORE_GAME_PROVENANCE_V1", **common, "planned": planned, "actual": asdict(audit)}),
        "validation": frozen(directory / "validation.json", validation),
        "outcome_proof": frozen(directory / "outcome-proof.json", original),
        "raw_metadata": frozen(directory / "raw-targets.json", {**raw_meta, **common})}
    np.savez_compressed(directory / "raw-targets.npz", **raw)
    inventory = collect.inventory(directory, seal=True)
    item = {**common, "path": str(directory), "inventory": inventory, "evidence": evidence,
            "validation": validation, "shard": shard}
    producer = {"run_uuid": run, "collection_uuid": collection, "generation": generation, "source": {"revision": source}, "actor": {"sha256": actor}}
    return SimpleNamespace(verifier=verifier, owners=owners, item=item, planned=planned, producer=producer, game=game,
                           adapter=adapter, asset_sha=asset_sha, raw=raw, raw_meta=raw_meta, gid=gid, source=source,
                           served=served, run=run, collection=collection, root=root, proof=proof)


@pytest.mark.parametrize("corruption", [None, "input", "policy", "outcome", "proof"])
@pytest.mark.parametrize("diagnostics", ["inactive", "active"])
def test_real_verified_raw_and_zarr_roundtrip_with_fake_owned_tb(tmp_path: Path, monkeypatch, corruption, diagnostics):
    from chess_anti_engine.core_generation_data import validate_stored_targets
    from chess_anti_engine.replay.shard import load_shard_arrays

    f = synthetic_game(tmp_path, monkeypatch, corruption, diagnostics)
    verifier, item, planned, producer, game, adapter = f.verifier, f.item, f.planned, f.producer, f.game, f.adapter
    raw, raw_meta, gid = f.raw, f.raw_meta, f.gid
    directory, proof = Path(item["path"]), f.proof
    try:
        if corruption is None:
            result, decoded = admit.verify_game(item, planned, producer, verifier, game, adapter,
                {"max_raw_decoded_bytes_per_game": 1 << 20, "max_shard_decoded_bytes_per_game": 8 << 20})
            assert result["eligible"] is True
            assert result["rows"] == 1
            assert decoded > 0
            if diagnostics == "active":
                loaded, _ = load_shard_arrays(directory / "shard", lazy=False, validate=True)
                for target, mask in (("search_wdl", "has_search_wdl"), ("categorical_target", "has_categorical")):
                    for removed in ((target,), (mask,), (target, mask)):
                        damaged = {key: value for key, value in loaded.items() if key not in removed}
                        with pytest.raises(ValueError, match="stored target"):
                            validate_stored_targets(raw, admit.stored_validation_view(damaged, raw), proof=proof,
                                                    metadata=raw_meta, game_id=gid, plies=[0])
        else:
            with pytest.raises(ValueError, match=r"differs|bytes|outcome|onehot"):
                admit.verify_game(item, planned, producer, verifier, game, adapter,
                    {"max_raw_decoded_bytes_per_game": 1 << 20, "max_shard_decoded_bytes_per_game": 8 << 20})
    finally:
        verifier.close()
    assert verifier.closed is True


def hashlib_sha(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def entry_case(tmp_path: Path, monkeypatch, *, generation: int = 2, fault: str | None = None,
               corruption=None, legacy: bool = False) -> SimpleNamespace:
    """Real one-game audit data; only runtime/native qualification is external."""
    import torch
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
    from chess_anti_engine.selfplay.config import DiffFocusConfig, OpponentConfig, SearchConfig, TemperatureConfig
    from chess_anti_engine.selfplay.opening import OpeningConfig
    from scripts import core_finite_collect as collect

    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    previous = tmp_path / "previous"
    previous.mkdir()
    actor_path = previous / "learner.pt"
    actor_path.write_bytes(b"synthetic actor identity only; not a checkpoint")
    actor_path.chmod(0o444)
    actor_sha = admit.learning.file_sha(actor_path)
    f = synthetic_game(producer_root, monkeypatch, corruption, generation=generation, actor_sha=actor_sha)
    f.verifier.close()
    model_config = ModelConfig(history_rep_fix=f.game.history_rep_fix, input_history_encoding=f.game.input_history_encoding,
                               input_extra_features=f.game.input_extra_features, policy_encoding=f.game.policy_encoding)
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(model_config)}
    prior_contract = frozen(tmp_path / "prior-contract.json", {"run_uuid": f.run, "source_revision": f.source})
    prior_stage = frozen(tmp_path / "prior-stage.json", {"generation": generation - 1, "contract_sha256": prior_contract["sha256"],
                         "generations": [], "parent": None, "output": str(previous)})
    checkpoint = {"path": str(actor_path), "sha256": actor_sha}
    boundary = frozen(previous / "boundary.json", {"status": "COMPLETED_CORE_GENERATION_V1", "run_uuid": f.run,
        "source_revision": f.source, "generation": generation - 1, "step": 7, "checkpoint": checkpoint,
        "contract_sha256": prior_contract["sha256"], "stage": prior_stage, "generation_manifests": [], "parent": None})
    prior_command = ["python", "-m", "scripts.core_finite_learner", "--contract", prior_contract["path"],
                     "--contract-sha256", prior_contract["sha256"], "--stage", prior_stage["path"], "--stage-sha256", prior_stage["sha256"]]
    owner_sha = "d" * 64
    prior_terminal = frozen(tmp_path / "prior-terminal.json", {"status": "PASS", "returncode": 0,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": owner_sha, "command": prior_command})
    predecessor = {"boundary": boundary, "owner_terminal": prior_terminal, "owner_sha256": owner_sha, "command": prior_command}
    if generation == 1:
        # Only the synthetic actor resource changes; the actual producer
        # validator still exercises its original G1 donor rule.
        monkeypatch.setattr(collect, "INITIAL_DONOR_SHA256", actor_sha)
    starts = tmp_path / "starts.epd"
    starts.write_text(f.planned["root_fen"] + " | \n")
    starts.chmod(0o444)
    cfg = {"game": f.game, "search": SearchConfig(simulations=64, fast_simulations=64, playout_cap_fraction=1),
           "temp": TemperatureConfig(), "opening": OpeningConfig(opening_fen_list_path=str(starts), opening_fen_dole_per_iter=1),
           "opponent": OpponentConfig(), "diff_focus": DiffFocusConfig(enabled=False)}
    profile = {key: asdict(value) for key, value in cfg.items()}
    producer: dict[str, Any] = {**f.producer, "schema": "CORE_FINITE_GENERATION_V1", "attempts": 1, "concurrent_slots": 1, "seed": 0,
        "actor": {"kind": "donor" if generation == 1 else "learner", **checkpoint, "expected_checkpoint_step": 7,
                  "architecture": arch, "predecessor": None if generation == 1 else predecessor},
        "source": {"revision": f.source, "checkout": str(admit.learning.REPO)},
        "owner": {"sha256": owner_sha, "seconds": 120}, "output_root": str(producer_root),
        "runtime": {"device": "cuda:0", "cpu_affinity": [20, 22], "threads": 2, "inner_seconds": 90},
        "stockfish": {"workers": 1, "nodes": 1, "multipv": 1, "hash_mb": 1},
        "serving": {"device": "cuda:0", "use_amp": True, "amp_dtype": "bfloat16"},
        "tablebase": {"path": "/synthetic-tables", "asset_inventory_sha256": f.asset_sha},
        "roster": frozen(tmp_path / "roster.json", [f.planned]), "starts": admit.ref(starts),
        "configs": profile, "resolved_profile_sha256": admit.digest(profile),
        "exclusions": {"prior": frozen(tmp_path / "prior-exposure.json", {"history_sha256": [],
                       "stored_input_float16_sha256": [], "coverage_scope": "synthetic fixture only", "donor_pretraining_exposure": "UNKNOWN"}),
                       "playing_starts": frozen(tmp_path / "playing-starts.json", []), "playing_start_count": 0,
                       "prior_overlap_action": "refuse", "playing_overlap_action": "refuse"}}
    overlap = collect.exclusions_gate({0: f.item}, producer)
    frozen(producer_root / "input-exclusions.json", overlap)
    if fault == "exclusions":
        row = f.item["validation"]["rows_detail"][0]
        producer["exclusions"]["playing_starts"] = frozen(tmp_path / "collision.json", [{"id": "synthetic",
            "input_sha256": row["stored_input_float16_sha256"], "mirror_sha256": row["stored_input_float16_mirror_sha256"]}])
        producer["exclusions"]["playing_start_count"] = 1
    producer_ref = frozen(tmp_path / "producer.json", producer)
    stage_admission = frozen(tmp_path / "producer-admission.json", {"status": "ROOT_ADMITS_CORE_GENERATION_V1",
        "protocol_sha256": producer_ref["sha256"], "run_uuid": f.run, "generation": generation,
        "source_revision": f.source, "actor_sha256": actor_sha})
    command = ["python", "-m", "scripts.core_finite_collect", "--execute", "--protocol", producer_ref["path"],
               "--protocol-sha256", producer_ref["sha256"], "--admission", stage_admission["path"],
               "--admission-sha256", stage_admission["sha256"]]
    terminal = {"status": "PASS", "returncode": 0, "owned_processes_closed": fault != "owner_unclosed",
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": owner_sha, "resource": "gpu", "seconds": 120,
        "command": command, "known_processes": {"2147483647": 1}}
    terminal_ref = frozen(tmp_path / "producer-terminal.json", terminal)
    serving = {"actor_sha256": actor_sha, "pre_state_sha256": f.served, "post_state_sha256": f.served,
        "checkpoint_model_state_sha256": f.served, "state_tensors": 494, "model_build_count": 1,
        "checkpoint_step": 8 if fault == "serving_step" else 7, "logical_generation": generation,
        "forwards": {"calls": 1}, "checkpoint_stat": collect.file_stamp(actor_path)}
    serving_ref = frozen(producer_root / "actor-serving.json", serving)
    frozen(producer_root / "resolved-profile.json", profile)
    frozen(producer_root / "stockfish-lifecycle.json", {"all_observed_births_closed": fault != "lifecycle"})
    frozen(producer_root / "completed-0.json", f.item)
    readout = {"status": "COLLECTED_CORE_GENERATION_V1", "protocol_sha256": producer_ref["sha256"],
        "run_uuid": f.run, "collection_uuid": f.collection if fault != "readout_collection" else "x",
        "generation": generation, "attempts": 1, "source_revision": f.source, "actor_sha256": actor_sha,
        "all_planned_attempts_accounted": True, "training_steps": 0, "root_acceptance_issued": False,
        "learning_admitted": False, "actor_serving": serving_ref, "pool_closed": True, "verification_handle_closed": True,
        "completed": {} if fault == "attempt_accounting" else {"0": f.item},
        "class_counts": {"NATURAL": 1, "STRICT_TB_ADJUDICATED": 0, "UNRESOLVED": 0},
        "resolved_nonempty_games": 1, "playing_and_exposure_gate_qualified": True}
    readout_ref = frozen(producer_root / "readout.json", readout)
    if fault == "producer_failure":
        frozen(producer_root / "failure.json", {"status": "FAILED_SYNTHETIC_PRODUCER"})
    inventory = {"path": str(producer_root), "files": collect.inventory(producer_root, seal=True)}
    required = set(admit.learning.SOURCE_FILES) | {"scripts/core_finite_admit.py", "scripts/core_generation_data_admit.py"}
    spec: dict[str, Any] = {"schema": admit.DATA_SPEC, "run_uuid": f.run, "collection_uuid": f.collection, "generation": generation,
        "source_revision": f.source, "source_files": {name: admit.learning.file_sha(admit.learning.REPO / name) for name in required},
        "architecture": arch, "actor_sha256": actor_sha, "actor_checkpoint_step": 7,
        "games_per_generation": 1, "max_games_per_generation": 1}
    p: dict[str, Any] = {"schema": admit.DATA_PROTOCOL, "data_spec": None, "producer_protocol": producer_ref, "producer_readout": readout_ref,
        "producer_inventory": frozen(tmp_path / "inventory.json", inventory),
        "producer_owner": {"terminal": terminal_ref, "owner_sha256": owner_sha, "command": command},
        "gates": {"attempts": 1, "min_resolved_games": 1, "min_total_rows": 1}, "runtime": {"inner_seconds": 30},
        "output_root": str(tmp_path / "audit-output"), "limits": {"max_metadata_bytes": 4 << 20,
            "max_inventory_files": 1000, "max_inventory_bytes": 32 << 20, "max_raw_decoded_bytes_per_game": 1 << 20,
            "max_shard_decoded_bytes_per_game": 8 << 20, "max_total_decoded_bytes": 16 << 20}}
    if fault == "owner_missing":
        p["producer_owner"]["terminal"] = {"path": str(tmp_path / "absent-terminal.json"), "sha256": "0" * 64}
    calls = []
    monkeypatch.setattr(admit, "runtime_guard", lambda _p: calls.append("runtime_guard"))
    monkeypatch.setattr(collect, "source_guard", lambda _p: calls.append("source_guard") or {"native_images": []})
    if not legacy:
        def forbidden_recipe(*_args, **_kwargs):
            pytest.fail("recipe-free auditor called the learner validator or a model/learning path")
        monkeypatch.setattr(admit.learning, "validate_contract", forbidden_recipe)
        monkeypatch.setattr(admit.learning, "make_trainer", forbidden_recipe)
        monkeypatch.setattr(torch, "load", forbidden_recipe)
        monkeypatch.setattr(torch.nn.Module, "__init__", forbidden_recipe)
    return SimpleNamespace(protocol=p, spec=spec, fixture=f, calls=calls, actor=checkpoint)


def bind_data_case(tmp_path: Path, case: SimpleNamespace) -> dict[str, str]:
    case.protocol["data_spec"] = frozen(tmp_path / "data-spec.json", case.spec)
    return frozen(tmp_path / "audit-protocol.json", case.protocol)


def test_recipe_free_public_entry_uses_real_audit_and_emits_unpromoted_candidate(tmp_path: Path, monkeypatch, capsys):
    case = entry_case(tmp_path, monkeypatch)
    pref = bind_data_case(tmp_path, case)
    assert data_admit.main(["--protocol", pref["path"], "--protocol-sha256", pref["sha256"]]) == 0
    output = Path(case.protocol["output_root"])
    _, result = admit.learning.read_ref(admit.ref(output / "readout.json"))
    _, generation = admit.learning.read_ref(result["generation_manifest"])
    _, wrapper = admit.learning.read_ref(generation["games"][0]["producer_proof"])
    assert result["status"] == "AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1"
    assert generation["schema"] == admit.DATA_CANDIDATE
    assert wrapper["status"] == admit.DATA_GAME_CANDIDATE
    for record in (result, generation, wrapper):
        assert record["data_spec"] == case.protocol["data_spec"]
        assert record["collection_uuid"] == case.spec["collection_uuid"]
        assert record["logical_generation"] == 2
        assert record["actor_checkpoint_step"] == 7
        assert record["learning_admitted"] is False
    assert generation["generation"] == 2
    assert generation["root_postclosure_promotion_required"] is True
    assert result["eligible_games"] == 1
    assert result["eligible_rows"] == 1
    assert result["decoded_bytes"] > 0
    assert case.calls == ["runtime_guard", "source_guard", "source_guard"]
    assert all(owner.closed for owner in case.fixture.owners)
    assert '"status":"AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1"' in capsys.readouterr().out
    contract_ref = {"path": "unused", "sha256": "e" * 64}
    stage = {"schema": "core_finite_stage_v1", "contract_sha256": contract_ref["sha256"], "generation": 1,
             "generations": [result["generation_manifest"]], "parent": None, "output": "unused"}
    with pytest.raises(ValueError, match="unadmitted generation"):
        admit.learning.metadata({"updates_per_generation": [1]}, stage, contract_ref)


@pytest.mark.parametrize("field", ["run_uuid", "collection_uuid", "generation", "source_revision", "architecture",
                                  "actor_sha256", "actor_checkpoint_step"])
def test_data_entry_refuses_producer_identity_mismatches(tmp_path: Path, monkeypatch, field):
    case = entry_case(tmp_path, monkeypatch)
    values = {"run_uuid": "33333333-3333-4333-8333-333333333333", "collection_uuid": "33333333-3333-4333-8333-333333333333",
              "generation": 3, "source_revision": "d" * 40, "architecture": {"different": True},
              "actor_sha256": "e" * 64, "actor_checkpoint_step": 8}
    case.spec[field] = values[field]
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match="specification/producer identity"):
        data_admit.execute(pref)


@pytest.mark.parametrize("field", ["donor", "lr", "optimizer", "updates_per_generation", "seed", "mirror_prob",
                                  "replay_generations", "batch_size", "runtime"])
def test_data_entry_rejects_recipe_fields(tmp_path: Path, monkeypatch, field):
    case = entry_case(tmp_path, monkeypatch)
    case.spec[field] = None
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match="data specification fields/schema"):
        data_admit.execute(pref)


@pytest.mark.parametrize("fault", ["missing_pin", "missing_core_pin", "missing_auditor_pin", "bad_pin", "bad_path",
                                  "minmax", "attempts", "games", "rows"])
def test_data_entry_source_and_gate_assertions_take_effect(tmp_path: Path, monkeypatch, fault):
    case = entry_case(tmp_path, monkeypatch)
    if fault == "missing_pin":
        del case.spec["source_files"]["scripts/core_generation_data_admit.py"]
    elif fault == "missing_core_pin":
        del case.spec["source_files"]["scripts/core_finite_learner.py"]
    elif fault == "missing_auditor_pin":
        del case.spec["source_files"]["scripts/core_finite_admit.py"]
    elif fault == "bad_pin":
        case.spec["source_files"]["scripts/core_finite_admit.py"] = "0" * 64
    elif fault == "bad_path":
        case.spec["source_files"]["../outside.py"] = "0" * 64
    elif fault == "minmax":
        case.spec["games_per_generation"] = 2
    elif fault == "attempts":
        case.protocol["gates"]["attempts"] = 2
    elif fault == "games":
        case.protocol["gates"]["min_resolved_games"] = 2
    else:
        case.protocol["gates"]["min_total_rows"] = 2
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match=r"source|game bounds|contract mismatch|insufficient"):
        data_admit.execute(pref)


@pytest.mark.parametrize("fault", ["owner_unclosed", "owner_missing", "producer_failure", "lifecycle", "serving_step",
                                  "readout_collection", "attempt_accounting", "exclusions"])
def test_data_entry_preserves_closed_producer_and_full_attempt_guards(tmp_path: Path, monkeypatch, fault):
    case = entry_case(tmp_path, monkeypatch, fault=fault)
    pref = bind_data_case(tmp_path, case)
    messages = {"owner_unclosed": "producer owner not exact closed PASS", "owner_missing": "absent-terminal.json",
                "producer_failure": "failed/partial producer publication", "lifecycle": "producer engine/verifier closure absent",
                "serving_step": "actor serving evidence differs", "readout_collection": "producer readout identity/status differs",
                "attempt_accounting": "not every fixed attempt accounted", "exclusions": "playing/exposure gate differs or failed"}
    error = FileNotFoundError if fault == "owner_missing" else ValueError
    with pytest.raises(error, match=messages[fault]):
        data_admit.execute(pref)


@pytest.mark.parametrize("corruption", ["input", "proof"])
def test_data_entry_reaches_real_raw_history_verification(tmp_path: Path, monkeypatch, corruption):
    case = entry_case(tmp_path, monkeypatch, corruption=corruption)
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match=r"differs|bytes"):
        data_admit.execute(pref)


def test_new_and_legacy_public_entries_reject_crossed_protocols(tmp_path: Path, monkeypatch):
    case = entry_case(tmp_path, monkeypatch)
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match="invalid audit protocol"):
        admit.main(["--protocol", pref["path"], "--protocol-sha256", pref["sha256"]])
    legacy = {**case.protocol, "schema": "CORE_FINITE_ADMISSION_V1", "learner_contract": case.protocol["data_spec"]}
    del legacy["data_spec"]
    legacy_ref = frozen(tmp_path / "legacy-protocol.json", legacy)
    with pytest.raises(ValueError, match="invalid audit protocol"):
        data_admit.main(["--protocol", legacy_ref["path"], "--protocol-sha256", legacy_ref["sha256"]])


@pytest.mark.parametrize(("field", "value"), [
    ("run_uuid", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"), ("collection_uuid", None),
    ("generation", True), ("generation", 0), ("generation", 2.0),
    ("actor_checkpoint_step", True), ("actor_checkpoint_step", -1),
    ("source_revision", "G" * 40), ("actor_sha256", "A" * 64), ("games_per_generation", False),
])
def test_data_entry_strict_specification_scalar_types(tmp_path: Path, monkeypatch, field, value):
    case = entry_case(tmp_path, monkeypatch)
    case.spec[field] = value
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match="invalid"):
        data_admit.execute(pref)


def test_candidate_game_wrapper_is_not_a_legacy_train_proof(tmp_path: Path, monkeypatch):
    case = entry_case(tmp_path, monkeypatch, generation=1)
    result = data_admit.execute(bind_data_case(tmp_path, case))
    _, candidate = admit.learning.read_ref(result["generation_manifest"])
    # Changing an envelope label alone is not promotion. The actual new game
    # wrapper still fails the existing learner's strict proof predicates.
    candidate["schema"] = "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1"
    ref = frozen(tmp_path / "relabeled-candidate.json", candidate)
    c: dict[str, Any] = {"run_uuid": case.spec["run_uuid"], "source_revision": case.spec["source_revision"],
         "updates_per_generation": [1], "games_per_generation": 1, "max_games_per_generation": 1}
    contract_ref = {"path": "unused", "sha256": "e" * 64}
    stage = {"schema": "core_finite_stage_v1", "contract_sha256": contract_ref["sha256"], "generation": 1,
             "generations": [ref], "parent": None, "output": "unused"}
    with pytest.raises(ValueError, match="TRAIN proof/identity predicates"):
        admit.learning.metadata(c, stage, contract_ref)


def test_legacy_entry_retains_recipe_validation_and_output_schemas(tmp_path: Path, monkeypatch):
    case = entry_case(tmp_path, monkeypatch, legacy=True)
    c: dict[str, Any] = {"schema": "core_finite_learning_v1", "run_uuid": case.spec["run_uuid"],
        "source_revision": case.spec["source_revision"], "source_files": case.spec["source_files"],
        "architecture": case.spec["architecture"], "donor": case.actor, "batch_size": 1,
        "games_per_generation": 1, "max_games_per_generation": 1, "updates_per_generation": [1],
        "replay_generations": 4, "seed": 0, "mirror_prob": 0,
        "runtime": {"device": "cpu", "use_amp": False, "torch_version": "unused", "numpy_version": "unused",
                    "torch_threads": 2, "cuda_devices": 0},
        "lr": {"peak": 1e-4, "floor": 1e-4, "warmup_steps": 0, "warmup_start": 1e-4, "first_period": 2},
        "limits": {"max_rows": 1, "max_decoded_window_bytes": 8 << 20}}
    p = {**case.protocol, "schema": "CORE_FINITE_ADMISSION_V1", "learner_contract": frozen(tmp_path / "learner-contract.json", c)}
    del p["data_spec"]
    result = admit.execute(frozen(tmp_path / "legacy-audit.json", p))
    _, generation = admit.learning.read_ref(result["generation_manifest"])
    _, wrapper = admit.learning.read_ref(generation["games"][0]["producer_proof"])
    assert result["status"] == "AUDITED_CORE_TRAIN_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1"
    assert generation["schema"] == "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1"
    assert wrapper["status"] == "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1"
    for record in (result, generation, wrapper):
        assert "data_spec" not in record
        assert "logical_generation" not in record
        assert "actor_checkpoint_step" not in record
    c["lr"]["peak"] = -1
    p["learner_contract"] = frozen(tmp_path / "bad-learner-contract.json", c)
    p["output_root"] = str(tmp_path / "bad-legacy-audit")
    with pytest.raises(ValueError, match="invalid LR bounds"):
        admit.execute(frozen(tmp_path / "bad-legacy-protocol.json", p))


@pytest.mark.parametrize(("raw_mask", "payload", "flag"), [([0], False, False), ([1], False, False),
                                                         ([0, 1], False, False), ([0], True, False), ([0], False, True)])
def test_optional_validation_view_never_repairs_active_or_ambiguous_pairs(raw_mask, payload, flag):
    import numpy as np

    n = len(raw_mask)
    arrays: dict[str, Any] = {"x": np.zeros((n, 175, 8, 8), dtype=np.float16),
              "policy_target": np.zeros((n, 1858), dtype=np.float16)}
    if payload:
        arrays["search_wdl"] = np.zeros((n, 3), dtype=np.float16)
    if flag:
        arrays["has_search_wdl"] = np.ones(n, dtype=np.uint8)
    original_keys = set(arrays)
    raw = {"has_search_raw": np.asarray(raw_mask, dtype=np.uint8), "has_categorical_raw": np.zeros(n, dtype=np.uint8)}
    view = admit.stored_validation_view(arrays, raw)
    assert set(arrays) == original_keys
    assert view is not arrays
    if flag:
        assert view["has_search_wdl"] is arrays["has_search_wdl"]
    elif not payload and not any(raw_mask):
        assert view["has_search_wdl"].dtype == np.uint8
        assert np.array_equal(view["has_search_wdl"], np.zeros(n, dtype=np.uint8))
    else:
        assert "has_search_wdl" not in view
