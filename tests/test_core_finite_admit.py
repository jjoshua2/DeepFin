"""CPU-only boundary and native-encoding controls; fake owned TB, no model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import core_finite_admit as admit


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


@pytest.mark.parametrize("corruption", [None, "input", "policy", "outcome", "proof"])
def test_real_verified_raw_and_zarr_roundtrip_with_fake_owned_tb(tmp_path: Path, monkeypatch, corruption):
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

    class FakeOwner:
        def __init__(self, _path):
            self.asset_inventory_sha256 = asset_sha
            self.tablebase = SimpleNamespace(wdl={}, dtz={})
            self.closed = False

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
    actor, served, source = "a" * 64, "b" * 64, "c" * 40
    verifier = VerifiedOutcomeVerifier(path="/synthetic-tables", asset_inventory_sha256=asset_sha,
        collection_uuid=collection, run_uuid=run, generation=1, source_revision=source,
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
    planned = {"start_ordinal": 0, "physical_game_id": collection + ":0", "bank": "train",
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
    common = {"run_uuid": run, "collection_uuid": collection, "generation": 1, "source_revision": source,
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
    original = {"schema": "CORE_PRODUCING_OUTCOME_PROOF_V1", **common,
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
    producer = {"run_uuid": run, "collection_uuid": collection, "generation": 1, "source": {"revision": source}, "actor": {"sha256": actor}}
    try:
        if corruption is None:
            result, decoded = admit.verify_game(item, planned, producer, verifier, game, adapter,
                {"max_raw_decoded_bytes_per_game": 1 << 20, "max_shard_decoded_bytes_per_game": 8 << 20})
            assert result["eligible"] is True
            assert result["rows"] == 1
            assert decoded > 0
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
