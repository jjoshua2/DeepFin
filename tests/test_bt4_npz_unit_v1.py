"""CPU-only one-game source seam and actual process ownership fixtures."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import py_compile
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from chess_anti_engine.source import bt4_npz_unit_v1 as unit

PREFIX = ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
          "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8")
MANIFEST = "a" * 64
STRICT = "b" * 64
SYZYGY = "c" * 64
ARCHIVE = b"synthetic archive, not NPZ"
GUARD = Path(__file__).resolve().parents[1] / "scripts/bt4_npz_exec_guard.py"
ACTUAL_1388_PROOF_LINE_SHA256 = (
    "d422febc15864197a793eec210c06e8807af505a748f1ac9726bf037fcf8a018")
ACTUAL_1388_TERMINAL_KEYS = frozenset({
    "final_fen", "first_terminal_ply", "game_id", "outcome_used_as_target",
    "pair_members", "raw_dtz", "raw_wdl", "result", "root_id", "rows",
    "signature", "source", "source_archive_sha256",
    "source_strict_receipt_sha256", "terminal_replayed", "termination",
})


def _identity(rows: int = 2) -> unit.GameIdentity:
    return unit.GameIdentity(MANIFEST, f"bt4_run11:{MANIFEST}", "root-0", 7,
                             unit.sha(ARCHIVE), STRICT, rows, PREFIX)


def _actual(identity: unit.GameIdentity) -> SimpleNamespace:
    return SimpleNamespace(source="BT4-v9",
                           source_manifest_sha=identity.manifest_sha256,
                           namespace=identity.namespace, root_id=identity.root_id,
                           game_id=identity.game_id,
                           archive_sha=identity.archive_sha256,
                           strict_receipt_sha=identity.strict_receipt_sha256,
                           gross_rows=identity.rows,
                           root={"uci_prefix": list(identity.root_prefix_uci)})


def _source_proof(identity: unit.GameIdentity) -> dict:
    return {"schema": "full512_raw_strict_source_proof_v1",
            "status": "PASS_SOURCE_PROOF", "source": "BT4-v9",
            "root_id": identity.root_id, "game_id": identity.game_id,
            "strict_receipt_sha256": STRICT,
            "archive_sha256": identity.archive_sha256,
            "gross_rows": identity.rows,
            "raw_archive_sha256": identity.archive_sha256,
            "checks": dict.fromkeys(("feed", "policy", "strict_receipt", "terminal"), True),
            "detail": "synthetic", "evidence": {"fixture": "synthetic"},
            "final_fen": "synthetic", "result": "1-0",
            "source_verifier_sha256": "d" * 64, "termination": "synthetic"}


def _replay(identity: unit.GameIdentity, *, wrong_uid: bool = False,
            fail_after_proof: bool = False
            ) -> Callable[[object, bytes, Callable[[dict], None]],
                           tuple[tuple[dict, bytes], ...]]:
    def replay(actual: object, raw: bytes,
               sink: Callable[[dict], None]
               ) -> tuple[tuple[dict, bytes], ...]:
        assert actual is not None
        assert raw == ARCHIVE
        proof = _source_proof(identity)
        sink({"source": "BT4-v9", "root_id": identity.root_id,
              "game_id": identity.game_id,
              "archive_sha256": identity.archive_sha256,
              "source_proof_sha256": unit.sha(unit.canonical(proof)),
              "source_proof": proof,
              "history_chain": {
                  "opening_uci": list(identity.root_prefix_uci),
                  "played_uci": ["a2a3"] * identity.rows,
                  "root_start_fen": "synthetic",
                  "row_index": [{} for _ in range(identity.rows)],
                  "schema": "tri_source_full_game_history_chain_v1",
                  "terminal_fen": "synthetic"},
              "strict_trace": [{} for _ in range(identity.rows + 1)],
              "terminal_fact": {
                  "final_fen": "synthetic", "first_terminal_ply": 0,
                  "game_id": identity.game_id, "outcome_used_as_target": True,
                  "pair_members": [], "raw_dtz": 0, "raw_wdl": 2,
                  "result": "1-0", "root_id": identity.root_id,
                  "rows": identity.rows, "signature": "synthetic",
                  "source": "BT4-v9",
                  "source_archive_sha256": identity.archive_sha256,
                  "source_strict_receipt_sha256": STRICT,
                  "terminal_replayed": True, "termination": "synthetic",
                  "syzygy_inventory_sha256": SYZYGY}})
        if fail_after_proof:
            raise RuntimeError("replay failed after provisional proof")
        rows = []
        for index in range(identity.rows):
            uid = [identity.manifest_sha256, identity.namespace,
                   identity.root_id, identity.game_id, len(PREFIX) + index]
            if wrong_uid and index == 0:
                uid[-1] = 0
            native = hashlib.sha256(unit.canonical(uid)).digest() * 1400
            assert len(native) == unit.ROW_BYTES
            rows.append(({"uid": uid, "source": "BT4-v9",
                          "provenance_locator_sha256": unit.sha(unit.canonical(uid)),
                          "input_digest": unit.sha(unit.wave.DOMAIN + native),
                          "history_stack_sha256": "history",
                          "repetition": "repetition", "rule50": 0,
                          "legal_context_sha256": "legal",
                          "teacher_query_sha256": "teacher",
                          "outcome": "1-0"}, native))
        return tuple(rows)
    return replay


def _old_rows(game: unit.BoundGame) -> tuple[tuple[dict, bytes], ...]:
    return tuple(({key: value for key, value in row.items()
                   if key in unit.OLD_ROW_KEYS}, native)
                 for row, native in game.rows)


def _old_proof(game: unit.BoundGame) -> dict:
    terminal = dict(game.proof["terminal_fact"])
    del terminal["syzygy_inventory_sha256"]
    return {**game.proof, "terminal_fact": terminal}


def test_saved_game_1388_old_proof_format_and_single_field_enrichment() -> None:
    path = Path(__file__).parent / "fixtures/bt4_game1388_old_proof_v8.json"
    raw = path.read_bytes()
    assert len(raw) == 96_844
    assert unit.sha(raw) == ACTUAL_1388_PROOF_LINE_SHA256
    proof = json.loads(raw)
    assert raw == unit.canonical(proof)
    assert set(proof["terminal_fact"]) == ACTUAL_1388_TERMINAL_KEYS
    assert proof["game_id"] == 1388
    assert proof["root_id"] == (
        "fp16_014a24839b0fd59e598557fc6b68498dc735f4cfc4196a7d5e2134d41e2b11c4")
    assert proof["source_proof_sha256"] == unit.sha(
        unit.canonical(proof["source_proof"]))
    expected = unit.GameIdentity(
        "b09ea044da4879aa9b27f5353f5370d62258c6ddf5cb83efff4fbcd039c561e1",
        "bt4_run11:b09ea044da4879aa9b27f5353f5370d62258c6ddf5cb83efff4fbcd039c561e1",
        proof["root_id"], 1388,
        "2cbccb13faea9c5bb715f57234e25b69107058971f52f2c750c0bd3577075a41",
        "5230b7dff0d63b071f07e92195dd653c404195958fd858d51a899cddc198edee",
        93, ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "g1f3",
             "e8g8", "h2h3", "d7d6", "c1f4", "c7c5", "d4d5", "a7a6",
             "a2a4", "e7e6"))
    enriched = unit.enrich_old_replay_proof(proof, proof, expected, SYZYGY)
    assert enriched["terminal_fact"]["syzygy_inventory_sha256"] == SYZYGY
    assert unit.canonical(_old_proof(unit.BoundGame((), enriched))) == raw
    # Matching omissions in both inputs must not turn into a complete proof.
    for section in (None, "source_proof", "terminal_fact", "history_chain"):
        original = proof if section is None else proof[section]
        for key in original:
            damaged = json.loads(raw)
            del (damaged if section is None else damaged[section])[key]
            with pytest.raises(unit.Hold, match="raw old replay proof"):
                unit.enrich_old_replay_proof(damaged, damaged, expected, SYZYGY)
    damaged = {**proof, "terminal_fact":
               {key: value for key, value in proof["terminal_fact"].items()
                if key != "source_strict_receipt_sha256"}}
    with pytest.raises(unit.Hold, match="raw old replay proof"):
        unit.enrich_old_replay_proof(damaged, damaged, expected, SYZYGY)
    conflicting = {**proof, "terminal_fact":
                   {**proof["terminal_fact"],
                    "syzygy_inventory_sha256": "0" * 64}}
    with pytest.raises(unit.Hold, match="raw old replay proof"):
        unit.enrich_old_replay_proof(conflicting, conflicting, expected, SYZYGY)


def test_literal_bt4_root_prefix_proof_and_exact_old_witness() -> None:
    identity = _identity()
    actual = _actual(identity)
    game = unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                                replay=_replay(identity),
                                syzygy_inventory_sha256=SYZYGY)
    assert [row["uid"][-1] for row, _ in game.rows] == [16, 17]
    old_rows = _old_rows(game)
    unit.compare_old_witness(game, old_rows=old_rows, old_proof=_old_proof(game),
                             old_routes=("BT4", "BT4"),
                             route=lambda _: "BT4", expected=identity,
                             syzygy_inventory_sha256=SYZYGY)
    old_raw = _old_proof(game)
    missing_terminal = dict(old_raw["terminal_fact"])
    del missing_terminal["source_strict_receipt_sha256"]
    with pytest.raises(unit.Hold, match="exact old terminal projection"):
        unit.compare_old_witness(
            game, old_rows=old_rows,
            old_proof={**old_raw, "terminal_fact": missing_terminal},
            old_routes=("BT4", "BT4"), route=lambda _: "BT4",
            expected=identity, syzygy_inventory_sha256=SYZYGY)
    contradictory = {**old_raw, "terminal_fact":
                     {**old_raw["terminal_fact"],
                      "syzygy_inventory_sha256": "0" * 64}}
    with pytest.raises(unit.Hold, match="exact old terminal projection"):
        unit.compare_old_witness(
            game, old_rows=old_rows, old_proof=contradictory,
            old_routes=("BT4", "BT4"), route=lambda _: "BT4",
            expected=identity, syzygy_inventory_sha256=SYZYGY)
    changed_proof = {**game.proof, "terminal_fact":
                     {**game.proof["terminal_fact"],
                      "syzygy_inventory_sha256": "0" * 64}}
    changed_rows = tuple(({**row, "syzygy_inventory_sha256": "0" * 64}, native)
                         for row, native in game.rows)
    with pytest.raises(unit.Hold, match="checked old proof/inventory"):
        unit.compare_old_witness(
            unit.BoundGame(changed_rows, changed_proof),
            old_rows=old_rows, old_proof=old_raw,
            old_routes=("BT4", "BT4"), route=lambda _: "BT4",
            expected=identity, syzygy_inventory_sha256=SYZYGY)
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.compare_old_witness(game, old_rows=old_rows,
                                 old_proof=_old_proof(game),
                                 old_routes=("wrong", "BT4"),
                                 route=lambda _: "BT4", expected=identity,
                                 syzygy_inventory_sha256=SYZYGY)
    changed = list(old_rows)
    note, native = changed[1]
    changed[1] = ({**note, "outcome": "0-1"}, native)
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.compare_old_witness(game, old_rows=tuple(changed),
                                 old_proof=_old_proof(game),
                                 old_routes=("BT4", "BT4"),
                                 route=lambda _: "BT4", expected=identity,
                                 syzygy_inventory_sha256=SYZYGY)
    deleted = list(old_rows)
    note, native = deleted[0]
    deleted[0] = ({key: value for key, value in note.items()
                   if key != "provenance_locator_sha256"}, native)
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.compare_old_witness(game, old_rows=tuple(deleted),
                                 old_proof=_old_proof(game),
                                 old_routes=("BT4", "BT4"),
                                 route=lambda _: "BT4", expected=identity,
                                 syzygy_inventory_sha256=SYZYGY)
    wrong_new = list(game.rows)
    note, native = wrong_new[0]
    wrong_new[0] = ({**note, "game_proof_sha256": "0" * 64}, native)
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.compare_old_witness(unit.BoundGame(tuple(wrong_new), game.proof),
                                 old_rows=old_rows, old_proof=_old_proof(game),
                                 old_routes=("BT4", "BT4"),
                                 route=lambda _: "BT4", expected=identity,
                                 syzygy_inventory_sha256=SYZYGY)
    wrong_new[0] = ({**note, "syzygy_inventory_sha256": "0" * 64}, native)
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.compare_old_witness(unit.BoundGame(tuple(wrong_new), game.proof),
                                 old_rows=old_rows, old_proof=_old_proof(game),
                                 old_routes=("BT4", "BT4"),
                                 route=lambda _: "BT4", expected=identity,
                                 syzygy_inventory_sha256=SYZYGY)


def test_wrong_root_identity_uid_and_provisional_proof_refused() -> None:
    identity = _identity()
    actual = _actual(identity)
    actual.root["uci_prefix"][0] = "d2d4"
    with pytest.raises(unit.Hold, match="exact BT4 game/root"):
        unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                             replay=_replay(identity),
                             syzygy_inventory_sha256=SYZYGY)
    actual = _actual(identity)
    with pytest.raises(ValueError, match="literal replay UID"):
        unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                             replay=_replay(identity, wrong_uid=True),
                             syzygy_inventory_sha256=SYZYGY)
    with pytest.raises(RuntimeError, match="provisional proof"):
        unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                             replay=_replay(identity, fail_after_proof=True),
                             syzygy_inventory_sha256=SYZYGY)
    with pytest.raises(ValueError, match="archive snapshot byte identity"):
        unit.read_bound_game(actual, identity, snapshot=lambda _: b"tampered",
                             replay=_replay(identity),
                             syzygy_inventory_sha256=SYZYGY)


def test_frozen_planner_npz_snapshot_and_replay_binding_is_single_read(
        tmp_path: Path) -> None:
    identity = _identity(rows=22)
    actual = _actual(identity)
    actual.path = tmp_path / "selected-game.npz"
    roster = tuple([actual] + [
        SimpleNamespace(source="BT4-v9", game_id=index + 8,
                        gross_rows=108) for index in range(255)] + [
        SimpleNamespace(source="Ceres-v8", game_id=index,
                        gross_rows=1) for index in range(256)])
    assert sum(item.gross_rows for item in roster
               if item.source == "BT4-v9") == 27_562
    names = ("adapter_grouped", "grouped_reader", "neural_replay",
             "supervisor", "replay_bridge", "tri_bridge", "gate_bridge",
             "row_bridge", "terminal_core", "tri_syzygy")
    modules: dict[str, ModuleType] = {}
    pins = []
    for name in names:
        path = tmp_path / f"{name}.py"
        path.write_text(f"# synthetic {name}\n")
        module = ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module
        pins.append(unit.FrozenFile(name, path, unit.sha(path.read_bytes())))
    setattr(modules["adapter_grouped"], "plan_from_pinned_metadata",
            lambda _: roster)

    class Meter:
        def __init__(self, _: int) -> None:
            self.bytes = 0
            self.paths: set[str] = set()

    calls = []

    def snapshot(archive, meter: Meter) -> bytes:
        assert archive.source == "BT4-v9"
        assert archive.path == actual.path
        assert archive.digest == identity.archive_sha256
        assert archive.games == (identity.game_id,)
        meter.bytes += len(ARCHIVE)
        meter.paths.add(str(archive.path))
        calls.append("snapshot")
        return ARCHIVE

    setattr(modules["grouped_reader"], "LogicalReadMeter", Meter)
    setattr(modules["grouped_reader"], "snapshot", snapshot)
    setattr(modules["supervisor"], "Archive", lambda *values: SimpleNamespace(
        source=values[0], path=values[1], digest=values[2], games=values[3]))

    def replay_one(source, selected, raw, **kwargs):
        assert source == "BT4-v9"
        assert selected is actual
        calls.append("replay")
        def raw_old_sink(proof: dict) -> None:
            terminal = dict(proof["terminal_fact"])
            del terminal["syzygy_inventory_sha256"]
            kwargs["proof_sink"]({**proof, "terminal_fact": terminal})
        return _replay(identity)(selected, raw, raw_old_sink)

    setattr(modules["neural_replay"], "replay_one", replay_one)
    old = unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                               replay=_replay(identity),
                               syzygy_inventory_sha256=SYZYGY)
    old_terminal = dict(old.proof["terminal_fact"])
    del old_terminal["syzygy_inventory_sha256"]
    assert set(old_terminal) == ACTUAL_1388_TERMINAL_KEYS
    assert len(ACTUAL_1388_PROOF_LINE_SHA256) == 64
    old_proof = {**old.proof, "terminal_fact": old_terminal}
    verifier_path = tmp_path / "raw_strict_bt4.py"
    verifier_path.write_text("def verify_raw(*args):\n    return True\n")
    namespace: dict = {}
    exec(compile(verifier_path.read_text(), str(verifier_path), "exec"), namespace)
    gate = SimpleNamespace(inventory_sha256=SYZYGY)
    verifier_sha = unit.sha(verifier_path.read_bytes())
    game, logical = unit.read_frozen_bt4_game(
        identity, adapter=modules["adapter_grouped"],
        grouped_reader=modules["grouped_reader"],
        neural_replay=modules["neural_replay"],
        supervisor=modules["supervisor"],
        imported_modules={name: modules[name] for name in names[4:]},
        module_pins=tuple(pins), gate=gate,
        verifier=namespace["verify_raw"], verifier_sha256=verifier_sha,
        verifier_path=verifier_path,
        strict_context={"gate": gate,
                        "source_verifier_sha256": verifier_sha,
                        "strict_receipt_sha256": STRICT},
        syzygy_inventory_sha256=SYZYGY,
        old_rows=_old_rows(old), old_proof=old_proof,
        old_routes=("BT4",) * identity.rows,
        route=lambda _: "BT4")
    assert game.rows == old.rows
    assert game.proof == old.proof
    assert logical == len(ARCHIVE)
    assert calls == ["snapshot", "replay"]
    gate.inventory_sha256 = "0" * 64
    with pytest.raises(unit.Hold, match="owned strict gate/context/inventory"):
        unit.read_frozen_bt4_game(
            identity, adapter=modules["adapter_grouped"],
            grouped_reader=modules["grouped_reader"],
            neural_replay=modules["neural_replay"],
            supervisor=modules["supervisor"],
            imported_modules={name: modules[name] for name in names[4:]},
            module_pins=tuple(pins), gate=gate,
            verifier=namespace["verify_raw"], verifier_sha256=verifier_sha,
            verifier_path=verifier_path,
            strict_context={"gate": gate,
                            "source_verifier_sha256": verifier_sha,
                            "strict_receipt_sha256": STRICT},
            syzygy_inventory_sha256=SYZYGY,
            old_rows=_old_rows(old), old_proof=old_proof,
            old_routes=("BT4",) * identity.rows,
            route=lambda _: "BT4")
    gate.inventory_sha256 = SYZYGY
    (tmp_path / "tri_bridge.py").write_text("# changed\n")
    with pytest.raises(unit.Hold, match="pinned file changed/SHA"):
        unit.read_frozen_bt4_game(
            identity, adapter=modules["adapter_grouped"],
            grouped_reader=modules["grouped_reader"],
            neural_replay=modules["neural_replay"],
            supervisor=modules["supervisor"],
            imported_modules={name: modules[name] for name in names[4:]},
            module_pins=tuple(pins), gate=gate,
            verifier=namespace["verify_raw"], verifier_sha256=verifier_sha,
            verifier_path=verifier_path,
            strict_context={"gate": gate,
                            "source_verifier_sha256": verifier_sha,
                            "strict_receipt_sha256": STRICT},
            syzygy_inventory_sha256=SYZYGY,
            old_rows=_old_rows(old), old_proof=old_proof,
            old_routes=("BT4",) * identity.rows,
            route=lambda _: "BT4")
    assert calls == ["snapshot", "replay"]


def test_fsynced_game_output_and_old_witness_file_join(tmp_path: Path) -> None:
    identity = _identity()
    actual = _actual(identity)
    game = unit.read_bound_game(actual, identity, snapshot=lambda _: ARCHIVE,
                                replay=_replay(identity),
                                syzygy_inventory_sha256=SYZYGY)
    payloads = {
        "rows.jsonl": b"".join(unit.canonical(row) for row, _ in _old_rows(game)),
        "native.bin": b"".join(native for _, native in game.rows),
        "proof.json": unit.canonical(_old_proof(game)),
        "routes.json": unit.canonical(["BT4"] * identity.rows),
    }
    for name, raw in payloads.items():
        (tmp_path / name).write_bytes(raw)
    files = unit.WitnessFiles(
        tmp_path / "rows.jsonl", unit.sha(payloads["rows.jsonl"]),
        tmp_path / "native.bin", unit.sha(payloads["native.bin"]),
        tmp_path / "proof.json", unit.sha(payloads["proof.json"]),
        tmp_path / "routes.json", unit.sha(payloads["routes.json"]))
    witness = unit.load_old_witness(files, identity)
    assert witness.rows == _old_rows(game)
    script = tmp_path / "unused_worker.py"
    script.write_text("# no launch\n")
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = unit.UnitClaim(
        source_sha256="d" * 64, config_sha256="e" * 64,
        route_sha256="f" * 64, witness_sha256=files.identity(),
        local_code_sha256=unit.local_code_hashes(),
        expected=identity, syzygy_inventory_sha256=SYZYGY,
        guard_sha256=unit.sha(GUARD.read_bytes()),
        worker_sha256=unit.sha(script.read_bytes()),
        interpreter_sha256=unit.sha(Path(argv[0]).read_bytes()),
        command_sha256=unit.sha(unit.canonical(argv)))
    attempt = tmp_path / "attempt_0000"
    attempt.mkdir()
    result = unit.write_game_output(attempt, claim, game, len(ARCHIVE), 0)
    assert result["outputs"]["native.bin"] == unit.sha(payloads["native.bin"])
    emitted = {name: (attempt / name).read_bytes() for name in unit.OUTPUT_NAMES}
    unit.verify_game_output(emitted, claim, witness, lambda _: "BT4")
    (attempt / "native.bin").write_bytes(b"changed" + payloads["native.bin"][7:])
    with pytest.raises(unit.Hold, match="old UID/native/context/outcome/route"):
        unit.verify_game_output(
            {name: (attempt / name).read_bytes() for name in unit.OUTPUT_NAMES},
            claim, witness, lambda _: "BT4")


_WORKER = '''from pathlib import Path
import hashlib, json, sys
attempt = Path(sys.argv[-1])
claim = (attempt.parent / "CLAIM.json").read_bytes()
payload = b"bad" if attempt.name == "attempt_0000" else b"ok"
(attempt / "payload.bin").write_bytes(payload)
result = {"schema": "bt4_npz_owned_one_game_result_v1",
          "claim_sha256": hashlib.sha256(claim).hexdigest(),
          "rows": 2, "logical_archive_bytes": 24,
          "logical_context_bytes": 0,
          "outputs": {"payload.bin": hashlib.sha256(payload).hexdigest()}}
(attempt / "RESULT.json").write_bytes((json.dumps(result, sort_keys=True,
    separators=(",", ":")) + "\\n").encode())
'''


def _claim(script: Path, argv: tuple[str, ...], *, wall: int = 5) -> unit.UnitClaim:
    return unit.UnitClaim(
        source_sha256="d" * 64, config_sha256="e" * 64,
        route_sha256="f" * 64, witness_sha256="1" * 64,
        local_code_sha256=unit.local_code_hashes(),
        expected=_identity(), syzygy_inventory_sha256=SYZYGY,
        guard_sha256=unit.sha(GUARD.read_bytes()),
        worker_sha256=unit.sha(script.read_bytes()),
        interpreter_sha256=unit.sha(Path(argv[0]).read_bytes()),
        command_sha256=unit.sha(unit.canonical(argv)), wall_seconds=wall)


def test_owned_resume_noop_and_corruption_hold(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)

    local = {**claim.local_code_sha256,
             "checkpointed_cursor_v2": "0" * 64}
    with pytest.raises(unit.Hold, match="pinned file changed/SHA"):
        unit.run_owned_unit(tmp_path, replace(claim, local_code_sha256=local),
                            argv=argv, output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert not (tmp_path / "CLAIM.json").exists()

    def verify(payloads: dict[str, bytes]) -> None:
        unit.need(payloads["payload.bin"] == b"ok",
                  "old witness differs")

    with pytest.raises(unit.Hold, match="old witness differs"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=verify)
    complete = unit.run_owned_unit(tmp_path, claim, argv=argv,
                                   output_names=("payload.bin",),
                                   verify_output=verify)
    assert complete["attempt"] == "attempt_0001"
    assert unit.run_owned_unit(tmp_path, claim, argv=argv,
                               output_names=("payload.bin",),
                               verify_output=verify) == complete
    (tmp_path / "attempt_0001" / "payload.bin").write_bytes(b"evil")
    with pytest.raises(unit.Hold, match="pinned file changed/SHA"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=verify)
    assert not (tmp_path / "attempt_0002").exists()


def test_symlink_unit_root_refused_before_claim(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    script = tmp_path / "unused.py"
    script.write_text("# no launch\n")
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    with pytest.raises(unit.Hold, match="owned unit path"):
        unit.run_owned_unit(linked, _claim(script, argv), argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert not (actual / "CLAIM.json").exists()


def _dead_or_zombie(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


def _await_file(path: Path) -> None:
    for _ in range(100):
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"missing child PID: {path}")


def test_watchdog_kills_process_tree(tmp_path: Path) -> None:
    script = tmp_path / "sleeping_worker.py"
    script.write_text('''from pathlib import Path
import subprocess, sys, time
attempt = Path(sys.argv[-1])
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
(attempt / "grandchild.pid").write_text(str(grandchild.pid))
time.sleep(30)
''')
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv, wall=1)
    with pytest.raises(unit.Hold, match="wall cap"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    pid_file = tmp_path / "attempt_0000" / "grandchild.pid"
    _await_file(pid_file)
    pid = int(pid_file.read_text())
    for _ in range(100):
        if _dead_or_zombie(pid):
            break
        time.sleep(0.02)
    assert _dead_or_zombie(pid)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_late_child_exit_cannot_seal_complete(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "quick_worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv, wall=1)
    state = {"exited": False}

    def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert all(env[key] == "1" for key in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert all(key not in env for key in
                   ("PYTHONOPTIMIZE", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT"))
        process = cast(subprocess.Popen[bytes], subprocess.Popen(*args, **kwargs))
        real_poll = process.poll

        def tracked_poll() -> int | None:
            status = real_poll()
            if status is not None:
                state["exited"] = True
            return status

        process.poll = tracked_poll
        return process

    monkeypatch.setattr(unit, "subprocess", SimpleNamespace(Popen=spawn))
    monkeypatch.setattr(unit, "time", SimpleNamespace(
        monotonic=lambda: 100.0 + (2.0 if state["exited"] else 0.0),
        sleep=time.sleep))
    with pytest.raises(unit.Hold, match="wall cap"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert state["exited"]
    assert (tmp_path / "attempt_0000" / "RESULT.json").exists()
    assert not (tmp_path / "COMPLETE.json").exists()


def test_parent_alarm_interrupts_sealing_without_complete(tmp_path: Path) -> None:
    script = tmp_path / "quick_worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv, wall=1)

    def blocked_verify(_payloads: dict[str, bytes]) -> None:
        time.sleep(30)

    with pytest.raises(unit.Hold, match="wall cap"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=blocked_verify)
    assert (tmp_path / "attempt_0000" / "RESULT.json").exists()
    assert not (tmp_path / "COMPLETE.json").exists()


def test_monitor_exception_kills_child_and_descendant(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "spawn_then_sleep.py"
    script.write_text('''from pathlib import Path
import subprocess, sys, time
attempt = Path(sys.argv[-1])
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
(attempt / "grandchild.pid").write_text(str(grandchild.pid))
time.sleep(30)
''')
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)
    pid_file = tmp_path / "attempt_0000" / "grandchild.pid"
    child_pids: list[int] = []

    def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = cast(subprocess.Popen[bytes], subprocess.Popen(*args, **kwargs))
        child_pids.append(process.pid)
        return process

    def broken_meter(_pid: int) -> int:
        _await_file(pid_file)
        raise RuntimeError("synthetic monitor failure")

    monkeypatch.setattr(unit, "subprocess", SimpleNamespace(Popen=spawn))
    monkeypatch.setattr(unit, "_rss_bytes", broken_meter)
    with pytest.raises(RuntimeError, match="synthetic monitor failure"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert child_pids
    grandchild_pid = int(pid_file.read_text())
    for _ in range(100):
        if (_dead_or_zombie(child_pids[0]) and
                _dead_or_zombie(grandchild_pid)):
            break
        time.sleep(0.02)
    assert _dead_or_zombie(child_pids[0])
    assert _dead_or_zombie(grandchild_pid)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_child_alarm_survives_exec(tmp_path: Path) -> None:
    script = tmp_path / "sleep_after_exec.py"
    script.write_text("import time; time.sleep(30)\n")
    interpreter = str(Path(sys.executable).resolve())
    env = {**os.environ, "BT4_EXPECTED_OWNER_PID": str(os.getpid()),
           "BT4_CHILD_WALL_SECONDS": "1",
           "BT4_INTERPRETER_PATH": interpreter}
    child = subprocess.Popen(
        [interpreter, str(GUARD), str(script), str(tmp_path / "unused")],
        start_new_session=True,
        env=env)
    assert child.wait(timeout=5) == -signal.SIGALRM


def test_parent_death_signal_kills_child(tmp_path: Path) -> None:
    outer_script = tmp_path / "outer.py"
    marker = tmp_path / "child.pid"
    ready = tmp_path / "child.ready"
    worker = tmp_path / "guarded_worker.py"
    worker.write_text('''from pathlib import Path
import sys, time
Path(sys.argv[1]).touch()
time.sleep(30)
''')
    outer_script.write_text('''from pathlib import Path
import os, subprocess, sys, time
owner_pid = os.getpid()
env = {**os.environ, "BT4_EXPECTED_OWNER_PID": str(owner_pid),
       "BT4_CHILD_WALL_SECONDS": "30",
       "BT4_INTERPRETER_PATH": sys.executable}
child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]],
                         start_new_session=True, env=env)
Path(sys.argv[4]).write_text(str(child.pid))
time.sleep(30)
''')
    outer = subprocess.Popen([sys.executable, str(outer_script),
                              str(GUARD), str(worker), str(ready), str(marker)])
    try:
        _await_file(marker)
        _await_file(ready)
        child_pid = int(marker.read_text())
        os.kill(outer.pid, signal.SIGKILL)
        outer.wait(timeout=5)
        for _ in range(100):
            if _dead_or_zombie(child_pid):
                break
            time.sleep(0.02)
        assert _dead_or_zombie(child_pid)
    finally:
        if outer.poll() is None:
            outer.kill()
            outer.wait()


def test_exec_guard_refuses_wrong_owner(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    script = tmp_path / "touch.py"
    script.write_text("from pathlib import Path; import sys; Path(sys.argv[1]).touch()")
    interpreter = str(Path(sys.executable).resolve())
    env = {**os.environ, "BT4_EXPECTED_OWNER_PID": str(os.getpid() + 1),
           "BT4_CHILD_WALL_SECONDS": "5",
           "BT4_INTERPRETER_PATH": interpreter}
    child = subprocess.Popen(
        [interpreter, str(GUARD), str(script), str(marker)],
        start_new_session=True, env=env)
    assert child.wait(timeout=5) == 127
    assert not marker.exists()


def test_alarm_pending_at_spawn_return_reaps_owned_child(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "sleeping_worker.py"
    script.write_text("import time; time.sleep(30)\n")
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)
    children: list[subprocess.Popen[bytes]] = []

    def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = cast(subprocess.Popen[bytes], subprocess.Popen(*args, **kwargs))
        children.append(process)
        # Reproduce expiry after OS launch but before Popen is assigned.
        os.kill(os.getpid(), signal.SIGALRM)
        return process

    monkeypatch.setattr(unit, "subprocess", SimpleNamespace(Popen=spawn))
    with pytest.raises(unit.Hold, match="wall cap"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert _dead_or_zombie(children[0].pid)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_resume_parses_authenticated_result_snapshot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)
    seen: list[bytes] = []

    def verify(payloads: dict[str, bytes]) -> None:
        seen.append(payloads["payload.bin"])

    complete = unit.run_owned_unit(
        tmp_path, claim, argv=argv, output_names=("payload.bin",),
        verify_output=verify)
    original_read = Path.read_bytes

    def reject_result_reread(path: Path) -> bytes:
        assert path.name != "RESULT.json", "must parse the authenticated bytes"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", reject_result_reread)
    assert unit.run_owned_unit(
        tmp_path, claim, argv=argv, output_names=("payload.bin",),
        verify_output=verify) == complete
    assert seen == [b"bad", b"bad"]


def test_deadline_during_receipt_publication_removes_completion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)
    original_write = unit.wave.atomic_write

    def expire_after_publication(path: Path, raw: bytes) -> None:
        original_write(path, raw)
        if path.name == "COMPLETE.json":
            raise unit.Hold("wall cap")

    monkeypatch.setattr(unit.wave, "atomic_write", expire_after_publication)
    with pytest.raises(unit.Hold, match="wall cap"):
        unit.run_owned_unit(tmp_path, claim, argv=argv,
                            output_names=("payload.bin",),
                            verify_output=lambda _: None)
    assert (tmp_path / "attempt_0000" / "RESULT.json").exists()
    assert not (tmp_path / "COMPLETE.json").exists()


def test_preblocked_owner_alarm_refused_before_work() -> None:
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        with pytest.raises(unit.Hold, match="exclusive owner alarm"):
            with unit.owned_deadline(1):
                raise AssertionError("blocked owner alarm admitted")
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def test_concurrent_root_owner_refused_and_released(tmp_path: Path) -> None:
    ready = tmp_path / "owner.ready"
    owner = subprocess.Popen([
        sys.executable, "-c",
        "import fcntl, pathlib, sys, time; "
        "lock = open(sys.argv[1], 'w'); "
        "fcntl.flock(lock, fcntl.LOCK_EX); "
        "pathlib.Path(sys.argv[2]).touch(); time.sleep(30)",
        str(tmp_path / ".bt4_owner.lock"), str(ready)])
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    argv = (str(Path(sys.executable).resolve()), str(GUARD), str(script))
    claim = _claim(script, argv)
    try:
        _await_file(ready)
        with pytest.raises(unit.Hold, match="already has an owner"):
            unit.run_owned_unit(tmp_path, claim, argv=argv,
                                output_names=("payload.bin",),
                                verify_output=lambda _: None)
        assert not (tmp_path / "CLAIM.json").exists()
        assert not (tmp_path / "attempt_0000").exists()
    finally:
        owner.kill()
        owner.wait(timeout=5)
    complete = unit.run_owned_unit(
        tmp_path, claim, argv=argv, output_names=("payload.bin",),
        verify_output=lambda _: None)
    assert complete["attempt"] == "attempt_0000"


def test_frozen_source_executes_pinned_bytes_not_timestamp_valid_pyc(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "_bt4_snapshot_source_fixture"
    path = tmp_path / f"{name}.py"
    path.write_text("VALUE = 'stale'\n")
    stamp = path.stat()
    cached = py_compile.compile(str(path), doraise=True)
    assert cached is not None and Path(cached).is_file()
    # Same size and mtime make the old pyc eligible to an ordinary import.
    path.write_text("VALUE = 'fresh'\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    pin = unit.FrozenFile(name, path, unit.sha(path.read_bytes()))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with unit.source_snapshot_imports((pin,)) as executed:
            module = importlib.import_module(name)
            assert module.VALUE == "fresh"
            assert executed == {path}
            unit.verify_modules({name: module}, (pin,))
    finally:
        sys.modules.pop(name, None)


def test_frozen_source_refuses_ambient_path_before_execution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "_bt4_snapshot_path_fixture"
    pinned = tmp_path / "pinned"
    ambient = tmp_path / "ambient"
    pinned.mkdir()
    ambient.mkdir()
    path = pinned / f"{name}.py"
    path.write_text("VALUE = 'pinned'\n")
    marker = tmp_path / "should-not-exist"
    (ambient / f"{name}.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).touch()\n")
    pin = unit.FrozenFile(name, path, unit.sha(path.read_bytes()))
    monkeypatch.syspath_prepend(str(ambient))
    try:
        with unit.source_snapshot_imports((pin,)):
            with pytest.raises(unit.Hold, match="path before execution"):
                importlib.import_module(name)
        assert not marker.exists()
    finally:
        sys.modules.pop(name, None)
