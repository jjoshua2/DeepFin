"""Private-packet BT4 one-game child; never selects an archive on its own.

The owner supplies a SHA-bound packet and watches this process for at most
600 seconds. This entrypoint has no corpus, target, trainer, or GPU authority.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

from chess_anti_engine.source import bt4_npz_unit_v1 as unit
from chess_anti_engine.source import checkpointed_candidate_v2 as candidate


def _file_ref(value: object) -> tuple[Path, str]:
    if type(value) is not dict:
        raise unit.Hold("private packet file reference")
    fields = cast(dict[str, object], value)
    path_text = fields.get("path")
    digest = fields.get("sha256")
    if (set(fields) != {"path", "sha256"} or
            type(path_text) is not str or type(digest) is not str or
            not unit.hex64(digest)):
        raise unit.Hold("private packet file reference")
    path = Path(path_text)
    unit.need(path.is_absolute(), "private packet absolute path")
    return path, digest


def _witness_files(value: object) -> unit.WitnessFiles:
    if type(value) is not dict:
        raise unit.Hold("old witness file references")
    fields = cast(dict[str, object], value)
    unit.need(set(fields) == {"rows_jsonl", "native_bin", "proof_json",
                              "routes_json"}, "old witness file references")
    rows, rows_sha = _file_ref(fields["rows_jsonl"])
    native, native_sha = _file_ref(fields["native_bin"])
    proof, proof_sha = _file_ref(fields["proof_json"])
    routes, routes_sha = _file_ref(fields["routes_json"])
    return unit.WitnessFiles(rows, rows_sha, native, native_sha,
                             proof, proof_sha, routes, routes_sha)


def run(packet_path: Path, attempt: Path) -> dict:
    unit.need(sys.flags.optimize == 0, "unoptimized owned interpreter")
    unit.need(packet_path.is_absolute() and attempt.is_absolute() and
              attempt.is_dir() and attempt.parent.joinpath("CLAIM.json").exists(),
              "owned packet/attempt path")
    claim = unit.claim_from_bytes(unit.bounded_file(
        attempt.parent / "CLAIM.json", 1 << 20))
    packet_raw = unit.pinned_file(packet_path, claim.config_sha256, 1 << 20)
    decoded = json.loads(packet_raw)
    if type(decoded) is not dict:
        raise unit.Hold("canonical private BT4 packet")
    packet = cast(dict[str, object], decoded)
    unit.need(packet_raw == unit.canonical(packet) and
              packet.get("schema") == "bt4_npz_one_game_source_packet_v1" and
              packet.get("witness_schema") == unit.WITNESS_SCHEMA and
              set(packet) == {"schema", "imports", "verifier",
                              "witness", "witness_schema"},
              "canonical private BT4 packet")
    verifier_pin_path, verifier_pin_sha = _file_ref(packet["verifier"])
    unit.pinned_file(verifier_pin_path, verifier_pin_sha, 1 << 20)
    refs_obj = packet["imports"]
    if type(refs_obj) is not list or len(refs_obj) < 11:
        raise unit.Hold("complete frozen imported sources")
    refs = cast(list[object], refs_obj)
    pins = []
    for ref in refs:
        if type(ref) is not dict:
            raise unit.Hold("frozen import reference")
        fields = cast(dict[str, object], ref)
        name = fields.get("module")
        unit.need(set(fields) == {"module", "path", "sha256"} and
                  type(name) is str and name.isidentifier(),
                  "frozen import reference")
        if type(name) is not str:
            raise unit.Hold("frozen import name")
        path, digest = _file_ref({"path": fields["path"],
                                  "sha256": fields["sha256"]})
        unit.pinned_file(path, digest, 1 << 20)
        pins.append(unit.FrozenFile(name, path, digest))
    unit.need(len({pin.module for pin in pins}) == len(pins) and
              unit.sha(unit.canonical({"imports": refs,
                                       "verifier": packet["verifier"]})) ==
                  claim.source_sha256,
              "frozen source list/claim")
    witness_files = _witness_files(packet["witness"])
    unit.need(witness_files.identity() == claim.witness_sha256,
              "old witness/claim")
    witness = unit.load_old_witness(witness_files, claim.expected)
    route_path = Path(candidate.__file__).resolve()
    unit.pinned_file(route_path, claim.route_sha256, 1 << 20)

    # Import paths are checked before execution and existing bytecode caches
    # cannot replace the exact source bytes authenticated by the private packet.
    source_pins = (*pins, unit.FrozenFile(
        "__bt4_raw_verifier__", verifier_pin_path, verifier_pin_sha))
    with unit.source_snapshot_imports(source_pins) as executed:
        for directory in reversed(tuple(dict.fromkeys(pin.path.parent for pin in pins))):
            sys.path.insert(0, str(directory))
        modules: dict[str, ModuleType] = {
            pin.module: importlib.import_module(pin.module) for pin in pins}
        unit.verify_modules(modules, tuple(pins))
        neural_child = modules["neural_child"]
        verifier, binder, verifier_sha = neural_child._load_verifier(unit.SOURCE)
        verifier_path = Path(verifier.__code__.co_filename)
        unit.need({pin.path for pin in source_pins} <= executed,
                  "frozen loader did not execute every authenticated source snapshot")
        unit.need(verifier_path.resolve() == verifier_pin_path.resolve() and
                  verifier_sha == verifier_pin_sha,
                  "accepted verifier differs from private source pins")
        grouped_reader = modules["grouped_reader"]
        # Reserve the full possible NPZ charge before reading replay context.
        # Packet, imports, witness, and executable reads have separate file caps.
        context_meter = grouped_reader.LogicalReadMeter(
            claim.archive_context_bytes - unit.MAX_ARCHIVE_BYTES)
        gate = neural_child._open_gate(unit.SOURCE)
        with gate:
            unit.need(gate.inventory_sha256 == claim.syzygy_inventory_sha256,
                      "opened strict Syzygy inventory")
            context = modules["replay_bridge"].bind_bt4_context(
                binder, gate, context_meter, verifier_sha)
            game, archive_bytes = unit.read_frozen_bt4_game(
                claim.expected, adapter=modules["adapter_grouped"],
                grouped_reader=grouped_reader,
                neural_replay=modules["neural_replay"],
                supervisor=modules["supervisor"],
                imported_modules={name: module for name, module in modules.items()
                                  if name not in {"adapter_grouped", "grouped_reader",
                                                  "neural_replay", "supervisor"}},
                module_pins=tuple(pins), gate=gate,
                verifier=verifier, verifier_sha256=verifier_sha,
                verifier_path=verifier_path, strict_context=context,
                syzygy_inventory_sha256=claim.syzygy_inventory_sha256,
                old_rows=witness.rows, old_proof=witness.proof,
                old_routes=witness.routes, route=candidate.route)
    unit.need(context_meter.bytes + archive_bytes <= claim.archive_context_bytes,
              "owned NPZ plus replay-context logical input cap")
    return unit.write_game_output(attempt, claim, game, archive_bytes,
                                  context_meter.bytes)


def main() -> int:
    unit.need(len(sys.argv) == 3, "private packet and owned attempt required")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
