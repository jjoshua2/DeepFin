"""One-game BT4 NPZ binding and owned CPU diagnostic unit.

The frozen adapter remains the only NPZ parser and strict replay remains the
only producer of source proof. This module grants no corpus or target credit.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from types import ModuleType
from typing import cast

from chess_anti_engine.source import checkpointed_candidate_v2 as candidate
from chess_anti_engine.source import checkpointed_cursor_v2 as cursor
from chess_anti_engine.source import checkpointed_sort as sort
from chess_anti_engine.source import checkpointed_wave_v2 as wave

SOURCE = "BT4-v9"
GUARD_PATH = Path(__file__).resolve().parents[2] / "scripts/bt4_npz_exec_guard.py"
CHILD_PATH = Path(__file__).resolve().parents[2] / "scripts/bt4_npz_one_game_child.py"
ROW_BYTES = 44_800
MAX_ROWS = 400
MAX_ARCHIVE_BYTES = 64 << 20
MAX_WALL_SECONDS = 600
MAX_RSS_BYTES = 4 << 30
MAX_OUTPUT_BYTES = 32 << 20
MAX_ARCHIVE_CONTEXT_BYTES = 128 << 20
MAX_METADATA_BYTES = 4096
WITNESS_SCHEMA = "bt4_tri_bridge_exact_row_projection_v1"
OLD_ROW_KEYS = frozenset({
    "uid", "source", "provenance_locator_sha256", "input_digest",
    "history_stack_sha256", "repetition", "rule50",
    "legal_context_sha256", "teacher_query_sha256", "outcome",
})
NEW_PROOF_KEYS = frozenset({"game_proof_sha256",
                            "syzygy_inventory_sha256"})
LOCAL_CODE_KEYS = frozenset({
    "bt4_npz_unit_v1", "checkpointed_cursor_v2",
    "checkpointed_wave_v2", "checkpointed_candidate_v2",
    "checkpointed_sort",
})
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class Hold(ValueError):
    """The source, witness, or owned-unit claim differs."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise Hold(why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def hex64(value: object) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


def _local_code_paths() -> dict[str, Path]:
    return {"bt4_npz_unit_v1": Path(__file__),
            "checkpointed_cursor_v2": Path(cursor.__file__),
            "checkpointed_wave_v2": Path(wave.__file__),
            "checkpointed_candidate_v2": Path(candidate.__file__),
            "checkpointed_sort": Path(sort.__file__)}


def local_code_hashes() -> dict[str, str]:
    return {name: sha(path.read_bytes())
            for name, path in _local_code_paths().items()}


def verify_local_code(pins: dict[str, str]) -> None:
    need(set(pins) == set(LOCAL_CODE_KEYS) and
         all(hex64(digest) for digest in pins.values()),
         "complete local reader/bridge code pins")
    for name, path in _local_code_paths().items():
        pinned_file(path.resolve(), pins[name], 1 << 20)


def pinned_file(path: Path, digest: str, cap: int) -> bytes:
    """Read a pinned regular file through no-follow components and exact SHA."""
    need(path.is_absolute() and hex64(digest) and 0 < cap <= MAX_ARCHIVE_BYTES,
         "pinned path/SHA/cap")
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                            os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=parent)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                 0 < before.st_size <= cap, "pinned file size/type")
            remaining = before.st_size
            blocks: list[bytes] = []
            while remaining:
                block = os.read(fd, min(remaining, 1 << 20))
                need(bool(block), "pinned file short read")
                blocks.append(block)
                remaining -= len(block)
            after = os.fstat(fd)
            def identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
                return (info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns)
            raw = b"".join(blocks)
            need(identity(before) == identity(after) and sha(raw) == digest,
                 "pinned file changed/SHA")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(parent)


@dataclass(frozen=True)
class FrozenFile:
    module: str
    path: Path
    sha256: str


def verify_modules(modules: dict[str, ModuleType], pins: tuple[FrozenFile, ...]) -> None:
    need(len(pins) == len({pin.module for pin in pins}), "duplicate source module")
    for pin in pins:
        module = modules.get(pin.module)
        file_name = getattr(module, "__file__", None)
        need(type(file_name) is str and
             Path(file_name).resolve() == pin.path.resolve(),
             f"frozen import path: {pin.module}")
        pinned_file(pin.path, pin.sha256, 1 << 20)


@dataclass(frozen=True)
class GameIdentity:
    manifest_sha256: str
    namespace: str
    root_id: str
    game_id: int
    archive_sha256: str
    strict_receipt_sha256: str
    rows: int
    root_prefix_uci: tuple[str, ...]


def locator_from_actual(actual: object, expected: GameIdentity) -> cursor.GameLocator:
    """Copy every source key; derive literal ply only from the pinned root."""
    root = getattr(actual, "root", None)
    prefix = root.get("uci_prefix") if type(root) is dict else None
    if type(prefix) is not list or not all(type(move) is str for move in prefix):
        raise Hold("exact BT4 game/root/16-ply identity")
    need(type(prefix) is list and tuple(prefix) == expected.root_prefix_uci and
         0 < len(prefix) <= 256 and
         getattr(actual, "source", None) == SOURCE and
         getattr(actual, "source_manifest_sha", None) == expected.manifest_sha256 and
         getattr(actual, "namespace", None) == expected.namespace and
         getattr(actual, "root_id", None) == expected.root_id and
         getattr(actual, "game_id", None) == expected.game_id and
         getattr(actual, "archive_sha", None) == expected.archive_sha256 and
         getattr(actual, "strict_receipt_sha", None) == expected.strict_receipt_sha256 and
         getattr(actual, "gross_rows", None) == expected.rows and
         0 < expected.rows <= MAX_ROWS and len(prefix) == 16,
         "exact BT4 game/root/16-ply identity")
    return cursor.GameLocator(
        SOURCE, expected.manifest_sha256, expected.namespace,
        expected.root_id, expected.game_id, expected.rows,
        expected.archive_sha256, expected.strict_receipt_sha256,
        ply_start=len(prefix), root_prefix_uci=tuple(prefix), actual=actual)


Snapshot = Callable[[object], bytes]
Replay = Callable[[object, bytes, Callable[[dict], None]],
                  tuple[tuple[dict, bytes], ...]]


@dataclass(frozen=True)
class BoundGame:
    rows: tuple[tuple[dict, bytes], ...]
    proof: dict


@dataclass(frozen=True)
class WitnessFiles:
    rows_jsonl: Path
    rows_sha256: str
    native_bin: Path
    native_sha256: str
    proof_json: Path
    proof_sha256: str
    routes_json: Path
    routes_sha256: str

    def identity(self) -> str:
        return sha(canonical({"schema": WITNESS_SCHEMA,
                              "row_keys": sorted(OLD_ROW_KEYS),
                              "rows": self.rows_sha256,
                              "native": self.native_sha256,
                              "proof": self.proof_sha256,
                              "routes": self.routes_sha256}))


@dataclass(frozen=True)
class OldWitness:
    rows: tuple[tuple[dict, bytes], ...]
    proof: dict
    routes: tuple[str, ...]


def _decode_rows(metadata: bytes, native: bytes,
                 expected: GameIdentity, *, enriched: bool
                 ) -> tuple[tuple[dict, bytes], ...]:
    lines = metadata.splitlines(keepends=True)
    need(len(lines) == expected.rows and
         len(native) == expected.rows * ROW_BYTES,
         "witness row/native count")
    result = []
    for index, line in enumerate(lines):
        need(0 < len(line) <= MAX_METADATA_BYTES + 1 and
             line.endswith(b"\n"), "bounded witness metadata row")
        note = json.loads(line)
        need(type(note) is dict and line == canonical(note) and
             set(note) == set(OLD_ROW_KEYS | NEW_PROOF_KEYS
                              if enriched else OLD_ROW_KEYS) and
             note.get("uid") == [expected.manifest_sha256,
                                 expected.namespace, expected.root_id,
                                 expected.game_id,
                                 len(expected.root_prefix_uci) + index] and
             note.get("source") == SOURCE,
             "literal old witness UID/order/source")
        result.append((note, native[index * ROW_BYTES:(index + 1) * ROW_BYTES]))
    return tuple(result)


def load_old_witness(files: WitnessFiles, expected: GameIdentity) -> OldWitness:
    metadata = pinned_file(files.rows_jsonl, files.rows_sha256,
                           MAX_ROWS * (MAX_METADATA_BYTES + 1))
    native = pinned_file(files.native_bin, files.native_sha256,
                         MAX_ROWS * ROW_BYTES)
    proof_raw = pinned_file(files.proof_json, files.proof_sha256, 4 << 20)
    routes_raw = pinned_file(files.routes_json, files.routes_sha256, 1 << 20)
    proof = json.loads(proof_raw)
    routes = json.loads(routes_raw)
    need(type(proof) is dict and canonical(proof) == proof_raw and
         type(routes) is list and len(routes) == expected.rows and
         all(type(value) is str and value for value in routes) and
         canonical(routes) == routes_raw,
         "canonical old proof/routes")
    return OldWitness(_decode_rows(metadata, native, expected,
                                   enriched=False),
                      proof, tuple(routes))


def _write_fsynced(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


OUTPUT_NAMES = ("rows.jsonl", "native.bin", "proof.json")


def write_game_output(attempt: Path, claim: UnitClaim,
                      game: BoundGame, logical_archive_bytes: int,
                      logical_context_bytes: int) -> dict:
    """Write the verified whole game and a claim-bound worker receipt."""
    need(attempt.is_absolute() and attempt.is_dir() and
         len(game.rows) == claim.expected.rows and
         0 < logical_archive_bytes <= MAX_ARCHIVE_BYTES and
         logical_context_bytes >= 0 and
         logical_archive_bytes + logical_context_bytes <= claim.archive_context_bytes,
         "complete output/physical archive cap")
    notes = [canonical(row) for row, _ in game.rows]
    need(all(0 < len(note) <= MAX_METADATA_BYTES + 1 for note in notes),
         "bounded emitted metadata row")
    rows = b"".join(notes)
    native = b"".join(data for _, data in game.rows)
    proof = canonical(game.proof)
    need(len(rows) <= claim.expected.rows * (MAX_METADATA_BYTES + 1) and
         len(native) == claim.expected.rows * ROW_BYTES and
         len(proof) <= 4 << 20,
         "one-game output sizes")
    payloads = dict(zip(OUTPUT_NAMES, (rows, native, proof), strict=True))
    for name, raw in payloads.items():
        _write_fsynced(attempt / name, raw)
    result = {"schema": "bt4_npz_owned_one_game_result_v1",
              "claim_sha256": sha(canonical(claim.as_json())),
              "rows": claim.expected.rows,
              "logical_archive_bytes": logical_archive_bytes,
              "logical_context_bytes": logical_context_bytes,
              "outputs": {name: sha(raw) for name, raw in payloads.items()}}
    _write_fsynced(attempt / "RESULT.json", canonical(result))
    wave.fsync_directory(attempt)
    return result


def verify_game_output(attempt: Path, claim: UnitClaim,
                       witness: OldWitness,
                       route: Callable[[list], object]) -> None:
    metadata = (attempt / OUTPUT_NAMES[0]).read_bytes()
    native = (attempt / OUTPUT_NAMES[1]).read_bytes()
    proof_raw = (attempt / OUTPUT_NAMES[2]).read_bytes()
    proof = json.loads(proof_raw)
    need(type(proof) is dict and canonical(proof) == proof_raw,
         "child proof canonical bytes")
    game = BoundGame(_decode_rows(metadata, native, claim.expected,
                                  enriched=True), proof)
    compare_old_witness(game, old_rows=witness.rows,
                        old_proof=witness.proof,
                        old_routes=witness.routes, route=route,
                        expected=claim.expected,
                        syzygy_inventory_sha256=claim.syzygy_inventory_sha256)


def read_bound_game(actual: object, expected: GameIdentity, *,
                    snapshot: Snapshot, replay: Replay,
                    syzygy_inventory_sha256: str) -> BoundGame:
    """A proof callback is provisional until complete replay and row validation."""
    loc = locator_from_actual(actual, expected)
    provisional: list[dict] = []

    def replay_once(inner: cursor.GameLocator, raw: bytes,
                    sink: Callable[[dict], None]
                    ) -> tuple[tuple[dict, bytes], ...]:
        need(inner == loc and not provisional, "replay called more than once")

        def capture(proof: dict) -> None:
            provisional.append(proof)
            sink(proof)

        return replay(actual, raw, capture)

    reader = cursor.ReplayGameReader(
        lambda inner: snapshot(inner.actual), replay_once,
        strict_receipt_sha256=expected.strict_receipt_sha256,
        syzygy_inventory_sha256=syzygy_inventory_sha256,
        archive_byte_cap=MAX_ARCHIVE_BYTES)
    rows = reader.read_game(loc)
    need(len(provisional) == 1 and
         all(len(native) == ROW_BYTES for _, native in rows),
         "whole-game proof/native length")
    return BoundGame(rows, provisional[0])


def compare_old_witness(game: BoundGame, *, old_rows: tuple[tuple[dict, bytes], ...],
                        old_proof: dict, old_routes: tuple[object, ...],
                        route: Callable[[list], object],
                        expected: GameIdentity,
                        syzygy_inventory_sha256: str) -> None:
    """Compare frozen bridge rows plus the separately checked proof/inventory."""
    terminal = game.proof.get("terminal_fact")
    old_terminal = old_proof.get("terminal_fact")
    if type(terminal) is not dict or type(old_terminal) is not dict:
        raise Hold("exact old terminal projection")
    need("syzygy_inventory_sha256" not in old_terminal and
         set(terminal) == set(old_terminal) | {"syzygy_inventory_sha256"},
         "exact old terminal projection")
    raw_projection = {**game.proof, "terminal_fact":
                      {key: value for key, value in terminal.items()
                       if key != "syzygy_inventory_sha256"}}
    need(len(game.rows) == len(old_rows) == len(old_routes) == expected.rows and
         set(game.proof) == set(old_proof) and
         canonical(raw_projection) == canonical(old_proof),
         "old whole-game proof/count differs")
    proof_sha = game.proof.get("source_proof_sha256")
    source_proof = game.proof.get("source_proof")
    inventory_sha = (terminal.get("syzygy_inventory_sha256")
                     if type(terminal) is dict else None)
    need(hex64(proof_sha) and
         inventory_sha == syzygy_inventory_sha256 and
         type(source_proof) is dict and
         sha(canonical(source_proof)) == proof_sha and
         source_proof.get("strict_receipt_sha256") ==
             expected.strict_receipt_sha256 and
         source_proof.get("archive_sha256") == expected.archive_sha256 and
         source_proof.get("root_id") == expected.root_id and
         source_proof.get("game_id") == expected.game_id,
         "checked old proof/inventory")
    for index, ((row, native), (old, old_native)) in enumerate(
            zip(game.rows, old_rows, strict=True)):
        uid = [expected.manifest_sha256, expected.namespace,
               expected.root_id, expected.game_id,
               len(expected.root_prefix_uci) + index]
        bridge_row = {key: value for key, value in row.items()
                      if key in OLD_ROW_KEYS}
        need(set(row) == set(OLD_ROW_KEYS | NEW_PROOF_KEYS) and
             set(old) == set(OLD_ROW_KEYS) and
             row.get("uid") == old.get("uid") == uid and
             row.get("game_proof_sha256") == proof_sha and
             row.get("syzygy_inventory_sha256") == inventory_sha and
             canonical(bridge_row) == canonical(old) and
             native == old_native and
             route(uid) == old_routes[index],
             f"old UID/native/context/outcome/route differs at {index}")


def enrich_old_replay_proof(raw_proof: dict, old_proof: dict,
                            expected: GameIdentity,
                            syzygy_inventory_sha256: str) -> dict:
    """Add one checked gate field for the new cursor; keep the old proof exact."""
    terminal = raw_proof.get("terminal_fact") if type(raw_proof) is dict else None
    if type(terminal) is not dict:
        raise Hold("raw old replay proof/gate inventory differs")
    need(hex64(syzygy_inventory_sha256) and
         "syzygy_inventory_sha256" not in terminal and
         canonical(raw_proof) == canonical(old_proof) and
         terminal.get("source_archive_sha256") == expected.archive_sha256 and
         terminal.get("source_strict_receipt_sha256") ==
             expected.strict_receipt_sha256 and
         terminal.get("terminal_replayed") is True and
         terminal.get("rows") == expected.rows,
         "raw old replay proof/gate inventory differs")
    return {**raw_proof, "terminal_fact":
            {**terminal,
             "syzygy_inventory_sha256": syzygy_inventory_sha256}}


def read_frozen_bt4_game(expected: GameIdentity, *,
                         adapter: ModuleType, grouped_reader: ModuleType,
                         neural_replay: ModuleType, supervisor: ModuleType,
                         imported_modules: dict[str, ModuleType],
                         module_pins: tuple[FrozenFile, ...], gate: object,
                         verifier: Callable, verifier_sha256: str,
                         verifier_path: Path,
                         strict_context: dict, syzygy_inventory_sha256: str,
                         old_rows: tuple[tuple[dict, bytes], ...],
                         old_proof: dict, old_routes: tuple[object, ...],
                         route: Callable[[list], object]) -> tuple[BoundGame, int]:
    """Call the frozen planner, no-follow NPZ reader and full strict replay."""
    modules = {"adapter_grouped": adapter,
               "grouped_reader": grouped_reader,
               "neural_replay": neural_replay,
               "supervisor": supervisor, **imported_modules}
    need({"adapter_grouped", "grouped_reader", "neural_replay",
          "supervisor", "replay_bridge", "tri_bridge", "gate_bridge",
          "row_bridge", "terminal_core", "tri_syzygy"}.issubset(modules) and
         set(modules) == {pin.module for pin in module_pins},
         "complete frozen source import pins")
    verify_modules(modules, module_pins)
    need(hex64(verifier_sha256) and
         verifier_path.resolve() ==
             Path(verifier.__code__.co_filename).resolve(),
         "accepted raw verifier path")
    pinned_file(verifier_path, verifier_sha256, 1 << 20)
    selected = cast(tuple[object, ...],
                    adapter.plan_from_pinned_metadata(SOURCE))
    need(type(selected) is tuple and len(selected) == 512 and
         sum(cast(int, getattr(item, "gross_rows")) for item in selected
             if getattr(item, "source", None) == SOURCE) == 27_562,
         "frozen selected BT4 roster")
    matches = [item for item in selected
               if getattr(item, "source", None) == SOURCE and
               getattr(item, "source_manifest_sha", None) ==
                   expected.manifest_sha256 and
               getattr(item, "namespace", None) == expected.namespace and
               getattr(item, "root_id", None) == expected.root_id and
               getattr(item, "game_id", None) == expected.game_id]
    need(len(matches) == 1, "selected BT4 game membership")
    actual = matches[0]
    loc = locator_from_actual(actual, expected)
    need(type(strict_context) is dict and
         strict_context.get("gate") is gate and
         strict_context.get("source_verifier_sha256") == verifier_sha256 and
         strict_context.get("strict_receipt_sha256") ==
             expected.strict_receipt_sha256 and
         getattr(gate, "inventory_sha256", None) == syzygy_inventory_sha256,
         "owned strict gate/context/inventory")
    meter = grouped_reader.LogicalReadMeter(MAX_ARCHIVE_BYTES)
    actual_path = getattr(actual, "path", None)
    need(isinstance(actual_path, Path), "selected NPZ locator path")

    def snapshot(inner: object) -> bytes:
        need(inner is actual and getattr(inner, "path", None) == actual_path,
             "exact selected NPZ path")
        archive = supervisor.Archive(SOURCE, actual_path,
                                     expected.archive_sha256,
                                     (expected.game_id,))
        raw = grouped_reader.snapshot(archive, meter)
        need(type(raw) is bytes and 0 < len(raw) <= MAX_ARCHIVE_BYTES and
             sha(raw) == expected.archive_sha256,
             "authenticated NPZ bytes")
        return raw

    def replay(inner: object, raw: bytes,
               sink: Callable[[dict], None]
               ) -> tuple[tuple[dict, bytes], ...]:
        need(inner is actual and loc.actual is actual,
             "replay selected locator")
        def add_checked_inventory(raw_proof: dict) -> None:
            need(getattr(gate, "inventory_sha256", None) ==
                     syzygy_inventory_sha256,
                 "raw old replay proof/gate inventory differs")
            sink(enrich_old_replay_proof(
                raw_proof, old_proof, expected, syzygy_inventory_sha256))

        return neural_replay.replay_one(
            SOURCE, actual, raw, gate=gate, verifier=verifier,
            verifier_sha256=verifier_sha256,
            strict_context=strict_context, proof_sink=add_checked_inventory)

    game = read_bound_game(actual, expected, snapshot=snapshot, replay=replay,
                           syzygy_inventory_sha256=syzygy_inventory_sha256)
    compare_old_witness(game, old_rows=old_rows, old_proof=old_proof,
                        old_routes=old_routes, route=route, expected=expected,
                        syzygy_inventory_sha256=syzygy_inventory_sha256)
    need(0 < meter.bytes <= MAX_ARCHIVE_BYTES and
         len(meter.paths) == 1, "one archive logical read")
    return game, meter.bytes


@dataclass(frozen=True)
class UnitClaim:
    source_sha256: str
    config_sha256: str
    route_sha256: str
    witness_sha256: str
    local_code_sha256: dict[str, str]
    expected: GameIdentity
    syzygy_inventory_sha256: str
    guard_sha256: str
    worker_sha256: str
    interpreter_sha256: str
    command_sha256: str
    wall_seconds: int = MAX_WALL_SECONDS
    rss_bytes: int = MAX_RSS_BYTES
    output_bytes: int = MAX_OUTPUT_BYTES
    # Only the NPZ snapshot and strict replay-context reads share this meter.
    # Packet, source imports, old witness, and interpreter have separate caps.
    archive_context_bytes: int = MAX_ARCHIVE_CONTEXT_BYTES

    def as_json(self) -> dict:
        return {"schema": "bt4_npz_owned_one_game_claim_v1",
                "credit": "CPU_DIAGNOSTIC_ZERO_CORPUS_TARGET_TRAINER_CREDIT",
                "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256,
                "route_sha256": self.route_sha256,
                "witness_sha256": self.witness_sha256,
                "local_code_sha256": self.local_code_sha256,
                "witness_schema": WITNESS_SCHEMA,
                "expected": {**self.expected.__dict__,
                             "root_prefix_uci": list(self.expected.root_prefix_uci)},
                "syzygy_inventory_sha256": self.syzygy_inventory_sha256,
                "guard_sha256": self.guard_sha256,
                "worker_sha256": self.worker_sha256,
                "interpreter_sha256": self.interpreter_sha256,
                "command_sha256": self.command_sha256,
                "wall_seconds": self.wall_seconds,
                "rss_bytes": self.rss_bytes,
                "archive_bytes": MAX_ARCHIVE_BYTES,
                "output_bytes": self.output_bytes,
                "archive_context_bytes": self.archive_context_bytes,
                "row_bytes": ROW_BYTES, "max_rows": MAX_ROWS}

    def validate(self) -> None:
        need(all(hex64(value) for value in (
            self.source_sha256, self.config_sha256, self.route_sha256,
            self.witness_sha256, self.syzygy_inventory_sha256,
            self.guard_sha256, self.worker_sha256,
            self.interpreter_sha256,
            self.command_sha256, self.expected.manifest_sha256,
            self.expected.archive_sha256,
            self.expected.strict_receipt_sha256)), "unit SHA pins")
        verify_local_code(self.local_code_sha256)
        need(0 < self.expected.rows <= MAX_ROWS and
             0 < self.wall_seconds <= MAX_WALL_SECONDS and
             0 < self.rss_bytes <= MAX_RSS_BYTES and
             0 < self.output_bytes <= MAX_OUTPUT_BYTES and
             MAX_ARCHIVE_BYTES < self.archive_context_bytes <=
                 MAX_ARCHIVE_CONTEXT_BYTES,
             "unit resource caps")


def claim_from_bytes(raw: bytes) -> UnitClaim:
    decoded = json.loads(raw)
    if type(decoded) is not dict:
        raise Hold("canonical unit claim")
    value = cast(dict[str, object], decoded)
    need(raw == canonical(value) and
         value.get("schema") == "bt4_npz_owned_one_game_claim_v1",
         "canonical unit claim")
    game_obj = value.get("expected")
    if type(game_obj) is not dict:
        raise Hold("unit expected game")
    game = cast(dict[str, object], game_obj)
    prefix_obj = game.get("root_prefix_uci")
    if type(prefix_obj) is not list or not all(
            type(move) is str for move in prefix_obj):
        raise Hold("unit root prefix")

    def text_key(values: dict[str, object], key: str) -> str:
        item = values.get(key)
        if type(item) is not str:
            raise Hold(f"unit string field: {key}")
        return item

    def int_key(values: dict[str, object], key: str) -> int:
        item = values.get(key)
        if type(item) is not int:
            raise Hold(f"unit integer field: {key}")
        return item

    local_obj = value.get("local_code_sha256")
    if type(local_obj) is not dict:
        raise Hold("unit local code pins")
    local = cast(dict[str, str], local_obj)

    claim = UnitClaim(
        source_sha256=text_key(value, "source_sha256"),
        config_sha256=text_key(value, "config_sha256"),
        route_sha256=text_key(value, "route_sha256"),
        witness_sha256=text_key(value, "witness_sha256"),
        local_code_sha256=local,
        expected=GameIdentity(
            text_key(game, "manifest_sha256"), text_key(game, "namespace"),
            text_key(game, "root_id"), int_key(game, "game_id"),
            text_key(game, "archive_sha256"),
            text_key(game, "strict_receipt_sha256"), int_key(game, "rows"),
            tuple(cast(list[str], prefix_obj))),
        syzygy_inventory_sha256=text_key(value, "syzygy_inventory_sha256"),
        guard_sha256=text_key(value, "guard_sha256"),
        worker_sha256=text_key(value, "worker_sha256"),
        interpreter_sha256=text_key(value, "interpreter_sha256"),
        command_sha256=text_key(value, "command_sha256"),
        wall_seconds=int_key(value, "wall_seconds"),
        rss_bytes=int_key(value, "rss_bytes"),
        output_bytes=int_key(value, "output_bytes"),
        archive_context_bytes=int_key(value, "archive_context_bytes"))
    claim.validate()
    need(claim.as_json() == value, "unit claim fields/caps differ")
    return claim


def _tree_pids(root: int) -> set[int]:
    result: set[int] = set()
    todo = [root]
    while todo:
        pid = todo.pop()
        if pid in result:
            continue
        result.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            todo.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


_owner_deadline: float | None = None


def _owner_alarm(_signum: int, _frame: object) -> None:
    raise Hold("wall cap")


@contextmanager
def owned_deadline(seconds: int) -> Generator[None, None, None]:
    """Bound the complete owner call, including witness reads and sealing."""
    global _owner_deadline
    if _owner_deadline is not None:
        need(time.monotonic() <= _owner_deadline, "wall cap")
        yield
        return
    need(signal.getsignal(signal.SIGALRM) == signal.SIG_DFL and
         signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
         "exclusive owner alarm required")
    previous = signal.signal(signal.SIGALRM, _owner_alarm)
    _owner_deadline = time.monotonic() + seconds
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        need(time.monotonic() <= _owner_deadline, "wall cap")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _owner_deadline = None
        signal.signal(signal.SIGALRM, previous)


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK,
                                           {signal.SIGALRM})
    try:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_owned_unit(root: Path, claim: UnitClaim, *, argv: tuple[str, ...],
                    output_names: tuple[str, ...],
                    verify_output: Callable[[Path], None]) -> dict:
    """Own and seal a bounded child attempt; previous failed attempts stay visible.

    The child writes output files into the attempt directory and a canonical
    RESULT.json listing each output SHA and exact logical archive bytes.
    """
    claim.validate()
    need(root.is_absolute() and root.exists() and root.is_dir() and
         not root.is_symlink() and
         type(argv) is tuple and len(argv) >= 3 and
         all(type(arg) is str and arg for arg in argv) and
         0 < len(output_names) <= 4 and len(set(output_names)) == len(output_names) and
         all(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None
             for name in output_names) and
         callable(verify_output),
         "owned unit path/command/outputs")
    arguments = iter(argv)
    interpreter_arg = next(arguments)
    guard_arg = next(arguments)
    worker_arg = next(arguments)
    need(sha(canonical(argv)) == claim.command_sha256 and
         Path(interpreter_arg).is_absolute() and
         Path(guard_arg) == GUARD_PATH and Path(worker_arg).is_absolute(),
         "owned command identity")
    pinned_file(Path(interpreter_arg), claim.interpreter_sha256,
                MAX_ARCHIVE_BYTES)
    pinned_file(Path(guard_arg), claim.guard_sha256, 1 << 20)
    pinned_file(Path(worker_arg), claim.worker_sha256, 1 << 20)
    claim_raw = canonical(claim.as_json())
    claim_path = root / "CLAIM.json"
    if claim_path.exists():
        need(claim_path.read_bytes() == claim_raw, "unit claim/source/config changed")
    else:
        wave.atomic_write(claim_path, claim_raw)
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        complete_raw = complete_path.read_bytes()
        complete = json.loads(complete_raw)
        need(complete_raw == canonical(complete) and
             complete.get("schema") == "bt4_npz_owned_one_game_complete_v1" and
             set(complete) == {"schema", "claim_sha256", "attempt",
                               "outputs", "result_sha256",
                               "logical_archive_bytes", "logical_context_bytes",
                               "rows", "peak_rss_bytes", "wall_seconds"} and
             complete.get("claim_sha256") == sha(claim_raw),
             "complete receipt/claim changed")
        need(type(complete.get("attempt")) is str and
             re.fullmatch(r"attempt_[0-9]{4}", complete["attempt"]) is not None and
             type(complete.get("outputs")) is dict and
             hex64(complete.get("result_sha256")) and
             type(complete.get("peak_rss_bytes")) is int and
             0 <= complete["peak_rss_bytes"] <= claim.rss_bytes and
             type(complete.get("wall_seconds")) in (int, float) and
             0 <= complete["wall_seconds"] <= claim.wall_seconds,
             "complete attempt/result shape")
        attempt = root / complete["attempt"]
        need(sha(pinned_file(attempt / "RESULT.json",
                             complete["result_sha256"], 1 << 20)) ==
             complete["result_sha256"], "complete result changed")
        result = json.loads((attempt / "RESULT.json").read_bytes())
        need(type(result) is dict, "complete result/source differs")
        archive_bytes = result.get("logical_archive_bytes")
        context_bytes = result.get("logical_context_bytes")
        need(type(result) is dict and
             set(result) == {"schema", "claim_sha256", "rows",
                             "logical_archive_bytes", "logical_context_bytes",
                             "outputs"} and
             result.get("schema") == "bt4_npz_owned_one_game_result_v1" and
             result.get("claim_sha256") == sha(claim_raw) and
             result.get("outputs") == complete["outputs"] and
             result.get("rows") == claim.expected.rows and
             type(archive_bytes) is int and
             0 < archive_bytes <= MAX_ARCHIVE_BYTES and
             type(context_bytes) is int and
             context_bytes >= 0 and
             archive_bytes + context_bytes <= claim.archive_context_bytes and
             result.get("logical_archive_bytes") ==
                 complete.get("logical_archive_bytes") and
             result.get("logical_context_bytes") ==
                 complete.get("logical_context_bytes"),
             "complete result/source differs")
        for name, digest in complete["outputs"].items():
            need(name in output_names and
                 hex64(digest) and
                 sha(pinned_file(attempt / name, digest,
                                 claim.output_bytes)) == digest,
                 "complete output changed")
        need(set(complete["outputs"]) == set(output_names),
             "complete output set")
        verify_output(attempt)
        return complete
    index = 0
    while (root / f"attempt_{index:04d}").exists():
        index += 1
    attempt = root / f"attempt_{index:04d}"
    attempt.mkdir(mode=0o700)
    stdout_path = attempt / "stdout.log"
    stderr_path = attempt / "stderr.log"
    start = time.monotonic()
    peak_rss = 0
    owner_pid = os.getpid()
    env = {"PATH": "/usr/bin:/bin",
           "HOME": os.environ.get("HOME", "/tmp"),
           "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
           "BT4_EXPECTED_OWNER_PID": str(owner_pid),
           "BT4_CHILD_WALL_SECONDS": str(claim.wall_seconds),
           "BT4_INTERPRETER_PATH": interpreter_arg}
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            (*argv, str(attempt)), stdout=stdout, stderr=stderr,
            # The exec guard installs PDEATH and SIGALRM before archive code.
            start_new_session=True, env=env)
        failure = None
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, sum(_rss_bytes(pid) for pid in
                                             _tree_pids(process.pid)))
                if time.monotonic() - start > claim.wall_seconds:
                    failure = "wall cap"
                elif peak_rss > claim.rss_bytes:
                    failure = "RSS cap"
                elif sum(path.stat().st_size for path in attempt.iterdir()
                         if path.is_file()) > claim.output_bytes:
                    failure = "output cap"
                if failure is not None:
                    break
                time.sleep(0.05)
        finally:
            # Also reap if monitoring itself raises. Kill surviving descendants
            # after an otherwise successful direct-child exit.
            _kill_group(process)
        exit_code = process.wait()
        stdout.flush()
        os.fsync(stdout.fileno())
        stderr.flush()
        os.fsync(stderr.fileno())
    if failure is not None or exit_code != 0:
        raise Hold(f"owned child failed: {failure or exit_code}; attempt={attempt.name}")
    result_path = attempt / "RESULT.json"
    need(result_path.exists() and result_path.stat().st_size <= 1 << 20,
         "owned result missing/oversize")
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    need(result_raw == canonical(result) and
         result.get("schema") == "bt4_npz_owned_one_game_result_v1" and
         set(result) == {"schema", "claim_sha256", "rows",
                         "logical_archive_bytes", "logical_context_bytes",
                         "outputs"} and
         result.get("claim_sha256") == sha(claim_raw) and
         result.get("rows") == claim.expected.rows and
         type(result.get("logical_archive_bytes")) is int and
         0 < result["logical_archive_bytes"] <= MAX_ARCHIVE_BYTES and
         type(result.get("logical_context_bytes")) is int and
         result["logical_context_bytes"] >= 0 and
         result["logical_archive_bytes"] +
             result["logical_context_bytes"] <= claim.archive_context_bytes and
         type(result.get("outputs")) is dict and
         set(result["outputs"]) == set(output_names),
         "owned result/source/cap")
    for name, digest in result["outputs"].items():
        need(hex64(digest) and
             sha(pinned_file(attempt / name, digest,
                             claim.output_bytes)) == digest,
             "owned output SHA")
    need(sum(path.stat().st_size for path in attempt.iterdir()
             if path.is_file()) <= claim.output_bytes,
         "owned final output cap")
    verify_output(attempt)
    elapsed = time.monotonic() - start
    need(elapsed <= claim.wall_seconds and
         (_owner_deadline is None or time.monotonic() <= _owner_deadline),
         "wall cap")
    complete = {"schema": "bt4_npz_owned_one_game_complete_v1",
                "claim_sha256": sha(claim_raw),
                "attempt": attempt.name,
                "outputs": result["outputs"],
                "result_sha256": sha(result_raw),
                "logical_archive_bytes": result["logical_archive_bytes"],
                "logical_context_bytes": result["logical_context_bytes"],
                "rows": result["rows"],
                "peak_rss_bytes": peak_rss,
                "wall_seconds": round(elapsed, 6)}
    wave.atomic_write(complete_path, canonical(complete))
    return complete


def run_owned_unit(root: Path, claim: UnitClaim, *, argv: tuple[str, ...],
                   output_names: tuple[str, ...],
                   verify_output: Callable[[Path], None]) -> dict:
    with owned_deadline(claim.wall_seconds):
        return _run_owned_unit(root, claim, argv=argv,
                               output_names=output_names,
                               verify_output=verify_output)


def _run_bt4_packet_unit(root: Path, claim: UnitClaim, *, packet_path: Path,
                        witness_files: WitnessFiles,
                        argv: tuple[str, str, str, str],
                        route: Callable[[list], object]) -> dict:
    """Bind the private frozen packet and old witness before child launch."""
    claim.validate()
    need(packet_path.is_absolute() and argv[2] == str(CHILD_PATH) and
         argv[3] == str(packet_path) and
         witness_files.identity() == claim.witness_sha256,
         "owned packet/witness identity")
    packet_raw = pinned_file(packet_path, claim.config_sha256, 1 << 20)
    packet = json.loads(packet_raw)
    need(type(packet) is dict and packet_raw == canonical(packet) and
         packet.get("schema") == "bt4_npz_one_game_source_packet_v1" and
         packet.get("witness_schema") == WITNESS_SCHEMA and
         type(packet.get("imports")) is list and
         type(packet.get("verifier")) is dict and
         sha(canonical({"imports": packet["imports"],
                        "verifier": packet["verifier"]})) ==
             claim.source_sha256,
         "private frozen source/claim")
    route_path = Path(route.__code__.co_filename)
    pinned_file(route_path, claim.route_sha256, 1 << 20)
    witness = load_old_witness(witness_files, claim.expected)

    def verify(attempt: Path) -> None:
        verify_game_output(attempt, claim, witness, route)

    return run_owned_unit(root, claim, argv=argv,
                          output_names=OUTPUT_NAMES,
                          verify_output=verify)


def run_bt4_packet_unit(root: Path, claim: UnitClaim, *, packet_path: Path,
                        witness_files: WitnessFiles,
                        argv: tuple[str, str, str, str],
                        route: Callable[[list], object]) -> dict:
    with owned_deadline(claim.wall_seconds):
        return _run_bt4_packet_unit(root, claim, packet_path=packet_path,
                                    witness_files=witness_files, argv=argv,
                                    route=route)
