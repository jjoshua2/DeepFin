"""Freeze 128 scalar-label rows from an independently audited tri-source replay.

This reads only the replay's terminal/receipt/index/game-proof metadata. It does
not read the canonical tensor spool, source archives, or selected targets. A
separate independent replay audit must authenticate the supplied receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from scripts import sf_dlite_value_sidecar as core

SAMPLE_DOMAIN = b"sf_dlite_unique_row_sample_v1\0"
SOURCE_PROOFS = ("BT4-v9", "Ceres-v8", "SF-d6")
MAX_RECEIPT = 128 << 20
MAX_INDEX = 64 << 20
MAX_PROOF = 64 << 20
MAX_EXTRACT = 16 << 20
ZERO = dict.fromkeys(("source", "unique", "target", "pack", "ingest",
                      "owner", "outcome"), 0)


def need(ok: bool, why: str) -> None:
    core.need(ok, why)


def object_value(value: Any, why: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise core.Hold(why)
    return value


def list_value(value: Any, why: str) -> list[Any]:
    if type(value) is not list:
        raise core.Hold(why)
    return value


def read_pinned(path: Path, sha256: str, maximum: int) -> bytes:
    core._sha_field(sha256, "extract input")
    need(path.is_file() and not path.is_symlink() and
         0 < path.stat().st_size <= maximum, "bounded regular extract input")
    raw = path.read_bytes()
    need(len(raw) == path.stat().st_size and core.digest(raw) == sha256,
         "pinned extract input bytes")
    return raw


def read_bounded(path: Path, maximum: int) -> bytes:
    need(path.is_file() and not path.is_symlink() and
         0 < path.stat().st_size <= maximum, "bounded regular proof file")
    raw = path.read_bytes()
    need(len(raw) == path.stat().st_size, "stable proof file length")
    return raw


def json_object(raw: bytes, why: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise core.Hold(why) from error
    return object_value(value, why)


def lines(raw: bytes, maximum: int) -> list[bytes]:
    rows = raw.splitlines(keepends=True)
    need(len(rows) <= maximum and b"".join(rows) == raw and
         all(line.endswith(b"\n") and 0 < len(line) <= 1 << 20
             for line in rows), "bounded complete extract JSONL")
    return rows


def sample_rank(uid: list[Any]) -> tuple[bytes, bytes]:
    core.uid_of(uid)
    encoded = core.canonical(uid) + b"\n"
    return hashlib.sha256(SAMPLE_DOMAIN + encoded).digest(), encoded


def extract_metadata(*, receipt_raw: bytes, terminal_raw: bytes,
                     index_raw: bytes, proof_raw: dict[str, bytes],
                     sample_size: int = 128,
                     expected_gross: int = 83991) -> tuple[bytes, bytes, dict[str, Any]]:
    """Join complete replay metadata and return selected index/proof lines."""
    need(type(sample_size) is int and 0 < sample_size <= 512,
         "bounded sample size")
    receipt = json_object(receipt_raw, "replay receipt JSON")
    terminal = json_object(terminal_raw, "replay terminal JSON")
    need(re.fullmatch(r"tri_source_83991_cpu_diagnostic_v[0-9]+",
                      str(receipt.get("schema"))) is not None and
         receipt.get("status") == "COMPLETE_DIAGNOSTIC_ZERO_CREDIT" and
         terminal.get("status") == receipt["status"] and
         terminal.get("receipt_sha256") == core.digest(receipt_raw) and
         receipt.get("credit") == terminal.get("credit") == ZERO,
         "terminal complete zero-credit receipt binding")
    resolution = object_value(receipt.get("resolution"), "resolution")
    same = object_value(resolution.get("two_pass_readback"),
                        "two-pass readback")
    dedup = object_value(resolution.get("provisional_dedup"),
                         "provisional dedup")
    winners = list_value(dedup.get("winners"), "dedup winners")
    need(same.get("rows") == expected_gross and
         same.get("index_sha256") == core.digest(index_raw) and
         dedup.get("gross_rows") == expected_gross and
         dedup.get("unique_rows") == len(winners) and
         len(winners) >= sample_size,
         "complete wave index/dedup receipt")
    index = lines(index_raw, expected_gross)
    need(len(index) == expected_gross, "gross wave index rows")
    ranked: list[tuple[tuple[bytes, bytes], dict[str, Any]]] = []
    seen_inputs: set[str] = set()
    seen_uids: set[tuple[str, str, str, int, int]] = set()
    for winner in winners:
        winner = object_value(winner, "winner object")
        uid = core.uid_of(winner.get("winner_uid"))
        digest = core._sha_field(winner.get("input_digest"), "winner input")
        ordinal = winner.get("winner_ordinal")
        need(type(ordinal) is int and 0 <= ordinal < expected_gross and
             uid not in seen_uids and digest not in seen_inputs,
             "unique winner UID/input")
        seen_uids.add(uid)
        seen_inputs.add(digest)
        ranked.append((sample_rank(list(uid)), winner))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:sample_size]
    selected_index_lines: list[bytes] = []
    selected_uid: set[tuple[str, str, str, int, int]] = set()
    for _, winner in selected:
        ordinal = winner["winner_ordinal"]
        note = json_object(index[ordinal], "selected index JSON")
        uid = core.uid_of(note.get("uid"))
        need(note.get("ordinal") == ordinal and
             uid == core.uid_of(winner["winner_uid"]) and
             note.get("input_digest") == winner["input_digest"] and
             note.get("source") in SOURCE_PROOFS,
             "selected winner/index identity")
        selected_uid.add(uid)
        selected_index_lines.append(index[ordinal])
    child_results = list_value(receipt.get("source_child_results"),
                               "child proof receipts")
    proof_pins: dict[str, str] = {}
    for item in child_results:
        item = object_value(item, "child result")
        if item.get("wave") == 1:
            source = item.get("source")
            if (type(source) is not str or source not in SOURCE_PROOFS or
                    source in proof_pins):
                raise core.Hold("unique wave1 child proof")
            proof_pins[source] = core._sha_field(item.get("proof_sha256"),
                                                 "child proof")
    need(set(proof_pins) == set(SOURCE_PROOFS) and
         set(proof_raw) == set(SOURCE_PROOFS), "three wave1 proof files")
    selected_proofs: list[bytes] = []
    selected_game_lines: dict[tuple[str, str, str, int, int], bytes] = {}
    found: set[tuple[str, str, str, int, int]] = set()
    for source in SOURCE_PROOFS:
        raw = proof_raw[source]
        need(core.digest(raw) == proof_pins[source], "wave1 proof SHA")
        for line in lines(raw, 256):
            game = json_object(line, "game proof JSON")
            chain = object_value(game.get("history_chain"),
                                 "history-chain game proof")
            row_index = list_value(chain.get("row_index"),
                                   "history-chain game proof")
            matches = []
            for entry in row_index:
                entry = object_value(entry, "row proof entry")
                uid = core.uid_of(entry.get("uid"))
                if uid in selected_uid:
                    need(uid not in found, "duplicate selected proof UID")
                    found.add(uid)
                    selected_game_lines[uid] = line
                    matches.append(uid)
            if matches:
                selected_proofs.append(line)
    need(found == selected_uid, "all selected complete history proofs")
    for index_line in selected_index_lines:
        note = json_object(index_line, "selected winner proof join")
        core.prepare_v6_winner(note, selected_game_lines[core.uid_of(note["uid"])])
    winners_out = b"".join(selected_index_lines)
    proofs_out = b"".join(selected_proofs)
    need(0 < len(winners_out) <= MAX_EXTRACT and
         0 < len(proofs_out) <= MAX_EXTRACT,
         "small-bank consumer input size")
    return winners_out, proofs_out, {
        "schema": "sf_dlite_128_metadata_extract_v1",
        "status": "METADATA_ONLY_ZERO_LABEL_AND_CORPUS_CREDIT",
        "sample_domain": SAMPLE_DOMAIN.decode("ascii"),
        "sample_size": sample_size,
        "gross_rows": expected_gross,
        "unique_rows": len(winners),
        "terminal_sha256": core.digest(terminal_raw),
        "receipt_sha256": core.digest(receipt_raw),
        "wave1_index_sha256": core.digest(index_raw),
        "wave1_proof_sha256": proof_pins,
        "winners_sha256": core.digest(winners_out),
        "proofs_sha256": core.digest(proofs_out),
        "proof_games": len(selected_proofs),
    }


def write_once(path: Path, raw: bytes) -> None:
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    need(path.read_bytes() == raw, "extract output readback")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--bt4-proof", type=Path, required=True)
    parser.add_argument("--ceres-proof", type=Path, required=True)
    parser.add_argument("--sf-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = read_pinned(args.receipt, args.receipt_sha256, MAX_RECEIPT)
    terminal = read_pinned(args.terminal, args.terminal_sha256, 1 << 20)
    index = read_pinned(args.index, args.index_sha256, MAX_INDEX)
    paths = dict(zip(SOURCE_PROOFS,
                     (args.bt4_proof, args.ceres_proof, args.sf_proof),
                     strict=True))
    proofs = {source: read_bounded(path, MAX_PROOF)
              for source, path in paths.items()}
    winners_out, proofs_out, manifest = extract_metadata(
        receipt_raw=receipt, terminal_raw=terminal, index_raw=index,
        proof_raw=proofs)
    need(not args.out.exists() and not args.out.is_symlink(),
         "fresh extraction output")
    args.out.mkdir(mode=0o700)
    try:
        write_once(args.out / "WINNERS.jsonl", winners_out)
        write_once(args.out / "PROOFS.jsonl", proofs_out)
        write_once(args.out / "MANIFEST.json", core.canonical(manifest) + b"\n")
    except BaseException as error:
        write_once(args.out / "FAILED.json",
                   core.canonical({"status": "FAILED_NO_CREDIT",
                                   "error": repr(error)}) + b"\n")
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
