"""Executable <=512-row diagnostic SF value sidecar for authenticated v6 winners.

This is a bounded small-bank adapter, not a corpus admission or training launch.
The operator must extract and pin the winner/proof/selected rows from separately
audited source and target receipts. Unreviewed full-bank labeling is out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.stockfish import uci as uci_module
from chess_anti_engine.stockfish import wdl as wdl_module
from scripts import sf_dlite_value_sidecar as core

MAX_INPUT = 16 << 20
MAX_OUTPUT = 64 << 20


def _runtime_source_hashes() -> dict[str, str]:
    return {
        "smallbank": core.file_digest(Path(__file__)),
        "dlite_core": core.file_digest(Path(core.__file__)),
        "stockfish_uci": core.file_digest(Path(uci_module.__file__)),
        "stockfish_wdl": core.file_digest(Path(wdl_module.__file__)),
        "sf_history_parser": core.file_digest(Path(core.corpus.__file__)),
    }


def _read_pinned(path: Path, expected_sha: str) -> bytes:
    core._sha_field(expected_sha, "input file")
    core.need(path.is_file() and not path.is_symlink() and
              0 < path.stat().st_size <= MAX_INPUT, "bounded regular input file")
    raw = path.read_bytes()
    core.need(len(raw) == path.stat().st_size and core.digest(raw) == expected_sha,
              "pinned small-bank input bytes")
    return raw


def _records(raw: bytes) -> list[tuple[bytes, dict[str, Any]]]:
    lines = raw.splitlines(keepends=True)
    core.need(0 < len(lines) <= 1024 and sum(len(line) for line in lines) == len(raw),
              "bounded complete JSONL")
    items = []
    for line in lines:
        core.need(line.endswith(b"\n") and 0 < len(line) <= 2**20,
                  "canonical bounded JSONL line")
        obj = json.loads(line)
        core.need(type(obj) is dict, "JSONL object")
        items.append((line, obj))
    return items


def load_small_bank(winners: bytes, proofs: bytes) -> list[core.PreparedRow]:
    winner_lines = _records(winners)
    proof_lines = _records(proofs)
    core.need(len(winner_lines) <= 512, "small-bank winner cap")
    proof_by_uid: dict[tuple[str, str, str, int, int], bytes] = {}
    for line, game in proof_lines:
        chain = game.get("history_chain")
        if type(chain) is not dict:
            raise core.Hold("v6 history proof schema")
        row_index = chain.get("row_index")
        if type(row_index) is not list:
            raise core.Hold("v6 history proof schema")
        for entry in row_index:
            uid = core.uid_of(entry.get("uid"))
            core.need(uid not in proof_by_uid, "duplicate proof UID")
            proof_by_uid[uid] = line
    prepared = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for _, winner in winner_lines:
        uid = core.uid_of(winner.get("uid"))
        core.need(uid not in seen and uid in proof_by_uid,
                  "unique winner/proof UID")
        seen.add(uid)
        prepared.append(core.prepare_v6_winner(winner, proof_by_uid[uid]))
    core.need(len({row.input_digest for row in prepared}) == len(prepared),
              "admitted-unique small-bank inputs")
    return prepared


def _new_output(out: Path, claim: dict[str, Any]) -> None:
    core.need(not out.exists() and not out.is_symlink(), "fresh one-shot output")
    out.mkdir(mode=0o700)
    try:
        _write(out / "CLAIM.json", claim)
    except BaseException as error:
        _write(out / "FAILED.json", {"status": "FAILED_NO_CREDIT",
                                     "error": repr(error)})
        raise


def _write(path: Path, value: Any) -> str:
    raw = core.canonical(value) + b"\n"
    core.need(len(raw) <= MAX_OUTPUT, "bounded output")
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    core.need(path.read_bytes() == raw, "output byte readback")
    return core.digest(raw)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    raw = b"".join(core.canonical(value) + b"\n" for value in values)
    core.need(len(raw) <= MAX_OUTPUT, "bounded output rows")
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    core.need(path.read_bytes() == raw, "output rows readback")
    return core.digest(raw)


def _output_bytes(out: Path) -> int:
    return sum(item.stat().st_size for item in out.iterdir() if item.is_file())


def run_label(*, winners: Path, winners_sha256: str,
              proofs: Path, proofs_sha256: str, stockfish: Path,
              stockfish_sha256: str, syzygy_path: str, depth: int,
              out: Path, engine_factory: Any = None,
              searcher_factory: Any = None) -> dict[str, Any]:
    rows = load_small_bank(_read_pinned(winners, winners_sha256),
                           _read_pinned(proofs, proofs_sha256))
    sources = _runtime_source_hashes()
    profile = {"depth": depth, "hash_mb": 8, "threads": 1,
               "syzygy_path": syzygy_path,
               "syzygy_50_move_rule": True, "syzygy_probe_limit": 6,
               "retain_syzygy_on_new_game": True,
               "uci_reset": "ucinewgame_then_serialized_readyok_per_row"}
    _new_output(out, {"schema": "sf_dlite_smallbank_label_v1", "depth": depth,
                      "winners_sha256": winners_sha256,
                      "proofs_sha256": proofs_sha256,
                      "stockfish_sha256": stockfish_sha256,
                      "requested_uci_profile": profile,
                      "runtime_source_sha256": sources,
                      "status": "CLAIMED_ZERO_CREDIT"})
    try:
        kw = {}
        if engine_factory is not None:
            kw["engine_factory"] = engine_factory
        if searcher_factory is not None:
            kw["searcher_factory"] = searcher_factory
        labels = core.label_bank(rows, depth=depth, stockfish=stockfish,
                                 stockfish_sha256=stockfish_sha256,
                                 syzygy_path=syzygy_path, **kw)
        raw_sha = _write_jsonl(out / "LABELS.jsonl", labels)
        core.need(_output_bytes(out) < MAX_OUTPUT - (1 << 20),
                  "total small-bank output reserve")
        core.need(_runtime_source_hashes() == sources,
                  "runtime source changed during small-bank labeling")
        receipt = {"schema": "sf_dlite_smallbank_label_v1",
                   "status": "COMPLETE_DIAGNOSTIC_ZERO_CORPUS_CREDIT",
                   "rows": len(rows), "depth": depth, "hash_mb": 8,
                   "winners_sha256": winners_sha256,
                   "proofs_sha256": proofs_sha256,
                   "stockfish_sha256": stockfish_sha256,
                   "requested_uci_profile": profile,
                   "runtime_source_sha256": sources,
                   "labels_sha256": raw_sha,
                   "scope": "side-to-move D-calibrated scalar facts only; no policy or active target"}
        _write(out / "COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        _write(out / "FAILED.json", {"status": "FAILED_NO_LABEL_CREDIT",
                                     "error": repr(error)})
        raise


def run_attach(*, winners: Path, winners_sha256: str,
               proofs: Path, proofs_sha256: str,
               labels: Path, labels_sha256: str,
               selected: Path, selected_sha256: str, out: Path) -> dict[str, Any]:
    rows = load_small_bank(_read_pinned(winners, winners_sha256),
                           _read_pinned(proofs, proofs_sha256))
    label_records = [x for _, x in _records(_read_pinned(labels, labels_sha256))]
    selected_records = [x for _, x in _records(_read_pinned(selected, selected_sha256))]
    core.need(len(rows) == len(label_records) == len(selected_records),
              "exact selected/label row count")
    sources = _runtime_source_hashes()
    _new_output(out, {"schema": "sf_dlite_smallbank_attach_v1",
                      "winners_sha256": winners_sha256,
                      "proofs_sha256": proofs_sha256,
                      "labels_sha256": labels_sha256,
                      "selected_sha256": selected_sha256,
                      "selected_route_seed": core.ROUTE_SEED,
                      "selected_route_domain_hex": core.ROUTE_DOMAIN.hex(),
                      "runtime_source_sha256": sources,
                      "status": "CLAIMED_ZERO_CREDIT"})
    try:
        candidate_rows = []
        for row, label, target_row in zip(rows, label_records, selected_records):
            core.need(target_row.get("teacher") in ("BT4", "Ceres"),
                      "chosen neural teacher identity")
            target = bytes.fromhex(target_row["target_hex"])
            mask = np.frombuffer(bytes.fromhex(target_row["legal_mask_hex"]),
                                 dtype=np.uint8).copy()
            candidate, proof = core.attach_main_wdl(row, label, target_row,
                                                      target, mask)
            candidate_rows.append({**proof, "target_hex": candidate.hex(),
                                   "selected_teacher": target_row["teacher"]})
        raw_sha = _write_jsonl(out / "CANDIDATE_TARGETS.jsonl", candidate_rows)
        core.need(_output_bytes(out) < MAX_OUTPUT - (1 << 20),
                  "total small-bank output reserve")
        core.need(_runtime_source_hashes() == sources,
                  "runtime source changed during small-bank attachment")
        receipt = {"schema": "sf_dlite_smallbank_attach_v1",
                   "status": "COMPLETE_DIAGNOSTIC_ZERO_CORPUS_CREDIT",
                   "rows": len(rows), "candidate_targets_sha256": raw_sha,
                   "labels_sha256": labels_sha256,
                   "selected_sha256": selected_sha256,
                   "selected_route_seed": core.ROUTE_SEED,
                   "selected_route_domain_hex": core.ROUTE_DOMAIN.hex(),
                   "runtime_source_sha256": sources,
                   "scope": "only main search_wdl changed; no SF policy or auxiliary"}
        _write(out / "COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        _write(out / "FAILED.json", {"status": "FAILED_NO_TARGET_CREDIT",
                                     "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "label", "attach"))
    parser.add_argument("--winners", type=Path)
    parser.add_argument("--winners-sha256")
    parser.add_argument("--proofs", type=Path)
    parser.add_argument("--proofs-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--stockfish", type=Path)
    parser.add_argument("--stockfish-sha256")
    parser.add_argument("--syzygy-path")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--labels-sha256")
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--selected-sha256")
    args = parser.parse_args()
    if args.mode == "inspect":
        print(json.dumps({"status": "SOURCE_ONLY_NO_REGISTERED_READ",
                          "max_rows": 512}))
        return
    common = {"winners": args.winners, "winners_sha256": args.winners_sha256,
              "proofs": args.proofs, "proofs_sha256": args.proofs_sha256,
              "out": args.out}
    core.need(all(value is not None for value in common.values()),
              "required small-bank paths and pins")
    if args.mode == "label":
        core.need(all(value is not None for value in
                      (args.stockfish, args.stockfish_sha256,
                       args.syzygy_path, args.depth)), "scalar profile args")
        print(json.dumps(run_label(**common, depth=args.depth,
                                   stockfish=args.stockfish,
                                   stockfish_sha256=args.stockfish_sha256,
                                   syzygy_path=args.syzygy_path), sort_keys=True))
    else:
        core.need(all(value is not None for value in
                      (args.labels, args.labels_sha256,
                       args.selected, args.selected_sha256)),
                  "selected target and label pins")
        print(json.dumps(run_attach(**common, labels=args.labels,
                                    labels_sha256=args.labels_sha256,
                                    selected=args.selected,
                                    selected_sha256=args.selected_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
