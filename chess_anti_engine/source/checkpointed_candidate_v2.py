"""Literal-UID, 100-byte candidate metadata for paired source segments.

Diagnostic only. The source reader and paired native-byte comparison must pass
before these records may enter a bounded external sort. No target is admitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterator

from chess_anti_engine.source import checkpointed_sort as sort
from chess_anti_engine.source import checkpointed_wave_v2 as wave

SCHEMA = "candidate_metadata_segment_v2"
ROUTE_DOMAIN = b"mixed500m/selected-teacher/v1\0"
ROUTE_SEED = "c8308d72aa70a137c34679f076959d9183e58dbf4cae3fef879b60fde7a2850b"


def route(uid: list[object]) -> str:
    """Frozen contract.route serialization with the unmodified five-field UID."""
    sort._uid(uid)
    payload = (json.dumps([ROUTE_SEED, *uid], separators=(",", ":"),
                          ensure_ascii=True) + "\n").encode("ascii")
    return "BT4" if hashlib.sha256(ROUTE_DOMAIN + payload).digest()[0] & 1 == 0 else "Ceres"


def _expected(store: wave.SegmentStore, number: int) -> dict:
    segment = store.verify_segment(number)
    comparison = store.verify_comparison(number)
    return {
        "schema": SCHEMA + "_claim",
        "segment": number,
        "wave_claim_sha256": wave.sha(store.claim_raw),
        "wave_receipt_sha256": wave.file_sha(store._path(number) / "RECEIPT.json"),
        "comparison_sha256": wave.file_sha(store.root / f"compare_{number:08d}.json"),
        "rows": segment["rows"],
        "paired_rows_sha256": comparison["paired_rows_sha256"],
        "code_sha256": wave.file_sha(Path(__file__)),
        "sort_code_sha256": wave.file_sha(Path(sort.__file__)),
        "record_format": "<32s32sQIIIIII4B",
        "record_bytes": sort.RECORD_BYTES,
    }


def _record_rows(store: wave.SegmentStore, number: int
                 ) -> tuple[bytes, bytes, int]:
    tables: dict[str, list[object]] = {"uid_prefixes": [], "contexts": [],
                                      "proofs": [], "provenances": []}
    ids: dict[str, dict[bytes, int]] = {name: {} for name in tables}
    records = bytearray()
    paired = hashlib.sha256()
    seen: set[tuple[str, str, str, int, int]] = set()
    count = 0
    for note, native in store.iter_rows(number):
        uid = sort._uid(note["uid"])
        sort._context(note["context"])
        wave.need(uid not in seen, "duplicate segment UID")
        seen.add(uid)
        wave.need(note["source"] == store.roster[(uid[0], uid[1])],
                  "roster source changed")
        values = {"uid_prefixes": uid[:3], "contexts": note["context"],
                  "proofs": note["game_proof_sha256"],
                  "provenances": note["provenance_locator_sha256"]}
        local: dict[str, int] = {}
        for name, value in values.items():
            key = wave.canonical(value)
            if key not in ids[name]:
                ids[name][key] = len(tables[name])
                tables[name].append(value)
            local[name] = ids[name][key]
        wave.need(len(wave.canonical(tables)) <= sort.MAX_TABLE_BYTES,
                  "candidate table byte cap")
        records.extend(sort.RECORD.pack(
            bytes.fromhex(note["input_digest"]),
            bytes.fromhex(note["input_bytes_sha256"]), note["ordinal"],
            local["uid_prefixes"], uid[3], uid[4], local["contexts"],
            local["proofs"], local["provenances"],
            sort.OUTCOMES.index(note["outcome"]),
            ("BT4-v9", "Ceres-v8", "SF-d6").index(note["source"]),
            sort.TEACHERS.index(route(list(uid))), 0))
        paired.update(wave.canonical(note))
        paired.update(native)
        count += 1
    wave.need(count == store.verify_segment(number)["rows"] and
              paired.hexdigest() == store.verify_comparison(number)["paired_rows_sha256"],
              "paired native proof before candidate admission")
    return bytes(records), wave.canonical(tables), count


def seal(store: wave.SegmentStore, number: int, path: Path) -> dict:
    """Seal metadata only after exact second-wave byte comparison."""
    wave.need(__import__("sys").flags.optimize == 0, "optimized Python forbidden")
    expected = _expected(store, number)
    records, tables, count = _record_rows(store, number)
    path = Path(path)
    if path.exists():
        return verify(store, number, path)
    stage = path.with_name("." + path.name + ".part")
    if stage.exists():
        import shutil
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    wave.atomic_write(stage / "CLAIM.json", wave.canonical(expected))
    wave.atomic_write(stage / "RECORDS.bin", records)
    wave.atomic_write(stage / "TABLES.json", tables)
    receipt = {"schema": SCHEMA + "_receipt",
               "claim_sha256": wave.sha(wave.canonical(expected)),
               "rows": count,
               "files": {name: {"sha256": wave.file_sha(stage / name),
                                "bytes": (stage / name).stat().st_size}
                         for name in ("RECORDS.bin", "TABLES.json")}}
    wave.atomic_write(stage / "RECEIPT.json", wave.canonical(receipt))
    wave.fsync_directory(stage)
    stage.rename(path)
    wave.fsync_directory(path.parent)
    return verify(store, number, path)


def verify(store: wave.SegmentStore, number: int, path: Path) -> dict:
    wave.need(__import__("sys").flags.optimize == 0, "optimized Python forbidden")
    path = Path(path)
    claim = (path / "CLAIM.json").read_bytes()
    expected = _expected(store, number)
    wave.need(claim == wave.canonical(expected), "candidate claim identity")
    raw = (path / "RECEIPT.json").read_bytes()
    receipt = json.loads(raw)
    wave.need(raw == wave.canonical(receipt) and
              receipt.get("schema") == SCHEMA + "_receipt" and
              receipt.get("claim_sha256") == wave.sha(claim) and
              receipt.get("rows") == expected["rows"] and
              set(receipt.get("files", {})) == {"RECORDS.bin", "TABLES.json"},
              "candidate receipt identity")
    wave.need({item.name for item in path.iterdir()} ==
              {"CLAIM.json", "RECEIPT.json", "RECORDS.bin", "TABLES.json"},
              "candidate file membership")
    for name, desc in receipt["files"].items():
        wave.need(wave.file_sha(path / name) == desc["sha256"] and
                  (path / name).stat().st_size == desc["bytes"],
                  "candidate file hash/size")
    wave.need(receipt["files"]["RECORDS.bin"]["bytes"] ==
              expected["rows"] * sort.RECORD_BYTES and
              receipt["files"]["TABLES.json"]["bytes"] <= sort.MAX_TABLE_BYTES,
              "candidate fixed-width/table cap")
    wave.need((path / "TABLES.json").stat().st_size <= sort.MAX_TABLE_BYTES,
              "candidate table byte cap")
    tables_raw = (path / "TABLES.json").read_bytes()
    tables = json.loads(tables_raw)
    wave.need(tables_raw == wave.canonical(tables) and
              set(tables) == {"uid_prefixes", "contexts", "proofs", "provenances"},
              "candidate table schema")
    with (path / "RECORDS.bin").open("rb") as stream:
        for index in range(expected["rows"]):
            values = sort.RECORD.unpack(stream.read(sort.RECORD_BYTES))
            digest, native_sha, ordinal, prefix, game, ply, context, proof, provenance, outcome, source, teacher, reserved = values
            wave.need(ordinal == number * sort.MAX_SEGMENT_ROWS + index and
                      prefix < len(tables["uid_prefixes"]) and
                      context < len(tables["contexts"]) and
                      proof < len(tables["proofs"]) and
                      provenance < len(tables["provenances"]) and
                      outcome < len(sort.OUTCOMES) and source < 3 and
                      teacher < len(sort.TEACHERS) and reserved == 0,
                      "candidate record bounds")
            uid = sort._uid([*tables["uid_prefixes"][prefix], game, ply])
            sort._context(tables["contexts"][context])
            wave.need(store.roster.get(uid[:2]) ==
                      ("BT4-v9", "Ceres-v8", "SF-d6")[source] and
                      sort.TEACHERS[teacher] == route(list(uid)) and
                      len(digest) == len(native_sha) == 32 and
                      sort._hex64(tables["proofs"][proof]) and
                      sort._hex64(tables["provenances"][provenance]),
                      "candidate UID/route/proof")
    # A locally recomputed receipt cannot change outcome or digest while the
    # paired native source remains sealed. This bounded diagnostic deliberately
    # re-decompresses one segment; it is not a 500M no-op cost claim.
    expected_records, expected_tables, count = _record_rows(store, number)
    wave.need(count == expected["rows"] and
              (path / "RECORDS.bin").read_bytes() == expected_records and
              tables_raw == expected_tables,
              "candidate differs from paired wave")
    return receipt


def iter_sort_entries(store: wave.SegmentStore, number: int, path: Path
                      ) -> Iterator[sort.SortEntry]:
    verify(store, number, path)
    with (Path(path) / "RECORDS.bin").open("rb") as stream:
        for index in range(store.verify_segment(number)["rows"]):
            values = sort.RECORD.unpack(stream.read(sort.RECORD_BYTES))
            yield sort.SortEntry(values[0].hex(), values[2], (number, index))
