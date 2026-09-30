"""Bounded whole-game reader cursor into literal-UID paired wave segments.

The callback signature matches the qualified replay_one/snapshot boundary, but
this diagnostic does not open a corpus by itself. Caller pins actual reader,
strict gate, Syzygy inventory and archive byte limits before any real use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
from collections.abc import Callable, Iterator

from chess_anti_engine.source import checkpointed_candidate_v2 as candidate
from chess_anti_engine.source import checkpointed_sort as sort
from chess_anti_engine.source import checkpointed_wave_v2 as wave

MAX_GAME_ROWS = 400
MAX_GAMES = 4096  # bounded diagnostic roster; pagination is separate work


@dataclass(frozen=True)
class GameLocator:
    source: str
    source_manifest_sha: str
    namespace: str
    root_id: str
    game_id: int
    rows: int
    archive_sha: str
    strict_receipt_sha: str
    actual: object = None


class GameReader(Protocol):
    def read_game(self, locator: GameLocator) -> tuple[tuple[dict, bytes], ...]: ...


class ReplayGameReader:
    """Adapter for actual snapshot(path, SHA) + replay_one(..., proof_sink=).

    ``replay`` is caller's bound replay_one closure and must invoke the
    frozen verifier/history/terminal gate. Proof comes only from its callback.
    """

    def __init__(self, snapshot: Callable[[GameLocator], bytes],
                 replay: Callable[[GameLocator, bytes, Callable[[dict], None]],
                                  tuple[tuple[dict, bytes], ...]],
                 *, strict_receipt_sha256: str,
                 syzygy_inventory_sha256: str,
                 archive_byte_cap: int = 64 << 20) -> None:
        wave.need(wave._hex64(strict_receipt_sha256) and
                  wave._hex64(syzygy_inventory_sha256) and
                  type(archive_byte_cap) is int and
                  0 < archive_byte_cap <= 512 << 20,
                  "strict/Syzygy/archive byte cap")
        self.snapshot = snapshot
        self.replay = replay
        self.strict = strict_receipt_sha256
        self.syzygy = syzygy_inventory_sha256
        self.archive_byte_cap = archive_byte_cap

    def read_game(self, locator: GameLocator) -> tuple[tuple[dict, bytes], ...]:
        wave.need(locator.strict_receipt_sha == self.strict,
                  "strict receipt mismatch")
        raw = self.snapshot(locator)
        wave.need(type(raw) is bytes and
                  0 < len(raw) <= self.archive_byte_cap and
                  wave.sha(raw) == locator.archive_sha,
                  "archive snapshot byte identity")
        proofs: list[dict] = []
        rows = self.replay(locator, raw, proofs.append)
        wave.need(type(rows) is tuple and len(rows) == locator.rows and
                  len(proofs) == 1, "whole-game replay/proof count")
        proof = proofs[0]
        source_proof = proof.get("source_proof") if type(proof) is dict else None
        if type(source_proof) is not dict:
            raise wave.Hold("complete checked replay proof/Syzygy identity")
        source_proof_raw = (json.dumps(source_proof, sort_keys=True,
                                       separators=(",", ":"), ensure_ascii=True,
                                       allow_nan=False).encode("ascii") + b"\n")
        wave.need(type(proof) is dict and
                  proof.get("source") == locator.source and
                  proof.get("root_id") == locator.root_id and
                  proof.get("game_id") == locator.game_id and
                  proof.get("archive_sha256") == locator.archive_sha and
                  wave._hex64(proof.get("source_proof_sha256")) and
                  wave.sha(source_proof_raw) == proof["source_proof_sha256"] and
                  source_proof.get("schema") ==
                      "full512_raw_strict_source_proof_v1" and
                  source_proof.get("status") == "PASS_SOURCE_PROOF" and
                  source_proof.get("source") == locator.source and
                  source_proof.get("root_id") == locator.root_id and
                  source_proof.get("game_id") == locator.game_id and
                  source_proof.get("strict_receipt_sha256") == self.strict and
                  source_proof.get("archive_sha256") == locator.archive_sha and
                  source_proof.get("gross_rows") == locator.rows and
                  proof.get("terminal_fact", {}).get("syzygy_inventory_sha256") ==
                      self.syzygy and
                  proof.get("terminal_fact", {}).get("terminal_replayed") is True and
                  proof.get("terminal_fact", {}).get("rows") == locator.rows,
                  "complete checked replay proof/Syzygy identity")
        answer = []
        for row, native in rows:
            wave.need(type(row) is dict and type(native) is bytes and
                      row.get("uid") == [locator.source_manifest_sha,
                                         locator.namespace, locator.root_id,
                                         locator.game_id, len(answer)],
                      "literal replay UID/ply sequence")
            answer.append(({**row,
                            "game_proof_sha256": proof["source_proof_sha256"],
                            "syzygy_inventory_sha256": self.syzygy}, native))
        return tuple(answer)


@dataclass(frozen=True)
class Cursor:
    game: int
    row: int
    game_sha256: str | None


def _take(rows: Iterator[tuple[dict, bytes]], count: int
          ) -> Iterator[tuple[dict, bytes]]:
    for _ in range(count):
        try:
            yield next(rows)
        except StopIteration as error:
            raise wave.Hold("source cursor truncated") from error


class SourceCursor:
    def __init__(self, locators: tuple[GameLocator, ...], reader: GameReader,
                 *, strict_receipt_sha256: str,
                 syzygy_inventory_sha256: str,
                 reader_code_sha256: str) -> None:
        wave.need(type(locators) is tuple and 0 < len(locators) <= MAX_GAMES and
                  wave._hex64(strict_receipt_sha256) and
                  wave._hex64(syzygy_inventory_sha256) and
                  wave._hex64(reader_code_sha256), "bounded source roster")
        expected = []
        key_labels: dict[tuple[str, str], str] = {}
        unique_games: set[tuple[str, str, str, int]] = set()
        for loc in locators:
            wave.need(type(loc) is GameLocator and
                      loc.source in ("BT4-v9", "Ceres-v8", "SF-d6") and
                      wave._hex64(loc.source_manifest_sha) and
                      wave._hex64(loc.archive_sha) and
                      loc.strict_receipt_sha == strict_receipt_sha256 and
                      type(loc.namespace) is str and bool(loc.namespace) and
                      type(loc.root_id) is str and bool(loc.root_id) and
                      type(loc.game_id) is int and 0 <= loc.game_id <= sort.U32 and
                      type(loc.rows) is int and 0 < loc.rows <= MAX_GAME_ROWS,
                      "bounded exact game locator")
            key = loc.source_manifest_sha, loc.namespace
            wave.need(key not in key_labels or key_labels[key] == loc.source,
                      "conflicting source namespace")
            key_labels[key] = loc.source
            game = (loc.source_manifest_sha, loc.namespace,
                    loc.root_id, loc.game_id)
            wave.need(game not in unique_games,
                      "duplicate literal source-qualified game")
            unique_games.add(game)
            expected.append({"source": loc.source,
                             "source_manifest_sha": loc.source_manifest_sha,
                             "namespace": loc.namespace,
                             "root_id": loc.root_id, "game_id": loc.game_id,
                             "rows": loc.rows, "archive_sha": loc.archive_sha,
                             "strict_receipt_sha": loc.strict_receipt_sha})
        self.locators = locators
        self.reader = reader
        self.strict_receipt_sha256 = strict_receipt_sha256
        self.syzygy_inventory_sha256 = syzygy_inventory_sha256
        self.roster = tuple((a, b, label) for (a, b), label in sorted(key_labels.items()))
        self.identity = wave.sha(wave.canonical({
            "schema": "source_cursor_roster_v2", "locators": expected,
            "strict_receipt_sha256": strict_receipt_sha256,
            "syzygy_inventory_sha256": syzygy_inventory_sha256,
            "reader_code_sha256": reader_code_sha256}))
        self.total_rows = sum(loc.rows for loc in locators)
        self.current = Cursor(0, 0, None)

    def _read(self, index: int) -> tuple[tuple[dict, bytes], ...]:
        loc = self.locators[index]
        rows = self.reader.read_game(loc)
        wave.need(type(rows) is tuple and len(rows) == loc.rows,
                  "complete bounded game")
        for ply, (row, native) in enumerate(rows):
            wave.need(type(row) is dict and
                      row.get("uid") == [loc.source_manifest_sha, loc.namespace,
                                         loc.root_id, loc.game_id, ply] and
                      row.get("source") == loc.source and
                      row.get("syzygy_inventory_sha256") ==
                          self.syzygy_inventory_sha256 and
                      type(native) is bytes,
                      "literal UID/source/whole-game row")
        return rows

    def rows_from(self, cursor: Cursor) -> Iterator[tuple[dict, bytes]]:
        wave.need(type(cursor) is Cursor and
                  0 <= cursor.game <= len(self.locators) and
                  (cursor.game < len(self.locators) or cursor.row == 0),
                  "cursor game bound")
        for index in range(cursor.game, len(self.locators)):
            rows = self._read(index)
            digest = hashlib.sha256()
            for row, native in rows:
                digest.update(wave.canonical(row))
                digest.update(native)
            game_sha = digest.hexdigest()
            start = cursor.row if index == cursor.game else 0
            wave.need(0 <= start < len(rows) and
                      (index != cursor.game or
                       ((start == 0 and cursor.game_sha256 is None) or
                        (start > 0 and cursor.game_sha256 == game_sha))),
                      "midgame resume complete-game hash")
            for ply in range(start, len(rows)):
                self.current = (Cursor(index, ply + 1, game_sha)
                                if ply + 1 < len(rows) else
                                Cursor(index + 1, 0, None))
                yield rows[ply]


class CursorPipeline:
    """Sealed segment cursor chain, paired replay, v2 records and sort inputs."""

    def __init__(self, root: Path, source: SourceCursor,
                 store: wave.SegmentStore) -> None:
        wave.need(store.claim["input_sha256"] == source.identity and
                  store.claim["roster"] == [list(item) for item in source.roster] and
                  store.claim["strict_receipt_sha256"] ==
                      source.strict_receipt_sha256 and
                  store.claim["syzygy_inventory_sha256"] ==
                      source.syzygy_inventory_sha256,
                  "cursor/wave source pins")
        self.root = Path(root)
        self.source = source
        self.store = store
        self.root.mkdir(parents=True, exist_ok=True)
        self.claim = {"schema": "source_cursor_pipeline_claim_v2",
                      "source_roster_sha256": source.identity,
                      "wave_claim_sha256": wave.sha(store.claim_raw),
                      "code_sha256": wave.file_sha(Path(__file__)),
                      "candidate_code_sha256": wave.file_sha(Path(candidate.__file__)),
                      "sort_code_sha256": wave.file_sha(Path(sort.__file__)),
                      "total_rows": source.total_rows}
        claim = self.root / "CLAIM.json"
        if claim.exists():
            wave.need(claim.read_bytes() == wave.canonical(self.claim),
                      "cursor source/config/code changed")
        else:
            wave.atomic_write(claim, wave.canonical(self.claim))

    def _cursor_path(self, segment: int) -> Path:
        return self.root / f"cursor_{segment:08d}.json"

    @staticmethod
    def _seal_sort_group(sorter: sort.CheckpointedSort,
                         entries: list[sort.SortEntry],
                         receipt_sha256: str, index: int) -> Path:
        identity = wave.sha(wave.canonical([receipt_sha256, index]))
        run = sorter.seal_source_run(f"candidate_{index:08d}", entries,
                                     source_identity_sha256=identity)
        expected = sorted(entries, key=lambda item: (item.key, item.input_index))
        wave.need(list(sorter.iter_run(run)) == expected,
                  "sort run differs from paired candidate metadata")
        return run

    def _position(self, cursor: Cursor) -> int:
        wave.need(0 <= cursor.game <= len(self.source.locators) and
                  (cursor.game < len(self.source.locators) or cursor.row == 0),
                  "cursor game position")
        prior = sum(loc.rows for loc in self.source.locators[:cursor.game])
        wave.need(cursor.game == len(self.source.locators) or
                  0 <= cursor.row < self.source.locators[cursor.game].rows,
                  "cursor row position")
        return prior + cursor.row

    def _read_cursor(self, segment: int, previous: str) -> tuple[dict, Cursor]:
        path = self._cursor_path(segment)
        raw = path.read_bytes()
        receipt = json.loads(raw)
        wave.need(raw == wave.canonical(receipt) and
                  receipt.get("schema") == "source_cursor_segment_v2" and
                  receipt.get("claim_sha256") == wave.sha(wave.canonical(self.claim)) and
                  receipt.get("segment") == segment and
                  receipt.get("previous_cursor_receipt_sha256") == previous and
                  receipt.get("wave_receipt_sha256") == wave.file_sha(
                      self.store._path(segment) / "RECEIPT.json"),
                  "cursor receipt chain/segment")
        self.store.verify_segment(segment)
        end = receipt["end"]
        cursor = Cursor(end[0], end[1], end[2])
        start = receipt["start"]
        start_cursor = Cursor(start[0], start[1], start[2])
        wave.need(type(cursor.game) is int and type(cursor.row) is int and
                  0 <= cursor.game <= len(self.source.locators) and
                  (cursor.game < len(self.source.locators) or cursor.row == 0) and
                  ((cursor.row == 0 and cursor.game_sha256 is None) or
                   (cursor.row > 0 and wave._hex64(cursor.game_sha256))),
                  "cursor end bound")
        wave.need(self._position(start_cursor) ==
                  segment * wave.SEGMENT_ROWS and
                  self._position(cursor) ==
                  segment * wave.SEGMENT_ROWS +
                  self.store.verify_segment(segment)["rows"] and
                  ((start_cursor.row == 0 and start_cursor.game_sha256 is None) or
                   (start_cursor.row > 0 and
                   wave._hex64(start_cursor.game_sha256))),
                  "cursor sealed ordinal coverage")
        return receipt, cursor

    def _compare_source_to_sealed(
            self, number: int, rows: Iterator[tuple[dict, bytes]],
            expected: int) -> None:
        comparison = self.store.verify_comparison(number)
        digest = hashlib.sha256()
        count = 0
        for (sealed, old_native), (row, native) in zip(
                self.store.iter_rows(number), _take(rows, expected),
                strict=True):
            note = wave._note(row, native,
                              number * wave.SEGMENT_ROWS + count,
                              self.store.roster, self.store.syzygy_sha256)
            wave.need(note == sealed and native == old_native,
                      "source differs from sealed paired wave")
            digest.update(wave.canonical(note))
            digest.update(native)
            count += 1
        wave.need(count == expected and
                  digest.hexdigest() == comparison["paired_rows_sha256"],
                  "sealed paired-wave source coverage")

    def wave1(self, *, after_seal: Callable[[int], None] | None = None,
              after_wave_seal: Callable[[int], None] | None = None,
              after_comparison: Callable[[int], None] | None = None) -> None:
        cursor = Cursor(0, 0, None)
        previous = "0" * 64
        segments = (self.source.total_rows + wave.SEGMENT_ROWS - 1) // wave.SEGMENT_ROWS
        for number in range(segments):
            path = self._cursor_path(number)
            if path.exists():
                receipt, end = self._read_cursor(number, previous)
                wave.need(receipt["start"] == [cursor.game, cursor.row,
                                                cursor.game_sha256],
                          "cursor chain continuity")
                cursor = end
                previous = wave.file_sha(path)
                continue
            start = cursor
            expected = min(wave.SEGMENT_ROWS,
                           self.source.total_rows - number * wave.SEGMENT_ROWS)
            source_rows = self.source.rows_from(start)
            existing = self.store._path(number).exists()
            if existing:
                comparison = self.store.root / f"compare_{number:08d}.json"
                if comparison.exists():
                    self._compare_source_to_sealed(
                        number, source_rows, expected)
                else:
                    self.store.compare_wave2(
                        number, _take(source_rows, expected))
                    if after_comparison is not None:
                        after_comparison(number)
            else:
                self.store.seal_wave1(number, _take(source_rows, expected),
                                      expected_rows=expected)
                if after_wave_seal is not None:
                    after_wave_seal(number)
            end = self.source.current
            wave.need(end != start, "cursor did not advance")
            receipt = {"schema": "source_cursor_segment_v2",
                       "claim_sha256": wave.sha(wave.canonical(self.claim)),
                       "segment": number, "start": [start.game, start.row,
                                                        start.game_sha256],
                       "end": [end.game, end.row, end.game_sha256],
                       "previous_cursor_receipt_sha256": previous,
                       "wave_receipt_sha256": wave.file_sha(
                           self.store._path(number) / "RECEIPT.json")}
            wave.atomic_write(path, wave.canonical(receipt))
            cursor = end
            previous = wave.file_sha(path)
            if after_seal is not None:
                after_seal(number)
        wave.need(cursor == Cursor(len(self.source.locators), 0, None),
                  "source cursor final coverage")

    def wave2_metadata_sort(self, sorter: sort.CheckpointedSort) -> list[Path]:
        wave.need(sorter.kind == "digest" and
                  sorter.source_sha256 == self.source.identity and
                  sorter.config_sha256 == self.store.claim["config_sha256"],
                  "sort source/config identity")
        runs = []
        cursor = Cursor(0, 0, None)
        previous = "0" * 64
        segments = (self.source.total_rows + wave.SEGMENT_ROWS - 1) // wave.SEGMENT_ROWS
        for number in range(segments):
            path = self._cursor_path(number)
            receipt, end = self._read_cursor(number, previous)
            wave.need(receipt["start"] == [cursor.game, cursor.row,
                                            cursor.game_sha256],
                      "cursor segment continuity")
            expected = self.store.verify_segment(number)["rows"]
            comparison = self.store.root / f"compare_{number:08d}.json"
            if comparison.exists():
                # Immutable source/code pins are in the claim. A completed
                # comparison is read back, never silently rerun on no-op.
                self.store.verify_comparison(number)
            else:
                source_rows = self.source.rows_from(cursor)
                self.store.compare_wave2(number, _take(source_rows, expected))
                wave.need(self.source.current == end, "paired replay cursor end")
            meta = self.root / f"candidate_{number:08d}"
            candidate.seal(self.store, number, meta)
            group = []
            receipt_sha = wave.file_sha(meta / "RECEIPT.json")
            for entry in candidate.iter_sort_entries(self.store, number, meta):
                group.append(entry)
                if len(group) == sort.MAX_RUN_ROWS:
                    index = len(runs)
                    runs.append(self._seal_sort_group(
                        sorter, group, receipt_sha, index))
                    group = []
            if group:
                index = len(runs)
                runs.append(self._seal_sort_group(
                    sorter, group, receipt_sha, index))
            cursor = end
            previous = wave.file_sha(path)
        return runs
