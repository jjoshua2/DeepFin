"""The BT4 shard dump's RIG JOIN KEY, its ``--restrict-keys`` filter, and — the
one that decides whether any of it was worth doing — whether the R/V/G rig's own
loader ACCEPTS the file.

The question this file exists to answer is not "does the dump run". It is "does
the CONSUMER accept what the dump wrote, and does it join to the same rows the
label pass keyed". So:

* the key-parity test drives ``scripts.rvg_label_pass.scan_corpus`` and
  ``scripts.bt4_policy_dump.iter_shard_rows`` — BOTH real, on the same shard —
  and compares what they produce. There is no expected key list written out
  here; a copied list would agree with a drifted local re-derivation of the
  fingerprint, which is the exact failure the test is aimed at;
* the consumer test drives the real
  ``chess_anti_engine.eval.rvg_surgery.RvgExternalPolicyIndex.load``, then
  breaks the file in the two ways that loader refuses (no ``v``, duplicate key)
  and pins that it does refuse.

⚑ THE ONLY STUB IS THE ONNX SESSION. ``open_session`` is replaced with a fake
graph, because the BT4 ``.onnx`` is a 700 MB runtime artifact that is not in
the tree (``data/`` is uncommitted) — so a test that loaded it would pass on
this machine and error everywhere else. Everything downstream of the logits —
key derivation, restriction, dedup, the header, the file bytes, the loader — is
the production code path. The same flow WAS run once against the real
``BT4-it332-vanilla-winner.onnx`` on CPU, and the rig loader accepted the
result; that run is not reproducible from a clean checkout, which is why it
lives in the commit message and not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.eval.rvg_surgery import (
    FINGERPRINT_BYTES,
    RVG_EXTERNAL_POLICY_SCHEMA_VERSION,
    RvgExternalPolicyIndex,
    position_fingerprints,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, POLICY_ENCODING_LC0_1858
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.replay.shard import save_local_shard_arrays
from scripts import bt4_policy_dump as dump
from scripts.rvg_label_pass import scan_corpus

SHARD_ENCODING = "lc0_root_legacy_meta"

#: Six rows, and each one is here to make a specific wrong fingerprint visible.
#: ⚑ THE FIRST FIXTURE WAS TOO WEAK AND THE MUTATION PASS SAID SO: with only
#: all-rights-or-no-rights positions in it, a copy of the digest that read the
#: castling planes in the WRONG ORDER produced identical keys (all four bits set
#: is order-invariant) and the parity test passed under the mutant. Rows [4] and
#: [5] were added for exactly that reason; do not trim this tuple back.
#:
#: ``[0]`` white to move, all four castling rights.
#: ``[1]`` the SAME position with BLACK to move and the same clock — the board
#:     is mirror-symmetric, so rows 0 and 1 are ONE position in the
#:     POV-normalized key space and must collapse to a single record.
#: ``[2]`` no castling rights and a non-zero halfmove clock (12) — a wrong
#:     rule50 SCALE quantizes it and changes the key.
#: ``[3]`` the opening position.
#: ``[4]`` ASYMMETRIC castling (white kingside + black queenside only) — the
#:     castling plane ORDER is observable only on a row like this.
#: ``[5]`` a legal en-passant square — pins the EP file (and its ``+1`` offset).
FIXTURE_FENS = (
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "8/8/6r1/1np5/p2k4/P7/8/2K5 w - - 12 40",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1",
    "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
)


# --------------------------------------------------------------- the fixture
def _canonical(board: chess.Board) -> chess.Board:
    """The white-to-move frame the planes are stored in."""
    return board if board.turn == chess.WHITE else board.mirror()


def write_shard(
    directory: Path, fens: tuple[str, ...] = FIXTURE_FENS, *, planes: int | None = None,
) -> Path:
    """A real replay shard both readers accept, written by the production writer.

    ``planes`` truncates ``x`` to that many planes, for the too-narrow case.
    """
    boards = [chess.Board(f) for f in fens]
    n = len(boards)
    xs = np.stack([
        encode_position(b, add_features=True, input_history_encoding=SHARD_ENCODING)
        for b in boards
    ]).astype(np.float16)
    if planes is not None:
        xs = xs[:, :planes]
    mask = np.zeros((n, COMPACT_POLICY_SIZE), dtype=np.uint8)
    policy = np.zeros((n, COMPACT_POLICY_SIZE), dtype=np.float16)
    for i, board in enumerate(boards):
        canon = _canonical(board)
        for move in canon.legal_moves:
            idx = compact_index_for_move(canon, move)
            if idx >= 0:
                mask[i, idx] = 1
                policy[i, idx] = 1.0
    directory.mkdir(parents=True, exist_ok=True)
    return save_local_shard_arrays(
        directory / "shard_000000.zarr",
        arrs={
            "x": xs,
            "policy_target": policy,
            "wdl_target": np.zeros(n, dtype=np.int8),
            "priority": np.ones(n, dtype=np.float32),
            "has_policy": np.ones(n, dtype=np.uint8),
            "legal_mask": mask,
            "has_legal_mask": np.ones(n, dtype=np.uint8),
            "game_id": np.arange(n, dtype=np.int64),
            "has_game_id": np.ones(n, dtype=np.uint8),
            "ply_index": np.arange(n, dtype=np.int32),
            "has_ply_index": np.ones(n, dtype=np.uint8),
        },
        meta={
            "input_history_encoding": SHARD_ENCODING,
            "policy_encoding": POLICY_ENCODING_LC0_1858,
        },
    )


def fixture_keys(fens: tuple[str, ...] = FIXTURE_FENS) -> list[str]:
    """Hex keys for the fixture rows, derived the way the RIG derives them.

    Used to build ``--restrict-keys`` inputs and to state expected counts. It
    calls the shared function, so it cannot certify a drifted copy of it — the
    parity test is what pins the derivation itself.
    """
    xs = np.stack([
        encode_position(chess.Board(f), add_features=True,
                        input_history_encoding=SHARD_ENCODING)
        for f in fens
    ]).astype(np.float16)
    return [
        k.hex()
        for k in position_fingerprints(xs, input_history_encoding=SHARD_ENCODING)
    ]


def test_the_fixture_can_see_what_it_claims_to_see() -> None:
    """⚑ THE FIXTURE'S OWN GUARD. Every test below is vacuous unless the rows
    actually differ in the ways their comments claim.

    (1) Rows 0 and 1 must COLLIDE: two different FENs (white to move vs black to
        move) that are ONE position once the encoding normalizes for side to
        move — a dump that deduped on the FEN, as this one used to, emits both.
    (2) Row 4 must differ from row 0 on castling rights ALONE, so a digest that
        reads the castling planes in the wrong order is visible.
    (3) Row 5 must differ from a copy of itself with no en-passant square, so
        the EP field of the digest is exercised.
    """
    keys = fixture_keys()
    assert keys[0] == keys[1]
    assert FIXTURE_FENS[0] != FIXTURE_FENS[1]
    assert len(set(keys)) == 5

    # (2) rows 0 and 4 share every plane except the castling bits.
    assert chess.Board(FIXTURE_FENS[0]).board_fen() == \
        chess.Board(FIXTURE_FENS[4]).board_fen()
    assert keys[0] != keys[4]

    # (3) the same board without the EP square is a different key.
    no_ep = FIXTURE_FENS[5].replace(" d6 ", " - ")
    with_ep_key, no_ep_key = fixture_keys((FIXTURE_FENS[5], no_ep))
    assert with_ep_key != no_ep_key


# ------------------------------------------------------------- the fake graph
class _FakeOutput:
    def __init__(self, name: str, width: int) -> None:
        self.name = name
        self.shape = ["batch", width]


class _FakeSession:
    """``logit[i] == i / 1858`` on every row — monotone in the slot index, so the
    emitted probabilities name the slots they were gathered from, and scaled so
    the softmax stays in a sane numeric range across 1858 entries."""

    def __init__(self) -> None:
        self._outs = [_FakeOutput("policy", COMPACT_POLICY_SIZE)]

    def get_outputs(self) -> list[_FakeOutput]:
        return self._outs

    def run(self, _names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = int(next(iter(feeds.values())).shape[0])
        row = np.arange(COMPACT_POLICY_SIZE, dtype=np.float32) / COMPACT_POLICY_SIZE
        return [np.tile(row, (batch, 1))]


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Replace the ONNX session only; every other stage stays production code."""
    monkeypatch.setattr(
        dump, "open_session",
        lambda *_a, **_kw: (_FakeSession(), "input", np.dtype(np.float32), ["Fake"]),
    )
    onnx = tmp_path / "fake.onnx"
    onnx.write_bytes(b"not a real graph; only its sha256 is read")
    return onnx


def dump_args(
    out: Path, shard_dir: Path, onnx: Path, **overrides: Any,
) -> argparse.Namespace:
    args = argparse.Namespace(
        out=str(out), rows=None, input_source=f"shards:{shard_dir}", resume=False,
        onnx=str(onnx), batch_size=8, threads=1, gpu_mem_gb=0.0, policy_output=None,
        limit=0, castle_examples=0, check_legal_mask=True, restrict_keys=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def read_dump(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "policy" in rec:
            rows.append(rec)
        else:
            header = rec
    return header, rows


# ------------------------------------------------------------------ key parity
def test_dump_keys_are_the_label_passs_own_key_derivation(tmp_path: Path) -> None:
    """⚑⚑ THE JOIN. Both sides run their REAL derivation over the same shard.

    ``scan_corpus`` is the label pass's row scanner and ``iter_shard_rows`` is
    the dump's; the assertion is that the (shard, row) -> key maps agree
    exactly. No expected key is written down here on purpose: a hardcoded list
    would still match a locally re-implemented digest as long as the list was
    regenerated from it, which is precisely the drift this test exists to catch.

    MUTANT (run, killed): re-implementing the fingerprint inside the dump as
    ``blake2b(x[:12] bits + castling + ep + clock)`` with any detail off — the
    rule50 scale, the castling plane ORDER, the ep ``+1`` — leaves a digest that
    is stable, 32 hex wide, and joins to nothing. Every such variant fails here.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)

    scan = scan_corpus(shard_dir, row_start=0, row_end=None, limit=0,
                       roundtrip_sample=len(FIXTURE_FENS))
    label_keys = {(r.shard, r.row): r.key.hex() for r in scan.rows}
    dump_keys = {
        (r.shard, r.row_index): r.key for r in dump.iter_shard_rows([str(shard_dir)])
    }

    assert label_keys, "the label pass found no rows; the fixture is wrong"
    assert dump_keys == label_keys
    # And the label pass's own plane-vs-board round-trip agreed on every row,
    # so "they match" is not two copies of one mistake.
    assert scan.fingerprint_roundtrip_checked == len(FIXTURE_FENS)
    assert scan.fingerprint_roundtrip_mismatch == 0
    assert scan.alignment_illegal == 0


def test_the_emitted_key_is_the_fingerprint_and_the_fen_moved_to_its_own_field(
    tmp_path: Path,
) -> None:
    """Shape of the change: ``key`` is 32 hex, ``fen`` is the canonical FEN."""
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    rows = list(dump.iter_shard_rows([str(shard_dir)]))
    assert len(rows) == len(FIXTURE_FENS)
    for row in rows:
        assert len(row.key) == FINGERPRINT_BYTES * 2
        bytes.fromhex(row.key)  # a real digest, not a FEN
        assert row.fen is not None
        assert row.fen.split()[1] == "w", "the fen field stays side-to-move canonical"
        assert row.identity()["fen"] == row.fen
    assert [r.key for r in rows] == fixture_keys()


def test_a_shard_too_narrow_to_fingerprint_is_refused(tmp_path: Path) -> None:
    """The guard fires as a REFUSAL, not a skip: a row that cannot be keyed
    cannot join, and emitting it under the empty-string key would make the rig
    refuse the whole file for a duplicate hours later."""
    shard_dir = tmp_path / "narrow"
    write_shard(shard_dir, planes=110)
    with pytest.raises(SystemExit, match="too narrow"):
        list(dump.iter_shard_rows([str(shard_dir)]))


# ------------------------------------------------------------- the restriction
def test_restrict_keys_keeps_exactly_the_enumerated_rows(tmp_path: Path) -> None:
    """PREDICTED FIRST, then asserted.

    The fixture holds 6 rows over 5 distinct keys (rows 0 and 1 collide). The
    restrict file names key[2] and key[3], so rows 0, 1, 4 and 5 are restricted
    away and rows 2 and 3 kept:
        kept = 2, restrict_skipped = 4, seen_rows = 6.

    MUTANT (run, killed): drop the ``restrict`` clause from ``source_rows`` and
    all 6 rows arrive as 5 distinct keys.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    keys = fixture_keys()
    restrict = {keys[2], keys[3]}

    stats: dict[str, int] = {}
    args = dump_args(tmp_path / "unused.jsonl", shard_dir, tmp_path / "unused.onnx")
    kept = list(dump.source_rows(args, set(), stats, restrict))

    assert [r.key for r in kept] == [keys[2], keys[3]]
    assert stats["seen_rows"] == 6
    assert stats["distinct_keys"] == 2
    assert stats["restrict_skipped"] == 4
    # ⚑ NOT an ordering claim. `seen` only ever receives keys that were KEPT, so
    # a restricted-away row cannot be charged to `collisions` no matter where
    # the restrict clause sits relative to the dedup — a mutation pass proved
    # that by flipping the two and watching this test still pass. The ordering
    # that IS observable is restrict-before-resume, pinned in the next test.
    assert stats["collisions"] == 0


def test_a_restricted_away_row_is_not_charged_to_the_resume_counter(
    tmp_path: Path,
) -> None:
    """The ordering claim that IS observable: restrict runs BEFORE the resume
    and terminal checks.

    Row 0's key is handed in as already-done AND left out of the restrict set.
    It must count as ``restrict_skipped``, not ``resume_skipped`` — the caller
    did not ask for it, so "already dumped" is not the reason it was passed
    over. Moving the restrict clause below the ``done`` check flips these two.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    keys = fixture_keys()
    stats: dict[str, int] = {}
    args = dump_args(tmp_path / "unused.jsonl", shard_dir, tmp_path / "unused.onnx")
    kept = list(dump.source_rows(args, {keys[0]}, stats, {keys[3]}))
    assert [r.key for r in kept] == [keys[3]]
    assert stats["restrict_skipped"] == 5
    assert stats["resume_skipped"] == 0


def test_without_the_restriction_the_same_fixture_yields_five_rows(
    tmp_path: Path,
) -> None:
    """The other direction of the same gate — and the dedup count that proves
    rows 0 and 1 collapsed to one record."""
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    stats: dict[str, int] = {}
    args = dump_args(tmp_path / "unused.jsonl", shard_dir, tmp_path / "unused.onnx")
    kept = list(dump.source_rows(args, set(), stats, None))
    assert len(kept) == 5
    assert stats["distinct_keys"] == 5
    assert stats["collisions"] == 1
    assert stats["restrict_skipped"] == 0


def test_restrict_file_skips_a_jsonl_provenance_header(tmp_path: Path) -> None:
    """A label file can be fed back in whole; only ``{``-leading lines are
    tolerated, matching ``rvg_label_pass.py``'s ``--restrict-to``."""
    keys = fixture_keys()
    path = tmp_path / "restrict.txt"
    path.write_text(
        json.dumps({"record": "provenance", "v": 2}) + "\n"
        + keys[2] + "\n\n" + keys[3].upper() + "\n",
        encoding="utf-8",
    )
    assert dump.load_restrict_keys(path) == {keys[2], keys[3]}


@pytest.mark.parametrize(
    ("line", "match"),
    [("not-hex-at-all", "not a hex key"), ("dead" * 4, "is 8 bytes")],
)
def test_restrict_file_refuses_a_key_it_cannot_use(
    tmp_path: Path, line: str, match: str,
) -> None:
    """⚑ A bad key is a REFUSAL, not a skip. Dropping it silently would restrict
    to fewer rows than the caller enumerated, and "the dump came out short" is
    not a symptom anyone traces back to a typo."""
    path = tmp_path / "restrict.txt"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        dump.load_restrict_keys(path)


def test_restrict_keys_is_refused_in_fen_mode(tmp_path: Path) -> None:
    """FEN mode keys records by FEN, so a hex key set could only match zero rows.
    An empty dump and no error is the silent wrongness this refusal prevents."""
    rows = tmp_path / "rows.txt"
    rows.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")
    restrict = tmp_path / "restrict.txt"
    restrict.write_text(fixture_keys()[0] + "\n", encoding="utf-8")
    args = argparse.Namespace(
        out=str(tmp_path / "out.jsonl"), rows=str(rows), input_source="fens",
        resume=False, onnx="/nonexistent.onnx", batch_size=8, threads=1,
        gpu_mem_gb=0.0, policy_output=None, limit=0, castle_examples=0,
        check_legal_mask=True, restrict_keys=restrict,
    )
    with pytest.raises(SystemExit, match="shard mode only"):
        dump.run(args)
    assert not (tmp_path / "out.jsonl").exists(), "a refused run must write nothing"


def test_an_empty_restrict_file_is_refused(tmp_path: Path) -> None:
    """It would dump nothing, successfully — say so instead."""
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    restrict = tmp_path / "restrict.txt"
    restrict.write_text(json.dumps({"record": "provenance"}) + "\n", encoding="utf-8")
    args = dump_args(tmp_path / "out.jsonl", shard_dir, tmp_path / "fake.onnx",
                     restrict_keys=restrict)
    with pytest.raises(SystemExit, match="enumerates no keys"):
        dump.run(args)


# ------------------------------------------------------- ⚑⚑ THE CONSUMER TEST
def test_the_rig_loader_accepts_a_shard_dump_and_joins_it(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """⚑⚑ THE ONE THAT MATTERS. The real ``RvgExternalPolicyIndex.load`` reads
    the real dump — no monkeypatching of the loader, no hand-built file.

    "Does the dump run" is not the question; this repo's signature defect is a
    value accepted and then silently ignored, so the question is whether the
    CONSUMER takes the file and lands its probabilities on the right rows. The
    join is asserted through ``weights_for``, which is what the rig calls: a key
    the dump emitted returns a vector with mass on it, and a key it never
    emitted returns ``None``.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    out = tmp_path / "dump.jsonl"
    assert dump.run(dump_args(out, shard_dir, fake_onnx)) == 0

    index = RvgExternalPolicyIndex.load(out)          # refuses by raising

    assert len(index) == 5                             # the collision collapsed
    keys = fixture_keys()
    # Every fixture key resolves, through the loader's PUBLIC accessor; the
    # length above then says there is nothing else in there.
    probe = np.zeros((1,), dtype=np.int64)
    assert all(index.weights_for(bytes.fromhex(k), probe) is not None
               for k in set(keys))

    header, rows = read_dump(out)
    assert header["record"] == "provenance"
    assert header["v"] == RVG_EXTERNAL_POLICY_SCHEMA_VERSION
    assert header["policy_encoding"] == POLICY_ENCODING_LC0_1858
    assert header["contract"]["key"].startswith(f"{FINGERPRINT_BYTES * 2}-hex")
    assert "fen" in header["contract"]["key"]
    assert header["restrict_keys"] is None

    # The join, through the rig's own accessor. `move_index` is the shard's
    # stored legal mask for that row, i.e. exactly what the rig would pass.
    row = next(r for r in rows if r["key"] == keys[2])
    board = chess.Board(row["fen"])
    move_index = np.array(
        [compact_index_for_move(board, m) for m in board.legal_moves], dtype=np.int64,
    )
    weights = index.weights_for(bytes.fromhex(keys[2]), move_index)
    assert weights is not None
    assert weights.shape == move_index.shape
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-3)
    assert index.weights_for(b"\x00" * FINGERPRINT_BYTES, move_index) is None


def test_the_loader_refuses_a_dump_whose_header_lost_its_version(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """The refusal direction of the same gate: strip ``v`` and the load fails.

    Without this, "the loader accepted it" would be evidence of nothing — a
    loader that accepts everything accepts a wrong file too.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    out = tmp_path / "dump.jsonl"
    dump.run(dump_args(out, shard_dir, fake_onnx))

    stripped = tmp_path / "no_version.jsonl"
    lines = out.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header.pop("v") == RVG_EXTERNAL_POLICY_SCHEMA_VERSION
    stripped.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n",
                        encoding="utf-8")
    with pytest.raises(SystemExit, match="declares no schema version"):
        RvgExternalPolicyIndex.load(stripped)


def test_the_loader_refuses_a_duplicate_key(tmp_path: Path, fake_onnx: Path) -> None:
    """Why the dump dedups on the FINGERPRINT and not on the FEN.

    Rows 0 and 1 of the fixture are two FENs sharing one fingerprint. Re-adding
    the second one to a finished dump reproduces exactly what FEN-keyed dedup
    would have written, and the rig refuses it — so the dedup change is load
    bearing, not cosmetic.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    out = tmp_path / "dump.jsonl"
    dump.run(dump_args(out, shard_dir, fake_onnx))

    _, rows = read_dump(out)
    doubled = tmp_path / "doubled.jsonl"
    text = out.read_text(encoding="utf-8")
    doubled.write_text(text + json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate key"):
        RvgExternalPolicyIndex.load(doubled)


def test_a_restricted_dump_is_loadable_and_carries_the_restrict_provenance(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """End to end with the filter on: 2 records, both enumerated, and a header
    that names the file the restriction came from."""
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    keys = fixture_keys()
    restrict = tmp_path / "restrict.txt"
    restrict.write_text(
        json.dumps({"record": "provenance"}) + "\n" + keys[2] + "\n" + keys[3] + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "dump.jsonl"
    dump.run(dump_args(out, shard_dir, fake_onnx, restrict_keys=restrict))

    index = RvgExternalPolicyIndex.load(out)
    assert len(index) == 2
    header, rows = read_dump(out)
    assert [r["key"] for r in rows] == [keys[2], keys[3]]
    assert header["restrict_keys"]["path"] == str(restrict)
    assert header["restrict_keys"]["keys"] == 2
    assert header["restrict_keys"]["bytes"] == restrict.stat().st_size
    assert header["restrict_keys"]["mtime_utc"]


# --------------------------------------------------------------------- resume
def test_resume_keys_off_the_new_key_and_stays_loadable(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """A resumed dump must (a) skip the fingerprints already present and (b)
    still load — the appended provenance line is a ``provenance`` record, so the
    loader classifies it as a header rather than choking on its missing key."""
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    keys = fixture_keys()
    out = tmp_path / "dump.jsonl"
    first = tmp_path / "first.txt"
    first.write_text(keys[2] + "\n", encoding="utf-8")
    dump.run(dump_args(out, shard_dir, fake_onnx, restrict_keys=first))
    assert dump.load_done_keys(out) == {keys[2]}

    dump.run(dump_args(out, shard_dir, fake_onnx, resume=True))
    header, rows = read_dump(out)
    assert [r["key"] for r in rows] == [keys[2], keys[0], keys[3], keys[4], keys[5]]
    assert header["resumed"] is True
    assert header["record"] == "provenance"
    assert len(RvgExternalPolicyIndex.load(out)) == 5


def test_resume_of_a_pre_rig_key_dump_is_refused_not_silently_restarted(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """⚑ A shard dump written BEFORE this commit passes the net/output/source
    checks (same onnx, same policy_output, source "shards") but keys its rows
    by FEN and declares no ``v``. Loading its keys into ``done`` can never skip
    a fingerprint-keyed row, so a --resume would print a nonzero resume count
    and then silently re-scan and re-emit every row — and the merged file would
    still open with a ``header`` record the rig loader cannot classify. The
    guard must refuse on the schema mismatch, before touching the file.

    MUTANT (run, killed): deleting the ``v`` check in ``_resume_guard``
    restores the silent full restart this test pins.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    out = tmp_path / "legacy_dump.jsonl"
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    legacy_header = {
        "record": "header",
        "onnx": {"path": str(fake_onnx), "sha256": dump.file_sha256(fake_onnx)},
        "policy_output": "policy",
        "input": {"source": "shards"},
    }
    legacy_row = {"key": fen, "fen": fen, "policy": {"e2e4": 1.0}}
    out.write_text(
        json.dumps(legacy_header) + "\n" + json.dumps(legacy_row) + "\n",
        encoding="utf-8",
    )
    before = out.read_bytes()

    with pytest.raises(SystemExit, match="key-schema"):
        dump.run(dump_args(out, shard_dir, fake_onnx, resume=True))
    assert out.read_bytes() == before, "the refusal must fire before any write"


def test_fen_mode_resume_of_a_fen_dump_still_goes_through_the_guard(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """A FEN-mode dump declares no ``v`` and neither does a FEN-mode run, so
    the schema check must pass (None == None) and the resume must actually
    happen: old keys skipped, the new FEN appended.

    MUTANT (run, killed): comparing the prior ``v`` against the hardcoded
    schema CONSTANT instead of this run's own header (None != 1) would refuse
    every legitimate FEN/FEN resume; so would a truthiness check. Nothing else
    in the suite drives a SUCCESSFUL resume through ``_resume_guard`` in FEN
    mode, so without this test both mutants survive.
    """
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    rows = tmp_path / "rows.txt"
    rows.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")
    out = tmp_path / "fen_dump.jsonl"
    dump.run(dump_args(out, tmp_path, fake_onnx, input_source="fens",
                       rows=str(rows)))
    rows.write_text(chess.STARTING_FEN + "\n" + fen2 + "\n", encoding="utf-8")
    dump.run(dump_args(out, tmp_path, fake_onnx, input_source="fens",
                       rows=str(rows), resume=True))

    header, recs = read_dump(out)
    assert [r["key"] for r in recs] == [chess.STARTING_FEN, fen2]
    assert header["resumed"] is True
    assert "v" not in header


def test_resume_refuses_a_dump_from_a_newer_key_schema(
    tmp_path: Path, fake_onnx: Path,
) -> None:
    """The check is prior-vs-THIS-RUN ``v``, not "prior v is missing": a shard
    dump declaring a v this writer does not produce must be refused even though
    its record type, net, output and source all match.

    MUTANT (run, killed): classifying by ``record`` instead of ``v`` passes the
    legacy-refusal test (record "header" differs) and both success tests
    (records match), but lets a v=999 provenance dump resume — only this test
    catches it. A truthiness check (`if not old_v`) also passes 999 as fine.
    """
    shard_dir = tmp_path / "shards"
    write_shard(shard_dir)
    out = tmp_path / "future_dump.jsonl"
    future_header = {
        "record": "provenance", "v": 999, "policy_encoding": "lc0_1858",
        "onnx": {"path": str(fake_onnx), "sha256": dump.file_sha256(fake_onnx)},
        "policy_output": "policy",
        "input": {"source": "shards"},
    }
    row = {"key": "00" * 16, "fen": chess.STARTING_FEN, "policy": {"e2e4": 1.0}}
    out.write_text(
        json.dumps(future_header) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    before = out.read_bytes()

    with pytest.raises(SystemExit, match="key-schema"):
        dump.run(dump_args(out, shard_dir, fake_onnx, resume=True))
    assert out.read_bytes() == before


def test_read_header_still_reads_a_pre_existing_header_record(tmp_path: Path) -> None:
    """Dumps written before the rename exist on disk; a resume must still be
    able to READ their provenance. For a shard-mode legacy dump that read is
    what lets the schema guard refuse with a real message instead of the blunt
    "no readable header"; a FEN-mode legacy dump (keys unchanged) may still
    legitimately resume through it."""
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"record": "header", "onnx": {"sha256": "abc"}}) + "\n",
                   encoding="utf-8")
    header = dump.read_header(old)
    assert header is not None
    assert header["onnx"]["sha256"] == "abc"
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"record": "something_else"}) + "\n", encoding="utf-8")
    assert dump.read_header(other) is None
