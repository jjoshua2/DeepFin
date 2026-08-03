"""audit-v2: feed the frozen audit set the input encoding production actually uses.

The v1 audit rulers (`scripts/audit_targets.py`, `scripts/value_regret.py`)
rebuild every input from the FEN alone, so `chess.Board(fen)` carries an EMPTY
move stack. Under the production encoding (`lc0_root_legacy_meta` + `v2_threats`,
175 planes) that leaves 93 of 175 planes structurally zero on every row -- the 7
non-current lc0 history slots (planes 13..103), the current-frame repetition
plane (12) -- and pins the side-to-move colour flag (plane 108) to 0, which is
the WRONG value on the ~51% of rows whose original position had black to move.
The audit SET is sound; the v1 RULER fed the net an input distribution it never
trained on.

audit-v2 replaces ONLY the input encoding. Positions, deep-SF labels, legal-move
indexing and the metrics are untouched. Two encodings are therefore selectable
everywhere the audit set is scored:

    fen_only   the v1 ruler: `chess.Board(fen)` -> empty move stack. DEFAULT,
               and bit-identical to the pre-2026-08-02 behaviour.
    stored     the production row recovered by position-key join against the
               replay snapshot the audit set was sampled from
               (`scripts/match_audit_rows.py`), with 1-ply children DERIVED
               from it by `pov_flip_slot`.

⚑ A RULER CHANGE INVALIDATES ITS RECORDS. A `stored` number and a `fen_only`
number are different rulers and must never share a table, a trend line or a
threshold. Every report line produced under either encoding names it; keep it
that way.

Child rows exist because the value ruler evaluates every legal child one ply
deep, so children need history too:

    child slot k+1 (planes 13(k+1) .. 13(k+1)+12) = pov_flip_slot(parent slot k)

`pov_flip_slot` swaps the us/them 6-plane halves and mirrors the rank axis --
the root-POV change from parent-to-move to child-to-move. That transform is not
asserted, it is VERIFIED bit-exactly against real consecutive-ply stored row
pairs (`scripts/verify_audit_history_transform.py`, regression-pinned by
`tests/test_audit_history_encoding.py` against a checked-in fixture).

What a derived child does NOT get: its own current-frame repetition plane (12).
That depends on positions older than the parent's 8-slot window, which no shift
can recover; it stays at whatever `encode_cboard` produced for the bare pushed
board (0). The snapshot verification measures the size of this gap directly --
4 misses in 28,539 slots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    LC0_HISTORY_FRAMES,
    LC0_HISTORY_ROOT_LEGACY_META,
    LC0_PLANES_PER_FRAME,
)

# The two input encodings the audit rulers accept. `fen_only` is the default
# everywhere and must stay bit-identical to the pre-audit-v2 behaviour.
INPUT_ENCODINGS: Final[tuple[str, ...]] = ("fen_only", "stored")
INPUT_ENCODING_DEFAULT: Final[str] = "fen_only"

# Stored rows are production rows, so they are only meaningful for a model that
# declares the encoding they were written under.
STORED_HISTORY_ENCODING: Final[str] = LC0_HISTORY_ROOT_LEGACY_META
STORED_EXTRA_FEATURES: Final[str] = "v2_threats"
STORED_PLANES: Final[int] = 175

# lc0 root layout: 8 slots x 13 planes (6 us, 6 them, 1 repetition), then the
# metadata block at 104 (4 castling planes, then the colour flag at 108).
SLOT_PLANES: Final[int] = LC0_PLANES_PER_FRAME
N_SLOTS: Final[int] = LC0_HISTORY_FRAMES
STM_PLANE: Final[int] = LC0_FULL.root_metadata_base + LC0_FULL.num_castling_planes


def normalize_input_encoding(encoding: str | None) -> str:
    """Validate an `--input-encoding` value, defaulting to `fen_only`."""
    enc = str(encoding or INPUT_ENCODING_DEFAULT).lower().strip()
    if enc not in INPUT_ENCODINGS:
        raise ValueError(
            f"unknown input encoding {encoding!r}; expected one of "
            f"{', '.join(INPUT_ENCODINGS)}"
        )
    return enc


def slot_bounds(slot: int) -> tuple[int, int]:
    """[lo, hi) plane range of history slot `slot` (0 = current frame)."""
    if not 0 <= slot < N_SLOTS:
        raise ValueError(f"history slot {slot} out of range 0..{N_SLOTS - 1}")
    lo = SLOT_PLANES * slot
    return lo, lo + SLOT_PLANES


def pov_flip_slot(slot: np.ndarray) -> np.ndarray:
    """Re-express one 13-plane history slot from the opposite root POV.

    `slot` is (13, 8, 8): planes 0..5 us pieces, 6..11 them pieces, 12
    repetition. Changing side to move swaps us/them and mirrors the board
    vertically; the repetition flag is POV-invariant.
    """
    arr = np.asarray(slot)
    if arr.shape != (SLOT_PLANES, 8, 8):
        raise ValueError(f"history slot must be {(SLOT_PLANES, 8, 8)}, got {arr.shape}")
    out = np.empty_like(arr)
    out[0:6] = arr[6:12, ::-1, :]
    out[6:12] = arr[0:6, ::-1, :]
    out[12] = arr[12]
    return out


def child_input_planes(
    parent_stored: np.ndarray, child_fen_planes: np.ndarray,
) -> np.ndarray:
    """Splice a child's real history onto its own FEN-only encoding.

    parent_stored     : (175, 8, 8) the parent's STORED production row.
    child_fen_planes  : (175, 8, 8) `encode_cboard` of the pushed child board.

    Everything except planes 13..103 and 108 comes from `child_fen_planes`,
    which is already correct for a bare board: slot 0, castling, rule50, EP and
    the v2_threats feature planes are all functions of the child position alone.
    """
    parent = _as_planes(parent_stored, "parent_stored")
    out = np.array(_as_planes(child_fen_planes, "child_fen_planes"), copy=True)
    for k in range(N_SLOTS - 1):
        src_lo, src_hi = slot_bounds(k)
        dst_lo, dst_hi = slot_bounds(k + 1)
        out[dst_lo:dst_hi] = pov_flip_slot(parent[src_lo:src_hi])
    out[STM_PLANE] = 1.0 - float(parent[STM_PLANE, 0, 0])
    return out


def root_input_planes(
    *, encoding: str, fen_planes: np.ndarray, stored_planes: np.ndarray | None,
) -> np.ndarray:
    """The root input the ruler feeds the net, under `encoding`.

    `fen_only` returns `fen_planes` UNCHANGED (identity — this is the property
    `tests/test_audit_history_encoding.py` pins bit-exactly); `stored` returns
    the stored production row.
    """
    enc = normalize_input_encoding(encoding)
    if enc == "fen_only":
        return fen_planes
    if stored_planes is None:
        raise ValueError("input encoding 'stored' needs the matched stored row")
    return _as_planes(stored_planes, "stored_planes")


def child_planes_for_encoding(
    *,
    encoding: str,
    child_fen_planes: np.ndarray,
    stored_parent: np.ndarray | None,
) -> np.ndarray:
    """The 1-ply child input the value ruler feeds the net, under `encoding`.

    `fen_only` returns `child_fen_planes` UNCHANGED (identity, pinned by test).
    """
    enc = normalize_input_encoding(encoding)
    if enc == "fen_only":
        return child_fen_planes
    if stored_parent is None:
        raise ValueError("input encoding 'stored' needs the matched stored parent row")
    return child_input_planes(stored_parent, child_fen_planes)


def verify_child_transform(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:
    """Bit-exact check of `pov_flip_slot` on real (parent, child) stored rows.

    `pairs` are consecutive-ply rows of the same game, both stored production
    rows. Compares every derivable child slot (1..7) and the colour flag. The
    repetition plane is counted separately because a repetition older than the
    parent's 8-slot window is unrecoverable by construction — that residual is
    a measured property of the transform, not a failure of it.
    """
    slot_ok = 0
    slot_total = 0
    rep_ok = 0
    stm_ok = 0
    for raw_parent, raw_child in pairs:
        parent = _as_planes(raw_parent, "parent")
        child = _as_planes(raw_child, "child")
        for k in range(N_SLOTS - 1):
            src_lo, src_hi = slot_bounds(k)
            dst_lo, dst_hi = slot_bounds(k + 1)
            got = pov_flip_slot(parent[src_lo:src_hi])
            want = child[dst_lo:dst_hi]
            slot_total += 1
            slot_ok += int(np.array_equal(got[:12], want[:12]))
            rep_ok += int(np.array_equal(got[12], want[12]))
        stm_ok += int(
            float(child[STM_PLANE, 0, 0]) == 1.0 - float(parent[STM_PLANE, 0, 0])
        )
    return {
        "pairs": len(pairs),
        "piece_slots_exact": slot_ok,
        "piece_slots_total": slot_total,
        "repetition_planes_exact": rep_ok,
        "repetition_planes_total": slot_total,
        "stm_flag_exact": stm_ok,
        "stm_flag_total": len(pairs),
    }


def _as_planes(x: np.ndarray, what: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != (STORED_PLANES, 8, 8):
        raise ValueError(
            f"{what} must be {(STORED_PLANES, 8, 8)} production planes, got {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# The matched-rows index
# ---------------------------------------------------------------------------


def default_matched_rows_path(audit_set: str | Path) -> Path:
    """Where `scripts/match_audit_rows.py` writes the index for an audit set.

    Sits next to the set under the same suffix-append convention as the
    shallow-SF cache (`<audit>.shallow_sf.jsonl`). Neither the audit set nor
    this index is checked in — both are build artifacts of a named snapshot.
    """
    p = Path(audit_set)
    return p.with_suffix(p.suffix + ".matched_rows.npz")


BUILD_HINT: Final[str] = (
    "build it with:\n"
    "  PYTHONPATH=. python3 scripts/match_audit_rows.py \\\n"
    "      --audit-set data/audit_set_v1.jsonl \\\n"
    "      --snapshot runs/parallel_candidate_replay_snapshots/"
    "current_live_20260602_202037_v2threats"
)


class MatchedAuditRows:
    """Stored production rows for the frozen audit set, keyed by position key.

    Produced by `scripts/match_audit_rows.py`. Rows that failed to match are
    absent from `keys` — a caller must restrict the scored set to `keys`, never
    silently feed a zero row.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise SystemExit(
                f"--input-encoding stored needs the matched-rows index "
                f"{self.path}, which does not exist. {BUILD_HINT}"
            )
        data = np.load(self.path, allow_pickle=False)
        # `x_stored`/`x_fen_only` are the current names; `x_v2`/`x_v1` are the
        # 2026-08-02 bank's, kept readable so the ledger's published numbers
        # stay reproducible from the artifact that produced them.
        stored_key = "x_stored" if "x_stored" in data else "x_v2"
        self._stored = data[stored_key]
        self._game_id = np.asarray(data["game_id"], dtype=np.int64)
        found = np.asarray(data["found"], dtype=bool)
        all_keys = [str(k) for k in data["key"]]
        self._index: dict[str, int] = {
            k: i for i, k in enumerate(all_keys) if found[i]
        }
        self.n_audit_rows = len(all_keys)
        self.n_matched = int(found.sum())
        self.input_history_encoding = _npz_str(data, "input_history_encoding")
        self.input_extra_features = _npz_str(data, "input_extra_features")
        self.snapshot = _npz_str(data, "snapshot")

    def __contains__(self, key: object) -> bool:
        return str(key) in self._index

    def stored_row(self, key: str) -> np.ndarray:
        """The (175, 8, 8) float32 stored production row for `key`."""
        return np.asarray(self._stored[self._row(key)], dtype=np.float32)

    def game_id(self, key: str) -> int:
        """Source game of the recovered row, for game-CLUSTERED resampling.

        Audit positions from the same game are not independent draws; a
        bootstrap that resamples positions rather than games understates its
        own CI. -1 when the shard did not carry `game_id`.
        """
        return int(self._game_id[self._row(key)])

    def _row(self, key: str) -> int:
        i = self._index.get(str(key))
        if i is None:
            raise KeyError(f"no stored row matched audit position {key!r}")
        return i

    def require_model_compatible(self, enc_kwargs: dict[str, str]) -> None:
        """Fail loudly when the checkpoint's encoding is not the stored one.

        Stored rows are bytes written under one specific layout. Feeding them
        to a model that declares a different history encoding or feature set
        would produce a number that silently means nothing, so this refuses
        rather than warns.
        """
        want = {
            "input_history_encoding": STORED_HISTORY_ENCODING,
            "input_extra_features": STORED_EXTRA_FEATURES,
        }
        bad = {k: v for k, v in want.items() if str(enc_kwargs.get(k)) != v}
        if bad:
            raise SystemExit(
                "--input-encoding stored requires a checkpoint encoded as "
                f"{want}; this checkpoint declares "
                f"{ {k: enc_kwargs.get(k) for k in want} }"
            )
        # The index itself records what it was built from. Older indexes
        # predate the field; say so instead of assuming.
        recorded = {
            "input_history_encoding": self.input_history_encoding,
            "input_extra_features": self.input_extra_features,
        }
        if all(v is None for v in recorded.values()):
            print(
                f"[audit-v2] NOTE: {self.path.name} predates encoding provenance; "
                f"assuming {want} (the only layout match_audit_rows.py writes)"
            )
            return
        mismatch = {
            k: recorded[k] for k, v in want.items()
            if recorded[k] is not None and recorded[k] != v
        }
        if mismatch:
            raise SystemExit(
                f"matched-rows index {self.path} was built from {mismatch}, "
                f"which is not the production stored layout {want}"
            )


def _npz_str(data: object, key: str) -> str | None:
    files = getattr(data, "files", ())
    if key not in files:
        return None
    return str(np.asarray(data[key]).reshape(-1)[0])  # pyright: ignore[reportIndexIssue]
