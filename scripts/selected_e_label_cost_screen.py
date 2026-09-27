#!/usr/bin/env python3
"""Selected-E sparse-row cost-screen kernel. The CLI is deliberately NO-LAUNCH.

The existing BT4 and Ceres sidecar writers require incompatible whole-shard
source contracts. This kernel accepts *already admitted* stored E rows and two
caller-owned sessions per arm; it never opens a model, bank, or CUDA provider.
It reuses the producers' input conversion, legal mapping and target routines.
An authenticated sparse-E input adapter and arena-gated sole-GPU wrapper are
required before a physical labeling rate can be measured.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from chess_anti_engine.encoding import ceres_tpg as tpg
from chess_anti_engine.moves import leela_index as mapping
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import bt4_derived_wdl_sidecar as bt4_feed
from scripts import bt4_policy_dump as bt4_policy
from scripts import bt4_policy_mix as bt4_target
from scripts import bt4_raw_corpus_sidecar as bt4_raw
from scripts import ceres_target_mix as ceres_policy
from scripts import ceres_value_mix as ceres_value

_ROUTE_DOMAIN = b"factorial58-E-teacher-row-v0\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
BATCH = 32
HISTORY = "lc0_root_legacy_meta"


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _require_sha(value: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected lowercase SHA-256")


def _compact_legal_mask(board: Any) -> np.ndarray:
    legal = np.zeros(1858, dtype=np.uint8)
    for move in board.legal_moves:
        legal[compact_index_for_move(board, move)] = 1
    return legal


@dataclass(frozen=True, eq=False)
class Row:
    """A caller-authenticated E row; this type does not authenticate a bank."""

    cohort_manifest_sha256: str
    shard_ordinal: int
    shard_rows: int
    row_offset: int
    input_sha256: str
    legal_sha256: str
    selected_teacher: str
    policy_target_sha256: str
    search_wdl_sha256: str
    x: np.ndarray
    legal: np.ndarray

    def __post_init__(self) -> None:
        for value in (self.cohort_manifest_sha256, self.input_sha256,
                      self.legal_sha256, self.policy_target_sha256,
                      self.search_wdl_sha256):
            _require_sha(value)
        if self.selected_teacher not in ("BT4", "Ceres"):
            raise ValueError("selected teacher roster differs")
        if (type(self.shard_ordinal) is not int or self.shard_ordinal < 0
                or type(self.shard_rows) is not int or self.shard_rows < 1
                or type(self.row_offset) is not int
                or not 0 <= self.row_offset < self.shard_rows):
            raise ValueError("invalid shard ordinal or row offset")
        x = np.asarray(self.x)
        legal = np.asarray(self.legal)
        if (x.dtype != np.float16 or x.shape != (175, 8, 8)
                or legal.dtype != np.uint8 or legal.shape != (1858,)
                or not np.isin(legal, (0, 1)).all() or not legal.any()):
            raise ValueError("invalid E input or legal mask")
        if (_digest(x.tobytes(order="C")) != self.input_sha256
                or _digest(legal.tobytes(order="C")) != self.legal_sha256):
            raise ValueError("E input or legal byte pin differs")
        x = np.ascontiguousarray(x).copy()
        legal = np.ascontiguousarray(legal).copy()
        x.flags.writeable = legal.flags.writeable = False
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "legal", legal)

    @property
    def key(self) -> tuple[str, int, int]:
        return self.cohort_manifest_sha256, self.shard_ordinal, self.row_offset


def teacher_for_row(row: Row, routing_seed_sha256: str) -> str:
    """Reviewed factorial58-E-teacher-row-v0 bit; source ID never enters it."""
    _require_sha(routing_seed_sha256)
    key = (_ROUTE_DOMAIN + bytes.fromhex(row.cohort_manifest_sha256)
           + row.shard_ordinal.to_bytes(8, "big")
           + row.row_offset.to_bytes(8, "big")
           + bytes.fromhex(routing_seed_sha256))
    return "BT4" if sha256(key).digest()[0] & 1 == 0 else "Ceres"


def admitted_roster(rows: Sequence[Row], seed: str) -> tuple[list[int], list[int]]:
    """Fail before any session call on duplicate or mismapped stored inputs."""
    if not rows:
        raise ValueError("empty screen roster")
    seen_keys: set[tuple[str, int, int]] = set()
    seen_inputs: set[str] = set()
    selected: dict[str, list[int]] = {"BT4": [], "Ceres": []}
    shard_sizes: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        if row.key in seen_keys or row.input_sha256 in seen_inputs:
            raise ValueError("duplicate E row or exact stored input")
        seen_keys.add(row.key)
        seen_inputs.add(row.input_sha256)
        shard_key = row.cohort_manifest_sha256, row.shard_ordinal
        if shard_key in shard_sizes and shard_sizes[shard_key] != row.shard_rows:
            raise ValueError("selected shard row count differs")
        shard_sizes[shard_key] = row.shard_rows
        # BT4's own stored-input decoder is the source of the compact legal map.
        planes = bt4_feed.stored_feed(row.x[None])
        board = bt4_policy.board_from_stored_x(
            row.x, planes[0], input_history_encoding=HISTORY)
        if not np.array_equal(_compact_legal_mask(board), row.legal):
            raise ValueError("E legal mask differs from reconstructed input")
        teacher = teacher_for_row(row, seed)
        if teacher != row.selected_teacher:
            raise ValueError("selected teacher roster differs")
        selected[teacher].append(index)
    if not selected["BT4"] or not selected["Ceres"]:
        raise ValueError("screen sample must contain both routed teachers")
    return selected["BT4"], selected["Ceres"]


@dataclass(frozen=True)
class BT4Session:
    session: Any
    input_name: str
    input_dtype: np.dtype[Any]
    policy_name: str
    wdl_name: str


@dataclass(frozen=True)
class CeresSession:
    session: Any


@dataclass(frozen=True)
class Observation:
    policy_target: np.ndarray
    search_wdl: np.ndarray
    feed_sha256: str
    head_sha256: str


def _bt4_batches(rows: Sequence[Row], indices: Sequence[int], binding: BT4Session,
                 calls: list[dict[str, Any]]) -> dict[int, Observation]:
    result: dict[int, Observation] = {}
    contract = bt4_raw.resolve_wdl_output(
        binding.session, {"output": binding.wdl_name, "kind": "probabilities"},
        binding.policy_name)
    if contract is None or binding.session.get_outputs()[
            bt4_policy.resolve_policy_output(binding.session, binding.policy_name)
            ].name != binding.policy_name:
        raise ValueError("BT4 named policy/WDL contract differs")
    if np.dtype(binding.input_dtype) not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("BT4 input dtype differs")
    for first in range(0, len(indices), BATCH):
        batch = list(indices[first:first + BATCH])
        xs = np.stack([rows[i].x for i in batch])
        planes = bt4_feed.stored_feed(xs)
        feed = planes.astype(binding.input_dtype, copy=False)
        begun = time.perf_counter()
        fetched = binding.session.run(
            [binding.policy_name, binding.wdl_name], {binding.input_name: feed})
        elapsed = time.perf_counter() - begun
        if len(fetched) != 2:
            raise ValueError("BT4 did not return both named heads")
        policy, wdl = fetched
        if (not isinstance(policy, np.ndarray) or policy.shape != (len(batch), 1858)
                or policy.dtype != np.float32 or not np.isfinite(policy).all()):
            raise ValueError("BT4 policy dtype/shape/finite check failed")
        bt4_raw.validate_wdl_values(wdl, len(batch), contract)
        calls.append({"teacher": "BT4", "rows": batch, "physical_rows": len(batch),
                      "feed_sha256": _digest(feed.tobytes(order="C")),
                      "head_sha256": _digest(policy.tobytes() + wdl.tobytes()),
                      "inference_seconds": elapsed})
        for local, index in enumerate(batch):
            board = bt4_policy.board_from_stored_x(
                rows[index].x, planes[local], input_history_encoding=HISTORY)
            _, _, dense = bt4_policy.compact_legal_policy(board, policy[local])
            selected = bt4_target._tempered_bt4_policy(
                dense[None], rows[index].legal[None], temperature=.5)[0]
            native = np.asarray(wdl[local], dtype=np.float64)
            native /= native.sum()
            result[index] = Observation(
                selected.astype(np.float16), native.astype(np.float16),
                _digest(feed[local].tobytes()),
                _digest(policy[local].tobytes() + wdl[local].tobytes()))
    return result


def _ceres_batches(rows: Sequence[Row], indices: Sequence[int], binding: CeresSession,
                   calls: list[dict[str, Any]]) -> dict[int, Observation]:
    result: dict[int, Observation] = {}
    for first in range(0, len(indices), BATCH):
        batch = list(indices[first:first + BATCH])
        feed = tpg.stored_x_to_ceres_tpg_bytes(
            np.stack([rows[i].x for i in batch]),
            input_history_encoding=HISTORY, history_rep_fix=True)
        physical = feed if len(batch) == BATCH else np.concatenate(
            [feed, np.repeat(feed[-1:], BATCH - len(batch), axis=0)])
        begun = time.perf_counter()
        fetched = binding.session.run(
            ["policy", "value", "value2"], {"squares_byte": physical})
        elapsed = time.perf_counter() - begun
        if len(fetched) != 3 or any(
            not isinstance(head, np.ndarray) or head.dtype != np.float16
            or head.shape != (BATCH, 1858 if j == 0 else 3)
            or not np.isfinite(head).all() for j, head in enumerate(fetched)
        ):
            raise ValueError("Ceres native fixed32 heads differ")
        calls.append({"teacher": "Ceres", "rows": batch, "physical_rows": BATCH,
                      "feed_sha256": _digest(physical.tobytes(order="C")),
                      "head_sha256": _digest(b"".join(h.tobytes() for h in fetched)),
                      "inference_seconds": elapsed})
        gather = mapping.leela_gather_indices(*tpg.ceres_tpg_gather_context(feed))
        for local, index in enumerate(batch):
            legal = np.flatnonzero(rows[index].legal)
            slots = gather[local, legal]
            if (np.any(slots < 0) or np.any(slots >= 1858)
                    or len(np.unique(slots)) != len(slots)):
                raise ValueError("Ceres legal Leela map differs")
            logits = fetched[0][local, slots]
            dense_logits = np.zeros((1, 1858), dtype=np.float16)
            dense_logits[0, legal] = logits
            policy = ceres_policy.policy_target(
                rows[index].legal[None].astype(np.float64), dense_logits,
                rows[index].legal[None], bt4_weight=0.,
                bt4_temperature=.5, ceres_temperature=.5)[0]
            value = (.6 * ceres_value.softmax(fetched[1][local:local + 1], .55)
                     + .4 * ceres_value.softmax(fetched[2][local:local + 1], 1.5))[0]
            result[index] = Observation(
                policy.astype(np.float16), value.astype(np.float16),
                _digest(feed[local].tobytes()),
                _digest(b"".join(h[local].tobytes() for h in fetched)))
    return result


def _collect(rows: Sequence[Row], bt4_indices: Sequence[int],
             ceres_indices: Sequence[int], bt4: BT4Session,
             ceres: CeresSession) -> tuple[dict[str, dict[int, Observation]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    result = {"BT4": _bt4_batches(rows, bt4_indices, bt4, calls),
              "Ceres": _ceres_batches(rows, ceres_indices, ceres, calls)}
    return result, calls


def matched_cpu_kernel(rows: Sequence[Row], routing_seed_sha256: str,
                       selected_bt4: BT4Session, selected_ceres: CeresSession,
                       dual_bt4: BT4Session, dual_ceres: CeresSession
                       ) -> dict[str, Any]:
    """Run S and D on matched rows; fake sessions support CPU-only fault tests.

    Caller owns physical sessions and all row authentication. Timing here is
    diagnostic only: it excludes sample admission, session open and readback.
    """
    bt4_rows, ceres_rows = admitted_roster(rows, routing_seed_sha256)
    selected, s_calls = _collect(
        rows, bt4_rows, ceres_rows, selected_bt4, selected_ceres)
    dual, d_calls = _collect(
        rows, bt4_rows, ceres_rows, dual_bt4, dual_ceres)
    extra, extra_calls = _collect(
        rows, ceres_rows, bt4_rows, dual_bt4, dual_ceres)
    for teacher in ("BT4", "Ceres"):
        dual[teacher].update(extra[teacher])
    d_calls.extend(extra_calls)
    if (sum(map(len, selected.values())) != len(rows)
            or any(len(component) != len(rows) for component in dual.values())):
        raise ValueError("selected or dual roster lost a row")
    bt4_set = set(bt4_rows)
    chosen = {index: selected["BT4" if index in bt4_set else "Ceres"][index]
              for index in range(len(rows))}
    for index in range(len(rows)):
        teacher = "BT4" if index in bt4_set else "Ceres"
        component = dual[teacher][index]
        if (chosen[index].policy_target.tobytes() != component.policy_target.tobytes()
                or chosen[index].search_wdl.tobytes() != component.search_wdl.tobytes()
                or chosen[index].feed_sha256 != component.feed_sha256
                or chosen[index].head_sha256 != component.head_sha256):
            raise ValueError(f"selected/dual target or raw-head byte mismatch at row {index}")
        if (_digest(chosen[index].policy_target.tobytes()) != rows[index].policy_target_sha256
                or _digest(chosen[index].search_wdl.tobytes()) != rows[index].search_wdl_sha256):
            raise ValueError(f"independently pinned selected target byte mismatch at row {index}")
    for teacher in ("BT4", "Ceres"):
        s = [c for c in s_calls if c["teacher"] == teacher]
        d = [c for c in d_calls if c["teacher"] == teacher][:len(s)]
        if [(c["rows"], c["feed_sha256"], c["head_sha256"]) for c in s] != [
                (c["rows"], c["feed_sha256"], c["head_sha256"]) for c in d]:
            raise ValueError(f"{teacher} selected physical call roster differs")
    return {"schema": 1, "qualification": "cpu-kernel-only-no-launch",
            "rows": len(rows), "selected_counts": {"BT4": len(bt4_rows),
                                               "Ceres": len(ceres_rows)},
            "selected_calls": s_calls, "dual_calls": d_calls,
            "selected_policy_sha256": _digest(b"".join(
                chosen[i].policy_target.tobytes() for i in range(len(rows)))),
            "selected_wdl_sha256": _digest(b"".join(
                chosen[i].search_wdl.tobytes() for i in range(len(rows))))}


def main(argv: Sequence[str] | None = None) -> None:
    # A reviewed sparse-E adapter and strict arena-gate verifier must be added
    # before exposing a real-session entry point. Never infer launch from args.
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps({"status": "NO-LAUNCH", "reason":
        "missing authenticated sparse-E adapter and strict postarena gate"},
        sort_keys=True))


if __name__ == "__main__":
    main()
