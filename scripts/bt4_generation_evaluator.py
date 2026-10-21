"""BT4 root-output reuse adapter for a future opt-in neural generator.

This module does not start games or write labels. A caller evaluates a verified
root once, retains its native teacher observation, and hands its mapped search
logits to ``run_gumbel_root_many_c`` only when requested. Bound leaf batches
use the same session.

The caller supplies the session and asserted model SHA-256. This adapter checks
the declared head/input shapes and actual tensors, but does not hash an ONNX
file, attest the execution provider, or produce a publishable sidecar record.
A future writer must enforce the raw collector's full provenance contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.cboard_encode import encode_cboard, encode_cboard_batch
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding, x_to_lc0_planes
from chess_anti_engine.eval.rvg_surgery import FINGERPRINT_BYTES, position_fingerprints
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, legal_move_indices
from scripts import gen_sf_rooted_corpus as corpus
from scripts.bt4_model_arithmetic import validate_model_wdl
from scripts.bt4_policy_dump import compact_legal_policy, resolve_policy_output
from scripts.bt4_prepared_root import BT4LegalSnapshot, BT4PreparedInput
from scripts.bt4_raw_corpus_sidecar import resolve_wdl_output

_ZERO_LOGIT = np.float32(-1e9)


def _feed_row_sha256(input_name: str, row: np.ndarray) -> str:
    """Hash exactly one submitted ONNX row plus its input schema."""
    digest = hashlib.sha256()
    digest.update(input_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(row.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(row.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(row).tobytes())
    return digest.hexdigest()


def _search_policy_logits(dense_policy: np.ndarray) -> np.ndarray:
    """Compact search logits whose legal softmax retains exact zero mass."""
    logits = np.full((COMPACT_POLICY_SIZE,), _ZERO_LOGIT, dtype=np.float32)
    positive = dense_policy > 0.0
    logits[positive] = np.log(dense_policy[positive]).astype(np.float32)
    return logits


def _search_wdl_logits(raw: np.ndarray, kind: str) -> np.ndarray:
    """Stable search-only logits; never alter the native WDL observation."""
    values = np.asarray(raw, dtype=np.float64)
    if kind == "probabilities":
        logits = np.full((3,), float(_ZERO_LOGIT), dtype=np.float64)
        positive = values > 0.0
        logits[positive] = np.log(values[positive])
    elif kind == "logits":
        with np.errstate(over="ignore"):
            logits = values - values.max()
        logits = np.maximum(logits, float(_ZERO_LOGIT))
    else:
        raise ValueError(f"unsupported WDL output kind {kind!r}")
    result = logits.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BT4 WDL cannot be represented as finite search logits")
    return result


@dataclass(frozen=True)
class BT4RootOutput:
    """In-memory root observation, not a complete corpus provenance record."""

    fen: str
    input_key: str
    source_key: bytes
    policy_t1: np.ndarray  # dense lc0_1858, float32; not search-improved policy
    wdl_raw: np.ndarray  # native dtype and named head; side-to-move POV
    policy_output: str
    wdl_output: str
    wdl_kind: str
    model_sha256: str
    input_name: str
    input_dtype: str
    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool
    onnx_feed_sha256: str = ""
    verified_session_sha256: str | None = None

    def search_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Fresh batched logits for search; root-only actors retain no derived copy."""
        return (_search_policy_logits(self.policy_t1)[None],
                _search_wdl_logits(self.wdl_raw, self.wdl_kind)[None])


@dataclass(frozen=True)
class _PreparedFeed:
    feed: np.ndarray
    feed_hashes: tuple[str, ...]
    root: bool


@dataclass(frozen=True)
class BT4PreparedExecution:
    """Validated, independently owned CPU input and immutable root identities."""
    owner: Any
    inference: _PreparedFeed
    fens: tuple[str, ...]
    input_keys: tuple[str, ...]
    source_keys: tuple[bytes, ...]
    legal: tuple[BT4LegalSnapshot, ...]


class BT4OnnxEvaluator:
    """One named BT4 policy+WDL session, bound to the C tree for leaf mapping.

    Native ONNX policy slots are Leela-ordered. Search expects our compact slots
    and expands those to its 4672 action space; passing native 1858 rows directly
    would silently mislabel legal moves. Every root and leaf goes through the
    shared legal conversion before it reaches that expansion.
    """

    def __init__(
        self, sess: Any, *, input_name: str, input_dtype: np.dtype[Any],
        policy_output: str | None, wdl_output: str, wdl_kind: str,
        input_history_encoding: str, input_extra_features: str,
        model_sha256: str, history_rep_fix: bool,
        verified_session_sha256: str | None = None,
        fixed_batch_size: int | None = None,
    ) -> None:
        if type(history_rep_fix) is not bool:
            raise ValueError("BT4 history_rep_fix must be an explicit boolean")  # pyright: ignore[reportUnreachable]
        self.history_rep_fix = history_rep_fix
        self._require_history_rep_fix()
        if not input_name or np.dtype(input_dtype) not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("BT4 input needs a named float16/float32 tensor")
        if wdl_kind not in ("logits", "probabilities"):
            raise ValueError("BT4 WDL output kind must be logits or probabilities")
        if len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
            raise ValueError("BT4 model_sha256 must be lowercase SHA-256")
        self.sess = sess
        self.input_name = input_name
        self.input_dtype = np.dtype(input_dtype)
        self.input_history_encoding = normalize_lc0_history_encoding(input_history_encoding)
        self.input_extra_features = input_extra_features
        self.expected_planes = input_plane_count(input_extra_features)
        outputs = sess.get_outputs()
        self.policy_output = outputs[resolve_policy_output(sess, policy_output)].name
        contract = resolve_wdl_output(
            sess, {"output": wdl_output, "kind": wdl_kind}, self.policy_output,
        )
        if contract is None:
            raise ValueError("BT4 requires an explicit native WDL output")
        self.wdl_contract = contract
        self.model_sha256 = model_sha256
        self.verified_session_sha256 = verified_session_sha256
        if fixed_batch_size is not None and fixed_batch_size < 1:
            raise ValueError("BT4 fixed batch size must be positive")
        self.fixed_batch_size = fixed_batch_size
        self._tree: Any | None = None
        self.root_calls = 0
        self.root_rows = 0
        self.leaf_calls = 0
        self.leaf_rows = 0

    def _require_history_rep_fix(self) -> None:
        """Refuse a process-global mode change before any CBoard is accessed."""
        actual = rep_fix.current()
        if actual is not self.history_rep_fix:
            raise RuntimeError(
                f"BT4 history_rep_fix is {actual!r}; expected {self.history_rep_fix!r}. "
                "Configure the process before constructing boards; the evaluator "
                "never changes this mode.",
            )

    def bind_tree(self, tree: Any | None) -> None:
        """Bind the same explicit MCTSTree passed to the C search, or unbind."""
        self._tree = tree

    def _checked_inputs(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if (
            arr.ndim != 4 or arr.dtype != np.dtype(np.float32)
            or arr.shape[1:] != (self.expected_planes, 8, 8)
        ):
            raise ValueError(
                f"BT4 evaluator expected float32 (N,{self.expected_planes},8,8), "
                f"got {arr.dtype} {arr.shape}",
            )
        return arr

    def _prepare_feed(self, x: np.ndarray, *, root: bool) -> _PreparedFeed:
        if self.fixed_batch_size is not None and len(x) != self.fixed_batch_size:
            raise ValueError("BT4 row count differs from fixed ONNX batch size")
        feed = np.array(x_to_lc0_planes(
            x, input_history_encoding=self.input_history_encoding,
        ), dtype=self.input_dtype, copy=True, order="C")
        hashes = tuple(_feed_row_sha256(self.input_name, row) for row in feed) if root else ()
        feed.flags.writeable = False
        return _PreparedFeed(feed, hashes, root)

    def _execute_feed(self, prepared: _PreparedFeed) -> tuple[np.ndarray, ...]:
        self._require_history_rep_fix()
        if prepared.root:
            # Count submitted calls/rows, including a session failure.
            self.root_calls += 1
            self.root_rows += len(prepared.feed)
        fetched = self.sess.run(
            [self.policy_output, self.wdl_contract["output"]],
            {self.input_name: prepared.feed},
        )
        # Take output ownership before the session owner may run another call.
        return tuple(np.array(value, copy=True, order="C") for value in fetched)

    def _finish_feed(self, prepared: _PreparedFeed, fetched: Any) -> tuple[np.ndarray, np.ndarray]:
        if len(fetched) != 2:
            raise ValueError("BT4 session returned the wrong number of named outputs")
        policy = np.asarray(fetched[0], dtype=np.float32)
        values = np.asarray(fetched[1])
        if policy.shape != (len(prepared.feed), COMPACT_POLICY_SIZE):
            raise ValueError(f"BT4 policy shape {policy.shape} is not (N,{COMPACT_POLICY_SIZE})")
        validate_model_wdl(values, len(prepared.feed), self.wdl_contract, self.model_sha256)
        return policy, values

    def _infer(self, x: np.ndarray, *, root: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
        prepared = self._prepare_feed(x, root=root)
        policy, values = self._finish_feed(prepared, self._execute_feed(prepared))
        return policy, values, list(prepared.feed_hashes)

    def evaluate_root(self, board: chess.Board, x: np.ndarray) -> BT4RootOutput:
        """Singleton convenience path with the same validation as batched roots."""
        root = np.asarray(x)
        return self.evaluate_roots([board], root[None])[0]

    def evaluate_roots(
        self, boards: Sequence[chess.Board], x_batch: np.ndarray,
    ) -> list[BT4RootOutput]:
        """Strict public boundary: independently verify every full board history."""
        prepared = self._prepare_roots(boards, x_batch)
        return list(self.finish(prepared, self.execute(prepared)))

    def evaluate_prepared_roots(self, payloads: Sequence[BT4PreparedInput]) -> list[BT4RootOutput]:
        prepared = self.prepare(payloads)
        return list(self.finish(prepared, self.execute(prepared)))

    def prepare(self, rows: Sequence[Any]) -> BT4PreparedExecution:
        """CPU-only validation/conversion shared by synchronous and staged APIs."""
        self._require_history_rep_fix()
        rows = tuple(rows)
        if not rows:
            raise ValueError("BT4 root boards and encoded batch must have equal nonzero length")
        if isinstance(rows[0], BT4PreparedInput):
            for payload in rows:
                if type(payload) is not BT4PreparedInput:
                    raise TypeError("BT4 internal roots require canonical prepared inputs")
                if (payload.model_sha256 != self.model_sha256
                        or payload.input_history_encoding != self.input_history_encoding
                        or payload.input_extra_features != self.input_extra_features
                        or payload.history_rep_fix is not self.history_rep_fix
                        or not payload._matches_board(payload._borrow_board())):
                    raise ValueError("BT4 prepared input belongs to another root or model")
            return self._prepare_roots(
                [payload._borrow_board() for payload in rows],
                np.stack([payload.x for payload in rows]), prepared=rows,
            )
        boards, inputs = zip(*rows, strict=True)
        return self._prepare_roots(boards, np.stack(inputs))

    def execute(self, prepared: BT4PreparedExecution) -> tuple[np.ndarray, ...]:
        """Session-owner-only stage; provider qualification wraps this boundary."""
        if prepared.owner is not self:
            raise ValueError("BT4 prepared execution belongs to another evaluator")
        return self._execute_feed(prepared.inference)

    def __call__(self, rows: Sequence[Any]) -> tuple[BT4RootOutput, ...]:
        prepared = self.prepare(rows)
        return self.finish(prepared, self.execute(prepared))

    def _prepare_roots(
        self, boards: Sequence[chess.Board], x_batch: np.ndarray,
        *, prepared: tuple[BT4PreparedInput, ...] | None = None,
    ) -> BT4PreparedExecution:
        """One canonical validation/inference boundary for both entry points.

        Rebuild verification CBoards independently from Python full histories,
        never from the persistent actor CBoards. No inference is attempted until
        all rows pass. This deliberately retains the second encoding operation.
        """
        self._require_history_rep_fix()
        arr = self._checked_inputs(x_batch)
        if not boards or len(boards) != len(arr):
            raise ValueError("BT4 root boards and encoded batch must have equal nonzero length")
        legal = (
            [BT4LegalSnapshot(board) for board in boards] if prepared is None
            else [payload.legal for payload in prepared]
        )
        expected = encode_cboard_batch(
            [CBoard.from_board(board) for board in boards],
            input_history_encoding=self.input_history_encoding,
            input_extra_features=self.input_extra_features,
        )
        if expected.shape != arr.shape:
            raise ValueError("BT4 verification encoding batch has the wrong shape")
        for idx, (row, reference) in enumerate(zip(arr, expected)):
            if not np.array_equal(row, reference):
                raise ValueError(f"BT4 root {idx} board/history does not match encoded input")
        source_keys = position_fingerprints(
            arr, input_history_encoding=self.input_history_encoding,
        )
        if len(source_keys) != len(arr) or any(
            len(key) != FINGERPRINT_BYTES for key in source_keys
        ):
            raise ValueError("BT4 root source fingerprint is unavailable")
        fens = [board.fen() for board in boards]
        input_keys = [corpus.input_tensor_key(row) for row in arr]
        if prepared is not None and any(
            (payload.fen, payload.input_key, payload.source_key)
            != (fen, key, source_key)
            for payload, fen, key, source_key in zip(prepared, fens, input_keys, source_keys)
        ):
            raise ValueError("BT4 prepared input identity disagrees with verified input")
        return BT4PreparedExecution(
            self, self._prepare_feed(arr, root=True), tuple(fens), tuple(input_keys),
            tuple(source_keys), tuple(legal),
        )

    def finish(self, prepared: BT4PreparedExecution, fetched: Any) -> tuple[BT4RootOutput, ...]:
        """CPU-only head validation and immutable legal-policy assembly."""
        if prepared.owner is not self:
            raise ValueError("BT4 prepared execution belongs to another evaluator")
        policy_rows, native_values = self._finish_feed(prepared.inference, fetched)
        fens, input_keys, source_keys, legal = (
            prepared.fens, prepared.input_keys, prepared.source_keys, prepared.legal,
        )
        feed_hashes = prepared.inference.feed_hashes
        outputs = []
        for idx, snapshot in enumerate(legal):
            dense = snapshot._dense_policy(fens[idx], policy_rows[idx])
            raw = native_values[idx].copy()
            for value in (dense, raw):
                value.flags.writeable = False
            outputs.append(BT4RootOutput(
                fen=fens[idx], input_key=input_keys[idx], source_key=source_keys[idx],
                policy_t1=dense, wdl_raw=raw,
                policy_output=self.policy_output, wdl_output=self.wdl_contract["output"],
                wdl_kind=self.wdl_contract["kind"], model_sha256=self.model_sha256,
                input_name=self.input_name, input_dtype=self.input_dtype.name,
                input_history_encoding=self.input_history_encoding,
                input_extra_features=self.input_extra_features,
                history_rep_fix=self.history_rep_fix,
                onnx_feed_sha256=feed_hashes[idx],
                verified_session_sha256=self.verified_session_sha256,
            ))
        return tuple(outputs)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a C tree's pending real leaves; return neutral padded rows."""
        self._require_history_rep_fix()
        if relations is not None:
            raise ValueError("BT4 evaluator does not support relation inputs")
        arr = self._checked_inputs(x)
        tree = self._tree
        if tree is None:
            raise RuntimeError("BT4 leaf evaluator is not bound to a C search tree")
        boards = tree.pending_leaf_cboards()
        legal_flat, legal_counts = tree.get_pending_legal_indices()
        n_real = len(boards)
        if (
            n_real == 0 or n_real > len(arr) or len(legal_counts) != n_real
            or int(np.sum(legal_counts)) != len(legal_flat)
        ):
            raise ValueError("BT4 pending leaf count disagrees with encoded batch")
        mapped: list[chess.Board] = []
        offset = 0
        for idx, cb in enumerate(boards):
            expected = encode_cboard(
                cb, input_history_encoding=self.input_history_encoding,
                input_extra_features=self.input_extra_features,
            )
            if not np.array_equal(arr[idx], expected):
                raise ValueError(f"BT4 leaf {idx} board/history does not match encoded input")
            board = chess.Board(cb.fen())  # Legal mapping only; feed keeps CBoard history.
            cb_legal = np.sort(np.asarray(cb.legal_move_indices(), dtype=np.int32))
            py_legal = np.sort(legal_move_indices(board))
            pending_legal = np.sort(np.asarray(
                legal_flat[offset:offset + int(legal_counts[idx])], dtype=np.int32,
            ))
            if not (np.array_equal(cb_legal, py_legal)
                    and np.array_equal(cb_legal, pending_legal)):
                raise ValueError(f"BT4 leaf {idx} C/Python legal indices disagree")
            offset += int(legal_counts[idx])
            mapped.append(board)
        policy_rows, native_values, _ = self._infer(arr[:n_real])
        self.leaf_calls += 1
        self.leaf_rows += n_real
        policy_logits = np.full((len(arr), COMPACT_POLICY_SIZE), _ZERO_LOGIT, dtype=np.float32)
        wdl_logits = np.zeros((len(arr), 3), dtype=np.float32)
        for idx, board in enumerate(mapped):
            _, _, dense = compact_legal_policy(board, policy_rows[idx])
            policy_logits[idx] = _search_policy_logits(dense)
            wdl_logits[idx] = _search_wdl_logits(native_values[idx], self.wdl_contract["kind"])
        return policy_logits, wdl_logits
