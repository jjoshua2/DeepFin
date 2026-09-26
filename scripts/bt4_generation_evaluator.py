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
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding, x_to_lc0_planes
from chess_anti_engine.eval.rvg_surgery import FINGERPRINT_BYTES, position_fingerprints
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, legal_move_indices
from scripts import gen_sf_rooted_corpus as corpus
from scripts.bt4_policy_dump import compact_legal_policy, resolve_policy_output
from scripts.bt4_raw_corpus_sidecar import resolve_wdl_output, validate_wdl_values

_ZERO_LOGIT = np.float32(-1e9)


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

    def search_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Fresh batched logits for search; root-only actors retain no derived copy."""
        return (_search_policy_logits(self.policy_t1)[None],
                _search_wdl_logits(self.wdl_raw, self.wdl_kind)[None])


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
        model_sha256: str,
    ) -> None:
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
        self._tree: Any | None = None
        self.root_calls = 0
        self.root_rows = 0
        self.leaf_calls = 0
        self.leaf_rows = 0

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

    def _infer(self, x: np.ndarray, *, root: bool = False) -> tuple[np.ndarray, np.ndarray]:
        feed = x_to_lc0_planes(
            x, input_history_encoding=self.input_history_encoding,
        ).astype(self.input_dtype, copy=False)
        if root:
            # Count submitted ONNX calls/rows, including a session failure.
            self.root_calls += 1
            self.root_rows += len(x)
        fetched = self.sess.run(
            [self.policy_output, self.wdl_contract["output"]], {self.input_name: feed},
        )
        if len(fetched) != 2:
            raise ValueError("BT4 session returned the wrong number of named outputs")
        policy = np.asarray(fetched[0], dtype=np.float32)
        values = np.asarray(fetched[1])
        if policy.shape != (len(x), COMPACT_POLICY_SIZE):
            raise ValueError(f"BT4 policy shape {policy.shape} is not (N,{COMPACT_POLICY_SIZE})")
        validate_wdl_values(values, len(x), self.wdl_contract)
        return policy, values

    def evaluate_root(self, board: chess.Board, x: np.ndarray) -> BT4RootOutput:
        """Singleton convenience path with the same validation as batched roots."""
        root = np.asarray(x)
        return self.evaluate_roots([board], root[None])[0]

    def evaluate_roots(
        self, boards: Sequence[chess.Board], x_batch: np.ndarray,
    ) -> list[BT4RootOutput]:
        """Validate every root, then retain aligned outputs from one ONNX call.

        History/shape/identity failures reject the entire batch before inference.
        The session is never asked to evaluate padding or an empty root batch.
        """
        arr = self._checked_inputs(x_batch)
        if not boards or len(boards) != len(arr):
            raise ValueError("BT4 root boards and encoded batch must have equal nonzero length")
        for idx, board in enumerate(boards):
            if board.legal_moves.count() == 0:
                raise ValueError(f"BT4 root {idx} has no legal move")
            expected = encode_cboard(
                CBoard.from_board(board),
                input_history_encoding=self.input_history_encoding,
                input_extra_features=self.input_extra_features,
            )
            if not np.array_equal(arr[idx], expected):
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
        policy_rows, native_values = self._infer(arr, root=True)
        outputs = []
        for idx, board in enumerate(boards):
            _, _, dense = compact_legal_policy(board, policy_rows[idx])
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
            ))
        return outputs

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a C tree's pending real leaves; return neutral padded rows."""
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
        policy_rows, native_values = self._infer(arr[:n_real])
        self.leaf_calls += 1
        self.leaf_rows += n_real
        policy_logits = np.full((len(arr), COMPACT_POLICY_SIZE), _ZERO_LOGIT, dtype=np.float32)
        wdl_logits = np.zeros((len(arr), 3), dtype=np.float32)
        for idx, board in enumerate(mapped):
            _, _, dense = compact_legal_policy(board, policy_rows[idx])
            policy_logits[idx] = _search_policy_logits(dense)
            wdl_logits[idx] = _search_wdl_logits(native_values[idx], self.wdl_contract["kind"])
        return policy_logits, wdl_logits
