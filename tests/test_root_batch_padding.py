"""Root-evaluation row padding: bucket the compiled path, not the broker path.

``_ROOT_BUCKETS`` exists to keep torch.compile / AOT from seeing a new input
shape on every network turn. That only helps an evaluator that owns a graph.
The broker client owns none, and ``SlotBroker`` re-pads the coalesced total
itself, so padding a broker request to 32 rows just ships zero rows across
shared memory and through the forward. Production selfplay is Stockfish-bound
at ~1 runnable game per network turn, which made that floor ~27% of every row
the clients submitted.

Both halves are pinned here: the broker path must send exactly the real rows,
the in-process path must still see an exact bucket.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.inference import MultiSlotInferenceClient, SlotInferenceClient
from chess_anti_engine.selfplay.network_turn import (
    _evaluate_root_batch,
    _padded_batch_size,
    _root_submit_size,
)
from chess_anti_engine.selfplay.state import SelfplayState

_EXTRA_FEATURES = "v2_threats"
_PLANES = input_plane_count(_EXTRA_FEATURES)
_POLICY_WIDTH = 1858


def _outputs(rows: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((rows, _POLICY_WIDTH), dtype=np.float32),
        np.zeros((rows, 3), dtype=np.float32),
    )


class _FakeBrokerClient:
    """Broker-style client: dense ``evaluate_encoded`` only, no pinned slots."""

    pads_batches_internally = True

    def __init__(self) -> None:
        self.submitted_rows: list[int] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert relations is None
        self.submitted_rows.append(int(x.shape[0]))
        return _outputs(int(x.shape[0]))


class _FakeDenseEvaluator:
    """Legacy dense evaluator that never declared the capability."""

    def __init__(self) -> None:
        self.submitted_rows: list[int] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert relations is None
        self.submitted_rows.append(int(x.shape[0]))
        return _outputs(int(x.shape[0]))


class _FakeCompiledEvaluator:
    """In-process evaluator with the pinned slot API (DirectGPU / AOT shape)."""

    def __init__(self) -> None:
        self.buffer_rows: list[int] = []
        self.forward_rows: list[int] = []
        self._buf = np.zeros((512, _PLANES, 8, 8), dtype=np.float32)

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        del slot
        self.buffer_rows.append(int(bsz))
        return self._buf[:bsz]

    def evaluate_inplace(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del copy_out, slot
        assert relations is None
        self.forward_rows.append(int(bsz))
        return _outputs(int(bsz))


def _fill_zeros(cboards: list[Any], buf: np.ndarray) -> None:
    """Stand-in for the C batch encoder; the row COUNT is what's under test."""
    assert len(cboards) <= buf.shape[0]
    buf[...] = 0


def _state(evaluator: object, *, n_boards: int) -> SelfplayState:
    stub = SimpleNamespace(
        evaluator=evaluator,
        cboards=[object() for _ in range(n_boards)],
        game=SimpleNamespace(
            input_history_encoding="legacy",
            input_extra_features=_EXTRA_FEATURES,
            record_relations=False,
        ),
        has_c_ply=True,
        batch_enc_146=_fill_zeros,
        batch_enc_146_bf16=None,
        batch_enc_146_lc0_root=_fill_zeros,
        batch_enc_146_lc0_root_bf16=None,
        batch_enc_146_lc0_root_legacy_meta=_fill_zeros,
        batch_enc_146_lc0_root_legacy_meta_bf16=None,
    )
    # _evaluate_root_batch only touches the attributes set above; SimpleNamespace
    # keeps the fixture to the surface under test instead of a full session.
    return cast("SelfplayState", cast("object", stub))


def test_broker_root_eval_submits_real_rows_only() -> None:
    client = _FakeBrokerClient()
    _, pol, wdl, _ = _evaluate_root_batch(_state(client, n_boards=1), [0])

    assert client.submitted_rows == [1], "broker request was padded to a bucket"
    assert pol.shape[0] == 1
    assert wdl.shape[0] == 1


def test_broker_root_eval_unpadded_across_the_bucket_ladder() -> None:
    for n in (1, 5, 32, 33, 100):
        client = _FakeBrokerClient()
        _evaluate_root_batch(_state(client, n_boards=n), list(range(n)))
        assert client.submitted_rows == [n]


def test_compiled_root_eval_still_gets_an_exact_bucket() -> None:
    for n, bucket in ((1, 32), (32, 32), (33, 64), (129, 256)):
        evaluator = _FakeCompiledEvaluator()
        _, pol, _, _ = _evaluate_root_batch(_state(evaluator, n_boards=n), list(range(n)))
        assert evaluator.buffer_rows == [bucket], f"n={n} lost its bucket"
        assert evaluator.forward_rows == [bucket], f"n={n} forwarded off-bucket"
        assert pol.shape[0] == n, "padded rows leaked into the caller's logits"


def test_dense_evaluator_without_the_capability_keeps_the_bucket_floor() -> None:
    evaluator = _FakeDenseEvaluator()
    _evaluate_root_batch(_state(evaluator, n_boards=3), [0, 1, 2])

    assert evaluator.submitted_rows == [32]


def test_root_submit_size_matrix() -> None:
    broker = _FakeBrokerClient()
    dense = _FakeDenseEvaluator()

    assert _root_submit_size(broker, 1, use_inplace=False) == 1
    assert _root_submit_size(dense, 1, use_inplace=False) == _padded_batch_size(1)
    # The pinned-slot path is bucketed even when the evaluator claims to pad:
    # ``get_input_buffer``/``evaluate_inplace`` hand the size straight to the
    # compiled graph.
    assert _root_submit_size(broker, 1, use_inplace=True) == _padded_batch_size(1)


def test_broker_clients_declare_that_they_pad_internally() -> None:
    single = SlotInferenceClient(slot_name="cae-test-unused", max_batch=64)
    multi = MultiSlotInferenceClient(slot_names=["cae-test-unused"], max_batch=64)
    try:
        assert single.pads_batches_internally is True
        assert multi.pads_batches_internally is True
    finally:
        single.close()
        multi.close()
