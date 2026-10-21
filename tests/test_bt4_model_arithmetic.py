"""Actual model-bound validation and both BT4 consumers; synthetic outputs only."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from scripts.bt4_generation_evaluator import BT4OnnxEvaluator
from scripts.bt4_model_arithmetic import validate_model_wdl
from scripts.bt4_raw_corpus_sidecar import validate_wdl_values
from scripts.bt4_root_policy_stepper import BT4RootPolicyStepper

SELECTED = "809db9c3c8063683bf8e7c08de8fec92ea4e37ad8c68c8b1befbde4bd1958961"
ORDINARY = "a" * 64
CONTRACT = {"dtype": "float32", "kind": "probabilities"}


class _Session:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get_outputs(self) -> list[Any]:
        value_type = "tensor(float)" if self.values.dtype == np.float32 else "tensor(double)"
        return [SimpleNamespace(name="policy", shape=["batch", 1858], type="tensor(float)"),
                SimpleNamespace(name="wdl", shape=["batch", 3], type=value_type)]

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "wdl"]
        rows = len(feed["planes"])
        return [np.zeros((rows, 1858), dtype=np.float32),
                np.broadcast_to(self.values, (rows, 3)).copy()]


def _evaluator(values: np.ndarray, model: str = SELECTED) -> BT4OnnxEvaluator:
    return BT4OnnxEvaluator(
        _Session(values), input_name="planes", input_dtype=np.dtype("float32"),
        policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
        input_history_encoding="lc0_root_legacy_meta", input_extra_features="v2_threats",
        model_sha256=model, history_rep_fix=False,
    )


def _stepper(model: str = SELECTED) -> tuple[BT4RootPolicyStepper, np.random.Generator]:
    rng = np.random.default_rng(301)
    stepper = BT4RootPolicyStepper(
        {0: chess.Board()}, {0: rng}, max_plies=8, syzygy_path="unused-above-six-pieces",
        input_history_encoding="lc0_root_legacy_meta", input_extra_features="v2_threats",
        history_rep_fix=False, model_sha256=model, outcome_mode="theoretical_wdl",
    )
    return stepper, rng


@pytest.fixture(autouse=True)
def _expected_rep_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same inert mode seam as the existing stepper tests: never change the
    # process-global native encoder mode in this synthetic arithmetic fixture.
    monkeypatch.setattr(rep_fix, "current", lambda: False)


def test_exact_artifact_evaluator_and_stepper_keep_raw_fp32_bytes() -> None:
    values = np.array([0.5, 0.5, 2**-11], dtype=np.float32)
    with pytest.raises(ValueError, match="probability range/mass"):
        validate_wdl_values(values[None], 1, CONTRACT)
    stepper, _rng = _stepper()
    prepared, finished = stepper.prepare_roots()
    assert prepared is not None
    assert not finished
    boards, encoded = prepared.inference_inputs()
    outputs = _evaluator(values).evaluate_roots(boards, encoded)
    assert outputs[0].wdl_raw.dtype == np.float32
    assert outputs[0].wdl_raw.tobytes() == values.tobytes()
    assert not outputs[0].wdl_raw.flags.writeable
    selected = stepper.apply_root_outputs(prepared, outputs, temperatures={0: 1.0})
    assert len(selected) == 1
    assert outputs[0].wdl_raw.tobytes() == values.tobytes()


def test_ordinary_evaluator_and_stepper_still_reject_same_mass_before_rng() -> None:
    values = np.array([0.5, 0.5, 2**-11], dtype=np.float32)
    stepper, rng = _stepper(ORDINARY)
    prepared, finished = stepper.prepare_roots()
    assert prepared is not None
    assert not finished
    boards, encoded = prepared.inference_inputs()
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="probability range/mass"):
        _evaluator(values, ORDINARY).evaluate_roots(boards, encoded)
    # A valid selected-artifact result cannot transfer its arithmetic exemption
    # to another model when the actual stepper validates the received output.
    output = replace(_evaluator(values).evaluate_roots(boards, encoded)[0], model_sha256=ORDINARY)
    with pytest.raises(ValueError, match="probability range/mass"):
        stepper.apply_root_outputs(prepared, [output], temperatures={0: 1.0})
    assert rng.bit_generator.state == before


def test_exact_artifact_consumers_refuse_changed_storage() -> None:
    stepper, rng = _stepper()
    prepared, finished = stepper.prepare_roots()
    assert prepared is not None
    assert not finished
    boards, encoded = prepared.inference_inputs()
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="FP32 probability storage"):
        _evaluator(np.array([0.5, 0.5, 0], dtype=np.float64)).evaluate_roots(boards, encoded)
    output = _evaluator(np.array([0.5, 0.5, 0], dtype=np.float32)).evaluate_roots(boards, encoded)[0]
    changed = output.wdl_raw.astype(np.float64)
    changed.flags.writeable = False
    with pytest.raises(ValueError, match="FP32 probability storage"):
        stepper.apply_root_outputs(prepared, [replace(output, wdl_raw=changed)], temperatures={0: 1.0})
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("error", [2**-10, -(2**-10)])
def test_existing_closed_mass_boundary_is_inclusive_and_unchanged(error: float) -> None:
    values = np.array([[0.5, 0.5 + error, 0]], dtype=np.float32)
    original = values.tobytes()
    validate_model_wdl(values, 1, CONTRACT, SELECTED)
    assert values.tobytes() == original


def test_next_representable_value_outside_existing_budget_refused() -> None:
    error = np.nextafter(np.float32(2**-10), np.float32(np.inf))
    values = np.array([[0.5, 0.5, error]], dtype=np.float32)
    with pytest.raises(ValueError, match="probability mass"):
        validate_model_wdl(values, 1, CONTRACT, SELECTED)


def test_selected_untyped_adapter_must_return_an_actual_array() -> None:
    with pytest.raises(ValueError, match="shape/storage"):
        validate_model_wdl([[0.5, 0.5, 0.0]], 1, CONTRACT, SELECTED)


@pytest.mark.parametrize(("values", "message"), [
    (np.array([[0.5, 0.5]], dtype=np.float32), "shape/storage"),
    (np.array([[0.5, 0.5, 0]], dtype=np.float64), "shape/storage"),
    (np.array([[0.5, 0.5, 0]], dtype=np.float16), "shape/storage"),
    (np.array([[np.nan, 0.5, 0.5]], dtype=np.float32), "finite range"),
    (np.array([[np.inf, 0, 0]], dtype=np.float32), "finite range"),
    (np.array([[-1e-8, 0.5, 0.5]], dtype=np.float32), "finite range"),
    (np.array([[1 + 1e-6, 0, 0]], dtype=np.float32), "finite range"),
    (np.array([[0.5, 0.5, 2**-9]], dtype=np.float32), "probability mass"),
])
def test_selected_shape_storage_finite_range_and_mass_negatives(values: np.ndarray, message: str) -> None:
    original = values.tobytes()
    with pytest.raises(ValueError, match=message):
        validate_model_wdl(values, 1, CONTRACT, SELECTED)
    assert values.tobytes() == original


@pytest.mark.parametrize("contract", [
    {"dtype": "float32", "kind": "logits"},
    {"dtype": "float16", "kind": "probabilities"},
])
def test_selected_contract_cannot_claim_other_precision_or_head(contract: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="FP32 probability storage"):
        validate_model_wdl(np.array([[0.5, 0.5, 0]], dtype=np.float32), 1, contract, SELECTED)


@pytest.mark.parametrize("model", [ORDINARY, SELECTED[:-1] + "0"])
def test_wrong_model_hash_has_no_fp32_arithmetic_exemption(model: str) -> None:
    with pytest.raises(ValueError, match="probability range/mass"):
        validate_model_wdl(np.array([[0.5, 0.5, 2**-11]], dtype=np.float32), 1, CONTRACT, model)


def test_ordinary_native_fp16_and_logits_contracts_are_unchanged() -> None:
    values = np.array([[0.5, 0.5, 2**-11]], dtype=np.float16)
    validate_model_wdl(values, 1, {"dtype": "float16", "kind": "probabilities"}, ORDINARY)
    logits = np.array([[100, -200, 30]], dtype=np.float32)
    validate_model_wdl(logits, 1, {"dtype": "float32", "kind": "logits"}, ORDINARY)
