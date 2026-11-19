"""Fictional same-owner GPU proof metadata for the exact thread-2 prover.

The old seven-b fingerprint cases keep their default fixture. These controls
exercise the new public metadata consumer, never a model, payload or GPU job.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import core_generation_holdout_ruler as ruler
from tests.test_core_same_gpu_payload_delivery import gpu_graph


@pytest.mark.parametrize("arm", ["A", "C"])
def test_thread2_prover_delivery_uses_exact_versioned_fingerprint(tmp_path: Path, arm: str):
    binding, train = gpu_graph(tmp_path, arm=arm, threaded=True)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    model = ruler.model_delivery(metadata, binding, arm)
    ruler.same_gpu_game_coverage(metadata, binding, model, train)
    assert model["step"] == 250
    assert model["tag"]["source_revision"] == "b" * 40
    metadata.finish()


@pytest.mark.parametrize("fault", ["thread_helper_unknown", "thread_missing_intra", "thread_missing_interop", "thread_intra",
    "thread_interop", "thread_bool_intra", "thread_float_interop", "thread_backend_intra", "thread_backend_interop",
    "thread_review_helper", "thread_review_findings", "thread_review_pins"])
def test_rehashed_thread2_prover_metadata_refusals(tmp_path: Path, fault: str):
    binding, _ = gpu_graph(tmp_path, fault, threaded=True)
    if fault == "thread_helper_unknown":
        expected = "unsupported same-GPU prover"
    elif fault in {"thread_review_helper", "thread_review_findings", "thread_review_pins"}:
        expected = "same-GPU prover source acceptance differs"
    else:
        expected = "same-GPU thread-2 evidence differs"
    with pytest.raises(ValueError, match=expected):
        ruler.model_delivery(ruler.Metadata(1 << 20, 8 << 20), binding, "A")
