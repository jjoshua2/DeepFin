"""CPU fake regressions for mapped findings 3 (capacity) and 5 (reload ownership)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import chess_anti_engine.inference as inf


class Package:
    def __init__(self, bucket):
        self.bucket = bucket
        self.calls = 0
        self.weight = 1.0
        self.fail_load = False
        self.entered = None
        self.release = None

    def get_constant_fqns(self):
        return ["weight"]

    def load_constants(self, constants, **kwargs):
        if self.fail_load:
            raise RuntimeError("injected rebind failure")
        self.weight = float(constants["weight"][0])

    def __call__(self, x):
        self.calls += 1
        assert x.shape[0] == self.bucket
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(2)
        return {
            "policy": torch.full((self.bucket, inf.POLICY_SIZE), self.weight),
            "wdl": torch.full((self.bucket, 3), self.weight),
        }


def evaluator(monkeypatch, tmp_path, buckets=(16,), max_batch=512):
    packages = {b: Package(b) for b in buckets}
    monkeypatch.setattr(inf, "load_aot_packages", lambda *args, **kwargs: packages)
    ev = inf.AOTEvaluator(tmp_path, device="cpu", max_batch=max_batch, input_planes=1)
    return ev, packages


@pytest.mark.parametrize("api", ["sync", "async", "inplace", "buffer", "bf16_buffer"])
def test_missing_largest_package_rejects_before_stage_or_launch(monkeypatch, tmp_path, api):
    ev, packages = evaluator(monkeypatch, tmp_path)
    x = np.zeros((17, 1, 8, 8), dtype=np.float32)
    ev._pinned_input.fill_(42)
    with pytest.raises(ValueError, match="17"):
        if api == "sync":
            ev.evaluate_encoded(x)
        elif api == "async":
            ev.evaluate_encoded_async(x)
        elif api == "inplace":
            ev.evaluate_inplace_async(17)
        elif api == "buffer":
            ev.get_input_buffer(17)
        else:
            ev.get_input_buffer_bf16_bits(17)
    assert packages[16].calls == 0
    assert torch.all(ev._pinned_input == 42)
    assert ev._max_batch == 16


@pytest.mark.parametrize("bsz", [1, 16])
@pytest.mark.parametrize("api", ["sync", "async", "inplace"])
def test_capacity_boundary_preserves_exact_output_rows(monkeypatch, tmp_path, bsz, api):
    ev, packages = evaluator(monkeypatch, tmp_path)
    x = np.zeros((bsz, 1, 8, 8), dtype=np.float32)
    if api == "sync":
        policy, wdl = ev.evaluate_encoded(x)
    elif api == "async":
        policy, wdl, receipt = ev.evaluate_encoded_async(x)
        assert receipt is None  # CPU fake completes synchronously.
    else:
        ev.get_input_buffer(bsz).fill(0)
        policy, wdl, receipt = ev.evaluate_inplace_async(bsz)
        assert receipt is None
    assert policy.shape == (bsz, inf.POLICY_SIZE)
    assert wdl.shape == (bsz, 3)
    assert packages[16].calls == 1


def test_missing_middle_bucket_uses_adequate_larger_package(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path, buckets=(16, 64))
    policy, wdl = ev.evaluate_encoded(np.zeros((17, 1, 8, 8), dtype=np.float32))
    assert policy.shape[0] == wdl.shape[0] == 17
    assert packages[16].calls == 0
    assert packages[64].calls == 1


def test_reload_waits_for_local_forward_then_resumes(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path)
    package = packages[16]
    package.entered, package.release = threading.Event(), threading.Event()
    reload_started = threading.Event()
    x = np.zeros((1, 1, 8, 8), dtype=np.float32)

    def reload():
        reload_started.set()
        ev.load_weights({"weight": torch.tensor([2.0])})

    with ThreadPoolExecutor(max_workers=2) as pool:
        forward = pool.submit(ev.evaluate_encoded, x)
        assert package.entered.wait(2)
        update = pool.submit(reload)
        assert reload_started.wait(2)
        assert package.weight == 1.0
        assert not update.done()
        package.release.set()
        old_policy, _ = forward.result(timeout=2)
        update.result(timeout=2)
    assert np.all(old_policy == 1.0)
    new_policy, _ = ev.evaluate_encoded(x)
    assert np.all(new_policy == 2.0)


class Event:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def record(self, stream):
        pass

    def synchronize(self):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(2)


class Stream:
    def synchronize(self):
        pass


@pytest.mark.parametrize("api", ["async", "inplace"])
def test_reload_drains_submitted_receipt_and_excludes_new_launches(
    monkeypatch, tmp_path, api,
):
    ev, packages = evaluator(monkeypatch, tmp_path)
    stream, event = Stream(), Event()
    # Emulate the GPU enqueue boundary using CPU outputs and controlled events.
    ev.device = "cuda"
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(ev, "_device_input", lambda *args, **kwargs: torch.zeros((16, 1, 8, 8)))
    monkeypatch.setattr(ev, "_device_input_inplace", lambda **kwargs: torch.zeros((16, 1, 8, 8)))
    monkeypatch.setattr(inf, "build_aot_constants", lambda source, *args, **kwargs: source)
    x = np.zeros((1, 1, 8, 8), dtype=np.float32)
    if api == "async":
        _, _, receipt = ev.evaluate_encoded_async(x)
    else:
        _, _, receipt = ev.evaluate_inplace_async(1)
    assert receipt is not None
    assert len(ev._pending) == 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        update = pool.submit(ev.load_weights, {"weight": torch.tensor([2.0])})
        assert event.entered.wait(2)
        later = pool.submit(ev.evaluate_encoded, x)
        assert packages[16].calls == 1
        assert packages[16].weight == 1.0
        assert not later.done()
        event.release.set()
        update.result(timeout=2)
        policy, _ = later.result(timeout=2)
    assert np.all(policy == 2.0)
    assert not ev._pending
    receipt.synchronize()  # Caller can still consume a drained receipt.
    assert event.calls == 2


def test_partial_rebind_poison_rejects_every_submission_and_retry(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path, buckets=(16, 32))
    packages[32].fail_load = True
    with pytest.raises(RuntimeError, match="AOT bucket 32 rejected"):
        ev.load_weights({"weight": torch.tensor([2.0])})
    assert packages[16].weight == 2.0
    assert packages[32].weight == 1.0
    x = np.zeros((1, 1, 8, 8), dtype=np.float32)
    for submit in (
        lambda: ev.evaluate_encoded(x),
        lambda: ev.evaluate_encoded_async(x),
        lambda: ev.evaluate_inplace_async(1),
        lambda: ev.get_input_buffer(1),
        lambda: ev.load_weights({"weight": torch.tensor([3.0])}),
    ):
        with pytest.raises(RuntimeError, match="poisoned"):
            submit()
    assert all(package.calls == 0 for package in packages.values())


def test_new_submission_waits_until_every_bucket_has_new_weights(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path, buckets=(16, 32))
    entered, release = threading.Event(), threading.Event()
    original = packages[32].load_constants

    def delayed_load(constants, **kwargs):
        entered.set()
        assert release.wait(2)
        original(constants, **kwargs)

    monkeypatch.setattr(packages[32], "load_constants", delayed_load)
    x = np.zeros((1, 1, 8, 8), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2) as pool:
        update = pool.submit(ev.load_weights, {"weight": torch.tensor([2.0])})
        assert entered.wait(2)
        assert packages[16].weight == 2.0
        assert packages[32].weight == 1.0
        forward = pool.submit(ev.evaluate_encoded, x)
        assert not forward.done()
        assert packages[16].calls == 0
        release.set()
        update.result(timeout=2)
        policy, _ = forward.result(timeout=2)
    assert np.all(policy == 2.0)
    assert packages[32].weight == 2.0
