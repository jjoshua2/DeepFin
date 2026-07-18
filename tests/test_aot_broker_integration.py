"""Non-GPU unit tests for SlotBroker AOT package integration helpers + wiring."""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import (
    AOTEvaluator,
    SlotBroker,
    _COMPILED_BATCH_BUCKETS,
    _aoti_load_package,
    aot_package_filename,
    build_aot_constants,
    load_aot_packages,
    model_constant_source,
    select_compiled_aot_buckets,
    should_use_aot_forward,
)
from chess_anti_engine.tune.distributed_runtime import _launch_inference_broker


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_aot_package_filename() -> None:
    assert aot_package_filename(128) == "chess_b128.pt2"
    assert aot_package_filename(1190) == "chess_b1190.pt2"


def test_select_compiled_aot_buckets_filters_by_max_batch() -> None:
    sel = select_compiled_aot_buckets(max_batch=256)
    assert sel == tuple(b for b in _COMPILED_BATCH_BUCKETS if b <= 256)
    assert 128 in sel
    assert 256 in sel
    assert 340 not in sel
    assert all(b in _COMPILED_BATCH_BUCKETS for b in sel)


def test_select_compiled_aot_buckets_full_ladder() -> None:
    sel = select_compiled_aot_buckets(max_batch=4096)
    assert sel == _COMPILED_BATCH_BUCKETS


def test_select_compiled_aot_buckets_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="max_batch must be positive"):
        select_compiled_aot_buckets(max_batch=0)


def test_should_use_aot_forward_routing() -> None:
    models: dict[int, Any] = {128: object(), 256: object()}
    assert should_use_aot_forward(models, 128) is True
    assert should_use_aot_forward(models, 256) is True
    # Exact key only — do not re-pick finer buckets.
    assert should_use_aot_forward(models, 200) is False
    assert should_use_aot_forward(models, 170) is False
    # Off / empty -> eager.
    assert should_use_aot_forward(None, 128) is False
    assert should_use_aot_forward({}, 128) is False


def test_build_aot_constants_from_state_dict() -> None:
    sd = {
        "trunk.w": torch.ones(2, 2),
        "head.b": torch.zeros(3),
        "unused.x": torch.tensor([1.0]),
    }
    constants = build_aot_constants(
        sd, ["trunk.w", "head.b"], device="cpu",
    )
    assert set(constants) == {"trunk.w", "head.b"}
    assert constants["trunk.w"].dtype == torch.bfloat16
    assert constants["trunk.w"].is_contiguous()
    assert "unused.x" not in constants


def test_build_aot_constants_fails_loud_on_missing_fqns() -> None:
    sd = {"trunk.w": torch.ones(2, 2)}
    with pytest.raises(KeyError, match=r"missing .* AOT constant"):
        build_aot_constants(sd, ["trunk.w", "missing.fqn"], device="cpu")


def test_build_aot_constants_preserves_non_float_dtypes() -> None:
    # Integer/bool lookup buffers (e.g. policy compact_to_full/to_valid) MUST
    # keep their dtype — casting an int index buffer to bf16 corrupts it and
    # makes load_constants fail with CUDA "invalid argument".
    sd = {
        "w": torch.ones(2, 2, dtype=torch.float32),
        "idx": torch.arange(4, dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
    }
    constants = build_aot_constants(sd, ["w", "idx", "mask"], device="cpu")
    assert constants["w"].dtype == torch.bfloat16      # float -> bf16
    assert constants["idx"].dtype == torch.int64       # int preserved
    assert constants["mask"].dtype == torch.bool        # bool preserved
    assert torch.equal(constants["idx"], torch.arange(4, dtype=torch.int64))


def test_model_constant_source_includes_non_persistent_buffers() -> None:
    # The packages externalize every constant, including non-persistent buffers
    # that state_dict() omits; model_constant_source must supply them or the
    # first forward reads a null device pointer -> CUDA illegal memory access.
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.register_buffer("persistent_buf", torch.ones(3))
            self.register_buffer("np_buf", torch.arange(4), persistent=False)

    m = M()
    src = model_constant_source(m)
    assert "np_buf" in src  # absent from state_dict()
    assert "np_buf" not in m.state_dict()
    assert "persistent_buf" in src
    assert "lin.weight" in src
    assert "lin.bias" in src


def test_load_aot_packages_skips_missing_and_loads_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "chess_b128.pt2").write_bytes(b"pkg128")
    (tmp_path / "chess_b256.pt2").write_bytes(b"pkg256")
    # 170 deliberately absent
    loaded_paths: list[str] = []

    def _fake_load(path: str) -> Any:
        loaded_paths.append(path)
        m = MagicMock()
        m.get_constant_fqns.return_value = ["w"]
        return m

    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package", _fake_load,
    )
    models = load_aot_packages(tmp_path, buckets=(128, 170, 256))
    assert set(models.keys()) == {128, 256}
    assert all(path.endswith((".pt2",)) for path in loaded_paths)
    assert len(loaded_paths) == 2


def test_aoti_load_package_primes_pytorch_codecache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_import(name: str) -> object:
        calls.append(("import", name))
        return object()

    def _fake_load(path: str) -> object:
        calls.append(("load", path))
        return object()

    # Resolve torch's lazy submodule before replacing the process-global
    # importlib function used by the helper under test.
    inductor = torch._inductor
    monkeypatch.setattr("chess_anti_engine.inference.importlib.import_module", _fake_import)
    monkeypatch.setattr(inductor, "aoti_load_package", _fake_load)

    _aoti_load_package("model.pt2")

    assert calls == [
        ("import", "torch._inductor.codecache"),
        ("load", "model.pt2"),
    ]


def test_load_aot_packages_fails_when_none_present(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No \.pt2 packages"):
        load_aot_packages(tmp_path, buckets=(128, 256))


def test_aot_evaluator_numpy_input_view_aliases_staging_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "chess_b16.pt2").write_bytes(b"pkg")
    package = MagicMock()
    package.get_constant_fqns.return_value = ["w"]
    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package", lambda _path: package,
    )

    evaluator = AOTEvaluator(tmp_path, device="cpu", max_batch=16, input_planes=3)

    assert evaluator._pinned_input.dtype == torch.float32
    assert np.shares_memory(evaluator._pinned_input_np, evaluator._pinned_input.numpy())
    evaluator._pinned_input_np.fill(2.5)
    assert torch.all(evaluator._pinned_input == 2.5)


def test_aot_evaluator_accepts_bit_packed_bf16_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "chess_b16.pt2").write_bytes(b"pkg")
    package = MagicMock()
    package.get_constant_fqns.return_value = ["w"]
    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package", lambda _path: package,
    )
    evaluator = AOTEvaluator(tmp_path, device="cpu", max_batch=16, input_planes=3)
    source = torch.linspace(-2.0, 2.0, 16 * 3 * 8 * 8).reshape(16, 3, 8, 8)
    expected = source.to(torch.bfloat16)
    bits = expected.view(torch.uint16).numpy().copy()

    staged = evaluator._device_input(bits, bucket=16)

    assert evaluator.supports_input_bf16_bits
    assert staged.dtype == torch.bfloat16
    assert torch.equal(staged, expected)
    assert np.array_equal(evaluator._pinned_input_bf16_bits_np, bits)


def test_aot_evaluator_input_slots_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "chess_b16.pt2").write_bytes(b"pkg")
    package = MagicMock()
    package.get_constant_fqns.return_value = ["w"]
    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package", lambda _path: package,
    )
    evaluator = AOTEvaluator(tmp_path, device="cpu", max_batch=16, input_planes=3)

    bits0 = evaluator.get_input_buffer_bf16_bits(4, slot=0)
    bits1 = evaluator.get_input_buffer_bf16_bits(4, slot=1)
    bits0.fill(0x3F80)
    bits1.fill(0x4000)
    f32_0 = evaluator.get_input_buffer(4, slot=0)
    f32_1 = evaluator.get_input_buffer(4, slot=1)
    f32_0.fill(1.0)
    f32_1.fill(2.0)

    assert evaluator.n_slots == 2
    assert not np.shares_memory(bits0, bits1)
    assert not np.shares_memory(f32_0, f32_1)
    assert np.all(bits0 == 0x3F80)
    assert np.all(bits1 == 0x4000)
    assert np.all(f32_0 == 1.0)
    assert np.all(f32_1 == 2.0)


# ---------------------------------------------------------------------------
# SlotBroker construction (OFF path + mocked ON path)
# ---------------------------------------------------------------------------


def _make_broker(
    tmp_path: Path,
    *,
    aot_dir: str | None = None,
    max_batch_per_slot: int = 8,
    num_slots: int = 1,
) -> SlotBroker:
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    return SlotBroker(
        publish_dir=publish_dir,
        num_slots=num_slots,
        max_batch_per_slot=max_batch_per_slot,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-aot-test-{uuid.uuid4().hex}",
        aot_dir=aot_dir,
    )


def test_slot_broker_aot_off_has_no_packages(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path, aot_dir=None)
    try:
        assert broker._aot_models is None
        assert broker._aot_constant_fqns == []
        assert should_use_aot_forward(broker._aot_models, 128) is False
    finally:
        broker.shutdown()


def test_slot_broker_aot_empty_string_is_off(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path, aot_dir="  ")
    try:
        assert broker._aot_models is None
    finally:
        broker.shutdown()


def test_slot_broker_aot_dir_loads_compiled_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    aot_dir = tmp_path / "aot"
    aot_dir.mkdir()
    # Capacity = 1 slot * 256 = 256 -> packages with b <= 256 from compiled ladder.
    for b in (128, 170, 256):
        (aot_dir / aot_package_filename(b)).write_bytes(b"x")
    # Not on compiled ladder — must not be selected by SlotBroker even if present.
    (aot_dir / "chess_b6.pt2").write_bytes(b"x")

    loaded_buckets: list[int] = []

    def _fake_load(path: str) -> Any:
        name = Path(path).name  # chess_b128.pt2
        bucket = int(name.removeprefix("chess_b").removesuffix(".pt2"))
        loaded_buckets.append(bucket)
        m = MagicMock()
        m.get_constant_fqns.return_value = ["layer.weight"]
        return m

    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package", _fake_load,
    )
    broker = _make_broker(
        tmp_path, aot_dir=str(aot_dir), max_batch_per_slot=256, num_slots=1,
    )
    try:
        assert broker._aot_models is not None
        assert set(broker._aot_models.keys()) == {128, 170, 256}
        assert 6 not in broker._aot_models
        assert broker._aot_constant_fqns == ["layer.weight"]
        assert should_use_aot_forward(broker._aot_models, 128) is True
        assert should_use_aot_forward(broker._aot_models, 200) is False
        assert set(loaded_buckets) == {128, 170, 256}
    finally:
        broker.shutdown()


def test_slot_broker_aot_dir_missing_packages_fails_loud(tmp_path: Path) -> None:
    empty = tmp_path / "empty_aot"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No \.pt2 packages"):
        _make_broker(tmp_path, aot_dir=str(empty), max_batch_per_slot=256)


def test_slot_broker_rebind_uses_build_aot_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constants construction + load_constants contract used by _ensure_model."""
    aot_dir = tmp_path / "aot"
    aot_dir.mkdir()
    (aot_dir / "chess_b128.pt2").write_bytes(b"x")

    aot_pkg = MagicMock()
    aot_pkg.get_constant_fqns.return_value = ["w"]

    monkeypatch.setattr(
        "chess_anti_engine.inference._aoti_load_package",
        lambda _path: aot_pkg,
    )
    broker = _make_broker(
        tmp_path, aot_dir=str(aot_dir), max_batch_per_slot=128, num_slots=1,
    )
    try:
        assert broker._aot_models is not None
        assert broker._aot_constant_fqns == ["w"]
        sd = {"w": torch.ones(2, 2), "other": torch.zeros(1)}
        constants = build_aot_constants(sd, broker._aot_constant_fqns, device="cpu")
        for model in broker._aot_models.values():
            model.load_constants(constants, check_full_update=False)
        aot_pkg.load_constants.assert_called_once_with(
            constants, check_full_update=False,
        )
        assert set(constants) == {"w"}
        # Missing expected fqn must fail loud (never silent no-op rebind).
        with pytest.raises(KeyError, match="missing"):
            build_aot_constants({"other": torch.zeros(1)}, broker._aot_constant_fqns, device="cpu")
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# CLI + distributed_runtime wiring
# ---------------------------------------------------------------------------


def _run_main_capture_broker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> dict[str, Any]:
    import chess_anti_engine.inference as inf

    captured: dict[str, Any] = {}

    class _FakeBroker:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.slot_names = ["slot-0"]

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(inf, "SlotBroker", _FakeBroker)
    monkeypatch.setattr("sys.argv", argv)
    assert inf.main() == 0
    return captured


def test_cli_aot_dir_default_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Per-trial argparse: --aot-dir defaults to None (flag absent)."""
    captured = _run_main_capture_broker_kwargs(
        monkeypatch,
        [
            "inference",
            "--publish-dir", str(tmp_path),
            "--slot-prefix", "t",
            "--num-slots", "1",
            "--max-batch-per-slot", "8",
            "--device", "cpu",
        ],
    )
    assert captured.get("aot_dir") is None


def test_cli_aot_dir_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    aot = str(tmp_path / "aot_pkgs")
    captured = _run_main_capture_broker_kwargs(
        monkeypatch,
        [
            "inference",
            "--publish-dir", str(tmp_path),
            "--slot-prefix", "t",
            "--num-slots", "1",
            "--max-batch-per-slot", "8",
            "--device", "cpu",
            "--aot-dir", aot,
        ],
    )
    assert captured.get("aot_dir") == aot


def test_launch_inference_broker_omits_aot_dir_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    class DummyProc:
        def poll(self) -> int | None:
            return None

    def _fake_popen(cmd: list[str], **_kwargs: Any) -> DummyProc:
        calls.append(list(cmd))
        return DummyProc()

    monkeypatch.setattr(
        "chess_anti_engine.tune.distributed_runtime.terminate_matching_processes",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "chess_anti_engine.tune.distributed_runtime.subprocess.Popen",
        _fake_popen,
    )

    publish_dir = tmp_path / "publish"
    trial_dir = tmp_path / "trial"
    publish_dir.mkdir()
    trial_dir.mkdir()

    _launch_inference_broker(
        config={
            "distributed_workers_per_trial": 2,
            "distributed_worker_device": "cuda",
            "distributed_server_root": str(tmp_path / "server"),
            "distributed_inference_aot_dir": "",
        },
        trial_id="trial_00000",
        publish_dir=publish_dir,
        trial_dir=trial_dir,
    )

    assert calls
    assert "--aot-dir" not in calls[0]


def test_launch_inference_broker_passes_aot_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    class DummyProc:
        def poll(self) -> int | None:
            return None

    def _fake_popen(cmd: list[str], **_kwargs: Any) -> DummyProc:
        calls.append(list(cmd))
        return DummyProc()

    monkeypatch.setattr(
        "chess_anti_engine.tune.distributed_runtime.terminate_matching_processes",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "chess_anti_engine.tune.distributed_runtime.subprocess.Popen",
        _fake_popen,
    )

    publish_dir = tmp_path / "publish"
    trial_dir = tmp_path / "trial"
    publish_dir.mkdir()
    trial_dir.mkdir()
    aot = str(tmp_path / "aot_models")

    _launch_inference_broker(
        config={
            "distributed_workers_per_trial": 2,
            "distributed_worker_device": "cuda",
            "distributed_server_root": str(tmp_path / "server"),
            "distributed_inference_aot_dir": aot,
        },
        trial_id="trial_00000",
        publish_dir=publish_dir,
        trial_dir=trial_dir,
    )

    assert calls
    cmd = calls[0]
    assert "--aot-dir" in cmd
    assert cmd[cmd.index("--aot-dir") + 1] == aot


def test_should_use_aot_forward_matches_process_batch_intent() -> None:
    """Document the routing table used by _process_batch_mode."""
    # OFF path: identical to pre-integration behaviour.
    assert should_use_aot_forward(None, 1024) is False
    # ON + covered exact bucket -> AOT dense branch.
    models = {b: SimpleNamespace() for b in (128, 256, 512)}
    assert should_use_aot_forward(models, 256) is True
    # ON + capacity-guard unbucketed total -> eager fallback (never crash).
    assert should_use_aot_forward(models, 300) is False
    assert should_use_aot_forward(models, 1190) is False
