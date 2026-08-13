"""The rulers' ``--onnx`` input path (scripts/net_source.py).

The defect this file is aimed at is NOT "does the code run" — it is a flag that
is accepted and then silently ignored. So the tests assert, in order:

* exactly one net source, with both/neither refused;
* the ONNX graph's tensor names are read (and ambiguity refused) rather than
  guessed;
* the ``--onnx`` path never falls back to a checkpoint;
* ``--gpu-mem-fraction`` REACHES onnxruntime — asserted on the ``providers=``
  argument ``InferenceSession`` was actually called with, not on the tuple our
  own helper built one line earlier — and the log line names the allocator it
  bounded rather than claiming "GPU memory capped";
* the CUDA guard reads ``session.get_providers()`` (what initialised) and not
  ``onnxruntime.get_available_providers()`` (what the wheel was compiled with,
  which on this box lists CUDA on a runtime whose every session is CPU);
* the policy the ruler ends up reading came through the BOARD-AWARE Leela
  remap, proven per legal move against the independent reference in
  ``moves/leela_index.py`` — a static table gives a different, non-crashing
  answer here, which is exactly how the O-O prior was once read 49-120x too
  small;
* ``scripts/value_regret.py --onnx`` scores that net end to end from its real
  ``main()``, on a frozen-audit-set file, with the checkpoint loader booby
  trapped.

The ONNX fixture is a tiny hand-built graph, not BT4: its policy output is
``[0, 1, ..., 1857]`` for every row, so "which Leela slot did the ruler read"
is directly observable in the returned logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding.model_inputs import encode_position_for_model
from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from chess_anti_engine.moves.leela_index import compact_index_for_move, leela_index_for_move
from scripts.net_source import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    NetSource,
    OnnxNetSpec,
    apply_gpu_mem_cap,
    gpu_mem_limit_bytes,
    net_source_from_args,
    onnx_providers_for_device,
    reject_stored_encoding_for_onnx,
    resolve_onnx_spec,
    validate_onnx_device,
    verify_onnx_session_device,
)

LC0_PLANES = 112


class _IndexEchoNet(torch.nn.Module):
    """policy[b, i] = i; wdl = softmax of a position-dependent logit triple.

    The policy makes the Leela->ours permutation readable off the output. The
    WDL is deliberately emitted as PROBABILITIES, the way LC0/Ceres value heads
    do, so the adapter's probs->log-probs branch is the one under test.
    """

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = planes.sum(dim=(1, 2, 3), keepdim=False)
        zero = rows * 0.0
        pol = torch.arange(COMPACT_POLICY_SIZE, dtype=torch.float32) + zero[:, None]
        wdl_logits = torch.stack([rows * 0.01, zero, -rows * 0.01], dim=-1)
        return pol, torch.softmax(wdl_logits, dim=-1)


class _TwoPolicyHeadNet(torch.nn.Module):
    """Two 1858-wide outputs: the ambiguity a resolver must refuse to guess at."""

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pol, wdl = _IndexEchoNet().forward(planes)
        return pol, pol * 2.0, wdl


class _NoPolicyNet(torch.nn.Module):
    """A graph with no 1858-wide head at all."""

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = planes.sum(dim=(1, 2, 3))
        return rows[:, None].expand(-1, 64), torch.softmax(
            torch.stack([rows, rows * 0.0, -rows], dim=-1), dim=-1,
        )


def _export(net: torch.nn.Module, path: Path, output_names: list[str]) -> Path:
    torch.onnx.export(
        net,
        (torch.zeros(2, LC0_PLANES, 8, 8),),
        str(path),
        input_names=["planes"],
        output_names=output_names,
        dynamic_axes={n: {0: "batch"} for n in ["planes", *output_names]},
        opset_version=17,
        dynamo=False,
    )
    return path


@pytest.fixture(scope="module")
def echo_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _export(
        _IndexEchoNet(), tmp_path_factory.mktemp("onnx") / "echo.onnx", ["policy", "wdl"],
    )


# --------------------------------------------------------------------------
# exactly one source, never a silent default
# --------------------------------------------------------------------------


def test_net_source_refuses_both_sources(echo_onnx: Path) -> None:
    spec = resolve_onnx_spec(echo_onnx)
    with pytest.raises(SystemExit, match="exactly one"):
        NetSource(checkpoint="ckpt", onnx=spec)


def test_net_source_refuses_neither_source() -> None:
    with pytest.raises(SystemExit, match="neither was given"):
        NetSource()


def test_args_with_both_flags_are_refused_before_the_graph_is_opened() -> None:
    args = _args(checkpoint="ckpt", onnx=Path("/nonexistent/never-opened.onnx"))
    with pytest.raises(SystemExit, match="exactly one"):
        net_source_from_args(args)


def test_a_name_override_without_onnx_is_refused_not_ignored() -> None:
    """The signature defect, in miniature: a value accepted and dropped."""
    args = _args(checkpoint="ckpt", onnx_policy_output="/output/policy")
    with pytest.raises(SystemExit, match="--onnx-policy-output given without --onnx"):
        net_source_from_args(args)


def test_a_missing_onnx_file_raises_rather_than_falling_back() -> None:
    with pytest.raises(SystemExit, match="no such file"):
        resolve_onnx_spec("/nonexistent/not-a-net.onnx")


def _args(**kw: object) -> argparse.Namespace:
    """A parsed-CLI stand-in carrying exactly the flags `add_net_source_args` adds."""
    defaults: dict[str, object] = {
        "checkpoint": None,
        "onnx": None,
        "onnx_input_name": None,
        "onnx_policy_output": None,
        "onnx_wdl_output": None,
    }
    return argparse.Namespace(**{**defaults, **kw})


# --------------------------------------------------------------------------
# graph resolution: read, don't guess
# --------------------------------------------------------------------------


def test_resolve_reads_the_names_off_the_graph(echo_onnx: Path) -> None:
    spec = resolve_onnx_spec(echo_onnx)
    assert (spec.input_name, spec.policy_output, spec.wdl_output) == (
        "planes", "policy", "wdl",
    )
    assert str(echo_onnx) in spec.label


def test_two_policy_heads_raise_instead_of_picking_one(tmp_path: Path) -> None:
    path = _export(
        _TwoPolicyHeadNet(), tmp_path / "two.onnx", ["policy", "policy2", "wdl"],
    )
    with pytest.raises(SystemExit, match="2 candidate policy outputs"):
        resolve_onnx_spec(path)
    # ...and naming one resolves it, so the guard is not just a blanket refusal.
    spec = resolve_onnx_spec(path, policy_output="policy2")
    assert spec.policy_output == "policy2"


def test_a_graph_without_an_1858_head_raises(tmp_path: Path) -> None:
    path = _export(_NoPolicyNet(), tmp_path / "nopolicy.onnx", ["notpolicy", "wdl"])
    with pytest.raises(SystemExit, match="no 1858-wide policy output"):
        resolve_onnx_spec(path)


def test_an_unknown_explicit_output_name_raises(echo_onnx: Path) -> None:
    with pytest.raises(SystemExit, match="is not an output of this graph"):
        resolve_onnx_spec(echo_onnx, policy_output="/output/policy")


def test_explicit_names_are_validated_too_not_taken_on_trust(echo_onnx: Path) -> None:
    """Naming all three must not buy a pass on opening the graph.

    A short-circuit there would accept a typo'd override at parse time and let
    it surface from ORT only after the audit set had loaded and SF had spent an
    hour labelling — the exact cost this resolution is placed early to avoid.
    """
    with pytest.raises(SystemExit, match="--onnx-input-name 'plane' is not in"):
        resolve_onnx_spec(
            echo_onnx, input_name="plane", policy_output="policy", wdl_output="wdl",
        )
    with pytest.raises(SystemExit, match="is not an output of this graph"):
        resolve_onnx_spec(
            echo_onnx, input_name="planes", policy_output="policy", wdl_output="wdl2",
        )
    spec = resolve_onnx_spec(
        echo_onnx, input_name="planes", policy_output="policy", wdl_output="wdl",
    )
    assert (spec.input_name, spec.policy_output, spec.wdl_output) == (
        "planes", "policy", "wdl",
    )


def test_cpu_device_never_offers_ort_the_cuda_provider() -> None:
    """`--device cpu` must be unable to allocate on a training GPU, not merely
    prefer not to."""
    assert onnx_providers_for_device("cpu") == (CPU_PROVIDER,)


# --------------------------------------------------------------------------
# --gpu-mem-fraction must REACH onnxruntime, not just be set somewhere
# --------------------------------------------------------------------------
#
# The defect: the cap was `torch.cuda.set_per_process_memory_fraction`, which
# bounds the torch caching allocator. On `--onnx` the net is not a torch module
# — it allocates through ORT's own CUDA arena, which only `gpu_mem_limit` in
# the CUDA provider OPTIONS bounds. The rulers passed bare provider NAMES and
# printed "GPU memory capped", so a ~700M session went onto the trainer's card
# uncapped while the log said otherwise.
#
# So the assertions below are made at the ORT BOUNDARY: the `providers=`
# argument `onnxruntime.InferenceSession` was actually called with. Asserting
# on the tuple our own helper returned one line earlier would prove nothing
# about whether it survives the trip through `NetSource.load` / `OnnxChessNet`.


class _ProvidersReported:
    """A real ORT session that REPORTS a chosen provider list.

    The delegate below always runs on CPU (these tests must never allocate on
    the live trainer's GPU), so `get_providers()` is overridden to stand in for
    a box whose CUDA EP loads. Only the RECORDED constructor argument is the
    thing under test; this wrapper exists so the rest of `OnnxChessNet.__init__`
    — dtype probe, WDL probe, a real `run()` — executes for real.
    """

    def __init__(self, real: object, reported: list[str]) -> None:
        self._real = real
        self._reported = reported

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def get_providers(self) -> list[str]:
        return list(self._reported)


def _capture_ort_providers(
    monkeypatch: pytest.MonkeyPatch, *, reported: list[str],
) -> list[object]:
    """Record every `providers=` handed to `onnxruntime.InferenceSession`."""
    import onnxruntime as ort

    real_cls = ort.InferenceSession
    seen: list[object] = []

    def _factory(
        path: str | Path, sess_options: object = None, providers: object = None,
    ) -> object:
        seen.append(providers)
        return _ProvidersReported(
            real_cls(path, sess_options, providers=[CPU_PROVIDER]), reported,
        )

    monkeypatch.setattr(ort, "InferenceSession", _factory)
    return seen


def _fake_card(monkeypatch: pytest.MonkeyPatch, total_bytes: int) -> None:
    """Report a card of a known size WITHOUT touching a GPU."""

    class _Props:
        total_memory = total_bytes

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _idx: _Props(),
    )


GIB = 1024 ** 3


def test_the_gpu_mem_cap_reaches_the_ort_session_constructor(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test: `--gpu-mem-fraction` must arrive at ORT.

    Read at the boundary — the `providers=` argument `InferenceSession` was
    called with. A bare `"CUDAExecutionProvider"` string there is the bug: ORT
    gets no arena limit and the session is unbounded next to the trainer.
    """
    spec = resolve_onnx_spec(echo_onnx)  # resolved BEFORE the recorder is armed
    seen = _capture_ort_providers(monkeypatch, reported=[CUDA_PROVIDER, CPU_PROVIDER])
    _fake_card(monkeypatch, 32 * GIB)

    NetSource(onnx=spec).load(device="cuda", gpu_mem_fraction=0.4)

    assert len(seen) == 1, "the scoring session must be built exactly once"
    assert seen[0] == [
        (CUDA_PROVIDER, {"device_id": 0, "gpu_mem_limit": int(0.4 * 32 * GIB)}),
        CPU_PROVIDER,
    ]


def test_an_indexed_cuda_device_reaches_ort_as_device_id(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--device cuda:1` must land on card 1, in ORT's own options.

    Previously refused outright, because provider NAMES carry no `device_id`
    and ORT would have used card 0 — here, the live trainer's. The options form
    closes that: the index is passed, so it can be honoured.
    """
    spec = resolve_onnx_spec(echo_onnx)
    seen = _capture_ort_providers(monkeypatch, reported=[CUDA_PROVIDER, CPU_PROVIDER])
    _fake_card(monkeypatch, 16 * GIB)

    NetSource(onnx=spec).load(device="cuda:1", gpu_mem_fraction=0.25)

    assert seen[0] == [
        (CUDA_PROVIDER, {"device_id": 1, "gpu_mem_limit": 4 * GIB}),
        CPU_PROVIDER,
    ]


def test_no_fraction_means_no_invented_gpu_mem_limit(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `--gpu-mem-fraction` must produce NO limit key, not a default one.

    A fabricated cap would be the mirror of the bug: a number in the options
    that no one asked for and that nothing in the log names.
    """
    spec = resolve_onnx_spec(echo_onnx)
    seen = _capture_ort_providers(monkeypatch, reported=[CUDA_PROVIDER, CPU_PROVIDER])

    NetSource(onnx=spec).load(device="cuda")

    assert seen[0] == [(CUDA_PROVIDER, {"device_id": 0}), CPU_PROVIDER]


def test_the_cpu_path_hands_ort_cpu_only_even_with_a_fraction(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--device cpu` stays structurally unable to reach the trainer's GPU."""
    spec = resolve_onnx_spec(echo_onnx)
    seen = _capture_ort_providers(monkeypatch, reported=[CPU_PROVIDER])

    NetSource(onnx=spec).load(device="cpu", gpu_mem_fraction=0.4)

    assert seen[0] == [CPU_PROVIDER]


def test_a_cpu_session_is_not_described_as_an_uncapped_cuda_arena(
    echo_onnx: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A CPU session has no CUDA arena, so neither "capped" nor "uncapped" is
    a true description of it. Saying "UNCAPPED" there would be the same
    over-claim in the opposite direction."""
    NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu", tag="audit")
    out = capsys.readouterr().out
    assert "no GPU memory allocated" in out
    assert "arena" not in out


def test_the_fraction_to_bytes_conversion_rounds_down_and_is_bounded() -> None:
    """The arithmetic ORT's absolute budget needs, checkable without a card."""
    assert gpu_mem_limit_bytes(0.4, 32 * GIB) == int(0.4 * 32 * GIB)
    assert gpu_mem_limit_bytes(1.0, 100) == 100
    # Rounded DOWN: a cap larger than the share asked for is not a cap.
    assert gpu_mem_limit_bytes(0.3, 10) == 3
    # ...but never zero, which ORT would read as "no arena at all".
    assert gpu_mem_limit_bytes(1e-12, 10) == 1
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(SystemExit, match="gpu-mem-fraction"):
            gpu_mem_limit_bytes(bad, 32 * GIB)


def test_a_fraction_on_cpu_is_reported_ignored_not_silently_dropped(
    capsys: pytest.CaptureFixture[str], echo_onnx: Path,
) -> None:
    """An accepted-and-dropped flag is the same defect one size down."""
    net = NetSource(onnx=resolve_onnx_spec(echo_onnx))
    apply_gpu_mem_cap(net=net, device="cpu", gpu_mem_fraction=0.4, tag="audit")
    out = capsys.readouterr().out
    assert "IGNORED" in out
    assert "capped" not in out.lower()


def test_the_torch_cap_never_claims_to_have_capped_the_onnx_session(
    capsys: pytest.CaptureFixture[str], echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The log line must name the allocator it bounded.

    The original read `GPU memory capped at fraction 0.4` after a torch-only
    call — true of the torch allocator, false of the ONNX session the `--onnx`
    run actually evaluates in.
    """
    calls: list[tuple[float, int]] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda frac, device: calls.append((frac, device)),
    )
    net = NetSource(onnx=resolve_onnx_spec(echo_onnx))
    apply_gpu_mem_cap(net=net, device="cuda:1", gpu_mem_fraction=0.4, tag="audit")

    out = capsys.readouterr().out
    assert calls == [(0.4, 1)]
    assert "TORCH GPU allocator capped" in out
    assert "ONNX session's CUDA arena is capped separately" in out


def test_the_ort_cap_is_reported_only_after_the_session_exists(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`verify_onnx_session_device` prints the ORT cap it can SEE was requested,
    alongside the providers the session actually came up on."""
    spec = resolve_onnx_spec(echo_onnx)
    _capture_ort_providers(monkeypatch, reported=[CUDA_PROVIDER, CPU_PROVIDER])
    _fake_card(monkeypatch, 32 * GIB)
    NetSource(onnx=spec).load(device="cuda", gpu_mem_fraction=0.5, tag="audit")
    out = capsys.readouterr().out
    assert "onnxruntime session on ['CUDAExecutionProvider'" in out
    assert f"CUDA arena capped at {16 * GIB} bytes" in out


# --------------------------------------------------------------------------
# a CUDA request must mean the device it says
# --------------------------------------------------------------------------


def _fake_providers(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: providers)


def test_the_cuda_guard_reads_the_session_not_the_compile_time_list(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parse-time guard CANNOT catch a CUDA session that fell back to CPU.

    `onnxruntime.get_available_providers()` is the list the wheel was COMPILED
    with. It stays populated when the provider's shared library cannot load —
    measured on this box with the GPU wheel: `get_available_providers()` returns
    `[Tensorrt, CUDA, CPU]` (so `validate_onnx_device('cuda')` passes) while
    every session it builds comes back `['CPUExecutionProvider']`, because
    `libonnxruntime_providers_cuda.so` needs a cuDNN that is not installed. ORT
    warns and drops; it does not fail. So the ONLY reading that can tell the two
    apart is the session's own.

    Below: a REAL session, really on CPU (no GPU is touched), asked about a
    `--device cuda` run. The compile-time list is faked to contain CUDA, which
    is exactly the state that makes the old guard a gate that cannot fail.
    """
    from chess_anti_engine.onnx.load import OnnxChessNet

    _fake_providers(monkeypatch, [CUDA_PROVIDER, CPU_PROVIDER])
    validate_onnx_device("cuda")  # the parse-time screen is happy — and wrong

    model = OnnxChessNet(
        echo_onnx,
        input_name="planes",
        policy_output_name="policy",
        wdl_output_name="wdl",
        providers=[CPU_PROVIDER],
    )
    assert model.session_providers() == [CPU_PROVIDER]
    with pytest.raises(SystemExit, match="DROPPED"):
        verify_onnx_session_device(model, "cuda")
    # ...and a CPU run of the same session is fine: nothing was claimed.
    verify_onnx_session_device(model, "cpu")


def test_a_cuda_request_without_a_cuda_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORT DROPS an unavailable provider and runs the next one silently.

    A `--device cuda` run of a BT4-sized net then becomes an hours-long CPU run
    that still calls itself CUDA.
    """
    _fake_providers(monkeypatch, [CPU_PROVIDER])
    with pytest.raises(SystemExit, match="no CUDAExecutionProvider"):
        validate_onnx_device("cuda")
    # ...and CPU is unaffected: a CPU-only onnxruntime runs CPU rulers fine.
    validate_onnx_device("cpu")


def test_the_device_is_validated_from_the_parsed_args_not_at_load_time(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check has to fire at parse time, or it fires after the SF pass."""
    _fake_providers(monkeypatch, [CPU_PROVIDER])
    args = _args(onnx=echo_onnx, device="cuda")
    with pytest.raises(SystemExit, match="no CUDAExecutionProvider"):
        net_source_from_args(args)


# --------------------------------------------------------------------------
# flag combinations that cannot mean anything
# --------------------------------------------------------------------------


def test_stored_encoding_with_onnx_is_refused(echo_onnx: Path) -> None:
    """Stored rows are 175-plane production input; a foreign net reads 112.

    Detected downstream anyway — but in `audit_targets` only after
    `_shallow_sf_records`, i.e. potentially an hour of Stockfish spent on a
    combination that was invalid when the flags were parsed.
    """
    onnx_net = NetSource(onnx=resolve_onnx_spec(echo_onnx))
    with pytest.raises(SystemExit, match="stored is not compatible with --onnx"):
        reject_stored_encoding_for_onnx(onnx_net, "stored")
    # The other three combinations are all legitimate and must pass through.
    reject_stored_encoding_for_onnx(onnx_net, "fen_only")
    reject_stored_encoding_for_onnx(NetSource(checkpoint="c"), "stored")
    reject_stored_encoding_for_onnx(NetSource(checkpoint="c"), "fen_only")


def test_both_rulers_reject_stored_plus_onnx_before_the_expensive_work() -> None:
    import inspect

    from scripts import audit_targets, value_regret

    expensive = {
        value_regret: "load_audit_set(",
        audit_targets: "_shallow_sf_records(",
    }
    guard = "reject_stored_encoding_for_onnx(net, args.input_encoding)"
    for module, costly_call in expensive.items():
        src = inspect.getsource(module.main)
        assert guard in src, f"{module.__name__}.main does not run the guard at all"
        assert src.index(guard) < src.index(costly_call)


def test_the_audit_dump_rows_carry_the_net_too() -> None:
    """`audit_targets --dump-per-position` must stamp the net like value_regret.

    Wiring pin rather than an end-to-end run: driving that dump needs a config,
    an audit set and a Stockfish labelling pass. A dump outlives its report and
    gets joined to other dumps, so an unstamped row is a number whose weights
    cannot be recovered.
    """
    import inspect

    from scripts import audit_targets

    src = inspect.getsource(audit_targets.main)
    dump_at = src.index("per_pos_dump.append({")
    assert '"net": net.label,' in src[dump_at:dump_at + 600]


# --------------------------------------------------------------------------
# loading: the requested net, or nothing
# --------------------------------------------------------------------------


def test_the_onnx_path_never_loads_a_checkpoint(
    echo_onnx: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.uci import model_loader

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("--onnx fell back to the checkpoint loader")

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _boom)
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    # The adapter must DECLARE LC0's input contract, or every caller encodes
    # production planes and the 112-plane slice silently means something else.
    assert model.input_history_encoding == "lc0_root"
    assert model.input_extra_features == "v1"
    assert model.policy_encoding == "az_4672"
    assert not model.training


def test_the_checkpoint_path_still_goes_through_the_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.uci import model_loader

    seen: list[str] = []

    class _Stub(torch.nn.Module):
        pass

    def _fake(path: str, **_k: object) -> torch.nn.Module:
        seen.append(path)
        return _Stub()

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _fake)
    model = NetSource(checkpoint="some/trainer.pt").load(device="cpu")
    assert seen == ["some/trainer.pt"]
    assert not model.training


# --------------------------------------------------------------------------
# the policy actually read is the board-aware remap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        # White to move, both castles legal, and a 7th-rank pawn plus a rook
        # that can slide onto the 8th rank -- the two families a static table
        # cannot map.
        "r3k2r/PP4pp/8/8/8/8/6PP/R3K2R w KQkq - 0 1",
        # Same, black to move: proves the orientation flip is applied too.
        "r3k2r/pp4pp/8/8/8/8/6PP/R3K2R b KQkq - 0 1",
    ],
)
def test_every_legal_move_reads_its_leela_slot(echo_onnx: Path, fen: str) -> None:
    """policy[i] == i in the graph, so the returned logit IS the Leela slot read.

    Compared per move against ``leela_index_for_move``, which derives the slot
    from the move itself rather than from the compact tables under test.
    """
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    board = chess.Board(fen)
    x = encode_position_for_model(model, board)
    out = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])
    pol = out["policy_own"][0].numpy()

    checked = 0
    castles = 0
    for move in board.legal_moves:
        full = move_to_index(move, board)
        if full < 0:
            continue
        want = leela_index_for_move(board, move)
        assert want >= 0
        assert float(pol[full]) == pytest.approx(float(want)), (
            f"{move.uci()} read Leela slot {pol[full]}, expected {want}"
        )
        checked += 1
        castles += int(board.is_castling(move))
    # Both fens keep all four castling rights, so a remap that dropped the
    # castling context would have had to fail one of the assertions above.
    assert castles == 2
    assert checked >= 20


def test_castling_reads_leelas_king_takes_rook_slot_not_the_slide(echo_onnx: Path) -> None:
    """The measured 49-120x castling-prior error, as an assertion.

    Leela's table has BOTH an ``e1g1`` slide entry and the ``e1h1`` castling
    entry, so reading the wrong one is silent. Our compact slot index differs
    from both, which is what makes this a real cross-convention check.
    """
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    x = encode_position_for_model(model, board)
    pol = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])["policy_own"][0]

    oo = chess.Move.from_uci("e1g1")
    read = float(pol[move_to_index(oo, board)])
    assert read == pytest.approx(float(LC0_1858_UCI_TO_IDX["e1h1"]))
    assert read != pytest.approx(float(LC0_1858_UCI_TO_IDX["e1g1"]))
    assert read != pytest.approx(float(compact_index_for_move(board, oo)))


def test_the_wdl_head_is_returned_as_logits_not_raw_probabilities(
    echo_onnx: Path,
) -> None:
    """The search softmaxes ``wdl``; feeding it probabilities crushes them."""
    model = NetSource(onnx=resolve_onnx_spec(echo_onnx)).load(device="cpu")
    x = encode_position_for_model(model, chess.Board())
    wdl = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...])["wdl"]
    assert float(wdl.sum()) < 0.0  # log-probs, not a row summing to 1
    np.testing.assert_allclose(
        torch.softmax(wdl, dim=-1).sum(dim=-1).numpy(), 1.0, atol=1e-5,
    )


# --------------------------------------------------------------------------
# end to end through the ruler's real main()
# --------------------------------------------------------------------------


def _audit_row(key: str, fen: str) -> str:
    board = chess.Board(fen)
    moves = [m.uci() for m in board.legal_moves][:6]
    return json.dumps({
        "key": key,
        "fen": fen,
        "phase": 1,
        "source": 0,
        "multipv": [
            {"move": m, "cp": 40 - 10 * i} for i, m in enumerate(moves)
        ],
        "wdl": [400, 400, 200],
        "nodes": 1_000_000,
        "depth": 40,
    })


@pytest.fixture
def mini_audit_set(tmp_path: Path) -> Path:
    path = tmp_path / "mini_audit.jsonl"
    path.write_text(
        "\n".join([
            _audit_row("a", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            _audit_row("b", "r3k2r/pp3ppp/2n5/3q4/8/2N5/PPP2PPP/R3K2R w KQkq - 0 12"),
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def test_value_regret_main_scores_the_onnx_net(
    echo_onnx: Path,
    mini_audit_set: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole CLI: --onnx in, a number out, and the checkpoint loader armed."""
    from chess_anti_engine.uci import model_loader
    from scripts import value_regret

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("value_regret --onnx loaded a checkpoint")

    dump = tmp_path / "per_position.jsonl"
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", _boom)
    monkeypatch.setattr("sys.argv", [
        "value_regret.py",
        "--onnx", str(echo_onnx),
        "--audit-set", str(mini_audit_set),
        "--device", "cpu",
        "--batch-size", "8",
        "--min-pieces", "0",
        "--dump-per-position", str(dump),
    ])
    value_regret.main()
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "cp (n=2)" in out
    # ⚑ The RESULT HEADER must name the net, not merely some earlier progress
    # line: a header reading "@ <checkpoint>" over an ONNX number is the whole
    # failure mode. Checked on the header line itself, because an
    # `f"onnx:..." in out` assertion passes on the [net-source] echo alone
    # (measured: that version survived deleting `net.label` from the header).
    headers = [ln for ln in out.splitlines() if ln.startswith("=== value-head")]
    assert len(headers) == 1
    assert f"@ onnx:{echo_onnx}" in headers[0]
    # A dump is a report too: every row must carry the net that produced it.
    rows = [json.loads(ln) for ln in dump.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {r["net"] for r in rows} == {f"onnx:{echo_onnx} "
                                        f"[in=planes policy=policy wdl=wdl]"}


def test_value_regret_main_refuses_a_missing_net(
    mini_audit_set: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import value_regret

    monkeypatch.setattr("sys.argv", [
        "value_regret.py", "--audit-set", str(mini_audit_set), "--device", "cpu",
    ])
    with pytest.raises(SystemExit, match="exactly one"):
        value_regret.main()


def test_both_rulers_resolve_their_net_before_any_scoring_work() -> None:
    """A late resolve would spend the audit set — or an hour of SF — first.

    ``_shallow_sf_records`` is the audit's Stockfish labelling pass; a bad
    ``--onnx`` surfacing after it is exactly the failure the other fail-fast
    guards in that ``main()`` were written for.
    """
    import inspect

    from scripts import audit_targets, value_regret

    expensive = {
        value_regret: "load_audit_set(",
        audit_targets: "_shallow_sf_records(",
    }
    for module, costly_call in expensive.items():
        src = inspect.getsource(module.main)
        assert "net_source_from_args(args)" in src
        parse_at = src.index("ap.parse_args()")
        resolve_at = src.index("net_source_from_args(args)")
        costly_at = src.index(costly_call)
        assert parse_at < resolve_at < costly_at


def test_audit_targets_net_candidates_takes_a_net_source_not_a_checkpoint() -> None:
    """No ``checkpoint=`` back door, and no default that could load something else."""
    import inspect

    from scripts import audit_targets

    params = inspect.signature(audit_targets._net_candidates).parameters
    assert "checkpoint" not in params
    assert params["net"].default is inspect.Parameter.empty
    assert params["net"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_onnx_spec_label_names_every_tensor_it_resolved() -> None:
    spec = OnnxNetSpec(
        path=Path("/tmp/x.onnx"), input_name="i", policy_output="p", wdl_output="w",
    )
    assert spec.label == "onnx:/tmp/x.onnx [in=i policy=p wdl=w]"
