"""``deepfin --selfcheck`` components and the UCI engine's startup gate.

Each component has a passing case on the real install and a failure case that
simulates the install defect it exists to catch (missing extension, ABI
mismatch, stale .so, zero feature planes, non-finite forward pass, wrong TB
answer, illegal bestmove).
"""
from __future__ import annotations

import importlib.machinery
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.uci import selfcheck as sc
from chess_anti_engine.utils.syzygy import syzygy_roots

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(fn) -> sc.CheckResult:
    return sc._run_check("x", fn)


# ── python ────────────────────────────────────────────────────────────────


def test_python_in_range_passes() -> None:
    assert _run(lambda: sc.check_python((3, 13, 1), gil_enabled=True)).status == "PASS"
    assert _run(lambda: sc.check_python((3, 10, 12), gil_enabled=True)).status == "PASS"


@pytest.mark.parametrize("version", [(3, 9, 18), (3, 10, 11), (3, 14, 0)])
def test_python_out_of_range_fails(version: tuple[int, int, int]) -> None:
    res = _run(lambda: sc.check_python(version, gil_enabled=True))
    assert res.status == "FAIL"
    assert "outside the supported range" in res.detail


def test_python_free_threaded_fails() -> None:
    res = _run(lambda: sc.check_python((3, 13, 1), gil_enabled=False))
    assert res.status == "FAIL"
    assert "free-threaded" in res.detail


# ── torch / CUDA ──────────────────────────────────────────────────────────

# torch 2.6.0+cu124's arch list: no Blackwell (sm_120) cubin and no PTX.
_CU124_ARCHES = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


class _FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def __mul__(self, k: float) -> _FakeTensor:
        return _FakeTensor(self.value * k)

    def sum(self) -> _FakeTensor:
        return _FakeTensor(self.value * 8)

    def item(self) -> float:
        return self.value


def _fake_torch(*, capability: tuple[int, int], arches: list[str], available: bool = True,
                op_value: float = 1.0) -> types.ModuleType:
    mod = types.ModuleType("torch")
    setattr(mod, "__version__", "9.9.9+cu999")
    setattr(mod, "version", types.SimpleNamespace(cuda="12.4"))
    setattr(mod, "cuda", types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 1 if available else 0,
        get_device_name=lambda i: "NVIDIA GeForce RTX 5090",
        get_device_capability=lambda i: capability,
        get_arch_list=lambda: arches,
    ))
    setattr(mod, "ones", lambda n, device=None: _FakeTensor(op_value))
    return mod


def test_torch_real_cpu_passes() -> None:
    res = _run(lambda: sc.check_torch(["cpu"]))
    assert res.status == "PASS", res.detail
    assert f"torch {torch.__version__}" in res.detail


def test_torch_cuda_wheel_too_old_for_gpu_fails_with_fix() -> None:
    fake = _fake_torch(capability=(12, 0), arches=_CU124_ARCHES)
    res = _run(lambda: sc.check_torch(["cuda:0"], fake))
    assert res.status == "FAIL"
    assert "sm_120" in res.detail
    assert "newer cuXXX wheel" in res.detail


def test_torch_cuda_wheel_supporting_gpu_passes() -> None:
    fake = _fake_torch(capability=(12, 0), arches=[*_CU124_ARCHES, "sm_100", "sm_120"])
    res = _run(lambda: sc.check_torch(["cuda:0"], fake))
    assert res.status == "PASS", res.detail
    assert "cuda:0=NVIDIA GeForce RTX 5090" in res.detail


def test_torch_auto_resolves_every_visible_gpu() -> None:
    fake = _fake_torch(capability=(12, 0), arches=_CU124_ARCHES)
    res = _run(lambda: sc.check_torch([], fake))
    assert res.status == "FAIL"  # auto picks cuda:0, which this wheel cannot run


def test_torch_wrong_tensor_result_fails() -> None:
    fake = _fake_torch(capability=(9, 0), arches=["sm_90"], op_value=3.0)
    res = _run(lambda: sc.check_torch(["cuda:0"], fake))
    assert res.status == "FAIL"
    assert "expected 16.0" in res.detail


def test_torch_cuda_requested_on_cpu_wheel_fails() -> None:
    fake = _fake_torch(capability=(9, 0), arches=[], available=False)
    res = _run(lambda: sc.check_torch(["cuda:0"], fake))
    assert res.status == "FAIL"
    assert "is_available() is False" in res.detail


@pytest.mark.parametrize(
    ("arches", "cap", "ok"),
    [
        (["sm_86"], (8, 9), True),         # same major, newer minor: cubin runs
        (["sm_90"], (8, 9), False),        # older device than the cubin
        (["sm_90"], (12, 0), False),       # cubins do not cross majors
        (["compute_90"], (12, 0), True),   # PTX JITs forward
        (["sm_120"], (12, 0), True),
        (["sm_90a"], (9, 0), True),
    ],
)
def test_arch_supports(arches: list[str], cap: tuple[int, int], ok: bool) -> None:
    assert sc._arch_supports(arches, cap) is ok


# ── extensions: import, symbols, ABI ──────────────────────────────────────


def test_extensions_real_install_passes() -> None:
    res = _run(sc.check_extensions)
    assert res.status == "PASS", res.detail


def test_extensions_missing_module_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # A `None` entry makes the import system raise ImportError for that name.
    monkeypatch.setitem(sys.modules, "chess_anti_engine.encoding._features_ext", None)
    res = _run(sc.check_extensions)
    assert res.status == "FAIL"
    assert "_features_ext is not importable" in res.detail


def test_extensions_abi_mismatch_fails() -> None:
    res = _run(lambda: sc.check_extensions(abi_required=lambda: 10_000))
    assert res.status == "FAIL"
    assert "ABI_VERSION=" in res.detail
    assert "< required 10000" in res.detail


def test_extensions_abi_exactly_one_below_requirement_fails() -> None:
    import chess_anti_engine.mcts._mcts_tree as tree

    actual = int(tree.ABI_VERSION)
    assert _run(lambda: sc.check_extensions(abi_required=lambda: actual)).status == "PASS"
    res = _run(lambda: sc.check_extensions(abi_required=lambda: actual + 1))
    assert res.status == "FAIL"
    assert f"ABI_VERSION={actual} < required {actual + 1}" in res.detail


def test_extensions_missing_symbol_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType("chess_anti_engine.encoding._features_ext")
    setattr(stub, "compute_extra_features", lambda *a: None)
    monkeypatch.setitem(sys.modules, "chess_anti_engine.encoding._features_ext", stub)
    res = _run(sc.check_extensions)
    assert res.status == "FAIL"
    assert "lacks compute_relation_matrices" in res.detail


# ── freshness (stale .so) ─────────────────────────────────────────────────


def _fake_tree(tmp_path: Path, *, so_mtime: int, src_mtime: int,
               records: bool = False) -> Path:
    """A checkout skeleton with the real freshness tool and every listed file."""
    from chess_anti_engine.ext_build_record import hash_inputs, write_record

    tool = sc._load_freshness_tool(REPO_ROOT)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "check_c_extensions_fresh.py", tmp_path / "scripts")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    for spec in tool.EXTENSION_SPECS:
        for dep in spec.dependencies:
            p = tmp_path / dep
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"source of {dep}\n", encoding="utf-8")
            os.utime(p, ns=(src_mtime, src_mtime))
        base = tmp_path.joinpath(*spec.module.split("."))
        out = base.with_name(base.name + suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"binary of {spec.module}\n", encoding="utf-8")
        os.utime(out, ns=(so_mtime, so_mtime))
        if records:
            hashes = hash_inputs(tmp_path, spec.dependencies)
            assert hashes is not None
            write_record(out, module=spec.module, input_hashes=hashes)
    return tmp_path


_FEATURES_IMPL = Path("chess_anti_engine") / "encoding" / "_features_impl.h"


def test_freshness_fresh_tree_passes(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    res = _run(lambda: sc.check_freshness(root))
    assert res.status == "PASS", res.detail
    assert "no build record" in res.detail


def test_freshness_stale_so_fails(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    os.utime(root / _FEATURES_IMPL, ns=(3_000_000_000, 3_000_000_000))
    res = _run(lambda: sc.check_freshness(root))
    assert res.status == "FAIL"
    assert "_features_ext is older than chess_anti_engine/encoding/_features_impl.h" in res.detail
    assert "checkout or a copy" in res.detail  # mtime fallback says it can be a false alarm


def test_freshness_ignores_extensions_the_engine_does_not_load(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    src = root / "chess_anti_engine" / "nnue" / "_nnue_ext.c"
    os.utime(src, ns=(3_000_000_000, 3_000_000_000))
    assert sc.freshness_problems(root) == []


def test_freshness_record_touch_only_does_not_refuse(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000, records=True)
    for rel in (_FEATURES_IMPL, Path("setup.py")):
        os.utime(root / rel, ns=(3_000_000_000, 3_000_000_000))  # newer, same content
    assert sc.freshness_problems(root) == []
    res = _run(lambda: sc.check_freshness(root))
    assert res.status == "PASS", res.detail
    assert "source content matches the build record" in res.detail


def test_freshness_record_real_edit_refuses(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000, records=True)
    (root / _FEATURES_IMPL).write_text("edited\n", encoding="utf-8")
    os.utime(root / _FEATURES_IMPL, ns=(1_000_000_000, 1_000_000_000))  # even with an OLD mtime
    problems = sc.freshness_problems(root)
    assert any("_features_ext" in p and str(_FEATURES_IMPL) in p for p in problems), problems
    assert any("_lc0_ext" in p for p in problems)  # shares the header


def test_freshness_record_replaced_artifact_refuses(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000, records=True)
    base = root / "chess_anti_engine" / "mcts" / "_mcts_tree"
    out = base.with_name(base.name + importlib.machinery.EXTENSION_SUFFIXES[0])
    out.write_text("a different binary\n", encoding="utf-8")
    problems = sc.freshness_problems(root)
    assert any("_mcts_tree" in p and "not the file its build record describes" in p
               for p in problems), problems


def test_freshness_record_missing_dependency_refuses(tmp_path: Path) -> None:
    """A dependency the record does not cover (a header added to the include
    graph since the build) is unverified, so stale."""
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000, records=True)
    base = root / "chess_anti_engine" / "encoding" / "_lc0_ext"
    rec = Path(str(base) + importlib.machinery.EXTENSION_SUFFIXES[0] + ".build-record.json")
    data = json.loads(rec.read_text(encoding="utf-8"))
    del data["inputs"][str(_FEATURES_IMPL)]
    rec.write_text(json.dumps(data), encoding="utf-8")
    problems = sc.freshness_problems(root)
    assert any("_lc0_ext" in p and f"{_FEATURES_IMPL} (not in the build record)" in p
               for p in problems), problems


def test_this_checkout_has_build_records() -> None:
    """setup.py's build_ext hook wrote records for the in-place build."""
    rows = sc._freshness_rows(REPO_ROOT)
    assert all(method == "content" and problem is None for _, method, problem in rows), rows


def test_freshness_without_sources_skips() -> None:
    res = _run(lambda: sc.check_freshness(None, use_source_root=False))
    assert res.status == "SKIP"


def test_source_root_is_this_checkout() -> None:
    assert sc.source_root() == REPO_ROOT


# ── encoder: fast path really taken ───────────────────────────────────────


def test_encoder_real_passes() -> None:
    res = _run(sc.check_encoder)
    assert res.status == "PASS", res.detail


def test_encoder_zero_feature_planes_fails() -> None:
    from chess_anti_engine.encoding.cboard_encode import encode_cboard

    def zero_features(cb, **kw):
        out = encode_cboard(cb, **kw).copy()
        out[112:] = 0.0
        return out

    res = _run(lambda: sc.check_encoder(c_encode=zero_features))
    assert res.status == "FAIL"
    assert "feature planes are all zero" in res.detail


def test_encoder_mismatch_with_reference_fails() -> None:
    def perturbed(board: chess.Board, version: str) -> np.ndarray:
        out = sc._python_reference_planes(board, version)
        out[120, 0, 0] += 1e-3
        return out

    res = _run(lambda: sc.check_encoder(reference=perturbed))
    assert res.status == "FAIL"
    assert "differs from the Python reference" in res.detail


def test_encoder_tree_leaf_encoder_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_anti_engine.mcts._mcts_tree as tree

    real = tree.batch_encode_146

    def wrong(cboards, out):
        real(cboards, out)
        out[:, 120, 0, 0] += 1.0

    monkeypatch.setattr(tree, "batch_encode_146", wrong)
    res = _run(sc.check_encoder)
    assert res.status == "FAIL"
    assert "tree leaf encoder differs" in res.detail


# ── movegen / perft ───────────────────────────────────────────────────────


def test_movegen_real_passes() -> None:
    res = _run(sc.check_movegen)
    assert res.status == "PASS", res.detail
    assert "perft d3=8902" in res.detail


def test_movegen_wrong_perft_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "PERFT_CASES", ((sc.ENCODER_FENS[0], 3, 8903),))
    res = _run(sc.check_movegen)
    assert res.status == "FAIL"
    assert "perft(3) = 8902, expected 8903" in res.detail


# ── C search ──────────────────────────────────────────────────────────────


def test_search_real_runs_through_c_tree() -> None:
    res = _run(sc.check_search)
    assert res.status == "PASS", res.detail
    assert f"{sc.SEARCH_SIMS} visits in the C tree" in res.detail


def test_search_not_c_tree_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_anti_engine.mcts.gumbel_c as gumbel_c

    def python_like(*_args, **_kwargs):
        return (None, [chess.Move.from_uci("e2e4")], None, None, object(), [0])

    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", python_like)
    res = _run(sc.check_search)
    assert res.status == "FAIL"
    assert "not the C MCTSTree" in res.detail


# ── checkpoint ────────────────────────────────────────────────────────────


def test_checkpoint_absent_skips() -> None:
    assert _run(lambda: sc.check_checkpoint(None)).status == "SKIP"


def test_checkpoint_tiny_passes(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    res = _run(lambda: sc.check_checkpoint(str(ckpt)))
    assert res.status == "PASS", res.detail
    assert "forward finite" in res.detail


def test_checkpoint_missing_tensor_fails(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    blob = torch.load(ckpt / "trainer.pt", weights_only=True)
    dropped = next(iter(blob["model"]))
    del blob["model"][dropped]
    torch.save(blob, ckpt / "trainer.pt")
    res = _run(lambda: sc.check_checkpoint(str(ckpt)))
    assert res.status == "FAIL"
    assert "left at random initialisation" in res.detail
    assert dropped in res.detail


def test_checkpoint_compile_prefix_is_a_note_not_a_failure(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    blob = torch.load(ckpt / "trainer.pt", weights_only=True)
    blob["model"] = {f"_orig_mod.{k}": v for k, v in blob["model"].items()}
    torch.save(blob, ckpt / "trainer.pt")
    res = _run(lambda: sc.check_checkpoint(str(ckpt)))
    assert res.status == "PASS", res.detail
    assert "_orig_mod. key prefix normalised" in res.detail


def test_checkpoint_v1_to_v2_input_migration_is_a_note(tmp_path: Path) -> None:
    """A v1 (146-plane) net served as v2_threats: the engine zero-pads 29 columns."""
    from chess_anti_engine.model import ModelConfig, build_model

    ckpt = sc._write_tiny_checkpoint(tmp_path)  # params.json says v2_threats
    torch.manual_seed(0)
    v1 = build_model(ModelConfig(kind="tiny", input_extra_features="v1"))
    torch.save({"model": v1.state_dict(), "step": 0}, ckpt / "trainer.pt")
    res = _run(lambda: sc.check_checkpoint(str(ckpt)))
    assert res.status == "PASS", res.detail
    assert "Zero-initialized 29 v2_threats input columns" in res.detail


def test_checkpoint_loader_refusal_fails(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    (tmp_path / "params.json").unlink()  # no embedded arch, no params.json: loader refuses
    res = _run(lambda: sc.check_checkpoint(str(ckpt)))
    assert res.status == "FAIL"
    assert "loader refuses" in res.detail


def test_checkpoint_non_finite_forward_fails(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)

    class NanEvaluator:
        def __init__(self, model, *, device: str) -> None:
            del model, device

        def evaluate_encoded(self, x: np.ndarray):
            return np.full((len(x), 4), np.nan), np.zeros((len(x), 3))

    res = _run(lambda: sc.check_checkpoint(str(ckpt), evaluator_factory=NanEvaluator))
    assert res.status == "FAIL"
    assert "non-finite" in res.detail


# ── syzygy ────────────────────────────────────────────────────────────────


def _fake_tb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tb"
    d.mkdir()
    (d / "KQvK.rtbw").write_bytes(b"")
    return d


def test_syzygy_absent_skips() -> None:
    assert _run(lambda: sc.check_syzygy(None)).status == "SKIP"


def test_syzygy_missing_directory_fails(tmp_path: Path) -> None:
    res = _run(lambda: sc.check_syzygy(str(tmp_path / "nope")))
    assert res.status == "FAIL"
    assert "do not exist" in res.detail


def test_syzygy_wrong_answer_fails(tmp_path: Path) -> None:
    d = _fake_tb_dir(tmp_path)
    res = _run(lambda: sc.check_syzygy(str(d), probe=lambda path, fen: 0))
    assert res.status == "FAIL"
    assert "expected 2" in res.detail


def test_syzygy_right_answers_pass(tmp_path: Path) -> None:
    d = _fake_tb_dir(tmp_path)
    answers = dict(sc.SYZYGY_CASES)
    res = _run(lambda: sc.check_syzygy(str(d), probe=lambda path, fen: answers[fen]))
    assert res.status == "PASS", res.detail


def test_syzygy_larger_tables_probed_only_when_present(tmp_path: Path) -> None:
    d = _fake_tb_dir(tmp_path)
    answers = {**dict(sc.SYZYGY_CASES), **{fen: w for _, _, fen, w in sc.SYZYGY_LARGER_CASES}}
    seen: list[str] = []

    def probe(_path: str, fen: str) -> int:
        seen.append(fen)
        return answers[fen]

    res = _run(lambda: sc.check_syzygy(str(d), probe=probe))
    assert res.status == "PASS", res.detail
    assert "not probed: 5-man" in res.detail
    assert sc.SYZYGY_LARGER_CASES[0][2] not in seen
    (d / "KQQRvK.rtbw").write_bytes(b"")
    res = _run(lambda: sc.check_syzygy(str(d), probe=probe))
    assert "5-man KQQRvK" in res.detail
    assert sc.SYZYGY_LARGER_CASES[0][2] in seen


def test_syzygy_wrong_five_man_answer_fails(tmp_path: Path) -> None:
    d = _fake_tb_dir(tmp_path)
    (d / "KQQRvK.rtbw").write_bytes(b"")
    answers = dict(sc.SYZYGY_CASES)
    res = _run(lambda: sc.check_syzygy(str(d), probe=lambda path, fen: answers.get(fen, 0)))
    assert res.status == "FAIL"
    assert sc.SYZYGY_LARGER_CASES[0][2] in res.detail


def _smoke_syzygy() -> Path | None:
    for root in syzygy_roots():
        candidate = root / "data" / "syzygy_3-4man"
        if candidate.is_dir() and any(candidate.glob("*.rtbw")):
            return candidate
    return None


def test_syzygy_real_tables_probe_correctly() -> None:
    path = _smoke_syzygy()
    if path is None:
        pytest.skip("3-4 man smoke Syzygy set (data/syzygy_3-4man) not present")
    res = _run(lambda: sc.check_syzygy(str(path)))
    assert res.status == "PASS", res.detail


# ── UCI exchange ──────────────────────────────────────────────────────────


def test_uci_exchange_real_engine_default_compiled_passes() -> None:
    res = _run(lambda: sc.check_uci(sc.SelfcheckOptions(devices=("cpu",))))
    assert res.status == "PASS", res.detail
    assert "default play path, torch.compile on" in res.detail


def test_uci_exchange_real_engine_fast_passes() -> None:
    res = _run(lambda: sc.check_uci(sc.SelfcheckOptions(devices=("cpu",), fast=True)))
    assert res.status == "PASS", res.detail
    assert "does NOT cover the default compiled play path" in res.detail


def test_uci_engine_rejects_unknown_flag_and_says_so() -> None:
    opts = sc.SelfcheckOptions(devices=("cpu",), fast=True, engine_args=("--hash", "256"))
    res = _run(lambda: sc.check_uci(opts))
    assert res.status == "FAIL"
    assert "unrecognized arguments: --hash 256" in res.detail


_FAKE_ENGINE = textwrap.dedent("""
    import sys
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            print("id name fake"); print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd.startswith("go"):
            print("{extra}", flush=True)
            print("bestmove {move}", flush=True)
        elif cmd == "quit":
            break
    sys.exit({rc})
""")


def _fake(move: str = "e2e4", *, extra: str = "info depth 1", rc: int = 0):
    script = _FAKE_ENGINE.format(move=move, extra=extra, rc=rc)
    return lambda ck, opts, cache: [sys.executable, "-c", script]


def _fake_check(**kw) -> sc.CheckResult:
    opts = sc.SelfcheckOptions(checkpoint="unused", devices=("cpu",))
    return _run(lambda: sc.check_uci(opts, command=_fake(**kw)))


def test_uci_illegal_bestmove_fails() -> None:
    res = _fake_check(move="e2e5")
    assert res.status == "FAIL"
    assert "not legal" in res.detail


def test_uci_legal_bestmove_from_fake_engine_passes() -> None:
    assert _fake_check().status == "PASS"


@pytest.mark.parametrize("marker", sc.UCI_FAILURE_MARKERS)
def test_uci_failure_marker_fails_despite_legal_move(marker: str) -> None:
    res = _fake_check(extra=f"info string {marker} simulated")
    assert res.status == "FAIL", marker
    assert marker in res.detail


def test_uci_timeout_reports_the_real_value() -> None:
    """--uci-timeout 0.5 must say 0.5s, not a rounded 0s."""
    silent = [sys.executable, "-c", "import time; time.sleep(30)"]
    opts = sc.SelfcheckOptions(checkpoint="unused", devices=("cpu",), uci_timeout_s=0.5)
    res = _run(lambda: sc.check_uci(opts, command=lambda ck, o, cache: silent))
    assert res.status == "FAIL"
    assert "within 0.5s" in res.detail, res.detail


def test_uci_nonzero_exit_after_quit_fails() -> None:
    res = _fake_check(rc=3)
    assert res.status == "FAIL"
    assert "exited 3 after quit" in res.detail


_FAKE_STARTUP_FAILURE = textwrap.dedent("""
    import sys
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            print("uciok", flush=True)
        elif cmd == "isready":
            print("Traceback (most recent call last):")
            print("torch._inductor.exc.InductorError: ImportError: GLIBCXX_3.4.32 not found")
            print("RuntimeError: evaluator warmup failed at batch=1")
            print("deepfin: engine startup failed: RuntimeError: evaluator warmup failed at batch=1",
                  flush=True)
            sys.exit(1)
""")


def test_uci_startup_failure_reports_engine_reason_and_root_cause() -> None:
    opts = sc.SelfcheckOptions(checkpoint="unused", devices=("cpu",))
    res = _run(lambda: sc.check_uci(
        opts, command=lambda ck, o, cache: [sys.executable, "-c", _FAKE_STARTUP_FAILURE]))
    assert res.status == "FAIL"
    assert "exit 1" in res.detail
    assert "engine startup failed: RuntimeError: evaluator warmup failed" in res.detail
    assert "GLIBCXX_3.4.32" in res.detail


def test_engine_command_flags_devices_and_private_cache() -> None:
    opts = sc.SelfcheckOptions(devices=("cuda:0", "cuda:1"))
    cmd = sc.engine_command("c", opts, "/tmp/private")
    assert cmd[cmd.index("--devices") + 1] == "cuda:0,cuda:1"
    assert cmd[cmd.index("--compile-cache-dir") + 1] == "/tmp/private"
    assert "--no-compile" not in cmd  # default play path
    assert sc.engine_command("c", sc.SelfcheckOptions(fast=True), "x")[-1] == "--no-compile"


def test_engine_command_operator_cache_and_flags_win() -> None:
    opts = sc.SelfcheckOptions(engine_args=("--walkers", "2", "--compile-cache-dir", "/op"))
    cmd = sc.engine_command("c", opts, "/tmp/private")
    assert cmd[-4:] == ["--walkers", "2", "--compile-cache-dir", "/op"]
    # argparse keeps the LAST occurrence, so the operator's cache dir is used.
    assert max(i for i, a in enumerate(cmd) if a == "--compile-cache-dir") > cmd.index("/tmp/private")


def test_parse_options_passes_unknown_flags_to_engine() -> None:
    opts, as_json = sc.parse_options(
        ["--selfcheck", "--json", "--walkers", "2", "--hash", "256", "--device", "cpu"])
    assert as_json
    assert opts.engine_args == ("--walkers", "2", "--hash", "256")
    assert opts.devices == ("cpu",)


def test_uci_mode_wording() -> None:
    assert "default play path" in sc.uci_mode(sc.SelfcheckOptions())
    assert "does NOT cover" in sc.uci_mode(sc.SelfcheckOptions(fast=True))
    assert "operator flags" in sc.uci_mode(sc.SelfcheckOptions(engine_args=("--no-compile",)))


# ── driver / output ───────────────────────────────────────────────────────


def test_main_fails_overall_when_any_component_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def broken() -> str:
        raise sc.CheckFailed("simulated")

    monkeypatch.setattr(sc, "check_python", broken)
    rc = sc.main(["--selfcheck", "--no-uci", "--device", "cpu"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert out[-1].startswith("SELFCHECK FAIL (")
    assert "skipped: checkpoint, syzygy, uci" in out[-1]
    assert any(line.startswith("FAIL") and "python" in line and "simulated" in line for line in out)
    assert any(line.startswith("SKIP") and "uci" in line for line in out)


def test_main_json_matches_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def broken() -> str:
        raise sc.CheckFailed("simulated")

    monkeypatch.setattr(sc, "check_python", broken)
    rc = sc.main(["--json", "--no-uci", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["overall"] == "FAIL"
    by_name = {r["name"]: r for r in payload["results"]}
    assert by_name["python"] == {"name": "python", "status": "FAIL", "detail": "simulated"}
    assert by_name["encoder"]["status"] == "PASS"
    assert by_name["checkpoint"]["status"] == "SKIP"
    assert payload["skipped"] == ["checkpoint", "syzygy", "uci"]
    assert payload["uci_mode"] is None


def test_main_passes_on_this_install(capsys: pytest.CaptureFixture[str]) -> None:
    rc = sc.main(["--no-uci", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.rstrip().splitlines()[-1].startswith("SELFCHECK PASS (")


def test_summary_line_names_mode_and_skips() -> None:
    results = [sc.CheckResult("a", "PASS", ""), sc.CheckResult("b", "SKIP", ""),
               sc.CheckResult("c", "SKIP", "")]
    assert sc.summary_line(results, "fast") == "SELFCHECK PASS (uci: fast; 2 skipped: b, c)"
    assert sc.summary_line(results[:1], None) == "SELFCHECK PASS"


# ── startup gate (refuse instead of silently degrade) ─────────────────────


def test_gate_passes_on_this_install() -> None:
    assert sc.startup_problems() == []


def test_gate_refuses_with_message() -> None:
    buf = io.StringIO()
    rc = sc.enforce_startup_gate(stream=buf, problems_fn=lambda: ["thing is stale"])
    assert rc == 2
    assert "refusing to start" in buf.getvalue()
    assert "thing is stale" in buf.getvalue()


def test_gate_opt_out_warns_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sc.ALLOW_UNVERIFIED_ENV, "1")
    buf = io.StringIO()
    assert sc.enforce_startup_gate(stream=buf, problems_fn=lambda: ["x"]) is None
    assert "WARNING, starting anyway" in buf.getvalue()


def test_gate_flags_zero_feature_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_anti_engine.encoding.cboard_encode as cbe

    real = cbe.encode_cboard

    def zero_features(cb, **kw):
        out = real(cb, **kw).copy()
        out[112:] = 0.0
        return out

    monkeypatch.setattr(cbe, "encode_cboard", zero_features)
    assert any("all zero" in p for p in sc.startup_problems(check_freshness=False))


def test_gate_flags_stale_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sc, "freshness_problems",
        lambda root: ["chess_anti_engine.mcts._mcts_tree is older than x.c"],
    )
    problems = sc.startup_problems(REPO_ROOT)
    assert problems == ["chess_anti_engine.mcts._mcts_tree is older than x.c (stale build)"]


def test_gate_flags_abi_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "required_abi", lambda: 10_000)
    problems = sc.startup_problems(check_freshness=False)
    assert any("ABI_VERSION=" in p and "< required 10000" in p for p in problems)


def _engine_subprocess(prelude: str, *args: str) -> subprocess.CompletedProcess[str]:
    code = prelude + textwrap.dedent("""
        import sys
        from chess_anti_engine.uci.__main__ import main
        sys.argv = ["deepfin", *sys.argv[1:]]
        raise SystemExit(main())
    """)
    env = {k: v for k, v in os.environ.items() if k != sc.ALLOW_UNVERIFIED_ENV}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-c", code, *args], input="uci\nisready\nquit\n",
        capture_output=True, text=True, timeout=120, env=env, cwd=REPO_ROOT, check=False,
    )


def test_engine_refuses_to_start_on_missing_extension(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    proc = _engine_subprocess(
        "import sys\nsys.modules['chess_anti_engine.encoding._features_ext'] = None\n",
        "--checkpoint", str(ckpt), "--device", "cpu", "--no-compile",
    )
    assert proc.returncode == 2, proc.stderr
    assert "uciok" not in proc.stdout
    assert "readyok" not in proc.stdout
    assert "refusing to start" in proc.stderr
    assert "_features_ext is not importable" in proc.stderr


def test_engine_refuses_to_start_on_stale_extension(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    prelude = textwrap.dedent("""
        import chess_anti_engine.uci.selfcheck as sc
        sc.freshness_problems = lambda root, modules=sc.REQUIRED_EXTENSIONS: [
            "chess_anti_engine.mcts._mcts_tree is older than chess_anti_engine/mcts/_mcts_tree.c"
        ]
    """)
    proc = _engine_subprocess(prelude, "--checkpoint", str(ckpt), "--device", "cpu", "--no-compile")
    assert proc.returncode == 2, proc.stderr
    assert "uciok" not in proc.stdout
    assert "stale build" in proc.stderr


def test_engine_starts_on_this_install(tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    proc = _engine_subprocess("", "--checkpoint", str(ckpt), "--device", "cpu", "--no-compile")
    assert proc.returncode == 0, proc.stderr
    assert "uciok" in proc.stdout
    assert "readyok" in proc.stdout


def test_cli_reports_engine_import_failure_as_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine.uci import cli

    monkeypatch.setitem(sys.modules, "chess_anti_engine.uci.__main__", None)
    assert cli.main(["--checkpoint", "x"]) == 2
    assert "refusing to start" in capsys.readouterr().err


def test_cli_selfcheck_does_not_import_the_engine() -> None:
    code = textwrap.dedent("""
        import sys
        from chess_anti_engine.uci import cli
        rc = cli.main(["--selfcheck", "--no-uci", "--device", "cpu", "--json"])
        print("ENGINE_IMPORTED", "chess_anti_engine.uci.__main__" in sys.modules)
        raise SystemExit(rc)
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""},
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ENGINE_IMPORTED False" in proc.stdout


# ── both entry points behave the same on a broken install ────────────────


def _entry_subprocess(prelude: str, entry: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `python -m chess_anti_engine.uci` (entry="module") or the `deepfin`
    console-script target (entry="script") with ``prelude`` applied first."""
    launch = {
        "module": "import runpy\nrunpy.run_module('chess_anti_engine.uci', run_name='__main__')\n",
        "script": "from chess_anti_engine.uci.cli import main\nraise SystemExit(main())\n",
    }[entry]
    code = f"import sys\nsys.argv = ['deepfin', *sys.argv[1:]]\n{prelude}\n{launch}"
    env = {k: v for k, v in os.environ.items() if k != sc.ALLOW_UNVERIFIED_ENV}
    env.update(PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, "-c", code, *args], input="uci\nisready\nquit\n",
        capture_output=True, text=True, timeout=180, env=env, cwd=REPO_ROOT, check=False,
    )


_HIDE_LC0 = "sys.modules['chess_anti_engine.encoding._lc0_ext'] = None"


@pytest.mark.parametrize("entry", ["module", "script"])
def test_entry_points_refuse_missing_core_extension(entry: str) -> None:
    proc = _entry_subprocess(_HIDE_LC0, entry, "--checkpoint", "unused", "--device", "cpu")
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout == ""
    assert "refusing to start" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("entry", ["module", "script"])
def test_entry_points_selfcheck_diagnoses_missing_core_extension(entry: str) -> None:
    proc = _entry_subprocess(_HIDE_LC0, entry, "--selfcheck", "--no-uci", "--device", "cpu")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "_lc0_ext is not importable" in proc.stdout
    assert proc.stdout.rstrip().splitlines()[-1].startswith("SELFCHECK FAIL")


@pytest.mark.parametrize("entry", ["module", "script"])
def test_entry_points_start_on_this_install(entry: str, tmp_path: Path) -> None:
    ckpt = sc._write_tiny_checkpoint(tmp_path)
    proc = _entry_subprocess("", entry, "--checkpoint", str(ckpt), "--device", "cpu",
                             "--no-compile")
    assert proc.returncode == 0, proc.stderr
    assert "uciok" in proc.stdout
    assert "readyok" in proc.stdout


# ── setup.py build-record hook ────────────────────────────────────────────


_CBOARD_H = "chess_anti_engine/encoding/_cboard_impl.h"
_NO_DEPENDS = """def _depends(module: str) -> list[str]:
    return []
"""


def _build_tree(dst: Path, *, depends: bool = True, scripts: bool = True) -> Path:
    """A minimal source tree: setup.py, its two helpers, every C input."""
    tool = sc._load_freshness_tool(REPO_ROOT)
    files = {"setup.py", "chess_anti_engine/ext_build_record.py"}
    if scripts:
        files.add("scripts/check_c_extensions_fresh.py")
    for spec in tool.EXTENSION_SPECS:
        files.update(spec.dependencies)
    for rel in files:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst / rel)
    if not depends:
        # An extension with no `depends` (the state three of them were in
        # before): setuptools then never recompiles it after a header edit.
        src = (dst / "setup.py").read_text(encoding="utf-8")
        start = src.index("def _depends(module: str) -> list[str]:")
        end = src.index("\n\n", start) + 1
        (dst / "setup.py").write_text(src[:start] + _NO_DEPENDS + src[end:], encoding="utf-8")
    return dst


def _setup_build(root: Path, *extra: str) -> list[str]:
    """Run the real `setup.py build_ext --inplace`; return the extensions it recorded."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CAE_EXT_") and k not in ("CFLAGS", "LDFLAGS", "LDSHARED")}
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", *extra],
        cwd=root, env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return sorted(line.split("/")[-1].split(".")[0]
                  for line in proc.stdout.splitlines() if line.startswith("build record: /"))


def _delete_inplace_objects(root: Path) -> None:
    """The lazy "clean rebuild": delete the in-place .so files, keep build/."""
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    gone = [p for p in (root / "chess_anti_engine").rglob(f"*{suffix}") if p.is_file()]
    assert len(gone) == 5, gone
    for p in gone:
        p.unlink()


def _edit_header(root: Path) -> None:
    # distutils compares WHOLE-SECOND mtimes (os.stat()[ST_MTIME]); an edit in
    # the same second as the link would look up to date. A human edit is never
    # that fast, so step past the second rather than weaken the check.
    time.sleep(1.1)
    with (root / _CBOARD_H).open("a", encoding="utf-8") as fh:
        fh.write("/* header-only edit */\n")


_ALL_EXTS = ["_features_ext", "_lc0_ext", "_mcts_tree", "_nnue_ext", "_perft_ext"]
_CBOARD_EXTS = ["_lc0_ext", "_mcts_tree", "_nnue_ext", "_perft_ext"]


def test_setup_header_edit_recompiles_on_a_lazy_rebuild(tmp_path: Path) -> None:
    """Header edit, delete the in-place .so, keep build/, rebuild: the header is
    a declared dependency, so the extensions that include it are recompiled and
    the gate passes on a genuinely fresh binary (real compiler, ~20 s on CPU)."""
    root = _build_tree(tmp_path)
    assert _setup_build(root) == _ALL_EXTS
    assert sc.freshness_problems(root) == []
    _edit_header(root)
    assert any("stale" in p or _CBOARD_H in p for p in sc.freshness_problems(root))
    _delete_inplace_objects(root)
    assert _setup_build(root) == _CBOARD_EXTS  # recompiled; _features_ext does not include it
    assert sc.freshness_problems(root) == []
    assert _setup_build(root) == []  # nothing compiled, nothing (re)recorded


def test_setup_copy_without_compile_gets_no_record(tmp_path: Path) -> None:
    """Without `depends`, the same lazy rebuild compiles nothing and copies the
    old object out of build/. No record may certify it: the gate must refuse,
    with the old record left in place and with no record at all (mtime
    fallback), and the forcing command in the refusal must clear it."""
    root = _build_tree(tmp_path, depends=False)
    assert _setup_build(root) == _ALL_EXTS
    _edit_header(root)
    _delete_inplace_objects(root)
    assert _setup_build(root) == []  # copied from build/, not compiled
    problems = sc.freshness_problems(root)
    assert any("_lc0_ext" in p and _CBOARD_H in p for p in problems), problems
    for record in (root / "chess_anti_engine").rglob("*.build-record.json"):
        record.unlink()
    problems = sc.freshness_problems(root)
    assert any("_lc0_ext is older than" in p for p in problems), problems
    assert sc.FORCE_REBUILD_COMMAND in sc.REBUILD_HINT
    cmd = sc.FORCE_REBUILD_COMMAND.split()
    assert cmd[:2] == ["python", "setup.py"]
    assert _setup_build(root, *cmd[4:]) == _ALL_EXTS
    assert sc.freshness_problems(root) == []


def test_setup_without_scripts_builds_without_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tree without scripts/: setup.py still loads (no depends, no records)."""
    import runpy

    import setuptools

    root = _build_tree(tmp_path, scripts=False)
    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    monkeypatch.chdir(root)
    ns = runpy.run_path(str(root / "setup.py"))
    assert ns["_FRESHNESS_TOOL"] is None
    assert len(captured["ext_modules"]) == 5
    assert all(ext.depends == [] for ext in captured["ext_modules"])


def test_setup_depends_come_from_the_freshness_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runpy

    import setuptools

    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    runpy.run_path(str(REPO_ROOT / "setup.py"))
    tool = sc._load_freshness_tool(REPO_ROOT)
    table = {spec.module: list(spec.dependencies) for spec in tool.EXTENSION_SPECS}
    got = {ext.name: list(ext.depends) for ext in captured["ext_modules"]}
    assert got == table
