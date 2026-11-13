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


def test_extensions_missing_symbol_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType("chess_anti_engine.encoding._features_ext")
    setattr(stub, "compute_extra_features", lambda *a: None)
    monkeypatch.setitem(sys.modules, "chess_anti_engine.encoding._features_ext", stub)
    res = _run(sc.check_extensions)
    assert res.status == "FAIL"
    assert "lacks compute_relation_matrices" in res.detail


# ── freshness (stale .so) ─────────────────────────────────────────────────


def _fake_tree(tmp_path: Path, *, so_mtime: int, src_mtime: int) -> Path:
    """A checkout skeleton with the real freshness tool and every listed file."""
    tool = sc._load_freshness_tool(REPO_ROOT)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "check_c_extensions_fresh.py", tmp_path / "scripts")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    for spec in tool.EXTENSION_SPECS:
        for dep in spec.dependencies:
            p = tmp_path / dep
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
            os.utime(p, ns=(src_mtime, src_mtime))
        base = tmp_path.joinpath(*spec.module.split("."))
        out = base.with_name(base.name + suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        os.utime(out, ns=(so_mtime, so_mtime))
    return tmp_path


def test_freshness_fresh_tree_passes(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    res = _run(lambda: sc.check_freshness(root))
    assert res.status == "PASS", res.detail


def test_freshness_stale_so_fails(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    src = root / "chess_anti_engine" / "encoding" / "_features_impl.h"
    os.utime(src, ns=(3_000_000_000, 3_000_000_000))
    res = _run(lambda: sc.check_freshness(root))
    assert res.status == "FAIL"
    assert "_features_ext is older than chess_anti_engine/encoding/_features_impl.h" in res.detail


def test_freshness_ignores_extensions_the_engine_does_not_load(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, so_mtime=2_000_000_000, src_mtime=1_000_000_000)
    src = root / "chess_anti_engine" / "nnue" / "_nnue_ext.c"
    os.utime(src, ns=(3_000_000_000, 3_000_000_000))
    assert sc.freshness_problems(root) == []


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
    assert "architecture mismatch" in res.detail
    assert dropped in res.detail


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


def test_uci_exchange_real_engine_passes() -> None:
    res = _run(lambda: sc.check_uci(None, ["cpu"]))
    assert res.status == "PASS", res.detail
    assert "bestmove" in res.detail


_FAKE_ENGINE = textwrap.dedent("""
    import sys
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            print("id name fake"); print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd.startswith("go"):
            print("bestmove {move}", flush=True)
        elif cmd == "quit":
            break
""")


def test_uci_illegal_bestmove_fails() -> None:
    script = _FAKE_ENGINE.format(move="e2e5")
    res = _run(lambda: sc.check_uci("unused", ["cpu"],
                                    command=lambda ck, dev: [sys.executable, "-c", script]))
    assert res.status == "FAIL"
    assert "not legal" in res.detail


def test_uci_legal_bestmove_from_fake_engine_passes() -> None:
    script = _FAKE_ENGINE.format(move="e2e4")
    res = _run(lambda: sc.check_uci("unused", ["cpu"],
                                    command=lambda ck, dev: [sys.executable, "-c", script]))
    assert res.status == "PASS", res.detail


def test_engine_command_passes_devices() -> None:
    assert sc.engine_command("c", ["cuda:0", "cuda:1"])[-2:] == ["--devices", "cuda:0,cuda:1"]
    assert sc.engine_command("c", ["cpu"])[-2:] == ["--device", "cpu"]
    assert "--no-compile" in sc.engine_command("c", [])


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
    assert out[-1] == "SELFCHECK FAIL"
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


def test_main_passes_on_this_install(capsys: pytest.CaptureFixture[str]) -> None:
    rc = sc.main(["--no-uci", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.rstrip().endswith("SELFCHECK PASS")


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
