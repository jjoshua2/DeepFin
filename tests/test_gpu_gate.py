"""GPU launch gate (chess_anti_engine/utils/gpu_gate.py): every clause fires, without the real device.

Every probe is replaced: nvidia-smi by a callable (or a fake binary on PATH for the CLI), /proc by a
fake tree, the GPU device set and fd stats by seams, HOME by a temp dir. No test imports torch.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.utils import gpu_gate as g

REPO = Path(__file__).resolve().parents[1]
UID = 4242
SELF = 500
GPU_RDEV = 0xBEEF


class FakeSmi:
    def __init__(self, memory=(2000,), util=(0,), fail: str | None = None, raw: dict | None = None):
        self.memory, self.util, self.fail, self.raw = memory, util, fail, raw or {}
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        kind = "memory" if "memory.used" in args[0] else "util"
        if self.fail == kind or self.fail == "all":
            raise RuntimeError(f"nvidia-smi {kind} exited 9")
        if kind in self.raw:
            return self.raw[kind]
        return "\n".join(str(v) for v in (self.memory if kind == "memory" else self.util)) + "\n"


class Proc:
    """A fake /proc tree; fd entries are files whose stat is given by `fd_stats`."""

    def __init__(self, root: Path):
        self.root = root
        self.fd_stats: dict[Path, object] = {}
        self.add(1, 0, ["init"], uid=0)
        self.add(400, 1, ["bash"])
        self.add(SELF, 400, ["python3", "scripts/arena_standard.py"])

    def add(self, pid, ppid, argv, *, uid=UID, environ=(), gpu_fd=False, unreadable=False):
        d = self.root / str(pid)
        (d / "fd").mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nPPid:\t{ppid}\n")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "environ").write_bytes(b"\0".join(e.encode() for e in environ) + (b"\0" if environ else b""))
        fd0 = d / "fd" / "0"
        fd0.write_text("")
        if gpu_fd:
            fd3 = d / "fd" / "3"
            fd3.write_text("")
            self.fd_stats[fd3] = SimpleNamespace(st_mode=stat.S_IFCHR | 0o600, st_rdev=GPU_RDEV)
        if unreadable:
            self.fd_stats[fd0] = PermissionError("denied")
        return d

    def fd_stat(self, path):
        v = self.fd_stats.get(path)
        if isinstance(v, BaseException):
            raise v
        if v is not None:
            return v
        return os.stat(path)


@pytest.fixture
def world(tmp_path):
    proc = Proc(tmp_path / "proc")
    home = tmp_path / "home"
    home.mkdir()
    smi = FakeSmi()

    def seams(**kw):
        base: dict[str, Any] = {"proc_root": proc.root, "home": home, "uid": UID, "pid": SELF, "nvidia_smi": smi,
                    "sleep": lambda s: None, "gpu_rdevs": lambda: {GPU_RDEV}, "fd_stat": proc.fd_stat}
        base.update(kw)
        return g.Seams(**base)

    return SimpleNamespace(proc=proc, home=home, smi=smi, seams=seams)


ENV: dict[str, str] = {}  # GPU visible, no override, no marker


def decide(world, env=None, **kw):
    seams = kw.pop("seams", None) or world.seams()
    return g.check("arena_standard", env=ENV if env is None else env, seams=seams, **kw)


def test_idle_gpu_passes_and_names_the_marker(world):
    d = decide(world)
    assert d.status == "OK"
    assert d.code == 0
    assert d.marker == f"{SELF}:arena_standard"


def test_memory_above_default_ceiling_refuses_and_prints_value_and_ceiling(world):
    world.smi.memory = (g.DEFAULT_MEMORY_CEILING_MIB + 1,)
    d = decide(world)
    assert d.code == g.EXIT_REFUSED
    assert str(g.DEFAULT_MEMORY_CEILING_MIB + 1) in d.reasons[0]
    assert str(g.DEFAULT_MEMORY_CEILING_MIB) in d.reasons[0]
    world.smi.memory = (g.DEFAULT_MEMORY_CEILING_MIB,)
    assert decide(world).code == 0


def test_default_ceiling_is_the_banked_idle_maximum_plus_margin():
    assert g.DEFAULT_MEMORY_CEILING_MIB == 3382 + 512


def test_baseline_file_sets_the_ceiling(world):
    f = world.home / "chess-artifacts/operations/GPU_IDLE_BASELINE"
    f.parent.mkdir(parents=True)
    f.write_text("idle_mib=2181 measured 2026-11-22\n")
    world.smi.memory = (2181 + 513,)
    d = decide(world)
    assert d.code == 8
    assert "idle_mib=2181" in d.reasons[0]
    world.smi.memory = (2181 + 512,)
    assert decide(world).code == 0


@pytest.mark.parametrize("text", ["idle_mib=abc", "no token", "idle_mib=5000"])
def test_malformed_baseline_refuses(world, text):
    f = world.home / "chess-artifacts/operations/GPU_IDLE_BASELINE"
    f.parent.mkdir(parents=True)
    f.write_text(text)
    assert decide(world).code == 8


def test_utilisation_at_ten_percent_refuses_nine_passes(world):
    world.smi.util = (9,)
    assert decide(world).code == 0
    world.smi.util = (10,)
    d = decide(world)
    assert d.code == 8
    assert "utilisation reached 10" in d.reasons[0]


@pytest.mark.parametrize("smi", [FakeSmi(fail="memory"), FakeSmi(fail="util"),
                                 FakeSmi(raw={"memory": "N/A\n"}), FakeSmi(raw={"util": ""})])
def test_every_nvidia_smi_failure_refuses(world, smi):
    d = decide(world, seams=world.seams(nvidia_smi=smi))
    assert d.code == 8
    assert any("probe failed" in r for r in d.reasons)


def test_listed_sibling_argv_refuses(world):
    world.proc.add(600, 400, ["python3", "-u", "scripts/match_vs_uci.py", "--engine-a", "x"])
    d = decide(world)
    assert d.code == 8
    assert "pid 600 (match_vs_uci.py)" in d.reasons[0]


def test_live_run_module_refuses(world):
    world.proc.add(601, 1, ["python3", "-m", "chess_anti_engine.run", "--mode", "tune"])
    d = decide(world)
    assert d.code == 8
    assert "-m chess_anti_engine.run" in d.reasons[0]


def test_cpu_only_environ_exempts_the_argv_clause_only(world):
    world.proc.add(602, 400, ["python3", "scripts/arena_standard.py"], environ=["CUDA_VISIBLE_DEVICES="])
    d = decide(world)
    assert d.code == 0
    assert any("exempt" in n for n in d.notes)
    world.proc.add(603, 400, ["python3", "scripts/arena_standard.py"], environ=["CUDA_VISIBLE_DEVICES="], gpu_fd=True)
    d = decide(world)
    assert d.code == 8
    assert "holds the GPU device: pid 603" in d.reasons[0]


def test_duplicate_cuda_visible_entry_is_not_exempt(world):
    world.proc.add(604, 400, ["python3", "scripts/arena_standard.py"],
                   environ=["CUDA_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES=0"])
    assert decide(world).code == 8


def test_unnamed_gpu_device_holder_refuses(world):
    world.proc.add(605, 1, ["some_unlisted_binary"], gpu_fd=True)
    d = decide(world)
    assert d.code == 8
    assert "pid 605" in d.reasons[0]


def test_other_users_are_not_scanned(world):
    world.proc.add(606, 1, ["python3", "-m", "chess_anti_engine.run"], uid=7, gpu_fd=True)
    assert decide(world).code == 0


def test_unreadable_fds_refuse_except_allowlisted_agents(world):
    world.proc.add(607, 1, ["/usr/bin/ssh-agent"], unreadable=True)
    assert decide(world).code == 0
    world.proc.add(608, 1, ["/usr/bin/other"], unreadable=True)
    d = decide(world)
    assert d.code == 8
    assert "cannot read fd" in d.reasons[0]


def test_ancestors_and_descendants_are_excluded(world):
    world.proc.add(400, 1, ["bash", "scripts/tcec_autotune.sh"], gpu_fd=True)  # the caller's parent
    world.proc.add(700, SELF, ["python3", "scripts/bench_uci_engine.py"], gpu_fd=True)  # own child
    assert decide(world).code == 0


def test_sibling_copy_of_the_same_launcher_refuses(world):
    world.proc.add(701, 400, ["python3", "scripts/arena_standard.py"])
    assert decide(world).code == 8


def test_no_gpu_device_with_nvidia_smi_present_refuses(world):
    d = decide(world, seams=world.seams(gpu_rdevs=set))
    assert d.code == 8
    assert "cannot identify the GPU device" in d.reasons[0]


def test_no_gpu_host_passes_without_probing(world, monkeypatch):
    monkeypatch.setattr(g.shutil, "which", lambda name: None)
    d = decide(world, seams=world.seams(nvidia_smi=None, gpu_rdevs=set))
    assert d.status == "OK no-gpu"


def test_nvidia_smi_absent_but_device_present_refuses(world, monkeypatch):
    monkeypatch.setattr(g.shutil, "which", lambda name: None)
    monkeypatch.setattr(g, "_run_smi", lambda args: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    d = decide(world, seams=world.seams(nvidia_smi=None))
    assert d.code == 8


def test_cpu_only_caller_passes_without_probing(world):
    smi = FakeSmi(fail="all")
    d = decide(world, env={"CUDA_VISIBLE_DEVICES": ""}, seams=world.seams(nvidia_smi=smi))
    assert d.status == "OK cpu-only"
    assert smi.calls == 0
    assert d.marker is None
    d = decide(world, device="cpu", seams=world.seams(nvidia_smi=smi))
    assert d.status == "OK cpu-only"
    assert smi.calls == 0


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "auto", None])
def test_gpu_devices_are_probed(world, device):
    world.smi.memory = (30000,)
    assert decide(world, device=device).code == 8


def test_override_passes_a_busy_gpu_and_records_what_it_saw(world):
    world.smi.memory = (30000,)
    d = decide(world, allow_reason="owner engine check")
    assert d.status == "OK allowed"
    assert d.code == 0
    assert any("owner engine check" in n and "30000" in n for n in d.notes)
    assert d.marker == f"{SELF}:arena_standard"


def test_override_passes_even_when_probes_fail(world):
    d = decide(world, allow_reason="pause window", seams=world.seams(nvidia_smi=FakeSmi(fail="all")))
    assert d.status == "OK allowed"


def test_env_twin_override(world):
    world.smi.memory = (30000,)
    assert decide(world, env={g.ENV_ALLOW: "wrapper reason"}).status == "OK allowed"


@pytest.mark.parametrize(("kw", "env"), [({"allow_reason": ""}, {}), ({"allow_reason": "   "}, {}), ({}, {g.ENV_ALLOW: ""})])
def test_empty_override_reason_is_a_usage_error(world, kw, env):
    with pytest.raises(g.GateUsageError):
        decide(world, env=env, **kw)


def test_children_of_a_gated_parent_pass_inherited_and_forgery_does_not(world):
    # parent P gated; sibling 1 runs on the GPU; sibling 2 (us) passes as inherited
    world.proc.add(800, 400, ["python3", "scripts/bench_multi_worker.py"])
    world.proc.add(801, 800, ["python3", "-m", "chess_anti_engine.worker"], gpu_fd=True)
    world.proc.add(802, 800, ["python3", "-m", "chess_anti_engine.worker"])
    env = {g.ENV_PASSED: "800:bench_multi_worker"}
    d = decide(world, env=env, seams=world.seams(pid=802))
    assert d.status == "OK inherited"
    # an unrelated process carrying the same (forged) variable gets the full check and refuses
    world.proc.add(900, 1, ["python3", "scripts/match_checkpoints.py"])
    d = decide(world, env=env, seams=world.seams(pid=900))
    assert d.code == 8
    assert any("does not name an ancestor" in n for n in d.notes)


def test_for_pid_must_be_an_ancestor(world):
    with pytest.raises(g.GateUsageError):
        decide(world, for_pid=999)
    assert decide(world, for_pid=400).marker == "400:arena_standard"


@pytest.mark.parametrize(("argv", "hit"), [
    (["python3", "-m", "chess_anti_engine.run"], "-m chess_anti_engine.run"),
    (["python3", "-mchess_anti_engine.worker"], "-m chess_anti_engine.worker"),
    (["python3", "-m", "chess_anti_engine.utils.gpu_gate", "--tool", "x"], None),
    (["python3", "-c", "open('scripts/arena_standard.py')"], None),
    (["/usr/bin/python3.12", "-X", "dev", "scripts/arena_standard.py"], "arena_standard.py"),
    (["bash", "scripts/train.sh", "start"], "train.sh"),
    (["bash", "-c", "sha256sum scripts/arena_standard.py"], None),
    (["/opt/lc0/lc0", "--backend=cuda"], "lc0"),
    (["dotnet", "Ceres.dll"], "Ceres.dll"),
    (["vim", "scripts/arena_standard.py"], None),
])
def test_argv_entry_point(argv, hit):
    assert g.argv_entry_point(argv) == hit


# ---------------------------------------------------------------- CLI and wrapper (fake nvidia-smi on PATH)


def fake_bin(tmp_path: Path, memory: int, util: int = 0) -> Path:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    smi = b / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu=memory.used) echo {memory};; *) echo {util};; esac\n")
    smi.chmod(0o755)
    return b


def cli_env(tmp_path: Path, memory: int, **extra) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAE_")}
    env.update(PATH=f"{fake_bin(tmp_path, memory)}:{env.get('PATH', '')}", HOME=str(tmp_path),
               CUDA_VISIBLE_DEVICES="", PYTORCH_NVML_BASED_CUDA_CHECK="1", PYTHONPATH=str(REPO))
    env[g.ENV_TEST] = "1"  # probe despite CUDA_VISIBLE_DEVICES= (empty); exit 97 after a pass
    env.update(extra)
    return env


@pytest.mark.parametrize("cmd", [[sys.executable, "-m", "chess_anti_engine.utils.gpu_gate"],
                                 ["bash", str(REPO / "scripts/gpu_gate.sh")]])
def test_cli_and_wrapper_exit_codes(tmp_path, cmd):
    busy = cli_env(tmp_path, 30000)
    r = subprocess.run([*cmd, "--tool", "t"], env=busy, capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == 8, r.stderr
    assert "REFUSED: GPU memory.used 30000 MiB" in r.stderr, r.stderr
    r = subprocess.run([*cmd, "--tool", "t", "--allow-busy-gpu", "cli test"], env=busy,
                       capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == g.EXIT_TEST_PASSED, r.stderr
    assert "BUSY-GPU OVERRIDE" in r.stderr
    assert "cli test" in r.stderr
    assert r.stdout.startswith(f"{g.ENV_PASSED}=")
    r = subprocess.run([*cmd, "--tool", "t", "--allow-busy-gpu", ""], env=busy, capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == 2
    r = subprocess.run([*cmd, "--tool", "t", "--for-pid", "999999999"], env=busy, capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == 2
    assert "not an ancestor" in r.stderr


def test_cli_without_test_seam_short_circuits_cpu_only(tmp_path):
    env = cli_env(tmp_path, 30000)
    env.pop(g.ENV_TEST)
    r = subprocess.run([sys.executable, "-m", "chess_anti_engine.utils.gpu_gate", "--tool", "t"],
                       env=env, capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == 0
    assert "OK cpu-only" in r.stderr
    assert r.stdout == ""


def test_gate_module_is_import_light():
    r = subprocess.run([sys.executable, "-c", "import sys, chess_anti_engine.utils.gpu_gate as m; "
                        "print('torch' in sys.modules, 'numpy' in sys.modules)"],
                       env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, check=False, timeout=60)
    assert r.stdout.strip() == "False False", r.stderr
