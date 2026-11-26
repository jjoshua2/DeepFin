"""Second #1518 review round: marker forgery, override marker, exact cpu skip, unreadable fd dir,
empty cmdline. Seams only; no torch, no real device."""

from __future__ import annotations

import stat
from types import SimpleNamespace

from typing import Any

import pytest

from chess_anti_engine.utils import gpu_gate as g
from tests.test_gpu_gate import GPU_RDEV, SELF, UID, FakeSmi, Proc


@pytest.fixture
def world(tmp_path):
    """The same fake world as tests/test_gpu_gate.py's fixture."""
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

BUSY = (30000,)


@pytest.mark.parametrize("head", ["1", str(SELF), "0"])
def test_marker_naming_init_self_or_zero_is_ignored_and_the_full_check_runs(world, head):
    world.smi.memory = BUSY
    d = g.check("t", env={g.ENV_PASSED: f"{head}:x"}, seams=world.seams())
    assert d.status == "REFUSED", (head, d)
    assert any("does not name a strict non-init ancestor" in n for n in d.notes)


def test_marker_naming_a_real_parent_still_inherits(world):
    world.smi.memory = BUSY
    d = g.check("t", env={g.ENV_PASSED: "400:parent"}, seams=world.seams())
    assert d.status == "OK inherited"
    assert d.marker == "400:parent"
    assert not any(n.startswith("overridden:") for n in d.notes)


@pytest.mark.parametrize("for_pid", [1, 0, SELF])
def test_for_pid_init_zero_or_self_is_a_usage_error(world, for_pid):
    with pytest.raises(g.GateUsageError, match="strict non-init ancestor"):
        g.check("t", env={}, for_pid=for_pid, seams=world.seams())


def test_for_pid_real_parent_names_the_marker(world):
    assert g.check("t", env={}, for_pid=400, seams=world.seams()).marker == "400:t"


def test_override_is_carried_in_the_marker_and_shown_loudly_by_the_child(world, capsys):
    world.smi.memory = BUSY
    parent = g.check("t", env={}, allow_reason="pause window", seams=world.seams(pid=400))
    assert parent.status == "OK allowed"
    assert parent.marker == "400:t:allowed"
    child = g.check("t2", env={g.ENV_PASSED: parent.marker}, seams=world.seams())
    assert child.status == "OK inherited"
    assert child.marker == parent.marker
    g._emit("t2", child)
    err = capsys.readouterr().err
    assert "BUSY-GPU OVERRIDE" in err
    assert "inherited from the parent gate" in err
    world.smi.memory = (2000,)
    clean = g.check("t", env={}, seams=world.seams(pid=400))  # an idle pass carries no suffix
    assert clean.marker == "400:t"


@pytest.mark.parametrize(("device", "skips"), [("cpu", True), ("0", False), ("gpu", False), ("GPU", False),
                                          ("mps", False), ("cpu:0", False), ("cuda:1", False), ("auto", False)])
def test_only_an_exact_cpu_device_skips(world, device, skips):
    world.smi.memory = BUSY
    d = g.check("t", env={}, device=device, seams=world.seams())
    assert (d.status == "OK cpu-only") is skips, (device, d)
    if not skips:
        assert d.status == "REFUSED"


def test_wholly_unreadable_fd_directory_refuses(world):
    """Reviewer mutant G9 survived: a /proc/<pid>/fd that cannot be listed must refuse."""
    p = world.proc.add(702, 1, ["node", "something"])
    (p / "fd").chmod(0)
    try:
        d = g.check("t", env={}, seams=world.seams())
    finally:
        (p / "fd").chmod(0o755)
    assert d.status == "REFUSED"
    assert any("cannot read the fds of pid 702" in r for r in d.reasons), d.reasons


def test_process_with_empty_cmdline_is_still_fd_scanned(world):
    p = world.proc.add(703, 1, ["placeholder"])
    (p / "cmdline").write_bytes(b"")
    fd3 = p / "fd" / "3"
    fd3.write_text("")
    world.proc.fd_stats[fd3] = SimpleNamespace(st_mode=stat.S_IFCHR | 0o600, st_rdev=GPU_RDEV)
    d = g.check("t", env={}, seams=world.seams())
    assert d.status == "REFUSED"
    assert any("pid 703" in r and "<empty cmdline>" in r for r in d.reasons), d.reasons


def test_zombie_with_empty_cmdline_and_unreadable_fds_is_skipped(world):
    """A real zombie (seen live: a defunct sandbox helper) lists no fds and its fd dir reads EACCES."""
    p = world.proc.add(705, 1, ["placeholder"])
    (p / "cmdline").write_bytes(b"")
    status = (p / "status").read_text()
    (p / "status").write_text(status.replace("Name:\tx\n", "Name:\tx\nState:\tZ (zombie)\n"))
    (p / "fd").chmod(0)
    try:
        assert g.check("t", env={}, seams=world.seams()).passed
        (p / "status").write_text(status.replace("Name:\tx\n", "Name:\tx\nState:\tS (sleeping)\n"))
        assert g.check("t", env={}, seams=world.seams()).status == "REFUSED"  # not a zombie: unreadable fds refuse
    finally:
        (p / "fd").chmod(0o755)


def test_vanished_cmdline_is_skipped(world):
    p = world.proc.add(704, 1, ["python3", "scripts/arena_standard.py"])
    (p / "cmdline").unlink()
    assert g.check("t", env={}, seams=world.seams()).passed
