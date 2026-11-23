"""Refuse an ACCIDENTAL GPU launch beside a running GPU job.

Purpose. On 2026-11-21 an agent-run launcher reached the GPU beside a live job, twice. GPU
out-of-memory in the live job is the failure that has actually happened. A launcher calls this gate
before it starts any GPU work: before importing torch where it can, and always before the first
child process that uses the GPU.

This is NOT a security control. It cannot authenticate anyone. It is designed to be cheap to pass
deliberately and impossible to pass by accident. There is exactly ONE override: the per-invocation
flag ``--allow-busy-gpu REASON``, or its environment twin ``CAE_ALLOW_BUSY_GPU=REASON`` for
wrappers that cannot pass flags. The reason must be non-empty, and the gate prints one loud stderr
line with the reason and what it saw. The owner's engine checks beside training, and an arena inside
an operator pause window, both pass the flag with a reason. Agents' briefs forbid them the flag.
That is a rule, not a mechanism.

The rule is the operator GPU check's rule (``operator_gpu_check.sh`` / ``gpu_guard.sh`` v3 /
``gpu_scan.sh`` v4 in the experiment banks), with one change: no mandatory baseline file. All
clauses run, and every probe fails closed:

* **Memory.** Every ``nvidia-smi --query-gpu=memory.used`` line is an integer <= the ceiling.
  The ceiling is ``idle_mib + 512`` from ``~/chess-artifacts/operations/GPU_IDLE_BASELINE``
  (``idle_mib=<int>``, <= 4,096) when that file exists, else ``DEFAULT_MEMORY_CEILING_MIB``
  (derivation at the constant). ``--query-compute-apps`` is NOT used: under WSL2 it does not
  list another process tree, so a launcher that checked only it was blind to the live run.
* **Utilisation.** ``nvidia-smi --query-gpu=utilization.gpu``: 5 samples one second apart,
  every line an integer, maximum < 10 %.
* **Process scan** (same uid; the caller's ancestors and descendants are excluded, so a sibling
  copy of the same launcher still refuses):
  - its argv is a GPU entry point (:mod:`chess_anti_engine.utils.gpu_entry_points`; the
    list is generated and kept in the repo, and ``tests/test_gpu_gate_coverage.py`` holds it
    to the code). A process whose exec environment holds exactly ``CUDA_VISIBLE_DEVICES=``
    (empty) is exempt from this clause only;
  - it holds the GPU device open: any fd that stats to a character device with the rdev of
    ``/dev/dxg`` (WSL2) or ``/dev/nvidia*``. This catches any CUDA process once its context
    exists, whatever its name;
  - its fds cannot be read, so the device clause cannot be checked
    (``/usr/bin/ssh-agent`` and ``/usr/bin/gpg-agent`` are allowlisted).

Memory and utilisation cannot see a job that is still loading. During that phase the process scan
is the defence.

**Multi-process launches.** A launcher that starts K GPU processes gates ONCE in the parent. On a
pass, the gate exports ``CAE_GPU_GATE_PASSED=<pid>:<tool>``. A child whose ancestor chain (read from
``/proc``, not trusted from the variable) contains that pid passes with ``OK inherited``. A forged or
stale value in an unrelated process is ignored, and the full check runs.

**No-GPU host.** With no ``/dev/dxg`` / ``/dev/nvidia*`` character device AND no ``nvidia-smi``
on PATH, there is no GPU and so no job to protect: ``OK no-gpu`` (CI, CPU-only hosts). Either one
present without the other is a broken probe and refuses.

**CPU short-circuit.** A process whose own environment holds exactly ``CUDA_VISIBLE_DEVICES=``
(empty), or whose resolved device is cpu, cannot reach the card. It passes with ``OK cpu-only``
without probing.

Exit codes (CLI and :func:`require_gpu_idle`): 0 pass, 8 refused (with the reason), 2 usage
error (an empty override reason, a bad ``--for-pid``). Live-run infrastructure (``scripts/train.sh``
and everything it starts) never calls this gate. The gate counts those processes as busy, which is
the point.

Test seam ``CAE_GPU_GATE_TEST=1``. It only makes the gate stricter or stops the program: it probes
even under ``CUDA_VISIBLE_DEVICES=`` (empty) or a cpu device, reports whether torch is imported at the decision,
and exits 97 right after a pass. The take-effect tests use it so that no gated tool runs past its
gate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.gpu_entry_points import GPU_ENTRY_POINT_NAMES

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 8
EXIT_TEST_PASSED = 97

FLAG = "--allow-busy-gpu"
ENV_ALLOW = "CAE_ALLOW_BUSY_GPU"
ENV_PASSED = "CAE_GPU_GATE_PASSED"
ENV_TEST = "CAE_GPU_GATE_TEST"
BASELINE_FILE = "~/chess-artifacts/operations/GPU_IDLE_BASELINE"

IDLE_MARGIN_MIB = 512
BASELINE_SANITY_CAP_MIB = 4096
# Default memory ceiling = the largest idle ``memory.used`` banked on this machine + the operator
# rule's 512 MiB margin. Measured, not round. Idle readings banked so far:
#   - 3,382 MiB: gpu_guard.sh v3 header (rl-run1-gen2 bank), 30 one-second samples, min = max,
#     utilisation 0-1 %, recorded as "this machine's IDLE baseline"; mem_20261122/RESULT.md
#     (rl-run1-root-full-width.opdiag) calls ~3,382 the "usual idle";
#   - 2026-11-22: 2,398 and 2,181 MiB (the gen-2 GPU_RELEASED_BY_OPERATOR lines); 2,145-2,169
#     (opdiag chunk_meas_20261122/log.txt); 1,021-1,276 (mem_wide*_20261122); 1,138 (mem_20261122).
# The idle level moves with the Windows desktop (WSL2), so the default takes the maximum, 3,382,
# giving 3,894. The live run alone holds tens of GB (22,104 MiB recorded 2026-11-21), far above
# this ceiling. Write GPU_IDLE_BASELINE to tighten it on a given day.
OBSERVED_IDLE_MAX_MIB = 3382
DEFAULT_MEMORY_CEILING_MIB = OBSERVED_IDLE_MAX_MIB + IDLE_MARGIN_MIB
UTIL_SAMPLES = 5
UTIL_LIMIT_PCT = 10  # refuse at >= this
FD_UNREADABLE_ALLOW = frozenset({"/usr/bin/ssh-agent", "/usr/bin/gpg-agent"})
# The gate's own CLI is a chess_anti_engine module but is not GPU work.
_NOT_GPU_MODULES = frozenset({"chess_anti_engine.utils.gpu_gate"})
_SMI_TIMEOUT_S = 20


class GateUsageError(ValueError):
    """An override given with an empty reason, or another invalid gate argument."""


@dataclass
class Seams:
    """Where the gate reads the world; tests replace these, production uses the defaults."""

    proc_root: Path = Path("/proc")
    dev_root: Path = Path("/dev")
    home: Path | None = None
    uid: int | None = None
    pid: int | None = None
    nvidia_smi: Callable[[Sequence[str]], str] | None = None
    sleep: Callable[[float], None] = time.sleep
    gpu_rdevs: Callable[[], set[int]] | None = None  # default: gpu_device_rdevs(dev_root)
    fd_stat: Callable[[Path], Any] = os.stat  # anything with st_mode / st_rdev


@dataclass
class GateDecision:
    code: int
    status: str  # OK | OK cpu-only | OK no-gpu | OK inherited | OK allowed | REFUSED
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    marker: str | None = None  # CAE_GPU_GATE_PASSED value to export on a pass

    @property
    def passed(self) -> bool:
        return self.code == EXIT_OK


# ---------------------------------------------------------------- probes


def _run_smi(args: Sequence[str]) -> str:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} exited {out.returncode}")
    return out.stdout


def _int_lines(text: str, what: str) -> list[int]:
    lines = [ln.strip() for ln in text.replace("\r", "").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"nvidia-smi printed no {what} line")
    values = []
    for ln in lines:
        if not ln.isdigit():
            raise RuntimeError(f"unreadable GPU {what} line {ln!r}")
        values.append(int(ln))
    return values


def memory_ceiling(home: Path) -> tuple[int, str]:
    """Return (ceiling MiB, where it came from). Raises RuntimeError on a malformed baseline file."""
    path = home / BASELINE_FILE.removeprefix("~/")
    if not path.exists():
        return DEFAULT_MEMORY_CEILING_MIB, (
            f"default: max banked idle {OBSERVED_IDLE_MAX_MIB} + {IDLE_MARGIN_MIB}; no {BASELINE_FILE}"
        )
    tokens = [t for t in path.read_text().split() if t.startswith("idle_mib=")]
    value = tokens[0].removeprefix("idle_mib=") if tokens else ""
    if not value.isdigit():
        raise RuntimeError(f"{BASELINE_FILE} carries no integer idle_mib=<MiB> ({value!r})")
    idle = int(value)
    if idle > BASELINE_SANITY_CAP_MIB:
        raise RuntimeError(f"{BASELINE_FILE} idle_mib {idle} > {BASELINE_SANITY_CAP_MIB} (sanity cap)")
    return idle + IDLE_MARGIN_MIB, f"{BASELINE_FILE} idle_mib={idle} + {IDLE_MARGIN_MIB}"


def probe_memory(smi: Callable[[Sequence[str]], str], home: Path) -> tuple[str | None, str]:
    """(reason or None, what it saw)."""
    try:
        ceiling, source = memory_ceiling(home)
        used = _int_lines(smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]), "memory")
    except Exception as exc:  # every probe fails closed
        return f"memory probe failed: {exc}", "memory: unreadable"
    seen = f"memory.used {'/'.join(map(str, used))} MiB, ceiling {ceiling} MiB ({source})"
    worst = max(used)
    if worst > ceiling:
        return f"GPU memory.used {worst} MiB > ceiling {ceiling} MiB ({source})", seen
    return None, seen


def probe_utilisation(smi: Callable[[Sequence[str]], str], sleep: Callable[[float], None]) -> tuple[str | None, str]:
    peak = 0
    try:
        for i in range(UTIL_SAMPLES):
            peak = max(peak, *_int_lines(smi(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]), "utilisation"))
            if i < UTIL_SAMPLES - 1:
                sleep(1.0)
    except Exception as exc:
        return f"utilisation probe failed: {exc}", "utilisation: unreadable"
    seen = f"utilisation max {peak} % over {UTIL_SAMPLES} samples"
    if peak >= UTIL_LIMIT_PCT:
        return f"GPU utilisation reached {peak} % ({UTIL_SAMPLES} samples; needs < {UTIL_LIMIT_PCT} %)", seen
    return None, seen


def argv_entry_point(argv: Sequence[str]) -> str | None:
    """The matched GPU entry point for this argv, else None (gpu_scan.sh v4 argv semantics)."""
    if not argv:
        return None
    a0 = os.path.basename(argv[0])
    if a0 in GPU_ENTRY_POINT_NAMES:
        return a0
    rest = list(argv[1:])
    if a0.startswith("python"):
        while rest:
            a = rest[0]
            if a == "-m" or (a.startswith("-m") and len(a) > 2):
                mod = rest[1] if a == "-m" and len(rest) > 1 else a[2:]
                if mod.startswith("chess_anti_engine") and mod not in _NOT_GPU_MODULES:
                    return f"-m {mod}"
                return None
            if a.startswith("-c"):
                return None
            if a in ("-X", "-W"):
                rest = rest[2:]
                continue
            if a == "--":
                rest = rest[1:]
                break
            if a.startswith("-"):
                rest = rest[1:]
                continue
            break
    elif a0 in ("bash", "sh", "dash", "zsh", "dotnet"):
        while rest:
            a = rest[0]
            if a0 != "dotnet" and (a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a[1:])):
                return None
            if a in ("-o", "+o"):
                rest = rest[2:]
                continue
            if a == "--":
                rest = rest[1:]
                break
            if a.startswith(("-", "+")):
                rest = rest[1:]
                continue
            break
    else:
        return None
    if rest and os.path.basename(rest[0]) in GPU_ENTRY_POINT_NAMES:
        return os.path.basename(rest[0])
    return None


def _read_status(pdir: Path) -> tuple[int, int] | None:
    uid = ppid = None
    try:
        for line in (pdir / "status").read_text().splitlines():
            if line.startswith("Uid:"):
                uid = int(line.split()[1])
            elif line.startswith("PPid:"):
                ppid = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    if uid is None or ppid is None:
        return None
    return uid, ppid


def _read_nul(path: Path) -> list[str] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return [s.decode("utf-8", "replace") for s in data.split(b"\0") if s]


def _cpu_only_environ(env_entries: Sequence[str] | None) -> bool:
    if env_entries is None:
        return False
    vals = [e for e in env_entries if e.startswith("CUDA_VISIBLE_DEVICES=")]
    return bool(vals) and all(e == "CUDA_VISIBLE_DEVICES=" for e in vals)


def gpu_device_rdevs(dev_root: Path) -> set[int]:
    rdevs = set()
    for path in [dev_root / "dxg", *sorted(dev_root.glob("nvidia*"))]:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if stat.S_ISCHR(st.st_mode):
            rdevs.add(st.st_rdev)
    return rdevs


def process_table(proc_root: Path) -> dict[int, tuple[int, int]]:
    table = {}
    for pdir in proc_root.iterdir():
        if pdir.name.isdigit():
            st = _read_status(pdir)
            if st is not None:
                table[int(pdir.name)] = st
    return table


def ancestors(pid: int, table: Mapping[int, tuple[int, int]]) -> set[int]:
    seen: set[int] = set()
    q: int | None = pid
    while q and q not in seen:
        seen.add(q)
        q = table[q][1] if q in table else None
    return seen


def descendants(roots: set[int], table: Mapping[int, tuple[int, int]]) -> set[int]:
    out = set(roots)
    changed = True
    while changed:
        changed = False
        for p, (_, pp) in table.items():
            if p not in out and pp in out:
                out.add(p)
                changed = True
    return out


def probe_processes(seams: Seams, self_pid: int, uid: int) -> tuple[str | None, str, list[str]]:
    """(reason or None, what it saw, notes)."""
    notes: list[str] = []
    rdevs = seams.gpu_rdevs() if seams.gpu_rdevs is not None else gpu_device_rdevs(seams.dev_root)
    if not rdevs:
        return ("cannot identify the GPU device (no character device at /dev/dxg or /dev/nvidia*)",
                "process scan: no GPU device", notes)
    try:
        table = process_table(seams.proc_root)
    except OSError as exc:
        return f"process scan failed: {exc}", "process scan: unreadable", notes
    mine = [p for p, (u, _) in table.items() if u == uid]
    if self_pid not in table:
        return "the process scan cannot see its own process", "process scan: incomplete", notes
    excluded = ancestors(self_pid, table) | descendants({self_pid}, table)
    checked = 0
    for p in sorted(mine):
        if p in excluded:
            continue
        pdir = seams.proc_root / str(p)
        argv = _read_nul(pdir / "cmdline")
        if not argv:
            continue  # exited, zombie or kernel thread
        checked += 1
        shown = " ".join(argv)[:200]
        hit = argv_entry_point(argv)
        if hit:
            if _cpu_only_environ(_read_nul(pdir / "environ")):
                notes.append(f"pid {p} ({hit}) runs with CUDA_VISIBLE_DEVICES= (empty): exempt from the argv clause")
            else:
                return f"GPU work is running: pid {p} ({hit}): {shown}", f"process scan: pid {p} {hit}", notes
        fd_dir = pdir / "fd"
        try:
            fds = sorted(fd_dir.iterdir())
        except FileNotFoundError:
            continue  # exited mid-scan
        except OSError as exc:
            if argv[0] in FD_UNREADABLE_ALLOW:
                notes.append(f"pid {p} ({argv[0]}) is allowlisted; its fds were not read")
                continue
            return (f"cannot read the fds of pid {p} ({shown[:120]}): {exc}; its GPU use cannot be checked",
                    "process scan: unreadable fds", notes)
        for fd in fds:
            try:
                st = seams.fd_stat(fd)
            except FileNotFoundError:
                continue  # fd closed mid-scan
            except OSError as exc:
                if argv[0] in FD_UNREADABLE_ALLOW:
                    notes.append(f"pid {p} ({argv[0]}) is allowlisted; its fds were not read")
                    break
                return (f"cannot read fd {fd.name} of pid {p} ({shown[:120]}): {exc}; its GPU use cannot be checked",
                        "process scan: unreadable fds", notes)
            if stat.S_ISCHR(st.st_mode) and st.st_rdev in rdevs:
                return (f"a process holds the GPU device: pid {p} via fd {fd.name}: {shown}",
                        f"process scan: pid {p} holds the GPU device", notes)
    return None, f"process scan: {checked} same-uid processes, no GPU job", notes


# ---------------------------------------------------------------- decision


def _allow_reason(allow_reason: str | None, env: Mapping[str, str]) -> str | None:
    if allow_reason is not None:
        if not allow_reason.strip():
            raise GateUsageError(f"{FLAG} needs a non-empty REASON")
        return allow_reason.strip()
    if ENV_ALLOW in env:
        if not env[ENV_ALLOW].strip():
            raise GateUsageError(f"{ENV_ALLOW} is set but empty; it needs a non-empty REASON")
        return env[ENV_ALLOW].strip()
    return None


def check(
    tool: str,
    *,
    allow_reason: str | None = None,
    device: str | None = None,
    env: Mapping[str, str] | None = None,
    for_pid: int | None = None,
    seams: Seams | None = None,
) -> GateDecision:
    """Decide; never prints or exits. Raises GateUsageError for an empty override reason."""
    env = os.environ if env is None else env
    seams = seams or Seams()
    reason = _allow_reason(allow_reason, env)
    if not tool or not tool.strip():
        raise GateUsageError("tool name is empty")
    test_mode = env.get(ENV_TEST) == "1"
    if (device is not None and not str(device).startswith("cuda") and str(device) not in ("auto", "")
            and not test_mode):
        return GateDecision(EXIT_OK, "OK cpu-only", notes=[f"device {device}"])
    if env.get("CUDA_VISIBLE_DEVICES", None) == "" and not test_mode:
        return GateDecision(EXIT_OK, "OK cpu-only", notes=["CUDA_VISIBLE_DEVICES is empty"])
    self_pid = seams.pid if seams.pid is not None else os.getpid()
    marker_pid = for_pid if for_pid is not None else self_pid
    marker = f"{marker_pid}:{tool}"
    notes: list[str] = []
    passed = env.get(ENV_PASSED)
    if passed:
        head = passed.split(":", 1)[0]
        try:
            table = process_table(seams.proc_root)
            lineage = ancestors(self_pid, table)
        except OSError:
            lineage = {self_pid}
        if head.isdigit() and int(head) in lineage:
            return GateDecision(EXIT_OK, "OK inherited", notes=[f"{ENV_PASSED}={passed} names an ancestor"], marker=passed)
        notes.append(f"{ENV_PASSED}={passed!r} does not name an ancestor of pid {self_pid}: ignored")
    if for_pid is not None:
        try:
            lineage = ancestors(self_pid, process_table(seams.proc_root))
        except OSError:
            lineage = set()
        if for_pid not in lineage:
            raise GateUsageError(f"--for-pid {for_pid} is not an ancestor of the gate (pid {self_pid})")
    smi = seams.nvidia_smi or _run_smi
    smi_present = seams.nvidia_smi is not None or shutil.which("nvidia-smi") is not None
    rdevs = seams.gpu_rdevs() if seams.gpu_rdevs is not None else gpu_device_rdevs(seams.dev_root)
    if not smi_present and not rdevs:
        # No GPU device node AND no nvidia-smi: this host has no GPU, so no GPU job can be beside us
        # (CI, CPU-only hosts). Either one present without the other refuses below (fail closed).
        return GateDecision(EXIT_OK, "OK no-gpu", notes=[*notes, "no /dev/dxg or /dev/nvidia* device and no nvidia-smi"])
    home = seams.home if seams.home is not None else Path.home()
    uid = seams.uid if seams.uid is not None else os.getuid()
    reasons, seen = [], []
    for r, s in (probe_memory(smi, home), probe_utilisation(smi, seams.sleep)):
        seen.append(s)
        if r:
            reasons.append(r)
    r, s, scan_notes = probe_processes(seams, self_pid, uid)
    seen.append(s)
    notes += scan_notes
    if r:
        reasons.append(r)
    if not reasons:
        return GateDecision(EXIT_OK, "OK", notes=[*notes, "; ".join(seen)], marker=marker)
    if reason is not None:
        notes.append(f"overridden: reason={reason!r}; saw: {'; '.join(reasons)}")
        return GateDecision(EXIT_OK, "OK allowed", reasons=reasons, notes=notes, marker=marker)
    return GateDecision(EXIT_REFUSED, "REFUSED", reasons=reasons, notes=notes)


def _test_report(env: Mapping[str, str]) -> None:
    if env.get(ENV_TEST) == "1":
        print(f"gpu_gate-test: torch_imported={'torch' in sys.modules}", file=sys.stderr, flush=True)


def _emit(tool: str, decision: GateDecision) -> None:
    for note in decision.notes:
        if note.startswith("overridden:"):
            print(f"gpu_gate: !!! {tool}: BUSY-GPU OVERRIDE ({FLAG} / {ENV_ALLOW}) {note}", file=sys.stderr, flush=True)
        else:
            print(f"gpu_gate: {tool}: note: {note}", file=sys.stderr, flush=True)
    if decision.passed:
        print(f"gpu_gate: {tool}: {decision.status}", file=sys.stderr, flush=True)
    else:
        for r in decision.reasons:
            print(f"gpu_gate: {tool}: REFUSED: {r}", file=sys.stderr, flush=True)
        print(f"gpu_gate: {tool}: refused to start GPU work beside a running GPU job. If that is deliberate, "
              f"rerun with {FLAG} \"<reason>\" (agents may not).", file=sys.stderr, flush=True)


def require_gpu_idle(tool: str, *, allow_reason: str | None = None, device: str | None = None) -> GateDecision:
    """Gate the current process: exit 8 on refusal, exit 2 on a usage error; on a pass export the marker."""
    try:
        decision = check(tool, allow_reason=allow_reason, device=device)
    except GateUsageError as exc:
        print(f"gpu_gate: {tool}: usage: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_USAGE) from None
    _emit(tool, decision)
    _test_report(os.environ)
    if not decision.passed:
        raise SystemExit(EXIT_REFUSED)
    if decision.marker:
        os.environ[ENV_PASSED] = decision.marker
    if os.environ.get(ENV_TEST) == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_TEST_PASSED)
    return decision


def add_allow_busy_gpu_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        FLAG, dest="allow_busy_gpu", metavar="REASON", default=None,
        help="Start even though another GPU job appears to be running (non-empty reason required; "
             "printed loudly). Not for agents. See docs/operations.md 'GPU launch gate'.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gpu_gate", description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--tool", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--for-pid", type=int, default=None,
                   help="the caller's pid (a shell passes $$); it must be an ancestor; the marker names it")
    add_allow_busy_gpu_argument(p)
    a = p.parse_args(argv)
    try:
        decision = check(a.tool, allow_reason=a.allow_busy_gpu, device=a.device, for_pid=a.for_pid)
    except GateUsageError as exc:
        print(f"gpu_gate: {a.tool}: usage: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    _emit(a.tool, decision)
    if not decision.passed:
        return EXIT_REFUSED
    if decision.marker:
        print(f"{ENV_PASSED}={decision.marker}", flush=True)
    if os.environ.get(ENV_TEST) == "1":
        return EXIT_TEST_PASSED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
