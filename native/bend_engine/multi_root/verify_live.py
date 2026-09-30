"""Opt-in persistent-root tests against the actual compiled Bend owner.

The gate replaces only the native model callback; its pipes never enter the
product. Ordinary pytest imports only the inexpensive transcript validator.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import select
import subprocess
import threading
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = "info string live_root "
WORK = "info string live_work "
CONTROL = "info string live_control "
TERMINAL = "fen k7/1Q6/2K5/8/8/8/8/8 b - - 150 1"


def integer(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {key}: {value!r}")
    return value


def validate(lines: list[str]) -> dict[str, Any]:
    """Reconcile identity and lifetime work without inventing model execution."""
    roots: dict[tuple[int, int], dict[str, Any]] = {}
    active: dict[int, int] = {}
    generations: set[int] = set()
    summaries: list[dict[str, Any]] = []
    for line in lines:
        if summaries and line.startswith((CONTROL + "admitted ", CONTROL + "removed ", ROOT)):
            raise ValueError("lifecycle event after final summary")
        if line.startswith(CONTROL + "admitted "):
            parts = line.removeprefix(CONTROL).split()
            if len(parts) != 3:
                raise ValueError("malformed admission")
            slot, gen = map(int, parts[1:])
            if not 1 <= slot <= 16 or not 1 <= gen <= 65535 or gen in generations:
                raise ValueError("invalid or recycled generation")
            if generations and gen <= max(generations):
                raise ValueError("non-monotonic generation")
            if slot in active and (slot, active[slot]) not in roots:
                raise ValueError("unreported generation replaced")
            active[slot] = gen
            generations.add(gen)
        elif line.startswith(CONTROL + "removed "):
            parts = line.removeprefix(CONTROL).split()
            if len(parts) != 3:
                raise ValueError("malformed removal")
            slot, gen = map(int, parts[1:])
            if active.get(slot) != gen or (slot, gen) not in roots:
                raise ValueError("removal precedes retirement")
            del active[slot]
        elif line.startswith(ROOT):
            obj = json.loads(line.removeprefix(ROOT))
            if not isinstance(obj, dict) or obj.get("schema") != "deepfin.live-root.v1":
                raise ValueError("invalid root record")
            slot, gen = integer(obj, "slot"), integer(obj, "generation")
            if active.get(slot) != gen or (slot, gen) in roots:
                raise ValueError("stale or duplicate root result")
            sent, executed = integer(obj, "dispatched_real_rows"), integer(obj, "executed_real_rows")
            accepted, wasted = integer(obj, "accepted_neural_rows"), integer(obj, "cancelled_rows")
            if sent != executed or executed != accepted + wasted:
                raise ValueError("root physical work mismatch")
            if integer(obj, "completed_simulations") < accepted:
                raise ValueError("accepted work exceeds completed simulations")
            if "arena_capacity" in obj or "arena_physical_slots" in obj:
                capacity = integer(obj, "arena_capacity")
                physical = integer(obj, "arena_physical_slots")
                if not 1 <= capacity <= 65536 or physical != 1 << max(12, (capacity - 1).bit_length()):
                    raise ValueError("invalid arena capacity or physical storage")
                if integer(obj, "used_nodes") > capacity:
                    raise ValueError("used nodes exceed arena capacity")
            roots[slot, gen] = obj
        elif line.startswith(WORK):
            obj = json.loads(line.removeprefix(WORK))
            if not isinstance(obj, dict) or obj.get("schema") != "deepfin.live-cohort-work.v1":
                raise ValueError("invalid lifetime record")
            summaries.append(obj)
    if len(summaries) != 1 or any((slot, gen) not in roots for slot, gen in active.items()):
        raise ValueError("missing/duplicate lifetime record or unfinished generation")
    work = summaries[0]
    if integer(work, "reported_generations") != len(roots):
        raise ValueError("generation count mismatch")
    for key in ("completed_simulations", "dispatched_real_rows", "executed_real_rows", "accepted_neural_rows", "cancelled_rows"):
        if integer(work, key) != sum(integer(r, key) for r in roots.values()):
            raise ValueError(f"lifetime {key} mismatch")
    if integer(work, "executed_real_rows") != integer(work, "accepted_neural_rows") + integer(work, "executed_wasted_rows"):
        raise ValueError("lifetime waste mismatch")
    if integer(work, "executed_wasted_rows") != integer(work, "cancelled_rows"):
        raise ValueError("cancelled/wasted mismatch")
    batch = integer(work, "batch_size")
    if batch not in (1, 2, 4, 8, 16) or integer(work, "physical_rows") != batch * integer(work, "forward_calls"):
        raise ValueError("physical batch/call mismatch")
    if integer(work, "physical_rows") != integer(work, "executed_real_rows") + integer(work, "padded_rows"):
        raise ValueError("padding mismatch")
    if integer(work, "unconfirmed_forward_rows") or integer(work, "unresolved_rows"):
        raise ValueError("summary precedes physical drain")
    return {"roots": {f"{s}:{g}": r for (s, g), r in roots.items()}, "work": work}


class Client:
    def __init__(self, binary: Path, *, gate: bool = False, simulations: int = 1, fault: str = "", arena: int = 4096, depth: int = 2,
                 environment: dict[str, str] | None = None, diagnostics: bool = True):
        self.events = self.release_fd = -1
        pass_fds: tuple[int, ...] = ()
        env = {**(os.environ if environment is None else environment),
               "DEEPFIN_COHORT_ASYNC": "1", "DEEPFIN_COHORT_ARENA_NODES": str(arena)}
        if fault:
            env["DEEPFIN_MULTI_TEST_FAULT"] = fault
        if gate:
            self.events, event_writer = os.pipe()
            release_reader, self.release_fd = os.pipe()
            pass_fds = (event_writer, release_reader)
            env.update(DEEPFIN_TEST_EVENT_FD=str(event_writer), DEEPFIN_TEST_RELEASE_FD=str(release_reader))
        self.proc = subprocess.Popen([str(binary), "--threads", "1", "--", str(simulations), str(depth), "0", str(int(diagnostics))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, pass_fds=pass_fds, env=env)
        for fd in pass_fds:
            os.close(fd)
        self.rows: queue.Queue[str | None] = queue.Queue()
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.readers = [threading.Thread(target=self._stdout, daemon=True), threading.Thread(target=self._stderr, daemon=True)]
        for reader in self.readers:
            reader.start()
        try:
            self.until("info string live_open ")
        except BaseException:
            self.close()
            raise

    def _stdout(self) -> None:
        assert self.proc.stdout
        for line in self.proc.stdout:
            text = line.rstrip("\n")
            self.lines.append(text)
            self.rows.put(text)
        self.rows.put(None)

    def _stderr(self) -> None:
        assert self.proc.stderr
        self.errors.extend(self.proc.stderr)

    def send(self, text: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def until(self, prefix: str, timeout: float = 5) -> str:
        end = time.monotonic() + timeout
        while True:
            line = self.rows.get(timeout=max(0.001, end - time.monotonic()))
            if line is None:
                raise AssertionError((prefix, self.proc.poll(), self.errors, self.lines[-8:]))
            if line.startswith(prefix):
                return line
            if time.monotonic() >= end:
                raise TimeoutError(prefix)

    def result(self, slot: int, gen: int) -> dict[str, Any]:
        while True:
            r = json.loads(self.until(ROOT).removeprefix(ROOT))
            if (r["slot"], r["generation"]) == (slot, gen):
                return r

    def started(self) -> None:
        assert self.events >= 0
        assert select.select([self.events], [], [], 5)[0], self.lines[-8:]
        assert os.read(self.events, 1) == b"S"

    def release(self) -> None:
        assert self.release_fd >= 0
        assert os.write(self.release_fd, b"R") == 1

    def finish(self, code: int = 0) -> dict[str, Any]:
        self.proc.wait(timeout=10)
        for reader in self.readers:
            reader.join(timeout=2)
            assert not reader.is_alive()
        assert self.proc.returncode == code, (self.proc.returncode, self.errors)
        if code:
            assert not any(x.startswith(WORK) for x in self.lines)
            return {"exit": code, "error": "".join(self.errors)}
        assert not self.errors, self.errors
        return validate(self.lines)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        for fd in (self.events, self.release_fd):
            if fd >= 0:
                os.close(fd)


def invalid_replacement_preserves_root(gate: Path) -> dict[str, Any]:
    """A rejected position must not cancel the existing admitted evaluation."""
    c = Client(gate, gate=True)
    try:
        c.send("add startpos")
        c.until(CONTROL + "admitted 1 1")
        c.started()
        before = len(c.lines)
        c.send("replace 1 1 startpos moves e2e5")
        c.until(CONTROL + "error invalid-position", 0.75)
        c.release()
        old = c.result(1, 1)
        assert (old["accepted_neural_rows"], old["cancelled_rows"]) == (1, 0)
        assert not any(x.startswith(CONTROL + "pending-install") for x in c.lines[before:])
        c.send("quit")
        result = c.finish()
        assert set(result["roots"]) == {"1:1"}
        return result
    finally:
        c.close()


def replacement(gate: Path) -> dict[str, Any]:
    c = Client(gate, gate=True)
    try:
        c.send("isready")
        c.until("info string live_ready", 0.75)
        c.send("add startpos")
        c.until(CONTROL + "admitted 1 1")
        c.started()
        c.send("replace 1 1 startpos moves e2e5")
        c.until(CONTROL + "error invalid-position", 0.75)
        c.send("replace 1 1 startpos moves e2e4")
        c.until(CONTROL + "pending-install 1 2", 0.75)
        c.send("add startpos\ncancel 1 2\nisready")
        c.until(CONTROL + "error lifecycle-busy", 0.75)
        c.until(CONTROL + "error stale-identity", 0.75)
        c.until("info string live_ready", 0.75)
        assert not any(x.startswith((ROOT, CONTROL + "admitted 1 2")) for x in c.lines)
        c.release()
        old = c.result(1, 1)
        assert (old["accepted_neural_rows"], old["cancelled_rows"]) == (0, 1)
        c.until(CONTROL + "admitted 1 2")
        c.started()
        c.send("cancel 1 1\ndeadline 1 1 0\nremove 1 1")
        for _ in range(3):
            c.until(CONTROL + "error stale-identity", 0.75)
        c.release()
        new = c.result(1, 2)
        assert (new["accepted_neural_rows"], new["cancelled_rows"]) == (1, 0)
        c.send("replace 1 2 startpos")
        c.until(CONTROL + "admitted 1 3")
        c.started()
        c.send("remove 1 3")
        c.until(CONTROL + "pending-remove 1 3", 0.75)
        assert not any(x.startswith(CONTROL + "removed 1 3") for x in c.lines)
        c.release()
        c.result(1, 3)
        c.until(CONTROL + "removed 1 3")
        c.send("add startpos moves d2d4")
        c.until(CONTROL + "admitted 1 4")
        c.started()
        c.release()
        c.result(1, 4)
        c.send("quit")
        result = c.finish()
        assert (result["work"]["executed_real_rows"], result["work"]["accepted_neural_rows"], result["work"]["cancelled_rows"]) == (4, 2, 2)
        return result
    finally:
        c.close()


def abandoned(gate: Path, *, quit_now: bool) -> dict[str, Any]:
    c = Client(gate, gate=True)
    try:
        c.send("add startpos")
        c.started()
        c.send("replace 1 1 startpos moves e2e4")
        c.until(CONTROL + "pending-install 1 2")
        c.send("quit" if quit_now else "stop")
        c.until(CONTROL + "abandoned 1 2", 0.75)
        c.until(CONTROL + ("quit" if quit_now else "stop"), 0.75)
        assert c.proc.poll() is None
        assert not any(x.startswith(WORK) for x in c.lines)
        c.release()
        if not quit_now:
            c.result(1, 1)
            c.send("replace 1 1 startpos")
            c.until(CONTROL + "admitted 1 3")
            c.started()
            c.release()
            c.result(1, 3)
            c.send("quit")
        r = c.finish()
        assert "1:2" not in r["roots"]
        return r
    finally:
        c.close()


def capacity(binary: Path) -> dict[str, Any]:
    c = Client(binary)
    try:
        for n in range(1, 17):
            c.send("add " + TERMINAL)
            c.until(CONTROL + f"admitted {n} {n}")
            r = c.result(n, n)
            assert r["accepted_neural_rows"] == 0
        c.send("add " + TERMINAL)
        c.until(CONTROL + "error capacity-or-generation-limit")
        c.send("remove 8 8")
        c.until(CONTROL + "removed 8 8")
        c.send("add " + TERMINAL)
        c.until(CONTROL + "admitted 8 17")
        c.result(8, 17)
        for bad in ("cancel 8", "deadline 8 1", "remove 8 8", "replace 0 1 startpos", "cancel 17 1", "replace 8 17 nonsense"):
            c.send(bad)
            c.until(CONTROL + "error ")
        assert c.proc.stdin
        c.proc.stdin.close()  # EOF closes intake, not a fictitious cancellation.
        r = c.finish()
        assert r["work"]["reported_generations"] == 17
        assert r["work"]["executed_real_rows"] == 0
        return r
    finally:
        c.close()


def expire_and_reuse(gate: Path) -> dict[str, Any]:
    c = Client(gate, gate=True)
    try:
        c.send("add startpos")
        c.started()
        c.send("deadline 1 1 0")
        c.until("info string cohort_control expired 1", 0.75)
        c.release()
        old = c.result(1, 1)
        assert old["deadline_expired"]
        assert old["cancelled_rows"] == 1
        c.send("replace 1 1 startpos")
        c.until(CONTROL + "admitted 1 2")
        c.started()
        c.release()
        new = c.result(1, 2)
        assert not new["deadline_expired"]
        assert new["deadline_offset_ms"] is None
        assert new["accepted_neural_rows"] == 1
        c.send("quit")
        return c.finish()
    finally:
        c.close()


def model_failure(gate: Path, fault: str) -> dict[str, Any]:
    c = Client(gate, gate=True, fault=fault)
    try:
        c.send("add startpos")
        c.started()
        c.send("replace 1 1 startpos moves e2e4")
        c.until(CONTROL + "pending-install 1 2")
        c.release()
        r = c.finish(2)
        assert not any(x.startswith((ROOT, CONTROL + "admitted 1 2")) for x in c.lines)
        return r
    finally:
        c.close()


def tree_snapshots(lines: list[str]) -> dict[tuple[int, int], list[list[int]]]:
    """Associate diagnostic tree rows with the immediately enclosing generation."""
    found: dict[tuple[int, int], list[list[int]]] = {}
    current: tuple[int, int] | None = None
    for line in lines:
        if line.startswith("info string live_result_begin "):
            if current is not None:
                raise ValueError("unfinished snapshot before next generation")
            a, b = map(int, line.removeprefix("info string live_result_begin ").split())
            current = (a, b)
            if current in found:
                raise ValueError("duplicate tree snapshot")
            found[current] = []
        elif line.startswith("info string cohort_node "):
            slot, index, raw = line.removeprefix("info string cohort_node ").split(" ", 2)
            node = json.loads(raw)
            if current is None or int(slot) != current[0] or int(index) != len(found[current]):
                raise ValueError("unscoped or reordered diagnostic tree row")
            if not isinstance(node, list) or len(node) != 29 or any(type(v) is not int for v in node):
                raise ValueError("malformed diagnostic tree row")
            found[current].append(node)
        elif line.startswith(ROOT):
            root = json.loads(line.removeprefix(ROOT))
            if current is None or current != (root["slot"], root["generation"]) or len(found[current]) != root["used_nodes"]:
                raise ValueError("incomplete generation snapshot")
            current = None
    if current is not None:
        raise ValueError("unfinished diagnostic tree snapshot")
    return found


def serial(reference: Path, position: str, simulations: int) -> tuple[dict[str, Any], list[list[int]]]:
    env = {**os.environ, "DEEPFIN_COHORT_ASYNC": "0"}
    env.pop("DEEPFIN_MULTI_TEST_FAULT", None)
    env.pop("DEEPFIN_MULTI_TEST_OPEN", None)
    got = subprocess.run([str(reference), "--threads", "1", "--", str(simulations), "2", "0", "1", position],
                         input="", capture_output=True, text=True, timeout=20, env=env, check=False)
    assert got.returncode == 0, (got.returncode, got.stderr)
    assert not got.stderr, got.stderr
    roots = [json.loads(x.removeprefix("info string cohort_root ")) for x in got.stdout.splitlines() if x.startswith("info string cohort_root ")]
    assert len(roots) == 1
    nodes: list[list[int]] = []
    for line in got.stdout.splitlines():
        if line.startswith("info string cohort_node "):
            slot, index, raw = line.removeprefix("info string cohort_node ").split(" ", 2)
            assert slot == "1"
            assert int(index) == len(nodes)
            nodes.append(json.loads(raw))
    assert len(nodes) == roots[0]["used_nodes"]
    return roots[0], nodes


def assert_serial(root: dict[str, Any], nodes: list[list[int]], control: tuple[dict[str, Any], list[list[int]]]) -> None:
    expected, expected_nodes = control
    assert nodes == expected_nodes, (root["slot"], root["generation"])
    for key in ("completed_simulations", "accepted_neural_rows", "rule_draw_replies", "used_nodes", "searched_move", "bestmove"):
        assert root[key] == expected[key], (key, root, expected)


def serial_reuse(binary: Path, reference: Path) -> dict[str, Any]:
    positions = ["startpos", "startpos moves e2e4 e7e5", "startpos moves g1f3 g8f6 f3g1 f6g8",
                 "fen 7k/P7/7K/8/8/8/8/8 w - - 0 1", "startpos moves e2e4 a7a6 e4e5 d7d5", TERMINAL, "startpos"]
    c = Client(binary, simulations=4)
    try:
        results = []
        for gen, position in enumerate(positions, 1):
            c.send(("add " if gen == 1 else f"replace 1 {gen-1} ") + position)
            c.until(CONTROL + f"admitted 1 {gen}")
            results.append(c.result(1, gen))
        c.send("quit")
        report = c.finish()
        snapshots = tree_snapshots(c.lines)
        compared = 0
        for r, position in zip(results, positions, strict=True):
            nodes = snapshots[1, r["generation"]]
            assert_serial(r, nodes, serial(reference, position, 4))
            compared += len(nodes)
        return {"status": "passed", "generations": len(results), "complete_nodes_compared": compared, "work": report["work"]}
    finally:
        c.close()


def shared_replacement(gate: Path, reference: Path) -> dict[str, Any]:
    """Replace one in-flight root while its batch partner continues normally."""
    c = Client(gate, gate=True, simulations=4)
    try:
        c.send("add startpos")
        c.started()
        c.send("add startpos moves e2e4")
        c.until(CONTROL + "pending-install 2 2")
        c.release()
        c.until(CONTROL + "admitted 2 2")
        c.started()  # root 1's second leaf and root 2's first share this batch.
        c.send("replace 1 1 startpos moves d2d4")
        c.until(CONTROL + "pending-install 1 3", 0.75)
        c.release()
        old = c.result(1, 1)
        assert (old["accepted_neural_rows"], old["cancelled_rows"]) == (1, 1)
        c.until(CONTROL + "admitted 1 3")
        for _ in range(3):
            c.started()
            c.release()
        unaffected = c.result(2, 2)
        c.started()
        c.release()
        new = c.result(1, 3)
        c.send("quit")
        report = c.finish()
        snapshots = tree_snapshots(c.lines)
        assert_serial(old, snapshots[1, 1], serial(reference, "startpos", 1))
        assert_serial(unaffected, snapshots[2, 2], serial(reference, "startpos moves e2e4", 4))
        assert_serial(new, snapshots[1, 3], serial(reference, "startpos moves d2d4", 4))
        assert (report["work"]["executed_real_rows"], report["work"]["accepted_neural_rows"], report["work"]["cancelled_rows"]) == (10, 9, 1)
        return report
    finally:
        c.close()


def eof_pending(gate: Path) -> dict[str, Any]:
    c = Client(gate, gate=True)
    try:
        c.send("add startpos")
        c.started()
        c.send("replace 1 1 startpos moves e2e4")
        c.until(CONTROL + "pending-install 1 2")
        assert c.proc.stdin
        c.proc.stdin.close()
        c.release()
        old = c.result(1, 1)
        assert old["cancelled_rows"] == 1
        c.until(CONTROL + "admitted 1 2")
        c.started()
        c.release()
        report = c.finish()
        assert report["roots"]["1:2"]["accepted_neural_rows"] == 1
        return report
    finally:
        c.close()


def startup_rejections(binary: Path) -> int:
    cases = [(flag, ["1", "2", "0", "0"]) for flag in ("0", "bad", "")]
    cases += [("1", args) for args in ([], ["0", "2", "0", "0"], ["257", "2", "0", "0"],
              ["1", "0", "0", "0"], ["1", "2", "257", "0"], ["1", "2", "0", "2"], ["a", "2", "0", "0"])]
    with tempfile.TemporaryDirectory(prefix="live-startup-") as directory:
        for i, (flag, args) in enumerate(cases):
            marker = Path(directory) / f"opened-{i}"
            env = {**os.environ, "DEEPFIN_COHORT_ASYNC": flag, "DEEPFIN_MULTI_TEST_OPEN": str(marker)}
            got = subprocess.run([str(binary), "--threads", "1", "--", *args], input="", capture_output=True, text=True, timeout=10, env=env, check=False)
            assert got.returncode == 2, (args, flag, got.returncode, got.stderr)
            assert not marker.exists(), marker
            assert not got.stdout, got.stdout
    return len(cases)


def arena_reuse(binary: Path) -> dict[str, Any]:
    """Exercise the selected capacity through actual root creation and replacement."""
    result: dict[str, Any] = {}
    for arena in (4096, 8192):
        c = Client(binary, simulations=256, depth=8, arena=arena)
        try:
            c.send("add startpos")
            first = c.result(1, 1)
            c.send("replace 1 1 startpos")
            second = c.result(1, 2)
            for r in (first, second):
                assert r["arena_capacity"] == arena
                assert r["arena_physical_slots"] == arena
                assert r["used_nodes"] <= arena
                if arena == 8192:
                    assert r["completed_simulations"] == 256
                    assert r["used_nodes"] > 4096
                else:
                    assert r["completed_simulations"] < 256
            for key in ("completed_simulations", "accepted_neural_rows", "used_nodes", "stop_code", "bestmove"):
                assert first[key] == second[key], (key, first, second)
            c.send("quit")
            report = c.finish()
            snapshots = tree_snapshots(c.lines)
            assert snapshots[1, 1] == snapshots[1, 2]
            result[str(arena)] = {"roots": report["roots"], "work": report["work"], "replacement_tree_bits_equal": True}
        finally:
            c.close()
    return result


def run(binary: Path, gate: Path, reference: Path) -> dict[str, Any]:
    return {"status": "passed", "scope": "compiled Bend live owner and actual worker with test-only callbacks; no neural-model or speed claim",
            "invalid_replacement_preserves_root": invalid_replacement_preserves_root(gate),
            "replacement": replacement(gate), "stop_abandons": abandoned(gate, quit_now=False),
            "quit_drains": abandoned(gate, quit_now=True), "capacity_and_eof": capacity(binary),
            "timer_reset": expire_and_reuse(gate), "eof_with_pending_replacement": eof_pending(gate),
            "serial_reuse": serial_reuse(binary, reference), "shared_replacement": shared_replacement(gate, reference),
            "arena_reuse": arena_reuse(binary), "startup_rejections": startup_rejections(binary), "failures": [model_failure(gate, x) for x in ("fail", "nan")]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    try:
        report = run(a.binary.resolve(), a.gate.resolve(), a.reference.resolve())
    except Exception as error:
        a.report.write_text(json.dumps({"status": "failed", "error": str(error), "live_runner_qualified": False}, indent=2) + "\n")
        raise
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "scope": report["scope"]}))


if __name__ == "__main__":
    main()
