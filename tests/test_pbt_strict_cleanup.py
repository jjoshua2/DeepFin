"""Disposable command spies only; no project imports or operational launches."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class PbtStrictCleanup(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        self.config = self.root / "configs/fixture.yaml"
        self.config.write_text(
            "distributed_inference_use_compile: true\n"
            "distributed_worker_use_compile: true\n"
            "distributed_inference_batch_wait_ms: 0.0\n", encoding="utf-8",
        )
        self.log = self.root / "calls"
        self.env = {
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "CHESS_ROOT": str(self.root), "TMPDIR": str(self.root),
            "SPY_LOG": str(self.log), "BENCH_BATCH_WAIT_CONFIG": str(self.config),
            "BENCH_BATCH_WAIT_BROKER_LOG": str(self.root / "absent-broker.log"),
            "TRAIN_MONITOR_LOG": str(self.root / "monitor.log"),
            "TRAIN_WORK_DIR": str(self.root / "run"),
            "WATCHDOG_PBT_MAX_ITERS": "2", "WATCHDOG_INTERVAL_SECONDS": "0",
        }
        self.spy("python3", 'printf "python\\n" >> "$SPY_LOG"\n')
        self.spy("ray", 'printf "ray %s\\n" "$*" >> "$SPY_LOG"\n')
        self.spy("sleep", ":\n")
        self.spy("pgrep", "exit 0\n")  # Always take the monitoring branch.
        self.spy("tail", 'exit "${TAIL_STATUS:-0}"\n')

    def spy(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def run_fixture(self, name: str) -> subprocess.CompletedProcess[str]:
        path = self.root / "scripts" / name
        source = SOURCE / "scripts" / name
        if name == "watchdog_pbt.sh" and os.environ.get("PBT_WATCHDOG_SOURCE"):
            source = Path(os.environ["PBT_WATCHDOG_SOURCE"])
        shutil.copyfile(source, path)
        # The copied benchmark owns only an inert python3 spy. Observe kill
        # without signalling; wait reaps that spy in its owning fixture shell.
        body = (
            'kill() { printf "kill %s\\n" "$*" >> "$SPY_LOG"; }\n'
            'wait() { builtin wait "$@"; local rc=$?; '
            'printf "wait %s\\n" "$*" >> "$SPY_LOG"; return "$rc"; }\n'
            'export -f kill wait\nexec bash "$1"\n'
        )
        return subprocess.run(
            ["bash", "-c", body, "inert-pbt-fixture", str(path)],
            cwd=self.root, env=self.env, capture_output=True, text=True,
            check=False, timeout=10,
        )

    def calls(self, prefix: str) -> list[str]:
        return [line for line in self.log.read_text().splitlines() if line.startswith(prefix)]

    def assert_cleanup(self, count: int) -> None:
        killed = [line.removeprefix("kill ") for line in self.calls("kill ")]
        waited = [line.removeprefix("wait ") for line in self.calls("wait ")]
        self.assertEqual(len(killed), count)
        self.assertEqual(waited, killed)
        self.assertTrue(all(pid.isdecimal() and int(pid) > 0 for pid in killed))
        self.assertEqual(self.calls("ray "), ["ray stop --force"] * count)
        self.assertEqual(len(self.calls("python")), count)
        self.assertEqual(list(self.root.glob("bench_batch_wait.*.yaml")), [])
        self.assertIn("use_compile: true", self.config.read_text())

    def test_failed_log_reaps_child_and_preserves_failure(self) -> None:
        self.env["TAIL_STATUS"] = "17"
        result = self.run_fixture("bench_batch_wait.sh")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assert_cleanup(1)

    def test_normal_benchmark_cleans_each_child_once(self) -> None:
        result = self.run_fixture("bench_batch_wait.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Done.", result.stdout)
        self.assert_cleanup(6)

    def test_failure_before_child_does_not_stop_ray(self) -> None:
        self.env["BENCH_BATCH_WAIT_CONFIG"] = str(self.root / "missing.yaml")
        result = self.run_fixture("bench_batch_wait.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertEqual(list(self.root.glob("bench_batch_wait.*.yaml")), [])

    def monitor_fixture(self, status: int) -> subprocess.CompletedProcess[str]:
        # The real monitor, guard and train script never enter this fixture.
        (self.root / "scripts/intentional_stop_guard.sh").write_text(
            'CAE_STOP_MARKER_DEFAULT="$CHESS_ROOT/stop"\n'
            'CAE_TUNE_DIR_DEFAULT="runs/fixture/tune"\n',
        )
        (self.root / "scripts/monitor_pbt.sh").write_text(
            'printf "monitor\\n" >> "$SPY_LOG"\nexit ' + str(status) + "\n",
        )
        (self.root / "scripts/train.sh").write_text(
            '[ "$#" -eq 1 ] && [ "$1" = owned-status ] || exit 99\n'
            'printf "owned-status\\n" >> "$SPY_LOG"\n',
        )
        return self.run_fixture("watchdog_pbt.sh")

    def test_monitor_status_one_continues(self) -> None:
        result = self.monitor_fixture(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("monitor"), ["monitor", "monitor"])
        self.assertEqual(self.calls("python"), [])

    def test_monitor_success_continues(self) -> None:
        result = self.monitor_fixture(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("monitor"), ["monitor", "monitor"])

    def test_unexpected_monitor_status_is_fatal(self) -> None:
        result = self.monitor_fixture(2)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls("monitor"), ["monitor"])


if __name__ == "__main__":
    unittest.main()
