"""Host result classification tests; not source-proof evidence."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from . import qualify
from .qualify import safe, semantic_rejection


class ClassificationTests(unittest.TestCase):
    def test_long_consumer_default_is_24_hours(self):
        self.assertEqual(qualify.DEFAULT_CONSUMER_TIMEOUT_SECONDS, 86_400)
        args = qualify.build_parser().parse_args(
            ["compiler", "--report", "unused.json"]
        )
        self.assertEqual(args.consumer_timeout_seconds, 86_400)
        override = qualify.build_parser().parse_args([
            "compiler", "--report", "unused.json", "--consumer-timeout-seconds", "1200"
        ])
        self.assertEqual(override.consumer_timeout_seconds, 1200)

    def test_consumer_default_propagates_to_child_and_log_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "consumer.log"
            child = __import__("unittest.mock").mock.Mock()
            child.poll.return_value = 0
            child.returncode = 0
            child.stdout = io.BytesIO()
            captured = {}

            def fake_popen(command, *, stdout, stderr, env, start_new_session):
                captured.update(stdout=stdout, stderr=stderr, env=env,
                                start_new_session=start_new_session)
                return child

            with patch.object(qualify.subprocess, "Popen", side_effect=fake_popen):
                result = qualify.invoke("bun", Path("compiler"), Path("entry"),
                                        qualify.DEFAULT_CONSUMER_TIMEOUT_SECONDS, log)
                self.assertEqual(captured["stdout"], qualify.subprocess.PIPE)
            self.assertFalse(result["timed_out"])
            self.assertEqual(result["wall_limit_seconds"], 86_400)
            self.assertEqual(result["log_path"], str(log))

    def test_short_explicit_control_deadline_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "control.log"
            with patch.object(qualify.subprocess, "Popen") as popen:
                popen.return_value.poll.return_value = 0
                popen.return_value.returncode = 0
                popen.return_value.stdout = io.BytesIO()
                result = qualify.invoke(
                    "bun", Path("compiler"), Path("entry"), 180, log
                )
            self.assertEqual(result["wall_limit_seconds"], 180)
            self.assertFalse(result["timed_out"])

    def test_output_limit_is_hard_and_interruption_reaps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "limited.log"
            with patch.object(qualify, "MAX_CONSUMER_LOG_BYTES", 4), patch.object(
                qualify.subprocess, "Popen"
            ) as popen:
                popen.return_value.poll.return_value = 0
                popen.return_value.returncode = 0
                popen.return_value.stdout = io.BytesIO(b"0123456789")
                with patch.object(qualify.os, "killpg"):
                    result = qualify.invoke(
                    "bun", Path("compiler"), Path("entry"), 180, log
                )
            self.assertTrue(result["output_limit_exceeded"])
            self.assertEqual(log.stat().st_size, 4)
            self.assertFalse(qualify.safe(result))

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "interrupted.log"
            with patch.object(qualify.subprocess, "Popen") as popen:
                child = popen.return_value
                child.stdout = io.BytesIO()
                child.poll.return_value = None
                child.wait.side_effect = [KeyboardInterrupt, None, None]
                child.returncode = -15
                with patch.object(qualify.os, "killpg") as killpg:
                    with self.assertRaises(KeyboardInterrupt):
                        qualify.invoke("bun", Path("compiler"), Path("entry"), 180, log)
                self.assertEqual(killpg.call_args_list, [
                    call(child.pid, qualify.signal.SIGTERM),
                    call(child.pid, qualify.signal.SIGKILL),
                ])

    def test_log_reader_failure_fails_closed(self):
        class BrokenOutput:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size):
                raise OSError("simulated log failure")

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "reader-error.log"
            with patch.object(qualify.subprocess, "Popen") as popen, patch.object(
                qualify.os, "killpg"
            ):
                child = popen.return_value
                child.poll.return_value = 0
                child.returncode = 0
                child.stdout = BrokenOutput()
                with self.assertRaisesRegex(RuntimeError, "output capture failed"):
                    qualify.invoke("bun", Path("compiler"), Path("entry"), 180, log)
                self.assertTrue(log.exists())

    def result(
        self, code: int | None = 0, text: str = "All terms check.\n",
        timed_out: bool = False,
    ) -> dict:
        return {"exit_code": code, "raw_text": text, "timed_out": timed_out}

    def test_only_exact_safe_success(self):
        self.assertTrue(safe(self.result()))
        self.assertFalse(safe({**self.result(), "output_limit_exceeded": True}))
        self.assertFalse(safe({**self.result(), "output_truncated": True}))
        for result in (
            self.result(1), self.result(-9), self.result(None, timed_out=True),
            self.result(text=""), self.result(text="All terms check.\nWARNING: unsafe"),
        ):
            self.assertFalse(safe(result))

    def test_timeout_cannot_be_a_rejection(self):
        text = "expected x observed y Location: Closed.public_builder_matches_recipe"
        self.assertTrue(semantic_rejection(self.result(1, text)))
        self.assertFalse(
            semantic_rejection({**self.result(1, text), "output_truncated": True})
        )
        for result in (
            self.result(None, text, True), self.result(-9, text),
            self.result(0, text), self.result(1, text, True),
        ):
            self.assertFalse(semantic_rejection(result))

    def test_unrelated_or_resource_diagnostic_is_not_semantic(self):
        text = "expected x observed y Location: Closed.public_builder_matches_recipe"
        for suffix in (
            "RangeError", "out of memory", "more than once", "unsafe", "TODO"
        ):
            self.assertFalse(semantic_rejection(self.result(1, text + " " + suffix)))
        self.assertFalse(
            semantic_rejection(self.result(1, "expected x observed y Location: other"))
        )
        self.assertFalse(semantic_rejection(self.result(1, "parse failed")))


if __name__ == "__main__":
    unittest.main()
