"""Host result classification tests; not source-proof evidence."""
from __future__ import annotations

import io
import json
import os
import sys
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

    def test_main_default_and_override_reach_actual_child_call_sites(self):
        # This exercises main's propagation only. Every child and proof receipt
        # is inert; this is not a compiler or theorem qualification.
        for override, expected in (([], 86_400), (["--consumer-timeout-seconds", "1200"], 1200)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                engine = root / "engine"
                suite = engine / "standalone/proofs/generator_contract"
                suite.mkdir(parents=True)
                for relative in ("standalone/proofs/table_preservation/focused.py",
                                 "standalone/proofs/table_preservation/_validation.py",
                                 "standalone/toolchain.json", "standalone/verify_compiler.js"):
                    path = engine / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("inert source identity fixture\n")
                report_path = root / "receipt.json"
                argv = ["qualify", str(root / "compiler"), "--report", str(report_path), *override]
                control = self.result(1, "expected x observed y Location: Closed.public_builder_matches_recipe")
                pin = qualify.subprocess.CompletedProcess([], 0, stdout="inert compiler identity\n", stderr="")
                with patch.object(sys, "argv", argv), patch.object(qualify, "SUITE", suite), \
                     patch.object(qualify, "ENGINE", engine), patch.object(qualify, "manifest"), \
                     patch.object(qualify, "load_closure", return_value=lambda *_: {}), \
                     patch.object(qualify, "replace"), patch.object(qualify.shutil, "copytree"), \
                     patch.object(qualify, "begin_report"), patch.object(qualify, "write_report") as writes, \
                     patch.object(qualify.subprocess, "run", return_value=pin) as compiler_calls, \
                     patch.object(qualify, "invoke", side_effect=[self.result(), control, control]) as children, \
                     patch("builtins.print"):
                    qualify.main()
                self.assertEqual([item.args[3] for item in children.call_args_list], [expected, 180, 180])
                actual_log = children.call_args_list[0].args[4]
                self.assertEqual(actual_log.parent.parent, root)
                self.assertEqual(actual_log.name, "checker.log")
                self.assertTrue(actual_log.parent.is_dir())
                self.assertEqual(writes.call_args.args[1]["consumer_log_path"], str(actual_log))
                self.assertEqual([item.kwargs["timeout"] for item in compiler_calls.call_args_list], [30, 30])
                self.assertEqual(writes.call_args.args[1]["default_consumer_timeout_seconds"], 86_400)
                self.assertEqual(writes.call_args.args[1]["effective_consumer_timeout_seconds"], expected)

    def test_main_nonpositive_override_invalidates_old_pass_before_children(self):
        for seconds in ("0", "-1"):
            with self.subTest(seconds=seconds), tempfile.TemporaryDirectory() as directory:
                report_path = Path(directory) / "receipt.json"
                report_path.write_text('{"closed_builder_gate":"PASS"}')
                argv = ["qualify", "unused-compiler", "--report", str(report_path),
                        "--consumer-timeout-seconds", seconds]
                with patch.object(sys, "argv", argv), patch.object(qualify, "invoke") as child, \
                     patch.object(qualify.subprocess, "run") as compiler:
                    with self.assertRaisesRegex(AssertionError, "finite and positive"):
                        qualify.main()
                child.assert_not_called()
                compiler.assert_not_called()
                receipt = json.loads(report_path.read_text())
                self.assertEqual(receipt["closed_builder_gate"], "NOT_COMPLETED")
                self.assertIn("finite and positive", receipt["failure"])

    def test_nonfinite_cli_overrides_are_not_admitted_by_integer_parser(self):
        for seconds in ("NaN", "inf", "-inf", "Infinity"):
            with self.subTest(seconds=seconds), patch.object(sys, "argv", [
                "qualify", "unused-compiler", "--report", "unused.json",
                "--consumer-timeout-seconds=" + seconds,
            ]), patch.object(qualify, "invoke") as child, \
                 patch.object(qualify.subprocess, "run") as compiler, \
                 patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    qualify.main()
                self.assertEqual(error.exception.code, 2)
                child.assert_not_called()
                compiler.assert_not_called()

    def test_invoke_invalid_wall_bounds_fail_before_log_or_child(self):
        for seconds in (0, -1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(seconds=seconds), patch.object(qualify.subprocess, "Popen") as child, \
                 patch.object(qualify.tempfile, "mkstemp") as log, \
                 patch.object(qualify.os, "killpg") as signal_call:
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    qualify.invoke("unused-bun", Path("unused-compiler"), Path("unused-entry"), seconds)
                child.assert_not_called()
                log.assert_not_called()
                signal_call.assert_not_called()

    def _capture_with_inert_child(self, payload: bytes, code: int, cap: int = 16 * 1024 * 1024) -> dict:
        # Popen, process-group signals and signal-handler registration are all
        # inert. Only temporary host log bytes and the actual wrapper are used.
        child = __import__("unittest.mock").mock.Mock()
        child.poll.return_value = code
        child.returncode = code
        child.stdout = io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(qualify.subprocess, "Popen", return_value=child), \
             patch.object(qualify, "MAX_CONSUMER_LOG_BYTES", cap), \
             patch.object(qualify.os, "killpg"), \
             patch.object(qualify.signal, "signal", return_value=qualify.signal.SIG_DFL):
            return qualify.invoke("inert-bun", Path("inert-compiler"), Path("inert-entry"),
                                  60, Path(directory) / "capture.log")

    def test_incomplete_capture_cannot_credit_any_shared_semantic_consumer(self):
        from . import qualify_castles, qualify_king_away
        from ..bitboard_inventory import qualify as bitboard
        from ..destination_factorization import qualify as destination
        from ..forward_attacks import qualify as forward

        classifiers = [qualify_castles.semantic_rejection, qualify_king_away.semantic_rejection,
                       bitboard.semantic_rejection, destination.semantic_rejection, forward.semantic_rejection]
        for location in ("step", "put", "public_builder_matches_recipe"):
            clean = ("expected x observed y Location: " + location + "\n").encode()
            hidden_prefix = b"WARNING unsafe backend failure\n" + b"x" * (1024 * 1024) + clean
            cases = (("complete", clean, 16 * 1024 * 1024, True, False, False),
                     ("tail-truncated", hidden_prefix, 16 * 1024 * 1024, False, False, True),
                     ("output-limit", clean, 4, False, True, False))
            for name, payload, cap, accepted, limited, truncated in cases:
                with self.subTest(location=location, capture=name):
                    result = self._capture_with_inert_child(payload, 1, cap)
                    self.assertEqual(result["process_exit_code"], 1)
                    self.assertEqual(result["exit_code"], 1 if accepted else None)
                    self.assertFalse(result["timed_out"])
                    self.assertEqual(result["output_limit_exceeded"], limited)
                    self.assertEqual(result["output_truncated"], truncated)
                    for classifier in classifiers:
                        self.assertEqual(classifier(result, location), accepted)
                    if location == "public_builder_matches_recipe":
                        self.assertEqual(qualify.semantic_rejection(result), accepted)

    def test_incomplete_capture_cannot_credit_any_shared_positive_consumer(self):
        from . import qualify_castles, qualify_king_away
        from ..bitboard_inventory import qualify as bitboard
        from ..destination_factorization import qualify as destination
        from ..forward_attacks import qualify as forward

        classifiers = [qualify.safe, qualify_castles.safe, qualify_king_away.safe,
                       bitboard.safe_result, destination.safe_result, forward.safe]
        clean = b"All terms check.\n"
        hidden_prefix = b"WARNING unsafe backend failure\n" + b"x" * (1024 * 1024) + clean
        cases = (("complete", clean, 16 * 1024 * 1024, True, False, False),
                 ("tail-truncated", hidden_prefix, 16 * 1024 * 1024, False, False, True),
                 ("output-limit", clean, 4, False, True, False))
        for name, payload, cap, accepted, limited, truncated in cases:
            with self.subTest(capture=name):
                result = self._capture_with_inert_child(payload, 0, cap)
                self.assertEqual(result["process_exit_code"], 0)
                self.assertEqual(result["exit_code"], 0 if accepted else None)
                self.assertFalse(result["timed_out"])
                self.assertEqual(result["output_limit_exceeded"], limited)
                self.assertEqual(result["output_truncated"], truncated)
                for classifier in classifiers:
                    self.assertEqual(classifier(result), accepted)

    def test_actual_main_distinct_reports_in_same_parent_own_distinct_persistent_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            suite = engine / "standalone/proofs/generator_contract"
            suite.mkdir(parents=True)
            for relative in ("standalone/proofs/table_preservation/focused.py",
                             "standalone/proofs/table_preservation/_validation.py",
                             "standalone/toolchain.json", "standalone/verify_compiler.js"):
                path = engine / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("inert source identity fixture\n")
            transcripts = (b"All terms check.\n", b"All terms check.\nWARNING: unsafe fixture\n")
            logs = []
            receipts = []
            def inert_child(_bun, _compiler, entry, _seconds, log_path=None):
                if entry.name == "consumer.bend":
                    transcript = transcripts[len(logs)]
                    logs.append(log_path)
                    log_path.write_bytes(transcript)
                    return {**self.result(text=transcript.decode()), "log_path": str(log_path)}
                return self.result(1, "expected x observed y Location: Closed.public_builder_matches_recipe")
            pin = qualify.subprocess.CompletedProcess([], 0, stdout="inert identity\n", stderr="")
            for name in ("one.json", "two.json"):
                report_path = root / name
                with patch.object(sys, "argv", ["qualify", "unused", "--report", str(report_path)]), \
                     patch.object(qualify, "SUITE", suite), patch.object(qualify, "ENGINE", engine), \
                     patch.object(qualify, "manifest"), patch.object(qualify, "load_closure", return_value=lambda *_: {}), \
                     patch.object(qualify, "replace"), patch.object(qualify.shutil, "copytree"), \
                     patch.object(qualify.subprocess, "run", return_value=pin), \
                     patch.object(qualify, "invoke", side_effect=inert_child), patch("builtins.print"):
                    if name == "one.json":
                        qualify.main()
                    else:
                        with self.assertRaisesRegex(AssertionError, "closed consumer did not return exact safe checker success"):
                            qualify.main()
                receipts.append(json.loads(report_path.read_text()))
            self.assertEqual(len(set(logs)), 2)
            self.assertEqual(receipts[0]["closed_builder_gate"], "PASS")
            self.assertEqual(receipts[1]["closed_builder_gate"], "NOT_COMPLETED")
            self.assertIn("closed consumer did not return exact safe checker success", receipts[1]["failure"])
            for path, receipt, transcript in zip(logs, receipts, transcripts):
                self.assertEqual(path.parent.parent, root)
                self.assertEqual(path.name, "checker.log")
                self.assertEqual(path.read_bytes(), transcript)
                self.assertEqual(receipt["consumer"]["raw_text"], transcript.decode())
                self.assertEqual(receipt["consumer_log_path"], str(path))
                self.assertEqual(receipt["consumer"]["log_path"], str(path))

    def test_supplied_log_is_exclusive_and_existing_evidence_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owned.log"
            path.write_bytes(b"prior evidence must remain unchanged")
            with patch.object(qualify.subprocess, "Popen") as child, \
                 patch.object(qualify.os, "killpg") as signal_call:
                with self.assertRaises(FileExistsError):
                    qualify.invoke("inert", Path("inert"), Path("inert"), 60, path)
            child.assert_not_called()
            signal_call.assert_not_called()
            self.assertEqual(path.read_bytes(), b"prior evidence must remain unchanged")

    def test_parent_exit_with_retained_stdout_cannot_credit_clean_tail(self):
        read_fd, write_fd = os.pipe()
        writer = [write_fd]
        os.write(write_fd, b"All terms check.\n")
        child = __import__("unittest.mock").mock.Mock()
        child.poll.return_value = 0
        child.returncode = 0
        child.stdout = os.fdopen(read_fd, "rb")
        def inert_group_cleanup(*_args):
            if writer[0] is not None:
                os.close(writer[0])
                writer[0] = None
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(qualify.subprocess, "Popen", return_value=child), \
                 patch.object(qualify, "OUTPUT_PIPE_GRACE_SECONDS", 0.01), \
                 patch.object(qualify.os, "killpg", side_effect=inert_group_cleanup), \
                 patch.object(qualify.signal, "signal", return_value=qualify.signal.SIG_DFL):
                result = qualify.invoke("inert", Path("inert"), Path("inert"), 60,
                                        Path(directory) / "capture.log")
            self.assertEqual(result["process_exit_code"], 0)
            self.assertIsNone(result["exit_code"])
            self.assertTrue(result["output_incomplete"])
            self.assertFalse(result["timed_out"])
            self.assertFalse(result["output_limit_exceeded"])
            self.assertFalse(result["output_truncated"])
            self.assertEqual(result["raw_text"], "All terms check.\n")
            self.assertFalse(qualify.safe(result))
        finally:
            inert_group_cleanup()
            child.stdout.close()

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
