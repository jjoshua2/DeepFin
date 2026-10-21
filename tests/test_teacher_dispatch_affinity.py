"""Actual-import CPU regressions for dispatcher affinity and acceptance barriers.

These fake backends replace only the session operation. The real dispatcher,
staged bridge, FIFO routing, Futures, threads and shutdown protocol run normally.
"""
from __future__ import annotations

import threading
import unittest

from chess_anti_engine.staged_execution import StagedHostBackend, guard_staged_backend
from chess_anti_engine.teacher_dispatch import TeacherDispatcher


class _Backend:
    def prepare(self, rows):
        return tuple(rows)

    def execute(self, prepared):
        return prepared

    def finish(self, _prepared, raw):
        return tuple(value * 10 for value in raw)


class TeacherAffinityTests(unittest.TestCase):
    def test_actual_guard_and_execute_run_on_dispatcher_not_session_executor(self):
        calls = []

        def guard():
            calls.append(threading.current_thread().name)
            self.assertEqual(threading.current_thread().name, "TeacherDispatcher")

        class Backend(_Backend):
            def execute(inner, prepared):
                # Also exercise a nested callback, as the real BT4 adapter does.
                guard()
                return super().execute(prepared)

        dispatcher = TeacherDispatcher(guard_staged_backend(Backend(), guard),
                                       target_rows=1, max_rows=2, batch_wait_ms=0)
        try:
            self.assertEqual(dispatcher.submit("first", (3,)).result(5), (30,))
            self.assertEqual(calls, ["TeacherDispatcher"] * 2)
        finally:
            dispatcher.close(timeout=5)

    def test_legacy_callable_guard_has_same_dispatcher_affinity(self):
        def guard():
            self.assertEqual(threading.current_thread().name, "TeacherDispatcher")

        dispatcher = TeacherDispatcher(guard_staged_backend(lambda rows: rows, guard),
                                       target_rows=1, max_rows=2, batch_wait_ms=0)
        try:
            self.assertEqual(dispatcher.submit("legacy", (7,)).result(5), (7,))
        finally:
            dispatcher.close(timeout=5)

    def test_next_prepare_overlaps_but_next_execute_waits_for_finish_acceptance(self):
        prepared_second = threading.Event()
        finishing_first = threading.Event()
        release_finish = threading.Event()
        executed_second = threading.Event()

        class Backend(_Backend):
            def prepare(inner, rows):
                if rows == (2,):
                    prepared_second.set()
                return super().prepare(rows)

            def execute(inner, prepared):
                if prepared == (2,):
                    executed_second.set()
                return super().execute(prepared)

            def finish(inner, prepared, raw):
                if prepared == (1,):
                    if not prepared_second.wait(5):
                        raise AssertionError("second prepare did not overlap first finish")
                    finishing_first.set()
                    if not release_finish.wait(5):
                        raise AssertionError("test finish release deadline")
                return super().finish(prepared, raw)

        dispatcher = TeacherDispatcher(Backend(), target_rows=1, max_rows=2, batch_wait_ms=0)
        first, second = dispatcher.submit_many((("first", (1,)), ("second", (2,))))
        try:
            self.assertTrue(finishing_first.wait(5))
            self.assertFalse(executed_second.wait(0.1), "next session raced unaccepted finish")
            self.assertFalse(first.done())
            release_finish.set()
            self.assertEqual(first.result(5), (10,))
            self.assertEqual(second.result(5), (20,))
            self.assertEqual(dispatcher.stage_stats["peak_slots"], 2)
        finally:
            release_finish.set()
            dispatcher.close(timeout=5)

    def _invalid_finish(self, *, raises):
        executed = []
        second_prepared = threading.Event()

        class Backend(_Backend):
            def prepare(inner, rows):
                if rows == (2,):
                    second_prepared.set()
                return super().prepare(rows)

            def execute(inner, prepared):
                executed.append(prepared)
                return super().execute(prepared)

            def finish(inner, prepared, raw):
                if not second_prepared.wait(5):
                    raise AssertionError("second prepare missing")
                if raises:
                    raise ValueError("invalid policy contract")
                return ()

        dispatcher = TeacherDispatcher(Backend(), target_rows=1, max_rows=2, batch_wait_ms=0)
        first, second = dispatcher.submit_many((("first", (1,)), ("second", (2,))))
        try:
            message = "invalid policy" if raises else "output count"
            for future in (first, second):
                with self.assertRaisesRegex(ValueError, message):
                    future.result(5)
            self.assertEqual(executed, [(1,)])
            with self.assertRaisesRegex(RuntimeError, "failed"):
                dispatcher.submit("later", (3,))
        finally:
            dispatcher.close(timeout=5)

    def test_wrong_output_count_stops_next_already_prepared_session(self):
        self._invalid_finish(raises=False)

    def test_finish_error_stops_next_already_prepared_session(self):
        self._invalid_finish(raises=True)

    def test_guard_refusal_prevents_session_and_fails_all_owned_requests(self):
        executed = []

        def refuse():
            raise RuntimeError("STOP before session")

        class Backend(_Backend):
            def execute(inner, prepared):
                executed.append(prepared)
                return super().execute(prepared)

        dispatcher = TeacherDispatcher(guard_staged_backend(Backend(), refuse),
                                       target_rows=1, max_rows=2, batch_wait_ms=0)
        futures = dispatcher.submit_many((("first", (1,)), ("second", (2,))))
        try:
            for future in futures:
                with self.assertRaisesRegex(RuntimeError, "STOP before session"):
                    future.result(5)
            self.assertEqual(executed, [])
        finally:
            dispatcher.close(timeout=5)

    def test_fifo_scatter_split_requests_and_complete_futures(self):
        dispatcher = TeacherDispatcher(_Backend(), target_rows=2, max_rows=8, batch_wait_ms=0)
        try:
            futures = dispatcher.submit_many((("a", (1, 2, 3)), ("b", (4,)), ("c", (5, 6, 7))))
            self.assertEqual([future.result(5) for future in futures],
                             [(10, 20, 30), (40,), (50, 60, 70)])
            self.assertEqual(dispatcher.histogram, {2: 3, 1: 1})
        finally:
            dispatcher.close(timeout=5)

    def test_close_timeout_retains_blocked_session_and_owner(self):
        entered = threading.Event()
        release = threading.Event()

        class Backend(_Backend):
            def execute(inner, prepared):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test session release deadline")
                return super().execute(prepared)

        dispatcher = TeacherDispatcher(Backend(), target_rows=1, max_rows=2, batch_wait_ms=0)
        future = dispatcher.submit("blocked", (1,))
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(TimeoutError, "ownership retained"):
                dispatcher.close(timeout=0.01)
            self.assertTrue(dispatcher.is_alive)
            self.assertFalse(future.done())
            release.set()
            self.assertEqual(future.result(5), (10,))
        finally:
            release.set()
            dispatcher.close(timeout=5)

    def test_stop_aware_prepare_wait_retains_blocked_preparation_on_close(self):
        preparing = threading.Event()
        release = threading.Event()
        stop = threading.Event()
        executed = []

        class Backend(_Backend):
            def prepare(inner, rows):
                preparing.set()
                if not release.wait(5):
                    raise AssertionError("test prepare release deadline")
                return super().prepare(rows)

            def execute(inner, prepared):
                executed.append(prepared)
                return super().execute(prepared)

        def control():
            self.assertEqual(threading.current_thread().name, "TeacherDispatcher")
            if stop.is_set():
                raise RuntimeError("STOP during preparation")

        dispatcher = TeacherDispatcher(Backend(), target_rows=1, max_rows=2,
                                       batch_wait_ms=0, control=control)
        future = dispatcher.submit("blocked", (1,))
        try:
            self.assertTrue(preparing.wait(5))
            stop.set()
            with self.assertRaisesRegex(RuntimeError, "STOP during preparation"):
                future.result(5)
            self.assertEqual(executed, [])
            with self.assertRaisesRegex(TimeoutError, "ownership retained"):
                dispatcher.close(timeout=0.01)
            self.assertTrue(dispatcher.is_alive)
        finally:
            release.set()
            dispatcher.close(timeout=5)

    def test_host_rejects_cross_thread_retirement_and_preserves_receipt(self):
        host = StagedHostBackend(_Backend())
        receipt = host.submit((1,))
        errors = []

        def retire_elsewhere():
            try:
                host.release(receipt)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=retire_elsewhere)
        thread.start()
        thread.join(5)
        try:
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["staged execution requires its dispatcher owner thread"])
            self.assertFalse(receipt.future.done())
            self.assertEqual(host.release(receipt), (10,))
        finally:
            host.close()


if __name__ == "__main__":
    unittest.main()
