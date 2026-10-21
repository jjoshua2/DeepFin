"""Actual-import owner/dispatcher/producer callback handoff regressions."""
from __future__ import annotations

import threading
import time
import unittest

from chess_anti_engine.producer_executor import ProducerExecutor
from chess_anti_engine.staged_execution import OwnerThreadCallbacks, guard_staged_backend
from chess_anti_engine.teacher_dispatch import TeacherDispatcher


class OwnerCallbackTests(unittest.TestCase):
    def test_dispatcher_guard_is_serviced_on_original_controller(self):
        owner = threading.get_ident()
        checked = []

        def check():
            self.assertEqual(threading.get_ident(), owner)
            checked.append(owner)

        bridge = OwnerThreadCallbacks(check, capacity=2, timeout=2, deadline=time.monotonic() + 5)
        dispatcher = TeacherDispatcher(guard_staged_backend(lambda rows: rows, bridge),
                                       target_rows=1, max_rows=2, batch_wait_ms=0,
                                       control=bridge)
        future = dispatcher.submit("guarded", (1,))
        try:
            deadline = time.monotonic() + 5
            while not future.done() and time.monotonic() < deadline:
                bridge.service()
                time.sleep(0.001)
            self.assertEqual(future.result(0), (1,))
            self.assertTrue(checked)
        finally:
            # Real close waits are bounded; keep servicing a final guard check
            # rather than blocking its original owner inside one long join.
            deadline = time.monotonic() + 5
            while dispatcher.is_alive and time.monotonic() < deadline:
                bridge.service()
                try:
                    dispatcher.close(timeout=0.01)
                except TimeoutError:
                    pass
            dispatcher.close(timeout=0)
            bridge.close()

    def test_producer_join_services_owner_guard_without_background_execution(self):
        owner = threading.get_ident()
        reached = threading.Event()
        checked = []

        def check():
            self.assertEqual(threading.get_ident(), owner)
            checked.append(owner)

        bridge = OwnerThreadCallbacks(check, capacity=1, timeout=2, deadline=time.monotonic() + 5)
        producers = ProducerExecutor(max_workers=1, capacity=1)

        def produce():
            reached.set()
            bridge()
            return 42

        future = producers.submit(produce)
        self.assertTrue(reached.wait(5))
        try:
            producers.finish(timeout=2, service=bridge.service)
            self.assertEqual(future.result(0), 42)
            self.assertEqual(checked, [owner])
        finally:
            bridge.close()
            producers.finish(timeout=2, service=bridge.service)

    def test_owner_stop_fails_waiting_producer_and_join_remains_quiescent(self):
        owner = threading.get_ident()
        reached = threading.Event()

        def check():
            self.assertEqual(threading.get_ident(), owner)
            raise RuntimeError("original owner STOP")

        bridge = OwnerThreadCallbacks(check, capacity=1, timeout=2, deadline=time.monotonic() + 5)
        producers = ProducerExecutor(max_workers=1, capacity=1)

        def produce():
            reached.set()
            bridge()

        future = producers.submit(produce)
        self.assertTrue(reached.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "original owner STOP"):
                producers.finish(timeout=2, service=bridge.service)
            self.assertTrue(future.done())
            with self.assertRaisesRegex(RuntimeError, "original owner STOP"):
                bridge()
        finally:
            bridge.close()

    def test_unserviced_timeout_cannot_execute_guard_later(self):
        checked = []
        errors = []
        bridge = OwnerThreadCallbacks(lambda: checked.append(True), capacity=1,
                                      timeout=0.02, deadline=time.monotonic() + 5)

        def call():
            try:
                bridge()
            except TimeoutError:
                errors.append("expired")

        thread = threading.Thread(target=call)
        thread.start()
        thread.join(2)
        try:
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["expired"])
            bridge.service()
            self.assertEqual(checked, [])
        finally:
            bridge.close()

    def test_background_service_is_refused_and_closed_bridge_fails_caller(self):
        checked = []
        errors = []
        bridge = OwnerThreadCallbacks(lambda: checked.append(True), capacity=1,
                                      timeout=1, deadline=time.monotonic() + 5)

        def serve_elsewhere():
            try:
                bridge.service()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=serve_elsewhere)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["owner callbacks require original controller thread"])
        bridge.close()
        with self.assertRaisesRegex(RuntimeError, "bridge closed"):
            bridge()
        self.assertEqual(checked, [])


if __name__ == "__main__":
    unittest.main()
