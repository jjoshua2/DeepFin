"""Host-verifier regression tests; no Bend compiler or native candidate is run.

Run with both `python3 test_harness.py` and `python3 -O test_harness.py`.
The synthetic transcripts model probe.bend's exact wire format. These are parser,
receipt, and dispatch-inventory tests, not new formal laws or native executions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import focused, verify_native as native
    from ._validation import begin_report, require, write_report
else:
    import focused
    import verify_native as native
    from _validation import begin_report, require, write_report


SIZES = [64, 1024, 131072, 131072]


def cells(size: int) -> list[str]:
    return [f'cell {i ^ 2779096485} {(i * 2654435761 + 12345) & 0xffffffff}' for i in range(size)]


def transcript(op: int = 1, context: int = 0) -> tuple[str, dict]:
    size = SIZES[context]
    snapshot = cells(size)
    outputs = ['word 4294967295 0'] if op == 0 else ['bool 0'] if op == 1 else ['move 4 6 0 2', 'end']
    lines = [f'before {size}', *snapshot, *outputs, f'after {size}', *snapshot, f'done {size}']
    return '\n'.join(lines) + '\n', {'input': [context, op]}


def edit_lines(text: str, edit) -> str:
    lines = text.splitlines()
    edit(lines)
    return '\n'.join(lines) + '\n'


class TranscriptTests(unittest.TestCase):
    def test_all_operation_records(self) -> None:
        for op in range(9):
            with self.subTest(op=op):
                text, case = transcript(op)
                result = native.parse_observation(text, case)
                self.assertEqual(result['cells'], 64)
                self.assertEqual(result['changed_cells'], [])
                self.assertEqual(result['moves'], 0 if op < 2 else 1)
                self.assertEqual(result['before_sha256'], result['after_sha256'])

    def test_all_complete_context_sizes(self) -> None:
        for context, size in enumerate(SIZES):
            with self.subTest(context=context):
                text, case = transcript(context=context)
                self.assertEqual(native.parse_observation(text, case)['cells'], size)

    def test_empty_move_list(self) -> None:
        text, case = transcript(2)
        self.assertEqual(native.parse_observation(text.replace('move 4 6 0 2\n', ''), case)['moves'], 0)

    def test_raw_u32_move_fields_not_claimed_legal(self) -> None:
        text, case = transcript(2)
        text = text.replace('move 4 6 0 2', 'move 4294967295 64 12 4294967295')
        self.assertEqual(native.parse_observation(text, case)['moves'], 1)

    def test_unchanged_answer_hash_format(self) -> None:
        text, case = transcript(0)
        result = native.parse_observation(text, case)
        self.assertEqual(result['answers_sha256'], hashlib.sha256(b'word 4294967295 0').hexdigest())

    def test_last_cell_corruption_is_detected(self) -> None:
        text, case = transcript()
        changed = edit_lines(text, lambda lines: lines.__setitem__(-2, 'cell 0 1'))
        with self.assertRaises(native.WrongValue):
            native.parse_observation(changed, case)
        diagnostic = native.parse_observation(changed, case, permit_corruption=True)
        self.assertEqual(diagnostic['changed_cells'], [63])
        self.assertEqual(diagnostic['cells'], 64)

    def test_permit_corruption_does_not_allow_truncation(self) -> None:
        text, case = transcript()
        truncated = text.splitlines()[:68]  # before64, answer, after, one returned cell
        with self.assertRaises(AssertionError):
            native.parse_observation('\n'.join(truncated) + '\n', case, True)

    def test_real_observe_wrapper_validates_subprocess_output(self) -> None:
        text, case = transcript()
        result = subprocess.CompletedProcess(['fake'], 0, text, '')
        with patch.object(native, 'run', return_value=result) as invoked:
            self.assertEqual(native.observe(Path('/not-a-native-run'), case)['cells'], 64)
            invoked.assert_called_once()

    def test_real_observe_wrapper_rejects_short_success(self) -> None:
        text, case = transcript()
        result = subprocess.CompletedProcess(['fake'], 0, '\n'.join(text.splitlines()[:68]) + '\n', '')
        with patch.object(native, 'run', return_value=result):
            with self.assertRaises(AssertionError):
                native.observe(Path('/not-a-native-run'), case)


# Each malformed transcript becomes an independently counted unittest test case.
INVALID_TRANSCRIPTS = {
    'empty': lambda s: '',
    'missing_final_terminator': lambda s: '\n'.join(s.splitlines()[:-1]) + '\n',
    'one_returned_cell_no_terminator': lambda s: '\n'.join(s.splitlines()[:68]) + '\n',
    'no_returned_cells': lambda s: '\n'.join(s.splitlines()[:67]) + '\n',
    'one_returned_cell_with_terminator': lambda s: '\n'.join(s.splitlines()[:68] + ['done 64']) + '\n',
    'last_cell_missing_with_terminator': lambda s: edit_lines(s, lambda x: x.pop(-2)),
    'extra_returned_cell': lambda s: edit_lines(s, lambda x: x.insert(-1, 'cell 0 0')),
    'trailing_garbage': lambda s: s + 'garbage\n',
    'extra_blank_line': lambda s: s + '\n',
    'wrong_before_size': lambda s: s.replace('before 64', 'before 63'),
    'wrong_after_size': lambda s: s.replace('after 64', 'after 63'),
    'wrong_done_size': lambda s: s.replace('done 64', 'done 63'),
    'wrong_cell_tag': lambda s: s.replace('cell ', 'word ', 1),
    'negative_cell': lambda s: edit_lines(s, lambda x: x.__setitem__(-2, 'cell -1 0')),
    'overflow_cell': lambda s: edit_lines(s, lambda x: x.__setitem__(-2, 'cell 4294967296 0')),
    'noninteger_cell': lambda s: edit_lines(s, lambda x: x.__setitem__(-2, 'cell x 0')),
    'signed_positive_cell': lambda s: edit_lines(s, lambda x: x.__setitem__(-2, 'cell +1 0')),
    'extra_cell_field': lambda s: edit_lines(s, lambda x: x.__setitem__(-2, 'cell 1 2 3')),
    'invalid_boolean': lambda s: s.replace('bool 0', 'bool 2'),
    'wrong_boolean_tag': lambda s: s.replace('bool 0', 'word 0'),
    'boolean_extra_field': lambda s: s.replace('bool 0', 'bool 0 0'),
    'boolean_multiple_records': lambda s: s.replace('bool 0', 'bool 0\nbool 1'),
    'incorrect_raw_seed': lambda s: s.replace('cell 2779096485 12345', 'cell 0 0'),
}


def malformed_test(transform):
    def test(self):
        text, case = transcript()
        with self.assertRaises(AssertionError):
            native.parse_observation(transform(text), case)
    return test


for label, transform in INVALID_TRANSCRIPTS.items():
    setattr(TranscriptTests, 'test_reject_' + label, malformed_test(transform))


class AnswerTests(unittest.TestCase):
    def test_reject_wrong_word_tag(self) -> None:
        text, case = transcript(0)
        with self.assertRaises(AssertionError):
            native.parse_observation(text.replace('word 4294967295 0', 'move 4294967295 0'), case)

    def test_reject_word_overflow(self) -> None:
        text, case = transcript(0)
        with self.assertRaises(AssertionError):
            native.parse_observation(text.replace('word 4294967295 0', 'word 4294967296 0'), case)

    def test_reject_bad_move_fields(self) -> None:
        text, case = transcript(2)
        for replacement in ('move x 6 0 2', 'move -1 6 0 2', 'move 4 6 0 4294967296', 'move 4 6 2', 'move 4 6 0 2 1'):
            with self.subTest(record=replacement), self.assertRaises(AssertionError):
                native.parse_observation(text.replace('move 4 6 0 2', replacement), case)

    def test_reject_missing_list_terminator(self) -> None:
        text, case = transcript(2)
        with self.assertRaises(AssertionError):
            native.parse_observation(text.replace('end\n', ''), case)

    def test_reject_invalid_case_selector(self) -> None:
        text, _ = transcript()
        for request in ([4, 1], [-1, 1], [0, 9], [True, 1], [0, '1']):
            with self.subTest(request=request), self.assertRaises(AssertionError):
                native.parse_observation(text, {'input': request})


class AlwaysOnTests(unittest.TestCase):
    def test_runtime_require_survives_optimization(self) -> None:
        with self.assertRaises(AssertionError):
            require(False, 'intentional failure')

    def test_no_debug_asserts_in_gate_sources(self) -> None:
        for name in ('focused.py', 'verify_native.py', '_validation.py'):
            nodes = ast.walk(ast.parse(Path(__file__).with_name(name).read_text()))
            self.assertFalse(any(isinstance(node, ast.Assert) for node in nodes), name)

    def test_existing_dispatch_partition(self) -> None:
        result = focused.dispatch_audit()
        self.assertEqual(result['disjoint_boolean_clauses'], 153)
        self.assertEqual(result['raw_u32_values_covered'], 1 << 32)

    def _reject_dispatch_edit(self, transform) -> None:
        text = (focused.SUITE / 'Queries.bend').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Queries.bend').write_text(transform(text))
            with patch.object(focused, 'SUITE', root):
                with self.assertRaises(AssertionError):
                    focused.dispatch_audit()

    def test_dispatch_missing_case_rejected(self) -> None:
        self._reject_dispatch_edit(lambda s: s.replace('    case ', '    removed ', 1))

    def test_dispatch_overlap_rejected(self) -> None:
        import re
        self._reject_dispatch_edit(lambda s: re.sub(r'^    case .*:$', '    case ' + ' '.join(['_'] * 32) + ':', s, count=1, flags=re.MULTILINE))

    def test_dispatch_wrong_width_rejected(self) -> None:
        import re
        self._reject_dispatch_edit(lambda s: re.sub(r'^    case .*:$', '    case _:', s, count=1, flags=re.MULTILINE))

    def test_source_success_requires_exact_output(self) -> None:
        focused.safe({'exit_code': 0, 'raw_text': 'All terms check.\n'})
        for receipt in ({'exit_code': 1, 'raw_text': 'All terms check.'},
                        {'exit_code': 0, 'raw_text': 'All terms check.\nWARNING'},
                        {'exit_code': 0, 'raw_text': ''}):
            with self.subTest(receipt=receipt), self.assertRaises(AssertionError):
                focused.safe(receipt)

    def test_native_failed_exit_and_stderr_rejected(self) -> None:
        for result in (subprocess.CompletedProcess(['fake'], 1, '', ''),
                       subprocess.CompletedProcess(['fake'], 0, '', 'warning')):
            with self.subTest(result=result), patch.object(native.subprocess, 'run', return_value=result):
                with self.assertRaises(AssertionError):
                    native.run(['fake'])


class ReceiptTests(unittest.TestCase):
    def test_begin_replaces_previous_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'report.json'
            write_report(p, {'native_gate': 'PASS'})
            begin_report(p, 'native_gate')
            self.assertEqual(json.loads(p.read_text())['native_gate'], 'NOT_COMPLETED')

    def test_complete_report_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'reports' / 'report.json'
            begin_report(p, 'focused_gate')
            write_report(p, {'focused_gate': 'PASS', 'test_only': True})
            self.assertEqual(json.loads(p.read_text()), {'focused_gate': 'PASS', 'test_only': True})
            self.assertEqual(list(p.parent.glob('.*.tmp')), [])

    def test_encoding_failure_leaves_nonpass_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'report.json'
            begin_report(p, 'native_gate')
            with self.assertRaises(TypeError):
                write_report(p, {'native_gate': 'PASS', 'unserializable': object()})
            self.assertEqual(json.loads(p.read_text())['native_gate'], 'NOT_COMPLETED')
            self.assertEqual(list(p.parent.glob('.*.tmp')), [])

    def test_focused_main_invalidates_old_pass_before_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'focused.json'
            write_report(p, {'focused_gate': 'PASS'})
            with patch.object(sys, 'argv', ['focused.py', '/missing', '--report', str(p)]), \
                    patch.object(focused.subprocess, 'run', side_effect=FileNotFoundError('test compiler')):
                with self.assertRaises(FileNotFoundError):
                    focused.main()
            self.assertEqual(json.loads(p.read_text())['focused_gate'], 'NOT_COMPLETED')

    def test_native_main_invalidates_old_pass_before_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'native.json'
            write_report(p, {'native_gate': 'PASS'})
            with patch.object(sys, 'argv', ['verify_native.py', '/missing', '--report', str(p)]), \
                    patch.object(native, 'run', side_effect=FileNotFoundError('test compiler')):
                with self.assertRaises(FileNotFoundError):
                    native.main()
            self.assertEqual(json.loads(p.read_text())['native_gate'], 'NOT_COMPLETED')


class CliReceiptTests(unittest.TestCase):
    """Exercise real CLI processes; the selected compiler is deliberately absent."""

    def _failed_cli(self, runner: str, gate: str, optimization: str | None) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "current.json"
            history = root / "historical.json"
            previous = b'{"status": "PASS", "historical": true}\n'
            history.write_bytes(previous)
            write_report(report, {gate: "PASS", "stale": True})
            command = [sys.executable]
            if optimization is not None:
                command.append(optimization)
            command += [str(Path(__file__).with_name(runner)), str(root / "compiler"),
                        "--report", str(report)]
            # No mocks: both entry points start, invalidate the old receipt and then
            # fail to execute the explicitly selected nonexistent Bun executable.
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False,
                env={**os.environ, "BUN": str(root / "missing-bun")},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)
            current = json.loads(report.read_text())
            self.assertEqual(current[gate], "NOT_COMPLETED")
            self.assertNotIn("stale", current)
            self.assertNotEqual(current["pid"], os.getpid())
            self.assertEqual(history.read_bytes(), previous)
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_focused_cli_normal_invalidates_old_pass(self) -> None:
        self._failed_cli("focused.py", "focused_gate", None)

    def test_focused_cli_optimized_invalidates_old_pass(self) -> None:
        self._failed_cli("focused.py", "focused_gate", "-O")

    def test_native_cli_normal_invalidates_old_pass(self) -> None:
        self._failed_cli("verify_native.py", "native_gate", None)

    def test_native_cli_optimized_invalidates_old_pass(self) -> None:
        self._failed_cli("verify_native.py", "native_gate", "-O")


if __name__ == '__main__':
    unittest.main(verbosity=2)
