"""Stdlib-only source-sliced tests: no scorer, model or project module imports.

The deliberately bounded extraction executes the real path helper and actual
early/late callsite statements, stopping at the model-acquisition boundary.
It does not establish full CLI/scorer integration or concurrent-mutator safety.
Owned fixture directories are retained for the qualification evidence.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Protocol, TypeGuard
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class _InputGuard(Protocol):
    def __call__(
        self, dump_path: str | Path | None, *, audit_set: str | Path,
        matched_rows: Path | None,
    ) -> None: ...


class _ReportWriter(Protocol):
    def __call__(
        self, path: Path, _materialised: list[dict[str, Any]], stamp: dict[str, Any], /,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _Fixture:
    root: Path
    audit: Path
    index: Path
    originals: dict[Path, bytes]
    # exec populates this heterogeneous namespace with actual extracted code.
    env: dict[str, Any]
    guard: _InputGuard
    preflight: list[ast.stmt]
    late_guard: ast.Expr
    default_matched_path: Callable[[str | Path], Path]
    write_report: _ReportWriter


@contextmanager
def _raises_system_exit(pattern: str) -> Generator[None, None, None]:
    caught_message = ""
    try:
        yield
    except SystemExit as exc:
        caught_message = str(exc)
    else:
        raise AssertionError("expected refusal before any write")
    assert re.search(pattern, caught_message), caught_message


def _execute(nodes: Sequence[ast.stmt], env: dict[str, Any]) -> None:
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<exact-path-slice>", "exec"), env)


class DumpInputPreservationTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._fixture: _Fixture | None = None

    @property
    def fixture(self) -> _Fixture:
        assert self._fixture is not None, "setUp must complete before fixture use"
        return self._fixture

    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="vt01-", dir=os.environ.get("VT01_FIXTURE_ROOT")))
        audit = root / "audit.jsonl"
        index = root / "matched.npz"
        audit.write_bytes(b'{"original":"audit"}\n')
        index.write_bytes(b'inert-matched-index-original\n')
        originals = {audit: audit.read_bytes(), index: index.read_bytes()}
        for path, data in originals.items():
            path.with_name(path.name + ".untouched-original").write_bytes(data)
        tree = ast.parse((ROOT / "scripts/value_regret.py").read_text())
        history = ast.parse((ROOT / "chess_anti_engine/eval/audit_history.py").read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        history_functions = {node.name: node for node in history.body if isinstance(node, ast.FunctionDef)}
        env: dict[str, Any] = {"Path": Path, "json": json}
        for node in history.body:
            if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id in ("INPUT_ENCODINGS", "INPUT_ENCODING_DEFAULT")):
                assert node.value is not None, "encoding constant must have a value"
                env[node.target.id] = ast.literal_eval(node.value)
        _execute([functions["_refuse_dump_input_aliases"],
                  history_functions["normalize_input_encoding"],
                  history_functions["default_matched_rows_path"]], env)
        guard: _InputGuard = env["_refuse_dump_input_aliases"]
        assert callable(guard)
        body = functions["main"].body
        parse_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets))
        net_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "net" for t in node.targets))
        preflight = body[parse_index + 1:net_index]
        def calls(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == name)
        assert any(calls(n, "_refuse_dump_input_aliases") for node in preflight for n in ast.walk(node))
        assert not any(calls(n, "apply_gpu_mem_cap") or calls(n, "value_1ply_regret")
                       for node in preflight for n in ast.walk(node))
        dump_block = next(node for node in body if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Attribute) and node.test.attr == "dump_per_position")
        late_guard = dump_block.body[0]
        assert isinstance(late_guard, ast.Expr)
        assert calls(late_guard.value, "_refuse_dump_input_aliases")
        writer = next(node for node in ast.walk(dump_block) if calls(node, "write_audit_cache"))
        force_value = next(k.value for k in writer.keywords if k.arg == "force")
        assert isinstance(force_value, ast.Constant)
        assert force_value.value
        cache_tree = ast.parse((ROOT / "chess_anti_engine/eval/audit_cache.py").read_text())
        cache_writer = next(node for node in cache_tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "write_audit_cache")
        tail_start = next(i for i, node in enumerate(cache_writer.body)
                          if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "mkdir")
        io_tail = ast.FunctionDef(
            name="writer_io_tail",
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
                               ("path", "materialised", "stamp")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=cache_writer.body[tail_start:], decorator_list=[],
        )
        _execute([io_tail], env)
        default_matched_path: Callable[[str | Path], Path] = env["default_matched_rows_path"]
        write_report: _ReportWriter = env["writer_io_tail"]
        assert callable(default_matched_path)
        assert callable(write_report)
        self._fixture = _Fixture(
            root, audit, index, originals, env, guard, preflight, late_guard,
            default_matched_path, write_report,
        )

    def assert_originals(self) -> None:
        for path, original in self.fixture.originals.items():
            assert path.read_bytes() == original
            assert path.with_name(path.name + ".untouched-original").read_bytes() == original

    def args(
        self, output: str | Path | None, encoding: str = "fen_only", index: Path | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(dump_per_position=output, audit_set=self.fixture.audit,
                               input_encoding=encoding, matched_rows=index)

    def run_preflight(self, args: SimpleNamespace) -> dict[str, Any]:
        env: dict[str, Any] = {**self.fixture.env, "args": args}
        _execute(self.fixture.preflight, env)
        return env

    def test_aliases_refuse_without_changing_original_bytes(self):
        for path, flag in ((self.fixture.audit, "--audit-set"), (self.fixture.index, "--matched-rows")):
            aliases = [path, Path(os.path.relpath(path))]
            symlink = self.fixture.root / (path.name + ".symlink")
            symlink.symlink_to(path.name)
            hardlink = self.fixture.root / (path.name + ".hardlink")
            os.link(path, hardlink)
            aliases.extend([symlink, hardlink])
            assert hardlink.resolve() != path.resolve()
            for output in aliases:
                with self.subTest(input=flag, output=str(output)):
                    with _raises_system_exit(flag):
                        self.fixture.guard(output, audit_set=self.fixture.audit, matched_rows=self.fixture.index)
                    self.assert_originals()

    def test_explicit_index_is_consumed_in_both_encodings(self):
        for encoding in ("fen_only", "stored"):
            with self.subTest(encoding=encoding):
                with _raises_system_exit("--matched-rows"):
                    self.run_preflight(self.args(self.fixture.index, encoding, self.fixture.index))
                self.assert_originals()

    def test_default_index_consumed_only_in_stored(self):
        default = self.fixture.default_matched_path(self.fixture.audit)
        default.write_bytes(b"inert-default-index\n")
        before = default.read_bytes()
        with _raises_system_exit("--matched-rows"):
            self.run_preflight(self.args(default, "stored"))
        assert default.read_bytes() == before
        env = self.run_preflight(self.args(default, "fen_only"))
        assert env["matched_path"] is None
        assert default.read_bytes() == before
        self.assert_originals()

    def test_preflight_refuses_before_net_acquisition_boundary(self):
        # Execute the exact preflight prefix only. A normal return reaches the
        # real next statement (net_source_from_args); a refusal cannot reach it.
        with _raises_system_exit("--audit-set"):
            self.run_preflight(self.args(self.fixture.audit))
        env = self.run_preflight(self.args(self.fixture.root / "new-report.jsonl"))
        assert env["matched_path"] is None
        self.assert_originals()

    def test_late_callsite_detects_output_retargeted_after_preflight(self):
        output = self.fixture.root / "report.jsonl"
        env = self.run_preflight(self.args(output, index=self.fixture.index))
        output.symlink_to(self.fixture.index.name)
        with _raises_system_exit("--matched-rows"):
            _execute([self.fixture.late_guard], env)
        self.assert_originals()

    def test_identity_errors_fail_closed_without_changing_originals(self):
        output = self.fixture.root / "report.jsonl"
        output.write_bytes(b"old-report\n")
        before = output.read_bytes()
        real_stat = Path.stat
        for failing in (self.fixture.audit, output):
            def stat(path, *args, _failing=failing, **kwargs):
                if path == _failing:
                    raise PermissionError("owned-fixture identity denied")
                return real_stat(path, *args, **kwargs)
            with (
                self.subTest(failing=str(failing)), patch.object(Path, "stat", stat),
                _raises_system_exit("cannot establish"),
            ):
                self.fixture.guard(output, audit_set=self.fixture.audit, matched_rows=self.fixture.index)
            assert output.read_bytes() == before
            self.assert_originals()
        with (
            patch.object(Path, "resolve", side_effect=RuntimeError("fixture symlink loop")),
            _raises_system_exit("cannot establish"),
        ):
            self.fixture.guard(output, audit_set=self.fixture.audit, matched_rows=self.fixture.index)
        assert output.read_bytes() == before
        self.assert_originals()

    def test_missing_consumed_input_is_not_treated_as_known_separate(self):
        with _raises_system_exit("cannot establish"):
            self.fixture.guard(self.fixture.root / "new-report", audit_set=self.fixture.audit,
                       matched_rows=self.fixture.root / "missing-index")
        self.assert_originals()

    def test_unrelated_report_replacement_and_no_dump_remain_allowed(self):
        output = self.fixture.root / "report.jsonl"
        self.fixture.guard(output, audit_set=self.fixture.audit, matched_rows=self.fixture.index)
        output.write_bytes(b"old-report\n")
        self.fixture.guard(output, audit_set=self.fixture.audit, matched_rows=self.fixture.index)
        self.fixture.write_report(output, [{"value": 1.0}], {"fixture_stamp": True, "rows": 1})
        assert json.loads(output.read_text().splitlines()[1]) == {"value": 1.0}
        self.fixture.guard(None, audit_set=self.fixture.root / "unused-missing-input", matched_rows=None)
        self.assert_originals()


if __name__ == "__main__":
    unittest.main(verbosity=2)
