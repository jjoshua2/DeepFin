"""Pure stamp-function fixtures; never import the panel/model/CLI runtime."""
from __future__ import annotations

import ast
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

KEY = "blindspot_panel_scorer_version"
VERSION = "wdl-logits-softmax-v1"


def _stamp_gate() -> Any:
    leaf = ast.parse(Path("chess_anti_engine/utils/audit_cache_format.py").read_text())
    assignments = [n for n in leaf.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    # Only exact literal format vocabulary; never project imports or functions.
    for node in assignments:
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            assert isinstance(call.func, ast.Name)
            assert call.func.id == "frozenset"
    reader = ast.parse(Path("scripts/paired_compare.py").read_text())
    gate = next(n for n in reader.body
                if isinstance(n, ast.FunctionDef) and n.name == "require_same_stamp")
    calls = {"isinstance", "version_a.strip", "version_b.strip", "a.stamp.get",
             "b.stamp.get", "SystemExit", "print", "sorted", "set", "json.dumps"}
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(gate))
    for call in (n for n in ast.walk(gate) if isinstance(n, ast.Call)):
        assert ast.unparse(call.func) in calls, ast.unparse(call.func)
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*future, *assignments, gate], type_ignores=[])
    namespace: dict[str, Any] = {"json": json}
    exec(compile(module, "<exact-paired-stamp-function>", "exec"), namespace)
    return namespace["require_same_stamp"]


def _check(a: dict[str, Any], b: dict[str, Any], *, refuses: bool) -> str:
    output = io.StringIO()
    rejected = False
    rejection_message = ""
    with redirect_stdout(output):
        try:
            _stamp_gate()(SimpleNamespace(stamp=a), SimpleNamespace(stamp=b),
                          label_a="baseline", label_b="candidate")
        except SystemExit as exc:
            rejected = True
            rejection_message = str(exc)
    if rejected and (KEY in a or KEY in b):
        assert "remeasure BOTH sides" in rejection_message
    assert rejected == refuses, (a, b, output.getvalue())
    return output.getvalue()


def test_corrected_panel_requires_matching_scorer_on_both_sides() -> None:
    corrected = {"audit_cache_format": 1, KEY: VERSION}
    for legacy in ({"audit_cache_format": 1}, {}):
        _check(corrected, legacy, refuses=True)
        _check(legacy, corrected, refuses=True)
    _check(corrected, {**corrected, KEY: "another-scorer"}, refuses=True)
    _check({**corrected, KEY: "another-scorer"}, corrected, refuses=True)
    _check(corrected, dict(corrected), refuses=False)


def test_malformed_panel_scorer_declarations_refuse() -> None:
    for invalid in (None, "", " ", 1, True, [], {}):
        malformed = {"audit_cache_format": 1, KEY: invalid}
        _check(malformed, dict(malformed), refuses=True)
        _check({"audit_cache_format": 1, KEY: VERSION}, malformed, refuses=True)
        _check(malformed, {"audit_cache_format": 1, KEY: VERSION}, refuses=True)


def test_legacy_and_unrelated_producers_keep_existing_behavior() -> None:
    core = {"audit_cache_format": 1, "policy_map_version": "map", "audit_ruler_version": "ruler"}
    _check(core, dict(core), refuses=False)
    assert "carries no provenance stamp" in _check({}, core, refuses=False)
    assert "declared by only one side" in _check(core, {**core, "other_extra": 1}, refuses=False)
    _check(core, {**core, "audit_ruler_version": "different"}, refuses=True)


def test_panel_writer_declares_corrected_scorer_and_legacy_anchor_label() -> None:
    tree = ast.parse(Path("scripts/blindspot_panel.py").read_text())
    writer = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "write_audit_cache")
    extra = next(k.value for k in writer.keywords if k.arg == "extra")
    assert isinstance(extra, ast.Dict)
    declared = {k.value: v.value for k, v in zip(extra.keys, extra.values, strict=True)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
    assert KEY in declared
    assert declared[KEY] == VERSION
    source = Path("scripts/blindspot_panel.py").read_text()
    assert "legacy baseline {base}; unrevalidated with corrected scorer" in source


def test_source_header_binding_and_monitor_refusal_status() -> None:
    writer = ast.parse(Path("chess_anti_engine/eval/audit_cache.py").read_text())
    stamp = next(n for n in writer.body
                 if isinstance(n, ast.FunctionDef) and n.name == "audit_cache_stamp")
    returned = next(n.value for n in stamp.body if isinstance(n, ast.Return))
    assert isinstance(returned, ast.Dict)
    assert any(k is None and isinstance(v, ast.Name) and v.id == "extra"
               for k, v in zip(returned.keys, returned.values, strict=True))
    cache = next(n for n in writer.body
                 if isinstance(n, ast.FunctionDef) and n.name == "write_audit_cache")
    cache_source = ast.unparse(cache)
    assert "stamp = audit_cache_stamp(**dict(extra or {}))" in cache_source
    assert "fh.write(json.dumps(stamp, sort_keys=True) + '\\n')" in cache_source
    reader = ast.parse(Path("scripts/paired_compare.py").read_text())
    load = next(n for n in reader.body
                if isinstance(n, ast.FunctionDef) and n.name == "load_dump")
    assert "stamp = dict(r)" in ast.unparse(load)
    returned_dump = next(n.value for n in load.body if isinstance(n, ast.Return))
    assert isinstance(returned_dump, ast.Call)
    assert isinstance(returned_dump.args[3], ast.Name)
    assert returned_dump.args[3].id == "stamp"
    monitor = Path("scripts/monitor_fen.sh").read_text()
    assert 'if [ "$2" != 0 ]; then\n            PROV="$PROV ${1}:REFUSED($2)"' in monitor
    assert '_prov "panel_$P" "$?"' in monitor


def test_no_cli_option_can_skip_stamp_gate_before_report_source_only() -> None:
    reader = ast.parse(Path("scripts/paired_compare.py").read_text())
    main = next(n for n in reader.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    # The gate is a top-level unconditional statement; no flag controls it.
    direct_calls = [(i, n.value) for i, n in enumerate(main.body)
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)]
    gate_i, gate = next((i, n) for i, n in direct_calls if ast.unparse(n.func) == "require_same_stamp")
    report_i, _ = next((i, n) for i, n in direct_calls if ast.unparse(n.func) == "report")
    assert gate_i < report_i
    assert [k.arg for k in gate.keywords] == ["label_a", "label_b"]
    flags = {n.args[0].value for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant)}
    # Negative controls: objective, metric, resampling and optional sample-size
    # choices do not reach the gate and cannot relax scorer compatibility.
    for flag in ("--metric-direction", "--field", "--mcnemar-at", "--cluster-key",
                 "--require-n", "--n-boot", "--join-key"):
        assert flag in flags
        _check({KEY: VERSION}, {}, refuses=True)
