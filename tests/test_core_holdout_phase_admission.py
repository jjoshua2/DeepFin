"""Fictional metadata: seal first, then bind actual SCORE/READ owner argv.

No process, numerical model, HOLD data or admission authority is created.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_holdout_read as reader_entry
from scripts import core_generation_holdout_ruler as ruler


def phase_graph(tmp_path: Path, *, reader: bool = False, fault: str = "",
                protocol_body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = {"schema": ruler.READ_SCHEMA if reader else ruler.SCORE_SCHEMA} | (protocol_body or {})
    protocol = ruler.write(tmp_path / "protocol.json", body)
    admission_path = tmp_path / "admission.json"
    module = "scripts.core_generation_holdout_read" if reader else "scripts.core_generation_holdout_score"
    normalized = ["/fiction/python", "-I", "-B", "-m", module,
                  "--protocol", protocol["path"], "--protocol-sha256", protocol["sha256"],
                  "--admission", str(admission_path), "--admission-sha256", "SELF_ADMISSION_SHA256"]
    admission: dict[str, Any] = {"status": ruler.ADMISSION, "protocol": protocol,
        "normalized_command": copy.deepcopy(normalized), "resource": "cpu" if reader else "gpu",
        "seconds": 900 if reader else 1200, "owner": {"sha256": "e" * 64}}
    admission.update({key: body[key] for key in ("source", "stage", "owner") if key in body})
    if fault == "legacy":
        admission["status"] = "ROOT_ADMITS_CORE_HOLDOUT_RULER_PHASE_V1"
    elif fault == "legacy_field":
        admission["command"] = list(normalized)
    elif fault == "missing_normalized":
        del admission["normalized_command"]
    elif fault == "wrong_normalized_type":
        admission["normalized_command"] = "not argv"
    elif fault == "normalized_prefix":
        admission["normalized_command"][0] = "/fiction/other-python"
    elif fault == "normalized_second_token":
        admission["normalized_command"][1] = "SELF_ADMISSION_SHA256"
    elif fault == "normalized_actual_hash":
        admission["normalized_command"][-1] = "0" * 64
    elif fault == "foreign_protocol":
        admission["protocol"] = {"path": protocol["path"], "sha256": "0" * 64}
    aref = ruler.write(admission_path, admission)
    command = list(normalized)
    command[-1] = aref["sha256"]
    return {"protocol": protocol, "admission": aref, "body": admission, "command": command}


@pytest.mark.parametrize("reader", [False, True])
def test_phase_admission_seals_before_actual_command_without_fixed_point(tmp_path: Path, reader: bool):
    graph = phase_graph(tmp_path, reader=reader)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    admission = metadata.read(graph["admission"])
    assert graph["command"][-1] == graph["admission"]["sha256"]
    assert graph["admission"]["sha256"] not in ruler.canonical(admission).decode()
    original = copy.deepcopy(graph["command"])
    ruler.phase_command(admission, graph["protocol"], graph["admission"], graph["command"], reader=reader)
    assert graph["command"] == original
    metadata.finish()


@pytest.mark.parametrize("reader", [False, True])
@pytest.mark.parametrize("fault", ["legacy", "legacy_field", "missing_normalized", "wrong_normalized_type", "foreign_protocol",
                                   "normalized_prefix", "normalized_second_token", "normalized_actual_hash"])
def test_rehashed_phase_admission_refusals(tmp_path: Path, reader: bool, fault: str):
    graph = phase_graph(tmp_path, reader=reader, fault=fault)
    expected = ("phase requires normalized V2 admission" if fault in {
        "legacy", "legacy_field", "missing_normalized", "wrong_normalized_type", "foreign_protocol"
    } else "normalized ruler phase command differs")
    with pytest.raises(ValueError, match=expected):
        ruler.phase_command(graph["body"], graph["protocol"], graph["admission"], graph["command"], reader=reader)


@pytest.mark.parametrize("reader", [False, True])
@pytest.mark.parametrize("fault", ["sha", "admission_path", "protocol_sha", "module", "trailing", "prefix", "empty", "duplicate"])
def test_actual_phase_argv_substitution_refusals(tmp_path: Path, reader: bool, fault: str):
    graph = phase_graph(tmp_path, reader=reader)
    command = list(graph["command"])
    expected = "ruler phase command suffix differs"
    if fault == "sha":
        command[-1] = "0" * 64
    elif fault == "admission_path":
        command[-3] = "/fiction/other-admission.json"
    elif fault == "protocol_sha":
        command[-5] = "0" * 64
    elif fault == "module":
        command[-9] = "scripts.core_generation_holdout_score" if reader else "scripts.core_generation_holdout_read"
    elif fault == "trailing":
        command.append("unexpected")
    elif fault == "prefix":
        command[0] = "/fiction/other-python"
        expected = "normalized ruler phase command differs"
    elif fault == "empty":
        command[0] = ""
    else:
        command[1:1] = ["--admission-sha256", graph["admission"]["sha256"]]
        expected = "ruler phase command flags ambiguous"
    with pytest.raises(ValueError, match=expected):
        ruler.phase_command(graph["body"], graph["protocol"], graph["admission"], command, reader=reader)


def closed_score_graph(tmp_path: Path, fault: str = "", *, protocol_body: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    graph = phase_graph(tmp_path, fault="normalized_prefix" if fault == "normalized" else "", protocol_body=protocol_body)
    command = graph["command"]
    terminal: dict[str, Any] = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": "gpu", "seconds": 1200, "wall_seconds": 1.0, "command": command,
        "source_revision": "f" * 40, "cwd": "/fiction/owner", "supervisor_sha256": "e" * 64,
        "supervisor_pid": 2147483645, "known_processes": {"2147483646": 1}}
    closure: dict[str, Any] = {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645,
        "supervisor_closed": True, "owned_births_closed": True, "known_processes": terminal["known_processes"]}
    if fault == "terminal_command":
        terminal["command"] = ["/fiction/other", *command[1:]]
    elif fault == "returncode":
        terminal["returncode"] = 1
    elif fault == "births":
        closure["known_processes"] = {"2147483646": 2}
    elif fault == "supervisor":
        closure["supervisor_closed"] = False
    elif fault == "cap":
        terminal["seconds"] = 1199
    owner = {"terminal": ruler.write(tmp_path / "terminal.json", terminal), "admission": graph["admission"],
        "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner", "closure": closure}
    return owner, command[-10:]


def test_closed_score_owner_uses_normalized_phase_without_legacy_fallback(tmp_path: Path):
    owner, suffix = closed_score_graph(tmp_path)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    terminal = ruler.closed_owner(metadata, owner, resource="gpu", suffix=suffix, seconds=1200, _ruler_phase=True)
    assert terminal["command"] == owner["command"]
    metadata.finish()
    with pytest.raises(KeyError, match="command"):
        ruler.closed_owner(ruler.Metadata(1 << 20, 8 << 20), owner, resource="gpu", suffix=suffix, seconds=1200)


@pytest.mark.parametrize("fault", ["normalized", "terminal_command", "returncode", "births", "supervisor", "cap"])
def test_closed_score_retains_actual_terminal_and_closure_refusals(tmp_path: Path, fault: str):
    owner, suffix = closed_score_graph(tmp_path, fault)
    expected = {"normalized": "normalized ruler phase command differs", "terminal_command": "owner not exact closed PASS",
                "returncode": "owner not exact closed PASS", "births": "closure evidence differs",
                "supervisor": "closure evidence differs", "cap": "owner cap/wall differs"}
    with pytest.raises(ValueError, match=expected[fault]):
        ruler.closed_owner(ruler.Metadata(1 << 20, 8 << 20), owner, resource="gpu", suffix=suffix, seconds=1200, _ruler_phase=True)


@pytest.mark.parametrize("fault", ["", "foreign_protocol"])
def test_reader_closed_observations_reaches_normalized_score_route(tmp_path: Path, fault: str):
    output = tmp_path / "scoring-output"
    output.mkdir()
    identities: dict[str, Any] = {name: {} for name in (
        "study_uuid", "design", "holdout", "parents", "models", "source", "runtime", "limits", "native_basis")}
    protocol_body = identities | {"outer_seconds": 1200, "stage": "fictional-score",
        "owner": {"sha256": "e" * 64}, "output": str(output)}
    owner, _ = closed_score_graph(tmp_path, protocol_body=protocol_body)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    admission = metadata.read(owner["admission"])
    pref = admission["protocol"]
    if fault:
        pref = ruler.write(tmp_path / "foreign-protocol.json", {"schema": ruler.SCORE_SCHEMA} | protocol_body)
    report = {"status": ruler.SCORE_PENDING, "protocol": pref, "source": identities["source"],
        "study_uuid": identities["study_uuid"], "holdout": identities["holdout"], "parents": identities["parents"],
        "model_deliveries": identities["models"], "optimizer_constructed": False, "optimizer_restored": False,
        "optimizer_updates": 0, "sampler_draws": 0, "closed_owner_attested_by_child": False,
        "row_logs": {label: {"path": str(output / (label + ".rows.jsonl"))} for label in ruler.metrics.LABELS}}
    report_ref = ruler.write(output / "readout.json", report)
    root = {"status": ruler.SCORE_ROOT, "protocol": pref, "readout": report_ref, "owner": owner}
    rref = ruler.write(tmp_path / "root.json", root)
    iref = ruler.write(tmp_path / "review.json", {"status": ruler.SCORE_REVIEW, "root_acceptance": rref,
        "protocol": pref, "readout": report_ref, "findings": []})
    p = identities | {"scoring": {"root": rref, "independent": iref}}
    if fault:
        with pytest.raises(ValueError, match="closed child command differs"):
            reader_entry.closed_observations(metadata, p)
    else:
        observed, terminal = reader_entry.closed_observations(metadata, p)
        assert observed == report
        assert terminal["command"] == owner["command"]
        metadata.finish()
