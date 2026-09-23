"""Synthetic NumPy-advice A/B contracts; no shard contents or loader runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_numpy_thp_hotload.py"
SPEC = importlib.util.spec_from_file_location("benchmark_numpy_thp_hotload", SCRIPT)
assert SPEC
assert SPEC.loader
thp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(thp)


def fake_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    kernel_thp = {
        "enabled": "madvise", "enabled_raw": "always [madvise] never",
        "defrag": "madvise", "defrag_raw": "always defer [madvise] never",
    }
    monkeypatch.setattr(thp, "read_kernel_thp_state", lambda: dict(kernel_thp))
    monkeypatch.setattr(thp, "git_head", lambda _source: "same-commit")
    monkeypatch.setattr(thp.subprocess, "check_output", lambda *args, **kwargs: "")
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "target_overlay.py").write_text("fixed candidate")
    candidate = {
        "path": str(source), "commit": "same-commit",
        "files_sha256": {
            "target_overlay.py": thp.sha256(source / "target_overlay.py"),
        },
    }
    shards = [
        {"path": str(tmp_path / f"shard_{index}.zarr"), "rows": 8192,
         "content_sha256": f"{index:064x}"}
        for index in range(8)
    ]
    receipt = tmp_path / "B.json"
    receipt.write_text(json.dumps({"shards": shards}))
    fixture = {
        "receipt": str(receipt), "receipt_sha256": thp.sha256(receipt),
        "shards": shards, "rows": 65536,
    }
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        **{name: thp.importlib.metadata.version(name) for name in thp.PACKAGE_NAMES},
    }
    original = {
        "fixture": fixture,
        "environment": environment,
        "sources": {"control": {"commit": "older"}, "candidate": candidate},
    }
    reference = tmp_path / "prepared-hotload-plan.json"
    reference.write_text(json.dumps(original))
    return {
        "status": "PREPARED_HELD_FOR_UNCONTENDED_CPU_SLOT",
        "runner_sha256": thp.sha256(SCRIPT),
        "fixture_reference": {"path": str(reference), "sha256": thp.sha256(reference)},
        "numpy_madvise_hugepage": dict(thp.NUMPY_ADVICE),
        "kernel_thp": kernel_thp,
        "order": list(thp.ORDER),
        "sources": {arm: dict(candidate) for arm in thp.ORDER},
        "environment": environment,
        "fixture": fixture,
        "bounds": {
            "seconds_per_arm": 180, "cpu_seconds_per_arm": 120,
            "max_arms": 4, "affinity": [16, 17], "nice": 19,
            "threads": 2, "minimum_memory_gib": 40,
            "minimum_disk_gib": 150, "new_output_limit_mib": 4,
            "gpu": False,
        },
    }


def receipt(plan: dict, arm: str, index: int, load: float, wall: float) -> dict:
    return {
        "status": "PASS", "arm": arm, "index": index,
        "source": plan["sources"][arm],
        "numpy_madvise_hugepage_env": thp.NUMPY_ADVICE[arm],
        "numpy_madvise_hugepage_realized": arm == "control",
        "kernel_thp_at_start": plan["kernel_thp"],
        "kernel_thp_at_end": plan["kernel_thp"],
        "plan": {"batch_array": [128]},
        "record_roster": [{"rows": 8192}],
        "decoded_array_hashes": {"x": {"shape": [1], "dtype": "|u1", "sha256": "a"}},
        "ordered_target_hashes": {"policy_target": {"shape": [1], "sha256": "b"}},
        "hot_load_seconds_per_shard": [load / 8] * 8,
        "wall_seconds": wall,
    }


def test_same_candidate_and_reference_fixture_are_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    thp.check_pins(plan, runner=SCRIPT)
    plan["sources"]["control"]["commit"] = "other"
    with pytest.raises(thp.Refused, match="original candidate"):
        thp.check_pins(plan, runner=SCRIPT)
    plan["sources"]["control"]["commit"] = "same-commit"
    plan["fixture"]["shards"][0]["rows"] = 8191
    with pytest.raises(thp.Refused, match="original candidate and fixture"):
        thp.check_pins(plan, runner=SCRIPT)
    plan["fixture"]["shards"][0]["rows"] = 8192
    Path(plan["fixture_reference"]["path"]).write_text("changed")
    with pytest.raises(thp.Refused, match="original eight-shard plan pin"):
        thp.check_pins(plan, runner=SCRIPT)


def test_child_environments_differ_only_in_advice_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "untrusted-parent-value")
    a = thp.child_environment(plan, "control")
    b = thp.child_environment(plan, "candidate")
    assert a["NUMPY_MADVISE_HUGEPAGE"] == "1"
    assert b["NUMPY_MADVISE_HUGEPAGE"] == "0"
    assert {key: value for key, value in a.items() if key != "NUMPY_MADVISE_HUGEPAGE"} == {
        key: value for key, value in b.items() if key != "NUMPY_MADVISE_HUGEPAGE"
    }
    monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
    with pytest.raises(thp.Refused, match="MALLOC_ARENA_MAX"):
        thp.child_environment(plan, "control")


def test_child_refuses_requested_or_realized_mode_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    for arm, requested, realized in (
        ("control", "1", True), ("candidate", "0", False),
    ):
        monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", requested)
        fake_numpy = SimpleNamespace(core=SimpleNamespace(
            multiarray=SimpleNamespace(_get_madvise_hugepage=lambda realized=realized: realized),
        ))
        assert thp.checked_numpy_advice(plan, arm, fake_numpy) is realized
        monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "wrong")
        with pytest.raises(thp.Refused, match="environment mismatch"):
            thp.checked_numpy_advice(plan, arm, fake_numpy)
        monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", requested)
        wrong_numpy = SimpleNamespace(core=SimpleNamespace(
            multiarray=SimpleNamespace(_get_madvise_hugepage=lambda realized=realized: not realized),
        ))
        with pytest.raises(thp.Refused, match="realization mismatch"):
            thp.checked_numpy_advice(plan, arm, wrong_numpy)


def test_parity_and_all_performance_terms_decide_local_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    items = [receipt(plan, arm, index, load, wall) for index, (arm, load, wall) in enumerate(
        zip(thp.ORDER, (10.0, 8.0, 8.2, 10.2), (20.0, 19.0, 19.2, 20.2), strict=True)
    )]
    passing = thp.compare(items, plan)
    assert passing["status"] == "PASS_EXACT_PARITY"
    assert passing["decision"] == "WARRANTS_FUTURE_TRAINER_TEST"
    assert passing["adjacent_pair_improvements"] == [True, True]

    items[1]["numpy_madvise_hugepage_realized"] = True
    with pytest.raises(thp.Refused, match="NumPy advice"):
        thp.compare(items, plan)
    items[1]["numpy_madvise_hugepage_realized"] = False
    items[1]["decoded_array_hashes"]["x"]["sha256"] = "changed"
    with pytest.raises(thp.Refused, match="decoded_array_hashes"):
        thp.compare(items, plan)
    items[1]["decoded_array_hashes"]["x"]["sha256"] = "a"

    items[1]["hot_load_seconds_per_shard"] = [11.0 / 8] * 8
    items[2]["hot_load_seconds_per_shard"] = [5.0 / 8] * 8
    assert thp.compare(items, plan)["decision"] == "DOES_NOT_MEET_LOCAL_BENEFIT_GATE"
    items[1]["hot_load_seconds_per_shard"] = [8.0 / 8] * 8
    items[2]["hot_load_seconds_per_shard"] = [8.2 / 8] * 8
    items[1]["wall_seconds"] = 22.0
    items[2]["wall_seconds"] = 22.2
    assert thp.compare(items, plan)["decision"] == "DOES_NOT_MEET_LOCAL_BENEFIT_GATE"


def test_parent_passes_arm_only_env_through_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "status": "ADMITTED", "plan_sha256": thp.sha256(plan_path),
        "admitted_by": "synthetic test", "quiet_workload_evidence": "fake",
        "kernel_thp": plan["kernel_thp"],
        "admitted_unix_seconds": time.time(),
    }))
    monkeypatch.setattr(thp, "check_resources", lambda *_args: None)
    monkeypatch.setattr(thp, "_proc_starttime", lambda _pid: 1)
    monkeypatch.setattr(thp, "supervise_child", lambda *_args: None)
    launches: list[dict[str, str]] = []

    class FinishedChild:
        pid = 123
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(*_args: object, **kwargs: object) -> FinishedChild:
        index = len(launches)
        launches.append(dict(kwargs["env"]))  # type: ignore[arg-type]
        output_dir = Path(kwargs["cwd"])  # type: ignore[arg-type]
        arm = thp.ORDER[index]
        thp.atomic_json(
            output_dir / f"arm_{index}.json",
            receipt(plan, arm, index, (10.0, 8.0, 8.2, 10.2)[index],
                    (20.0, 19.0, 19.2, 20.2)[index]),
            thp.ARM_LIMIT,
        )
        return FinishedChild()

    monkeypatch.setattr(thp.subprocess, "Popen", fake_popen)
    output_dir = tmp_path / "output"
    thp.run(plan, plan_path, admission, output_dir)
    assert [env["NUMPY_MADVISE_HUGEPAGE"] for env in launches] == ["1", "0", "0", "1"]
    assert len({
        tuple(sorted((key, value) for key, value in env.items()
                     if key != "NUMPY_MADVISE_HUGEPAGE"))
        for env in launches
    }) == 1
    assert thp.load_json(output_dir / "summary.json")["decision"] == "WARRANTS_FUTURE_TRAINER_TEST"


def test_fresh_admission_still_precedes_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "HELD"}))
    monkeypatch.setattr(thp, "check_resources", lambda *_args: None)
    monkeypatch.setattr(
        thp.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("child launched without admission"),
    )
    output_dir = tmp_path / "output"
    with pytest.raises(thp.Refused, match="ADMITTED"):
        thp.run(plan, plan_path, admission, output_dir)
    assert thp.load_json(output_dir / "summary.json")["status"] == "FAILED"


def test_kernel_thp_parser_and_drift_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sysfs = tmp_path / "transparent_hugepage"
    sysfs.mkdir()
    (sysfs / "enabled").write_text("always [madvise] never\n")
    (sysfs / "defrag").write_text("always defer [madvise] never\n")
    monkeypatch.setattr(thp, "THP_SYSFS", sysfs)
    expected = thp.read_kernel_thp_state()
    assert expected["enabled"] == expected["defrag"] == "madvise"
    assert thp.checked_kernel_thp_state({"kernel_thp": expected}) == expected
    (sysfs / "enabled").write_text("[always] madvise never\n")
    with pytest.raises(thp.Refused, match="kernel THP state drift"):
        thp.checked_kernel_thp_state({"kernel_thp": expected})


def test_admission_and_receipt_require_kernel_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    admission = tmp_path / "admission.json"
    admitted = {
        "status": "ADMITTED", "plan_sha256": thp.sha256(plan_path),
        "admitted_by": "synthetic test", "quiet_workload_evidence": "fake",
        "admitted_unix_seconds": time.time(),
    }
    admission.write_text(json.dumps(admitted))
    with pytest.raises(thp.Refused, match="admission kernel THP pin mismatch"):
        thp.check_admission(plan, plan_path, admission)
    admitted["kernel_thp"] = plan["kernel_thp"]
    admission.write_text(json.dumps(admitted))
    thp.check_admission(plan, plan_path, admission)
    arm = receipt(plan, "control", 0, 10.0, 20.0)
    thp.check_mode_receipt(arm, plan, 0)
    arm["kernel_thp_at_end"] = {**plan["kernel_thp"], "enabled": "always"}
    with pytest.raises(thp.Refused, match="kernel THP receipt mismatch"):
        thp.check_mode_receipt(arm, plan, 0)
