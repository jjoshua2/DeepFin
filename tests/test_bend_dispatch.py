"""No compiler/model required: reject stale or fabricated service plans."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from native.bend_engine.service_profile.dispatch import (
    PACKAGE_ENV,
    PLAN_ENV,
    gather_caps,
    policy,
)

REPORT = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/evidence/native-service-profile/report.json"
)


def report() -> dict:
    return json.loads(REPORT.read_text())


def compile_policy(r: dict, batch: int = 4, **overrides: Any) -> dict:
    t = next(t for t in r["targets"] if t["batch"] == batch)
    args: dict[str, Any] = {
        "package_sha": t["identity"]["package_sha256"],
        "batch": batch,
        "channels": 175,
        "checkpoint_sha": r["checkpoint_sha256"],
        "encoding": r["encoding"],
        "host": r["host"],
    }
    args.update(overrides)
    return policy(r, **args)


def test_recorded_profile_recomputed_and_package_bound() -> None:
    r = report()
    original = copy.deepcopy(r)
    for batch in (1, 4):
        p = compile_policy(r, batch)
        assert p["environment"][PACKAGE_ENV] == p["package_sha256"]
        assert list(map(int, p["environment"][PLAN_ENV].split())) == [
            batch,
            *p["caps_by_active_roots"],
        ]
        assert p["caps_by_active_roots"] == (
            [1] * 16 if batch == 1 else [1, 2, 3, 4, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]
        )
        assert p["deadline_guarantee"] is False
    assert r == original


def test_nonmonotonic_costs_can_change_the_real_row_cap() -> None:
    target = {
        "batch": 4,
        "cells": [
            {"physical_batch": 4, "real_rows": i, "p95_ns": cost}
            for i, cost in enumerate((20, 10, 100, 90), 1)
        ],
    }
    caps = gather_caps(target)
    assert caps[:6] == [1, 2, 2, 2, 2, 2]


def test_receding_plan_does_not_repeatedly_take_small_remainder() -> None:
    target = {
        "batch": 4,
        "cells": [
            {"physical_batch": 4, "real_rows": i, "p95_ns": 100} for i in range(1, 5)
        ],
    }
    # The equal-work optimizer's lexicographic plan is [1,4], not a gather cap of 1.
    assert gather_caps(target)[4] == 4


@pytest.mark.parametrize(
    "key", ["runtime_dispatch_changed", "gpu_qualified", "strength_qualified"]
)
@pytest.mark.parametrize("bad", [True, None, 0])
def test_wrong_source_scope(key: str, bad: object) -> None:
    r = report()
    r[key] = bad
    with pytest.raises(ValueError, match=r"scope"):
        compile_policy(r)


@pytest.mark.parametrize(
    "key",
    [
        "machine",
        "cpu_model",
        "logical_cpus",
        "torch_version",
        "torch_threads",
        "interop_threads",
    ],
)
def test_host_mismatch_rejected(key: str) -> None:
    r = report()
    host = dict(r["host"])
    host[key] = "different"
    with pytest.raises(ValueError, match=r"host/runtime"):
        compile_policy(r, host=host)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("status", "failed"),
        ("schema", "x"),
        ("accepted_neural_rows", 1),
        ("useful_eps", 2),
    ],
)
def test_failed_or_wrong_units(key: str, bad: object) -> None:
    r = report()
    r[key] = bad
    with pytest.raises(ValueError, match=r"expected|source profile|accepted"):
        compile_policy(r)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("package_sha", "f" * 64),
        ("checkpoint_sha", "e" * 64),
        ("channels", 146),
        ("encoding", {}),
        ("batch", True),
    ],
)
def test_identity_mismatch(key: str, bad: Any) -> None:
    r = report()
    kwargs: dict[str, Any] = {key: bad}
    if key == "batch":
        with pytest.raises(
            ValueError, match=r"package|checkpoint|encoding|shape|identity"
        ):
            policy(
                r,
                r["targets"][1]["identity"]["package_sha256"],
                bad,
                175,
                r["checkpoint_sha256"],
                r["encoding"],
                r["host"],
            )
    else:
        with pytest.raises(
            ValueError, match=r"package|checkpoint|encoding|shape|identity"
        ):
            compile_policy(r, **kwargs)


@pytest.mark.parametrize(
    "mutation", ["p95", "sample", "warmup", "count", "identity", "repeat", "occupancy"]
)
def test_raw_sample_and_summary_consistency(mutation: str) -> None:
    r = report()
    t = r["targets"][1]
    if mutation == "p95":
        t["cells"][0]["p95_ns"] += 1
    elif mutation == "sample":
        t["runs"][0]["native"]["samples"][0]["service_ns"] = 0
    elif mutation == "warmup":
        t["runs"][0]["native"]["samples"][0]["warmup"] = False
    elif mutation == "count":
        t["runs"][0]["native"]["samples"].pop()
    elif mutation == "identity":
        t["runs"][0]["identity"]["checkpoint_sha256"] = "f" * 64
    elif mutation == "repeat":
        t["runs"] = t["runs"][:1]
    elif mutation == "occupancy":
        t["runs"][0]["native"]["samples"][0]["real_rows"] = 5
    with pytest.raises(
        ValueError, match=r"sample|integer|occupancy|process|identity|cells|matrix"
    ):
        compile_policy(r)


def test_duplicate_package_target_is_not_silently_selected() -> None:
    r = report()
    r["targets"].append(copy.deepcopy(r["targets"][1]))
    with pytest.raises(ValueError, match=r"one measured target"):
        compile_policy(r)


def test_native_decision_audit_checks_the_configured_limit() -> None:
    from native.bend_engine.multi_root.verify_dispatch import (
        BATCH,
        PREFIX,
        STEP,
        observations,
    )

    caps = [1] + [2] * 15
    lines = [PREFIX + " ".join(map(str, [4, *caps])), STEP + "6 2 4", BATCH + "1 2 4"]
    assert observations(lines, caps)["real_rows_per_call"] == [2]
    with pytest.raises(ValueError, match="ignored gather"):
        observations([*lines[:-1], BATCH + "1 3 4"], caps)
    with pytest.raises(ValueError, match="configured table"):
        observations([lines[0], STEP + "6 4 4", lines[-1]], caps)
    with pytest.raises(ValueError, match="acknowledgment"):
        observations(lines[1:], caps)
    with pytest.raises(ValueError, match="default path"):
        observations(lines, None)
