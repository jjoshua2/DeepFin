"""Actual CPU model composition under gather overrides; no service/strength claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from .verify import oracle_check
from .verify_dispatch import PREFIX, STEP, observations
from .verify_live_model import model_identity, project, session


def run(binary: Path, package: Path, checkpoint: Path, oracle: Path) -> dict[str, Any]:
    import torch
    from chess_anti_engine.encoding import rep_fix
    from native.bend_engine.standalone.verify_neural import fixtures
    from native.bend_engine.standalone.verify_rules import position

    torch.set_num_threads(2)
    rep_fix.apply(True)
    manifest, loaded, eager = model_identity(package, checkpoint)
    if manifest["batch"] != 4:
        raise ValueError("this qualification requires a batch-four model")
    reports, controls = [], {}
    with tempfile.TemporaryDirectory(prefix="dispatch-model-") as temporary:
        base = Path(temporary)
        for name, caps in [
            ("greedy", None),
            ("one", [1] * 16),
            ("two", [1] + [2] * 15),
        ]:
            env = (
                None
                if caps is None
                else {
                    "DEEPFIN_COHORT_DISPATCH": " ".join(map(str, [4, *caps])),
                    "DEEPFIN_COHORT_PROFILE_PACKAGE_SHA256": manifest["sha256"],
                }
            )
            for case, roots, shared in [
                ("reuse", fixtures(), False),
                ("shared", fixtures()[:2], True),
            ]:
                positions = [position(r).removeprefix("position ") for r in roots]
                lines, ids, trace = session(
                    binary,
                    package,
                    positions,
                    base / f"{name}-{case}",
                    True,
                    shared=shared,
                    dispatch_environment=env,
                )
                decisions = observations(lines, caps)
                # Only the independently checked policy metadata is omitted from
                # the old identity adapter. Original neural bytes are untouched.
                projected = project(
                    [x for x in lines if not x.startswith((PREFIX, STEP))], ids, 4, True
                )
                numerical = oracle_check(
                    projected,
                    roots,
                    trace,
                    oracle,
                    manifest["channels"],
                    4,
                    4,
                    2,
                    0,
                    eager,
                    loaded.encoding.input_history_encoding,
                )
                if name == "greedy":
                    controls[case] = projected["roots"]
                elif projected["roots"] != controls[case]:
                    raise ValueError("gather override changed fixed-work model results")
                if name == "one" and max(decisions["real_rows_per_call"]) != 1:
                    raise ValueError("singleton gather override did not reach model")
                if name == "two":
                    quiet, qids, _ = session(
                        binary,
                        package,
                        positions,
                        base / f"quiet-{case}",
                        False,
                        shared=shared,
                        dispatch_environment=env,
                    )
                    observations(quiet, caps)
                    q = project(
                        [x for x in quiet if not x.startswith((PREFIX, STEP))],
                        qids,
                        4,
                        False,
                    )
                    if q["roots"] != projected["roots"]:
                        raise ValueError("diagnostics changed profiled model result")
                    for key in (
                        "forward_calls",
                        "executed_real_rows",
                        "accepted_neural_rows",
                        "padded_rows",
                    ):
                        if q["work"][key] != projected["work"][key]:
                            raise ValueError("diagnostics changed profiled work")
                reports.append(
                    {"policy": name, "case": case, "dispatch": decisions, **numerical}
                )
    return {
        "status": "passed",
        "scope": "CPU fixture with synthetic gather overrides; no measured speedup",
        "batch": 4,
        "package_sha256": manifest["sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "cases": reports,
        "model_qualified": True,
        "gpu_qualified": False,
        "speed_qualified": False,
        "inflight_cancellation_injected": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "package", "checkpoint", "oracle", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    report: dict[str, Any] = {"status": "failed"}
    try:
        report = run(a.binary, a.package, a.checkpoint, a.oracle)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
