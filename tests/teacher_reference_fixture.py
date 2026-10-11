"""Materialize pinned CPU references with declared neutral path derivatives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def reference_sources(tmp_path: Path) -> Path:
    fixtures = Path(__file__).with_name("fixtures") / "teacher_reference"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if manifest["schema"] != 2:
        raise ValueError("explicit original and CPU fixture identities required")
    root = tmp_path / "retained_reference"
    root.mkdir(exist_ok=True)
    for record in manifest["files"]:
        raw = (fixtures / record["fixture"]).read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("retained reference fixture bytes changed")
        changed = any(record["path_relocations"].values())
        if changed != (record["sha256"] != record["original_sha256"]):
            raise ValueError("declared reference path transformation differs")
        target = root / record["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("materialized retained reference changed")
        else:
            target.write_bytes(raw)
    return root


def cpu_config(reference: Path, runtime: Path) -> dict[str, Any]:
    """Explicit unqualified CPU relocation; never a production qualified plan."""
    config = json.loads((reference / "scripts/dual_companion_config.json").read_text())
    def ref(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config.update(python=sys.executable, v9_source=ref(reference / "reference/compare.py"),
                  cleanup_source=ref(reference / "cleanup/finite_natural_gpu_gap_bridge_v2.py"))
    plan = config["qualified_plan"]
    plan["status"] = "CPU_REFERENCE_FIXTURE_NOT_QUALIFICATION"
    plan["teacher_env"]["ceres"] = {"LD_LIBRARY_PATH": ""}
    paths = {"runtime": str(runtime), "tpg": str(reference / "encoding/ceres_tpg.py"),
             "adapter": str(reference / "ad/c3_backend.py")}
    plan["cpu_fixture_paths"] = paths
    plan["source_pins"] = {str(path): ref(path)["sha256"] for path in (
        reference / "reference/compare.py", Path(paths["tpg"]), Path(paths["adapter"]))}
    return config
