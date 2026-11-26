#!/usr/bin/env python3
"""Regenerate tests/ci_file_durations.json from CI junit reports.

    python scripts/ci_file_durations.py <dir-or-xml>...

Pass the downloaded ``cpu-test-results*`` artifacts (directories are searched
for ``cpu.xml`` and ``cpu-serial.xml``). The file only balances the shards of
``scripts/validate.py cpu --shard K/N``; it never decides which tests run.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPORT_PHASE = {"cpu.xml": "xdist", "cpu-serial.xml": "serial"}


def durations_from_junit(reports: list[Path]) -> dict[str, dict[str, float]]:
    """Seconds per test file, summed over its test cases, keyed by phase."""
    totals: dict[str, dict[str, float]] = {"serial": {}, "xdist": {}}
    for report in reports:
        phase = totals[REPORT_PHASE[report.name]]
        for case in ET.parse(report).getroot().iter("testcase"):
            # classname is the dotted module, plus the class for a method.
            parts = (case.get("classname") or "").split(".")
            module = next((index for index, part in enumerate(parts) if part.startswith("test_")), None)
            if module is None:
                continue
            name = "/".join(parts[:module + 1]) + ".py"
            phase[name] = phase.get(name, 0.0) + float(case.get("time") or 0.0)
    return {phase: {name: round(seconds, 1) for name, seconds in sorted(files.items())}
            for phase, files in totals.items()}


def main(argv: list[str]) -> int:
    reports: list[Path] = []
    for arg in argv:
        path = Path(arg)
        reports += sorted(found for found in path.rglob("*.xml") if found.name in REPORT_PHASE) if path.is_dir() else [path]
    if not reports:
        print(__doc__, file=sys.stderr)
        return 2
    out = ROOT / "tests/ci_file_durations.json"
    out.write_text(json.dumps(durations_from_junit(reports), indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} from {len(reports)} reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
