"""Build records: what source content each compiled extension was built from.

``setup.py`` writes ``<extension file>.build-record.json`` beside every
extension it actually (re)builds: the sha256 of the built file and of every
source it was compiled from (the transitive ``.c``/``.h`` list in
``scripts/check_c_extensions_fresh.py``, plus ``setup.py`` itself, which owns
the flags and macros). The UCI engine's startup gate and ``deepfin
--selfcheck`` compare those hashes with the sources on disk, so a timestamp-only
change (a branch switch and back, ``cp``/``scp`` without ``-p``) no longer reads
as a stale build, while a real edit still does.

Stdlib only, and loaded by path from ``setup.py`` (which cannot import the
package: that would need its build dependencies). The training gate
(``scripts/check_c_extensions_fresh.py``, used by ``train.sh`` and
``validate.py``) keeps its own mtime comparison and does not read these files.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RECORD_SUFFIX = ".build-record.json"
RECORD_VERSION = 1
# Recorded for diagnosis only; the build inputs that decide staleness are the
# hashed files. setup.py reads these to choose -march/-flto/sanitizers.
_BUILD_ENV_KEYS = (
    "CC", "CFLAGS", "CAE_EXT_NATIVE", "CAE_EXT_LTO", "CAE_EXT_SANITIZE", "CAE_EXT_WERROR",
)

Verdict = Literal["ok", "stale", "artifact_changed", "no_record"]


@dataclass(frozen=True)
class RecordCheck:
    verdict: Verdict
    detail: str


def record_path(ext_path: Path) -> Path:
    return ext_path.with_name(ext_path.name + RECORD_SUFFIX)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_record(ext_path: Path, *, module: str, root: Path, inputs: Iterable[str]) -> dict:
    return {
        "version": RECORD_VERSION,
        "module": module,
        "artifact_sha256": sha256_file(ext_path),
        "inputs": {rel: sha256_file(root / rel) for rel in sorted(set(inputs))},
        "build_env": {k: os.environ[k] for k in _BUILD_ENV_KEYS if k in os.environ},
    }


def write_record(ext_path: Path, *, module: str, root: Path, inputs: Iterable[str]) -> Path:
    path = record_path(ext_path)
    record = make_record(ext_path, module=module, root=root, inputs=inputs)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def check_record(ext_path: Path, *, root: Path, inputs: Iterable[str]) -> RecordCheck:
    """Compare an extension's build record with the sources under ``root``.

    ``inputs`` is the current dependency list; a dependency the record does not
    cover (a header added to the include graph since the build) is unverified
    and counts as stale.
    """
    path = record_path(ext_path)
    if not path.is_file():
        return RecordCheck("no_record", f"no build record beside {ext_path.name}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        recorded: dict[str, str] = dict(record["inputs"])
        artifact = str(record["artifact_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return RecordCheck("no_record", f"unreadable build record {path.name} ({exc})")
    if sha256_file(ext_path) != artifact:
        return RecordCheck(
            "artifact_changed",
            f"{ext_path.name} is not the file its build record describes (replaced or "
            "copied in after the build)",
        )
    changed = []
    for rel in sorted(set(inputs) | set(recorded)):
        want = recorded.get(rel)
        source = root / rel
        if want is None:
            changed.append(f"{rel} (not in the build record)")
        elif not source.is_file():
            changed.append(f"{rel} (missing)")
        elif sha256_file(source) != want:
            changed.append(rel)
    if changed:
        return RecordCheck("stale", "source content changed since the build: " + ", ".join(changed))
    return RecordCheck("ok", f"{len(recorded)} source hashes match the build record")
