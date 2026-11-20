"""sha256 provenance of the nets and openings an arena plays.

The arena's game-log fingerprint and result rows name checkpoints by PATH. A
path says nothing about the bytes behind it: a checkpoint rewritten between two
runs (or between a run and its pooling) keeps its path, and every downstream
reader would silently attribute the games to the new file. These helpers hash
what a seat actually loads, once, so a result row and a game-log header can say
which bytes were played and a later reader can check them.

A seat is one of:

* a checkpoint file;
* a checkpoint directory, which the loader resolves to ``<dir>/trainer.pt``;
* a JSON spec (for example a research ``*.hybrid.json`` wrapper) whose string
  values that resolve to a checkpoint (a file, or a directory's ``trainer.pt``)
  are the nets it is built from. A spec that carries ``<key>_sha256`` beside
  ``<key>`` is VERIFIED: a mismatch raises, and so does a member that does not
  resolve, since the spec would otherwise claim nets it does not load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """A seat's files cannot be hashed, or a spec's recorded sha256 is wrong."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(path: str | Path) -> Path:
    """The file the model loader reads for ``path`` (``model_loader._resolve_trainer_pt``)."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir() and (p / "trainer.pt").is_file():
        return p / "trainer.pt"
    raise ProvenanceError(f"no checkpoint file at {path}")


def seat_files(path: str | Path) -> list[dict[str, str]]:
    """Every file a seat is made of, primary first: ``[{"role", "path", "sha256"}, ...]``."""
    primary = resolve_checkpoint(path)
    files = [{"role": "spec" if primary.suffix == ".json" else "checkpoint",
              "path": str(primary), "sha256": sha256_file(primary)}]
    if primary.suffix != ".json":
        return files
    try:
        spec = json.loads(primary.read_text())
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"{primary}: a .json seat that is not a readable JSON spec ({exc})") from exc
    if not isinstance(spec, dict):
        raise ProvenanceError(f"{primary}: a .json seat must be a JSON object")
    seen = {str(primary)}
    for key, value in spec.items():
        if not isinstance(value, str) or not value or key.endswith("_sha256"):
            continue
        recorded = spec.get(f"{key}_sha256")
        try:
            member = resolve_checkpoint(value)
        except ProvenanceError as exc:
            if recorded is not None:
                # A recorded sha256 says this key IS a net; skipping it would skip its verification.
                raise ProvenanceError(f"{primary}: {key} has {key}_sha256 recorded but {exc}") from exc
            continue            # a string that is not a checkpoint (a mode, a label)
        digest = sha256_file(member)
        if recorded is not None and recorded != digest:
            raise ProvenanceError(f"{primary}: {key}_sha256 {recorded} != sha256 of {member} ({digest})")
        if str(member) not in seen:
            files.append({"role": key, "path": str(member), "sha256": digest})
            seen.add(str(member))
    return files


def seat_provenance(path: str | Path) -> dict[str, Any]:
    """``{"path", "sha256", "files"}``; ``sha256`` is the primary file's (the spec's, for a spec)."""
    files = seat_files(path)
    return {"path": str(path), "sha256": files[0]["sha256"], "files": files}


def openings_sha256(path: str | Path | None) -> str | None:
    """sha256 of an openings source file (book or FEN list); None when there is no file."""
    if path is None:
        return None
    p = Path(path)
    return sha256_file(p) if p.is_file() else None
