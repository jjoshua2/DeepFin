"""Stage lineage: refuse a broken chain of sealed stages, and unseal downstream.

Stages are ordered ``(generation, collect < train < eval)``; generation 0 is
the init stage. A sealed stage's manifest records the sha256 of every input it
consumed. ``verify_stage`` re-hashes the inputs that belong to the run's own
lineage (files inside the run dir, plus the TRAIN/EVAL openings) and refuses a
mismatch, so a redo of an earlier stage can never be silently built upon.

``unseal_after`` implements ``--redo-downstream``: every sealed stage after the
redone one is MOVED (rows and models kept) to
``<run>/superseded/<UTC stamp>[-N]/gen_NNNN/<stage>`` (``-N`` keeps two redos in the
same second apart), and the run-level files
derived from them are rewound: arena ledger rows of the unsealed evaluations,
the derived presentation ledger, the probe set (when generation 1's train or
collect is unsealed), and the duplicate-sketch / openings-seen files from the
first unsealed collect on. The opening cursor lives in the collect manifests,
so it rewinds with them.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

from chess_anti_engine.simple_rl.common import (
    MANIFEST, SHA_CACHE, read_manifest, sha256_tree, stat_stamp, write_json_atomic,
)

STAGE_ORDER = ("collect", "train", "eval")
# Inputs outside the run dir that are part of the lineage (the split's files).
LINEAGE_EXTERNAL_INPUTS = frozenset({"train_openings", "openings"})


def stage_key(generation: int, stage: str) -> tuple[int, int]:
    return (int(generation), -1 if stage == "init" else STAGE_ORDER.index(stage))


def sealed_stages(run_dir: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for gdir in sorted(Path(run_dir).glob("gen_[0-9][0-9][0-9][0-9]")):
        g = int(gdir.name.split("_")[1])
        if g == 0:
            if (gdir / MANIFEST).is_file():
                out.append((0, "init"))
            continue
        out += [(g, s) for s in STAGE_ORDER if (gdir / s / MANIFEST).is_file()]
    return sorted(out, key=lambda gs: stage_key(*gs))


def stage_dir(run_dir: Path, generation: int, stage: str) -> Path:
    gdir = Path(run_dir) / f"gen_{int(generation):04d}"
    return gdir if stage == "init" else gdir / stage


def later_sealed(run_dir: Path, generation: int, stage: str) -> list[tuple[int, str]]:
    k = stage_key(generation, stage)
    return [gs for gs in sealed_stages(run_dir) if stage_key(*gs) > k]


class _ShaCache:
    """sha256 of a file keyed by (size, mtime_ns, inode), persisted in the run
    dir so a resume does not re-read every model of every generation. A
    directory is always hashed in full."""

    stamp = staticmethod(stat_stamp)

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / SHA_CACHE
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        except (OSError, ValueError):
            self.data = {}
        self.dirty = False

    def sha(self, p: Path) -> str:
        if p.is_dir():
            return sha256_tree(p)
        key = str(p.resolve())
        stamp = self.stamp(p.stat())
        hit = self.data.get(key)
        if hit and hit["stamp"] == stamp:
            return str(hit["sha256"])
        digest = sha256_tree(p)
        self.remember(p, stamp, digest)
        return digest

    def remember(self, p: Path, stamp_before: list[int], digest: str) -> None:
        """Cache ``digest`` only if the file's stamp did not move while it was hashed."""
        if self.stamp(p.stat()) == stamp_before:
            self.data[str(p.resolve())] = {"stamp": stamp_before, "sha256": digest}
            self.dirty = True

    def save(self) -> None:
        if self.dirty:
            write_json_atomic(self.path, self.data)


def _fresh_superseded_dir(run_dir: Path) -> Path:
    """A new ``superseded/<UTC stamp>[-N]`` dir; never one an earlier redo used."""
    base = Path(run_dir) / "superseded"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for n in itertools.count():
        d = base / (stamp if n == 0 else f"{stamp}-{n}")
        try:
            d.mkdir()
        except FileExistsError:
            continue
        return d
    raise AssertionError("unreachable")


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_stage(run_dir: Path, generation: int, stage: str, *, cache: _ShaCache | None = None) -> None:
    """Refuse (SystemExit) when a sealed stage's lineage inputs changed."""
    run_dir = Path(run_dir)
    manifest = read_manifest(stage_dir(run_dir, generation, stage))
    if manifest is None:
        return
    own = cache or _ShaCache(run_dir)
    bad = []
    for name, ref in sorted(manifest.get("inputs", {}).items()):
        if not isinstance(ref, dict) or "sha256" not in ref:
            continue
        p = Path(ref["path"])
        if not (_inside(p, run_dir) or name in LINEAGE_EXTERNAL_INPUTS):
            continue
        now = own.sha(p) if p.exists() else None
        if now != ref["sha256"]:
            bad.append(f"{name}: {p} is {now and now[:12]}, sealed as {str(ref['sha256'])[:12]}")
    if cache is None:
        own.save()
    if bad:
        raise SystemExit(f"gen {generation} {stage}: sealed inputs no longer match their upstream "
                         f"({'; '.join(bad)}). An upstream stage was redone after this one was sealed; "
                         "redo from that stage with --redo-downstream")


def verify_before(run_dir: Path, generation: int, stage: str, *, inclusive: bool) -> None:
    """Verify every sealed stage before (and, if inclusive, at) (generation, stage)."""
    cache = _ShaCache(run_dir)
    k = stage_key(generation, stage)
    try:
        for g, s in sealed_stages(run_dir):
            if stage_key(g, s) < k or (inclusive and stage_key(g, s) == k):
                verify_stage(run_dir, g, s, cache=cache)
    finally:
        cache.save()


def unseal_after(run_dir: Path, generation: int, stage: str, *, include_self: bool) -> dict[str, Any]:
    """Move every sealed stage after (generation, stage) to ``superseded/`` and
    rewind the run-level files derived from them. Returns what was done."""
    from chess_anti_engine.simple_rl.evaluate import LEDGER, replace_ledger_rows

    run_dir = Path(run_dir)
    k = stage_key(generation, stage)
    targets = [gs for gs in sealed_stages(run_dir) if stage_key(*gs) > k]
    dest = _fresh_superseded_dir(run_dir) if targets else None
    moved = []
    for g, s in targets:
        assert dest is not None
        src = stage_dir(run_dir, g, s)
        dst = dest / f"gen_{g:04d}" / s
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        moved.append(f"gen_{g:04d}/{s}")
    unsealed = set(targets) | ({(int(generation), stage)} if include_self else set())
    rewound: list[str] = []
    for g in sorted({g for g, s in unsealed if s == "eval"}):
        if (run_dir / LEDGER).exists():
            replace_ledger_rows(run_dir / LEDGER, g, [])
            rewound.append(f"{LEDGER} gen {g}")
    if any(s == "train" for _, s in unsealed) and (run_dir / "presentation_ledger.json").exists():
        (run_dir / "presentation_ledger.json").unlink()  # derived; the next train rewrites it
        rewound.append("presentation_ledger.json")
    if {(1, "train"), (1, "collect")} & unsealed and (run_dir / "probe.npz").exists():
        (run_dir / "probe.npz").unlink()
        rewound.append("probe.npz")
    collect_gens = [g for g, s in unsealed if s == "collect"]
    if collect_gens:
        first = min(collect_gens)
        for sub in ("dup_sketch", "openings_seen"):
            for p in (run_dir / sub).glob("through_gen_*"):
                if int(p.stem.split(".")[0].rsplit("_", 1)[1]) >= first:
                    p.unlink()
                    rewound.append(f"{sub}/{p.name}")
    return {"moved_to": str(dest) if moved else None, "moved": moved, "rewound": rewound}
