#!/usr/bin/env python3
"""Run one frozen common-input batch with one or two bounded source lanes.

Only derive -> snapshot -> (adapt existing BT4 + rank) -> qualify. No inference,
training, target mixing, resume or automatic retry. See docs/common_input_batch.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import time
from typing import Any

from scripts.corpus_selection_schema import validate_selection_metadata

TOOLS = {
    "derive_corpus_targets.py",
    "adapt_raw_bt4_sidecars.py",
    "sf_d9_rank_sidecar.py",
}
STAGES = {"derive", "snapshot", "adapt", "rank", "qualify"}
BOOTSTRAP = "from numcodecs import blosc\nblosc.set_nthreads(2)\n"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def identity(path):
    s = Path(path).lstat()
    require(stat.S_ISREG(s.st_mode), f"not regular file: {path}")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def pinned(item):
    path = Path(item["path"])
    require(
        path.is_absolute() and path == path.resolve() and len(item["sha256"]) == 64,
        "invalid metadata pin",
    )
    before = identity(path)
    require(
        sha(path) == item["sha256"] and identity(path) == before,
        f"changed metadata pin: {path}",
    )
    return path


def selected(source):
    return read(pinned(source["selection"]))["shards"]


def matches_selection(proof, source):
    if not isinstance(proof, dict):
        return False
    item = source["selection"]
    original = read(pinned(item))
    requested = original.pop("shards")
    expected = {
        **original,
        "path": str(Path(item["path"]).resolve()),
        "sha256": item["sha256"],
        "order": "original corpus shard order; limit applies after selection",
    }
    got = {k: v for k, v in proof.items() if k != "shards"}
    # The shared consumer uses original corpus order, even if selection JSON was reversed.
    return got == expected and sorted(
        proof.get("shards", []), key=lambda e: e["source_shard"]
    ) == sorted(requested, key=lambda e: e["source_shard"])


def source_storage(source):
    entries = read(pinned(source["source_metadata"]))
    wanted = {
        str(Path(source["source_dir"]) / e["source_shard"]) for e in selected(source)
    }
    require(
        len(entries) == len(wanted) and {e["source_path"] for e in entries} == wanted,
        "source stat universe differs",
    )
    for entry in entries:
        expected = [
            entry[k] for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
        ]
        require(
            identity(entry["source_path"]) == expected,
            f"selected raw storage changed: {entry['source_path']}",
        )
        side = Path(entry["sidecar_path"])
        require(
            side.parent == Path(source["sidecar_dir"]) and not side.is_symlink(),
            "foreign/aliased sidecar",
        )
        pinned(entry["sidecar_attrs_snapshot"])
        require(
            sha(side / ".zattrs") == entry["sidecar_attrs_snapshot"]["sha256"],
            "teacher attrs changed",
        )


def available_cpus():
    return os.sched_getaffinity(0)


def validate_manifest(plan):
    require(
        set(plan) - {"overlap_adapt_rank"}
        == {
            "schema",
            "state",
            "checkout",
            "commit",
            "python",
            "runtime_qualification",
            "preregistration",
            "pins",
            "max_concurrent_sources",
            "derive_options",
            "limits",
            "sources",
        },
        "unknown or missing manifest fields",
    )
    require(
        plan["schema"] == 1
        and type(plan["max_concurrent_sources"]) is int
        and plan["max_concurrent_sources"] in (1, 2),
        "invalid concurrency/schema",
    )
    require(
        type(plan.get("overlap_adapt_rank", False)) is bool,
        "overlap_adapt_rank must be boolean",
    )
    sources = plan["sources"]
    require(1 <= len(sources) <= 2, "one or two source lanes required")
    require(
        plan["max_concurrent_sources"] <= len(sources),
        "concurrency exceeds source lanes",
    )
    state = Path(plan["state"])
    require(
        state.is_absolute() and state == state.resolve() and state.parent.is_dir(),
        "canonical state parent required",
    )
    limits = plan["limits"]
    require(
        set(limits)
        == {
            "wall_seconds_including_kill",
            "numeric_threads",
            "nice",
            "ionice_class",
            "CUDA_VISIBLE_DEVICES",
            "new_output_cache_bytes",
            "minimum_free_bytes",
            "adapter_index_cache_bytes",
            "rank_index_cache_bytes",
        },
        "unknown resource limit",
    )
    require(
        type(limits["wall_seconds_including_kill"]) is int
        and limits["wall_seconds_including_kill"] > 30,
        "invalid deadline",
    )
    require(
        limits["numeric_threads"] == 2
        and limits["nice"] == 19
        and limits["ionice_class"] == 3
        and limits["CUDA_VISIBLE_DEVICES"] == "",
        "two-thread low-priority CPU-only runtime required",
    )
    for field in (
        "new_output_cache_bytes",
        "minimum_free_bytes",
        "adapter_index_cache_bytes",
        "rank_index_cache_bytes",
    ):
        require(type(limits[field]) is int and limits[field] > 0, f"invalid {field}")
    opts = plan["derive_options"]
    require(
        set(opts)
        == {
            "scheme",
            "policy_observation",
            "value_observation",
            "value_scheme",
            "temp",
            "floor",
            "workers",
            "row_provenance",
            "seed",
            "rows_per_shard",
        },
        "unknown derive option",
    )
    require(
        {
            k: opts[k]
            for k in (
                "scheme",
                "policy_observation",
                "value_observation",
                "value_scheme",
                "temp",
                "floor",
                "workers",
                "row_provenance",
            )
        }
        == {
            "scheme": "uniform-d9",
            "policy_observation": "phase0",
            "value_observation": "latest-phase",
            "value_scheme": "search",
            "temp": 0.0005,
            "floor": 0,
            "workers": 2,
            "row_provenance": True,
        },
        "unsupported common-input semantics",
    )
    require(
        type(opts["seed"]) is int
        and opts["seed"] >= 0
        and type(opts["rows_per_shard"]) is int
        and opts["rows_per_shard"] > 0,
        "invalid shuffle/output geometry",
    )
    ids, all_refs, affinities, all_outputs = set(), set(), [], []
    inputs = [
        Path(plan["checkout"]),
        Path(plan["python"]).resolve(),
        pinned(plan["runtime_qualification"]),
        pinned(plan["preregistration"]),
    ]
    for source in sources:
        name = source["source_id"]
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", name)
            and name not in ids,
            "invalid/duplicate source id",
        )
        ids.add(name)
        affinity = source["cpu_affinity"]
        require(
            isinstance(affinity, list)
            and len(affinity) == 2
            and len(set(affinity)) == 2
            and all(type(c) is int and c >= 0 for c in affinity),
            "two distinct CPU ids required",
        )
        require(
            set(affinity) <= set(available_cpus()),
            "lane CPUs outside available affinity",
        )
        affinities.append(set(affinity))
        source_root, side_root = Path(source["source_dir"]), Path(source["sidecar_dir"])
        require(
            all(
                p.is_absolute() and p == p.resolve() and p.is_dir()
                for p in (source_root, side_root)
            ),
            "canonical source/sidecar required",
        )
        inputs.extend([source_root, side_root])
        inputs.extend(
            pinned(source[field])
            for field in (
                "source_manifest",
                "selection",
                "closed_bt4_receipts",
                "source_metadata",
            )
        )
        require(
            Path(source["source_manifest"]["path"]) == source_root / "manifest.json",
            "original source manifest required",
        )
        selection = read(source["selection"]["path"])
        validate_selection_metadata(
            selection,
            source_dir=source_root,
            source_config_sha256=read(source["source_manifest"]["path"]).get(
                "config_sha256"
            ),
            source_manifest_sha256=source["source_manifest"]["sha256"],
        )
        entries = selection["shards"]
        require(
            type(source["physical_rows"]) is int
            and sum(e["rows"] for e in entries) == source["physical_rows"],
            "selected row count differs",
        )
        for entry in entries:
            n = entry["source_shard"]
            key = (str(source_root), n)
            require(key not in all_refs, "overlapping source selections")
            all_refs.add(key)
        for field in ("support_drop_ceiling", "missing_result_count_ceiling"):
            require(
                type(source[field]) is int
                and 0 <= source[field] < source["physical_rows"],
                f"invalid {field}",
            )
        require(
            type(source["missing_result_fraction_ceiling"]) in (int, float)
            and math.isfinite(source["missing_result_fraction_ceiling"])
            and 0 <= source["missing_result_fraction_ceiling"] < 1
            and source["missing_result_count_ceiling"]
            == math.floor(
                source["physical_rows"] * source["missing_result_fraction_ceiling"]
            ),
            "missing-result count/fraction disagree",
        )
        outputs = [
            Path(source[k]) for k in ("derived_output", "adapted_output", "rank_output")
        ]
        require(
            len(set(outputs)) == 3
            and all(
                p.is_absolute() and p == p.resolve() and p.parent == state / name
                for p in outputs
            ),
            "distinct canonical lane outputs required",
        )
        all_outputs.extend(outputs)
        source_storage(source)
    require(
        plan["max_concurrent_sources"] == 1 or not (affinities[0] & affinities[1]),
        "concurrent lane affinities overlap",
    )
    require(
        all(
            out != p and out not in p.parents and p not in out.parents
            for out in all_outputs
            for p in inputs
        ),
        "output overlaps input",
    )


def verify(plan):
    cwd = plan["checkout"]
    require(
        subprocess.check_output(
            ["git", "-C", cwd, "rev-parse", "HEAD"], text=True
        ).strip()
        == plan["commit"],
        "runtime revision changed",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", cwd, "diff", "--name-only", "HEAD"], text=True
        ).strip(),
        "tracked runtime changed",
    )
    runtime = read(pinned(plan["runtime_qualification"]))
    require(
        runtime["status"] == "qualified"
        and all(runtime[k] == plan[k] for k in ("checkout", "commit", "python")),
        "qualified runtime binding differs",
    )
    require(
        all(
            runtime.get("features", {}).get(k) is True
            for k in ("closed_shard_selection", "support_exclusion_requires_result")
        ),
        "missing qualified consumer features",
    )
    required = {str(Path(cwd) / "scripts" / s) for s in TOOLS} | {
        str(Path(plan["python"]).resolve()),
        str(Path(__file__).resolve()),
        str(Path(cwd) / "scripts" / "corpus_selection_schema.py"),
        "/usr/bin/timeout",
        "/usr/bin/time",
    }
    require(
        required <= set(plan["pins"])
        and all(plan["pins"].get(p) == h for p, h in runtime["pins"].items()),
        "missing runtime pins",
    )
    for path, digest in plan["pins"].items():
        require(sha(path) == digest, f"changed runtime/input pin: {path}")


def usage(state, *, stable=False):
    total = 0

    def error(exc):
        if not isinstance(exc, FileNotFoundError) or stable:
            raise exc

    for root, dirs, files in os.walk(state, onerror=error):
        for name in [*dirs, *files]:
            path = Path(root) / name
            try:
                st = path.lstat()
            except FileNotFoundError:
                if stable:
                    raise
                continue
            require(not stat.S_ISLNK(st.st_mode), f"output symlink: {path}")
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
    return total


class Guard:
    def __init__(self, plan, deadline):
        self.plan, self.deadline = plan, deadline
        self.last_size = float("-inf")
        self.peak_observed_output_bytes = 0
        self.minimum_observed_free_bytes = None

    def check(self, *, force=False, stable=False):
        state = Path(self.plan["state"])
        limits = self.plan["limits"]
        require(time.time() < self.deadline, "overall deadline reached")
        require(
            not any((p / "STOP").exists() for p in (state, state.parent)),
            "STOP requested",
        )
        free = shutil.disk_usage(state).free
        self.minimum_observed_free_bytes = (
            free
            if self.minimum_observed_free_bytes is None
            else min(free, self.minimum_observed_free_bytes)
        )
        require(free >= limits["minimum_free_bytes"], "free-space reserve reached")
        now = time.monotonic()
        if force or now - self.last_size >= 60:
            size = usage(state, stable=stable)
            self.peak_observed_output_bytes = max(size, self.peak_observed_output_bytes)
            require(
                size <= limits["new_output_cache_bytes"],
                "aggregate output/cache cap reached",
            )
            self.last_size = now


def command(plan, script, *args):
    require(script in TOOLS, "unregistered tool")
    return [
        plan["python"],
        str(Path(plan["checkout"]) / "scripts" / script),
        *map(str, args),
    ]


def actual_exclusions(source, summary):
    require(
        matches_selection(summary.get("source_selection"), source),
        "derived selection differs",
    )
    realized = summary["realized"]
    support = realized.get("policy_support_exclusions", [])
    ns, nn, nr, nw = [
        realized.get(k, 0) if k == "rows_dropped_policy_support" else realized.get(k)
        for k in [
            "rows_dropped_policy_support",
            "rows_dropped_no_result",
            "rows_read",
            "rows_written",
        ]
    ]
    require(
        all(type(n) is int and n >= 0 for n in [ns, nn, nr, nw]),
        "invalid drop/row counters",
    )
    require(
        ns == len(support) <= source["support_drop_ceiling"],
        "support drop cap/count differs",
    )
    require(
        summary.get("max_policy_support_misses", 0) == source["support_drop_ceiling"],
        "wrong support budget",
    )
    require(nn <= source["missing_result_count_ceiling"], "missing-result cap exceeded")
    require(
        realized["rows_dropped_envelope"] == 0
        and nr == source["physical_rows"]
        and nw == nr - ns - nn,
        "unexpected survival/drop accounting",
    )
    if ns:
        require(
            summary.get("policy_support_misses_file") == "policy_support_misses.jsonl",
            "missing actual support ledger",
        )
        path = Path(source["derived_output"]) / "policy_support_misses.jsonl"
        require(
            not path.is_symlink() and path.is_file(), "support ledger is not regular"
        )
        require(
            [json.loads(line) for line in path.read_text().splitlines()] == support,
            "support ledger differs from summary",
        )
    else:
        require(
            summary.get("policy_support_misses_file") is None,
            "unexpected zero-count support ledger",
        )
    universe = {e["source_shard"]: e["rows"] for e in selected(source)}
    excluded = set()
    for ref in support:
        key = (ref["source_shard"], ref["source_row"])
        require(
            ref["source_dir"] == source["source_dir"]
            and key[0] in universe
            and type(key[1]) is int
            and 0 <= key[1] < universe[key[0]]
            and key not in excluded,
            "foreign, duplicate or out-of-range support reference",
        )
        require(
            ref["reason"] == "selected_phase0_policy_support"
            and ref["policy_depth"] == 9
            and ref["full_history_input_key_verified"] is True,
            "unsupported exclusion reason",
        )
        excluded.add(key)
    return excluded


def verify_rank_accounting(source, summary, rank):
    realized = summary["realized"]
    require(
        matches_selection(rank.get("source_selection"), source),
        "rank selection differs",
    )
    require(
        rank["raw_rows_read"] == source["physical_rows"]
        and rank["rows_dropped_no_result"] == realized["rows_dropped_no_result"]
        and rank["row_provenance"].get("rows_dropped_policy_support", 0)
        == realized.get("rows_dropped_policy_support", 0)
        and rank["rows"] == realized["rows_written"],
        "independent rank/drop accounting differs",
    )
    require(
        rank["row_provenance"]["join"]
        == "source-qualified-physical-row-and-full-history-keys-v1",
        "missing independent raw eligibility/injectivity proof",
    )


def verify_complement(source, summary, rank, covered):
    verify_rank_accounting(source, summary, rank)
    survived = sum(sum(bits) for bits in covered.values())
    omitted = source["physical_rows"] - survived
    require(
        survived == summary["realized"]["rows_written"]
        and omitted
        == summary["realized"].get("rows_dropped_policy_support", 0)
        + rank["rows_dropped_no_result"],
        "injective eligible mapping does not cover complete survivor complement",
    )
    return omitted


def stop_owned_group(child, grace: float = 5):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Group members can survive the group leader; always reap the owned group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def snapshot_derived(source):
    from scripts.adapt_raw_bt4_sidecars import storage_identity

    root = Path(source["derived_output"])
    summary = read(root / "derive_targets_summary.json")
    require(
        summary["realized"]["rows_read"] == source["physical_rows"],
        "raw row accounting differs",
    )
    require(
        summary["realized"]["rows_dropped_envelope"] == 0, "unexpected envelope drops"
    )
    actual_exclusions(source, summary)
    write(
        root.parent / "derived_identity.json",
        {
            "source_summary_sha256": sha(root / "derive_targets_summary.json"),
            "storage_identity": storage_identity(root),
            "realized": summary["realized"],
        },
    )


def qualify(source):
    import numpy as np
    import zarr
    from scripts import corpus_row_provenance as provenance, bt4_policy_mix as mix
    from scripts.adapt_raw_bt4_sidecars import storage_identity

    importlib.import_module("numcodecs.blosc").set_nthreads(2)
    root = Path(source["derived_output"])
    before = read(root.parent / "derived_identity.json")
    require(
        storage_identity(root) == before["storage_identity"],
        "derived inputs changed during adapter/rank",
    )
    summary_path = root / "derive_targets_summary.json"
    summary = read(summary_path)
    summary_sha = sha(summary_path)
    excluded = actual_exclusions(source, summary)
    require(summary_sha == before["source_summary_sha256"], "derived summary changed")
    paths = sorted(root.glob("shard_*.zarr"))
    names = {p.name for p in paths}
    require(
        names == {e["path"] for e in summary["shards"]}
        and len(paths) == len(summary["shards"]),
        "source membership differs",
    )
    side_root, rank_root = Path(source["adapted_output"]), Path(source["rank_output"])
    for other in (side_root, rank_root):
        require(
            {p.name for p in other.glob("shard_*.zarr")} == names,
            "sidecar membership differs",
        )
    side = read(side_root / mix.SIDECAR_SUMMARY)
    rank = read(rank_root / "sf_d9_rank_sidecar_summary.json")
    n = summary["realized"]["rows_written"]
    require(
        side["rows"] == rank["rows"] == n
        and side["source_dir"] == rank["source_dir"] == str(root),
        "sidecar summary source/count differs",
    )
    require(
        rank["source_derive_summary_sha256"] == summary_sha
        and rank["top_k"] == 3
        and rank["raw_dir"] == source["source_dir"]
        and rank["row_provenance"]["raw_shards_read_once"] == len(selected(source))
        and rank["raw_rows_read"] == source["physical_rows"]
        and rank["row_provenance"]["policy_observation"] == "phase0"
        and rank["row_provenance"]["value_observation"] == "latest-phase",
        "rank observation or raw coverage differs",
    )
    verify_rank_accounting(source, summary, rank)
    side_payloads = {e["path"]: e for e in side["adapter"]["written_shards"]}
    require(set(side_payloads) == names, "adapter payload lineage inventory differs")
    by_name = {e["source_shard"]: e for e in selected(source)}
    covered = {name: bytearray(e["rows"]) for name, e in by_name.items()}
    digest = hashlib.sha256()
    rows = 0
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        x = group["x"]
        if not isinstance(x, zarr.Array):
            raise ValueError("source x is not an array")
        count = int(x.shape[0])
        stamp = dict(group.attrs)["derive_row_provenance"]
        require(
            sha(path / provenance.FILENAME) == stamp["sha256"], "row provenance changed"
        )
        refs = provenance.read(path / provenance.FILENAME, rows=count)
        for ref in refs:
            require(
                ref["source_dir"] == source["source_dir"]
                and ref["source_shard"] in by_name,
                "foreign original source",
            )
            bitmap = covered[ref["source_shard"]]
            offset = ref["source_row"]
            require(
                0 <= offset < len(bitmap) and not bitmap[offset],
                "duplicate/out-of-bounds source row",
            )
            require(
                (ref["source_shard"], offset) not in excluded,
                "excluded source row survived",
            )
            bitmap[offset] = 1
            digest.update(json.dumps(ref, sort_keys=True).encode())
        _, keys, key_sha, policy_sha = mix._sidecar_identity(group, path)
        attrs = mix._validate_sidecar(
            side_root / path.name,
            source_path=path,
            source_keys=keys,
            source_key_sha=key_sha,
            source_policy_sha=policy_sha,
            onnx_sha=side["onnx"]["sha256"],
            providers=side["providers"],
            policy_output=side["policy_output"],
        )
        require(attrs["source_dir"] == str(root), "BT4 original derived parent differs")
        mix._validate_sf_rank_sidecar(
            rank_root / path.name,
            source_group=group,
            source_path=path,
            source_summary_sha256=summary_sha,
            required_top_k=3,
        )
        policy = zarr.open_group(str(side_root / path.name), mode="r")[
            mix.SIDECAR_POLICY_FIELD
        ]
        payload_digest = hashlib.sha256()
        for start in range(0, count, 256):
            p = np.asarray(policy[start : start + 256])
            payload_digest.update(np.ascontiguousarray(p).tobytes())
            legal = np.asarray(group["legal_mask"][start : start + 256]) != 0
            require(
                np.all(np.isfinite(p))
                and np.all(p >= 0)
                and np.all(p[~legal] == 0)
                and np.allclose(p.sum(axis=1, dtype=np.float64), 1, atol=2e-6, rtol=0),
                "BT4 legal normalized mass differs",
            )
        require(
            payload_digest.hexdigest()
            == attrs["bt4_policy_sha256"]
            == side_payloads[path.name]["bt4_policy_sha256"],
            "adapted payload changed after publication",
        )
        require(
            attrs["row_provenance_sha256"] == stamp["sha256"]
            and attrs["source_derive_summary_sha256"] == summary_sha,
            "adapter source lineage differs",
        )
        rows += count
    require(rows == n, "emitted row total differs")
    omitted = verify_complement(source, summary, rank, covered)
    require(
        storage_identity(root) == before["storage_identity"],
        "derived storage changed during qualification",
    )
    write(
        root.parent / "common_input_qualification.json",
        {
            "status": "complete",
            "rows": rows,
            "physical_rows": source["physical_rows"],
            "derive_realized": summary["realized"],
            "source_selection": summary["source_selection"],
            "independent_rank_missing_result_rows": rank["rows_dropped_no_result"],
            "verified_support_exclusion_rows": len(excluded),
            "omitted_rows": omitted,
            "complement_proof": "rank eligibility plus injective full-history physical joins and exact universe cardinality; omitted IDs reconstructible from selection universe minus emitted refs",
            "per_raw_shard_survivors": {
                name: sum(bits) for name, bits in covered.items()
            },
            "source_qualified_input_sequence_sha256": digest.hexdigest(),
            "unchanged_derived_storage_identity": before["storage_identity"],
            "summary_pins": {
                str(p): sha(p)
                for p in [
                    summary_path,
                    side_root / mix.SIDECAR_SUMMARY,
                    rank_root / "sf_d9_rank_sidecar_summary.json",
                ]
            },
            "scope": "All emitted rows admitted by source-bound BT4/rank consumers, unique source physical rows and immutable derived non-policy lineage; no training schedule execution.",
        },
    )


def descendants(pid):
    found = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = (
                Path(f"/proc/{current}/task/{current}/children").read_text().split()
            )
        except FileNotFoundError:
            continue
        for item in children:
            child = int(item)
            if child in found:
                continue
            try:
                argv = (
                    Path(f"/proc/{child}/cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode()
                )
            except FileNotFoundError:
                continue
            found[child] = argv
            pending.append(child)
    return found


def stage(name, argv, plan, source, guard):
    require(name in STAGES, "unregistered stage")
    guard.check(force=True)
    state = Path(plan["state"]) / source["source_id"]
    metrics = state / (name + ".time.json")
    require(not os.path.lexists(metrics), "existing stage timing")
    timed = [
        "/usr/bin/time",
        "-o",
        str(metrics),
        "-f",
        '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"filesystem_inputs":%I,"filesystem_outputs":%O,"exit_code":%x}',
        *argv,
    ]
    started = time.time()
    observed = {}
    with (state / (name + ".log")).open("xb") as log:
        # Inherits the lane timeout's process group: outer timeout still owns the
        # stage and multiprocessing children if either Python coordinator dies.
        child = subprocess.Popen(
            timed, cwd=plan["checkout"], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            write(
                state / (name + ".started.json"),
                {"pid": child.pid, "argv": timed, "start_unix": started},
            )
            while child.poll() is None:
                observed.update(descendants(child.pid))
                guard.check()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            require(
                child.returncode == 0,
                f"{source['source_id']}.{name} exit {child.returncode}",
            )
            guard.check()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise  # lane/coordinator failure cleans the complete owned lane group
    measured = read(metrics)
    require(measured["exit_code"] == 0, "stage resource status differs")
    write(
        state / (name + ".completed.json"),
        {
            "pid": child.pid,
            "argv": timed,
            "start_unix": started,
            "end_unix": time.time(),
            "exit_code": child.returncode,
            "resources": measured,
            "observed_owned_descendants": observed,
            "descendant_observation": "sampled; short-lived children may exit between observations",
        },
    )


def adapt_and_rank(adapt_argv, rank_argv, plan, source, guard):
    """Own just the two independent post-snapshot stages on the lane's CPU pair."""
    state = Path(plan["state"]) / source["source_id"]
    active: dict[str, tuple[subprocess.Popen, Any, list[str], float, dict]] = {}
    try:
        for name, argv in (("adapt", adapt_argv), ("rank", rank_argv)):
            guard.check(force=True)
            metrics = state / (name + ".time.json")
            require(not os.path.lexists(metrics), "existing stage timing")
            # Inherit the existing lane timeout group, affinity and environment.
            # Neither child gets a new deadline or an independently detached group.
            timed = [
                "/usr/bin/time",
                "-o",
                str(metrics),
                "-f",
                '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"filesystem_inputs":%I,"filesystem_outputs":%O,"exit_code":%x}',
                *argv,
            ]
            log = (state / (name + ".log")).open("xb")
            started = time.time()
            try:
                child = subprocess.Popen(
                    timed,
                    cwd=plan["checkout"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log.close()
                raise
            active[name] = (child, log, timed, started, {})
            write(
                state / (name + ".started.json"),
                {
                    "pid": child.pid,
                    "argv": timed,
                    "start_unix": started,
                    "deadline_unix": guard.deadline,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
            )
        while active:
            guard.check()
            for name, (child, log, timed, started, observed) in list(active.items()):
                observed.update(descendants(child.pid))
                if child.poll() is None:
                    continue
                code = child.returncode
                require(code == 0, f"{source['source_id']}.{name} exit {code}")
                child.wait()
                log.close()
                del active[name]
                measured = read(state / (name + ".time.json"))
                require(measured["exit_code"] == 0, "stage resource status differs")
                guard.check(force=True)
                write(
                    state / (name + ".completed.json"),
                    {
                        "pid": child.pid,
                        "argv": timed,
                        "start_unix": started,
                        "end_unix": time.time(),
                        "exit_code": code,
                        "resources": measured,
                        "deadline_unix": guard.deadline,
                        "observed_owned_descendants": observed,
                        "descendant_observation": "sampled; short-lived children may exit between observations",
                    },
                )
            if active:
                time.sleep(1)
    finally:
        # Signal all siblings before waiting through cleanup. This also handles
        # receipt-write failures and the lane's SIGTERM handler raising SystemExit.
        for child, *_ in active.values():
            try:
                child.terminate()
            except ProcessLookupError:
                pass
        for child, log, *_ in active.values():
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.close()
        # Any failure propagates to existing coordinator lane-group cleanup,
        # which also terminates descendants of these direct time wrappers.


def internal_command(plan, manifest, digest, action, source):
    return [
        plan["python"],
        str(Path(__file__).resolve()),
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        digest,
        "--action",
        action,
        "--source",
        source["source_id"],
    ]


def lane(plan, manifest, digest, source, deadline):
    verify(plan)
    source_storage(source)
    guard = Guard(plan, deadline)
    parent = Path(plan["state"]) / source["source_id"]
    parent.mkdir(exist_ok=True)
    opts = plan["derive_options"]
    argv = command(
        plan,
        "derive_corpus_targets.py",
        "--corpus",
        source["source_dir"],
        "--out",
        source["derived_output"],
        "--limit",
        source["physical_rows"],
        "--max-policy-support-misses",
        source["support_drop_ceiling"],
        "--source-shards",
        source["selection"]["path"],
    )
    for key in (
        "scheme",
        "policy_observation",
        "value_observation",
        "value_scheme",
        "temp",
        "floor",
        "seed",
        "rows_per_shard",
        "workers",
    ):
        argv += ["--" + key.replace("_", "-"), str(opts[key])]
    argv += ["--row-provenance"]
    stage("derive", argv, plan, source, guard)
    stage(
        "snapshot",
        internal_command(plan, manifest, digest, "snapshot", source),
        plan,
        source,
        guard,
    )
    summary_path = Path(source["derived_output"]) / "derive_targets_summary.json"
    summary = read(summary_path)
    receipts = [
        json.loads(line)
        for line in Path(source["closed_bt4_receipts"]["path"]).read_text().splitlines()
    ]
    require(receipts, "empty closed teacher receipt")
    receipt = receipts[0]
    attrs = read(Path(source["sidecar_dir"]) / receipt["sidecar"] / ".zattrs")
    mapping = {
        "schema": 1,
        "derived_summary": {"path": str(summary_path), "sha256": sha(summary_path)},
        "teacher": {
            "onnx": {"path": attrs["onnx_path"], "sha256": receipt["onnx_sha256"]},
            "policy_output": receipt["policy_output"],
            "providers": receipt["providers"],
            "remap": receipt["remap_provenance"],
        },
        "sources": [
            {
                "source_dir": source["source_dir"],
                "sidecar_dir": source["sidecar_dir"],
                "manifest": source["source_manifest"],
                "receipts": source["closed_bt4_receipts"],
            }
        ],
    }
    mapping_path = parent / "adapter_manifest.json"
    write(mapping_path, mapping)
    adapt_argv = command(
        plan,
        "adapt_raw_bt4_sidecars.py",
        "--manifest",
        mapping_path,
        "--expected-manifest-sha256",
        sha(mapping_path),
        "--out",
        source["adapted_output"],
        "--max-index-bytes",
        plan["limits"]["adapter_index_cache_bytes"],
    )
    rank_argv = command(
        plan,
        "sf_d9_rank_sidecar.py",
        "--raw",
        source["source_dir"],
        "--shards",
        source["derived_output"],
        "--out",
        source["rank_output"],
        "--limit",
        source["physical_rows"],
        "--top-k",
        3,
        "--seed",
        opts["seed"],
        "--rows-per-shard",
        opts["rows_per_shard"],
        "--max-provenance-cache-bytes",
        plan["limits"]["rank_index_cache_bytes"],
        "--source-shards",
        source["selection"]["path"],
        "--expected-rows",
        summary["realized"]["rows_written"],
        "--expected-shards",
        len(summary["shards"]),
        "--expected-source-summary-sha256",
        sha(summary_path),
    )
    if plan.get("overlap_adapt_rank", False):
        adapt_and_rank(adapt_argv, rank_argv, plan, source, guard)
    else:
        stage("adapt", adapt_argv, plan, source, guard)
        stage("rank", rank_argv, plan, source, guard)
    stage(
        "qualify",
        internal_command(plan, manifest, digest, "qualify", source),
        plan,
        source,
        guard,
    )
    verify(plan)
    source_storage(source)
    guard.check(force=True)
    write(
        parent / "lane_complete.json",
        {
            "status": "complete",
            "overlap_adapt_rank": plan.get("overlap_adapt_rank", False),
            "end_unix": time.time(),
            "qualification_sha256": sha(parent / "common_input_qualification.json"),
            "peak_sampled_aggregate_output_bytes": guard.peak_observed_output_bytes,
            "minimum_sampled_free_bytes": guard.minimum_observed_free_bytes,
        },
    )


def worker_environment(plan):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(Path(plan["state"]) / "python_bootstrap")
        + os.pathsep
        + plan["checkout"],
        CHESS_ANTI_ENGINE_LIVE_CONFIG=str(
            Path(plan["checkout"]) / "configs/pbt2_small.yaml"
        ),
    )
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLOSC_NTHREADS",
    ):
        env[key] = "2"
    return env


def lane_command(plan, manifest, digest, source, deadline):
    remaining = deadline - time.time() - 30
    require(remaining > 0, "preflight exhausted wall cap")
    return [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{remaining:.3f}s",
        "/usr/bin/nice",
        "-n",
        "19",
        "/usr/bin/ionice",
        "-c",
        "3",
        "/usr/bin/taskset",
        "-c",
        ",".join(map(str, source["cpu_affinity"])),
        *internal_command(plan, manifest, digest, "lane", source),
        "--deadline",
        str(deadline),
    ]


def fresh_outputs(plan):
    state = Path(plan["state"])
    require(
        not any(
            os.path.lexists(state / name)
            for name in (
                "started.json",
                "completed.json",
                "failed.json",
                "python_bootstrap",
            )
        ),
        "prior attempt exists; preserve it",
    )
    for source in plan["sources"]:
        parent = state / source["source_id"]
        require(
            not any(
                os.path.lexists(parent / name)
                for name in (
                    "lane.log",
                    "lane.started.json",
                    "lane_complete.json",
                    "adapter_manifest.json",
                    "derived_identity.json",
                    "common_input_qualification.json",
                )
            ),
            "prior source lane exists",
        )
        for kind in STAGES:
            require(
                not any(
                    os.path.lexists(parent / (kind + suffix))
                    for suffix in (
                        ".log",
                        ".time.json",
                        ".started.json",
                        ".completed.json",
                    )
                ),
                "prior stage exists",
            )
        for field in ("derived_output", "adapted_output", "rank_output"):
            p = Path(source[field])
            require(
                not os.path.lexists(p) and not os.path.lexists(str(p) + ".writing"),
                "existing output/partial refused",
            )


def execute(plan, manifest, digest, deadline=None):
    started = time.time()
    deadline = (
        started + plan["limits"]["wall_seconds_including_kill"]
        if deadline is None
        else deadline
    )
    require(
        math.isfinite(deadline)
        and started
        < deadline
        <= started + plan["limits"]["wall_seconds_including_kill"],
        "invalid outer deadline",
    )
    validate_manifest(plan)
    verify(plan)
    fresh_outputs(plan)
    state = Path(plan["state"])
    state.mkdir(exist_ok=True)
    guard = Guard(plan, deadline)
    guard.check(force=True, stable=True)
    # Exclusive attempt claim before creating any bootstrap or lane outputs.
    write(
        state / "started.json",
        {
            "pid": os.getpid(),
            "overlap_adapt_rank": plan.get("overlap_adapt_rank", False),
            "start_unix": started,
            "deadline_unix": deadline,
            "manifest_sha256": digest,
        },
    )
    active: dict[str, tuple[subprocess.Popen, Any]] = {}
    completed = []
    pending = list(plan["sources"])
    try:
        boot = state / "python_bootstrap"
        boot.mkdir()
        (boot / "sitecustomize.py").write_text(BOOTSTRAP)
        while pending or active:
            guard.check()
            while pending and len(active) < plan["max_concurrent_sources"]:
                source = pending.pop(0)
                name = source["source_id"]
                parent = state / name
                parent.mkdir(exist_ok=True)
                argv = lane_command(plan, manifest, digest, source, deadline)
                log = (parent / "lane.log").open("xb")
                try:
                    child = subprocess.Popen(
                        argv,
                        env=worker_environment(plan),
                        cwd=plan["checkout"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[name] = (child, log)
                write(
                    parent / "lane.started.json",
                    {
                        "pid": child.pid,
                        "start_unix": time.time(),
                        "argv": argv,
                        "cpu_affinity": source["cpu_affinity"],
                        "numeric_threads": 2,
                        "overlap_adapt_rank": plan.get("overlap_adapt_rank", False),
                        "manifest_sha256": digest,
                    },
                )
            for name, (child, log) in list(active.items()):
                if child.poll() is None:
                    continue
                code = child.returncode
                stop_owned_group(child)
                log.close()
                del active[name]
                require(code == 0, f"lane {name} exit {code}")
                receipt = read(state / name / "lane_complete.json")
                require(
                    receipt["status"] == "complete"
                    and receipt["qualification_sha256"]
                    == sha(state / name / "common_input_qualification.json"),
                    "lane completion differs",
                )
                completed.append(name)
            if active:
                time.sleep(1)
        verify(plan)
        for source in plan["sources"]:
            source_storage(source)
        guard.check(force=True, stable=True)
        write(
            state / "completed.json",
            {
                "status": "complete",
                "start_unix": started,
                "end_unix": time.time(),
                "manifest_sha256": digest,
                "completed_sources": completed,
                "peak_sampled_aggregate_output_bytes": guard.peak_observed_output_bytes,
                "minimum_sampled_free_bytes": guard.minimum_observed_free_bytes,
            },
        )
    except BaseException as exc:
        # Signal every owned lane first, so a failing source cancels its sibling
        # promptly rather than waiting sequentially through grace periods.
        for child, _ in active.values():
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child, log in active.values():
            stop_owned_group(child)
            log.close()
        write(
            state / "failed.json",
            {
                "status": "failed",
                "error": repr(exc),
                "start_unix": started,
                "end_unix": time.time(),
                "manifest_sha256": digest,
                "completed_sources": completed,
                "peak_sampled_aggregate_output_bytes": guard.peak_observed_output_bytes,
                "minimum_sampled_free_bytes": guard.minimum_observed_free_bytes,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--action",
        choices=["validate", "execute", "lane", "snapshot", "qualify"],
        default="validate",
    )
    parser.add_argument("--source")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()

    def terminate(_signum, _frame):
        raise SystemExit("termination requested")

    signal.signal(signal.SIGTERM, terminate)
    path = pinned(
        {"path": str(args.manifest.resolve()), "sha256": args.expected_manifest_sha256}
    )
    plan = read(path)
    if args.action == "execute":
        require(args.deadline is not None, "execute requires the outer launch deadline")
        return execute(plan, path, args.expected_manifest_sha256, args.deadline)
    if args.action == "validate":
        validate_manifest(plan)
        verify(plan)
        fresh_outputs(plan)
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_LAUNCHED",
                    "manifest_sha256": args.expected_manifest_sha256,
                }
            )
        )
        return
    # Every internal entry still checks the frozen runner and metadata binding.
    require(
        plan["pins"].get(str(Path(__file__).resolve())) == sha(__file__),
        "runner pin differs",
    )
    source = next(s for s in plan["sources"] if s["source_id"] == args.source)
    if args.action == "snapshot":
        return snapshot_derived(source)
    if args.action == "qualify":
        return qualify(source)
    require(
        args.deadline is not None and math.isfinite(args.deadline),
        "lane deadline required",
    )
    return lane(plan, path, args.expected_manifest_sha256, source, args.deadline)


if __name__ == "__main__":
    main()
