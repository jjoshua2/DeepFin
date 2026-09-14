"""Bounded physical-row join for the explicit G10 Downside300 policy path.

No rank cache or full-corpus row dictionary. One derived shard's requested rows
are retained while each referenced raw shard is streamed. SF values stay intact.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from scripts import corpus_row_provenance as provenance
from scripts import derive_corpus_targets as derive
from scripts import sf_d9_rank_sidecar as rank
from scripts.bt4_policy_dump import file_sha256

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class SelectedG10:
    def __init__(self, args: argparse.Namespace, summary: dict[str, Any],
                 raw_dir: Path, source: Path,
                 observation: Callable[[dict[str, Any], str], Any]):
        self.observation = observation
        self.raw_dir, self.source, self.summary = raw_dir, source, summary
        path = Path(args.selected_g10_roster).resolve()
        self.metadata = {path: args.expected_selected_g10_roster_sha256}
        require(file_sha256(path) == self.metadata[path], "selected roster pin differs")
        selection = json.loads(path.read_text())
        manifest_path = raw_dir / "manifest.json"
        self.metadata[manifest_path] = file_sha256(manifest_path)
        manifest = derive.corpus.read_launch_manifest(raw_dir)
        self.config = summary["corpus"]["config_sha256"]
        derive.validate_selection_metadata(
            selection, source_dir=raw_dir, source_config_sha256=self.config,
            source_manifest_sha256=self.metadata[manifest_path],
        )
        staircase = [{"depth": 9, "width": "all"}, {"depth": 10, "width": "8"},
                     {"depth": 12, "width": "4"}]
        require(manifest["config_sha256"] == self.config
                and manifest["row_schema"] == 3
                and manifest["staircase_parsed"] == staircase
                and summary["corpus"]["staircase_parsed"] == staircase
                and summary["corpus"]["staircase_gate"].get("policy") == "g10",
                "selected G10 raw configuration differs")
        scheme = summary["scheme"]
        require(scheme["canonical"] == "uniform-d9" and scheme["kind"] == "uniform"
                and scheme["depth"] == 9 and scheme.get("value_depth") is None
                and scheme.get("policy_observation") == "phase0"
                and scheme.get("value_observation") == "latest-phase"
                and scheme["value_source"] == "deepest_phase_covering",
                "selected G10 requires phase0 policy/latest-phase value")
        require(summary["value_scheme"]["name"] == "search"
                and summary.get("value_target_postprocess") is None
                and summary.get("policy_target_postprocess") is None,
                "selected G10 requires original SF source values and policy")
        require(summary["temp_requested"] == 0.0005 and summary["floor_requested"] == 0
                and summary["cp_map"]["cp_slope"] == 0.006
                and summary["cp_map"]["cp_draw_width"] == 120,
                "selected G10 SF target contract differs")
        inp = summary["input"]
        require(inp["input_history_encoding"] == "lc0_root_legacy_meta"
                and inp["history_rep_fix"] is True and inp["zero_history"] is False
                and inp["input_extra_features"] == "v2_threats",
                "selected G10 history contract differs")
        require(summary["corpus"]["dir"] == raw_dir.name
                and summary["seed_effect"] == "permutes rows WITHIN each shard; changes no target value",
                "selected source directory/shuffle contract differs")
        realized = summary["realized"]
        rows = realized["rows_written"]
        require(type(rows) is int and rows > 0
                and realized["input_key_verified"] == realized["support_checks"] == rows
                and realized.get("rows_dropped_envelope", 0) == 0
                and sum(s["rows"] for s in summary["shards"]) == rows
                and summary["row_provenance"]["schema"] == 1
                and summary["row_provenance"]["path_in_shard"] == provenance.FILENAME,
                "selected G10 full retained-row proof missing")
        # Actual G10 selected products can be finalized while global generation continues.
        # Never rewrite corpus_complete; each stored shard must pass shard_contract.
        old_selection = summary.get("source_selection")
        if old_selection:
            require(all(old_selection[k] == selection[k] for k in selection),
                    "selected roster differs from source derivation")
        self.entries = {entry["source_shard"]: entry for entry in selection["shards"]}
        self.order = {name: i for i, name in enumerate(self.entries)}
        self.starts: dict[str, int] = {}
        total = 0
        for name, entry in self.entries.items():
            self.starts[name] = total
            total += entry["rows"]
        self.limit = summary["limit_requested"]
        require(type(self.limit) is int and 0 < self.limit <= total
                and self.limit == realized["rows_read"], "selected raw prefix differs")
        self.namespace = provenance._namespace(raw_dir, self.config)[0]
        self.exclusions, exclusion_path, exclusion_sha = rank._policy_support_exclusions(
            summary, source_dir=source, raw_dir=raw_dir, raw_config=self.config,
        )
        if exclusion_path is not None:
            assert exclusion_sha is not None
            self.metadata[exclusion_path] = exclusion_sha
        self.last_physical: tuple[int, int] = (-1, -1)
        self.raw_rows_decoded = 0
        self.raw_proofs: dict[str, Any] = {}
        self.producer_hashes = {str(Path(__file__)): file_sha256(Path(__file__)),
                               str(Path(provenance.__file__)): file_sha256(Path(provenance.__file__))}

    def join(self, spec: dict[str, Any], guard: Callable[[], None],
             raw_cap: int | None) -> list[Any]:
        rows = spec["rows"]
        require(type(rows) is int and 0 < rows <= 8192, "selected derived shard exceeds8192")
        shard = self.source / spec["path"]
        group: Any = zarr.open_group(str(shard), mode="r")
        stamp = spec["row_provenance"]
        require(dict(group.attrs).get("derive_row_provenance") == stamp
                and stamp["schema"] == 1 and stamp["rows"] == rows
                and stamp["record_bytes"] == provenance.RECORD_DTYPE.itemsize
                and stamp["path"] == provenance.FILENAME, "selected provenance stamp differs")
        ref_path = shard / provenance.FILENAME
        require(file_sha256(ref_path) == stamp["sha256"], "selected provenance hash differs")
        refs = provenance.read(ref_path, rows=rows)
        requests: dict[str, dict[int, tuple[int, dict[str, Any]]]] = {}
        physical = []
        for offset, ref in enumerate(refs):
            name, index = ref["source_shard"], ref["source_row"]
            require(ref["source_dir"] == str(self.raw_dir)
                    and ref["source_config_sha256"] == self.config
                    and ref["source_namespace"] == self.namespace and name in self.entries,
                    "selected provenance source differs")
            require(0 <= index < self.entries[name]["rows"]
                    and self.starts[name] + index < self.limit
                    and (name, index) not in self.exclusions,
                    "excluded or out-of-range physical row")
            targets = requests.setdefault(name, {})
            require(index not in targets, "duplicate physical row")
            targets[index] = (offset, ref)
            physical.append((self.order[name], index))
        # Existing derivation shuffles only within each emitted shard. This constant-
        # space range check detects cross-shard duplicates without a 35M-entry set.
        require(min(physical) > self.last_physical, "physical shard ranges overlap or regress")
        self.last_physical = max(physical)
        ordered: list[Any] = [None] * rows
        games, plies = group["game_id"][:], group["ply_index"][:]
        x_array = group["x"]
        require(x_array.dtype == np.dtype(np.float16) and x_array.shape[0] == rows
                and math.prod(x_array.shape) * x_array.dtype.itemsize <= 256 * 1024**2,
                "selected x working set exceeds256MiB")
        x = np.asarray(x_array[:])
        for name, targets in requests.items():
            guard()
            path = self.raw_dir / name
            before = derive._selection_stat(path)
            require(path.stat().st_size <= 64 * 1024**2, "selected raw file exceeds64MiB")
            require(file_sha256(path) == self.entries[name]["source_sha256"],
                    "selected raw content pin differs")
            count = 0
            for index, raw in enumerate(derive.iter_corpus_rows(path)):
                require(raw_cap is None or self.raw_rows_decoded < raw_cap,
                        "selected pilot raw-row cap exhausted")
                if index % 128 == 0:
                    guard()
                self.raw_rows_decoded += 1
                count += 1
                require(count <= self.entries[name]["rows"], "selected raw count exceeds roster")
                if index not in targets:
                    continue
                offset, ref = targets[index]
                require(raw.get("result") is not None, "selected row has no result")
                obs = self.observation(raw, self.config)
                require(all(raw[k] == ref[k] for k in ("worker_id", "game_id", "ply", "input_key"))
                        and obs.game == games[offset] and obs.ply == plies[offset]
                        and ref["stored_input_key"] == derive.corpus.input_tensor_key(
                            x[offset]),
                        "selected raw/stored history identity differs")
                ordered[offset] = obs
            require(count == self.entries[name]["rows"] and derive._selection_stat(path) == before,
                    "selected raw count or storage changed")
            self.raw_proofs[str(path)] = {"identity": rank._file_identity(path),
                                         "sha256": self.entries[name]["source_sha256"],
                                         "rows_per_pass": count}
        require(all(obs is not None for obs in ordered), "missing selected raw rows")
        return [obs for obs in ordered if obs is not None]

    def copy_exclusion_evidence(self, writing: Path) -> None:
        evidence = self.source / derive.POLICY_SUPPORT_MISSES_FILE
        if evidence in self.metadata:
            target = writing / evidence.name
            shutil.copy2(evidence, target)
            require(file_sha256(target) == self.metadata[evidence], "copied exclusion evidence differs")

    def verify(self) -> None:
        for path, digest in self.metadata.items():
            require(file_sha256(path) == digest, "selected metadata changed")
        for path, proof in self.raw_proofs.items():
            require(rank._file_identity(Path(path)) == proof["identity"], "selected raw changed")
