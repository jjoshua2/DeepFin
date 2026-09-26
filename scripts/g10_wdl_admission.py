"""Consume completed common-input receipts, without replaying raw G10 payloads.

The caller-supplied receipt SHA is the trust anchor. This checks its consistency
and frozen derived storage, not the historical qualifier's work a second time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from scripts.adapt_raw_bt4_sidecars import storage_identity
from scripts.bt4_policy_dump import file_sha256
from scripts.sf_policy_rewrite import require


def same(left: Any, right: Any) -> bool:
    # Historical summaries contain NaN diagnostics. Match their serialized
    # metadata while preserving integer/boolean distinctions.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def admit(
    path: Path, expected_sha: str, source: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    """Admit exactly one previously qualified source/batch derived namespace."""
    require(sha(expected_sha) and file_sha256(path) == expected_sha,
            "G10 qualification pin differs")
    q = json.loads(path.read_text())
    require(q.get("status") == "complete", "G10 qualification is not complete")
    root = source.parent
    require(path.resolve() == root / "common_input_qualification.json"
            and source.name == "derived", "G10 qualification/source namespace differs")
    paths = [source / "derive_targets_summary.json",
             root / "bt4/bt4_policy_sidecar_summary.json",
             root / "rank/sf_d9_rank_sidecar_summary.json"]
    pins = q["summary_pins"]
    require(set(pins) == {str(p) for p in paths}, "G10 summary namespaces differ")
    require(all(sha(pins[str(p)]) and file_sha256(p) == pins[str(p)] for p in paths),
            "G10 qualified summary pin differs")
    side, rank = (json.loads(p.read_text()) for p in paths[1:])
    realized = summary["realized"]
    rows, physical = q["rows"], q["physical_rows"]
    require(type(rows) is int and rows > 0 and type(physical) is int and physical >= rows
            and same(q["derive_realized"], realized), "G10 realized accounting differs")
    require(rows == realized["rows_written"] == side["rows"] == rank["rows"]
            and physical == realized["rows_read"] == rank["raw_rows_read"]
            and side["source_dir"] == rank["source_dir"] == str(source)
            and rank["source_derive_summary_sha256"] == pins[str(paths[0])],
            "G10 sidecar source/count differs")
    require(summary["corpus"]["corpus_complete"] is False
            and summary["corpus"]["staircase_gate"]["policy"] == "g10"
            and summary["scheme"]["policy_observation"] == "phase0"
            and summary["scheme"]["value_observation"] == "latest-phase"
            and rank["kind"] == "sf_d9_rank_gap_sidecar" and rank["top_k"] == 3
            and rank["raw_config_sha256"] == summary["corpus"]["config_sha256"],
            "G10 source/observation contract differs")
    raw_dir = Path(rank["raw_dir"])
    require(raw_dir.is_absolute() and str(raw_dir.resolve()) == str(raw_dir),
            "G10 raw namespace is not canonical")
    adapter = side["adapter"]
    require(adapter["derived_summary"] == {"path": str(paths[0]), "sha256": pins[str(paths[0])]},
            "G10 adapter summary differs")
    verified = adapter["verified_raw_shards"]
    inventory = []
    for name, item in sorted(verified.items()):
        receipt = item["receipt"]
        shard = receipt["source_shard"]
        require(Path(shard).name == shard and shard not in (".", "..")
                and name == str(raw_dir / shard) and sha(receipt["source_sha256"])
                and type(receipt["positions"]) is int and receipt["positions"] > 0,
                "G10 verified raw inventory differs")
        inventory.append({"source_shard": shard, "rows": receipt["positions"],
                          "source_sha256": receipt["source_sha256"]})
    require(bool(inventory) and sum(e["rows"] for e in inventory) == physical,
            "G10 selected physical inventory differs")
    provenance = rank["row_provenance"]
    require(provenance["join"] == "source-qualified-physical-row-and-full-history-keys-v1"
            and provenance["policy_observation"] == "phase0"
            and provenance["value_observation"] == "latest-phase"
            and provenance["raw_shards_read_once"] == len(inventory),
            "G10 inherited join differs")
    selection = summary.get("source_selection")
    require(same(rank.get("source_selection"), selection), "G10 rank selection differs")
    if selection is None:
        require(q.get("source_selection") is None, "G10 prefix selection differs")
    else:
        require(all(same(q["source_selection"].get(k), v) for k, v in selection.items()),
                "G10 qualification selection differs")
        require(same(selection["shards"], inventory), "G10 selected shards differ")
        selected_path = Path(selection["path"])
        require(sha(selection["sha256"]) and file_sha256(selected_path) == selection["sha256"],
                "G10 selection pin differs")
        selected = json.loads(selected_path.read_text())
        require(selected["source_dir"] == str(raw_dir)
                and selected["source_config_sha256"] == rank["raw_config_sha256"]
                and same(selected["shards"], inventory)
                and all(item["source_manifest_sha256"] == selected["source_manifest_sha256"]
                        for item in verified.values()), "G10 selection source binding differs")
    survivors = q["per_raw_shard_survivors"]
    require(set(survivors) == {e["source_shard"] for e in inventory}
            and all(type(survivors[e["source_shard"]]) is int
                    and 0 <= survivors[e["source_shard"]] <= e["rows"] for e in inventory)
            and sum(survivors.values()) == rows, "G10 survivor inventory differs")
    missing = realized["rows_dropped_no_result"]
    excluded = realized.get("rows_dropped_policy_support", 0)
    require(type(missing) is int and type(excluded) is int and min(missing, excluded) >= 0
            and physical == rows + missing + excluded
            and rank["rows_dropped_no_result"] == missing
            and rank.get("rows_dropped_policy_support", 0) == excluded
            and len(realized.get("policy_support_exclusions", [])) == excluded,
            "G10 exclusion accounting differs")
    # The original prefix receipt predates these explicit complement counters.
    # Preserve that historical scope; do not invent a stronger retrospective proof.
    for key, expected in [("independent_rank_missing_result_rows", missing),
                          ("verified_support_exclusion_rows", excluded),
                          ("omitted_rows", missing + excluded)]:
        require((selection is None and key not in q) or same(q.get(key), expected),
                "G10 qualified complement differs")
    specs = summary["shards"]
    require(summary["row_provenance"]["path_in_shard"] == "row_provenance.npz"
            and all(s["row_provenance"]["path"] == "row_provenance.npz"
                    and s["row_provenance"]["rows"] == s["rows"]
                    and sha(s["row_provenance"]["sha256"]) for s in specs)
            and sha(q["source_qualified_input_sequence_sha256"]),
            "G10 row provenance is missing")
    require(sha(q["unchanged_derived_storage_identity"])
            and storage_identity(source) == q["unchanged_derived_storage_identity"],
            "G10 qualified derived storage changed")
    return {
        "profile": "frozen-g10-common-derived-v1",
        "qualification": {"path": str(path.resolve()), "sha256": expected_sha},
        "source_dir": str(source), "raw_source_dir": str(raw_dir),
        "source_config_sha256": rank["raw_config_sha256"],
        "summary_pins": pins,
        "selected_raw_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        "source_selection": selection,
        "source_qualified_input_sequence_sha256": q["source_qualified_input_sequence_sha256"],
        "derived_storage_identity": q["unchanged_derived_storage_identity"],
    }
