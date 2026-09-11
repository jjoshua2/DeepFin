#!/usr/bin/env python3
"""Audit how saved SF search should arbitrate BT4/Ceres bootstrap teachers.

The tool is CPU-only and performs no teacher inference or new Stockfish search. It
reuses the provenance-qualified G10/raw-BT4 join from ``tactical300_calibration``
and can optionally consume exact row-aligned completed Ceres sidecars. The audit
measures policy regret on the actually rescored d10/d12 roster, SF pairwise-ranking
reliability, simple active-search routing rules, teacher complementarity, and value
calibration where native BT4 WDL plus both Ceres value heads are available.

Later G10 searches are narrowed and adaptively selected. Metrics therefore retain
coverage/adjudicability explicitly; omitted moves never receive invented scores.
This produces diagnostics and a bounded Ceres-labeling selection, not training data,
a training admission, or a playing-strength claim.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.stockfish import wdl as sf_wdl
from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import adaptive_sf_value as adaptive
from scripts import bt4_policy_mix as bt4_mix
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import ceres_derived_sidecar as ceres
from scripts import ceres_target_mix as ceres_policy
from scripts import ceres_value_mix as ceres_value
from scripts import corpus_row_provenance as provenance
from scripts import derive_corpus_targets as derive
from scripts import tactical300_calibration as tactical
from scripts.bt4_policy_dump import file_sha256

SUMMARY = "teacher_adjudication_audit.json"
SELECTION = "teacher_adjudication_ceres_selection.json"
SCHEMA = 1
BT4_TEMPERATURE = 0.5
CERES_POLICY_TEMPERATURE = 0.5
RANK_GAPS = (100.0, 300.0, 500.0, 1000.0)
SELECTION_QUOTA = 1024
SELECTION_STRATA = (
    "deeper_reversal",
    "d9_bt4_large_conflict",
    "d9_bt4_other_conflict",
    "agreement_control",
)


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _legal_map(board: chess.Board) -> tuple[dict[str, int], np.ndarray]:
    mapping: dict[str, int] = {}
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        index = compact_index_for_move(board, move)
        require(index not in mapping.values(), "duplicate compact legal index")
        mapping[move.uci()] = index
        legal[index] = True
    require(bool(mapping), "terminal row is outside audit scope")
    return mapping, legal


def _normalize(policy: np.ndarray, legal: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(policy, dtype=np.float64)
    require(values.shape == (COMPACT_POLICY_SIZE,), f"{name} policy width differs")
    require(
        bool(np.isfinite(values).all()) and bool(np.all(values >= 0)),
        f"invalid {name} policy",
    )
    require(not bool(np.any(values[~legal] != 0)), f"{name} policy has illegal mass")
    total = float(values[legal].sum())
    require(math.isfinite(total) and total > 0, f"{name} policy has no legal mass")
    return values / total


def bt4_policy(board: chess.Board, raw_policy: np.ndarray) -> np.ndarray:
    """Reconstruct B100's one-node BT4 T=0.5 target in float64."""
    _mapping, legal = _legal_map(board)
    values = np.asarray(raw_policy, dtype=np.float32)
    require(values.shape == (COMPACT_POLICY_SIZE,), "raw BT4 policy width differs")
    result = bt4_mix._tempered_bt4_policy(
        values[None, :], legal[None, :], temperature=BT4_TEMPERATURE
    )[0]
    return _normalize(result, legal, name="BT4 T0.5")


def dense_ceres_policy(bank: Any, row: int, legal: np.ndarray) -> np.ndarray:
    """Expand compact Ceres logits and softmax at the registered policy T=0.5."""
    offsets = np.asarray(bank["legal_offsets"][row : row + 2], dtype=np.int64)
    lo, hi = int(offsets[0]), int(offsets[1])
    indices = np.asarray(bank["legal_indices"][lo:hi], dtype=np.int64)
    logits = np.asarray(bank["policy_logits"][lo:hi], dtype=np.float64)
    require(
        len(indices) == int(np.count_nonzero(legal))
        and bool(np.array_equal(np.sort(indices), np.flatnonzero(legal)))
        and bool(np.isfinite(logits).all()),
        "Ceres compact legal roster differs",
    )
    dense = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float64)
    scaled = (logits - float(np.max(logits))) / CERES_POLICY_TEMPERATURE
    exp = np.exp(scaled)
    dense[indices] = exp / float(exp.sum())
    return _normalize(dense, legal, name="Ceres T0.5")


def entropy(policy: np.ndarray) -> float:
    positive = np.asarray(policy, dtype=np.float64)
    positive = positive[positive > 0]
    return float(-np.sum(positive * np.log(positive)))


def top_set(policy: np.ndarray, mapping: dict[str, int]) -> set[str]:
    best = max(float(policy[index]) for index in mapping.values())
    return {move for move, index in mapping.items() if float(policy[index]) == best}


def js_divergence(left: np.ndarray, right: np.ndarray, legal: np.ndarray) -> float:
    p = _normalize(left, legal, name="left")
    q = _normalize(right, legal, name="right")
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def arithmetic_mix(left: np.ndarray, right: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return _normalize(0.5 * left + 0.5 * right, legal, name="arithmetic mix")


def geometric_mix(left: np.ndarray, right: np.ndarray, legal: np.ndarray) -> np.ndarray | None:
    """Equal-weight log-opinion pool without reviving a teacher's zero support."""
    p = _normalize(left, legal, name="geometric left")
    q = _normalize(right, legal, name="geometric right")
    support = legal & (p > 0) & (q > 0)
    if not bool(support.any()):
        return None
    result = np.zeros_like(p)
    result[support] = np.sqrt(p[support] * q[support])
    return _normalize(result, legal, name="geometric mix")


def conditional_policy_metrics(
    board: chess.Board,
    policy: np.ndarray,
    final_scores: dict[str, float] | None,
) -> dict[str, float] | None:
    """Evaluate only the saved deeper roster and expose its covered policy mass."""
    if final_scores is None or not final_scores:
        return None
    values = np.asarray(list(final_scores.values()), dtype=np.float64)
    if bool(np.any(np.abs(values) > sf_wdl.SF_CP_CLAMP_CP)):
        return None
    mapping, legal = _legal_map(board)
    p = _normalize(policy, legal, name="candidate")
    require(set(final_scores) <= set(mapping), "deeper roster contains illegal move")
    scored_mass = float(sum(p[mapping[move]] for move in final_scores))
    if scored_mass <= 0:
        return {
            "scored_mass": 0.0,
            "conditional_regret_cp": math.nan,
            "conditional_best_mass": 0.0,
        }
    best = max(final_scores.values())
    regret = 0.0
    best_mass = 0.0
    for move, score in final_scores.items():
        weight = float(p[mapping[move]]) / scored_mass
        regret += weight * (best - score)
        if score == best:
            best_mass += weight
    return {
        "scored_mass": scored_mass,
        "conditional_regret_cp": float(regret),
        "conditional_best_mass": float(best_mass),
    }


def tactical300_preview(row: dict[str, Any], base: np.ndarray) -> np.ndarray:
    """Preview only the ordinary >300cp/50% transfer geometry from PR #640."""
    board = chess.Board(str(row["fen"]))
    mapping, legal = _legal_map(board)
    p = _normalize(base, legal, name="Tactical300 base").copy()
    scores = tactical._d9_scores(row, set(mapping))
    tactical._validate_score_domain(scores)
    if any(abs(score) > sf_wdl.SF_CP_CLAMP_CP for score in scores.values()):
        return p
    best_score = max(scores.values())
    winners = {move for move, score in scores.items() if score == best_score}
    lower = [score for score in scores.values() if score < best_score]
    if not lower or best_score - max(lower) <= 300.0:
        return p
    recipients = np.asarray([mapping[move] for move in winners], dtype=np.int64)
    donors = np.asarray(
        [mapping[move] for move in mapping if move not in winners], dtype=np.int64
    )
    donor_mass = float(p[donors].sum())
    transfer = 0.5 * donor_mass
    p[donors] *= 0.5
    recipient_mass = float(p[recipients].sum())
    if recipient_mass > 0:
        p[recipients] += transfer * p[recipients] / recipient_mass
    else:
        p[recipients] += transfer / len(recipients)
    return _normalize(p, legal, name="Tactical300 preview")


def _metric_cell() -> dict[str, float | int]:
    return {
        "rows": 0,
        "scored_mass_sum": 0.0,
        "regret_sum_cp": 0.0,
        "best_mass_sum": 0.0,
    }


def _add_policy_metric(
    cell: dict[str, float | int], metric: dict[str, float] | None
) -> None:
    if metric is None or not math.isfinite(metric["conditional_regret_cp"]):
        return
    cell["rows"] = int(cell["rows"]) + 1
    cell["scored_mass_sum"] = float(cell["scored_mass_sum"]) + metric["scored_mass"]
    cell["regret_sum_cp"] = float(cell["regret_sum_cp"]) + metric[
        "conditional_regret_cp"
    ]
    cell["best_mass_sum"] = float(cell["best_mass_sum"]) + metric[
        "conditional_best_mass"
    ]


def _routing_cell() -> dict[str, int]:
    return {"eligible": 0, "selected": 0, "reversals": 0, "reversals_captured": 0}


def _rank_cell() -> dict[str, int]:
    return {"constraints": 0, "confirmed": 0, "contradicted": 0, "ties": 0, "unscored": 0}


def new_aggregate() -> dict[str, Any]:
    return {
        "rows": 0,
        "ordinary_rows": 0,
        "policy": {
            "bt4": _metric_cell(),
            "tactical300_preview": _metric_cell(),
            "ceres": _metric_cell(),
            "arithmetic50": _metric_cell(),
            "geometric50": _metric_cell(),
        },
        "ranking": {str(int(gap)): _rank_cell() for gap in RANK_GAPS},
        "routing": {
            name: _routing_cell()
            for name in (
                "all",
                "bt4_disagrees_d9",
                "d9_gap_le_100",
                "bt4_top_lt_0.50",
                "conflict_or_low_margin",
                "conflict_or_low_confidence",
            )
        },
        "position_strata": {},
        "neural_pair": {
            "rows": 0,
            "top_agree": 0,
            "top_disagree": 0,
            "js_sum": 0.0,
            "bt4_entropy_sum": 0.0,
            "ceres_entropy_sum": 0.0,
            "adjudicable_disagreements": 0,
            "deeper_bt4": 0,
            "deeper_ceres": 0,
            "deeper_tie": 0,
            "deeper_third": 0,
        },
        "value": {
            name: {"rows": 0, "brier_sum": 0.0, "cross_entropy_sum": 0.0}
            for name in (
                "sf_saved",
                "bt4_native",
                "ceres_primary",
                "ceres_secondary",
                "ceres_dual",
                "registered_sf50_bt425_ceres25",
            )
        },
    }


def _position_strata(board: chess.Board, row: dict[str, Any]) -> list[str]:
    legal_count = board.legal_moves.count()
    ply = int(row["ply"])
    return [
        "in_check" if board.is_check() else "not_in_check",
        (
            "legal_le_10"
            if legal_count <= 10
            else "legal_11_30"
            if legal_count <= 30
            else "legal_gt_30"
        ),
        "ply_lt_40" if ply < 40 else "ply_40_79" if ply < 80 else "ply_ge_80",
    ]


def _update_position_strata(
    aggregate: dict[str, Any],
    strata: list[str],
    *,
    reversal: bool,
    bt4_regret: float | None,
) -> None:
    for name in strata:
        cell = aggregate["position_strata"].setdefault(
            name,
            {
                "rows": 0,
                "reversals": 0,
                "regret_rows": 0,
                "bt4_regret_sum_cp": 0.0,
            },
        )
        cell["rows"] += 1
        cell["reversals"] += int(reversal)
        if bt4_regret is not None and math.isfinite(bt4_regret):
            cell["regret_rows"] += 1
            cell["bt4_regret_sum_cp"] += bt4_regret


def _deeper_reversal(
    final_scores: dict[str, float] | None, d9_best: set[str]
) -> bool | None:
    if final_scores is None or not d9_best <= set(final_scores):
        return None
    values = np.asarray(list(final_scores.values()), dtype=np.float64)
    if bool(np.any(np.abs(values) > sf_wdl.SF_CP_CLAMP_CP)):
        return None
    best_d9 = max(final_scores[move] for move in d9_best)
    return max(final_scores.values()) > best_d9


def _update_ranking(
    aggregate: dict[str, Any],
    d9: dict[str, float],
    d9_best: set[str],
    final: dict[str, float] | None,
) -> None:
    best_score = max(d9.values())
    for threshold in RANK_GAPS:
        cell = aggregate["ranking"][str(int(threshold))]
        for move, score in d9.items():
            if move in d9_best or best_score - score <= threshold:
                continue
            cell["constraints"] += 1
            if final is None or move not in final or not d9_best <= set(final):
                cell["unscored"] += 1
                continue
            d9_final = max(final[item] for item in d9_best)
            other = final[move]
            if d9_final > other:
                cell["confirmed"] += 1
            elif other > d9_final:
                cell["contradicted"] += 1
            else:
                cell["ties"] += 1


def _update_routing(
    aggregate: dict[str, Any],
    *,
    disagreement: bool,
    gap: float | None,
    top_probability: float,
    reversal: bool | None,
) -> None:
    if reversal is None:
        return
    rules = {
        "all": True,
        "bt4_disagrees_d9": disagreement,
        "d9_gap_le_100": gap is not None and gap <= 100.0,
        "bt4_top_lt_0.50": top_probability < 0.5,
        "conflict_or_low_margin": disagreement or (gap is not None and gap <= 100.0),
        "conflict_or_low_confidence": disagreement or top_probability < 0.5,
    }
    for name, selected in rules.items():
        cell = aggregate["routing"][name]
        cell["eligible"] += 1
        cell["selected"] += int(selected)
        cell["reversals"] += int(reversal)
        cell["reversals_captured"] += int(reversal and selected)


def _softmax3(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(1, 3)
    return ceres_value.softmax(values, temperature)[0]


def _normalized_wdl(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    require(
        p.shape == (3,) and bool(np.isfinite(p).all()) and bool(np.all(p >= 0)),
        "invalid WDL",
    )
    total = float(p.sum())
    require(total > 0, "zero WDL mass")
    return p / total


def _wdl_loss(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    p = _normalized_wdl(prediction)
    t = _normalized_wdl(target)
    brier = float(np.sum((p - t) ** 2))
    cross_entropy = float(-np.sum(t * np.log(np.clip(p, 1e-12, 1.0))))
    return brier, cross_entropy


def _update_value(
    aggregate: dict[str, Any], name: str, prediction: np.ndarray, target: np.ndarray
) -> None:
    brier, ce = _wdl_loss(prediction, target)
    cell = aggregate["value"][name]
    cell["rows"] += 1
    cell["brier_sum"] += brier
    cell["cross_entropy_sum"] += ce


def _deeper_wdl(final_scores: dict[str, float] | None) -> np.ndarray | None:
    if final_scores is None or not final_scores:
        return None
    score = max(final_scores.values())
    return sf_wdl.cp_to_wdl(
        score,
        None,
        slope=float(bt4_mix.SOURCE_TARGET_CONTRACT["cp_slope"]),
        draw_width_cp=float(bt4_mix.SOURCE_TARGET_CONTRACT["cp_draw_width"]),
    ).astype(np.float64)


def _selection_stratum(
    *, disagreement: bool, gap: float | None, reversal: bool | None
) -> str:
    if reversal:
        return "deeper_reversal"
    if disagreement and gap is not None and gap > 300.0:
        return "d9_bt4_large_conflict"
    if disagreement:
        return "d9_bt4_other_conflict"
    return "agreement_control"


class Selector:
    """Deterministically retain the smallest hashes per preregistered stratum."""

    def __init__(self, quota: int = SELECTION_QUOTA) -> None:
        require(type(quota) is int and quota > 0, "selection quota must be positive")
        self.quota = quota
        self.heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
            name: [] for name in SELECTION_STRATA
        }

    def add(self, stratum: str, identity: dict[str, Any]) -> None:
        require(stratum in self.heaps, "unknown selection stratum")
        stable = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(stable.encode()).hexdigest()
        priority = int(digest, 16)
        item = {**identity, "stratum": stratum, "priority_sha256": digest}
        heap = self.heaps[stratum]
        entry = (-priority, digest, item)
        if len(heap) < self.quota:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    def selected(self) -> list[dict[str, Any]]:
        result = [entry[2] for heap in self.heaps.values() for entry in heap]
        return sorted(result, key=lambda item: (item["stratum"], item["priority_sha256"]))


class CeresManifest:
    """Optional exact row-aligned completed Ceres bank for the same derived G10 source."""

    def __init__(self, path: Path, digest: str, source: Path, source_summary_sha: str) -> None:
        require(_hex64(digest) and file_sha256(path) == digest, "Ceres manifest pin differs")
        self.path = path
        self.digest = digest
        self.data = json.loads(path.read_text())
        require(self.data.get("schema") == 1, "unsupported Ceres audit manifest")
        require(
            Path(self.data["source"]).resolve() == source
            and self.data["source_summary_sha256"] == source_summary_sha,
            "Ceres manifest source differs from G10 audit source",
        )
        entries = self.data["entries"]
        require(isinstance(entries, list), "Ceres manifest entries must be a list")
        self.entries = {entry["shard"]: entry for entry in entries}
        require(len(self.entries) == len(entries), "duplicate Ceres manifest shard")
        self.states: dict[Path, str] = {}

    def load(self, shard: str, source_group: Any, rows: int) -> Any | None:
        entry = self.entries.get(shard)
        if entry is None:
            return None
        path = Path(entry["ceres"]).resolve()
        require(
            path.is_dir() and not path.name.endswith(".writing"),
            "completed Ceres shard required",
        )
        binding = entry["ceres_binding"]
        require(
            binding["shard"] == shard and binding["rows"] == rows,
            "Ceres shard binding differs",
        )
        state = ceres.shared.storage_identity(path)
        attrs = ceres.verify_cached(path, binding)
        group: Any = zarr.open_group(str(path), mode="r")
        ceres_policy.check_ceres_alignment(source_group, group, attrs, rows)
        self.states[path] = state
        return group

    def guard(self) -> None:
        require(file_sha256(self.path) == self.digest, "Ceres manifest changed")
        require(
            all(
                ceres.shared.storage_identity(path) == state
                for path, state in self.states.items()
            ),
            "Ceres input changed during audit",
        )


def analyze_row(
    aggregate: dict[str, Any],
    selector: Selector,
    *,
    raw_row: dict[str, Any],
    raw_bt4: np.ndarray,
    ref: dict[str, Any],
    derived_shard: str,
    derived_row: int,
    derived_wdl: np.ndarray,
    raw_bt4_wdl: np.ndarray | None = None,
    ceres_bank: Any | None = None,
) -> None:
    board = chess.Board(str(raw_row["fen"]))
    mapping, legal = _legal_map(board)
    d9 = tactical._d9_scores(raw_row, set(mapping))
    tactical._validate_score_domain(d9)
    final, _depth, _reason = adaptive.select(raw_row, set(mapping))
    if final is not None:
        tactical._validate_score_domain(final)
    bt4 = bt4_policy(board, raw_bt4)
    bt4_top = top_set(bt4, mapping)
    bt4_top_probability = max(float(bt4[index]) for index in mapping.values())
    aggregate["rows"] += 1

    ordinary = not any(abs(score) > sf_wdl.SF_CP_CLAMP_CP for score in d9.values())
    if not ordinary:
        return
    aggregate["ordinary_rows"] += 1
    d9_best, gap = tactical._ordinary_d9_best(d9)
    disagreement = d9_best.isdisjoint(bt4_top)
    reversal = _deeper_reversal(final, d9_best)

    bt4_metric = conditional_policy_metrics(board, bt4, final)
    _add_policy_metric(aggregate["policy"]["bt4"], bt4_metric)
    preview = tactical300_preview(raw_row, bt4)
    _add_policy_metric(
        aggregate["policy"]["tactical300_preview"],
        conditional_policy_metrics(board, preview, final),
    )
    _update_ranking(aggregate, d9, d9_best, final)
    _update_routing(
        aggregate,
        disagreement=disagreement,
        gap=gap,
        top_probability=bt4_top_probability,
        reversal=reversal,
    )
    _update_position_strata(
        aggregate,
        _position_strata(board, raw_row),
        reversal=bool(reversal),
        bt4_regret=(
            None if bt4_metric is None else bt4_metric["conditional_regret_cp"]
        ),
    )

    identity = {
        "source_dir": str(Path(ref["source_dir"]).resolve()),
        "derived_shard": derived_shard,
        "derived_row": int(derived_row),
        "game_id": int(ref["game_id"]),
        "ply": int(ref["ply"]),
        "raw_shard": str(ref["source_shard"]),
        "physical_row": int(ref["source_row"]),
        "input_key": str(ref["input_key"]),
    }
    selector.add(
        _selection_stratum(disagreement=disagreement, gap=gap, reversal=reversal),
        identity,
    )

    if ceres_bank is None:
        return
    cpolicy = dense_ceres_policy(ceres_bank, derived_row, legal)
    ctop = top_set(cpolicy, mapping)
    pair = aggregate["neural_pair"]
    pair["rows"] += 1
    agree = not bt4_top.isdisjoint(ctop)
    pair["top_agree"] += int(agree)
    pair["top_disagree"] += int(not agree)
    pair["js_sum"] += js_divergence(bt4, cpolicy, legal)
    pair["bt4_entropy_sum"] += entropy(bt4)
    pair["ceres_entropy_sum"] += entropy(cpolicy)

    _add_policy_metric(
        aggregate["policy"]["ceres"], conditional_policy_metrics(board, cpolicy, final)
    )
    arithmetic = arithmetic_mix(bt4, cpolicy, legal)
    _add_policy_metric(
        aggregate["policy"]["arithmetic50"],
        conditional_policy_metrics(board, arithmetic, final),
    )
    geometric = geometric_mix(bt4, cpolicy, legal)
    if geometric is not None:
        _add_policy_metric(
            aggregate["policy"]["geometric50"],
            conditional_policy_metrics(board, geometric, final),
        )

    if not agree and final is not None and bt4_top <= set(final) and ctop <= set(final):
        pair["adjudicable_disagreements"] += 1
        bscore = max(final[move] for move in bt4_top)
        cscore = max(final[move] for move in ctop)
        if bscore > cscore:
            pair["deeper_bt4"] += 1
        elif cscore > bscore:
            pair["deeper_ceres"] += 1
        else:
            final_best = max(final.values())
            if bscore == final_best:
                pair["deeper_tie"] += 1
            else:
                pair["deeper_third"] += 1

    target = _deeper_wdl(final)
    if target is None or raw_bt4_wdl is None or "value2_logits" not in ceres_bank:
        return
    primary = _softmax3(np.asarray(ceres_bank["value_logits"][derived_row]), 0.55)
    secondary = _softmax3(np.asarray(ceres_bank["value2_logits"][derived_row]), 1.5)
    dual = 0.6 * primary + 0.4 * secondary
    sf_saved = _normalized_wdl(derived_wdl)
    bt4_value = _normalized_wdl(raw_bt4_wdl)
    registered = 0.5 * sf_saved + 0.25 * bt4_value + 0.25 * dual
    for name, prediction in (
        ("sf_saved", sf_saved),
        ("bt4_native", bt4_value),
        ("ceres_primary", primary),
        ("ceres_secondary", secondary),
        ("ceres_dual", dual),
        ("registered_sf50_bt425_ceres25", registered),
    ):
        _update_value(aggregate, name, prediction, target)


def finalize(aggregate: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(aggregate))
    for cell in result["policy"].values():
        rows = int(cell["rows"])
        cell["mean_scored_mass"] = cell["scored_mass_sum"] / rows if rows else None
        cell["mean_conditional_regret_cp"] = (
            cell["regret_sum_cp"] / rows if rows else None
        )
        cell["mean_conditional_best_mass"] = (
            cell["best_mass_sum"] / rows if rows else None
        )
    for cell in result["routing"].values():
        eligible = int(cell["eligible"])
        reversals = int(cell["reversals"])
        cell["search_fraction"] = cell["selected"] / eligible if eligible else None
        cell["reversal_capture_rate"] = (
            cell["reversals_captured"] / reversals if reversals else None
        )
    for cell in result["position_strata"].values():
        cell["reversal_rate"] = (
            cell["reversals"] / cell["rows"] if cell["rows"] else None
        )
        cell["mean_bt4_regret_cp"] = (
            cell["bt4_regret_sum_cp"] / cell["regret_rows"]
            if cell["regret_rows"]
            else None
        )
    pair = result["neural_pair"]
    rows = int(pair["rows"])
    pair["mean_js"] = pair["js_sum"] / rows if rows else None
    pair["mean_bt4_entropy"] = pair["bt4_entropy_sum"] / rows if rows else None
    pair["mean_ceres_entropy"] = pair["ceres_entropy_sum"] / rows if rows else None
    for cell in result["value"].values():
        rows = int(cell["rows"])
        cell["mean_brier"] = cell["brier_sum"] / rows if rows else None
        cell["mean_cross_entropy"] = (
            cell["cross_entropy_sum"] / rows if rows else None
        )
    return result


def audit(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    out: Path,
    ceres_manifest_path: Path | None = None,
    expected_ceres_manifest_sha256: str | None = None,
    start_shard: int = 0,
    max_shards: int | None = None,
    max_raw_rows: int = 100000,
    max_index_bytes: int = 8 * 1024**3,
) -> dict[str, Any]:
    require(_hex64(expected_manifest_sha256), "adapter manifest SHA256 required")
    require(
        start_shard >= 0 and (max_shards is None or max_shards > 0),
        "invalid shard slice",
    )
    manifest_path = manifest_path.resolve()
    manifest_pin = {"path": str(manifest_path), "sha256": expected_manifest_sha256}
    manifest = json.loads(adapter.pin(manifest_pin).read_text())
    require(manifest.get("schema") == 1, "unsupported adapter manifest")
    summary_path = adapter.pin(manifest["derived_summary"])
    source = summary_path.parent
    summary = json.loads(summary_path.read_text())
    require(
        summary.get("row_provenance", {}).get("path_in_shard") == provenance.FILENAME,
        "derivation has no row provenance",
    )
    paths = sorted(source.glob("shard_*.zarr"))
    written = {entry["path"]: entry for entry in summary["shards"]}
    require(
        bool(paths) and set(written) == {path.name for path in paths},
        "derived inventory differs",
    )
    stop = (
        len(paths)
        if max_shards is None
        else min(len(paths), start_shard + max_shards)
    )
    selected = paths[start_shard:stop]
    require(bool(selected), "selected shard slice is empty")

    inputs = adapter.RawInputs(manifest, max_raw_rows, max_index_bytes)
    optional_ceres: CeresManifest | None = None
    if ceres_manifest_path is not None or expected_ceres_manifest_sha256 is not None:
        if ceres_manifest_path is None or expected_ceres_manifest_sha256 is None:
            raise ValueError("Ceres manifest path and SHA256 are both required")
        optional_ceres = CeresManifest(
            ceres_manifest_path.resolve(),
            expected_ceres_manifest_sha256,
            source,
            manifest["derived_summary"]["sha256"],
        )

    out = out.resolve()
    writing = out.with_name(out.name + ".writing")
    protected = [source, manifest_path, *[Path(value) for value in inputs.sources]]
    protected += [spec.out_dir for spec, _receipts in inputs.sources.values()]
    if optional_ceres is not None:
        protected.append(optional_ceres.path)
    require(
        all(
            out != path and out not in path.parents and path not in out.parents
            for path in protected
        ),
        "output overlaps inputs",
    )
    require(not out.exists() and not writing.exists(), "new output required; no adoption")
    writing.mkdir(parents=True)
    cache_dir = writing / "._raw_identity_cache"
    inputs.cache_dir = cache_dir
    cache_dir.mkdir()

    aggregate = new_aggregate()
    selector = Selector()
    derived_states: dict[Path, str] = {}
    started = time.monotonic()
    rows_analyzed = 0
    shard_receipts: list[dict[str, Any]] = []
    try:
        for path in selected:
            derived_states[path] = adapter.storage_identity(path)
            group: Any = zarr.open_group(str(path), mode="r")
            x = np.asarray(group["x"][:])
            rows = len(x)
            require(
                rows == written[path.name]["rows"] and rows > 0,
                "derived row count differs",
            )
            stamp = dict(group.attrs).get("derive_row_provenance")
            if not isinstance(stamp, dict):
                raise ValueError("provenance attribute is not a mapping")
            require(
                stamp == written[path.name].get("row_provenance"),
                "provenance pin mismatch",
            )
            provenance_path = path / provenance.FILENAME
            require(
                file_sha256(provenance_path) == stamp["sha256"],
                "corrupted row provenance",
            )
            refs = provenance.read(provenance_path, rows=rows)
            game_ids = np.asarray(group["game_id"][:])
            ply_indices = np.asarray(group["ply_index"][:])
            legal = np.asarray(group["legal_mask"][:]) != 0
            sf_values = np.asarray(group["search_wdl"][:])
            cbank = (
                optional_ceres.load(path.name, group, rows)
                if optional_ceres is not None
                else None
            )
            grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
            seen: set[tuple[str, str, int]] = set()
            for index, ref in enumerate(refs):
                identity = (
                    str(ref["source_namespace"]),
                    str(ref["source_shard"]),
                    int(ref["source_row"]),
                )
                require(
                    identity not in seen,
                    "duplicate source-qualified derived row",
                )
                seen.add(identity)
                grouped.setdefault(
                    (str(ref["source_dir"]), str(ref["source_shard"])), []
                ).append((index, ref))

            for requests in grouped.values():
                first_ref = requests[0][1]
                raw_group, records = inputs.get(first_ref)
                offsets = {int(ref["source_row"]) for _index, ref in requests}
                raw_path = Path(str(first_ref["source_dir"])) / str(
                    first_ref["source_shard"]
                )
                source_rows = tactical._raw_rows(raw_path, offsets)
                raw_offsets = np.asarray(
                    [int(ref["source_row"]) for _index, ref in requests],
                    dtype=np.int64,
                )
                policies = np.asarray(
                    raw_group[raw.POLICY_FIELD].oindex[raw_offsets, :]
                )
                values = (
                    np.asarray(raw_group[raw.WDL_FIELD].oindex[raw_offsets, :])
                    if raw.WDL_FIELD in raw_group
                    else None
                )
                for request_index, (derived_index, ref) in enumerate(requests):
                    offset = int(ref["source_row"])
                    require(
                        0 <= offset < len(records),
                        "physical source row outside closed shard",
                    )
                    record = records[offset]
                    require(
                        ref["input_key"] == record["input_key"].tobytes().hex(),
                        "raw input key mismatch",
                    )
                    require(
                        ref["stored_input_key"]
                        == record["stored_input_key"].tobytes().hex()
                        == adapter.corpus.input_tensor_key(x[derived_index]),
                        "stored input key mismatch",
                    )
                    require(
                        all(
                            int(ref[field]) == int(record[field])
                            for field in ("worker_id", "game_id", "ply")
                        ),
                        "raw physical identity mismatch",
                    )
                    require(
                        int(game_ids[derived_index]) == int(ref["game_id"])
                        and int(ply_indices[derived_index]) == int(ref["ply"]),
                        "derived game/ply alignment mismatch",
                    )
                    raw_row = source_rows[offset]
                    derive._check_row_identity(
                        raw_row, str(ref["source_config_sha256"])
                    )
                    board = chess.Board(str(raw_row["fen"]))
                    _mapping, expected_legal = _legal_map(board)
                    require(
                        np.array_equal(expected_legal, legal[derived_index]),
                        "legal support differs",
                    )
                    analyze_row(
                        aggregate,
                        selector,
                        raw_row=raw_row,
                        raw_bt4=policies[request_index],
                        ref=ref,
                        derived_shard=path.name,
                        derived_row=derived_index,
                        derived_wdl=sf_values[derived_index],
                        raw_bt4_wdl=(
                            None if values is None else values[request_index]
                        ),
                        ceres_bank=cbank,
                    )
                    rows_analyzed += 1

            require(
                adapter.storage_identity(path) == derived_states[path],
                "derived source changed",
            )
            require(
                file_sha256(provenance_path) == stamp["sha256"],
                "row provenance changed",
            )
            shard_receipts.append(
                {
                    "path": path.name,
                    "rows": rows,
                    "row_provenance_sha256": stamp["sha256"],
                    "source_storage_identity": derived_states[path],
                    "ceres_present": cbank is not None,
                }
            )

        require(rows_analyzed == aggregate["rows"], "aggregate row count differs")
        require(
            all(
                adapter.storage_identity(path) == state
                for path, state in (inputs.stable | derived_states).items()
            ),
            "verified input changed before publication",
        )
        for item in [manifest_pin, manifest["derived_summary"], *inputs.pins]:
            adapter.pin(item)
        if optional_ceres is not None:
            optional_ceres.guard()

        selected_rows = selector.selected()
        selection_payload = {
            "schema": 1,
            "kind": "teacher-adjudication-bounded-ceres-selection",
            "quota_per_stratum": SELECTION_QUOTA,
            "strata": list(SELECTION_STRATA),
            "rows": len(selected_rows),
            "selected": selected_rows,
            "new_ceres_inference": 0,
            "status": "SELECTION_ONLY_NOT_CERES_LABELS",
        }
        (writing / SELECTION).write_text(
            json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
        )
        final = {
            "schema": SCHEMA,
            "status": "COMPLETE_DIAGNOSTIC_NOT_TRAINING_ADMISSION",
            "kind": "sf-bt4-ceres-teacher-adjudication-audit",
            "adapter_manifest": manifest_pin,
            "derived_summary": manifest["derived_summary"],
            "ceres_manifest": (
                None
                if optional_ceres is None
                else {"path": str(optional_ceres.path), "sha256": optional_ceres.digest}
            ),
            "rows": rows_analyzed,
            "shards": len(selected),
            "slice": {"start_shard": start_shard, "max_shards": max_shards},
            "new_teacher_inference": 0,
            "new_stockfish_search": 0,
            "training_admission": False,
            "playing_strength_result": False,
            "metrics": finalize(aggregate),
            "ceres_selection": {
                "path": SELECTION,
                "rows": len(selected_rows),
                "sha256": file_sha256(writing / SELECTION),
            },
            "shard_receipts": shard_receipts,
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Saved d10/d12 rosters are narrowed/adaptive and are calibration evidence, not ground truth.",
                "Policy regret is conditional on actually rescored moves and always reports covered mass.",
                "Ceres metrics exist only on exact row-aligned completed Ceres shards explicitly supplied.",
                "The bounded Ceres selection is not a qualified selected bank and launches no inference.",
            ],
        }
        (writing / SUMMARY).write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n"
        )
        inputs.cache.clear()
        shutil.rmtree(cache_dir)
        require(not out.exists(), "output appeared before publication")
        os.replace(writing, out)
        return final
    except BaseException:
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ceres-manifest", type=Path)
    parser.add_argument("--expected-ceres-manifest-sha256")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-raw-rows", type=int, default=100000)
    parser.add_argument("--max-index-bytes", type=int, default=8 * 1024**3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        out=args.out,
        ceres_manifest_path=args.ceres_manifest,
        expected_ceres_manifest_sha256=args.expected_ceres_manifest_sha256,
        start_shard=args.start_shard,
        max_shards=args.max_shards,
        max_raw_rows=args.max_raw_rows,
        max_index_bytes=args.max_index_bytes,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "shard_receipts"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
