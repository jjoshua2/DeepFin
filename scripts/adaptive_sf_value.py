"""Select saved G10 value observations; no inference or policy construction."""

from __future__ import annotations

import math
from typing import Any

SELECTOR = "g10-adaptive-final-or-d9-v1"
VALUE_SOURCE = "g10-adaptive-final-root-candidates-with-latest-d9-fallback-v1"


def _lines(
    block: dict[str, Any], legal: set[str], width: int, *, complete: bool
) -> dict[str, float]:
    lines = block["lines"]
    moves = [str(line[1]) for line in lines]
    ranks = [line[0] for line in lines]
    if (
        len(set(moves)) != len(moves)
        or not set(moves) <= legal
        or any(type(rank) is not int for rank in ranks)
        or ranks != list(range(1, len(lines) + 1))
        or not all(math.isfinite(float(line[2])) for line in lines)
    ):
        raise ValueError("invalid_roster")
    if complete and (block.get("complete") is not True or len(lines) != width):
        raise ValueError("incomplete_roster")
    return {str(line[1]): float(line[2]) for line in lines}


def validate_baseline(row: dict[str, Any], legal: set[str]) -> None:
    """Reject ambiguity in any block consumed by the existing composite d9 read."""
    phases = row["phases"]
    if (
        not phases
        or phases[0].get("index") != 0
        or phases[0].get("depth_requested") != 9
    ):
        raise ValueError("invalid_phase0_identity")
    if phases[0].get("searchmoves") is not None:
        raise ValueError("restricted_phase0")
    pending = set(legal)
    for index in range(len(phases) - 1, -1, -1):
        phase = phases[index]
        blocks = [b for b in phase["per_depth"] if b["depth"] == 9]
        if not blocks:
            continue
        selected = blocks[-1]
        used = pending & {str(line[1]) for line in selected["lines"]}
        if not used:
            continue
        if len(blocks) != 1:
            raise ValueError("ambiguous_baseline_depth")
        allowed = legal if index == 0 else set(phase.get("searchmoves") or [])
        if not allowed <= legal:
            raise ValueError("illegal_baseline_searchmoves")
        _lines(selected, allowed, len(legal), complete=index == 0)
        pending -= used
    if pending:
        raise ValueError("missing_baseline_moves")
    # Phase0 is also the policy support and must be complete independently of overlays.
    base = [b for b in phases[0]["per_depth"] if b["depth"] == 9]
    if (
        len(base) != 1
        or set(_lines(base[0], legal, len(legal), complete=True)) != legal
    ):
        raise ValueError("invalid_phase0_baseline")


def _requested(
    phase: dict[str, Any], index: int, depth: int, expected: list[str], legal: set[str]
) -> dict[str, float]:
    width = len(expected)
    if (
        phase.get("index") != index
        or phase.get("depth_requested") != depth
        or phase.get("width_realized") != width
        or phase.get("searchmoves") != (None if index == 0 else expected)
    ):
        raise ValueError(f"phase{index}_identity_or_ancestor_roster")
    blocks = phase["per_depth"]
    selected = [b for b in blocks if b["depth"] == depth]
    # The generator uses its deepest width-satisfying block to choose the next roster.
    # Do not pretend the requested block supplied that roster when another did.
    full = [b for b in blocks if len(b["lines"]) >= width]
    if len(selected) != 1 or not full or max(b["depth"] for b in full) != depth:
        raise ValueError(f"phase{index}_requested_block")
    values = _lines(selected[0], legal, width, complete=True)
    if set(values) != set(expected):
        raise ValueError(f"phase{index}_move_roster")
    return values


def select(
    row: dict[str, Any], legal: set[str]
) -> tuple[dict[str, float] | None, int, str]:
    """Return selected scores/depth/reason, or explicit unchanged-d9 fallback.

    Bad baseline or a different experiment's identity is fatal. Malformed later
    observations never remove a row or masquerade as a successful deeper read.
    Missing schema fields or malformed line shapes remain fatal schema corruption;
    fallback is limited to schema-valid missing/incomplete/inconsistent observations.
    """
    validate_baseline(row, legal)
    gate = row.get("staircase_gate") or {}
    if (
        gate.get("policy") != "g10"
        or gate.get("adaptive") is not True
        or gate.get("decision_depth") != 10
        or gate.get("decision_after_phase") != 1
        or gate.get("threshold_cp") != 10
        or gate.get("extend_when") != "margin_cp<=threshold_cp"
        or gate.get("metric") != "effective_cp_rank1_minus_rank2"
        or gate.get("no_margin_action") != "stop"
        or gate.get("extended_depth") != 12
        or gate.get("extended_phase") != 2
        or gate.get("extended_width") != 4
    ):
        raise ValueError("not_registered_g10_identity")
    try:
        phases = row["phases"]
        p0 = _requested(phases[0], 0, 9, sorted(legal), legal)
        if len(phases) < 2:
            raise ValueError("missing_d10")
        p1 = _requested(phases[1], 1, 10, list(p0)[:8], legal)
        if gate.get("decision_depth_observed") != 10:
            raise ValueError("decision_depth")
        scores = list(p1.values())
        if len(scores) == 1:
            extended = False
            margin = None
            reason = "fewer_than_two_moves"
        else:
            margin = scores[0] - scores[1]
            extended = margin <= 10
            reason = (
                "margin_at_or_below_threshold" if extended else "margin_above_threshold"
            )
        if (
            type(gate.get("extended")) is not bool
            or gate["extended"] != extended
            or gate.get("margin_cp") != margin
            or gate.get("reason") != reason
            or len(phases) != (3 if extended else 2)
        ):
            raise ValueError("gate_or_phase_count")
        if extended:
            final = _requested(phases[2], 2, 12, list(p1)[:4], legal)
            return final, 12, "selected_d12"
        return p1, 10, "selected_single_move_d10" if margin is None else "selected_d10"
    except ValueError as exc:
        return None, 9, f"fallback:{exc}"
