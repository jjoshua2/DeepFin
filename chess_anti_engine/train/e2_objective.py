"""Immutable full objective contract for E2 paired validation; no optimizer loop.

Includes every current production compute_loss default, not only changed knobs.
Non-scalar optional policies are unsupported and fail closed. Model/optimizer/
batch-trace digests are pinned declarations, not proof those executors loaded them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
from typing import Any

from chess_anti_engine.train.e2_batch_adapter import compute_pair_losses
from chess_anti_engine.train.losses import compute_loss

FIXED = {
    "sf_wdl_frac": 0.0,
    "use_adjusted_wdl_target": False,
    "wdl_terminal_outcome_plies": 0,
    "sf_search_dampen_sf_low": 0.0,
    "sf_search_dampen_sf_high": 0.0,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FrozenObjective:
    payload_json: str
    sha256: str


def freeze_objective(
    overrides: dict[str, Any],
    *,
    donor_sha256: str,
    architecture_sha256: str,
    optimizer_plan_sha256: str,
    batch_trace_sha256: str,
) -> FrozenObjective:
    defaults = {
        name: p.default
        for name, p in inspect.signature(compute_loss).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY and name != "search_wdl_frac"
    }
    if set(overrides) - set(defaults) or "search_wdl_frac" in overrides:
        raise ValueError("unknown or arm-specific objective override")
    if not {"w_wdl", "w_categorical"} <= set(overrides):
        raise ValueError("explicit positive WDL and categorical weights required")
    effective = defaults | overrides
    for name, value in effective.items():
        if value is not None and type(value) not in (bool, int, float, str):
            raise ValueError("non-scalar objective unsupported: " + name)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("nonfinite objective")
        if (
            name.startswith("w_")
            and value is not None
            and (type(value) not in (int, float) or value < 0)
        ):
            raise ValueError("nonnegative numeric objective weight required")
    for key in ("w_wdl", "w_categorical"):
        if type(effective[key]) not in (int, float) or effective[key] <= 0:
            raise ValueError("positive head weight required")
    if any(effective[key] != value for key, value in FIXED.items()):
        raise ValueError("E2 mixture controls changed")
    # Resolve the only scalar auxiliary alias, so its realized weight is explicit.
    if effective["w_sf_volatility"] is None:
        effective["w_sf_volatility"] = effective["w_volatility"]
    pins: dict[str, Any] = {
        "donor": donor_sha256,
        "architecture": architecture_sha256,
        "optimizer_plan": optimizer_plan_sha256,
        "batch_trace": batch_trace_sha256,
    }
    if any(
        not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
        for v in pins.values()
    ):
        raise ValueError("exact identity pins required")
    payload = {
        "schema": 1,
        "effective_shared_kwargs": effective,
        "pins": pins,
        "steps_per_arm": 64,
        "batch_size": 256,
        "fresh_optimizer_each_arm": True,
        "augmentation": "none",
        "arm_search_fractions": [1.0, 0.8],
        "semantics": "pinned declarations; actual executor identities remain separate gates",
    }
    raw = canonical(payload)
    return FrozenObjective(raw, hashlib.sha256(raw.encode()).hexdigest())


def objective_losses(
    contract: FrozenObjective, expected_sha256: str, pair, outputs_a, outputs_b
):
    if (
        hashlib.sha256(contract.payload_json.encode()).hexdigest() != contract.sha256
        or contract.sha256 != expected_sha256
    ):
        raise ValueError("objective contract identity mismatch")
    payload = json.loads(contract.payload_json)
    # Reconstruct against actual current signature/defaults and require exact bytes.
    pins = payload["pins"]
    verified = freeze_objective(
        payload["effective_shared_kwargs"],
        donor_sha256=pins["donor"],
        architecture_sha256=pins["architecture"],
        optimizer_plan_sha256=pins["optimizer_plan"],
        batch_trace_sha256=pins["batch_trace"],
    )
    if verified != contract:
        raise ValueError("objective structure mismatch")
    shared = {
        k: v for k, v in payload["effective_shared_kwargs"].items() if k not in FIXED
    }
    return compute_pair_losses(pair, outputs_a, outputs_b, shared_kwargs=shared)
