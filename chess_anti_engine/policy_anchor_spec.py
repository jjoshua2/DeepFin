"""Strict, I/O-free frozen-reference policy identity for finite-family metadata."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

SCHEMA = "frozen_reference_policy_anchor_v1"
DIRECTION = "reference_to_student_forward_KL"
MASK = "original_network_turn_has_policy_with_explicit_legal_mask_v1"
CHECKPOINT_KEY = "frozen_reference_policy_anchor_v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_spec(value: Any, architecture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "direction", "mask", "coefficient", "reference"}:
        raise ValueError("invalid frozen policy anchor fields")
    if value["schema"] != SCHEMA or value["direction"] != DIRECTION or value["mask"] != MASK:
        raise ValueError("unsupported frozen policy anchor semantics")
    weight = value["coefficient"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("invalid frozen policy anchor coefficient")
    reference = value["reference"]
    if weight == 0:
        if reference is not None:
            raise ValueError("zero anchor requires null reference")
    else:
        if not isinstance(reference, dict) or set(reference) != {"checkpoint", "architecture", "checkpoint_step", "historical_source_revision"}:
            raise ValueError("invalid frozen policy reference")
        checkpoint = reference["checkpoint"]
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
            raise ValueError("invalid frozen policy checkpoint reference")
        if type(checkpoint["path"]) is not str or not Path(checkpoint["path"]).is_absolute() or type(checkpoint["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", checkpoint["sha256"]) is None:
            raise ValueError("invalid frozen policy checkpoint identity")
        if canonical(reference["architecture"]) != canonical(architecture):
            raise ValueError("frozen policy reference architecture differs")
        if type(reference["checkpoint_step"]) is not int or reference["checkpoint_step"] < 0:
            raise ValueError("invalid frozen policy checkpoint step")
        historical = reference["historical_source_revision"]
        if historical is not None and (type(historical) is not str or re.fullmatch("[0-9a-f]{40}", historical) is None):
            raise ValueError("invalid declared historical reference provenance")
    return json.loads(canonical(value))

