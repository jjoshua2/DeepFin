"""Explicit strict-RL startup dispatch. No I/O, Torch or Ray imports."""
from __future__ import annotations
import hashlib
import json
import uuid
from typing import Any

SCHEMA = "strict_rl_startup_contract_v1"
MANDATORY_SETTINGS = ("lr", "optimizer", "lr_schedule", "batch_size", "accum_steps")
EXACT_RESTORE_SUPPORTED = False


class StartupIntentError(ValueError):
    """Strict startup refusal; callers must not downgrade it to a warning."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def contract_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def is_strict(config: dict[str, Any]) -> bool:
    return (
        "rl_startup_intent" in config or "rl_startup_contract" in config
        or config.get("bootstrap_handoff_mode", "legacy") != "legacy"
    )


def _declared(config: dict[str, Any]) -> str:
    if not is_strict(config):
        return "legacy"
    intent = config.get("rl_startup_intent")
    if intent not in ("fresh_bootstrap", "exact_restore"):
        raise StartupIntentError("strict RL needs explicit startup intent")
    contract = config.get("rl_startup_contract")
    required = {"schema", "run_uuid", "donor_sha256", "model_contract_sha256", "settings"}
    if not isinstance(contract, dict) or set(contract) != required or contract["schema"] != SCHEMA:
        raise StartupIntentError("missing or unsupported strict startup contract")
    try:
        if str(uuid.UUID(contract["run_uuid"])) != contract["run_uuid"]:
            raise ValueError("noncanonical UUID")
        for key in ("donor_sha256", "model_contract_sha256"):
            value = contract[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid digest")
        settings = contract["settings"]
        if not isinstance(settings, dict) or not set(MANDATORY_SETTINGS).issubset(settings):
            raise ValueError("missing bound startup settings")
        if any(key.startswith("rl_startup_") or key in {"resume", "bootstrap_handoff_mode"} for key in settings):
            raise ValueError("reserved startup setting")
        canonical(contract)
        for key, expected in settings.items():
            if key not in config or canonical(config[key]) != canonical(expected):
                raise ValueError("changed or missing bound setting: " + key)
    except (TypeError, ValueError, AttributeError) as exc:
        raise StartupIntentError("invalid or conflicting strict startup contract: " + str(exc)) from exc
    if config.get("mode") == "salvage" or config.get("salvage_seed_pool_dir"):
        raise StartupIntentError("strict RL conflicts with salvage")
    if "resume" in config and type(config["resume"]) is not bool:
        raise StartupIntentError("resume must be an explicit boolean")
    if intent == "fresh_bootstrap":
        if config.get("resume", False):
            raise StartupIntentError("ambiguous fresh bootstrap plus resume")
        if config.get("bootstrap_handoff_mode") != "fresh_rl_complete_v1":
            raise StartupIntentError("fresh intent requires complete donor mode")
        if config.get("bootstrap_checkpoint_sha256") != contract["donor_sha256"]:
            raise StartupIntentError("donor identity conflicts with startup contract")
        if contract_digest(config.get("bootstrap_model_contract")) != contract["model_contract_sha256"]:
            raise StartupIntentError("model contract conflicts with startup contract")
    else:
        if config.get("bootstrap_handoff_mode", "legacy") != "legacy" or any(
            config.get(key) is not None for key in (
                "bootstrap_checkpoint", "bootstrap_checkpoint_sha256", "bootstrap_model_contract"
            )
        ):
            raise StartupIntentError("exact restore refuses stale bootstrap instructions")
    return intent


def validate_declared(config: dict[str, Any]) -> str:
    intent = _declared(config)
    if intent == "exact_restore":
        raise StartupIntentError("strict exact RL restoration is not qualified")
    return intent


def resolve_startup_intent(
    config: dict[str, Any], *, has_checkpoint: bool, saved_contract: dict[str, Any] | None = None,
) -> str:
    intent = _declared(config)
    if intent == "fresh_bootstrap" and has_checkpoint:
        raise StartupIntentError("fresh bootstrap refuses an existing checkpoint")
    if intent == "exact_restore":
        if not has_checkpoint or saved_contract is None:
            raise StartupIntentError("exact restore requires checkpoint and authenticated saved contract")
        if canonical(saved_contract) != canonical(config["rl_startup_contract"]):
            raise StartupIntentError("saved startup contract conflicts")
    return intent


def enforce_startup(
    config: dict[str, Any], checkpoint: Any, *, saved_contract: dict[str, Any] | None = None,
) -> str:
    intent = resolve_startup_intent(config, has_checkpoint=checkpoint is not None, saved_contract=saved_contract)
    if intent == "exact_restore":
        # Deliberately no config flag can qualify a missing restoration implementation.
        raise StartupIntentError("strict exact RL restoration is not qualified")
    return intent


def validate_saved_overlay(saved: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Refuse silent omission, replacement or unqualified resume of strict intent."""
    if not (is_strict(saved) or is_strict(incoming)):
        return
    if "rl_startup_intent" not in incoming or "rl_startup_contract" not in incoming:
        raise StartupIntentError("strict startup intent/contract may not disappear through omission")
    if is_strict(saved) and canonical(saved.get("rl_startup_contract")) != canonical(incoming["rl_startup_contract"]):
        raise StartupIntentError("saved startup contract may not be replaced by YAML")
    if is_strict(saved) and saved.get("rl_startup_intent") != incoming["rl_startup_intent"]:
        raise StartupIntentError("fresh-to-exact transition is not qualified")
    validate_declared(dict(saved) | dict(incoming))


def refuse_strict_resume(config: dict[str, Any]) -> None:
    if is_strict(config):
        raise StartupIntentError("strict saved-state resume/filtering is not qualified")
