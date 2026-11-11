"""Opt-in fresh RL donor loading. No Torch import or device work at import time."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

MODE = "fresh_rl_complete_v1"
KEYS = ("bootstrap_handoff_mode", "bootstrap_checkpoint_sha256", "bootstrap_model_contract")


def canonical(value: Any) -> str:
    """JSON canonicalization also reconciles tuple/list manifest representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_config(config: dict[str, Any]) -> bool:
    from chess_anti_engine.tune.startup_intent import validate_declared
    validate_declared(config)
    mode = config.get("bootstrap_handoff_mode", "legacy")
    if mode not in ("legacy", MODE):
        raise ValueError("unsupported bootstrap_handoff_mode")
    if mode == "legacy":
        if any(config.get(key) is not None for key in KEYS[1:]):
            raise ValueError("strict donor identity fields require fresh_rl_complete_v1")
        return False
    if config.get("mode") == "salvage":
        raise ValueError("fresh RL conflicts with salvage driver mode")
    if not isinstance(config.get("bootstrap_checkpoint"), str) or not config["bootstrap_checkpoint"].strip():
        raise ValueError("fresh RL requires an explicit donor path")
    digest = config.get("bootstrap_checkpoint_sha256")
    if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
        raise ValueError("fresh RL requires an exact lowercase SHA256")
    contract = config.get("bootstrap_model_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("fresh RL requires a complete architecture/encoding contract")
    canonical(contract)
    for key in ("bootstrap_zero_policy_heads", "bootstrap_reinit_volatility_heads"):
        if config.get(key, False) is not False:
            raise ValueError(f"{key} must be false in fresh RL")
    if config.get("salvage_seed_pool_dir"):
        raise ValueError("fresh RL conflicts with seed warmstart")
    if config.get("resume", False) is not False:
        raise ValueError("fresh RL conflicts with automatic resume")
    return True


def validate_startup(config: dict[str, Any], checkpoint: Any) -> bool:
    active = validate_config(config)
    if active and checkpoint is not None:
        raise ValueError("fresh RL conflicts with an existing trial checkpoint")
    return active


def validate_architecture(contract: dict[str, Any], donor_arch: Any, recipient_arch: dict[str, Any]) -> None:
    if not isinstance(donor_arch, dict) or canonical(donor_arch) != canonical(contract):
        raise ValueError("donor architecture/encoding differs from pinned contract")
    if canonical(recipient_arch) != canonical(contract):
        raise ValueError("recipient architecture/encoding differs from pinned contract")


def normalize_state(state: Any, model_state: Mapping[str, Any]) -> dict[str, Any]:
    """Only outer compile wrappers may differ; no QKV, plane or head migrations."""
    if not isinstance(state, Mapping) or not state or not all(isinstance(k, str) for k in state):
        raise ValueError("missing donor tensor mapping")
    def strip(mapping: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
        out = {}
        for key, value in mapping.items():
            name = key.removeprefix("_orig_mod.")
            if name in out:
                raise ValueError("ambiguous compile-prefix tensor identity")
            out[name] = (key, value)
        return out
    donor = strip(state)
    target = strip(model_state)
    if donor.keys() != target.keys():
        raise ValueError("donor tensor keys differ; migrations are forbidden")
    result = {}
    for key, (target_key, tensor) in target.items():
        value = donor[key][1]
        if value.shape != tensor.shape or value.dtype != tensor.dtype:
            raise ValueError(f"donor tensor shape/dtype differs: {key}")
        result[target_key] = value
    return result


def load_fresh_donor(
    *, config: dict[str, Any], trainer: Any, checkpoint: Any, restore: Any,
    torch_module: Any, schema_version: int,
) -> dict[str, Any]:
    """Fail before publication. A failed partially mutated Trainer must be discarded."""
    from chess_anti_engine.tune.startup_intent import enforce_startup
    if enforce_startup(config, checkpoint) != "fresh_bootstrap":
        raise ValueError("strict donor load requires fresh startup dispatch")
    if not validate_startup(config, checkpoint):
        raise ValueError("strict donor loader requires explicit opt-in")
    if restore.seed_warmstart_used:
        raise ValueError("fresh RL conflicts with restored seed state")
    recipient_arch = {"_schema_version": schema_version, **asdict(trainer._model_config)}
    path = Path(config["bootstrap_checkpoint"])
    # Nonblocking open lets fstat reject a FIFO without waiting for a writer.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("donor must be a regular file")
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != config["bootstrap_checkpoint_sha256"]:
            raise ValueError("donor SHA256 mismatch")
        stream.seek(0)
        payload = torch_module.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("donor changed during authenticated load")
    if not isinstance(payload, dict):
        raise ValueError("donor requires a checkpoint mapping")
    validate_architecture(config["bootstrap_model_contract"], payload.get("arch"), recipient_arch)
    state = payload.get("model")
    if state is None:
        state = payload.get("model_state_dict")
    if isinstance(state, Mapping):
        for value in state.values():
            if not isinstance(value, torch_module.Tensor) or not bool(torch_module.isfinite(value).all()):
                raise ValueError("donor contains a non-tensor or nonfinite tensor")
    state = normalize_state(state, trainer.model.state_dict())
    # Shared Parameters must not accept conflicting aliases with last-write-wins behavior.
    aliases = {}
    for name, parameter in trainer.model.named_parameters(remove_duplicate=False):
        previous = aliases.setdefault(id(parameter), name)
        if not torch_module.equal(state[previous], state[name]):
            raise ValueError("donor shared-parameter aliases disagree")
    trainer.model.load_state_dict(state, strict=True)
    trainer.reset_optimizer_reference_weights()
    trainer._init_swa()
    provenance = {"mode": MODE, "sha256": digest, "architecture": config["bootstrap_model_contract"]}
    trainer.fresh_rl_donor_provenance = provenance
    return provenance
