"""Configuration checks for the opt-in local generation/learning phase barrier."""
from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

STOP_BEFORE_TRAINING_KEY = "distributed_stop_selfplay_before_training"


def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def validate_sequential_selfplay_config(config: Mapping[str, Any]) -> bool:
    """Only owned local producers can be covered by process-exit verification."""
    enabled = config.get(STOP_BEFORE_TRAINING_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{STOP_BEFORE_TRAINING_KEY} must be a boolean")
    if not enabled:
        return False
    for key in ("distributed_inference_shared_broker", "distributed_async_test_eval"):
        if config.get(key, False) is not False:
            raise ValueError(f"{STOP_BEFORE_TRAINING_KEY} requires {key}=false")
    host = str(config.get("distributed_server_host", "127.0.0.1") or "127.0.0.1")
    if not _is_loopback(host):
        raise ValueError(f"{STOP_BEFORE_TRAINING_KEY} requires a loopback server host")
    for key in ("distributed_server_url", "distributed_server_public_url"):
        value = config.get(key)
        if value:
            parsed = urlparse(str(value))
            if parsed.scheme not in {"http", "https"} or not _is_loopback(parsed.hostname):
                raise ValueError(f"{STOP_BEFORE_TRAINING_KEY} requires a loopback {key}")
    if int(config.get("distributed_workers_per_trial", 1)) < 1:
        raise ValueError(f"{STOP_BEFORE_TRAINING_KEY} requires owned local workers")
    return True
