from __future__ import annotations

from importlib import metadata

# Bump this when you make backward-incompatible changes to the server<->worker protocol
# (manifest schema, upload payloads, etc.).
PROTOCOL_VERSION = 1

PACKAGE_NAME = "chess-anti-engine"


def package_version() -> str:
    try:
        return str(metadata.version(PACKAGE_NAME))
    except Exception:
        return "0.0.0"


PACKAGE_VERSION = package_version()
