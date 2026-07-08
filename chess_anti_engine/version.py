from __future__ import annotations

from importlib import metadata

# Bump this when you make backward-incompatible changes to the server<->worker protocol
# (manifest schema, upload payloads, etc.).
# 2: server-doled blind-spot FEN seeding — the manifest carries a per-request
#    ``dole_fen_seeds`` flag + ``opening_fen_dole_per_iter`` reco key that a worker
#    must act on. A pre-dole worker ignores both, so if it won the single
#    per-iteration dole claim it would silently drop the seed batch; the exact
#    protocol match (_check_worker_compat) now 426s such workers so they update
#    or stop before they can poll.
PROTOCOL_VERSION = 2

PACKAGE_NAME = "chess-anti-engine"


def package_version() -> str:
    try:
        return str(metadata.version(PACKAGE_NAME))
    except Exception:
        return "0.0.0"


PACKAGE_VERSION = package_version()
