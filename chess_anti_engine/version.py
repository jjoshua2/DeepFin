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
# 3: the seed-dole claim moved OFF the unauthenticated manifest GET onto an
#    authenticated ``POST /v1/seed_dole_claim``. The GET is now side-effect-free
#    and always answers ``dole_fen_seeds: false``; a server that supports the
#    claim advertises ``seed_dole_claim_endpoint`` + ``manifest_revision``.
#
#    ⚑ THE BUMP IS THE WHOLE SAFETY MECHANISM HERE, because the failure it
#    prevents is silent. A version-2 worker only ever reads ``dole_fen_seeds``;
#    against a version-3 server that field is permanently false, so the worker
#    would poll happily, report healthy, upload shards, and NEVER seed a single
#    blind-spot position. Nothing in the fleet's own telemetry distinguishes
#    that from a run where seeding is working. 426ing the worker converts an
#    invisible training-quality regression into a loud refusal.
#
#    ⚑ ROLLOUT ORDER IS NOT OPTIONAL: workers first, server second. A worker
#    carrying this code still honours the legacy field when the server does not
#    advertise the endpoint, so worker-first is safe; server-first is the
#    silently-unseeded fleet described above, for as long as the gap lasts.
PROTOCOL_VERSION = 3

PACKAGE_NAME = "chess-anti-engine"

# Worker-computed sha256 of the shard tarball, verified by the server against
# the digest it computes over what it RECEIVED.
#
# ⚑ Defined once and imported by BOTH sides on purpose. A header is a string
# matched by name across a process boundary, so two literals are one typo away
# from a check that silently never fires -- the failure this repo sees most:
# a value accepted and then ignored. There is no NOT-sent case to distinguish
# from a MISSPELLED case at the server, because both look like "header absent",
# and absent is the backward-compatible path.
#
# Not a PROTOCOL_VERSION bump: the server verifies the digest only when it is
# present, so old workers and new servers interoperate unchanged.
UPLOAD_CONTENT_SHA256_HEADER = "X-CAE-Content-SHA256"


def package_version() -> str:
    try:
        return str(metadata.version(PACKAGE_NAME))
    except Exception:
        return "0.0.0"


PACKAGE_VERSION = package_version()
