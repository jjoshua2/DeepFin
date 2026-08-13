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
# ⚑⚑ NOT BUMPED TO 3 FOR THE SEED-DOLE CLAIM CHANGE, AND THE REASON IS THAT
#    THERE IS NO SAFE ORDER IN WHICH TO DEPLOY THE BUMP.
#
#    `_check_worker_compat` requires EXACT equality (`got_p != req_p` ->
#    426), not a minimum. So the bump breaks BOTH directions of a rolling
#    deploy: a v3 worker against the not-yet-updated v2 server is refused at
#    the manifest poll, which means it never reaches the legacy
#    `dole_fen_seeds` fallback that was supposed to make worker-first safe;
#    and v2 workers against a v3 server are refused too. "Workers first" and
#    "server first" both take the fleet to zero for the length of the window.
#
#    The cutover mechanism is `min_worker_version` instead, which is already
#    published on every manifest (`distributed_runtime.py`) and IS a `>=`
#    comparison (`version_lt`), so it can exclude stale workers without
#    breaking the transition.
#
#    ⚑ THE RESIDUAL RISK, STATED RATHER THAN PAPERED OVER: a worker running
#    OLD code against a NEW server reads `dole_fen_seeds: false` forever and
#    is silently never seeded. `min_worker_version` only catches that if the
#    PACKAGE VERSION IS ACTUALLY BUMPED in the same deploy -- it is compared
#    against `PACKAGE_VERSION`, so shipping this change without a version bump
#    leaves the gate unable to fire. Bumping the package version is therefore a
#    REQUIRED deploy step for this change, not a nicety.
PROTOCOL_VERSION = 2

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
