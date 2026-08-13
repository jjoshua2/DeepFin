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
# 3: `search_wdl_draw_mode` reco key — it selects how the stored `search_wdl`
#    TARGET COLUMN is constructed. A pre-#409 worker drops the unknown key, is
#    not restart-keyed by it (its own _RECO_RESTART_KEYS predates it), and keeps
#    uploading `net_raw` rows into the same replay window as the new workers'
#    `parametric_q` rows — with NO column distinguishing the two, so the mixture
#    is unrecoverable after the fact and the readout is silently confounded.
#    ⚑ min_worker_version cannot catch this: it is PACKAGE_VERSION, which an
#    in-tree `git pull` leaves identical. Same shape as the bump to 2 above, and
#    the same fix — _check_worker_compat 426s the stale worker so it updates or
#    stops before it can poll. Bumped even though production workers are
#    same-machine and restart with the trainer: that is a property of today's
#    deployment, not of the protocol, and it is the remote-worker case this
#    field exists for.
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
