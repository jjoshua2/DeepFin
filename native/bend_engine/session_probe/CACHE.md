# Opt-in legal-move reuse in persistent PUCT sessions

The diagnostic session frontend now accepts a **startup-only** setting:

```sh
DEEPFIN_SESSION_MOVE_CACHE_BITS=6 path/to/session --threads 1
```

Absent or zero keeps the original uncached `Search.prepare` path and allocates no
MoveCache arrays. One or two ASCII decimal digits are required; values 1..16 enable
`2^(bits-1)` cached positions (32 at six bits), with the existing collision-safe
MoveCache. Whitespace, signs, longer strings and out-of-range values are errors
before loading the root/attack tables. Leading zeros within two digits are accepted.
This does not change standalone UCI, the multi-root stack, worker configuration,
search policy, model settings or live training. The default remains disabled.

## Lifetime and controls

One cache belongs to the native connection, not to each newly reset PUCT tree.
It survives completed searches, cancellation/bad evaluator replies, and accepted
or rejected root advances. Those operations do not change the legality of an
already-stored board's moves. Closing the connection drops the owner. There is no
eviction: a full-cache miss generates normally; a cached empty list is a hit.

At the existing idle `ready` boundary, two commands are available:

```
cache
clear_cache
```

Each returns `cache BITS HITS FILLS BYPASSES` followed by the ordinary `ready`.
`clear_cache` replaces the cache with an empty one of the same capacity and zeroes
these counters; it does **not** reset the root or epoch. While disabled, either
command reports `cache 0 0 0 0`. Extra operands are errors. Neither command is
accepted while the process waits for an evaluator reply; the existing synchronous
transport fails closed on a wrong reply record. These are not UCI commands.

Counts describe legal-list lookups from unevaluated search leaves since the last
clear, not simulations, neural calls, or every legal-generation call in the
process. Already-terminal/depth-cutoff nodes use the existing stored node value
without another legal lookup. Root-advance validation intentionally keeps its
original direct generation and is not counted. A leaf can fill the legal cache
before its evaluator reply fails or its tree expansion runs out of space. No
counter claims that such a search simulation completed. The existing 1,024-command
and 256-simulation bounds keep these U32 counters from wrapping.

Retained ordered move lists are the only reusable payload. Search nodes, visits,
values, replies, parent paths and optional claim actions are freshly processed by
the existing search/host code. A hit never stands in for a neural result or draw
adjudication. Root history still belongs to the host; the structural cache does
not store or reconstruct it. All attack tables in one connection are unchanged.

## Implementation boundary

`CachedSearch.bend` owns the optional cache and wraps the existing selector and
`Search.prepare_moves` ticket/terminal/capacity path. Disabled operation delegates
to `Search.prepare` directly. Enabled operation substitutes only the legal list
for an unevaluated selected leaf. `Search.bend`, MoveCache, board identity, move
application, reply validation and the independent tree reference are unchanged.
The cache is carried alongside `Live` while IO suspends for each reply.

This is an integration into the real **diagnostic PUCT session frontend**, not
production Gumbel/self-play adoption. The startup setting is deliberately not
propagated to a live YAML or advertised as a throughput improvement. The retained
payload and clear operation have memory costs; capacity is not a byte budget or
a bound on transient allocator RSS. No new timing panel is run here.

## Qualification

```sh
python -m pytest tests/test_bend_session_move_cache.py
python -m native.bend_engine.session_probe.cache_probe \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --baseline-root /path/to/e58c821d-checkout --bun bun --cc clang-18 \
  --output artifacts/NEW-session-cache --python-chess
```

The baseline frontend is hash-pinned to the previous published source. Both
ordinary and UBSan builds compare complete bidirectional search transcripts
(actions in order, evaluator exchanges, paths, all node fields and best moves)
with absent/zero settings, a one-entry saturated cache, and a 32-entry cache.
The initial disabled search also matches the separately built parent frontend.
Existing independent CBoard/PUCT checks validate every snapshot, not just the
reported best move. Python-chess additionally checks real draw and optional-claim
history suites with caching enabled. Small host-reply tests verify that a warm
legal hit still accepts a new draw or claim response rather than reusing a prior
value; they are protocol controls, not claims that the starting board is drawn.

Repeated searches, clear/retry, a reversible knight cycle, twelve evaluator error
paths, capacity, depth cutoff, changed policy, castling, promotion, en passant and
terminal roots are covered. All admitted settings have startup/clear smokes;
invalid settings and malformed/in-flight clear requests are rejected. A separate
native mutant clears the cache before every search; it must still pass chess
semantics but fail the persistent-reuse check, demonstrating the option is used.
The compiler's known foreign-IO report is checked exactly, not suppressed.

Raw session transcripts, per-mode results, compiler/command outcomes and source
hashes are retained. Use a fresh output directory. Local checks may omit
`--python-chess` when it is unavailable; that does not count as the full history
qualification. CI uses the locked CPU environment and includes that flag.
Self-review only; no independent review, formal proof, deployment or measured
engine speedup is claimed.

## Completed hosted qualification

Run https://github.com/jjoshua2/DeepFin/actions/runs/36249315140 completed all preceding validation stages: 139 focused Python tests without skips, explicit static checks, actual generic/UBSan sessions with independent CBoard and Python-chess checking, existing automatic-draw and optional-claim suites with caching enabled, the reset-each-epoch negative control, and whole-repository Ruff/Basedpyright/Vulture. Exact source identities and counts are in cache-validation.json; raw transcripts/reports are in the session-cache-qualification artifact. This is not a speed result or a production enablement. The source manifest identifies this document before its appended readout. Self-review only.
