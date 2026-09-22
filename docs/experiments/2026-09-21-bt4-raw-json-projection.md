# Raw BT4 JSON input projection (2026-09-21)

## Preregistration

Compare the main `7939609c903f2a59a57bc52f282fc39ad60182fa` stdlib reader
with a raw-BT4-only projected reader. Preserve the eleven consumed fields, including
verification's `worker_id`, with no coercion or invented defaults. Leave general
Stockfish target parsing, validation, model, feed, and output handling unchanged.

Use the existing complete 8,236-row `run06_g10/w00-00000.jsonl.zst` shard
(SHA256 `3c93d2d127bc6d11db9eb31425e0c795c96b417106d737632ef4b689569ae17d`).
Compare every consumed value and ordered encoded planes, input keys, game/ply IDs,
board legal moves, and canonical LC0 feeds. Exercise both labeling and verification
through the projected reader in deterministic tests. Test malformed skipped JSON,
missing/null/type preservation, duplicate keys, nonfinite numbers, giant integers,
Unicode, and both codecs against the stdlib reference.

Success requires exact parity and lower full-shard reader-plus-encoding wall time;
any parity or acceptance-contract mismatch blocks publication until corrected.
Alternate baseline/candidate order for two paired CPU-only runs, cores 18 and 19,
two library threads, nice 19, batch 128, total benchmark cap ten minutes. Use banked
inputs without inference or changes to the active runtime. This measures CPU input
preparation, not GPU producer throughput or full-corpus end-to-end speed.

## Readout

The final eleven-field reader preserves `worker_id` for optional verification
identity receipts as well as the ten labeling fields. `Any`/`UNSET` preserve types
and distinguish absent fields from null. JSON rejected by the accelerator falls
back to stdlib at the original iterator stack depth. Conservative opening-bracket
and long-digit guards preserve reference recursion and integer-limit failures,
including unused fields. Missing optional `msgspec` uses stdlib throughout;
`msgspec>=0.20.0` is now a direct ONNX-extra dependency (lock: 0.21.1).

All 8,236 consumed-field projections match the original complete JSON objects.
Every ordered plane, input-key, game-ID, ply, worker-ID, legal-move, and canonical
LC0-feed hash matches across all four full-shard runs with locked msgspec 0.21.1:

| Order | Reader | Whole CPU preparation (s) | Reader stage (s) |
| --- | --- | ---: | ---: |
| 1 | reference | 8.537776 | 4.714288 |
| 2 | projected | 5.062255 | 1.457700 |
| 3 | projected | 5.256441 | 1.497505 |
| 4 | reference | 8.742240 | 4.800000 |

The paired aggregate is **1.67x faster for whole CPU preparation** and
3.22x for the reader stage. Whole preparation includes encoding, canonical
conversion, identity hashing and legal-move checks. The final guards' overhead is
included. This warm-cache, one-shard component screen establishes no GPU producer
or 500M end-to-end throughput improvement. No runtime was adopted or GPU used.

The same final code with ambient msgspec 0.20.0 also passed full-shard parity
(reference 7.929/7.779s, candidate 4.853/5.015s). The locked-version benchmark took
31.10s; final-code ambient qualification took 28.01s. Earlier prototype receipts
are retained separately: the 2,048-row ten-field parser-only 33x result omitted
`worker_id` and compatibility guards and is **not the qualified speedup**. A
pre-recursion-guard full-shard run is likewise superseded.

Validation: 62 raw-sidecar and raw-WDL tests passed against locked msgspec 0.21.1
in a disposable `/tmp` install; 38 focused tests passed with ambient 0.20.0.
Tests cover both codecs, field types/null/absence, duplicate keys, nonfinite and
large numbers, malformed skipped values, nesting fallback, and actual label/verify
reader calls with identity and canonical-feed receipts. No model/feed code changed.

Artifact root: `/home/josh/chess-artifacts/operations/bt4-raw-json-projection-20260921`. `complete.json` records hashes of the exact benchmark,
implementation, patch, and result files; `raw-projection-full-shard-20260921.json`
contains all final measurements and ordered hashes. Source corpus hash is pinned
above. The original general reader is unchanged from the recorded main base.

Independent reviewer `factorial_receipt_review` approved the code, benchmark and
readout. It independently passed 38 tests, 128 actual gzip/zstd iterator cases at
recursion limits 100/1000, optional-dependency absence, and 1,018 JSON value cases.
The original deep-nesting acceptance mismatch was corrected before these final
qualifications. Its complete receipt is banked as
`raw-bt4-projected-json-independent-review-20260921.json`.

Whole-repository `scripts/lint.sh` ran with the existing host Python paths: Ruff
and Vulture passed; the 14 type diagnostics match unchanged main exactly. The
comparison against both baseline logs is banked in
`raw-projection-lint-baseline-comparison.json`. A first sandboxed lint attempt
could not discover host dependencies and is preserved separately. Focused type
checking passed with zero diagnostics; `uv lock --check --offline` passed.

[Compact qualification receipt](evidence/2026-09-21-bt4-raw-json-projection.json)
contains the paired measurements, ordered hashes and supporting artifact hashes.


