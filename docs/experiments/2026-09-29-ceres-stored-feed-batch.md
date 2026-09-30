# Ceres stored-feed batch conversion, September 29, 2026

The full 58,820-winner selected/dual cost pilot exposed avoidable CPU work in
its Ceres inference producer. In its first selected arm, 919 fixed-32 Ceres
calls spent 25.0816 s converting feeds out of 91.5409 s of measured call wall;
the inference component was 64.7418 s. The producer's `row_bind.feed_for_call`
ran full `bind` for every Ceres row, building an unused BT4 feed and legal-map
context and invoking the Ceres TPG encoder one row at a time. The full bind is
still required in independent target/readback validation. This record tests a
physical feed path that checks staged source hash, finite values, and exact
stored-f16 round trip, then converts a whole Ceres call in one TPG batch.

The CPU probe used the frozen selected roster, staged rows, and already saved
S1 feeds from the ongoing cost pilot. Its first eight Ceres calls were 256
distinct rows. All eight new feeds equaled the saved physical feed bytes and
their recorded SHA-256 values. With rows preloaded and five warmed conversion
rounds, median time for those eight calls was 0.222191 s with full bind,
0.096588 s with Ceres-only per-row conversion, and 0.049159 s with Ceres-only
batch-32 conversion. The last is a 77.875% reduction in this narrow CPU
conversion stage. These timings exclude stage reads and neural inference.
The probe read 11,468,800 staged bytes and 2,244,608 saved feed bytes.

A second, deterministic stratified probe read 512 **additional** staged rows
(22,937,600 bytes) and the selected metadata, with no model or GPU access and
no full-stage hash. The sample contained 252 BT4-v9 and 260 Ceres-v8 source
rows, 263 black and 249 white roots, nine en-passant positions, 82 rows with
repetition, all observed castling-right counts from zero through four, and
rule50 values in zero, low, middle and high bins. The final UID component was
0–7 for nine rows, 8–15 for ten, and at least 16 for 493; true real-history
depth was not independently classified. All eight history slots and all 137
features are included in each exact feed-byte comparison. Main's tracked TPG
encoder, the frozen snapshot encoder, and the per-row full bind agreed for all
512 records. Repeat-last physical padding agreed at the actual selected-tail
shape, 209 real plus 303 padded slots for batch 512, and at 17 plus 15 for
batch 32. Main's pure TPG batch-512 conversion median was 0.039611 s versus
0.058050 s for the frozen snapshot; this timing excludes raw-row validation,
stacking and all inference.

The reusable [stored-feed builder](../../chess_anti_engine/encoding/ceres_stored_feed.py)
accepts staged little-endian float32 rows and their source SHA-256 values,
requires the qualified source's corrected legacy-history profile explicitly,
checks the exact-f16 storage domain, invokes the tracked batch TPG encoder,
and repeats the last byte record for fixed physical tails. It does not replace
full legal-context validation in the independent readback. The current frozen
cost-pilot source and running jobs were not modified. Its externally pinned
producer is outside the tracked main package, so these results do not yet
measure a live integrated throughput gain or change corpus/500M credit.

Compact evidence: [first eight calls](evidence/2026-09-29-ceres-feed-first8.json)
and [stratified 512 rows](evidence/2026-09-29-ceres-feed-stratified512.json).
The original CPU probe files were
`/tmp/ceres-feed-conversion-first8-cpu-probe-20260929.json` (SHA-256
`df04aa79ed79e64d9ba9d5d343511177e6db0768b2407a446b1da6f6a5a0f866`)
and `/tmp/ceres-feed-stratified512-cpu-probe-20260929.json` (SHA-256
`1138876f03eebd6e94d2a4dcc4e222258f9342477a200cbae7ea05b0aaf5276b`).
