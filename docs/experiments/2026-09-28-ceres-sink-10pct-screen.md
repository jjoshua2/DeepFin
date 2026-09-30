# Ceres v11 nested sink: independent source-only 10% screen

Question and predeclared engineering threshold: can a reviewable replacement for redundant sink work save at least 10% of the measured 1,704.563-second Ceres source stage, while retaining strict row, history, legal/feed, WDL/DTZ and publication proofs? This is a read-only receipt and source screen against the accepted run11 baseline; it has no new registered timing or statistical strength claim.

Verdict: **HOLD_NO_DEFENSIBLE_LOW_RISK_10_PERCENT_CANDIDATE**. No registered
payload, model, GPU, or Syzygy was opened for this check. It reviews saved small
receipts, source, prior independently reviewed timing, and synthetic outputs;
there is no new throughput credit or source change.

The accepted run11 child receipt (SHA-256
`d87be4aa81f1f8eefd101c1dfebb2f1ae5a3b1dee795524eca31be9a02475356`)
records 2,048 games, 257,402 published-before-dedup rows and 32 ZIP shards.
Actor wall is 1,692.399 s, source-stage wall is 1,704.563 s, and inclusive
sink/readback is 470.699 s. The last of six cumulative progress snapshots gives
459.452 s of nested per-game sink; subtraction leaves 11.247 s of final tail.
Ten percent of source-stage wall is **170.456 s**, requiring elimination of
37.10% of that nested sink bucket. These are exact receipt arithmetic, not
subcomponent timings.

The accepted v11 writer (`memory_queue.py` SHA-256
`cfdf06d3a9da20a5607b9c65f326d2e99d21a4d24edeba3b41f5ce68863e08c6`)
does not time its inner `accept`, ZIP build, hashes, or three verifier roles.
`accept` checks stored-X-to-TPG feed equality, per-row feed hashes, complete
initial history and terminal replay. `flush` writes the ZIP and fully verifies
the private `.writing` archive before rename. The ledger fully verifies the
published archive after rename. `finalize` gives the final shard an extra full
published read. Dropping a check without an independently reviewed equivalent
would change the history, legal/feed or postpublication trust contract.

The prior independently reviewed, original-bank **read-only** A/B terminal
(SHA-256 `055586a94662cac297da0e8dd1860019b0dc14e28e756a697c9177e3f6eb60dd`)
timed the first 31 shards' published full reader at 125.120–126.508 s. Even
ideal deletion is just 7.34–7.42% of source-stage wall and leaves 43.948–45.336
s to reach 10%. The test did not exercise private-to-published rename or prove
the compact substitute safe; the independent terminal review explicitly holds
production trust and actor-wall savings. Removing only the final duplicate
reader cannot save more than the whole 11.247-s final tail (0.66% of source
wall). Neither verifier edit qualifies as a low-risk 10% candidate.

The sealed 125-row synthetic sink profile (manifest SHA-256
`1aae6736d0907dea7dc3cb420e87aa67221eb172837795679c60928f4e331a3a`)
ran three legal draw-claim games through accepted `accept`, finalization and
three strict full readers. Its median feed reconstruction was 15.39 ms and
trajectory digest 9.31 ms per game. Naively multiplying by 2,048 gives 31.52
and 19.06 s; even synthetic whole `accept` plus trajectory digest scales to
157.73 s, below the 170.46-s threshold. These are **not bounds on real game
cost**: the knight-cycle ZIPs are tiny and highly compressible, and the three
archive SHA values differ across runs. The fixture proves each run's internal
strict readback, not cross-run output-byte identity. Feed and trajectory checks
also serve distinct provenance requirements.

Next bounded step, if this sink remains a priority: add timers only in a frozen
disposable writer copy around `accept` collection/feed/history replay,
trajectory digest, ZIP construction/hash, private verify, published verify,
and final tail. First require source-only synthetic parity and an independent
review of the timers; do not rewrite or defer checks. A later separately gated,
quiet-host one- or two-shard representative timing can test whether any
single **replaceable** subcomponent reaches 170.456 s when scaled by exact
row/game/shard counts. Preserve full ZIP bytes, row/history/legal/feed and
WDL/DTZ provenance, both publication trust boundaries, all-attempt strict and
physical-call readback, and zero credit until complete. No actor change or
registered-payload timing is authorized by this screen.
