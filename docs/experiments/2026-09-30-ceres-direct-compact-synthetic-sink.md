# Ceres direct-compact sink: bounded synthetic CPU screen

**Readout, September 30, 2026.** The isolated candidate avoids materializing
transient stacked copies of the complete raw policy head and reconstructed feed
after checking each original row. The control is the run08 queue with matching
phase timers. This was a question about sink CPU cost, not a replacement for
strict source, physical-call, or saved-game qualification. The candidate has
not been integrated into the live Ceres producer.

Eight four-row synthetic completed games and one zero-accepted-row discard
were constructed once with full raw policy, feed and value heads. The same nine
objects went through two warmups and six alternating ABBA/BAAB timing blocks:
26 fresh-root arms total. Each arm published 32 accepted rows, nine game
records and 47,210 ZIP bytes. The harness reopened every output ZIP through
`verify_shard` and the publication ledger, then compared decoded array hashes,
semantic manifests, ordered game records and terminal counts across arms.
The pinned completed-game full-head digests and canonical discard-metadata
digest were unchanged before and after the run. The benchmark completed in
4.27 seconds with a 180-second outer watchdog, a 170-second child alarm,
2 GiB RSS gate and 64 MiB output-tree cap. The output used 1,398,699 bytes;
numerical peak RSS was **not recorded**.

Paired block sink-wall savings were **+3.564%, +3.643%, −5.102%, −30.192%,
+21.822%, and +0.437%**. The median was +2.000%, but sign changes and large
swings swamp that small center. Median sink wall was 63.097 ms for control
and 62.995 ms for candidate. Collection was 11.590 versus 11.364 ms; accept
was 19.830 versus 19.174 ms; flush was 43.106 versus 43.470 ms. The common
flush verification (~27 ms) and ZIP writing (~11 ms) dominated the tiny
fixture. **This screen gives no practical case for live adoption.**

Run08 chunk 7 measured 150.266 seconds in the sink, 682.813 seconds in source
and 935.807 seconds in source plus readback. Eliminating its entire sink would
cap source and combined-stage wall savings at 22.01% and 16.06%. Mechanically
applying the synthetic median gives 0.440% and 0.321%; these are shape
illustrations, **not** measured or expected production speedups. The fixture
does not include real actor trajectories, GPU calls, Syzygy decisions, a
strict saved-game replay or a physical-call ledger. Run08 legal-only ZIPs
cannot reconstruct the full raw policy heads needed for this comparison.

The first harness attempt stopped before timing because it combined an older
synthetic discard class with a newer run08 trajectory digest. The corrected
run02 pins the archived actor and matching digest together. The current run08
actor retains audit-only discard rows; real chunks 22 and 29 completed with
discards. The failed run01 is preserved as a fixture-integration failure,
**not** a live source defect.

The [compact evidence record](evidence/2026-09-30-ceres-direct-compact-synthetic-sink.json)
contains all six pairs, component medians, source hashes and scope. The full
code, output journal and ZIPs are sealed outside every checkout at
`<artifact-root>/operations/ceres-direct-compact-synthetic-record-20260930`
(`SEAL.json` SHA-256 `c5fb0199f4b16e3d140c0d7e086298d2bef3d60553e5de60b59ed818dea5a12b`;
`BENCHMARK.json` SHA-256 `9c640a9586a94df7c74f405b58677d633fc84637a09636e0c9644369f37d6804`).
Its SHA256SUMS and all 124 files were independently reopened after copying.
The private proof preserves the original execution path bindings and pinned
environment; its public evidence map uses archive-relative source paths.
A future adoption decision needs a
separately reviewed real full-head actor fixture and closed strict and physical
ledger readback, while keeping the completed-chunk resume bound under an hour.
