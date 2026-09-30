# Ceres live128 proof-writer A/B/B/A screen

Four sequential, single-GPU arms replayed the same 128 frozen run14 roots
(global game IDs 7168–7295) through the full Ceres source actor, strict
six-man WDL/DTZ readback and physical-call ledger. A used the original ZIP
writer; B used the one-use saved32 proof writer, which reuses a private-file
semantic proof after checking that the published file has the same physical
bytes. The frozen model, batch32 schedule, source roster, rule50 outcome
policy and all-attempt reader were the same. Every arm started with fresh
output roots and carried zero owner or corpus-admission credit.

The preregistered small-screen signal required exact per-game semantic and
all-attempt parity, at least 5% mean B source-stage saving, and positive
A-minus-B source-stage differences in both halves. The screen passed. Every
arm produced 16,390 accepted roots from 128 completed games, including 71
strict Syzygy and 57 natural terminals, with no discards. Each made 609
physical model calls: 16,390 real rows plus 3,098 padding rows. The four
arms had identical in-memory trajectory, physical-feed stream, decoded
per-game saved arrays and row/game metadata, and strict per-game outcomes.
ZIP physical SHA-256 values differed, as expected for the writer change;
all 12 published ZIPs were reopened with matching SHA-256 and passing CRC.

| Arm | Writer | Source stage (s) | Strict readback (s) | Physical readback (s) | Measured stage sum (s) | Actor (s) | Sink (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | Original | 115.461 | 22.821 | 14.227 | 152.509 | 103.434 | 29.189 |
| B1 | Proof reuse | 112.456 | 22.819 | 14.213 | 149.488 | 101.375 | 20.956 |
| B2 | Proof reuse | 103.787 | 23.337 | 14.225 | 141.348 | 92.379 | 21.706 |
| A2 | Original | 121.517 | 23.328 | 14.217 | 159.062 | 110.177 | 29.025 |

The mean source stage was 118.489 seconds for A and 108.121 seconds for B:
10.368 seconds, or **8.75% of A**, faster. The two paired source-stage
differences were +3.005 and +17.730 seconds. Mean sink time fell from 29.107
to 21.331 seconds (26.72%); mean actor time fell from 106.806 to 96.877
seconds (9.30%). Adding the three separately measured source, strict and
physical readback intervals gives 155.786 seconds for A and 145.418 seconds
for B, a 6.65% reduction in that **measured source-and-readback sum**. Strict
and physical readback times themselves were effectively unchanged. This sum
does not include every supervisor or downstream pipeline cost.

The writer's proof transfer assumes exclusive fresh output roots and no
concurrent mutation of the private and published bank files. It is not an
fsync durability claim. The strict whole-game and physical-call readbacks
remain separate, later checks. Four 128-game arms are a positive screen for a
larger sustained producer comparison; they do not establish a production
throughput multiplier, corpus admission, or time to 500M accepted positions.

Source-host artifacts: reviewed packet
`/tmp/ceres-full-prefix-byteproof-abba-screen-v1-20260929/arms.json`
(SHA-256 `8b07cdf132fbfb32068dc101b42cedd79133ebb6e5520f2107ac987887c869c6`),
full decoded-array readback
`/tmp/ceres-full-prefix-byteproof-abba-screen-v1-20260929/readback.json`
(SHA-256 `f6f54578b7141977fdd70da96cca2d29a12c4444c29cd93f79b37213856daf85`),
and a separate physical/strict/timing receipt audit
`/tmp/ceres-live128-writer-independent-audit-20260929.json`
(SHA-256 `4f14b504fea64bdfd4e35061427307e638f0f19d1bafc9b98cb8b9bc01b2e8b9`).
The latter independently rehashed all 12 ZIPs and checked CRC, strict per-game
receipts, timing records and cross-arm counts; decoded-array equality comes
from the first readback.
