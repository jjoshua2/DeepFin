# Frozen S7 selector: recovered full16 analysis

The frozen cheap selector `ply >= 80 && piece_count == 7` **missed its
predeclared discovery gate** on the 16 whole run07 holdout shards. It selected
11,990 of 132,234 retained rows, or 9.06726%, within the 12% search cap. It
captured 15.19317% of complete ordinary depth-9 BT4 deficit mass above 300 cp,
below the required 20%. A per-shard same-count random control captured
9.23808%; the paired advantage was +5.95509 percentage points, with a 2,000
whole-game-within-shard bootstrap 95% interval of [+3.87336, +8.17350]
points. The random comparison is positive, but the conjunction of frozen
criteria fails. S7 should not enter a sparse Stockfish policy-correction
training comparison on this evidence. A changed selector needs a new untouched
holdout; this diagnostic says nothing about playing strength.

This is a **recovered analysis from a failed terminal**, not a completed
physical-run receipt. The reviewed v2 operator joined all 16 raw shards and
saved per-game sufficient statistics, but stopped while publishing its
208,531-byte receipt: its writer passed an unsupported 4-MiB small-read cap
where the reader accepts at most 1 MiB. The receipt was quarantined, the
terminal remained empty, and source, corpus, target, outcome and completion
credit stay zero. No raw archives were reread to repair that status.

An independent metadata-only audit verified the exact claim, failure and
quarantined-receipt hashes; all 16 shard identities and 681 game identities;
row and deficit-mass sums; and the frozen metric, bootstrap and decision
functions. Its [path-free public projection](evidence/2026-09-29-s7-full16-recovery-audit.json)
(SHA-256 `619c1b896ebd6cb20f89cad7e372bd4611f3df3072bbf9f6a2b77da0eb033b2b`)
records the exact lineage and limits; it removes only the local `run_root` and
identifies the immutable full independent audit by SHA-256
`60b81aaf5f9cf333bf62449572a0860d8cd62eebc367c44686898d5e9a7fc294`.
The reviewed operator packet SHA-256 is
`401b5150c2b9a1ee0dd38709cbbfdc28f0a88c386d29a02eebbe058c1e61df2c`;
the independent source-review SHA-256 is
`e134487ce8e4cc8add7d67f94be56db74a4b5d842fd0ab00747ce26680138696`;
the quarantined receipt SHA-256 is
`feec7a03051bb775ca3d6900650d5906867a932b0d54f78df35a27c62e8bd4c2`.
The audit did not independently reread source payload or upgrade the failed
terminal to COMPLETE.

The frozen population was 132,647 raw rows, 132,234 retained rows and 413
provenance exclusions, from 681 progress-listed qualified games (680 with
retained rows). The quarantined receipt reported 93,375 ordinary depth-9 rows,
38,859 mate-domain rows, 13 invalid depth-9 panels only among excluded raw
rows, zero duplicate raw input keys, and 11,003.912685 total ordinary deficit
mass. Mate-domain rows contributed to selector search cost and zero ordinary
deficit mass. The recorded pre-failure cost was 154.116 seconds, 527.532 MB
process read/write bytes, 254.218 MB peak RSS, and 910.676 MB peak address
space, within the frozen caps. These are recorded observations before the
publication error, not terminal resource readback.

An unlaunched source successor fixes the small-read cap and has a synthetic
208,531-byte regression. Reopening all 16 source shards solely to obtain a
successful terminal would not change the missed frozen capture threshold.
The broader [saved-dose readout](2026-09-28-sf-late-position-dose.md) remains
useful for designing a *new* selector with a new holdout, separately from
source-mix or value-target strength tests.
