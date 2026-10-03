# Checked partial: destination count factorization

Status: partial; no end-to-end scan_after occurrence theorem and no PR.

Branch: proof/bend-scan-after-predicate-20261003
Parent commit: 1c7fbc68bc81f7b59ae1b6eb903ffb5450381dd5 (PR1006 head)
Parent tree: bc35d5d17cff166ee155d378b4f3e1be8cf548a3
Checker: Bend 2.0.21 + U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae
Pinned source fingerprint: 84 files, SHA-256 d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4
Checker file: /tmp/deepfin-king-away-checker-aaeb9bc/bend2/main.ts
Bun: /home/josh/.bun/bin/bun v1.4.2
Checked proof: native/bend_engine/standalone/proofs/destination_factorization/QueryCount.bend
Proof SHA-256: 753d02315235654fd036bce73e3ce4a5edccb5cd64d29223708398afadd528a4

Result: All terms check.

Command (run from worktree):
ulimit -v 6291456 && ulimit -f 64 && timeout --signal=TERM 86400s taskset -c 0,1 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 BEND_NO_TELEMETRY=1 /home/josh/.bun/bin/bun --smol /tmp/deepfin-king-away-checker-aaeb9bc/bend2/main.ts native/bend_engine/standalone/proofs/destination_factorization/QueryCount.bend --check-only

The checked module proves structural list-sum equalities for Spec.tally and factors a specification-level Ply emission count through destination equality, including promotion tags 1 through 4. It does not yet connect this count to the actual Chess.destinations producer. The unfinished composition requires proving destination_equal(dst, query) agrees with the query's U32 destination so PR1000's all-U32 bit_squares frequency theorem can replace the full destination fold. No claim is made here about legal move correctness, target geometry, or board validity.
