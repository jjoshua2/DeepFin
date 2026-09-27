# Reconcile and qualify saved non-slider reversal proofs

## Candidate and scope

This continuation updates existing PR #910 on exact head
`4d0d51fcc24b3d814a3e57d0dc676ffd514af33a`. That baseline includes the three
accepted-castle destination laws added after the saved reversal package was made.
The accepted-destination suite and every inherited native source are unchanged.

All 18 saved primary files from local commit
`717b56793e1de754536114a4ae31c742728241e9` are restored byte-for-byte under
`proofs/attack_reversal/`, together with its original local readout and receipts.
The two saved index/migration pages are not used to replace the newer pages.
The original 202/672 totals and local-only status remain historical; the active
baseline is 203/672 and this increment contributes two laws and 17 controls,
for modular 205/689 after successful qualification. No duplicate laws are counted.

The two contracts cover coordinate-list and actual computed-mask reversal for
knight, king and both pawn directions, for two on-board endpoints. They do not
prove slider blocker reversal, a universal initialized forward attacked witness,
whole-generator starting/transit safety, or historical castling rights.

## Validation plan and current status

Fresh qualification is pending. The saved local consumer/controls/native receipts
are preserved, not represented as a current hosted pass. The planned bounded run
executes certificate reproducibility, host tests in three Python modes, the full
consumer and 17 controls, four native build modes under optimized Python, original
compiler/pin checks and unchanged configured locked-environment repository lint.

Successful publication must recheck this exact remote parent and all source hashes.
No production/runtime, prior theorem, compiler input, permanent workflow, GPU,
training, search, benchmark, perft budget or application responsibility changes.
Self-review only unless a separate reviewer is explicitly recorded. An empty set
of review threads is not independent review.
