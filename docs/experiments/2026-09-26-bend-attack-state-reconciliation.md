# Initialized attack reconciliation and retained table state

## Decision and exact preservation

Update PR #897 instead of opening a duplicate composition PR. The supplied local
attack_composition variant and active initialized_attacks variant have different
implementations of the same three contract domains: typed piece queries, attacked,
and singleton in_check, with the same symbolic initialization and full-array results.
All 15 saved primary files are preserved byte-for-byte as .txt in
evidence/bend-initialized-attacked/saved-alternative-7d6f3b19/source, alongside the
original readout and selected original receipts. The archive manifest gives hashes
and restoration instructions. Neither implementation is overwritten. This is a
scope comparison, not a newly proved equivalence or an independent reviewer.

No additional public law is counted: totals remain modular186/581.

## New executable state regression

Source `3a3bdce49916aa52c56cad50f756f50eff694ed9` adds only verify_table_state.py to the active native
suite. It reuses the existing forward-coordinate reference and instruments a
disposable native probe to plant and read a nonzero sentinel at unused slot131071.
No production, compiler, inherited proof or existing test file changes.

Generic, portable, native-target and UBSan each pass384 sequential query requests
(128 base queries across three initializers),5760 query fields and three sentinel
reads. Modes repeat fixtures. Two actual-code mutants are compiled/executed using
generic C flags. Replacing the returned table preserves the first masks and attack
Boolean but changes the following in_check from1 to0. Erasing ONLY the unused slot
preserves EVERY query field; the sentinel alone rejects this state corruption.
These are deliberate regressions, not production bugs. One sentinel is not a
full-buffer comparison, pointer-identity guarantee or native lifetime proof.

## Fresh versus retained qualification

Hosted run **36289048263** freshly executes both the unchanged
original native verifier (42270 requests/634050 fields per mode, four original
mutants, nine malformed requests per mode) and the new state verifier. Original
native reports agree with the prior qualified report except C compiler identity.
Compiler16-law/seven-control tests,12 pin tests and unchanged locked-CPU repository
lint pass. Repository lint retains its configured scope; it is not relabeled as
a new explicit all-native-Python type-check. The added file is syntax-compiled
and exercised by the native regression.

The expensive full three-law importing consumer and19 source controls are retained
from exact-source run36286418567/job108527941137, NOT rerun or counted twice here.
All585 qualified native-source entries are checked unchanged. The historical
alternate local lint failure remains in its archive; no fresh execution of those
archived alternative proofs is claimed. No complete186-law aggregate ran.

## Next acceptance and trust

Universal forward/reverse attack correspondence, singleton/valid-side propagation
through castling and complete semantic legal-move correctness remain separate.
No new mathematics is claimed for this test/publication increment. Self-review
only. Existing checker/Base, native lowering/storage, ABI, toolchain, library, OS
and hardware boundaries remain. No merge, deployment, live process, Python
application responsibility, model/GPU, search, training or perft change.
