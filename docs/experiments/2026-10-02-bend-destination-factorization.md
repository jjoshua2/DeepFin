# Actual destination factorization, 2 October 2026

Outcome: **PASS for four structural component contracts** in a local-only worktree
based on current main `269105298285b6098ffbf80405cb18ec33186b38`. This advances the
ordinary-generator proof foundation; independent bitboard inventory and complete
legal-generator correctness remain open. No push, PR or merge is authorized for
this increment.

The [theorem guide](../../native/bend_engine/standalone/proofs/destination_factorization/README.md)
states the exact domains and proof decomposition. The importing consumer proves:

```text
Chess.destinations(n,empty,bb,src,pawn,ep,Factor.emit(keys,src,pawn,ep,tail))
  == Factor.emit(Chess.bit_squares(n,empty,bb,keys),src,pawn,ep,tail)
Chess.destinations(n,empty,bb,src,pawn,ep,tail)
  == Spec.expand(Chess.bit_squares(n,empty,bb,Nil),src,pawn,ep,tail)
Spec.count(Chess.destinations(n,empty,bb,src,pawn,ep,tail),query)
  == Spec.tally(Chess.bit_squares(n,empty,bb,Nil),src,pawn,ep,query,Spec.count(tail,query))
Chess.scan_after(src,pawn,ep,tail,(table,targets))
  == (table,Spec.expand(Chess.bit_squares(64,is_zero(targets),targets,Nil),src,pawn,ep,tail))
```

The first three quantify over arbitrary budget and empty flag, bitboard, source,
pawn flag, EP square and tail, with no premises; the first also allows an arbitrary
key accumulator and the third compares every field of an arbitrary Ply query.
The scan theorem uses the actual fixed budget and consistent initial empty flag,
with an arbitrary affine Array and tail. Existing checked complete-array reification
derives its internal witness; neither table initialization nor a caller certificate
is assumed. Exact order and complete array equality are retained.

Promotion reuses accepted `promotion_choices_exact` with its checked body loaded.
It emits qp 1,2,3,4 with flag zero, with no qp-zero ordinary occurrence. Tail counts
are carried explicitly. The separate supplied-key specification calls neither the
actual generator nor the actual bit scanner. Actual `bit_squares` remains in the
public theorem: independent coverage, boundedness and distinctness are unfinished.
Budget zero on a nonempty bitboard already refutes an arbitrary-budget whole-BB
inventory claim. An eventual complete-BB theorem needs budget 64 and
`empty=is_zero(bb)`, or sufficient budget and consistent empty. Actual closed
boundary examples and a symbolic promotion qp-zero exclusion lemma also check.

## Recovery and qualification

The [Muse recovery receipt](evidence/bend-destination-factorization/muse-recovery.json)
records the old clean live-base worktree `a38988814a39c1801c2f3017b3359cbe19bdb1ce`
and a hashed 980-event log. There were no Bend files, no saved proof patch and no
completed proof-writing command. Its investigation was reused; no unverified
candidate was promoted. The current-main open-PR read was empty before work began.

The [final source receipt](evidence/bend-destination-factorization/source.json) has
SHA-256 `429764be7aca12d6cea474fd647e589162c2010384cae4d4c98b16c2b6de11fb`. Its 45 identities cover the transitive
consumer closure, gate, guide, tests and imported host policy. The existing pin is
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 plus U64, with 84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The gate checks identical compiler fingerprints before and after, and unchanged
source identities before PASS. All 249 inherited proof inputs retain their hashes;
the inherited receipt remains
`6e20c442ca16b43a80803c3d2156bf6471da7414a6080d5507cccda6e432f800`.

The structural consumer and complete consumer returned exit 0, no timeout and exact
`All terms check.` in 0.696 and
3.959 seconds. Their output SHA-256 is
`3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84`.
Five disposable-copy controls reject actual ordinary flag, promotion rank, EP tag,
bit-square key and disconnected-consumer changes at the intended semantic obligation.
No timeout, backend, missing-file, quantity or kind failure receives credit.

Final gate CPU was 11.009 seconds; wall time
13.641 seconds. All saved prototype, gate and host runs are summarized in
[cpu-accounting.json](evidence/bend-destination-factorization/cpu-accounting.json):
311.247 measured subprocess CPU seconds. Earlier lightweight
reads/writes and tool-output-only focused checks are not included in that measurement. Every proof
check was bounded at 180 seconds (controls 120), below the authorized 900-second
limit, with two-core affinity, numerical thread settings 2 and nice 10. Total new
CPU remains inside 5400 seconds. No compiler, production engine, existing proof,
training checkout, arena, dataset, GPU, lease, credentials or configuration was
changed. The only implementation additions are this component's proof and gate.

## Host checks and independent review

Three fail-closed wrapper tests passed in normal, -O and -OO modes. All 12 unchanged
compiler-pin tests passed. Focused ruff, basedpyright and vulture pass. Whole-repository
lint reports 277 missing-environment/type diagnostics on both candidate and a clean
current-main archive. Their complete logs are byte-identical after path normalization,
SHA-256 `ae73bcb4598825963dbc9af5e758afe512b6d0dc69205ca2751a7d32b093cf9a`.
The [compact host receipt](evidence/bend-destination-factorization/host-validation.json)
records both failures honestly; no whole-repository lint PASS is claimed. Full raw
logs and failed wrong-module/missing-login-PATH attempts remain in the local handoff.
No dependency or configuration change was made.

A separate read-only reviewer found no theorem defect and requested after-check
compiler identity verification. That correction is implemented and included in the
final PASS above. The [final independent review](evidence/bend-destination-factorization/independent-review.md)
is bound to the checked proof manifest and source receipt. No publication or
native/whole-program correctness is claimed.
