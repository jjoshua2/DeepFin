# Actual filter mask exclusion and ordinary post-move ray subset

Under explicit source ownership, the strict source-index bound, and actual
Chess.filter_requires(sensitive,m)==False, Filter.source_exclusion derives

    U64.and(U64.bit(U32.to_nat(src)),rays) == U64.zero()

sensitive is the exact castle_safety.Spec.sensitive mask:
Chess.color(board,Chess.get_turn(board)) AND (rays OR Chess.get_kings(board)).
The generic importing exclude consumer accepts an actual Ply and uses its actual
Spec.source; there is no independent or mismatched source argument.

Bits.or_false_right extracts the actual filter test from its flag==1 OR test.
Indexed Word induction derives a False ray observation from the owned source and
False sensitive observation. decoder.Bits.test64 supplies correspondence to the
real test_bit under Nat.is_lt(U32.to_nat(src),64n)==True. A structural bit-first
intersection proof plus existing bit_encoding then yields the zero U64 mask.
All bounded-index certificates required by those actual bridges are retained.
No Boolean bypass is returned unconditionally, and no desired-result premise
supplies the zero intersection.

Filter.path_disjoint explicitly accepts
Lattice.subset(Path.attack(xs,Chess.occupied(board)),rays).
Actual ray_clear_disjoint.Algebra.disjoint_subset transfers source/rays exclusion
to source/OLD path-attack exclusion. Filter.actual_ray calls PR1027
Ordinary.actual_ray to derive the actual post-move Tables.ray subset of the old
actual ray for exactly promotion=0,flag=0. The input-only trace, actual fuel,
stopped=False/mask=False and shared arbitrary accumulator are preserved.

The importing consumer checks generic exclusion and this actual board/filter/ray
composition. A covered example has old ray6, post-move ray2 and arbitrary shared
accumulator. The source is owned; actual bypass and coverage are separately
checked from the input board/mask/move, rather than supplied by the desired result.

Necessary-premise witnesses:
- Without ownership, own0/rays4/source2 produces actual bypassFalse but source
  intersection4. Without bypass, own4/rays4/source2 is owned but filterTrue and
  the intersection remains4. Controls falsely claiming4==0 must reject.
- With ownership and actual bypass but rays2 lacking old-path coverage, source2
  is the old first blocker. Actual old ray6 becomes14 after ordinary move2->0.
  Coverage isFalse, and the control falsely claiming14 subset6 must reject.
- An EP flag requires the full filter even when the sensitive mask is zero.

These are total-function witnesses, not legal-move assertions. Candidate source
ownership, input ray coverage from actual lookup geometry, legal-board/table
invariants, unconditional bypass retention and full legal-move correctness remain
open. No raw Pro code, desired-result witness or whole-board claim is credited.

## Qualification

Exact base PR1027 c7b988e35847d4c94871f7c2ac8925bb33b0471d,
tree61e397b38eb9e2498cfb1a6faf6a40d21a07273e.

From the repository root, choose fresh external evidence paths:

    python3 -m native.bend_engine.standalone.proofs.filter_ray_exclusion.qualify_filter_ray_exclusion /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json --report /outside/checkout/qualification-001.json --evidence-dir /outside/checkout/checks-001

Require Bend 2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, Bun 1.4.2,
all 84 checker files and fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
Exact-base closure/check/evidence/mutation/strict-classifier helpers and source
dependencies are hashed and pinned. Clean published source must match HEAD Git
blobs; every reused dependency must match the exact base.

Each positive and negative allowance is 86400 seconds, CPUs1,3, at most2CPU,
6GiB AS/RSS checks, and16MiB per-output caps. P2 CPUs30-31 are excluded.
Prior evidence, qualified source and compiler identity must remain unchanged.

Nine declaration-local controls reject missing ownership/bypass/coverage through
the concrete counterexamples and weakened generic certificates, an actual-ray
disconnection, lost shared accumulator, and an unconditional EP bypass claim.
Only designated typed-obligation failures count; parser/import/resource/timeout/
linearity failures do not count. Internal independent exact-head review precedes
stacked draft publication. No merge, live adoption, runtime changes, P2/GPU or
new installation.
