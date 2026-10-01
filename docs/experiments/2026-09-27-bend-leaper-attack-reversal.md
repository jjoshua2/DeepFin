# Non-slider attack reversal: coordinates and actual computed masks

## Result and source

This continuation is based on exact PR #910 head `e452fcc965c09fe52010c65ff9b25abcb48ec57d`. Qualified local source
commit: `583aa888bbcd223fdca6a62a204ff374a9257bd3`, tree `eeecc9d67230bc38f07b0b4ccfcb57c16b1929c4`. All 645 inherited native-source files
remain unchanged; the 18-file `proofs/attack_reversal/` suite makes 663 candidate
entries. Its subsequent evidence commit changes documentation only.

Two related public laws establish one new forward/reverse attack correspondence:

- `leaper_coordinate_membership_reverses` equates target membership in the accepted
  independent coordinate lists for all four leaper classes and all on-board source
  and destination squares. Knight and king directions are self-reversing; pawn
  reversal exchanges white and black.
- `computed_leaper_mask_reverses` establishes the same bit-test equality for the
  actual `Slots.value` mask computations used by `Tables.extras`, composing the
  unchanged `Computed.actual` certificate with the new structural reflection.

```text
bit(actual_mask(kind, source), target)
  = bit(actual_mask(reverse(kind), target), source)
```

Both endpoints are Nat indices below 64. This covers all 16,384 class/source/target
triples, including nonedges and equal endpoints. It is a universal bounded source
result, not a theorem inferred from a native sample. The two public interfaces are
related corollaries, not two independent discoveries.

## Proof structure

The four certificate files contain 256 square/class rows. They cover the 1,280
entries in the coordinate-target schedules, including 328 off-board markers and
952 directed on-board edges. The generator emits constructor proofs whose equalities
and impossible-premise branches are checked by Bend. It contributes no assumed
axiom or host-supplied bitboard answer. `generate_certificates.py` reproduced all
four files exactly. Generic list induction transports each certificate to any
member, and Boolean reflection supplies both directions of the equality.

`Mask.bend` establishes mask/list membership by structural Word and list induction,
using the existing checked U64 bit-projection and bitwise-algebra producers. It
handles arbitrary Nat entries in a list, including saturated indices, but only
observes bounded bit positions. A small Data predicate replaces an invalid draft
that attempted to duplicate an affine function closure. The actual computation
connection uses the earlier checked mask producer, not an additional assumed
lookup table or a shadow runtime.

The off-board marker is consequential: `Geometry.targets(Knight,0)` contains 64,
while bit 64 of its U64 mask is false. The consumer checks that distinction, a
real white-pawn forward edge, the incorrect same-color reversal, the correct
black-pawn reversal, a nonattacking king self-square and a high-limb knight edge.
The endpoint bounds are not silently dropped.

## Fresh local qualification

The complete consumer returned exit 0 and exactly `All terms check.` in
27.708 seconds. The final full source gate and all 17 controls
completed successfully in 114.578 seconds, on unchanged source bytes.
The controls comprise eight expected/observed source-refinement rejections, eight
manifest/import checks and one synthetic warning-output unit. The synthetic unit
is not a compiler execution. The entries and exact diagnostics are preserved.

The semantic controls exercise wrong pawn reversal, wrong king steps, an actual
knight offset, omitted endpoint bounds, wrong public reverse class, lost high-limb
reflection, and an incorrect all-zero computed mask. Some target the supporting
certificate or existing computation producer rather than rerunning every large
public composition. Parse/ownership errors, missing imports, crashes and timeouts
are not credited as semantic rejection.

All 15 host regression methods passed under ordinary Python, `-O`, and optimization
level 2. These are 15 repeated host tests, not 45 distinct tests or additional laws.
They cover framing, complete counts, bounded fields, warning rejection, report
replacement, and the distinction between reciprocal and correct attack masks.

The native driver itself ran under Python `-O`. Generic, forced-portable,
native-target and UBSan builds all passed. Each mode checked 3,072 distinct actual
mask requests / 6,144 U32 fields, then 196,608 forward/reverse endpoint comparisons
with 11,424 positive edges. Those pair counts are the same 16,384 geometric triples
repeated over four initialization contexts and three occupancy choices; they are
not 196,608 independent geometries. Build modes repeat the fixtures again.

The unchanged `attack_geometry/probe.bend` executes actual `Tables.build()` or the
three previously defined nonzero/partial/sentinel initialization contexts and actual
`Chess.attack`. It does not execute the new proof model or receive expected answers.
A signed-coordinate forward oracle independently checks every returned mask before
reciprocity is tested. Nine malformed bounded batches were rejected per mode.

Two actual-code corruptions compiled and executed in generic mode. Changing one
knight offset failed both reciprocity and the forward oracle. Swapping BOTH pawn
storage directions is subtler: all 49,152 reciprocity comparisons in the tested
full-builder/three-occupancy context still pass, but the independent oracle rejects
white a1's missing b2 attack (observed mask 0, expected 512). This is a deliberately
mutated runtime, not a discovered production bug. It demonstrates that reciprocity
alone does not establish correct geometry; all-zero masks are a second host-tested
example of the same limitation.

The original compiler 16-law/seven-control suite and all 12 compiler-pin tests passed.
Compiler source remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs,
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Bun is 1.4.2. The source runner used its smaller-heap runtime option; compiler source
and checking semantics were not edited. C compiler identity and exact commands are
in the native and execution reports. Every one of the 36 manifest entries matches
before and after the source and native gates.

## Failures, review and scope

The unchanged whole-repository lint command failed because Ruff, Basedpyright and
Vulture are unavailable. That failed receipt is committed. This is not a locked
Python 3.13 development-environment qualification or a new hosted run. The existing
parent's successful hosted checks remain attached to their original source hashes.

An initial outer tool invocation interrupted the first aggregate gate after some
controls, leaving NOT_COMPLETED; it receives no aggregate pass credit. The final
bounded worker reran the complete gate and native verification serially and finished.
Draft parser/match, dependent rewrite, affine-function and Nat boundary errors were
corrected before that successful run. Available exploratory logs are retained in
the review package, not conflated with the final receipts. A final staged-diff
check caught one extra blank line at the end of Certificate.bend. After removing
it, the complete source/control gate, native verifier, host tests and compiler
gates were rerun on the exact bytes now committed; earlier pass receipts are
retained separately and are not reassigned to changed sources.

Self-review only; no separate reviewer, aggregate 202-law wrapper, merge, deployment,
force push, production-code change, previous-proof change or Python application
migration occurred. The symbolic proof trusts the pinned checker and Base. Native
lowering, C toolchain, memory, operating system and hardware remain boundaries.
No full-buffer, pointer-identity or native-lifetime result is established here.

Active modular totals become 202 laws / 672 controls when this local increment is
included (parent 200/655 plus 2/17). It has NOT been pushed. A write action is not
available in this session, and ordinary Git transport cannot resolve github.com.
The package targets a guarded non-force update of existing PR #910, refusing a
changed head or dependency. It does not bypass Actions approval or merge the PR.

This closes non-slider direction reversal at coordinate and actual computation
interfaces. Blocker-aware slider reversal, a single combined initialized attack
witness in forward semantics, accepted-move castling safety and historical rights
remain separate. Existing initialized non-slider storage/query certificates are
unchanged; native initialized-query evidence is not mislabeled as a new formal
initialized-pair theorem.
