# Actual array address normalization — September 22, 2026

## Scope and acceptance

Parent is PR #826 at `fa1afc75ee652d5b13f5788abfc36d9545010df9`.
The recovered implementation baseline is the parent's previous documentation/evidence
commit `e8ce93c36a848e87b687aabc0c0483f709c3267a`, tree
`624f40f1a795931bf1964b03c838f05b41ccf2f2`. The final parent only adds its
reconciliation note, which is preserved by the remote base tree. All baseline
native/proof files used locally are unchanged. No old seven-contract patch is applied.
Compiler stays `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
with the same 84-source fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This record follows constructive local development, before hosted qualification;
it is not a backdated preregistration. Acceptance requires the unchanged 74-law
parent, all new laws/controls, the supported public-API native probe, original
compiler checks and unchanged whole-repository lint. Checks stay bounded/opt-in.
No production process, perft budget, model, GPU or training work is changed.

## Five laws that passed the local focused gate

| Law | Actual guarantee and premise |
| --- | --- |
| `bounded_mask_identity` | Every U32 i<131072 equals i & (131072-1). |
| `bounded_mask_injective` | Equal masked addresses imply equal original addresses when both are in range. |
| `bounded_read_direct` | The actual Array.get wrapper equals the entire result of Array.get.go at the original index, given observed capacity and range. |
| `bounded_write_direct` | Actual Array.set equals Array.set.fin of Array.swap.go at the original index, with the same premises. |
| `bounded_separation_direct` | The existing separation predicate equals its route test at the original unmasked indices, when capacity and both bounds hold. |

Bounds quantify over every interior address, not finite endpoint examples.
Structural Word comparison proves the arithmetic without expanding machine bounds
into gigantic unary Nats. Certified affine reification establishes the actual size
pair. The consumer derives the real Array.new capacity for arbitrary seeds and
retains zero/interior/last-address uses and the excluded-capacity alias counterexample.
No new axiom, unsafe proof, foreign equality witness or weakened accepted law.

## Local checks and construction outcomes

The focused five-law importing consumer and all 14 new rejection controls pass.
Six semantic controls reject a shortened mask, an inclusive or oversized bound,
wrong real Base read/write masks, and a truncated existing separation query.
Eight policy/manifest controls retain all obligations/imports and reject holes,
foreign inputs, symlinks and unsafe warnings even with raw status zero. Crashes,
timeouts and missing-file diagnostics do not count as valid semantic rejection.
The unchanged parent aggregate is reserved for hosted qualification, not claimed
as locally rerun. The same is true of whole-repository lint.

The supported public-API native probe passes generic, forced-portable, native-target
and UBSan modes: 74 rows per mode (64 in-range, ten out-of-range wrappers), each
using an actual 131072-cell allocation. Six malformed requests reject per mode.
The observation covers initial/post-write queries, capacity and masked address;
it is not a full-array-cell comparison. Repeated modes use the same fixtures.
The independent reference uses unsigned remainder and known seed/write values.

Two development limits are retained. A Nat-based proof draft overflowed the checker
stack while expanding a machine bound; the final Word-comparison proof preserves
the entire original domain and leaves the checker unchanged. A redundant consumer
that repeated complete Array.new/read/write expressions hit its 25-second local
limit; the final consumer checks real capacity once and exercises the universal
API laws on symbolic arrays instead of expanding complete initialized trees.

The first native probe attempted direct calls to internal Array.get.go/swap.go.
C generation failed with `an open Array element type`. This is **not** classified
as a passing compiler or native test. The final supported public-API probe is a
separate, narrower native test; the internal routines remain source-qualified only.
The original probe text and failed command output are preserved as evidence. No
compiler edit, weakened source theorem or ignored native failure was used.

## Remaining acceptance and trust

The next missing route theorem must connect distinct bounded integers to distinct
leaves of a complete depth-17 tree. Numeric mask injectivity alone does not prove
that for arbitrary nonuniform shapes. The actual prefix-plus-interior address
bounds must also be derived, then combined with the frame/ordinal laws to prove
computed final table contents and independent blocker-ray lookup refinement.
No one of these remaining claims follows from the five laws above.

Self-review only, not independent review. Source results trust the pinned checker
and Base; native lowering, allocation/lifetime, effects/ABI, C toolchain, libraries,
OS and hardware remain separate boundaries. Historical fork TypeScript and
diagnostic-depth issues are not fixed or suppressed. Python export, external
references, data/control orchestration and training remain; C++/LibTorch/AOTI is
transitional inference. No new Python responsibility moved into Bend and no
full-engine/model/GPU, perft, strength or performance qualification is claimed.
