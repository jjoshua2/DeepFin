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


## Hosted qualification and publication

Hosted run **35759797335**, temporary workflow commit `6348d9f69e2f8871b895949d5567ec9211347ac2`, passes **79 accepted laws and 140 rejection controls**, supported public-API tests in four native modes, original compiler source/pin checks and unchanged whole-repository lint on source `a26e59f7168519c7de9720b9edb7967ef0650d97`.

The aggregate invokes the unchanged 74-law/126-control prefix gate, then the five-law/14-control normalization gate. The focused contract/control/source report matches the saved local canonical identity; the complete native report matches except the recorded C compiler version. Every candidate native/proof source hash is checked before and after qualification. No previous law or gate was omitted or modified.

The native probe executes only supported public Array.get/set/size APIs, using a complete 131072-cell allocation for each fixture. Each mode passes 74 rows (64 in-range and ten out-of-range wrapper cases), and rejects six malformed requests. Observations are query values, capacity and mask values, not complete returned-cell comparisons. Repeated modes are not disjoint datasets or exhaustive arbitrary-input native tests.

Tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1), locked Python 3.13 CPU development environment and uv 0.12.10. Original compiler source laws and seven controls including cyclic-template rejection pass, and all 12 compiler-pin tests pass. Whole-repository Ruff/Basedpyright/Vulture passes unchanged; it was not rerun locally without its locked environment.

The exploratory direct-internal-routine C generation failure remains unresolved: open Array element type. Its original source and command-output bytes are committed with SHA-256 identities. The passing public-wrapper native probe is explicitly narrower and does not qualify direct calls to Array.get.go/swap.go. Source equalities involving those routines do check. No compiler edit, continue-on-error or relabeling of the failed probe was used.

The complete source tree 166ca47788da93ad29a9c5da7b3668fb35f89c94 matches the locally prepared index with the final parent reconciliation note preserved. Local baseline proof/native bytes came from the previous parent evidence commit; its final note is the only later change. Publication adds evidence, index and matrix documentation only after qualification. Full logs remain in the 30-day artifact bend-address-normalization-qualification; compact reports are committed.

Mask injectivity is a numeric result. Distinct bounded integers reaching distinct leaves of a balanced tree, prefix-plus-interior bounds, clear-path certificates, final computed table contents and independent blocker-ray lookup remain separate unfinished targets. Observed capacity alone does not require a balanced tree. No source theorem assumes its desired read value as a premise.

Branch `feat/bend-address-normalization-20260922` is advanced by a documentation/evidence commit only. No merge, force push, deployment or live-process change; the temporary workflow is excluded from this feature branch. Self-review only, not independent review. No production code, previous proof, compiler input, existing test or routine perft budget changes. No full-engine/model/GPU/training/benchmark result. Python export, data/control orchestration, external references and training, and transitional C++/LibTorch/AOTI inference remain dependencies. Historical compiler TypeScript/diagnostic-depth limits are not fixed or suppressed.
