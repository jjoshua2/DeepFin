# Subset successor and full-width borrow continuation

## Scope recorded before proof construction

Base PR #806: `1654a945e946440885675b5bc1f7d0ee80f80708`.
Compiler stays `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, with the
existing 84-input fingerprint. No protected checker or production source edits.

Target the actual `(s-mask)&mask` step: connect split-U32 subtraction and borrow
to full-width Word arithmetic, then connect masked arithmetic to compact
successor and the Nat-indexed ordinal theorem where completed. Keep the full
accepted target visible; do not substitute a bounded test or vacuous premise.

Budget: isolated local constructive source proofs and small CPU-native probes,
one compiler at a time; bounded hosted source/native/lint qualification only.
No full engine/model rebuild, perft, training, GPU, deployment or merge. Existing
proof gates and statements stay unchanged. Success requires source status zero
and exactly `All terms check.`, negative controls, and independent native checks
of the actual operations. Every achieved law and remaining obligation is recorded
separately. Self-review only unless a separate reviewer is actually obtained.


## Constructive result and source boundaries

Seven new universally quantified laws are proved in `proofs/successor/LAWS.bend`
and explicitly discharged in its `PROOF.bend`. The suite imports the actual
`Subsets.bend`, the existing Nat-indexed `at` recurrence, and existing Base U64
operations; there is no replacement runtime enumerator and no production edit.

1. `subtraction_refinement`: the real two-U32 `U64.sub` equals full-width
   `Word.sub`, including the actual low-half comparison and high-half borrow.
2. `step_successor`: for every mask-contained state, actual next-state PEXT value
   is the compact successor, wrapping to zero at mathematical capacity.
3. `sequence_ordinal`: for every Nat `i < 2^popcount(mask)`, the PEXT value of
   actual `at(i,mask)` is exactly `i`.
4. `sequence_nonduplicating`: equal states with indices in that range have equal
   indices.
5. `sequence_coverage`: every mask-contained state has a bounded recurrence index;
   the constructive witness is its actual extracted value.
6. `cycle_endpoint`: state at capacity is zero.
7. `sequence_periodic`: state at `i + capacity` equals state at `i`, for all Nat i.

The exact capacity-to-popcount equality is reused from #806, not newly assumed.
Empty and full-width masks are included. Theorems use unbounded Nat powers; no
attempt is made to execute a 2^64-state recurrence. The successor specification
uses an explicit conditional, not a new theorem about Base's Nat remainder code.

`Borrow` establishes general ADC concatenation, final-carry/comparator equivalence,
and high-half decrement/borrow. `Step` connects the production operation to a
structural observation and then to its numeric successor; `Ordinal` inducts over
the actual recurrence. `Value` proves injectivity of Word/U64 numeric observations,
which supplies constructive coverage and the cycle endpoint. Those helpers and
specifications are not application substitutions or imported equality axioms.

## Local executed evidence

Bun 1.4.2, Clang 17.0.0, verified compiler pin above, isolated 4-GiB container.
**PASS:** the new aggregate checks 40 accepted laws in total (seven new, 17 prior
engine/index and 16 inherited U64), all 26 prior negative controls, and 16 new
negative controls. It requires status zero AND exactly `All terms check.`.
The importing consumer checks all-mask use and concrete cross-half/high-bit cases.

The new controls reject missing/hole/omitted proofs, the incorrect inclusive
ordinal bound, removed membership, incorrect wrap, reversed or unmasked actual
steps, a stuck actual recurrence, missing inherited proof and unsafe/foreign
imports. Three disposable Base mutations fail the new `Borrow.bend` directly:
omitted borrow, borrowing on equal halves, and truncated high result. They do not
rely on an older closed fixture to catch the mistake. Import/manifest rejection is
not misrepresented as a semantic counterexample.

A source-checked nonmember counterexample is retained: state 2 with mask 5
produces compact next index 3, while blindly applying compact successor predicts
1. The mask-membership condition is material, not vacuous. Empty-mask index 1
also demonstrates why the ordinal bound must be strict.

**PASS:** actual native U64.sub and Subsets.next/PEXT on 1,605 distinct operand
pairs per mode, in generic, forced-portable, native-target and UBSan C. Each mode
includes 815 low-half-borrow cases, 790 non-borrow cases, 234 compact wraps and
1,371 advancing states; populations 0..64 are represented. Four invalid/budget
requests reject per mode. Full-width BigInt subtraction and direct bit-position
scatter/gather of the next index are independent references. Same fixtures repeat
across environments, not disjoint or exhaustive arbitrary-U64 data sets.
Every native output SHA-256 is
`55a842ac6c1f3fb3197dadc94eac304e11b262029e15b78f96bb352c345d1af2`.

The separate original fork source gate passes 16 laws and seven negative controls,
including cyclic-template rejection. All 12 compiler-pin contracts pass. During
construction, match-order, rewrite-direction and affine-use errors in proof code
were corrected, plus consumer syntax and an incorrect draft counterexample.
Accepted law statements and compiler code were not weakened or modified.

Local source/native/inherited reports and lossless pin logs are committed under
`evidence/bend-subset-successor/`. Full-engine/static-runtime tests and model/GPU
qualification are not rerun or claimed in this proof-only increment. Whole-repo
lint and hosted confirmation are recorded below only after actual execution.

## Interpretation, review and remaining work

The general P1 subset-order, complete/nonduplicating coverage and period targets
are now **law proved**, conditional on the pinned checker/Base. Chess-mask
population bounds and P2 affine table/lookup refinement remain open. The next
acceptance connects these ordinal laws to actual U32 size/offset arithmetic,
initialized Array regions and lookup, with an independent ray specification.
Source proof does not prove C lowering, runtime/ABI, C compiler or hardware.

Self-review checked the real split-U32 implementation link, strict domains, empty
and full masks, induction over the imported recurrence, constructive witness and
inherited import graph. No independent review was obtained or claimed. Compiler
fork strict-TypeScript failures remain separately unresolved and unsuppressed.

Nothing newly moves from Python into Bend. Export, production/data control and
training still have Python dependencies; C++/LibTorch/AOTI remains transitional
model execution. No whole-engine/model build, perft, trained-checkpoint/GPU test,
training, speed/strength measurement, merge or deployment belongs to this increment.


## Hosted qualification

Run **35655691011**, temporary workflow commit `7881299921b8640ed5b14cc379d72f84eff42860`, passed every qualification stage before publication on exact PR #806 parent `1654a945e946440885675b5bc1f7d0ee80f80708`. Candidate executable/proof/test sources matched the local source hashes before and after qualification. The unchanged compiler fingerprint was checked again.

The aggregate source gate proves all seven new universal laws, retaining 17 earlier engine/index laws and 16 original U64 laws: 40 accepted laws total. All 16 new negative controls and 26 prior controls pass. The new source report is byte-identical to the local report. The original fork source gate (16 laws/seven controls) and all 12 compiler-pin contracts also pass separately. No accepted obligation, compiler fingerprint or checker was weakened.

All four native modes pass 1,605 distinct operand pairs each: 815 low-half borrows, 790 without borrow, 234 successor wraps and 1,371 advances. Mask populations 0..64 and four malformed/budget rejections are covered. Mode results and source identities match the local gate exactly; hosted compiler `Ubuntu clang version 18.1.3 (1ubuntu1)`, Bun `1.4.2`. The shared output digest is `55a842ac6c1f3fb3197dadc94eac304e11b262029e15b78f96bb352c345d1af2`. These are repeated environments, not exhaustive arbitrary-U64 testing.

The unchanged whole-repository lint gate passes Ruff, Basedpyright and Vulture in the locked CPU development environment. No failed check was suppressed. Original logs are retained losslessly as JSON strings with SHA-256 identities. The two duplicated characters in the first compressed transport fragment were corrected before decompression, then its exact expected blob hash and the complete original patch SHA-256 were required; this does not change any candidate source.

No production source, full engine binary, perft, model export/forward, GPU, trainer, benchmark or full inherited native suite was changed or rerun. The full ordinal law is now a source proof, not an inference from the bounded native samples. It uses the imported recurrence and actual subtraction; the conditional cyclic-successor specification is not a separate proof of Base Nat.mod/div. Chess-mask population/offset bounds and P2 affine table/lookup refinement remain open. Source proofs trust the pinned checker/Base; native lowering/runtime/ABI/toolchain/hardware remain separately tested/trusted. Self-review only.

Reports and exact source identities are committed under this record's evidence directory and retained in the 30-day `bend-subset-ordinal-qualification` artifact. Patch SHA-256: `6c53eb780dd2296aee7cd4438eda06002d58e4c6d077995e1e58fc38932f1455`. Publication is create-only on `feat/bend-subset-ordinal-20260921`. No force push, merge, deployment or live-process change. The temporary workflow and compressed transport are absent from the clean feature branch.
