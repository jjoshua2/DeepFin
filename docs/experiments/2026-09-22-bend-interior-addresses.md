# Interior addresses and actual array normalization

## Baseline and bounded acceptance

Continue #826 at fa1afc75ee652d5b13f5788abfc36d9545010df9, tree
5d9df02609c7296fd6b4a26118b1d202b1ff8911. Its final reconciliation note is retained.
Local construction used the code-identical evidence ancestor e8ce93c36a848e87b687aabc0c0483f709c3267a;
the final parent adds only that note. The original six-law prefix implementation
and separate unqualified seven-law draft are preserved, not reconciled by deletion.

Compiler stays aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae: Bend 2.0.21 + U64, all
84 source inputs, fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
Repository guidance/development/branch lifecycle and the experiment index were read.
Only isolated proof/test/docs work is in scope; no merge, deployment, live-process
change, model/GPU/training run or ordinary perft-budget increase.

This acceptance record follows constructive development and precedes hosted
qualification; it is not a backdated preregistration. Full acceptance requires the
five new public contracts and all 21 new controls, the unchanged 74-law/126-control
parent aggregate, four native modes, original compiler source/pin checks and
unchanged whole-repository lint. Counts are acceptance targets until reports pass.

## Five source targets and actual implementation connection

| Law | Guarantee and premise |
| --- | --- |
| allocation_mask_exact | U32 x<131072 implies x AND 131071 equals x. |
| allocation_mask_injective | Equal masked U32s within that bound imply equal original U32s. |
| prefix_interior_certificate | Every absolute member of a valid half-open actual block is >=512, <131072 and unchanged by allocation masking. |
| ordered_normalized_addresses_distinct | Members of distinct ordered blocks retain distinct normalized U32 addresses. |
| interior_read_uses_unmasked_address | With actual array source size 131072 and valid block membership, public Array.get equals internal Array.get.go at x as a complete array/read pair. |

Mask uses Word induction, not 131072 host-generated proof cases. Certified imports
prior prefix PROOF and actual-size producers; callers do not supply desired size
or no-overflow equalities. Interval bridges endpoint inequalities to arbitrary
members. Read reuses certified affine reification and the actual Array.size identity.
No protected checker edit, new axiom, proof hole or unsafe/foreign witness.

The premises are not the desired output equality. Membership is two numeric
inequalities; the size premise observes the actual source API. The consumer gives
first/last allocation values, genuine rook/bishop interiors, distant blocks, a
size witness for an arbitrary seed, and explicit alias counterexamples.

These are **absolute address** statements. Establishing membership for prefix+
relative-index addition and the actual incrementing fill sequence is still needed.
The normalized-number inequality is not a route theorem for arbitrary nonuniform
arrays: a source-checked ragged tree sends in-range indices 4 and 5 to one leaf.
Regular depth-17 shape, route injectivity, clear-write certificates, final computed
contents and independent blocker-ray lookup equality remain separate targets.

## Local development evidence and failures

The structural Mask, Interval and actual-array Read modules passed the unmodified
CLI. All 21 new controls passed in the explicitly labeled controls-only mode;
that mode does not claim the combined consumer or inherited source aggregate.
The final public consumer/aggregate remain pending at this prequalification record.

A first combined draft consumer reached its 600-second local limit without a
verdict; neither success nor semantic rejection is inferred. A development-only
observer of the unmodified checker identified the inherited finite Facts module
as expensive; observer output is not CLI/ownership qualification. The observer
first needed the loader's explicit seen-map argument; this harness error is not a
compiler defect. No compiler file was edited. Constructive binding/pattern errors
were fixed before acceptance, without restricting the stated theorem domains.

The first native probe attempted a direct call to internal Array.get.go. Native
lowering failed with `Error: an open Array element type`. The failure is retained;
standalone native lowering of that internal helper is **not qualified**. The final
probe uses actual public reads with original, explicitly masked, and outside-alias
arguments. This changes the probe, not the source law or compiler.

The final public-API native probe passed generic, forced-portable, native-target,
and UBSan modes locally. Each mode checks all 131072 allocation addresses and
393216 returned U64 values: 107648 slider entries, 512 metadata/extra entries, and
22912 slack locations, plus 131072 outside aliases. Seven invalid input requests
are rejected per mode. The independent reference uses signed geometry and direct
compact scatter. No proof predicate or external reference supplies candidate values.
These are repeated fixed table fixtures, not exhaustive arbitrary inputs or a
native proof of the five source contracts.

## Review and next acceptance

Self-review only; no independent review is claimed. Pinned checker/Base semantics,
native lowering, affine storage/lifetime, ABI, compiler, OS and hardware remain
trust boundaries. Existing compiler-fork strict-TypeScript/diagnostic-depth issues
are not repaired or suppressed. Whole-repository lint and hosted aggregate results
must be recorded separately; inherited green CI is not this candidate's result.

No application responsibility moved from Python to Bend. Export, external references,
data/control orchestration and training remain; C++/LibTorch/AOTI stays transitional
inference. No whole-engine, model/GPU, strength or performance result is added.
The next decisive acceptance is relative-index membership and complete regular-tree
path separation, then frame certificates for final initialized contents and actual
lookup against independent blocker rays. Numeric mask injectivity is only a step.


## Hosted qualification and publication

Hosted run **35761225905**, temporary workflow commit `f4a29f5a9e359db4cda3ba08dd9c48590047c936`, passes **79 accepted laws and 147 rejection controls**, the four-mode public-read native gate, original compiler source/pin checks and unchanged whole-repository lint on source `196229c1ebd2fa93774dfd1250e8580f620354a7`.

The five new laws and 21 controls are checked before the unchanged 74-law/126-control parent, so a new failure stops promptly without omitting inherited qualification. Controls-only mode cannot satisfy the aggregate. All required prefix/layout proof producers are in the checked import graph. Positive source witnesses include genuine interior addresses, a satisfiable allocation-size premise, and both out-of-range masking and nonuniform-route counterexamples.

Every native mode checks all 131072 allocation addresses and 393216 returned values using actual Tables.build and actual public Array.get. Public reads, explicitly normalized arguments and outside aliases match the independent geometric reference; seven invalid requests are rejected per mode. Reports match the local native result except the separately recorded C compiler identity. Modes repeat a fixed table fixture, not disjoint datasets or exhaustive arbitrary arrays.

The source Array.get.go equality is qualified at source level only. The earlier direct internal-helper native probe failed with an open Array element type. The successful native probe intentionally exercises only the public API; it is not evidence that the internal helper now compiles. No compiler file was changed or failure suppressed.

Hosted tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1), locked Python 3.13 CPU development environment and uv 0.12.10. All 12 compiler-pin checks and the original 16-law/seven-control source suite pass. Whole-repository Ruff/Basedpyright/Vulture passes unchanged, resolving the local missing-tools lint gap for this candidate.

The candidate tree and every recorded source hash are checked before and after qualification. Subsequent publication adds only documentation and compact reports. Full logs have 30-day retention in bend-interior-address-qualification; compact reports are committed. The original local timeout and direct-helper compiler failure remain documented as failed or incomplete attempts, not proof passes.

These are absolute half-open interval membership laws, not proof that every relative-index addition or incrementing writer produces such a member. Distinct normalized U32 values do not by themselves separate paths in arbitrary nonuniform arrays. Regular-depth shape/path certificates and their connection to relative PEXT indices, clear-write conditions, final computed contents and independent blocker-ray lookup remain open.

Branch `feat/bend-interior-addresses-20260922` receives a fast-forward documentation/evidence commit only after all checks. No merge, force push, deployment, live-process change, production source change or additional Python-to-Bend migration. Temporary workflow and development transport are absent from the feature branch. Self-review only; no independent code review, full-engine/model/GPU, perft, training, strength or speedup qualification is claimed.
