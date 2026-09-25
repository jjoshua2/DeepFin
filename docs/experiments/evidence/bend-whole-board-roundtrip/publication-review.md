# Whole-board reconstruction publication review

## Exact qualification and source identity

Hosted run **36085950814**, job **107917774878**, completed every stage on its first attempt. The five-law focused gate, seventeen controls, four native actual-observation modes, both behavioral mutations, original compiler source/pin checks and unchanged repository lint passed. The separate structural snapshot lowering probe reproduced its known nonzero result; it is not counted as successful native snapshot execution.

Qualified source: `92737251af190ebddba5b83d146e462ae1049ec6`, tree `a88dfe56fbef013fb9e52bd68582a3212e5b9561`.
Evidence head: `a99bb30d27848180bbe937f13a8d4f6d07fcd7ec`, tree `93f69816bdaa6db37874203f6aa7d5f58005a874`.
Parent: #879 at `d8629e31a55ceda25edef836f8f34a895416dba4`, tree `12a512d0866f8cd8c7f03831a5077b461fe9ffc1`.

Downloaded artifact **10843827073**, ZIP SHA-256 `aee60233ff3414204d197dd24d314f5449a77d4e98216170b672c4e1d1aba9fd`, reproduces all **3,670 tracked paths**, the exact evidence-head tree and original commit object. All **416 candidate native-source hashes** match the inspected local candidate. The entire focused report matches locally except measured consumer seconds; the complete native report matches except the recorded C compiler version. The archived lowering failure also reproduces exactly. No original report was edited to force equality.

The source-only publication patch is 57,803 bytes, SHA-256 `23ccb80147a3cf530a228e561697fa6adece6cb7d705d613b70ec59cbdafa2ed`. The exact binary transport blob matched its locally computed Git identity. Application to the complete parent reproduces the qualified tree and passes Git whitespace checks. The final publication review changes documentation only; it adds no logical source, native test or qualification count.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs and fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. Fork PR #2 was refreshed and still records strict-TypeScript diagnostics; no compiler gate was modified or suppressed.

## Reconciliation rather than duplicate decoding

The supplied local occupied-decoding patch at `c2d327dfd1d6d814554a211f7f6ee4ab7e2c922b` and published #879 are different implementations. The former has four laws and 17 controls under `proofs/decode/`; the latter has four laws and 18 controls under `proofs/decoder/`, including a guarded empty/occupied observation. Neither was silently substituted for the other or overwritten.

This continuation uses #879's published exact-source 127-law/350-control baseline. The local patch remains preserved. Its explicit empty-square observation is recovered as a consequence of the published decoder producers in the new five-law suite. The other four new public contracts establish whole-Board reconstruction, snapshot length, injectivity on consistent Boards and square-order correspondence. This is not a claim to have newly reproved occupied decoding or to have added five independent discoveries.

## Precise proof and runtime scopes

`restore(observe(board)) == board` is proved for every actual Board satisfying the existing per-square partition. All eight bitboards and three arbitrary U32 metadata values are included. Snapshot length and bounded square-order correspondence do not require consistency; the inverse and injectivity do. Snapshots use ordered low/high lists of 32 abstract cells each, plus metadata.

There is no unrestricted inverse over arbitrary malformed Snapshot lists, Invalid cells or arbitrary raw Occupant tags. Metadata preservation does not prove metadata validity, and representation consistency is not chess legality. The new functions are proof-interface operations, not a replacement engine Board representation.

The native verifier does NOT compile or execute structural `Spec.observe` or `Spec.restore`. It executes actual Position operations and real Chess occupancy/decoder/getter observations at every square, then independently reconstructs all eight bitboards externally and compares complete raw Board fields and metadata. Each mode passes 1,026 Boards, 65,664 square observations and 150,822 output fields, with nine malformed requests. Inputs cover 768 fresh square/kind/color insertions, 192 mixed Boards, 64 metadata cases, empty and initial Boards. Generic, portable, native-target and UBSan repeat these fixtures, not disjoint or exhaustive board/game sets.

The two native mutations are an ACTUAL wrong pawn tag and a TEST-PROBE traversal that skips squares. Both compile and execute before incorrect values are rejected. The traversal mutation is not mislabeled a production implementation defect. Neither native test establishes structural snapshot lowering or native ownership/lifetime correctness.

The direct structural snapshot probe still fails pinned C generation with `Error: an arity over 255`. Its original code and fresh hosted nonzero receipt are committed. This is an unresolved limitation of the tested lowering path, not proof that no alternative implementation can ever compile. It is not an accepted semantic rejection control. No compiler modification was made to avoid reporting the limitation.

## Supplementary paired-orientation test

Round-trip equality alone can conceal two matching errors. In one isolated source copy, BOTH the low/high lists in `Spec.observe` and the corresponding reconstruction arguments in `Spec.restore` were exchanged. The unchanged `Roundtrip.bend` still passed with exactly `All terms check.` in 3.969 seconds. The unchanged `Projection.bend` then failed ordinarily at `Projection.structural` in 3.997 seconds. The independent square-order property detects an orientation error that a left-inverse test alone would miss.

Original Spec SHA-256: `418bc772a8d8d650b6a266f9d1bcee7ac19571f9efb2c3d7422fb731b9a26880`.
Paired-mutant Spec SHA-256: `92b95b53f574cc27164c59fa001c631aad91345f30cc7c57d59fe96d6a9060ca`.
Roundtrip success-log SHA-256: `3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84`.
Projection failure-log SHA-256: `ed9861aa8960df299f5c4436d65e75c92ef44128909b67e196c493e3992732c1`.

To reproduce, in a disposable qualified checkout exchange the two `cells(32n,Planes{...})` arguments to `Snapshot` in `observe`, and exchange `columns(32n,low)` with `columns(32n,high)` in `restore`. Leave `square` and every proof unchanged. Check Roundtrip.bend and Projection.bend separately using the pinned main.ts. Require safe status-zero output for the first and status one with expected/observed diagnostics at structural for the second. No native execution of this paired mutant is claimed. Its driver and complete receipts are retained in the review package; it adds no public law or registered control.

## Evidence accounting and retained failures

The focused command executes all five public laws and 17 controls in one successful gate. Eight controls require ordinary semantic/refinement failures at specified locations, eight enforce manifests/imports/no holes/unsafe/symlinks, and one tests synthetic zero-exit warning output. The synthetic test is not a compiler execution. Crashes, missing dependencies, affine-use errors and timeouts do not count as valid semantic rejection.

Exact-source parent127/350 evidence is retained on 404 manifest entries and its successful qualified job, giving **modular132-law/367-control coverage**. The full132-law wrapper was not executed. New source and native reports are actual executions, not just inherited receipts.

Initial dependent-constructor binders, argument ordering and import placement were corrected before acceptance. A first focused attempt passed the consumer and three controls then stopped on a nonunique mutation-site assertion; the exact site was fixed. Another enclosing command timed out without a complete retained result and receives no pass credit. The final file-streamed supervised local focused gate completed in45.768 seconds. Local lint failed for missing tools; fresh hosted Ruff/Basedpyright/Vulture resolves that environment gap but does not relabel the old failure. All original available logs and receipts remain in the review package.

## Remaining work and trust

Next decisive P3 acceptance is parser-derived fresh insertion, then invariant-preserving removal and move/special-move operations. A validated abstract Snapshot inverse in the other direction, valid metadata, legal reachability, king safety and move-generation soundness/completeness/no duplicates remain distinct obligations. Prior P2 results and their direct closed-builder limitation are unchanged.

Self-review only, not independent review. Exact hashes establish provenance rather than a second reviewer. The pinned checker/Base, native lowering/storage, ABI, C/C++ toolchain, libraries, OS and hardware remain trust boundaries. No production runtime, old accepted law, compiler input, permanent workflow, routine perft, search, model/GPU, training or benchmark change. No additional Python application responsibility moved into Bend; export, references, data/control orchestration and training remain dependencies, with C++/LibTorch/AOTI transitional inference. No merge, force push, deployment or live-process action.
