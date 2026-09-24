# Stored-header publication reconciliation

## Exact source and scope

This continuation found the already-qualified branch `feat/bend-stored-headers-20260923` at `9d774ce97b9f9db3cc27a45d894c9d86d8ec2d51`, based on #865 at `a2bb4ddd10868adadea9c7cd39c79030f5263940`, but no pull request for it. It preserves that implementation and makes it reviewable rather than repeating or replacing accepted proofs. This record changes no executable source.

Qualified implementation: `eceb6329cbc000469884513c660c2a615060f83b`, tree `267fc59aadc1fca9c68acae8222a742c45f5f13d`. Its existing successor changes documentation/evidence only. Compiler remains `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs, fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. Repository instructions, development guidance, branch lifecycle and experiment index were read; no merge, deployment, force push or live-process change.

The three public contracts are `own_block_header`, `stored_header_after_tables`, and `stored_header_after_extras`. The first promotes the exact prior supplementary one-block proof. The other two establish correctness of a selected mask/prefix header through later blocks and final extras, including the complete updated array. The numeric domain is `before + 1 + after + start_key <= 128`; complete depth-17 shape is required for arbitrary arrays and derived from actual allocation in the pipeline theorem. Extras satisfy their square budget <=64. No desired incoming header value, size equality, or preservation result is assumed by the caller.

These are not three additional mathematical results produced by this reconciliation. Independent relevant-mask geometry and final slider-data contents remain separate obligations. The universal domains include full table/extras parameter instances; no separately closed normalization of `Tables.build()` is claimed.

## Newly executed reconstruction and report checks

Downloaded and verified:

- Full parent archive, artifact `10782873708`, ZIP SHA-256 `ee3541e9e0353071fd379fdeccfc7654b0bafe70fd3170fa4b4d7423c0a02194`.
- Stored-header qualification, artifact `10784497803`, ZIP SHA-256 `9e9b11686c16e87836e25997502850fffe468437b7e74777bd8ea83dbe6bcf05`.

Indexing all 3,418 original tracked archive paths, including ignored tracked paths and modes, reproduces the complete parent tree `cb7f23738248dae857a1cb29a680d8c762599a66` and original parent commit. Applying the unchanged 83,967-byte saved patch (SHA-256 `ff61e8bb67d5c1fefc046c72947d03d65420c9a96949a43e47db7f7957767f0a`) passes Git whitespace checks and reproduces the complete qualified candidate tree and original source commit object. All **315** candidate native-source manifest entries match.

The original `Facts.bend`, `Header.bend` and `OneBlock.bend` bytes match the prior supplementary-source archive exactly. The stored focused report's source closure matches the reconstructed files. Retained parent report digests and logical-source hashes were checked directly. Of the older strengthened parent's 298 manifest entries, 297 match; the sole difference is its already-recorded extras README clarification. Both exact README hashes agree with the explicit exception record. This is not a claim that an old 298-entry manifest matches every later documentation byte.

GitHub reports all stages of qualification run **35938924299**, job **107442237117**, successful. Its source gate passed the three public laws, importing consumer and 17 controls. Its native verifier, compiler source/pin checks and unchanged repository lint passed. Parent job receipts identify successful runs 35920944024 and 35924465983 with their recorded IDs and source identities.

**Qualification is modular:** the exact-source 100-law/226-control parent result is retained with the three-law/17-control focused result. The full 103-law/243-control wrapper was not executed by that qualification or this reconciliation. The new controls comprise eight ordinary semantic/refinement failures, eight manifest/import checks and one exact-output wrapper unit check; the last is not falsely counted as a new compiler execution.

## Native and pin checks newly rerun

Commands on the reconstructed qualified source:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/headers/verify_native.js /path/to/pinned/bend --report /tmp/header-native.json
bun test native/bend_engine/standalone/verify_compiler.test.js
```

Both complete successfully. Generic, forced-portable, native-target and UBSan modes each pass **nine buffers / 1,179,648 public cell-value comparisons**, all 128 chess keys across the cases, and nine invalid requests per mode. All 12 compiler-pin tests pass. Local Bun is 1.4.2, Clang is 17.0.0; hosted Clang was 18.1.3. The complete new native JSON equals the hosted JSON except the recorded `cc` field.

New native JSON SHA-256: `502d62256a9778b0f49b39bba8dfb8cd0b40e4c75b74e0648f4608bc83df8ffa`.
Hosted native JSON SHA-256: `f4a0983a04d06a46b5e0a7a3d16706a66dc74d2fa57d625def8fb8916b33be37`.

Native cases start from fresh distinct index-derived values; the candidate computes its own prefix using actual production masks and executes actual table/extras loops. The independent oracle supplies no candidate input or answer. Full-cell comparisons include untouched headers/slack and full/partial/suffix executions. Repeated modes and reruns are not disjoint datasets or exhaustive arbitrary-input coverage.

## Newly executed wrong-prefix mutation

In an isolated copy only, the actual stored prefix was changed from `U64.from_u32(at)` to `U64.from_u32(U32.inc(at))`. The unchanged native driver compiled and executed that candidate, then rejected **array cell 128: observed `0 513`, expected `0 512`**, exit 1. The driver stops at that generic-mode failure; no four-mode pass is claimed for the corrupted implementation.

Original Tables.bend SHA-256: `8453d5d7b69d2228ac3dae071c41ec79102566705008f30de1bf14db9fa54b7e`.
Mutated Tables.bend SHA-256: `d3022e9fcfa628c9776379a47f1e9c62943f41ac81e49859ba5c776a89125881`.
Diagnostic log SHA-256: `f40728ac648fe9394e2ae8ee57735b8b5cd2e1e398239dcbca362275278e07cf`.

This is supplementary behavioral validation, not a new public source law or an additional counted source rejection control. The real candidate's source bytes remain unchanged.

## Checks not repeated and remaining work

No full source aggregate, focused source consumer/control run, compiler source suite, or whole-repository lint was repeated for this documentation-only reconciliation. Their retained hosted results and exact logical source identities are distinct from the newly executed native, pin, tree and report checks. The earlier failed header/extras development branch remains untouched and is not counted as a pass. A local streaming-execution tool was unavailable before execution; the ordinary synchronous native command then completed successfully.

Self-review only, not independent review. Hash/report agreement is not independent code review. No production code, prior accepted law, compiler input, existing test or routine perft budget changed. Python export/references/data/control/training and transitional C++/LibTorch/AOTI remain dependencies. Existing compiler TypeScript and normalization/raw-helper limitations are not fixed or suppressed. Source checker/Base and native lowering, allocation/lifetime, ABI, C/C++ toolchain, libraries, OS and hardware remain trust boundaries.

The next decisive P2 work is preserving computed slider-data entries across later metadata and other-block writes, composing the stored-header, fill-content and extras results into actual lookup, and proving equality with independently specified blocker-ray attacks. Correct final headers alone do not establish that remaining data-content or geometric theorem.
