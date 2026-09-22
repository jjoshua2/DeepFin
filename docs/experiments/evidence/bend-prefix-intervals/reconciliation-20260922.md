# Prefix continuation reconciliation — September 22, 2026

## Selected published source, without replacing the saved candidate

The continuation found the already-qualified `feat/bend-prefix-intervals-20260922`
branch at `e8ce93c36a848e87b687aabc0c0483f709c3267a`, complete tree
`624f40f1a795931bf1964b03c838f05b41ccf2f2`. Its implementation commit is
`34d40b0a8e897fb48536a40fee1c2ea443431b70`; the following commit adds evidence and
documentation. Parent remains PR #822, `b85b4b3620a815150736e4a6ac2858d372160917`.
The branch had no pull request when inspected. This note prepares its reviewable
publication; it does not merge, deploy, rewrite history or change executable code.

The separately saved **seven-contract** candidate is local commit
`17c1cd439e0c2f4be67df91dde17aa04d7996065`, tree
`09cf6bbace75ca5c677d4e2208772e1b545e4661`. Its aggregate remains **NOT QUALIFIED**:
the retained status says stopped without a complete verdict. Its restored consumer
was terminated with status -15 after 646.387 seconds; that is not a proof rejection
or a proof pass. Its own successful native/reference and compiler checks are not
substituted for source qualification.

The saved patch and bundle remain in the conversation's original package, unchanged:

- `bend-prefix-certificates.patch`: SHA-256
  `a8552d016f6a4355022c0a67d8f1fd10c94577a8c08e5ea93c4a61f44f2d8e80`.
- `bend-prefix-certificates.zip`: SHA-256
  `d612c88db1daa9cab419fbf4df1133854182d0d45a92d3503f1f010c6c6cc0a9`.

These are different implementations occupying the same `proofs/prefix/` namespace;
do not apply both patches together. The six-law branch is **not** the seven-law
commit with a passing label, and this reconciliation does not certify an automatic
translation between their differently defined schedules. No previously accepted
law from the 68-law parent is removed or weakened.

## Contract-by-contract scope comparison

| Saved seven-contract target | Related qualified support | Difference retained explicitly |
| --- | --- | --- |
| `table_loop_split` | `tables_follow_prefix` equates the actual loop with an explicit-prefix schedule over the complete affine array. | The qualified public theorem requires `n+k<=128` and starts at the corresponding prefix. It is not the saved unrestricted split for arbitrary counts, initial key and address. |
| `prefix_key` | The qualified schedule uses `U32.from_nat(k)` directly; `Builder.refine` connects the actual incrementing key to its recursive call. | The saved universal `key_after(n,0)` identity is not reproduced as a named law over all Nat n. |
| `prefix_contiguous` | `Spec.prefix` unfolds to previous prefix plus independent size; `Certified.size` supplies actual-size equality for k<128. | This is qualified-domain support, not a new acceptance of the saved unrestricted statement. |
| `prefix_step_safe` | `prefix_in_allocation`, `step_no_overflow`, and checked `Facts` state/transition certificates cover supported endpoint bounds, widening, strict increase and endpoint masking. | The saved bundled `step_ok` proposition itself was not checked on this branch. Interior-address path separation remains open. |
| `prefix_regions_ordered` | `blocks_ordered` proves actual block end <= later prefix for every i<j<128. | Related goal over the qualified branch's schedule; no cross-spec equality is assumed. |
| `terminal_prefix` | `logical_end` proves the qualified schedule ends at 108160. | This does not qualify the old schedule or its entire consumer. |
| `build_prefix_split` | Native tests exercise actual `Tables.build`; the public loop theorem covers arbitrary initial affine arrays within its bounded key domain. | The saved partitioned closed-build source theorem remains unqualified. Full native-buffer comparison does not discharge it. |

The qualified branch additionally exposes cumulative U32-to-U64 widening for all
129 endpoints. Its `Certified` module explicitly imports the existing layout proof
to construct actual-size certificates. The caller does not assume these equalities.
The comparison schedule retains actual header writes and the original fill loop;
it does not replace them with externally computed table values.

## Evidence checked during this reconciliation

The completed qualification is **run 35747413010, job 106812475132**. All steps,
including the source gate, native gate, original compiler checks, locked-environment
lint and publication, report success. Artifact **10703794104** was downloaded;
its ZIP SHA-256 is
`70542b1414fb37201952d4ad0e2eb75289d826617bd3759f92eb6fe787481914`.
The retained reports contain 74 accepted laws (six plus 68 inherited), 126 controls
(19 plus 107 inherited), and the unchanged whole-repository Ruff/Basedpyright/Vulture
pass. This is retained evidence on unchanged sources, not a new aggregate run.

Read-only recovery run **35754763235** recovered the exact qualified delta and
original commit objects. The full restored tree matched the published tree, and
all **216 native-source manifest entries**, plus every source/native report digest,
matched their inspected file bytes. The recovered source differs from the baseline
only in proof/test/documentation files. No recovery workflow is in the feature diff.

A new local execution of the **unchanged** native prefix verifier passed generic,
forced-portable, native-target and UBSan modes: five complete buffers and **655360
cell comparisons per mode**, with all 128 chess keys and seven invalid requests
rejected per mode. The full local native report equals the retained hosted report
except for the separately recorded Clang version (local 17, hosted 18.1.3).
The local original compiler-pin suite also passed **12 tests, zero failures**.
Compiler identity remains the same 84-input fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`, revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2.

The complete source aggregate and whole-repository lint were **not rerun** during
this documentation-only reconciliation. Their exact-source hosted results remain
valid evidence; neither is attributed to the stalled seven-contract package.
Repeated native runs use the same fixtures, not new disjoint datasets. Local
comparison reports and command output are retained in the conversation's
reconciliation evidence package; hosted compact reports remain committed here.

## Remaining acceptance and review boundary

This review checked the public laws, certificate producer, actual builder bridge,
order/finite-domain proof structure, fail-closed gates and independent native
reference. It is **self-review**, not independent code review. No compiler change,
new axiom, weakened old law, production-code change or routine test-budget increase
is introduced. Historical compiler TypeScript/diagnostic limits are unresolved.

The next P2 target remains normalized-path and clear-write certificates for bounded
interior addresses, followed by final computed table contents and actual lookup
against an independent blocker-ray specification. Numeric interval ordering and
native full-buffer parity are not those source theorems. The saved unrestricted
split/closed-build targets are preserved as separate unqualified work, not erased.
No Python responsibility moved into Bend; export, external references, data/control
orchestration and training remain, with C++/LibTorch/AOTI transitional inference.
No full-engine/model/GPU, perft, strength or performance requalification is claimed.
