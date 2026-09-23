# Fill-contents publication reconciliation — September 23, 2026

This continuation makes the saved and already-qualified three-law increment
reviewable. It does not create three additional laws or relabel retained test
results as newly executed checks. No production source, proof, verifier, compiler,
existing test, or workflow is changed by this reconciliation.

## Identities checked again

- Dependency: PR #860, `feat/bend-fill-intervals-20260922` at
  `0d0b325be4e8b0790b482aebbe096000fe989819`.
- Complete dependency tree: `540dc223e40a84f4bb3a2626e0591d8b3ff0b621`.
- Saved patch: 225,819 bytes, SHA-256
  `ba473c4cc0ec106f7c8cd5bfddeeaebfaf78c293c9dab6d586dc729975b9b866`.
- Qualified source commit: `94e380d218d1713828fd192427282588bf7df910`.
- Complete qualified source tree: `731bf2a90704324e1dd34727d3f1aaad07162a67`.
- Refreshed pre-reconciliation head: `dba39a1b1cc53461bc3b7e2b6c42942eab7b46f3`.
  GitHub's comparison confirms that every change since the qualified source
  commit is documentation or evidence, not executable or proof source.
- Compiler branch still resolves to `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`.
  The unchanged 84-input fingerprint is
  `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

The source archive from run 35898058655, artifact 10768225491, was downloaded
again and its ZIP SHA-256 matched
`56c7c697ec84c619895e14cbfb8347f67cfce3a1f2d2d27f99a4bdeec6adbe3e`.
An isolated Git repository was reconstructed from all archived tracked paths and
recorded executable/symlink modes. An initial naive git add omitted 178 archived
paths covered by the repository's ignore rules; the tree mismatch correctly
stopped verification. Indexing those original archive paths reproduced the exact
dependency tree. No archive or source bytes were edited to resolve the mismatch.

The unchanged saved patch applied cleanly, passed git diff --check, and reproduced
the complete qualified source tree. Its bytes are identical to saved-source.patch
in the hosted artifact. The original qualified commit object also reproduced its
recorded Git SHA. All 283 native-source manifest entries matched local file bytes.
These are new reconstruction/hash checks, not new theorem or runtime executions.

## Retained qualification, freshly inspected

Run **35896688343**, job **107302258875**, has completed successfully at every
stage, including publication. Its artifact 10768704502 was downloaded again; ZIP
SHA-256 matched
`8314fe4d6106b6c8e4678e47b48b402ca6a638105b8f9f3982c9f00e5b37c476`.

The report records 98 accepted laws, including the unchanged 95-law parent and
three contents laws, plus 212 rejection controls (195 inherited and 17 new).
Every new control's category, diagnostic hash, excerpt and outcome matches the
saved local controls report. The native report matches the saved local report
except its separately recorded C compiler identity: four modes, 452 fixtures per
mode, all 128 keys, 384 full-block samples, 412 written entries, 40 untouched
controls and 264 entries followed by later writes. Every mode rejects six invalid
requests and has output SHA-256
`ae998c57111da290688df18aa4cd19df97121b6164afab25dd2a88ea17ea895c`.

The retained original compiler source gate passes; the pin log records 12 pass
and zero fail; unchanged repository lint ends with lint: OK. The underlying
Ruff/Basedpyright/Vulture checks were completed by the qualification job.
The full source aggregate, native suite and repository lint were **not rerun**
for this documentation-only publication reconciliation. The earlier local-only
and missing-tool statements remain historical records, not the current status.

## Scope, parallel work and review

The three public contracts remain bounded_fill_entry, bounded_zero_fill_read and
full_block_entry. They connect actual fill execution to its selected Tables.slider
value; the latter two include the complete final array in the returned pair.
They do not prove all stored mask/prefix headers, persistence through the entire
builder's later phases, or equality with an independent ray specification.
The supplementary arbitrary-start pair/native checks already recorded in
review-supplement.md were read, not newly executed or added to aggregate counts.

The parallel `feat/bend-fill-frame-20260923` branch at
`f5db02af035cfbdbde58a2e578bc98250bfddf7d` is preserved untouched. Its common
ancestor with this branch is the older relative-address head, not this contents
head. Its source/results are not silently incorporated or counted here.

Self-review only, not independent review. Hash comparison does not constitute
independent code review. The pinned checker/Base and native lowering, array
allocation/lifetime, ABI, toolchain, libraries, OS and hardware remain trust
boundaries. No Python responsibility moves into Bend; export, references,
data/control/training and transitional C++/LibTorch/AOTI remain dependencies.
No merge, deployment, full-engine/model/GPU/perft increase, training, performance
or strength result is introduced.

Next acceptance remains stored-header correctness and composition of inside-block
contents with outside-block framing across later metadata/other-block/extras
writes, then actual lookup equality with independently specified blocker rays.
