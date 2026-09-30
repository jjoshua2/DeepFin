# Parser-prefix publication review

## Exact qualification

Run **36144541680**, job **108102300933**, passed every stage on its first attempt: exact recovery, focused proofs/controls, four native modes and both real-code mutations, original compiler tests, locked CPU development environment, unchanged repository lint and publication.

Qualified source: `c3e1fe63d48790859685dc91eabf9012c2378495`, tree `e5417c334f702e6d5876d905f601d3ccad551ddf`.
Evidence head: `210084c6694df96ce8097b37371524c8edb0cbfe`, tree `87a293647f5ecbd95f201227231eff1fddb4f5a3`.
Parent: #880 at `0aa3794344f3cd2fcb683251d9cb905809bd1560`, tree `8dc290537eae8058dd7999c3c6d6748a4f0f26c7`.

Artifact **10869077841**, ZIP SHA-256 `b254ce0db1f94076fa6f341264f3b4c5278f9d31fcc657c8cf65faff53ef7a7c`, reconstructs all **3,694 tracked paths**, the exact evidence tree and original commit object. All **429 native-source manifest entries** match the inspected local candidate. The entire focused report matches except consumer seconds (4.713 local,1.797 hosted). The complete native report matches except C compiler identity. No original report was normalized to force agreement.

The 62,332-byte source patch has SHA-256 `8d137c655e8bd1e15b714736780910b3d85dde1021f663c4c56cf79ae08e9f37`. Its exact application to the complete parent reproduces the qualified source tree and passes whitespace checks. The parent itself was recovered by applying its final documentation-only archive to the existing complete evidence tree and checking its original raw commit object. This is not a partial-file baseline claim.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs and fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. The compiler fork's strict-TypeScript diagnostics and the parent's structural snapshot lowering limitation remain unresolved.

## Evidence categories and scope

Five new universal laws and all17 controls ran together in the focused command. Eight controls are ordinary intended semantic/refinement failures; eight enforce manifest/import safety; one synthetic unsafe-warning unit is not another compiler execution. Missing names/imports, malformed terms, affine errors, crashes and timeouts do not count as semantic rejection. Safe success requires exit0 and exactly `All terms check.`.

Exact-source parent132/367 is retained on416 source hashes and its successful job/report, giving **modular137-law/384-control coverage**. The full137-law wrapper was not run. The new source and native results are actual executions, not merely retained parent tests.

Native generic/portable/native-target/UBSan each passes1,295 input cases and89,355 fields across three complete Layout observations: prefix, suffix-after-prefix and concatenated traversal. Each state includes file/rank, validity, acceptance, all eight Board planes and all three metadata fields. Nine malformed probe requests are rejected per mode. There are768 single-piece placements, every split of empty/start layouts, malformed and Unicode inputs, cursor boundaries and three different initial Board seeds. Of these,830 final states are accepted and382 prefixes are invalid. Distinct input tuples do not mean distinct strings, and modes repeat fixtures rather than adding disjoint coverage.

The candidate receives test strings/cursors/metadata/seed selections, not reference-produced parser results. The independent reference tracks64 square sets of kinds/colors plus a cursor interpreter. Neither proof predicates nor a shadow state supplied by the reference executes in the candidate. Native bounds are Unicode strings without embedded NUL, combined length<=512 and initial file0..9/rank0..8; source laws quantify over arbitrary Strings/Layout values.

The actual invalidity-revival mutation compiles/runs and fails row0/field25, observed1 rather than0. The actual character-discard mutation compiles/runs and fails row0/field23, observed0 rather than8. The latter would still satisfy some structural properties, which is why exact typed-transition behavior is separately proved and tested. These are deliberate corruption controls, not production defects.

## Supplementary raw-layout domain checks

Two additional closed facts were newly source-checked locally. A caller-supplied true validity flag does not itself certify the initial cursor: starting file at U32_MAX and parsing `18` wraps to0 then reaches8, allowing that raw rank to finish. Also, raw placement traversal does not clear a caller-supplied Board: an all-empty layout string preserves an existing pawn. The outer FEN entry point avoids these arbitrary-start examples by supplying its own empty Board and fixed cursor. Neither observation is a demonstrated production bug.

These facts delimit the next freshness theorem: it must follow the actual initialized parser trajectory rather than assume that the flag alone entails cursor bounds or an initially empty Board. They do not weaken any of the five accepted laws, and they are not added to registered counts or native coverage.

Exact source, SHA-256 `7ae1ce70f02b4a2e451c82d9e465a5bc229e472aba8c853a6ae0c9cb9ae3c638`:

```bend
# Supplementary raw-layout domain checks; not public parser contracts.
import Base
import ./Position.bend as P
import ../legal_probe/Chess.bend as Chess

# A caller-supplied true flag is not itself a certificate of cursor bounds.
# Max U32 plus digit1 wraps file to zero; the following8 completes this raw rank.
def raw_true_flag_does_not_imply_bounded_start() ->
  {P.layout_result(P.layout("18",P.Layout{P.empty(),4294967295,0,True{}})) == Some{P.empty()} : Maybe<&2,Chess.Board>}:
  {==}

# Placement traversal does not clear a caller-supplied Board. The outer FEN
# entry point avoids this by starting from its own empty Board.
def raw_layout_does_not_initialize_board() ->
  {P.layout_result(P.layout("8/8/8/8/8/8/8/8",P.Layout{P.put(0,True{},0,P.empty()),0,7,True{}})) ==
    Some{P.put(0,True{},0,P.empty())} : Maybe<&2,Chess.Board>}:
  {==}
```

Copy the block, with one trailing newline, to a disposable qualified checkout as `native/bend_engine/standalone/ParserDomainReview.bend` and check it with the pinned main.ts. The check completed with exact safe success in1.986seconds. Changing the first expected result to None rejects at `raw_true_flag_does_not_imply_bounded_start`; changing the second expected Board to empty rejects at `raw_layout_does_not_initialize_board`. Both status1 diagnostics are ordinary expected/observed failures, not crashes. Full logs/receipts are in the conversation review ZIP. Their SHA-256 values are `98c42090a711efbf9ea21a3120fb7ff7a6834957f1e288fd65091634a383fed3` and `d93456df108eb77dd082d024ed2ef862bb51e30821692176022bee906a0657c5` respectively. No native execution of this supplementary file is claimed.

## Failures retained and next acceptance

The original local lint command failed for missing tools; hosted Ruff/Basedpyright/Vulture resolves that environment gap without relabeling the old result. Several enclosing native command limits interrupted mutation/report completion after baseline modes had passed. The bounded supervised final rerun completed successfully in55.385seconds. Draft proof dispatch/match-order failures, an unintended mutation rejection location and a missing forward helper name were corrected before final acceptance. A trailing empty line in Actual.bend was removed before final local and hosted source rechecks; no term or domain changed. Original available logs and hashes remain in the review ZIP.

This final review adds documentation only. The qualified primary proof/test sources are unchanged; no redundant full suite or lint was run for it. Its supplementary source facts/diagnostics are labeled separately. Self-review only, not independent review; hash agreement establishes provenance, not a second reviewer.

The next decisive P3 obligation is bounded, nonoverlapping insertion addresses on actual accepted parser paths, connected to the existing fresh-insertion invariant. These prefix-safety results do not yet prove parser-wide freshness, partition preservation, full six-field FEN semantics, legal reachability or rollback of a raw rejected Layout. Metadata, king safety, move application and generation completeness remain separate.

No production function, earlier accepted law, compiler input, permanent workflow or routine perft budget changed. No search, model/GPU, training or benchmark run was added. No additional Python application responsibility moved into Bend; export/references/data/control/training and transitional C++/LibTorch/AOTI remain dependencies. Checker/Base, native lowering/storage, ABI, toolchain, libraries, OS and hardware remain trust boundaries. No merge, force push, deployment or live-process operation.
