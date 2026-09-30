# Castling cannot use the final-filter shortcut

This opt-in source suite proves that an actual generated castling move is forced
through the current side's real child-board check. It reuses `castle_chain`'s full
producer provenance, not an assumed empty input tail or a replacement generator.

## Public contracts

| Law | Guarantee and conditions |
| --- | --- |
| `owned_king_source_requires_check` | An owned king at a source below 64 requires checking for every ray mask, destination, promotion and raw flag. No whole-Board consistency premise. |
| `blocker_stage_uses_sensitive_mask` | Exact complete-pair refinement of actual `filter_blockers` to `own & (rays | kings)` and actual fast filtering. |
| `generated_castle_requires_check` | Actual `legal_moves` membership and flag 2 imply the mandatory check for every ray mask. No independent Board-consistency, source-bound or target-freshness assumption. |
| `generated_castle_takes_checked_step` | The actual fast step equals the complete pair returned by actual `make_move`, `in_check` for the **input** side, and `filter_after`. Generation and filtering tables are separate affine inputs, not silently duplicated. |
| `positive_check_rejects_current_move` | A true check result leaves the complete actual table-and-accumulator pair unchanged. An equal move already in the accumulator is not removed. |
| `complete_generator_equals_forced_castle_checks` | Actual `Chess.legal_moves` equals the audit-only `Forced.legal_moves`, which explicitly ORs flag 2 into mandatory checking. Includes complete returned array and ordered move list, for any actual Board and input array. |

The sixth law lifts the branch result through the actual ordinary scan, both
castling-side calls and final full/fast filter. The proof derives each candidate's
ordinary-or-guarded-castling classification using the earlier checked producers.
`Forced` is a comparison implementation, not a production engine change. It reuses
actual attack/check/metadata operations and alters only the mandatory-check choice.

## Commands

Use the unchanged compiler pin recorded by `standalone/toolchain.json`. Run from
the repository root with Bun available and one native compiler invocation at a time:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/castle_safety/consumer.bend
bun native/bend_engine/standalone/proofs/castle_safety/focused.js \
  /path/to/pinned/bend --report /tmp/castle-safety-focused.json
CC=clang bun native/bend_engine/standalone/proofs/castle_safety/verify_native.js \
  /path/to/pinned/bend --report /tmp/castle-safety-native.json
CC=clang bun native/bend_engine/standalone/proofs/castle_safety/verify_forced.js \
  /path/to/pinned/bend --report /tmp/castle-safety-forced.json
```

The focused gate runs the importing consumer and 19 controls: ten intended
source/refinement failures, eight import/manifest checks, and a synthetic warning
output check that is not another compiler execution. Successful source checks must
return zero and exactly `All terms check.`. Missing dependencies, syntax/ownership
errors, crashes and timeouts are not accepted as semantic rejections.

`verify_native` builds actual current tables, probes the actual required-check
mask, filter decision and child-side check, and compares complete ordered lists
and all returned child-Board fields to an external square-set/coordinate reference.
Inputs contain raw data, never expected answers. Squares are bounded to 0..63.
Raw nonking or unowned-source examples demonstrate that an arbitrary flag-2 value
alone need not require checking; the initialized producer's guard matters.

`verify_forced` separately compiles the unchanged parent chain probe and the audit
`generator_probe`. It compares their **entire ordered output**, including all
noncastling moves and child Boards. The inherited independent move-set oracle is
still narrower: exact lists for direct filter/two-side cases and only the castling
subset for full generation. Full noncastling equality between two programs is not
independent validation of noncastling legality/completeness. The audit-only mutant
that disables required checking is distinctly labeled, not called a production bug.

Both native drivers run generic, forced-portable, native-target and UBSan builds.
Modes and paired programs repeat fixtures, not disjoint or exhaustive legal games.
The native transcript observes Board data, not complete table contents or lifetimes.
The formal complete-pair equality does not state that the returned table equals the
original table; it relates the two actual-array computations.

## Limits

The claimed property is that generated castling cannot bypass the existing final
check, **not that the shared `in_check` implementation computes independent chess
attack semantics correctly**. Rights history, legal metadata, king counts,
reachability, full legal-move soundness/completeness and duplicate freedom remain
separate. Raw Board metadata and source arrays remain arbitrary in the source laws.
The consumer's small zero-table satisfiability examples are not correct-table or
legal-chess certificates. No source premise assumes a desired check result.

Pinned checker/Base, native lowering/storage, ABI, C compiler, OS and hardware
remain trust boundaries. Existing compiler, closed-builder and snapshot-lowering
limitations are unchanged. No production function, older accepted proof, permanent
workflow, routine perft budget, model/GPU, search, training or benchmark changed.
