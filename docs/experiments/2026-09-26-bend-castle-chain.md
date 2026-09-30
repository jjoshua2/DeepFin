# Castling provenance through the actual final generator

## Preregistered scope

Base PR #891 at b3bbfd5c38f8647cece0ce1d4d541e26fbb6e29f. Preserve all existing
source, proofs and compiler inputs. Add opt-in source laws linking actual ordinary
scan, both castle_side calls and the final filter to the existing guarded castling
update theorem. Do not assume a caller tail is empty or castling-free except where
derived from the actual initialized scan. Retain exact member values including
promotion/flag. No king-safety semantics, rights-history or legal completeness claim.

Acceptance: importing consumer with exactly safe success; intended proof/refinement
mutations; bounded current-pin native generator/reference probes; original compiler
source/pin gates and the unchanged repository lint command. No perft, model/GPU,
search, training or benchmark. One native compilation at a time; retain failures.
Self-review only. Publication and tests must be reported as actually completed.


## Completed local source and native checks

Local branch: `feat/bend-castle-chain-20260926`. Exact published parent is PR #891,
`b3bbfd5c38f8647cece0ce1d4d541e26fbb6e29f`, still unchanged at the final connector refresh. The compiler branch also
still resolves to `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`. No publishing action is exposed in this session;
direct Git access fails DNS. No new PR, hosted run, merge, deployment or force push
is claimed. The completed local commit, patch and prerequisite bundle are delivered
separately from this source-indexed record.

### Five public contracts

| Law | Actual guarantee and assumptions |
| --- | --- |
| `ordinary_scan_excludes_castling` | Actual scan from an empty list cannot produce flag 2, for arbitrary key lists, Board and affine tables. |
| `final_filter_preserves_member_origin` | Membership in the actual final-filter result entails membership in its input list. Covers both full and fast paths. |
| `filtered_castling_suffix_origin` | Membership after both actual castle sides and final filtering is either inherited from the arbitrary initial tail or has an exact guarded castling certificate. |
| `legal_castle_member_has_producer` | Actual full-generator membership plus flag 2 entails a kingside or queenside guard and exact complete Ply. No caller-tail-absence premise remains. |
| `legal_castle_member_preserves_representation` | The same member applied to an input-consistent Board preserves the partition. Initial rook-target freshness is derived. |

`Spec.castle` is an explicit disjunction of the two side certificates, each containing
the actual input guard and full Ply equality. The guard supplies the selected rights
bit, owned source-king/corner-rook bits and empty between path. The previously accepted
`castle_emission/Guard.consume` derives target freshness and invokes the public castling
update theorem. Input consistency is required only for the final preservation law.

### Actual composition, not an assumed empty tail

The proof follows `Chess.scan`, `castle_side(True)`, `castle_side(False)` and the actual
`filter_prepare` call in `Chess.legal_moves`. Scan destinations use promotion-zero or
Boolean-derived flags, so they cannot introduce flag 2. The two side calls preserve a
pointwise classification while attaching guards to new moves. Both full and fast filters
preserve that classification without inventing members. Structural list induction
extracts the certificate of a final flag-2 member.

Dependent proof continuations retain actual affine arrays as single-owned inputs
through every call. No oracle table, replacement generator or assumed final-list
classification is supplied. These results do not assert returned-array identity,
physical allocation or lifetime. The positive provenance facts hold even for arbitrary
source tables, so no semantic attack correctness is inferred from them.

The arbitrary-tail suffix law still permits old entries explicitly. The consumer checks
that a final filter alone can retain an arbitrary invalid preexisting castling move,
whereas the initialized actual generator does not inherit that tail. A small zero-table
example produces both routes, checks ordering and uses both public full-generator
contracts. That example demonstrates a satisfiable source domain, not legal chess or a
certified attack table; native tests use real current `Tables.build()` instead.

### Executed gates

The final focused gate completed with exit 0 in 61.767
seconds, with the consumer taking 8.276 seconds and exactly
`All terms check.`. All five laws and 18 controls ran in that command. Nine controls
require ordinary expected/observed diagnostics at the intended semantic/refinement
locations, eight enforce manifests/imports/no holes/unsafe/symlinks, and one is a
synthetic warning-output unit, not another compiler execution.

The actual implementation corruptions forge a castling flag during ordinary scan,
invent moves in fast or full filtering, bypass the castling guard, inject a forged scan
tail, or repeat kingside instead of queenside. Public-premise mutations omit membership,
flag 2 or input consistency. Each is rejected by the final source gate. Missing imports,
parser/affine errors, crashes and timeouts do not count as semantic rejection.

The native gate completed normally with exit 0 in 37.350
seconds. Generic, forced-portable, native-target and UBSan each pass:

- 1,062 distinct requests: 386 final-filter, 386 filtered-two-side suffix, and 290 actual
  full-generator requests;
- 9,241 complete returned child Boards / 175,579 U32 Board fields;
- seven malformed bounded batches rejected;
- four destination-only attacked castles rejected by the final generator;
- 39 full-generator cases retaining both castling choices.

The complete output contains 1,542 flag-2 and 7,699 other move records per mode. These
counts include arbitrary-tail filter/suffix operations and are not counts of distinct
legal chess moves. All four modes repeat the same fixtures. Output SHA-256:
`6dde6b9e3ad7b9ec0b4c27c4acb823bd06ec6c0d6bc2fb2d15cc0ecbecd02d10`.

The native candidate builds actual tables once per bounded batch, then executes actual
Position/Chess operations. The independent reference uses 64-square kind/color sets,
coordinate attacks, path clearance and a separately expressed filtering decision.
**The complete ordered move list is checked for filter/suffix requests. For full
`legal_moves`, only the ordered castling subset has an independent move-set oracle.**
Every returned move's full raw child Board is nevertheless compared, including other
moves and inherited-tail entries. This is not a proof or test qualification of the full
noncastling legal move set. Native Boards are consistent with exactly one king per
color; source laws do not require those extra test-domain conditions.

Three actual-code regressions compile and execute normally before independent output
rejection: a forged flag-2 scan tail, duplicate kingside stage, and disabling destination
rejection. The first two fail in the first 64-request batch. The last fails at batch
index 256. No crash or compile failure receives behavioral-rejection credit.

Original compiler source checks pass all 16 laws and seven controls, including the
cyclic-template regression, in 9.088 seconds.
All 12 compiler-pin tests pass with zero failures. Local compiler is Bun 1.4.2 and
Clang 17. The same unchanged 84-input Bend source pin is used everywhere.

### Repository lint and qualification limits

The unchanged whole-repository `./scripts/lint.sh` ran once after final primary source
construction and failed with exit 1: Ruff, Basedpyright and Vulture are absent. Its
original output and receipt are committed. No newly hosted lint pass or independent
review is claimed, and the parent's successful lint does not qualify these new files.

Exact-source parent166/487 is retained on its committed report, verified successful job
steps and all 511 unchanged native-source entries. The new completed 5/18 gate therefore
yields **modular171 laws/505 controls**, not a freshly executed full171-law wrapper.
All 524 final native-source entries and the 49-entry focused dependency closure are
verified. The native report's 12 code-file hashes still match final executable sources;
its run preceded addition of the README only. The final focused run follows the finished
native verifier and replaces the earlier report that included its placeholder hash.

### Retained development history

An initial filter proof draft was rejected for matching a computed result outside the
required binder order. Dependent continuations resolved actual-array ownership without
weakening the statement. A draft list contradiction used the wrong equality orientation;
it was corrected using existing word equality lemmas. Original failed drafts/logs remain
in the review package. The final completed focused/native gates, not those drafts, are
the accepted local evidence. The IO probe's empty direct invocation log is not counted
as a source theorem check; its generated native executable is tested separately.

### Remaining obligations and trust boundaries

A guarded producer route is not historical castling-right validity, independent metadata
correctness, legal reachability, or semantic start/transit/destination king safety.
The new provenance theorem would not by itself prove that an attack checker computes
chess attacks. Source list membership also does not establish multiplicity, ordering,
no duplicates or generation completeness; those stronger properties remain distinct.

Next decisive acceptance is to prove that generated castling takes the required king-
safety path through the optimized final filter, then connect actual check predicates to
independent attack semantics and metadata conditions. Whole legal-move soundness AND
completeness, not just castling provenance, remain P3 obligations.

Self-review only. The pinned checker/Base, native lowering/storage, ABI, C/C++ compiler,
libraries, OS and hardware remain trust boundaries. Existing strict-TypeScript,
snapshot-lowering and literal closed-builder limitations are unchanged. No production
runtime, earlier accepted law, compiler input, permanent workflow, routine perft,
search, model/GPU, training or benchmark changed. No additional Python application
responsibility moved into Bend; export, references, data/control/training and
transitional C++/LibTorch/AOTI remain dependencies.
