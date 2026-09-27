# Bend-owned history rules and automatic search draws

## Preregistration

Base: #799, `496284b233de9cc3727ddd4b79dc5720d6ed945e`.
Keep today's source-verified U64 compiler `fd1df81707fd758f749a9570ccb5b12b1bb2fea3`.
Question: can the standalone executable own history-based rule decisions, rather
than querying Python, while retaining its directly launched no-interpreter runtime?

Implement automatic fivefold/75-move and a conservative insufficient-material
subset in Bend. Use full piece/color/turn/castling identity and only legally
available en-passant rights. Reconstruct each selected search leaf from the
Bend-owned played history and actual tree ancestors; validate its full board.
Use the existing terminal-zero reply and cache. Mate/stalemate remain the native
search's first terminal checks. Do not force optional threefold/fifty-move claims.
No compiler, search-core, C transport, neural model or production changes.

Deciding checks, before running new runtime tests: compare current rule facts and
complete unchanged histories with external python-chess across clock boundaries,
repetition histories, lost rights, legal/pinned EP, material and terminal cases.
Test rules below root, cached revisits, identical-board/historyless controls,
invalid-command transactionality, rewinds, and unchanged perft semantics. Preserve
the original standalone verifier unmodified. Run generic/portable/native/UBSan and
static-only empty chroot; only external tests use Python. Do not claim exhaustive
dead-position solving, claim-choice migration, neural inference or performance.

Budget: isolated CPU development, at most two compiler processes, one runtime
thread, one bounded hosted confirmation (10 minutes). Additional runs only repair
concrete failures. Native tests are opt-in; no recurring perft-depth/test expansion.
No live process, training, GPU, merge or deployment. Recovery: discard isolated
branch. Self-review; independent review is not currently available.

Rules source: FIDE Laws articles 9.2.3 and 9.6 (and 5.2 for terminal outcomes),
https://handbook.fide.com/chapter/E012023 . The implemented material detector is
only a sufficient subset of the general dead-position rule.

## Implementation

The new Rules.bend (113 lines) and SearchHistory.bend (54 lines) contain the rule
and ancestry logic. Main's selected-leaf path now reconstructs the complete
Bend-owned played history plus the actual parent chain, verifies the reconstructed
full board, and computes the automatic rules before invoking the material evaluator.
No Python/CBoard or native C callback supplies a rule decision or leaf history.
Only existing native Base/IO infrastructure remains outside authored Bend logic.

Repetition uses full structural piece/color/turn/castling equality, not a hash.
An EP square contributes only if a generated legal EP move exists; a pinned EP
capture is excluded. Older EP states are normalized by the same Bend generator.
Count includes the current board and the reversible window in stored history.
The 512-played-ply protocol limit plus 32 search plies bounds the scan at 544;
missing pre-FEN history is never fabricated. Parent indices must decrease and be
in range; reconstruction rejects inconsistent history before a rule/evaluator reply.

The existing status-3 canonical draw reply (0/1/0, no policy) stores terminal zero
in the unchanged search. Later visits use that cache. Search still checks mate and
stalemate before asking for evaluation. Each newly adjudicated leaf emits one
rule_draw diagnostic; the read-only rules command returns reason/count/halfmove.
A drawn root with legal moves gets an explicit legal-protocol-fallback label, not
a claimed searched continuation. GUI outcome/claim handling remains separate.

The material rule is deliberately sufficient, not exhaustive: bare kings, one minor
piece in total, or bishops only on one color. Two knights and opposite-color
bishops are not forced draws. Claimable threefold/fifty-move positions continue
searching; optional claim-action choice is not yet migrated into this entry point.

## Local readout

The new verifier and the unchanged original standalone verifier pass on Clang17
with today's exact source-verified compiler, in generic, UBSan and static modes.
The static executable was tested in an empty chroot containing only deepfin-bend.
Python/python-chess clients execute outside it. Final module-invocation test also
passed there. All new rule reports are byte-identical across these local runs.
Focused Ruff passes. Basedpyright and the remaining portable/native C lanes were
confirmed in the hosted run below, not assumed from local execution.

Before native compilation, Bend's checker caught a destructuring-order issue in
new helper code; helper boundaries were corrected without changing the compiler.
Local Ruff identified two test-code style issues, corrected without suppressions.
No chess assertion, depth limit or numerical test was relaxed.

## Hosted readout: PASS

[Run 35516520704](https://github.com/jjoshua2/DeepFin/actions/runs/35516520704),
job **106093284584**, passed every stage, including exact source integrity,
Ruff, Basedpyright, 12 existing compiler-pin contracts, all five executable lanes
and clean source publication. Exact tested implementation commit:
`78bcc2d0ef03dc1cff8fa66be4a694637cb8c93c`, directly on #799's head. This subsequent
readout changes only this document. Published application/verifier blobs match the
locally checked sources. No development workflow or patch payload is in the PR.

Bun 1.4.2, Clang 18.1.3, Linux x86-64, one runtime thread, -O1 -ffp-contract=off
-Werror=shift-count-overflow. build.txt verifies **fd1df81707fd758f749a9570ccb5b12b1bb2fea3**,
Bend 2.0.20 plus U64, all 84 compiler/Base/effect files and fingerprint
`88f7505294c77f8187396aaeefd4d1845a6194d7e64b0ab9a982455bf2d8d38b`.
No compiler source or pin change was necessary. No Torch/model dependency is
installed or executed by this gate.

| Executed environment | Rule-position/history comparisons | New search cases | Original exact children | Original searched roots | Real UCI-client plies |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic C | 116 | 11 | 137 | 51 | 8 |
| Portable U64 helpers | 116 | 11 | 137 | 51 | 8 |
| Native CPU target | 116 | 11 | 137 | 51 | 8 |
| UndefinedBehaviorSanitizer C | 116 | 11 | 137 | 51 | 8 |
| Static, otherwise empty chroot | 116 | 11 | 137 | 51 | 8 |

These are the same deterministic fixtures repeated across modes, not disjoint
aggregate datasets. The new verifier compares full unchanged played boards/clocks/
histories and rule facts against python-chess, including 64 seeded positions,
clock boundaries, 0-5 repetition cycles, lost castling rights, both EP colors,
pinned-illegal EP, pawn/capture/EP/promotion resets, material subsets and mate/
stalemate. A 512-ply synthetic analysis history exercises the declared parser bound;
it is not presented as a legal game continuing beyond its automatic ending.

The 11 search cases include root draws, a fifth repetition below the root and its
identical-board/historyless control, capture into insufficient material, the
75-move boundary, cache behavior, no-forced-claim cases and mate precedence.
A drawn root completes 12 simulations with exactly one rule_draw notice. The
below-root fifth repetition appears at node 1 only with the actual prior history.
The engine's own history reconstruction also checks its full board against every
selected native ticket. This verifier checks rule decisions, result counts, legal
best moves and root history; it does not claim an independent complete-tree/value
trace comparison for the new standalone integration.

Perft semantics remain independent of draw rules. Every original suite returns
startpos3=8902, Kiwipete3=97862, endgame4=43238; repeated-draw startpos3 still gives
8902. The unchanged suite also rejects 23 invalid transactions per lane and checks
partial input/readiness, repeated stop and subsequent search recovery. All five
real UCI clients play the same legal eight material-evaluator plies. These are
not learned games or strength evidence, and no speed comparison is made.

The isolated ELF is static with no INTERP segment. The entire new runtime root is:

```
deepfin-bend
```

There is no Python/Bun, shell, dynamic loader, shared-library file, repository,
attack-table file or helper process in that root. The external client uses standard
pipes; the host kernel/stdio and statically linked native runtime remain. This
qualifies runtime dependencies, not sandbox security or a proof of correctness.

The initial hosted attempt stopped before compilation because Basedpyright found
an implicit sibling import in the external verifier. It was replaced with an
explicit relative import and documented module invocation; the corrected test and
README bytes were hash-checked in the successful run. No engine source, assertion,
rule, test case, compiler input or validation requirement was relaxed.

## Evidence

Artifact **bend-owned-draw-confirmation**, ID **10606908156**, 30-day retention.
ZIP SHA-256: `e1ea3d79f4e27ffe7edf972acbc0ed8861dbb9d1d09880675cbb6e178af60e56`.
All five new rule reports have the identical SHA-256 (also identical locally):
`002f1c059bc5d4caee872ecef691573d0126e1294a21d3301e22367b84105c1f`.
Original regression JSON identities:
- generic: `534a3f2cd4668e769676892e2ae5afe3fe228406cc8584a37a2e172cedc3b4ea`;
- portable: `20b092204db753c5daa508fac444b543b06c91cf5fe789cef8a7f2ac4cdb6e86`;
- native: `8ec428e60ad1b122f32b5117fa0b729449c8c72f266afe166ba7c9ea606884d4`;
- UBSan: `b0e8513d126132df3af11db0afd4163e1a70151d56c975f4063594d58497a192`;
- isolated: `2af7ef73f3870dfd3d21a455ca2620a112681dc335f0033057d1d02ba1b5fa66`.

Static executable: `20138d20acaa90c91f96417d69eea35dcbbce51bc342e24a0bae1304f7c939fe`.
Generated C: `5ead932529be930d3e50b5323bc6a69c42b81f98489ccc42f1336985b2b7911a`.
Original applied patch: `cda3986f47af92efc1e43194634125b11ebd19d8d277b3599eadcc5a015098d1`.
Final verifier: `c593dffe9f561af09305e4481dae980a97f077dc4229c6b6a5a4dcc0356d250c`.
The upload contains compact reports/build identities, not engine binaries or
compiler/model artifacts. Hashes identify tested bytes, not universal reproducible
build or cross-machine portability promises.

## Reproduce / limits / next decision

In an isolated checkout with a new output directory:

```sh
bash native/bend_engine/standalone/build.sh build/bend_owned_rules
./build/bend_owned_rules/deepfin-bend --threads 1

# External oracle only; never launched/linked by the engine:
python -m native.bend_engine.standalone.verify_rules \
  --report artifacts/bend-owned-rules.json \
  --command ./build/bend_owned_rules/deepfin-bend --threads 1
```

The engine build/run itself needs no Python. Native rules and the original external
verifier remain opt-in; no new permanent workflow, ordinary pytest search, deeper
perft, model export or benchmark is added. Search.bend, Chess.bend, poll.c, compiler
inputs, prior Python scaffolding and production paths remain untouched.

Optional claim choices, exhaustive dead positions, neural encoding/inference,
batched scheduling, training, subtree reuse and production Gumbel parity are not
established. History work has not been performance-measured. Cached terminals are
path-specific in a fresh tree; transpositions/reuse need their own history-safety
review. Resource exhaustion can precede rule evaluation and must not be called a
draw. Cooperative stop still does not preempt a pure computation or blocked output.

Self-reviewed, not independently reviewed or formally proven. Nothing merged or
deployed. The dedicated confirmation is green; the PR's wider checks are separate.
The next substantial migration is Bend-authored neural input encoding, reusing the
history reconstruction rather than bringing back a Python controller.
