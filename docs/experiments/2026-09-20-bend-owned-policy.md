# Bend-owned policy boundary

## Predeclared scope

Base #801 (fe658a2680ec7941742c045feff0ccb806a898ad). Continue the
Bend-everywhere migration: generate policy maps and translate legal Bend moves
without NumPy, Python, CBoard or generated table files in the engine.

The fork advanced to 806b373a7a4509479054da74a82271c6a4c25014 on September 20,
adding U64 reference laws. Adopt its verified source fingerprint separately
from application changes; keep native compiler/kernel bytes unchanged. Run the
existing standalone checks on that pin before adding the new policy source.

Acceptance: exact complete full/compact/mirror lookup equality; legal move
indices and reverse resolution for both colors, castling, EP and every promotion;
no fallback for a nonlegal or out-of-range index/key. All metadata validation
precedes diagnostic output. Same-board/different-history positions must have
identical policy mappings, unlike their history tensors. Read-only commands
preserve the complete played history, reject during active search, and work in
a static-only empty runtime. Preserve unchanged encoding/rules/UCI/perft tests.

This does not add missing classical input planes, normalize neural logits, run a
model, or change material search. Claims and null moves are outside model policy.
No C runtime addition, proof of compiler correctness, speedup or strength claim.

Budget: isolated CPU, at most two compiler processes, one engine thread; one
ten-minute hosted confirmation with reruns only for concrete failures. New native
checks remain opt-in; no perft increases or new recurring CI workload. Nothing
is merged, deployed, trained or changed in live production. Recovery is dropping
the isolated branch. Self-review only unless an independent reviewer is obtained.

## Implementation

Policy.bend constructs the compact vocabulary by enumerating all 4672 full
geometry slots in order. It owns full-to-compact, compact-to-full and square-pair
lookup arrays, with logical bounds distinct from physical capacity. Mirroring
flips files in oriented coordinates; Black's perspective separately flips ranks.
Normal/queen-promotion rays and dedicated N/B/R promotion slots remain distinct.
Policy.Space is an explicit sum type. Resolution uses the current generated legal
entries, retaining the original Ply flags, not a numeric-ID fallback move.

PolicyDiagnostic exposes read-only table, legal-list, UCI/key and full/compact
lookup commands. It validates command shape and prepares all legal entries before
emitting a response. A nonexistent legal match is an error; null/claim identifiers
are never converted into real moves. It builds disposable maps for each diagnostic;
this is not an optimized inference hot path or a model integration. The reusable
typed map owner may be retained by a future Bend inference controller.

No authored C/effects, model/search/evaluator, historical encoding or rule code
was changed. Default material search does not call the policy diagnostic. Native
array lookup and code generation still belong to the trusted compiler boundary.

## Local observations

Before adding policy code, the unchanged standalone verifier passed on the new
806b373... pin: 137 exact legal children, 51 searches, 23 invalid transactions,
canonical perft counts and eight legal UCI-client plies. Comparing both compiler
snapshots found only bend2/base.bend changed under bend2: native compiler, checker
implementation and effect files are byte-identical. The source-law runner
u64_proofs.js passed, including its negative controls. This is evidence for the
fork's source laws, not a proof of the new policy adapter.

The initial native policy run passed all 17,156 scalar table values, 2963 legal
move comparisons over 173 position fixtures, 2156 exact UCI/key/full/compact
resolution checks, 663 rejected requests and all 176 distinct promotion choices
(8 files, legal forward/capture directions, 4 pieces, 2 colors). That preliminary
run had no C reference. The final implementation, including Policy.Space, passed
those same checks with the actual CBoard reference in generic, UBSan and static
empty-chroot executions on Clang 17. Their policy JSON reports are byte-identical.
The final generic executable also passes all unchanged encoding, rule and original
standalone checks. Ruff and the 12 existing compiler-pin contracts pass locally.
Basedpyright and all five runtime environments were then checked in hosted CI.

The Bend checker rejected computed tuple destructuring, a missing copy annotation
and an unsupported forward reference during implementation. Helper boundaries and
explicit ownership fixed those without compiler changes. A raw integer space tag
was replaced with Policy.Space before final compilation. Ruff findings in the
external test were corrected without relaxing assertions. The existing pin test
was updated to assert the new exact revision/fingerprint, not bypassed.

## Hosted readout: PASS

[Run 35521725171](https://github.com/jjoshua2/DeepFin/actions/runs/35521725171),
job **106106848892**, passed every stage on the first confirmation attempt:
source integrity, static checks, compiler laws, all five runtime environments,
unchanged regression suites and clean source publication. Exact tested executable
source commit: **0dac6336fd764bc5c64711a4bba13a35b6a1e5bb**, directly on #801.
This later follow-up changes only this experiment record. The feature branch
contains no temporary workflow or patch transport. The applied patch SHA-256 was
checked before tests; published Policy/PolicyDiagnostic/Protocol blobs match the
locally tested source. Generated C also has the same hash locally and in CI.

Toolchain: Bun 1.4.2, Clang 18.1.3, Linux x86-64, one runtime thread,
-std=c11 -O1 -ffp-contract=off -Werror=shift-count-overflow. External tests use
NumPy 2.2.6 and chess 1.11.2. No Torch, model package, weights or inference process
is installed/executed by this confirmation. The build verifies the new U64 pin
**806b373a7a4509479054da74a82271c6a4c25014**, 84 source files and fingerprint
**5c8949a6d84f3365108e7182e70e3fc1bb8d378ff7a9db884d0a3d4611c62135**.
The fork's source-law runner separately passes 16 laws, its importing application
and standard fixture, plus six negative proof controls. No new native oracle
campaign or formal proof of this application is inferred from that source gate.

| Executable | Scalar map values | Legal moves vs Python and C | Exact reverse/encode checks | Rejected requests | Promotion variants |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic C | 17156 | 2963 | 2156 | 663 | 176 |
| Portable U64 helpers | 17156 | 2963 | 2156 | 663 | 176 |
| Native CPU target | 17156 | 2963 | 2156 | 663 | 176 |
| UndefinedBehaviorSanitizer | 17156 | 2963 | 2156 | 663 | 176 |
| Static, otherwise empty runtime | 17156 | 2963 | 2156 | 663 | 176 |

These are the same fixtures repeated across builds, not disjoint datasets.
All five policy reports are byte-identical, also identical to the three final
local reports. The table test checks ALL 4672 full slots and their mirror map,
ALL 1858 compact slots and their inverse/mirror map, and ALL 4096 square-pair
entries, including invalid geometry. Those arrays match the existing Python
reference exactly; no numerical tolerance or sampled-only map claim. CBoard
independently confirms every legal full ID in 173 position fixtures (2963 moves).
It does not supply the candidate's maps or decide a candidate move.

Round trips cover both turn perspectives, both castling colors, EP, and every
legal promotion geometry for all files and four promotion pieces. Wrong/missing
special flags, absent promotion, null/claim identifiers, nonlegal geometric slots,
negative/overflow IDs, physical-capacity-but-outside-logical-shape IDs, and malformed
commands are rejected. No fallback move is substituted. Entire root/history
snapshots remain unchanged; busy commands and reset behavior also pass. Different
move orders reaching the same board have identical policy IDs, deliberately
contrasting the earlier history-encoding requirement.

The unchanged inherited suites pass in every lane:
- 112-plane history: 505 complete Python and 501 complete C bitwise tensor
  comparisons; four explicitly Python-only clock cases beyond CBoard's range.
  The ordered tensor hash stays cb7a7e6688a73c5bfbf734769cd6c5f1d3a04609167bf019a4cbfa79d5eb6bbf.
- Draw rules/history: 116 position facts and 11 search cases.
- Standalone engine: 137 exact legal children, 51 searches, 23 rejected transactions
  and eight independent UCI-client plies; same legal material-evaluator moves.
- Perft: 8902 / 97862 / 43238, unchanged; draw rules do not prune perft.

Ruff, Basedpyright and all 12 existing compiler-pin contracts pass. There were no
hosted source fixes, suppressed checks or changed acceptance thresholds. This is
not a trained-game, engine-speed, GPU or production-Gumbel result. The old material
search/evaluator remains the ordinary engine behavior.

## Interpreter-free runtime

The static ELF has no INTERP segment. The full filesystem inventory in the newly
created runtime root is exactly:

```
deepfin-bend
```

All four external verifiers communicate via pipes through sudo chroot. No Python,
Bun, shell, dynamic loader, shared-library file, repository, table file or helper
executable is present inside the engine's root. Python and the existing CBoard
extension run only outside as test references. Host kernel/stdio and statically
linked native runtime remain. This is dependency evidence, not a security sandbox
or universal correctness proof. No handwritten C/effect was added to the engine.

## Evidence

Artifact **bend-owned-policy-confirmation**, ID **10608592431**, 30-day retention.
ZIP SHA-256: cff28054333f189d30256e822aa9dfde3147367d1fec9972967d90497759a0d7.
All five policy reports: f49a209aa546daff21c0d796642bd1c2aed2882ef0a4119a05ed76a82c38a5eb.
Complete map output: 2f4c0524e2b7d3a2c81d6c53b770f97c07dc246dc2c776a937b1e6d05f82f618.
Ordered legal results: 95137e609a695f1b22f981b9cb5d00db3326cc17d13d966ed88e98f1a32516b1.
All five history reports: 72b4466260668b3ed5b05879720e49b1a5000cf4dd4fde4fb0e933760a810bff.
All five rule reports: 002f1c059bc5d4caee872ecef691573d0126e1294a21d3301e22367b84105c1f.
Source-law report: 1182b4ad3aa3823168e33c2e538e9c40b849454092bc85ebe68883c2b2a52bc7.
Static executable: 0bb66e862ac3779751d3f48ea466297d470d652b4d6a8dc132f14aa2eb1484b5.
Generated C: 1a91b81338f1259685e9f26649c6b7e346320fbceb77e524c1d74a28462f575e.
Applied source patch: d99b8384b1d6bd99fac931cf677a1bb00cc52e2117cbc9c9ad7f69c13a92bfe7.

Only compact reports and build/commit identities were uploaded, not executables,
compiler archives, model files or full table dumps. Hashes identify tested bytes,
not guaranteed binary reproducibility or cross-host machine-code portability.

## Reproduce / remaining scope

```sh
bash native/bend_engine/standalone/build.sh build/bend_owned_policy
./build/bend_owned_policy/deepfin-bend --threads 1
```

At the initial root, policy encode e2e4 prints key 1804, full ID 877 and compact
slot 304. policy decode lc0_1858 304 returns the same exact legal move. Use policy
legal or policy tables for inspection. The README documents the external oracle
command. Build and engine runtime need no Python; tests may use it externally.

This implements a reusable policy component and diagnostics, not an end-to-end
neural controller. Maps are currently built per diagnostic request; retaining them
and connecting logit gathering/normalization require later integration. Missing
34 v1 / 63 v2_threats classical planes remain unimplemented and are not zero-filled.
Neural execution, batching, training, optional claim choices in this entry point,
subtree reuse and throughput remain separate Bend migration work. Read-only text
output is synchronous and can block under stdout backpressure. No timing guarantee.

Ten intended files, no new permanent workflow, no ordinary pytest native traversal,
no deeper perft, no recurring native policy run or benchmark, and no production
modification. Existing source-law changes are adopted through the compiler pin;
no compiler code was changed here. Self-reviewed, not independently reviewed or
formally proven. Nothing merged/deployed. Broader PR checks remain separate from
this complete focused confirmation. Next: version-specific classical feature
planes, then a complete Bend-owned input/output boundary for native inference.
