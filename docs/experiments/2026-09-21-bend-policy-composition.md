# Bend policy mapping and complete-input composition

## Plan before execution

Base #802, 5032ba20d5c6e45630300d409a063de590f2ac21. Preserve current
Bend2.0.21 U64 compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae and its manifest.
An existing unpublished policy prototype at bf2d50ada2d0b3d718a987c4d24de7a010ce1ad2
was discovered on feat/bend-owned-policy. Reuse its mapping and verifier rather
than overwrite that branch or replay its older pin/Protocol changes onto #802.

Hypothesis: the Bend policy vocabulary can compose with the complete 146/175-plane
encoder without any runtime Python or C chess helper. Add a typed prepared-input
value pairing an explicitly shaped tensor with exact legal moves and both index
spaces, constructed from ONE validated Game. Keep search/material defaults unchanged.

Acceptance: exhaustive geometry/permutation tables match project Python mappings;
legal moves and all promotions map and resolve exactly with independent CBoard
cross-checks. Prepared requests must match the existing complete-input diagnostic
bit-for-bit and match legal policies from the same hypothetical descendant.
Same-board/different-history inputs must differ while policy IDs agree. Reject
malformed/late-illegal paths before output; preserve root/history and busy behavior.
No array capacity as model width, sentinel as move, geometry as legality, or missing
features as zeros. Explicitly test zero legal moves. No inference/softmax claim.

Budget: isolated CPU, at most two C compilers, one engine thread. One opt-in
15-minute hosted confirmation, retries only to correct findings. Generic, portable,
native, UBSan and static/empty-chroot runtime. Preserve old perft/UCI/rule/history/
complete-input verifiers and 12 pin tests; no recurring workflow or deeper perft.
Python and C oracles remain external. No GPU, training, checkpoint, production,
merge or deployment. Recovery discards the isolated new branch, preserving earlier
work. Self-review only; no independent reviewer is available in this session.

## Implementation and recovered work

Policy.bend, PolicyDiagnostic.bend and verify_policy.py are reused byte-for-byte
from bf2d50ada2d0b3d718a987c4d24de7a010ce1ad2. That earlier branch predates #802;
it remains untouched. Its older compiler manifest and command dispatcher are NOT
used. The mapping dependencies match the current source. Earlier run 35521725171
is background evidence, not qualification of this composition or compiler.

The new EvaluationInput.prepare accepts one already-validated Position.Game,
layout/version, attack table and generated Policy.Maps. It constructs exact legal
entries and complete 146/175-plane input from that same Game, retaining the map
and input/table owners in one Prepared value. The Game must come from Protocol
or SearchHistory; this is not a second arbitrary-FEN validator.

Policy maps are generated from geometry in Bend without a Python lookup file.
Private packed moves, full 4672 actions and compact 1858 slots are distinct spaces.
Encoding consumes generated legal moves. Reverse resolution searches the current
legal entries, preserving promotion/castling/EP flags rather than treating a
geometric slot as a legal move. Logical bounds precede Array access; padding,
sentinel and private-claim IDs cannot act as model slots.

The read-only policy and encode_request commands coexist with both existing
encoding diagnostics. A complete optional path (at most 32 plies) is checked
before output. A late illegal move emits no partial request and never updates
the root. Terminal boards may have empty legal lists, not fabricated moves. The
text format is diagnostic, not an inference ABI. Diagnostic maps are rebuilt per
command; the typed API retains their owner for a future persistent caller.

No compiler, build manifest, C effect, existing encoder/rule/search core, old
verifier or production change. Material search is unchanged. No actual search
ticket is routed to a model here, and no logits or softmax are used.

## Local checks and compilation memory

Focused Ruff and all 12 unchanged Bun pin contracts pass locally. The new
EvaluationInput and EvaluationDiagnostic modules pass Bend checking individually.
A duplicate Base type name was renamed Preparation before publication, without
suppressing the checker or changing the compiler.

The combined main could not be qualified locally within this container's 4 GiB
limit alongside its existing services: a build was OOM-killed and bounded
small-heap checks timed out. No unrelated service was stopped. The unmodified
baseline main check already used about 2.2 GiB here. These component checks are
NOT local native-executable qualification. The hosted run below supplies that.

## Hosted readout: PASS

[Run 35583416338](https://github.com/jjoshua2/DeepFin/actions/runs/35583416338),
job **106281171747**, passed every stage on the first combined confirmation:
source identities, external C reference, Ruff, Basedpyright, 12 pin contracts,
five executable environments, all inherited suites and clean source publication.
Exact tested implementation: **cda2745391d9cdaf636c2ae0aea7557751aa26c3**, directly
on #802. Subsequent usage/index and this readout change documentation only. The
clean feature contains no temporary workflow or patch file.

Compiler: **aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae**, Bend 2.0.21 plus U64,
84 verified inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Bun 1.4.2, Clang 18.1.3, Linux x86-64, one engine thread, flags
`-std=c11 -O1 -ffp-contract=off -Werror=shift-count-overflow`.
Python 3.13, chess, NumPy and the original C encoder are external test tools.
No Torch or model package is used.

| Environment | Legal moves, Python/C | Exact resolution checks | Paired requests, Python/C |
| --- | ---: | ---: | ---: |
| Generic C | 2963 | 2156 | 236 |
| Portable U64 | 2963 | 2156 | 236 |
| Native CPU | 2963 | 2156 | 236 |
| UBSan C | 2963 | 2156 | 236 |
| Static, empty chroot | 2963 | 2156 | 236 |

These repeat the same cases across environments, not disjoint datasets. Each
policy lane checks 173 positions, 663 rejected requests and 176 distinct promotion
moves spanning every file, forward/capture direction, piece and color. All 4672
full slots, 1858 compact slots, 4096 square pairs and mirror permutations match
the project reference: 17,156 scalar table values. Every legal full ID also matches
CBoard. Wrong/missing special flags, illegal moves, invalid geometry, padding and
private claim IDs are rejected. All five policy reports are byte-identical.

Each paired-request lane checks 236 requests, 2616 legal entries, eight empty
terminal requests and 60 bitwise comparisons with the separate encode_input
command. Input and legal IDs match external CBoard for every request. Python
matches exactly except the existing predeclared pawn-storm allowance on planes
173/174 (absolute 1.2e-7, relative 0); maximum difference is 5.960464477539063e-8.
No tolerance was added or broadened. Same-board/different-history requests have
identical legal policy entries but different history tensors. Nine malformed or
unsupported requests, busy rejection and reset checks pass. All five paired
reports are byte-identical.

Unchanged inherited suites pass in every environment:
- Full input: 714 Python / 706 C tensors, same ordered digest as #802.
- History: 505 Python / 501 C tensors, exact bits and same digest as #801.
- Rules: 116 position/history cases and 11 search cases.
- Original engine: 137 exact children, 51 searches, 23 invalid transactions,
  eight standard-client plies, readiness/stop/recovery and canonical perft counts
  8902 / 97862 / 43238 at unchanged depths 3 / 3 / 4. Draws do not prune perft.

The complete hosted build command, including source fetch/check and C compilation,
finished in **63.22 seconds**, maximum RSS **5,421,960 KiB (about 5.17 GiB)**,
exit status 0. This is a practical build-resource observation as application size
grows. It is NOT engine runtime memory, inference throughput, or a controlled
compiler-performance comparison. No runtime speedup follows from these checks.

## Interpreter-free evidence

The static ELF has no INTERP segment. The complete runtime filesystem is:

```
deepfin-bend
```

All six verifiers run externally over pipes to
`sudo chroot ROOT /deepfin-bend --threads 1`. No Python, Bun, shell, dynamic loader,
shared-library file, repository, generated policy/feature data or helper executable
exists inside that root. Host kernel/stdio and statically linked native runtime
remain. This is dependency evidence, not security isolation or a universal theorem.
No handwritten C policy or encoder implementation was added.

## Evidence

Artifact **bend-policy-composition-confirmation**, ID **10631573246**, 30-day retention.
ZIP SHA-256: `d103dd44dc826ede6dad1495e07b5f041ee20d16f517cfdcd0976c7999bb9247`.
Every policy JSON: `f49a209aa546daff21c0d796642bd1c2aed2882ef0a4119a05ed76a82c38a5eb`.
Policy table payload: `2f4c0524e2b7d3a2c81d6c53b770f97c07dc246dc2c776a937b1e6d05f82f618`.
Ordered legal entries: `95137e609a695f1b22f981b9cb5d00db3326cc17d13d966ed88e98f1a32516b1`.
Every paired JSON: `49e856b91ed0c56e5ad301054ee4c92128cda4c44112ea134b6cd53f25ff3746`.
Paired payloads: `b1ecdf36701cbbb636706c087cec32f5df9a82d1ecdc6722eb209d1194108c44`.
Build-resource report: `9243f92b8b8c76e5e7d5e58a282851456d6b63010529670c87b19b9213cc19c0`.
Static executable: `c14627a44d09f47ec76bfe634f8c56b2d744a860c9444303971037873332cca8`.
Generated C: `e8e112a182508dbfea6745173d8bc6baf545a273bf58b47a3c7ffd7ca1fb85a2`.
Only compact reports/build identities were uploaded, not binaries or model data.
Hashes identify tested bytes, not guaranteed reproducible builds or portability.

## Reproduction and remaining boundary

```sh
bash native/bend_engine/standalone/build.sh build/bend_policy
./build/bend_policy/deepfin-bend --threads 1
```

At idle, use `policy legal`, `policy encode e2e4`, `policy tables`, or
`encode_request lc0_root_legacy_meta v2_threats moves e2e4 e7e5`.
The README documents reverse decoding and both optional external verifiers.
Build and engine runtime need no Python. No new permanent workflow, recurring
native traversal, ordinary pytest execution, deeper perft or benchmark is added.
The one-off documentation publication does not rerun or change tested code.

Self-reviewed, not independently reviewed or formally proven. Review covered
orientation, promotion/special flags, bounds versus capacity, legal-list-only
decoding, map/table ownership, shared history, empty terminals and transactional
parsing. Prepared objects assume a validated Game and generated Maps; future FFI
consumers must not bypass that boundary. Diagnostic output is synchronous and can
block. No timing, strength or throughput promise.

Actual leaf-ticket integration, model forward, legal-logit normalization, response
validation, batching, GPU and training remain separate. This is a verified input
and policy component, not a neural-playing engine. Python references remain test
and migration tools, not a reintroduced controller. No merge, deployment or
production changes. Broader PR CI is separate from the focused confirmation.
