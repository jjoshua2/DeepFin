# Bend-authored application, no Python runtime

## Revised question and predeclared hosted gate

Owner clarified Bend everywhere, not permanent Python orchestration. Base #797:
3b6047ddfc08d9aed43da37aff8f2db218f84d98. Preserve the older Python scaffolding as
reference, but implement the standalone application in Bend. No production/live
changes, neural checkpoint, GPU, training, perft increase or deployment.

Acceptance: compile one directly launched diagnostic executable. UCI/FEN parsing,
played-history ownership, table initialization, evaluator and search/control
choices must live in Bend. Native exceptions are limited to raw IO/clock/sleep and
the compiler runtime, not chess/application decisions. A static executable must
pass the external tests in an otherwise empty chroot. Test tools may use Python
outside that root; no interpreter/packages/launcher/data can be inside it.

Check exact full board/clocks/history for 137 legal transitions from 10 roots,
40 seeded history extensions, 51 bounded searches, invalid transactional inputs,
partial lines, readiness/stop during outstanding analysis, repeated stop and
subsequent recovery. Use an independent python-chess UCI client for eight played
plies. Compare startpos perft3=8902, Kiwipete3=97862 and canonical endgame4=43238
(no deeper default). Also compare all 108160 used Bend-built attack-table words
with the existing C reference in a separately linked test executable. Run generic,
portable, native, UBSan and static-isolated variants. No performance threshold.

Budget: isolated Linux CPU, two concurrent compilation processes at most, native
threads=1, ten-minute hosted cap. No permanent new workflow or ordinary pytest
native execution; the verifier stays opt-in. Failure recovery is discarding this
isolated feature, not resetting user branches. Self-review only.

## Implementation and compiler regression

New standalone modules own ASCII token/checked-number parsing, structural FEN
validation, legal move replay, full played board/move history, clock updates,
command state, limits, material evaluator and uniform priors. Search/Chess remain
unchanged pure Bend modules. Tables.bend enumerates every occupancy and initializes
slider/leaper tables once, without a C callback or a table file. Only poll.c is
new handwritten runtime code: 54 lines of generic bounded nonblocking stdin packet
transport. Base provides output, monotonic time and sleep. No exec, CBoard, Python
or model worker call exists in the candidate. The C oracle is separately linked.
The five new application .bend modules comprise 782 lines including comments.

This work exposed U64.from_u32's codegen bug on a variable u32: emit_alias could
reuse that C local and then extract its high half using a width-32 shift. Fork
PR #3 explicitly casts to u64 before splitting. Before/after hosted regression:
https://github.com/jjoshua2/bend/actions/runs/35510038201. Old code fails with
-Werror=shift-count-overflow; fixed normalization/JS and four C modes return eight
correct sign/wrap-boundary values. Existing U64/array tests remain green. New
standalone pin d9b9bce9ce4c02583ca1a548cfd239b368c3fc77, 74 source/effect files,
combined fingerprint 9aad7eaa045f981f8d8ba444dfa807e69f8d769d8934d03854544eb090e3ba5a.
Other DeepFin probes keep their prior compiler pin and behavior. No kernel edit.

## Local readout

Clang17, Bun1.4.2, pinned compiler with the exact widening fix. Dynamic and UBSan
executables pass the full raw-client suite. The static build passes the same
suite plus the independent python-chess UCI client inside an empty chroot whose
only file is the engine. There is no dynamic interpreter segment or libpython
requirement. The test driver runs outside the root through standard pipes.

Per raw-client suite: 10 root fixtures, 137 exact legal transitions, 51 bounded
searches, 23 invalid position/go transactions, all three canonical perft counts,
overlong/NUL record rejection, partial-line readiness, one bestmove after repeated
stop, and new-root recovery. Eight real-client plies are legal and the engine keeps
running. They are material-evaluator moves, not learned-strength evidence.
All 108160 Bend/C table words compare byte-for-byte, serialized table SHA-256
37cf5ae16f1709ef3220c30f60fbd142bc32be6315016dd037a6a8e631c5a663.
Ruff passes the external verifier. No assertion/tolerance was relaxed.

Two implementation errors were caught before publication: the starting-position
black-pawn constant initially occupied the wrong rank (perft caught it), and
adding fallback labeling required marking a reused Bend number as copyable (the
checker rejected it). Both were corrected before the final native tests.

## Hosted readout: PASS

[Run 35511076293](https://github.com/jjoshua2/DeepFin/actions/runs/35511076293),
job 106078976739, passed every stage on the first hosted confirmation, including
source integrity, Ruff, Basedpyright, all runtime modes, table equality and clean
publication. Exact executable source commit:
`df9b171c378d4ba373d38cdf447dc82ab2b970e6`, directly on #797. This subsequent update
changes only this experiment record. The published application/build/test blobs
match the local files. No temporary workflow or patch payload is in the feature.

Toolchain: Bun 1.4.2, Clang 18.1.3, Linux x86-64, -O1 -ffp-contract=off,
-Werror=shift-count-overflow, one native runtime thread. No Torch or NumPy is
installed by this confirmation: Python and python-chess are external test tools.

| Executed build | Exact legal children | Raw-client searches | Invalid transactions | Real UCI-client plies | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Generic C | 137 | 51 | 23 | 8 | PASS |
| Portable U64 helpers | 137 | 51 | 23 | 8 | PASS |
| Native CPU target | 137 | 51 | 23 | 8 | PASS |
| UBSan | 137 | 51 | 23 | 8 | PASS |
| Static, empty chroot | 137 | 51 | 23 | 8 | PASS |

Counts are repeats of the same deterministic fixtures, not disjoint unique cases.
Every build returns perft counts 8902/97862/43238. Raw tests compare every retained
prior board, move and clock against python-chess, including 40 seeded history
extensions. Invalid late moves, malformed FEN/limits, overlong and NUL records do
not change the accepted root. Partial-line polling, isready while bounded analysis
is outstanding, busy-position rejection, repeated stop and subsequent search pass.
This does not establish preemption during a pure search step or a latency bound.
Observed stop responses were about 10ms on this workload; no speed claim is made.

The independent python-chess engine client plays the same legal eight plies in
all five environments: b1a3 a7a5 a1b1 a5a4 b1a1 a8a5 a1b1 a5b5. This is deliberately
simple material/uniform-policy chess, not a learned or strong engine.

The isolation step verifies there is no ELF INTERP segment, then copies only the
static executable into a newly created root. Its full inventory is:

```
deepfin-bend
```

The external verifier invokes `sudo chroot ROOT /deepfin-bend --threads 1` and
runs both test clients over pipes. There is no Python, Bun, shell, dynamic loader,
shared-library file, repository, checkpoint or attack data in that filesystem.
The standard runtime/library code is statically linked; the Linux kernel and open
stdio descriptors remain host-provided. This is a runtime-dependency demonstration,
not a security-isolation proof or a claim that no C runtime exists.

Finally, all 108160 used table entries produced by Tables.bend match the separately
linked C reference exactly, with the same SHA-256 as the local comparison. Neither
that C reference nor Python supplies an attack table to the deployed executable.

## Evidence

Artifact bend-owned-standalone-confirmation, ID 10605277072, 30-day retention.
ZIP SHA-256: 336bc450648f622ad3649d5b46a1f48d07069b1b8c57477b5371579b66a95493.
- Applied source patch: 310304478be342294a7dd98ed0854632e561bebcdd710aaf56df1c44feb2a1dd.
- Generic report: d3ca98d7b6b23e03c4404b970ec00f801e0c1fa9eea3ef9021b5b7f1d9444f5a.
- Portable report: 16dacbcae39df39bb632dd55a4a10383226f2e4305c205a1fd1ba19469e0c161.
- Native report: 19ff12ccd5641a5f86aee959ca9bcea2108ad12a98aab1301431b9a5bd7c3b26.
- UBSan report: 9345bf819faafbb889e7228fbb645653e207cef2c1940b6fced44eb87a17ebf0.
- Isolated report: 5fe4d8a9c183b2ef8fce9efe5c89789f766e6c2591882f43be792bf7054c2c6f.
- Static executable: 92670ad8e8c18c1d350a1d76879a483ac0b2e224fd5240d649726ab85aad1eab.
- Generated C: 46a88e9eec4289f29b0e094f5c612b2f0558a5719f03478bb9c0421adf9a7d74.

Only compact reports and build/commit identities were uploaded, not executables
or toolchains. Hashes identify the tested artifacts, not portable/reproducible
binary byte guarantees. Build recipes and ownership exceptions are in the README.

## Scope / next decision

The no-Python-runtime diagnostic milestone is narrower than a complete Python-free
DeepFin. History/clocks are owned by Bend but draw adjudication/claim options are
not ported into this entry point yet. Model encoding/inference/training and batched
scheduling are also absent. These remain Bend migration work, not permanent Python
exceptions. Old tests and scaffolding remain as independent migration references.

The UCI subset lacks clock/increment allocation, options, ponder/searchmoves and
continuous infinite search. Search is bounded to 256 simulations/4096 nodes/horizon32.
Stop is cooperative between steps; initialization, FEN replay, synchronous explicit
perft and stdout backpressure are not preemptible. Finite event-loop fuel and
bounded input/history avoid pretending this is a production daemon. No latency,
throughput, optimal play, full UCI or proof-of-correctness claim.

Run/build/ownership details: native/bend_engine/standalone/README.md. No permanent
workflow or recurring native test/benchmark is added. Existing core files and
perft depths are unchanged. Nothing is merged/deployed; independent review is not
claimed. The wider PR checks are separate from this focused hosted confirmation.
