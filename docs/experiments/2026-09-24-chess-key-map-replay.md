# Numeric maps on actual CBoard position keys

**Final qualified scope:** eight bounded legal walks per seed, at most 64 plies
each, with 2,048 buckets for encounter replay. The original plan and its fixed-
budget amendment are retained below. This is correctness qualification, not a
performance result or a production cache integration.

## Original scope and preregistration

Parent: #876 at `4364c213561e417386c6f2dab229e0b63d42075b`. The repaired
explicit CPU target's checks passed. The old synthetic-key benchmarks do not
establish performance or correctness at a real cache boundary. This increment
checks source-derived chess keys and documents that boundary, without changing
any map implementation or adopting a cache.

Generate four deterministic legal walks of up to 64 plies from each of four
fixed FENs: opening, castling-rich middlegame, capturable en passant and promotion.
Python-chess generates sorted legal moves; the seeded selection is not an engine
policy. Every observed key is obtained from the checkout's actual compiled
`CBoard.from_board(...).transposition_key`, never a substitute Python hash or
Polyglot key. Preserve each root, walk ID, move path, FEN and full key in the raw
report. Record producer-source and extension-binary hashes and python-chess
version. If the extension resolves outside the inspected checkout, fail.

Each corpus must supply at least 128 distinct keys within at most 260 records;
there is no unbounded retry to obtain a desired distribution. First-encounter
order selects 64 stored keys and 64 genuinely absent keys without conditioning
on the map's bucket assignment. Use the existing 256-operation hit/miss/churn
cycles and same retained slot budget for both implementations. Operations are
still a synthetic schedule; these are not production access logs, selfplay games,
complete search trees, or an evaluation-cache workload.

In addition, replay every observed position in encounter order as get/put/get,
with a stable numeric ID per unique key. A separate 1,024-bucket map permits all
bounded records without eviction. Repeated roots/positions exercise existing
hits and replacements, while the original synthetic fixtures still cover full
rejection and deletion. Both native implementations must match every returned
value/disposition/size against the independent dictionary oracle. Retain all
32 old cases plus four new replay cases in explicit BMI2/POPCNT and UBSan builds.
Twelve new operation cases at zero/one/three cycles give 144 driver executions.
No timer threshold or performance verdict is part of this qualification.

## Identity boundaries

The existing position graph verifies canonical piece/occupancy, turn, castling
and conservative pseudo-legal en-passant state after a hash match. The numeric
map only compares U64 keys. The fixture producer rejects a repeated key attached
to different canonical fields rather than merging it or pretending the key
alone is an exact board identity. An injected collision tests this refusal.
This producer guard is not a production collision-handling implementation.

Six native-key relationships are checked: a reversible knight cycle; a changed
halfmove clock; capturable, noncapturable and pinned en passant; and changed
castling rights. Same structural key does NOT authorize copying a history-
sensitive neural value. Pinned EP is intentionally a conservative split, not
claimed identical to the legality-exact repetition key. Replacing the producer's
transposition key with its raw persisted position hash must fail the capturable-
EP check. Raw history paths are retained; no neural encoding or model is run.

## Validation and recovery

Run the existing synthetic driver qualification and all prior map Python tests,
then the new `--chess` qualification without `--measure`. The complete hosted
check includes actual extension construction in the locked CPU environment,
source/compiler verification, all original four-mode map/mutation checks,
explicit static checks, and repository-wide Ruff/Basedpyright/Vulture. Use one
compiler job, two Torch threads and no GPU/live process. Each child retains the
existing 60-second timeout; the hosted workflow has a 12-minute cap. This is
correctness qualification, not a new speed experiment. Failures stay recorded.

```sh
python -m pytest tests/test_bend_map_chess.py
python -m native.bend_engine.u64_map_probe.benchmark \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-chess-map --chess
```

The producer runs only with `--chess`; existing synthetic workloads and their
historical result identities are unchanged. `--chess --measure` is available
for a separately budgeted timing experiment, but is not used here. There is no
comparison to a stronger map in this increment and no change to production
cache keys, eviction, models, compiler, defaults, or running training. Self-review
only; no independent review or formal proof. No merge or deployment.

## Initial local checks

Fourteen pure producer/admission tests passed locally without global conftest;
the local environment lacks python-chess and the compiled project extension,
so the three engine-dependent tests were not included in that local result.
The full hosted result follows below.

### Preserved first static failure

Run 36032177266 passed source checks, the locked build and Ruff, then Basedpyright rejected the circular import between the benchmark and the new chess fixture producer. No hosted Python or native replay had run. The unchanged frozen Workload schema now lives in workload.py and both producers import it, rather than suppressing the cycle diagnostic. The existing benchmark still exposes Workload through its import. Fixtures, native implementations, oracles and expected results are unchanged. Fourteen pure tests passed locally after this refactor; full hosted qualification remained a separate gate.

### Fixed corpus-budget amendment

Run 36032729272 passed Ruff and Basedpyright, and 101 of 102 Python cases. All six native identity relationships and the raw-hash substitution negative passed. The corpus reproducibility case rejected the promotion seed because four bounded random legal walks did not supply the required 128 distinct keys. No native map replay or performance measurement ran. This is a fixture-coverage failure, not an observed map mismatch.

The fixed budget is now eight walks per seed, still capped at 64 plies each, for at most 520 observations per corpus. All four seeds use the same expanded budget and deterministic RNG sequence; no keys or seeds are selected by hash bucket. The minimum 128 distinct-key requirement, 64-entry operation workloads and all equality/identity expectations are unchanged. The encounter replay now reserves 2,048 buckets / 1,024 entries so every possible observed key fits without eviction. Oversize/replay tests use the corresponding 520-record bound. This supersedes the original four-walk/1,024-bucket plan above and was recorded before native replay. No performance result was rerolled.

Failure artifact 10823596752 retains the 102-case JUnit report (101 passed, one failed), ZIP SHA-256 dbf0b580d095ff914e471a364776ddd035443d81ce3505640d876c55c9d08f43. Fourteen pure tests passed locally after the bound amendment; engine-dependent tests remained a hosted gate.

## Completed hosted qualification

Run https://github.com/jjoshua2/DeepFin/actions/runs/36033116416, job 107746508557, completed every stage successfully. All 102 Python cases passed without skips (17 new), including native-key identity checks, rejection of the raw-hash substitution and full reconstruction of every recorded legal path. The original four-mode map, three executed mutations, synthetic comparison and whole-repository Ruff/Basedpyright/Vulture passed.

Both maps passed 36 exact dictionary traces / 12,061 operations per explicit-target/UBSan build: 32 prior cases plus four chess encounter streams. All 144 new operation-driver checks passed, in addition to 180 unchanged synthetic driver checks. Build modes reuse fixtures, not independent games. No performance panel ran.

| Corpus | Position observations | Distinct keys | Revisits |
|---|---:|---:|---:|
| opening | 520 | 513 | 7 |
| castling | 483 | 476 | 7 |
| ep | 341 | 328 | 13 |
| promotion | 329 | 317 | 12 |

The 1,673 recorded observations contain 1,634 distinct keys overall. Their
encounter replay adds 5,019 get/put/get operations. This is generated legal-
position coverage, not a representative production lookup-frequency estimate.

The raw report retains every FEN/move path/key plus all operation definitions and checks. Compact source, extension, corpus and expected-trace hashes are committed under evidence/chess-key-map-replay/. The tested-source manifest describes preregistration before the documentation-only readout. Same-key/different-history examples and the intentional raw-hash/collision negatives enforce test boundaries; they do not implement safe neural caching. No engine, map algorithm, cache key, compiler, model, live setting, merge or deployment changed. Self-review only.

### Evidence verification and persistent CI

Successful artifact: 10822733454, `chess-key-map-replay`; ZIP SHA-256
`23dfe29cb49d6ef7c83d382e9e436fbcea6824a4bf5c746e0d9ce372f5c8d7d4`.
Tested staging commit: `b6cb9d568d58dbe70505b4f2ad0e0dea6ad3d7ad`.
Published source-qualified commit: `340e26a621d4f4757d3945b8dd18146ee003a2ad`.

The downloaded artifact was rechecked outside the runner: all 102 JUnit cases,
128 original-map traces, 128 synthetic comparison traces, 144 chess comparison
traces, all 324 driver output/input hashes and parsed states, corpus record
hashes and replay construction matched. Captured stderr was empty. The complete
raw artifact has finite retention; committed generator code and source/corpus
hashes preserve the reproducible workload definition and compact evidence.

The subsequent workflow-only wiring and this documentation cleanup do not
change the qualified Python/Bend sources. Persistent numeric-map CI now invokes
`--chess` without `--measure`, includes all map tests, and triggers on the relevant
encoding and graph-identity sources. It uses read-only permissions and the same
locked CPU/compiler setup. New-head PR CI is separate from the dedicated run;
no historical artifact is an input to future correctness checks.
