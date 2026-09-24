# Numeric maps on actual CBoard position keys

## Scope and preregistration

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

## Readout

Pending hosted qualification. Fourteen pure producer/admission tests passed
locally without global conftest; the local environment lacks python-chess and
the compiled project extension, so the three engine-dependent tests are not
included in that local result.
