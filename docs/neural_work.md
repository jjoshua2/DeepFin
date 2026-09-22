# Neural work and fixed-budget comparisons

PR1's schema is `deepfin.neural-work.v1`. Its purpose is attribution, not a claim
of higher throughput or playing strength. Search simulations, network rows and
backend forward calls are different units. One batch can execute many rows; a
terminal/cache hit can complete a simulation with no network call.

## Counter contract

| Field | Meaning |
| --- | --- |
| `completed_simulations` | Completed search backups, not network requests. |
| `forward_calls` | Actual backend forward invocations, not batches merely gathered. |
| `dispatched_real_rows` | Non-padding input rows admitted to those forwards. |
| `executed_real_rows` | Real rows whose forward completion is confirmed. |
| `accepted_neural_rows` | Executed rows actually accepted into this search. |
| `padded_rows` | Extra physical rows dispatched, including unsuccessful forwards. |
| `failed_forward_rows` | Dispatched real rows whose forward did not confirm completion. Actual work within a failing/hung kernel is unknown. |
| `cancelled_rows`, `stale_rows`, `rejected_rows`, `failed_rows` | Mutually exclusive logical request dispositions in a per-search ledger; these can include requests rejected before dispatch. Not additive with physical counters. |
| `executed_wasted_rows` | Confirmed executed rows with a non-accepted final disposition. |
| `unresolved_rows` | Submitted rows without a logical final disposition. |
| `real_batch_histogram`, `physical_batch_histogram` | Dispatch frequencies by real and padded batch size, respectively. |
| `useful_eps`, `executed_eps` | Accepted/confirmed-real rows divided by the report's wall interval. Null when the interval is zero. |

Do not add logical outcomes to physical work totals. A cancelled request may
still run; it must not increase useful EPS, but it still consumes computation.
Backend internal warmup/autotuning kernels are not extra request rows. Bank cold
startup separately and exclude warmup searches in warmed comparisons.

`NeuralWorkLedger` is a thread-safe **bounded-run** observer/admission helper.
Request/batch IDs are unique for its entire run. It retains tombstones until the
run is discarded, rejects duplicate dispatch, and completes/resolves exactly
once. Admission reserves **real rows**, not calls or padding; failures and
cancellations do not refund that reservation. It does not implement an async
search scheduler. It is not suitable as an unbounded lifetime broker registry.

## Current integration and explicit limits

**Standalone Bend engine:** a single `info string neural_work {JSON}` is emitted
with the final result per finished search. An infinite search that finishes its
bounded work holds the report until `stop`; its wall interval excludes that idle
hold. The existing synchronous batch-one native product counts a
returned model forward as one executed row and counts acceptance only after
`Search.resume` increases completion. Capacity/reply rejection cannot increase
acceptance. Material evaluation and proven/terminal draws never count as neural
work. These are observations around the unchanged PUCT core, not Gumbel parity.

The native bridge is still CPU F32 batch one. Consequently real/physical batch
histograms have only size one, and padding is zero. A fatal backend/invalid-logit
error exits without a successful search report; the harness fails closed on that
absence. It does not invent the physical work done before the fatal error. Native
asynchronous cancelled/stale dispositions belong to the later lifecycle PR.

**Python per-trial `SlotBroker`:** `CAE_NEURAL_WORK=1` adds a JSON line to the
existing periodic broker reports. It observes eager, legal-policy, legal-row and
AOT dispatch paths after input/metadata preparation, at the actual forward call.
CUDA completion uses the already-existing synchronization; no new GPU sync is
introduced. CPU execution remains counted if subsequent output handling fails.
Clamped rows, malformed requests and a missing model use the actual gather
semantics, not client-claimed rows. Existing served counters are unchanged.

A broker cannot observe search acceptance. Its accepted/simulation/disposition
counts and useful EPS are **null**, not zero or delivered-row estimates. Its
explicit scope is cumulative broker lifetime, including startup and idle time;
that denominator is not interchangeable with a warmed per-search interval.
`SharedSlotBroker`, standalone Python evaluators and worker search acceptance are
not wired by this PR. No broad all-backend coverage is implied.

## Native controls and timing

`go evals N depth 32` stops on executed real neural rows, with N in 1..65536.
`go movetime MS depth 32` uses the existing monotonic deadline, 1..60000 ms.
Neither silently substitutes the old default 64-simulation budget. The existing
4096-node arena, depth limit and a 65536-simulation safety guard still apply;
any early stop/underfilled budget is reported and must not be scored as an equal
budget. Explicit `go nodes N` retains its existing 1..256 contract. A material
engine cannot satisfy a neural budget and returns zero neural rows.

Append `profile` for optional clock samples. By default per-phase values are
null. Counters and the final wall measurement remain enabled. Clock granularity
is one millisecond; fast phases may measure zero. Exact fixed-point formatting
avoids rounded integer counters or float wall timestamps. The internal search
interval excludes model startup and initial position/root legal validation. The
harness additionally records command-to-bestmove elapsed time.

The current backend has no separate H2D/GPU/D2H phases, so those values are null.
Selection and cached backup happen in one `Search.prepare` call and are reported
as `selection_and_cached_backup_seconds`, not falsely separated. Likewise
`backend_and_transport_seconds` includes flatten/marshaling and native model
execution, and `normalization_and_diagnostics_seconds` includes current per-leaf
notices. Encoding and neural reply backup are measured separately. History/rule
checks and other gaps mean the phases are not a partition of wall time.

Model calls, encoding and output are still nonpreemptible. A movetime request can
overshoot. This PR measures/rejects unfair comparisons; it does not solve stop
responsiveness or disable the inherited per-leaf diagnostics.

## Running a comparison

Prepare a JSON array of engines (argv arrays, no shell), with identical trusted
model/encoding declarations and each source revision. For example:

```json
[
  {"label":"baseline", "command":["/path/baseline","--threads","1"],
   "model_id":"checkpoint/package identity", "encoding_id":"v2_threats/root-legacy-meta/repfix",
   "source_revision":"baseline commit"},
  {"label":"candidate", "command":["/path/candidate","--threads","1"],
   "model_id":"checkpoint/package identity", "encoding_id":"v2_threats/root-legacy-meta/repfix",
   "source_revision":"candidate commit"}
]
```

Set `DEEPFIN_BEND_MODEL_PACKAGE` to the bound package (or use `/usr/bin/env` in
argv for different compatible artifacts). Declarations are required metadata,
not cryptographic verification of the executable's model. The native product
independently enforces its bound package identity. Use the same checkpoint,
encoding/history conventions, search settings, thread count and root order.

The corpus contains one full UCI `position startpos moves ...` or
`position fen ... moves ...` command per line. Prefer move histories rather
than FEN-only reconstructions for repetition-sensitive examples. Blank/comment
lines are ignored.

```sh
PYTHONPATH=. python scripts/bench_neural_search.py \
  --engines engines.json --positions positions.txt --output observations.jsonl \
  --evals 256 --movetime-ms 1000 --repeats 2 --wall-tolerance-ms 10
```

Each block starts a fresh engine and excludes one warmup search. Engine order
reverses on alternate blocks; each position is tested at both budgets. JSONL
preserves metadata, corpus hash, root identity, bestmove, all counters, internal
wall time, external decision time, budget shortfall/overrun and comparability.
Output creation is exclusive to protect existing observations.

The harness reconciles histogram rows/padding/calls, accepted/executed/dispatched
counts, EPS denominators and final simulation count. Missing/duplicate reports,
unknown schemas or contradictory fields are errors, never zero-cost successes.
At fixed rows it requires the full executed budget with no overshoot. At fixed
wall time it requires the requested interval and no more than the declared
wall-overrun tolerance (default 10 ms), for both the internal search interval and
the external command-to-bestmove interval. Slow go preparation or result output
cannot be hidden by the internal clock. Unresolved requests are not a completed
comparison; resolved accepted/wasted rows must reconcile with execution.
Early termination or unfair timing leaves
`comparable=false` and the CLI exits nonzero. Wasted rows still cost budget.

These observations are not Elo estimates. Use controlled paired games and
uncertainty for strength claims; do not pool correlated roots as independent
strength samples or mix throughput profiles with timing disabled/enabled.

## Validation entry points

Ordinary focused tests:

```sh
python -m pytest tests/test_neural_work.py tests/test_neural_search_benchmark.py \
  tests/test_broker_served_count.py tests/test_inference_slot_protocol.py
```

`work_probe.bend` is a small optional compiled observer check; compile it with
the same verified compiler as `build.sh`, then run:

```sh
PYTHONPATH=. python native/bend_engine/standalone/verify_work.py \
  --probe /path/work-probe --command /path/material-engine --threads 1
```

`verify_neural.py` also reconciles each search's counters against its actual
native forward trace, including zero-forward terminal searches. That verifier
requires the external model/oracle setup; merely adding its assertions is not
proof it has run. No model export, full native build or extra permanent workflow
is added to ordinary pytest.
