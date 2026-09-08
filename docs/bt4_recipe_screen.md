# B100/H20 arena package

`scripts/bt4_recipe_screen.py` runs the two registered development comparisons
through the existing owned-stage supervisor. Training is separate: both final
checkpoint identities and completed training/schedule receipts must already
qualify. The launcher does not retry, resume, select another recipe or promote a
checkpoint. It preserves historical-control limitations.

| Stage | Allocation | Execution |
| --- | --- | --- |
| Low | 100 simulations; 500-pair cap; ordered SPRT 0/+15, alpha .05, beta .10, first128 then64 | Rolling256, evaluator4096 |
| High | 400 simulations; fixed first128 pairs | Rolling128, evaluator4096 |

A valid low H0, H1 or inconclusive result always proceeds to high. An operationally
invalid low halts the package and preserves its raw bank, terminal output and
reader failure. The high reader recertifies the pinned low manifest and reports
the registered exploratory first128 aligned-pair percentile bootstrap: 10,000
PCG64 resamples, seed20260903. Stopped low intervals remain descriptive.

## Frozen inputs

The schema1 manifest has `profile:"B100_H20"`, a new absolute `output` directory,
and these fields:

- `candidate` / `reference`: `{role,path,sha256}`, respectively B100 and the qualified
  H20 checkpoint. `candidate_training` / `reference_training` pin existing complete
  training receipts; their run summary, checkpoint and canonical schedule must agree.
- `runtime`: the compact qualified old-CUDA overlay identity. The supported image
  is the reviewed two-file ordered-SPRT overlay at commit `1b8e5036290789ec3a037339af0150e9f6fa76d8`.
  It retains Python3.10.12, Torch2.11/CUDA12.8, NumPy1.26.2 and the original native
  payloads. Its location is explicit, rather than changing shared launcher globals.
- `book`, `live_config`, `preregistration`: exact path/SHA pins. The original PGN
  development book and explicit overlay `configs/pbt2_small.yaml` are required.
- `launcher_sha256`, `reader_sha256`, `supervisor_sha256`: frozen source identities.
- `preparation`: pin of the CPU preparation receipt, added after the first command
  below. Omit it from the prospective preparation manifest.

The coordinator uses a qualified CPU Python environment for the current tool and
reader. The arena child uses the pinned old executable, working directory and
explicit environment; model loading on CUDA occurs only under the shared lease.
The shared helper's optional `cwd` and `env` arguments must be supplied together.
Existing callers retain their original defaults.

```bash
PYTHONPATH=. python scripts/bt4_recipe_screen.py --prepare \
  --manifest /absolute/prospective.json --expected-manifest-sha256 SHA \
  --out /absolute/new-preparation.json

PYTHONPATH=. python scripts/bt4_recipe_screen.py --execute \
  --manifest /absolute/final.json --expected-manifest-sha256 SHA
```

Preparation CPU-loads each final checkpoint once through the frozen model loader,
records its actual dynamic-relation flag, and computes the uncapped leaf requirement
with the actual resolved search and pool. No compile, inference or CUDA initialization
occurs. The post-load arena requirement must independently equal the prepared
value and remain within4096 in the completed readout.

Preparation calls the unchanged PGN sampler at500 and128 pairs, each from a fresh
NumPy seed42, and checks the entire root-FEN/move-history/endpoint prefix. It banks
a500-entry panel and full search/execution settings. Arenas still use the original
PGN CLI, seed42 and16 opening plies; no new opening format or sampling scheme is
introduced. Input metadata must remain stable through preparation and each arena.
The reader can check the bank's endpoints; actual initial history consumption
continues to rely on this qualified producer and frozen sampler.

## Bounds and receipts

Each arena gets an independent surviving GNU timeout: TERM at5370 seconds and
KILL30 seconds later, within its5400-second allowance. Its internal deadline is
**5340 seconds**, leaving30 seconds for normal completion before TERM. Setting the
internal deadline to5370 would race the supervisor rather than qualify an ordinary
deadline result. The combined arena charge is at most3 GPU hours; the completed
candidate training charge must be at most4.5 hours, retaining the7.5-hour package
bound. CPU preparation has a 300-second surviving cap (TERM at 270 seconds,
KILL 30 seconds later), with a 305-second outer subprocess wait. Each reader
retains its separate 120-second surviving cap.

The preparation allowance follows existing banked startup evidence: H20's C100,
G100 and C400 stages took approximately 100.69, 104.91 and 104.54 seconds from process start to
the game-log header, before checkpoint loads. The same PGN sampling happens before
that header; preparation additionally CPU-loads two models and checks both opening
prefixes. Its former TERM-at-90-seconds allowance was therefore too short. These
historical timings justify a bounded allowance, not a new benchmark or a promise
that preparation will finish within it.

Lease waiting is outside GPU stage charges and checks STOP and the150GiB reserve.
STOP, source changes or child failure halt the package; only owned process groups
are cleaned up. A coordinator SIGKILL can leave incomplete receipts, but the child
retains the lease and independent timeout. Reconcile that evidence before any new
attempt; this tool offers no adoption or retry mode.

Every stage banks the actual command/process receipt, launch proof, pinned reader
manifest, raw game/result records and reader stdout/stderr. Top-level completion
requires both certified stages. Existing elapsed-charge checks remain separate
from scientific stopping. No GPU allocator fraction, throughput improvement or
production adoption is claimed by this tooling.
