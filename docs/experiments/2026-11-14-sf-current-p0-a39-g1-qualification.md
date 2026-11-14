# Actual A39 / original G1 SF-P0 consumer qualification

Status: **closed owner PASS, root and independent actual engineering acceptance
with no findings**. This is four engineering updates, not the proposed 512-update
research comparison or evidence of playing strength. The earlier
[default-CUDA exact-zero NONPASS](2026-11-14-sf-current-p0-cuda-qualification.md)
is preserved.

## Criterion fixed before execution

The question was whether the actual full A39 network could consume original G1
through the new strict SF-P0 family at B256 with AMP, using the intended targets
and fresh optimizer state. The two branches were alpha 0 and native SF alpha 0.2,
each fresh from the same unchanged donor, with two applied updates/512 views.
They shared seed 2026111452, mirror probability 0.5, source and data. Both used
engineering-only constant LR 1e-6, zero warmup and period 8192. A two-update
horizon cannot use the separately proposed scientific warmup of 256 steps.

The predeclared criterion required exact source/data/native identities, primary
fields, row draws and captured RNG controls; finite model/optimizer state and
updates; actual AMP observations; the original within-call main-WDL formula and
loss scale; and observed teacher-health coverage with explicit gaps. The formula
check used rtol 1.3e-6 / atol 1e-5 on detached float32 reference calculations.
Those settings address within-call rounding, not cross-run model distance.
There was no model-equality predicate, retrospective zero-test pass, automatic
retry, reseed, dose extension or adoption rule.

Independent source review found that the first helper could pass zero orphan
flags with an entirely unmeasured health population. The corrected helper
validates binary masks, requires a positive aggregate checked population and
reports qualified-but-unchecked rows. It does not assume that every qualified
row has sufficient raw diagnostics. The original source packet remains unrun;
the corrected packet was independently accepted before execution.

## Frozen route and observed result

The child used frozen runtime `d6b9b978b36faaeeb4bcfcc6ce3f6333bef7323a`, its
isolated qualification environment and six native images built at
`7cd077b5a687d89ab7ee2fe768dba7dab4c65db2`. The previously accepted typing-only
transfer retained all native inputs. Full source and dependency metadata guards
ran before and after the computation. The outer supervisor remained R7 `ba9d6de`;
that identifies the owner, not the learner source.

Original A39 remained `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
The original data identity remained producing run `b0db4b6b-2bb3-4753-80d6-96825decbbee`,
source L090, G1 manifest `35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878`:
all 256 games / 24,827 rows. New engineering learner identities did not relabel
the producing run. The positive branch authenticated the real carrier's root,
independent and both closed producer/validator chains before loading it.

The owner completed in **102.897895416 seconds**, under a fresh 1,500-second
cap and 1,470-second inner limit. Both branches observed 494 state tensors,
61,444,448 unique parameters and inputs shaped `(256,175,8,8)`. Inputs were
float32; actual CUDA autocast and WDL outputs were bfloat16. Default backend
flags stayed unchanged, including deterministic algorithms off and cuDNN TF32 on.

| Observation | Alpha 0 | Native alpha 0.2 |
|---|---:|---:|
| Applied updates / sampled views | 2 / 512 | 2 / 512 |
| New SF-P0 prepare / wrap / overlay calls | 0 / 0 / 0 | 1 / 1 / 2 |
| New SF-P0 teacher mass | 0 | 100.4 |
| Genuine outcome mass | 512 | 411.6 |
| Positive qualified / invalid / unavailable rows | not overlaid | 502 / 3 / 7 |
| Positive health checked / gaps / orphans | not the P0 route | 502 / 0 / 0 |
| Torch peak allocated bytes | 8,592,873,984 | 8,587,735,552 |
| Torch peak reserved bytes | 8,860,467,200 | 8,950,644,736 |

Each branch constructed one Trainer, strictly loaded the fresh donor once and
used two host batches, forwards and original loss calls. All 438 active NadamW
state entries had counter 2; losses/state were finite, with zero skipped/retried
updates or clipping. Exact paired controls matched: journal hash, complete
membership, all five captured RNG states and primary field bytes before/after
the existing single mirror stream. These are route and custody observations,
not a proof of deterministic GPU tensor updates.

The zero branch deliberately retained legacy P1 fields already in replay. Its
`sf_wdl_effective_frac=1` and `sf_eval_pv_orphan_frac=0.01171875` are inactive
legacy-label diagnostics: actual SF loss share and new P0 teacher mass were zero.
They are not positive P0 health measurements. The positive branch replaced the
teacher context through the admitted P0 route and recorded 502 measured checks
with zero gaps/orphans. No blanket zero-orphan claim is made for both branches.

Process peak RSS was 3,764,891,648 bytes. Torch allocator and process RSS counters
do not measure total Windows/device/context memory. Observation copies and
synchronizations also make this unsuitable as a throughput benchmark.

## Scope and retained evidence

This qualifies the finite Python execution route for this model, bank, batch,
precision and four-update engineering horizon. It does not qualify direct CLI,
same-family continuation, the scientific LR schedule or 512-update dose, new
arena serving, teacher quality, sustained RL, playing strength or deployment.
The two saved engineering checkpoints remain unadopted. Original CUDA NONPASS,
prior negative experiments and unused confirmations remain unchanged.

All locations below are relative to
`~/chess-artifacts/operations/rl-week-one-20261112-v1/`.

| Evidence | Location | SHA256 |
|---|---|---|
| Corrected source preparation | `core-SF-current-P0-A39-G1-qualification-preparation-v2/manifest.json` | `02d041b5c0815c2413fa26e27c1abbc373652dfb68ff03e526f8af560450c6df` |
| Original source finding | `core-SF-current-P0-A39-G1-qualification-independent-source-review-v1/review.json` | `ddedf463e890e2cb513d4ec8d7c735dfcf7d0dfd951e9a90d5c82e282ff654f4` |
| Corrected independent source acceptance | `core-SF-current-P0-A39-G1-qualification-independent-source-review-v2/review.json` | `966e8683f6dbcddf9cbefc246252125e1b83fb6a1f01e3ced812cdecd57534e3` |
| Complete owned readout | `core-SF-current-P0-A39-G1-qualification-results-v1/readout.json` | `db2f128de658b9369d7c65e1b7a1a03f5fcff6b99281b807a9bb295434580b15` |
| Root engineering acceptance | `core-SF-current-P0-A39-G1-qualification-bindings-v1/root-acceptance.json` | `3f6b06572a08354c5cc7394f734cb04096dea0ac95593a77e5c6690754e09c08` |
| Independent actual acceptance | `core-SF-current-P0-A39-G1-qualification-independent-actual-review-v1/review.json` | `0d1ce613ecee48566b774b40ae70f4b979e56eae22d2f203bdecf0b8e879c1ec` |

Root reconciliation verified all 245 recorded owners closed. Week GPU owner wall
time at that cutoff was 54,532.46119880414 seconds, including failed attempts and
earlier experiments. The next scientific direction still needs complete frozen
roles, common measurement/source bindings and published preregistration before
research execution. No Rofchade, Stockfish or Lc0 strength milestone is achieved.
