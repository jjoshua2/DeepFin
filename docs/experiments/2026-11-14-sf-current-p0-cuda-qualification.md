# SF-P0 CUDA qualification and default-backend repeatability

Status: **original zero-weight CUDA qualification NONPASS; observational
repeatability diagnostic complete; two separately scoped positive teacher
controls passed their assertions**. The diagnostic cutoff was 2026-11-14
17:09:57 UTC; the positive controls were reconciled by root at 17:13:35 UTC,
then independently accepted with no findings. No learning
recipe, full GPU-consumer qualification, checkpoint adoption or playing-strength
claim follows.

## Question and frozen execution

The [CPU consumer qualification](2026-11-14-sf-current-p0-finite-consumer-qualification.md)
passed 22 new and 21 legacy controls. Source was published by
[PR #1316](https://github.com/jjoshua2/DeepFin/pull/1316), merge
`056cdac1a7611dc658cf74e4597b2c88db098d9d`. GPU execution used the separately
frozen runtime `d6b9b978b36faaeeb4bcfcc6ce3f6333bef7323a`, its isolated
Python 3.13 environment, six qualified native images and two CPU threads.
The qualified loaded Python/native inputs match published source; the entire
local runtime is not claimed identical to the later main tree.

All three GPU owners had a 300-second hard cap and 270-second inner limit, under
separate, newly identified resource scopes. All actual wall time, including
the failed owner, counts toward the week budget. They used synthetic Tiny
networks, batch size 2, float32 tensors and no AMP. Default backend settings
were retained: deterministic algorithms off, cuDNN deterministic off,
cuDNN benchmark off, cuDNN TF32 on, matmul TF32 off, matmul precision highest.
Float32 tensor dtype therefore does not establish IEEE-only computation.

## Original three-case qualification: failed first case

The original plan selected the unchanged serialized test bodies for zero
SF weight and two positive cases: alpha 0.5, native/stored teacher variants,
forced mirror probability 1. The zero case ran first and required **exact**
model, optimizer and scheduler equality against the legacy learner. There
was no numerical tolerance or automatic retry.

The owner failed after 10.419166275 seconds with
`ValueError('checkpoint continuation did not restore model.trunk.0.weight exactly')`.
Both legacy and new zero branches completed two applied updates/four views
each. Captured RNG states, draw journals and replay membership matched.
The first model comparison failed; optimizer/scheduler comparisons, the
later poisoned-sidecar refusal and both positive teacher cases were unrun.
The helper's final custody and weak-reference assertions were not reached.
Root and independent review accepted this as a closed failed attempt, not
GPU qualification.

## Separately selected observational repeatability diagnostic

Before this diagnostic ran, root fixed one new CPU donor seed 2026111407,
one shared synthetic bank, trainer seed 45, mirror probability 0.5 and four
fresh two-update branches in this order: OLD1, OLD2, NEWZERO1, NEWZERO2.
The donor differs from the original failed fixture. No failed checkpoint
payload was reused. The four comparison pairs below were fixed in advance.
The diagnostic reports exact identity, byte identity and descriptive numeric
differences; it selects no tolerance or replacement qualification criterion.

The owner passed its completion/custody checks in 11.534799741 seconds,
with eight actual applied updates and four released Trainers. All four pairs
had identical captured RNG, draw journal and membership controls. Backend
flags stayed unchanged. State comparisons ran inside the GPU owner after
loading its own checkpoints onto CPU. Root and independent actual review
accepted the closed diagnostic with no findings, within this scope.

| Fixed comparison | Model exact? | Model maximum absolute difference | Optimizer exact? | Scheduler exact? |
|---|---|---:|---|---|
| OLD1 / OLD2 | no | 2.9103830456733704e-11 | no | yes |
| OLD1 / NEWZERO1 | no | 5.820766091346741e-11 | no | yes |
| NEWZERO1 / NEWZERO2 | no | 5.820766091346741e-11 | no | yes |
| OLD2 / NEWZERO2 | yes | 0 | yes | yes |

All comparisons had zero noncomparable fields. The model differences were
confined to one tensor per differing pair. The complete readout retains
per-field counts, byte hashes, absolute differences and L2 distances.
Torch peak allocated memory was 80,375,296 bytes; this excludes other GPU
allocations and is not total device/context memory.

The old-versus-old result shows default CUDA is not bit repeatable on this
new donor. It does not establish the unique cause of the original failure,
and one exact cross pair does not qualify the new path. The original exact
zero criterion remains NONPASS. Neither result tests the actual A39 model,
original G1 bank, B256/AMP route, sustained learning or playing strength.

## Separate positive-control execution

Root next selected only the two previously unrun positive test bodies, under
fresh scope `e66010da-35dc-4496-bdf5-0af53fbc1d04`. The zero control was not
rerun. The original assertions and their existing numerical settings were
retained; no model-equivalence tolerance was introduced. The new owner again
had 300 seconds, with a 270-second inner limit and no automatic retry.

Native alpha 0.5 / mirror 1 ran first, then stored alpha 0.5 / mirror 1.
Both bodies returned normally, with two fresh Trainers and four total applied
CUDA updates/eight views. The owner passed in 10.417294601 seconds. Each
variant drew one qualified teacher row and three invalid-label outcome
fallback rows: teacher mass 0.5, outcome mass 3.5. The synthetic unavailable
row was not drawn at loss in these cases. Existing assertions checked the
within-call blended WDL formula, total loss scale, original row sampling,
single mirror stream, masks and predecessor health diagnostics. Source,
native, backend and Trainer-release guards completed. Root and independent
actual review accepted this closed positive-only result.

This qualifies these two Tiny synthetic positive assertions only. It does
not combine with the diagnostic to turn the original zero NONPASS into a
pass. Actual A39/G1/B256/AMP execution and direct CLI remain unqualified.
The next engineering decision is qualification of that actual A39/G1 route;
a proposed SF-anchor training comparison remains unselected and unadmitted.

## Retained evidence

All paths below are under
`~/chess-artifacts/operations/rl-week-one-20261112-v1/`.
JSON references bind the source, plan, admission, exact argv, native images,
owner terminal and prior failure. Source-only reviews are distinct from
actual readout acceptance.

| Evidence | Relative location | SHA256 |
|---|---|---|
| Original preparation | `core-SF-current-P0-finite-consumer-GPU-synthetic-preparation-v1/preparation.json` | `4cd5fa6385b74824dbf2d56b3af8a0df5c3ccce60b4409d2bc18d1c94af3321c` |
| Closed original failure | `core-SF-current-P0-finite-consumer-GPU-synthetic-bindings-v1/root-failure-readout.json` | `ee39ca70c2b0d333f6413adb06f283b5ebffb6bf8cc25b3ed8692498bf05b414` |
| Independent original failure review | `core-SF-current-P0-finite-consumer-GPU-synthetic-independent-failure-review-v1/review.json` | `4f7a864198ff2c7ac592fc4e433f5685f79a83bb3cd25ec21d083a2d30a82128` |
| Diagnostic preparation | `core-SF-current-P0-CUDA-repeatability-preparation-v1/preparation.json` | `95f09b107857dad170d0139a80de0f93b113095214202b7deab7ab2b3ee71a93` |
| Independent diagnostic source review | `core-SF-current-P0-CUDA-repeatability-independent-source-review-v1/review.json` | `25b540a0bf0237ec4312d0ff24cf3e79a0f835b5d2d8edfd75ee0a353ab99740` |
| Complete diagnostic readout | `core-SF-current-P0-CUDA-repeatability-results-v1/readout.json` | `0f5ccce34f92e2fa7f51c37e16bc019ff31766ab73fd933df35fbb079db350de` |
| Root diagnostic-only acceptance | `core-SF-current-P0-CUDA-repeatability-bindings-v1/root-readout-acceptance.json` | `4ca84b5032c0f2ead66c3e9765363e35e9963eaaa76acc1fc951e2804ee331f5` |
| Independent diagnostic actual review | `core-SF-current-P0-CUDA-repeatability-independent-actual-review-v1/review.json` | `88d6701b74bba913a5c91fd24f943e9be1be9ddd009b4930fa9681092f2aeb80` |
| Positive-only preparation | `core-SF-current-P0-CUDA-positive-preparation-v1/preparation.json` | `67c85ec9420396f3eb17d14b49e74d890a055b9030889cdd21e3c0115198dde7` |
| Independent positive source review | `core-SF-current-P0-CUDA-positive-independent-source-review-v1/review.json` | `13f444623fb91cee9384ac5068aa3b71d9889e3f494a44b4927edb6f65418ed6` |
| Positive complete readout | `core-SF-current-P0-CUDA-positive-results-v1/readout.json` | `8ea7f01204f5f671986a1e2450e40d771c4f5bb34bc79036added3c482a83f7c` |
| Root positive-only acceptance | `core-SF-current-P0-CUDA-positive-bindings-v1/root-acceptance.json` | `e6f41459858e8320a11de9012c63e627f23ad6068a52597b1e4149c6708f41a2` |
| Independent positive actual review | `core-SF-current-P0-CUDA-positive-independent-actual-review-v1/review.json` | `00fea106a710db057a47b7702e116f2bdf76a1df0f2a971da1911d3fa4401b46` |

The root closure census verified all 243 recorded owners closed and all
recorded process births absent. Actual week GPU owner wall time at that
cutoff was 54,419.146008787095 seconds, including earlier experiments.
The broader first-week goal remains active: no verified gain over multicore
Rofchade, Stockfish or Lc0 has been established.
The later positive-control census verified all 244 owners closed, with actual
week GPU owner wall time 54,429.56330338813 seconds.
