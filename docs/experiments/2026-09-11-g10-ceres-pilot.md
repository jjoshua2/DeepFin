# G10 Ceres first-four-shard convenience pilot

The preregistered task was to collect C3-768-30-pre8-I8 policy and both raw value-logit heads for the first four whole original run06 G10 derived shards (0..3, 32,768 rows, 1,024 batch-32 calls, zero padding). This is a convenience prefix, **not representative G10**. Same-row teacher disagreement is a diagnostic, not an Elo estimate or a new training queue.

Schedule only after the original18.91M Ceres collection is terminal, during final saved-output qualification or CPU target preparation, and without delaying the registered Ceres training anchors. Parent reviews this plan and publishes the preregistration before GPU spending. There is no automatic launch, wait daemon, resume or retry.

The original complete admitted run06 batch remains 262,079 rows/32 shards. Source summary `ab214d52665ee4ad6ccf756c0515344dc9653c3b51de15965fc74323198b7fd9`, G10 qualification `d8d71e5e1c73f193710757c36db3e07b36fbc6a9962ed28c318d0cd3742d5c1c`, and adapter `8b81f9633050f8e2f286259671e3cefb39bfb30dffa29c7d76608723d36505c8` are pinned in `plan.json`. Exact selected specs include each original row-provenance hash. No synthetic smaller source manifest, B100 substitute or adaptive-value source is used.

## Execution and reuse

`run_chunk.py` is a fresh narrow copy of the accepted `chunk_adapter_v2/run_chunk.py`. Its differences are fixed G10 range0..3 with exact original full shard specs, two G10 qualification flags, and a 1200-second deadline. No collector/numerical/provider implementation was changed. The mapped-library helper is copied byte-identically. `integration.diff` records the entire adapter change.

The unchanged existing `ceres_collection_batches.py` receives a one-chunk `driver.plan.json`. It owns a new process group and TERM/KILL cleanup; the collector retains its shared GPU lease and own child cleanup. `command.prepared.txt` adds an independently surviving external 1170-second TERM plus30-second KILL bound. Driver and wrapper default invocations only validate; the prepared launch command contains explicit `--execute` and is not executed here.

Runtime `/tmp/deepfin-ceres-selected-bank-runtime`, commit `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`; Python `/tmp/deepfin-ceres-collector-ort129/bin/python` (CPython3.13.15, **ONNX Runtime1.29.0**, NumPy2.2.6), accepted CUDA13/cuDNN package and loaded-library pins. Model `data/ceres/C3-768-30-pre8-I8/C3-768-30-pre8-I8.onnx`, SHA `44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3`. Profile `ceres-c3-fixed32-compact-v2`, outputs policy/value/value2, `--retain-value2 --pad-final-batch`, primary WDL output value/logits, native float16. Exact package, runtime source, native-extension and mapped-library evidence is retained in the plan; large libraries/model were not rehashed during preparation.

Fresh output: `/home/josh/projects/chess/data/lc0/ceres_compact_sidecars/g10_run06_first4_v1`. The collector generates a fresh timestamped invocation directory under this output; freshness is enforced before execution. No existing output is relabeled or modified.

Bounds: CPUs4,5 with all numeric threads2 and nice19/ionice3; GPU0 arena8GiB; sampled device memory<=12GiB; host MemAvailable>=16GiB; per-process sampled RSS<=12GiB; SSD reserve150GiB; combined output/state128MiB; STOP at pilot, driver, Ceres readiness and output paths. The 1200-second inclusive external/driver deadline covers metadata admission, lease wait, model/library verification, calls, writes and cleanup. Existing telemetry retries only rc3/timeouts, with failure receipts and bounded logical-query deadline; inference is never retried. These are inherited measured sampling guards, not hard process-RSS quotas.

Cost basis: a completed131,072-row original-corpus recovery chunk took532.3155seconds. A simple quarter-row ratio is133.08seconds, but fixed session setup/library hashes and full G10 metadata admission do not scale with row count. Use the20-minute ceiling and report actual elapsed/runtime/call counts rather than claim a measured G10 speedup.

Fresh host preflight must establish original collection is done, the GPU lease is available, CPU affinity is appropriate, model/library stat and WSL boot/mount identities remain valid, and resources are free. Historical GPU/boot identities in this prepared plan are not a new host observation. If changed, stop and review the evidence instead of weakening checks.

## Completion and prospective diagnostic

Successful collection requires driver terminal0, wrapper completion, complete actual invocation, all4 complete saved shards/32,768 rows, exact original source bindings, both logits heads, fixed32 call accounting, first-call CUDA provider proof and first/final loaded-library/project evidence. Independently qualify saved outputs before creating the small audit-specific Ceres manifest from actual accepted bindings. A terminal collection alone is not training-corpus qualification.

Then reuse the existing policy audit with the same four source shards and the already pinned BT4 policy adapter. Retain per-row qualified source/game/shard/row identities and source-qualified game clustering. Descriptively report Ceres/BT4 top-move agreement, Jensen–Shannon divergence and each teacher's entropy; compare prespecified existing arithmetic and geometric mixture coverage and common-paired deeper-SF regret, with explicit conditional-regret versus all-row coverage denominators and mate/invalid/unscored exclusions. Preserve fixed exact-T300 preview semantics if included. No outcome-driven mixture/temperature grid or promotion criterion is introduced here. Confirm the actual audit's temperatures/weights from its frozen implementation before a later separately reviewed diagnostic invocation; this collection does not choose them or silently qualify a new downstream runtime.

Bank both raw Ceres WDL heads for a later explicitly authenticated native-BT4-WDL join. The present G10 policy adapter is policy-only; current audit cannot make the intended three-way value comparison from these inputs. Do not claim value agreement/calibration is available merely because both Ceres heads were stored. No new native-WDL join or value-audit code is included in this pilot.

## Publication and scheduling record

This pilot follows the [completed SF constraint diagnostic](2026-09-11-sf-negative-constraints-screen.md) and preserves the [registered Ceres training anchors](2026-09-11-ceres-weighted-bootstrap.md). To avoid CPU contention if collection overlaps final original-corpus qualification, run the existing CPU-only qualification on cores 2,3 and this pilot on its pinned cores 4,5. Target materialization retains cores 0,1. These are resource allocations, not changes to either teacher or verifier.

Preparation and independent review passed without reading payloads or loading a model. The preparation snapshot below preceded the completed collection and qualification reported next. The [preparation receipt](evidence/g10-ceres-first4-prepared-20260911.json) pins the reviewed plan, adapter and driver. Both Ceres heads will be banked for future value work; their presence alone does not complete the missing native-BT4 join.

## Completed collection and saved qualification, September 12

The original-corpus collector finished before this pilot launched. The existing
one-chunk driver completed all **32,768 rows in four shards** in **152.47 seconds**,
with 1,024 fixed-32 calls and zero padding. Root collection session 60278 exited
zero. This is a measured whole-pilot cost, not a speedup comparison.

The independently reviewed post-collection adapter then reused the existing saved
chunk auditor unchanged. Qualification session 34959 exited zero with
`PASS_SAVED_G10_CERES_FIRST4_NOT_TRAINING`. All 36 saved arrays passed the existing
content, shape and identity checks, retaining both raw value heads, legal rosters,
source-column digests, provider evidence, endpoint runtime observations and terminal
invocation proofs. The adapter matched the original G10 source storage and its
historically qualified source-column hashes without rereading source features,
native BT4 values, model or library payloads.

[Compact completion evidence](evidence/g10-ceres-first4-completed-20260912.json)
pins the driver, reviewed preparation and actual saved qualification
(`d82997934ae7b5a5dca31c1b9ce3588b854c4158bc98e5b9c5edc5d759d477d5`).
The actual four-shard policy-consumer manifest has SHA256
`972da2fabb2d1c721c6ca6b737d7404064147ecf92944ebf98392aeeaf87755e`;
it was assembled from accepted attributes and is paired with the terminal
qualification receipt. It does not admit a training corpus.

At that qualification milestone the prospective policy diagnostic had not run.
Its subsequent completed result is reported below; there are still no playing results. Both Ceres value heads are saved, but the
explicit native-BT4 value join remains outstanding. The convenience prefix retains
its sampling limitation; these four shards do not establish representative G10
behavior or replace the registered original-corpus training comparisons.

## Completed policy diagnostic, September 12

The small saved-bank comparison gives mixed evidence: Ceres and its mixtures reduce conditional deeper-SF regret in the prespecified balanced-score stratum, but increase it over the whole ordinary-score prefix. This is useful evidence of differing rankings, not grounds to cancel the registered Ceres training or promote a new mixture.

All 32,768 rows in four shards were banked and have Ceres inputs. The frozen ordinary-d9 analysis excludes 8,775 mate-domain rows, leaving 23,993. Another 618 ordinary rows lack an eligible nonmate final ruler. All five policies have the **same 23,375 valid-positive-mass rows**; there are no missing policies or zero-coverage rows in that eligible set. Conditional regret renormalizes each candidate over the saved, narrowed SF roster. Consequently, reduced coverage can hide errors outside that roster.

| Policy | Mean scored mass on eligible final rosters | Common-paired regret change versus BT4 |
| --- | ---: | ---: |
| BT4 T0.5 | 81.887% | 0 cp |
| Ceres T0.5 | 80.290% | +13.242 cp |
| Arithmetic 50/50 | 81.088% | +6.113 cp |
| Geometric 50/50 | 81.273% | +4.643 cp |
| Existing Tactical300 preview | 82.211% | −60.191 cp |

Positive change is worse according to this conditional SF ruler. These are descriptive means, not Elo or an independent tactical-correctness test. The Tactical300 result repeats a direction already seen in the larger SF-only diagnostic and does not establish a training gain.

The fixed balanced stratum, absolute phase-zero d9 best score <=100 cp, contains 2,250 ordinary rows (2,247 common-valid). BT4 conditional regret is 10.533 cp; Ceres reduces it by 2.066 cp, arithmetic by 1.051 cp, geometric by 2.036 cp, and Tactical300 by 0.636 cp. Scored mass is respectively 87.866%, 87.275%, 87.571%, 87.836%, and 87.874%. The geometric mixture preserves nearly BT4's roster coverage while approaching Ceres's conditional regret in this stratum. It is a plausible later contrast, not a selected winner from this convenience prefix. Balanced positions are only 9.38% of ordinary rows; other positions are not automatically irrelevant or decided games.

At d9 best >100 cp, Ceres's paired delta is +20.331 cp; below −100 cp it is +7.787 cp. Arithmetic/geometric deltas have the same sign in these strata. The full readout retains all prespecified position strata, not just the favorable balanced one.

Neural top sets overlap on 20,182/23,993 ordinary positions (84.12%). Mean JS divergence is 0.021651 nats; mean BT4/Ceres entropy is 1.227863/1.351206 nats. Of 3,811 top disagreements, 2,573 (67.52%) have both entire top sets scored by the saved SF roster. In these, the best scored member favors BT4 on 1,308 and Ceres on 1,099, with 128 ties at the final maximum and 38 ties below a third move. The other 1,238 disagreements are unadjudicated, not teacher losses. Comparing each top set's maximum does not certify every tied move in that set.

Next decision: retain the already registered arithmetic policy and separate value training anchors. Do not tune temperatures/thresholds on these four shards or infer that Ceres is globally inferior from SF self-agreement. The balanced geometric result justifies retaining geometric mixing as one distinct later candidate if the actual trained anchors warrant it. Additional G10 collection was stopped by the existing GPU memory guard before inference; preserve the user's graphics workload and failed receipt rather than retry automatically. A future wider saved cohort should check whether the balanced/global pattern survives, without making it a prerequisite for the registered original-corpus training.

That policy-only invocation had **no value result**: every value comparison had zero rows and null loss means because the native-BT4 value join was absent. The later authenticated value pass below fills that missing input without changing these historical policy results.


The diagnostic completed in 376.82 seconds with 2,000,152 KiB peak RSS; root
session 50016 exited zero. A separate bounded one-pass bank readout took 6.53
seconds and 53,196 KiB. It verified the bank identity, unique source/derived
rows, exact layout, policy aggregates and entropy/JS sums. Source-qualified
game sufficient statistics remain available for all 167 ordinary game groups.
Independent scientific review passed without rescanning the bank.

The [full descriptive readout](evidence/g10-ceres-policy-readout-20260912.json)
contains every fixed stratum and paired denominator. The [original audit summary](evidence/g10-ceres-policy-summary-20260912.json)
preserves all original metrics, including unavailable values as null means with
zero rows. Both are byte-identical copies of the completed host artifacts.
[Compact progress evidence](evidence/g10-ceres-policy-progress-20260912.json)
pins terminal status, plans, reviews and the local bank. No new source scan,
inference or bootstrap was used to prepare this publication.

## Remaining cohort preregistration and preserved failure

Before reading these policy outcomes, the next collection was registered for
shards 4–19 and 20–31 of the same original run06 source: 229,311 new stored
rows, 7,166 fixed-32 calls and exactly one padded input on the final 8,127-row
shard. It preserved the qualified first four shards. Two 1,200-second chunk
ceilings sat within a 2,400-second total allocation including the inter-chunk
pause, using the same frozen teacher/runtime and existing resource guards.
It introduced no new teacher weights or training promotion claim.

The first remaining chunk stopped **before inference** after 5.2257 seconds;
root session 90935 exited one. At 2026-09-12T02:01:23.014076+00:00 the wrapper
recorded `sampled device memory above12GiB`. The exact triggering MiB value
was not persisted, so the supported statement is only that it exceeded 12,288
MiB. A later observation at 2026-09-12T02:03:12.516662+00:00 showed
20,204 MiB device memory used and 52% utilization. That later observation is
not the failed sample. Windows graphics activity supported external contention;
per-process graphics counters are not additive device-residency measurements.

No user graphics process was stopped, no guard was weakened, and no automatic
retry or GPU reallocation occurred. The failed v1 remains preserved and adds
zero qualified rows. Further collection is deferred while the existing CPU
materialization continues; the full original-corpus registered training anchors
retain priority. Completing this convenience cohort would still require saved
qualification of its final partial shard and does not replace those anchors.

## Completed authenticated value diagnostic, September 12

The native BT4 join works on the actual saved G10 bank, and it fills the previously missing six-way value comparison. The scientific result mainly exposes how closely the evaluation ruler follows the original SF labels. It does not establish that keeping pure SF values is best for training or that Ceres/BT4 value mixing is harmful.

The run banked all 32,768 original rows. Exactly 23,993 rows are common to all six value predictions; 8,775 rows are excluded because the d9 roster contains mate-domain scores. There are no other value exclusions. The included set has 16,745 d10 final rulers (including 86 single-move cases) and 7,248 d12 rulers.

This value denominator is **618 rows larger than the policy denominator**. The value code maps the maximum saved final SF score through the original fixed CP-to-WDL conversion even when the final roster contains a mate-coded score. Policy regret instead excludes any mate-domain final roster. The 618 included value rows with such rosters are reported separately; this does not imply that each roster's maximizing move is itself mate-coded.

All results below use the same 23,993 included rows. Brier is the sum of squared errors over win/draw/loss; cross-entropy uses natural logs. Lower means closer agreement with this fixed SF-derived ruler.

| Fixed prediction | Mean Brier | Mean cross-entropy |
| --- | ---: | ---: |
| Saved SF | 0.000316 | 0.292345 |
| Native BT4 | 0.034222 | 0.473302 |
| Ceres primary T0.55 | 0.049225 | 0.792925 |
| Ceres secondary T1.5 | 0.035772 | 0.420418 |
| Ceres dual, 60/40 | 0.043075 | 0.483361 |
| Registered SF50/BT425/Ceres-dual25 | 0.009383 | 0.309598 |

The low saved-SF error is not an independent validation. The ruler reuses SF, saved adaptive search and the original source's CP-to-WDL mapping. At a declared descriptive tolerance of Brier <=1e-6 (WDL L2 distance <=0.001), 8,865/23,993 rows (36.95%) are almost unchanged from the saved SF prediction; 2,427 have exactly zero banked SF Brier. No SF prediction was reconstructed or invented: these counts use its actual banked loss. The mean ruler entropy is 0.291207 nats, so saved SF cross-entropy exceeds it by only about 0.001139 nats. Closeness across the cohort is much stronger than the near-zero count alone conveys.

The fixed balanced stratum, absolute phase-zero d9 best score <=100 cp, has 2,250 included rows. Mean Brier/CE are SF 0.000463/1.086962, BT4 0.258230/2.010652, primary 0.356411/3.136971, secondary 0.261855/1.823039, dual 0.312337/2.111680, and registered blend 0.069701/1.187164. Its ruler entropy is 1.086276 nats, close to the maximum ln(3), while the actual game-outcome calibration of these predictions is not measured. Saved-SF closeness also persists here: 741/2,250 losses meet the same tolerance. All other fixed score and position strata are retained in the machine readout.

Ceres secondary has lower cross-entropy than native BT4 by 0.052884 nats on the common cohort, but slightly higher Brier by 0.001550. The registered dual has higher Brier by 0.008854 and higher CE by 0.010060 than BT4. Primary has substantially higher CE against this particular ruler. These different metric rankings are reasons to keep head/calibration questions explicit, not to fit a new temperature or declare one teacher universally superior. The SF-containing registered blend is naturally closer to an SF ruler; that does not isolate Ceres's incremental value.

Next decision: retain the registered value training and its comparisons against B100 and B100V50. Those comparisons are more discriminating about whether Ceres adds useful training supervision than further optimizing agreement with nearly the same SF labels. If playing evidence later motivates a different Ceres value-head combination, this bank can describe its failure pattern, but the current pass selects no new weights, temperatures or teacher. No independent value accuracy, game calibration or Elo result is established.

The actual diagnostic completed in 509.61 seconds with 1,999,496 KiB peak RSS; root session 48260 exited zero. A single bounded readout pass took 15.08 seconds and 64,712 KiB, with two CPU cores and the GPU hidden. It verified the exact bank hash, unique source/derived identities, four-shard layout, shared six-way inclusion, and every aggregate loss against the original summary. All nonvalue metrics exactly match the previous policy-only summary; the old policy bank was not reread. Source-qualified statistics for 167 included game groups are retained. No inference, source scan, bootstrap, repeated bank pass or calibration fitting was performed.


Full WDL losses do not isolate the search quantity `Q = W − L`: draw-probability
errors also contribute. This readout therefore does not rank heads by search
value quality. The subsequent Q-versus-draw decomposition below addresses this distinction;
it remains agreement with the SF ruler and does not fit a new recipe.

The [full value readout](evidence/g10-ceres-value-readout-20260912.json) preserves
all fixed strata, common paired differences and the near-zero tolerance. The
[original machine summary](evidence/g10-ceres-value-summary-20260912.json) and
readout are byte-identical host artifacts. [Compact execution and validation evidence](evidence/g10-ceres-value-progress-20260912.json)
pins the actual bank, terminal, runtime and independent scientific review.

The runtime is commit `21aec39249c17a44145a4435f2ca9abbcf93cb4e` on
[PR #655](https://github.com/jjoshua2/DeepFin/pull/655), an open stack based on
`fix/teacher-adjudication-coverage`, not merged into main by this publication.
Fourteen focused tests, scoped types with zero findings, whole Ruff/Vulture and
independent source reviews supported the bounded research run. Whole-repository
type checks timed out and remain unresolved; a completed data diagnostic does
not turn those checks into a passing upstream validation. The existing source,
failed collection evidence and current CPU materialization are unchanged.

## Completed Q/draw decomposition, September 12

The decomposition confirms that most neural-versus-SF WDL Brier disagreement is draw-mass disagreement on this sample. MCTS consumes Q=W-L; WDL Brier by itself does not quantify scalar-Q error. This remains agreement with the same SF-derived ruler, not independent Q accuracy or Elo.

All 23,993 included rows reproduce all six prior banked losses. The 8,775 d9 mate-domain rows remain excluded and retain their original identities with null unavailable d9 scores in the new bank. The fixed balanced stratum contains 2,250 included rows.

For normalized WDL, Brier = 0.5*(Qprediction-Qruler)^2 + 1.5*(Dprediction-Druler)^2. The draw share below is the share of summed Brier, not an average of per-row ratios.

| Fixed prediction | Q MSE, all | Draw share of Brier, all | Q MSE, balanced | Draw share, balanced |
| --- | ---: | ---: | ---: | ---: |
| Saved SF | 0.000405 | 35.88% | 0.000835 | 9.77% |
| Native BT4 | 0.014579 | 78.70% | 0.038686 | 92.51% |
| Ceres primary T0.55 | 0.023565 | 76.06% | 0.063453 | 91.10% |
| Ceres secondary T1.5 | 0.019216 | 73.14% | 0.067674 | 87.08% |
| Ceres dual 60/40 | 0.021567 | 74.97% | 0.064021 | 89.75% |
| Registered SF50/BT425/Ceres25 | 0.004344 | 76.85% | 0.011858 | 91.49% |

In the balanced stratum, the ruler's mean draw probability is 0.339029; native BT4 predicts 0.657477, Ceres primary 0.725489, secondary 0.623582 and dual 0.684727. These probabilities differ materially under the fixed teacher mappings; game-outcome calibration was not measured. Draw disagreement contributes 92.51% of BT4's balanced Brier, 91.10% of primary's, 87.08% of secondary's and 89.75% of dual's.

The earlier lower cross-entropy of secondary versus primary does not translate into uniformly better scalar agreement: on balanced rows primary Q MSE is 0.063453 versus secondary 0.067674. Across all included rows secondary instead has lower Q MSE, 0.019216 versus primary 0.023565. Native BT4 has lower Q MSE than either on both fixed cohorts. This is descriptive evidence about these fixed temperatures and the SF ruler; it selects no new head, mixture or temperature and does not displace registered training anchors.

The machine readout retains signed Q/draw errors, prediction means and p05/p25/p50/p75/p95 quantiles for all prior fixed strata. A compact bank preserves original source-qualified identities, inclusion, strata, ruler Q/D and the six actual prediction Q/D pairs, so future decompositions need not reread arrays.

Authentication and limits: original SF search_wdl hashes were witnessed now under unchanged storage stamps from the completed value audit. They are explicitly not historical payload hashes. Native BT4 and both Ceres heads matched their existing accepted decoded hashes and attributes. All six per-row Brier/CE losses matched the prior bank within maximum absolute difference 1.78e-15; the Brier decomposition residual was at most 5.01e-16. No raw FEN, history, feature/legal arrays, policy payloads, model or engine was read.

One initial attempt stopped after 6.07 seconds because the reader incorrectly required float16 for native WDL; actual accepted native WDL is float32. That attempt read the first SF value array but no native payload or retained bank. The corrected, independently reviewed attempt read 983,040 decoded value bytes plus one retained-bank pass, completed in 42.27 seconds at 395,704 KiB peak RSS, and exited zero (session16452). Total execution was 48.34 seconds, within the original 120-second budget. Both attempts are retained. No inference, fit, bootstrap or raw-source join ran.

The [full Q/draw readout](evidence/g10-value-q-draw-readout-20260912.json)
is a byte-identical copy of the reviewed machine result, retaining all fixed
strata and quantiles. [Compact provenance and execution receipts](evidence/g10-value-q-draw-progress-20260912.json)
pin both attempts, the reviewed corrected plan, actual terminal and local Q/draw
bank. Publication did not reread arrays or either row bank.

A possible later controlled question is whether changing draw supervision while
holding each target's Q fixed improves training. That would distinguish changes
in draw supervision and shared-network learning from changes in scalar value
targets. It is a hypothesis, not a queued experiment or validated improvement.
The registered training anchors retain priority.
