# G10 Ceres first-four-shard convenience pilot

Current collection status: **3,858,023 G10 rows / 472 shards are qualified**
across the original run06 increment, both complete next96 cohorts and the complete
run06 large cohort, including policy and both value-logit heads. No further Ceres
collection is launched. The completed science below still uses the original
32,768-row first-four-shard sample. Collection does not enlarge those diagnostic
results. The dated sections retain the actual execution history.

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


## Completed same-move policy complementarity, September 12

Ceres reduces mean probability on deeper-confirmed SF-flagged moves from
**0.22032% to 0.13675% in balanced positions**, but increases it from **2.17833%
to 2.30303% across all ordinary positions**. This measures teacher allocations
relative to saved SF scores, not verified mistakes or Elo. About 60% of flagged
mass cannot be adjudicated under the conservative criterion, and this four-shard
prefix is not a representative sample of G10.

The fixed comparison uses moves more than 300 cp below the d9 maximum, with BT4
T0.5, Ceres T0.5 and their arithmetic 50/50 mixture measured on exactly the same
sets. A constraint is confirmed only if every d9 winner still outranks that move
in the saved final roster. Outranking any prior winner contradicts that conservative
constraint; equality with the lowest winner is a tie. Missing the flagged move or
any prior winner makes the outcome unavailable. Final mate-domain scores retain
the existing ranking semantics and explicit flags.

All 32,768 source rows were banked: 23,993 ordinary rows, 8,775 excluded d9
mate-domain rows, and 167 source-qualified game groups. All earlier aggregate
policy/value metrics match exactly. Included ordinary rows retain move indices,
scores, availability, both teacher probabilities and existing row identities;
future interpretation can reuse these observations without another raw join.

| Fixed stratum | Rows | Mean flagged mass, BT4 / Ceres / arithmetic50 | Mean confirmed flagged mass, BT4 / Ceres / arithmetic50 |
| --- | ---: | --- | --- |
| All ordinary | 23,993 | 6.42126% / 6.84613% / 6.63369% | 2.17833% / 2.30303% / 2.24068% |
| Balanced, abs(d9 best) <=100 cp | 2,250 | 0.49096% / 0.31315% / 0.40205% | 0.22032% / 0.13675% / 0.17854% |
| Outside balanced | 21,743 | 7.03494% / 7.52217% / 7.27855% | 2.38095% / 2.52720% / 2.45407% |
| Outside T300's gap gate | 22,247 | 4.65956% / 4.96639% / 4.81298% | 1.16181% / 1.21746% / 1.18963% |
| Balanced, outside that gate | 2,006 | 0.47076% / 0.32313% / 0.39694% | 0.19344% / 0.14206% / 0.16775% |
| Inside T300's gap gate | 1,746 | 28.86835% / 30.79710% / 29.83272% | 15.13056% / 16.13506% / 15.63281% |

Each mean uses all ordinary rows in its stratum, including zero-flag rows; no
renormalization over the final roster is applied. Arithmetic50 is necessarily
halfway and supplies no independent evidence. On all ordinary rows, summed
confirmed flagged mass changes from 522.6475 to 552.5655: Ceres removes 66.0001
from the same moves but adds 95.9182 elsewhere within that confirmed set. Those
sums are probability across positions, not counts of errors or games. In balanced
rows, the corresponding mass falls from 4.9573 to 3.0769, with 2.7536 removed and
0.8732 added: a 37.93% net reduction. One balanced contradicted constraint carries
only 7.9434e-6 BT4 mass and 5.9301e-6 Ceres mass; there are no balanced tied constraints.

Unavailable mass remains explicit: 915.0908 BT4 and 988.1030 Ceres summed across
ordinary rows. Ceres therefore does not globally make an SF constraint redundant,
while the balanced result shows complementary teacher allocations relative to the
saved SF ruler. The gate-complement result establishes a distinct mechanism to
consider: an all-move veto can act when several SF good moves are nearly tied,
whereas T300 requires an isolated best-next-lower gap. Neither result establishes
that applying a veto improves training. Keep the registered strength anchors;
no threshold sweep, new training recipe or teacher promotion follows this readout.

The independently reviewed, pinned source pass exited zero in 336.412 seconds
(child 336.12 seconds, peak RSS 2,001,572 KiB), within the 900-second CPU-only
budget. Its inherited terminal label still says VALUE_DIAGNOSTIC; the exact plan
identifies complementarity, and the six value metrics were retained for consistency.
One bank-only readout exited zero in 9.50 seconds at 36,900 KiB, verified the bank
hash, row uniqueness, exact four-shard layout and arithmetic reconstruction within
3.33e-16. It performed no inference, source-array reread, bootstrap or training.

The [full fixed-stratum readout](evidence/g10-policy-complementarity-readout-20260912.json)
and [original audit summary](evidence/g10-policy-complementarity-summary-20260912.json)
are byte-identical copies of the completed artifacts. [Execution, validation and
review evidence](evidence/g10-policy-complementarity-progress-20260912.json)
pins the plan, runtime and local move bank. Runtime `5375064b56714b6b16f3a6c552efe75c7855386d`
is on open [PR #659](https://github.com/jjoshua2/DeepFin/pull/659), stacked on
[PR #655](https://github.com/jjoshua2/DeepFin/pull/655). Sixteen focused tests,
scoped host types with zero findings, configured whole Ruff/Vulture and independent
source/plan review passed. Related whole-project type timeouts remain unresolved;
this main documentation publication does not merge that code stack or imply a
whole-type pass. No active runtime was replaced.


## Remaining 28 shards completed and full bank qualified

The reviewed fresh v2 attempt collected original shards 4–19 and 20–31, preserving
qualified shards 0–3 and the original source namespace. The two chunks contain
131,072 and 98,239 real rows respectively: 229,311 new rows plus 32,768 retained
rows equals 262,079 across exactly 32 shards. Final shard 31 contains 8,127 rows;
one padding row completes its final fixed-32 call. Padding is not stored as a
source row. The new chunks made 4,096 and 3,070 calls.

Collection exited zero in 985.2826 seconds (individual chunks 549.5639 and
405.7090 seconds, plus the registered inter-chunk pause and supervision). The
previous v1 attempt failed before inference after 5.2257 seconds. Both attempts
remain preserved; combined collection elapsed time is approximately 990.5083
seconds within the original 2,400-second allocation. V2 conservatively used a
2,394-second inclusive cap, with a 2,364-second external TERM plus 30-second KILL
bound. No automatic retry or additional allocation was introduced.

The first fresh preflight stopped before collection because nonblocking GPU flock
returned errno 11. A subsequent read-only host lock inventory showed no GPU lock;
an exact-path probe then found the lease free. The final immediate preflight
recorded 3,364 MiB device use, 4% utilization, 51.99 GiB available RAM and
296.59 GiB free SSD, with the shared lease available. All observations, including
the failed preflight, are retained. No Windows process was stopped or changed.

The collector remained at `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`, with the
same fixed-32 profile, model and provider/library pins. No source-cache runtime
was adopted. The 8 GiB ORT allocator cap, 12 GiB sampled device guard, CPU 4–5,
two threads, RAM and 150 GiB disk reserve guards remained unchanged. The running
original20M CPU materializer was preserved.

After successful collection, the independently reviewed CPU-only saved-label
qualification exited zero in 16.19 seconds at 72,460 KiB peak RSS, within its
600-second ceiling. It audited all 28 new shards and reused the accepted first
four through pinned qualification, attributes, source and unchanged output
storage witnesses, without rereading their label payloads. The existing auditor
was copied with only exact per-spec row counts and corresponding fixed-32 padding
counts replacing its 8,192-row assumptions; payload hashes, row/roster identity,
provider execution, call accounting and loaded-library proofs remained required.
No raw source features, model or library payloads were rescanned by qualification.

The resulting policy-consumer manifest is
`scratchpad/bt4_joint20/g10_ceres_remaining28_v2/post_collection/completed/ceres_policy_manifest.json`,
SHA-256 `69101b506b70bab8f94ee6520eeab2f8080b8c59295f29612389dde0a2032f36`.
Its complete qualification receipt has SHA-256
`401f190c04a33d2403e0e7bb06c8781d45d44909fb6cecd682cb0feaa1bddeb4`.
Both value heads are retained in the authenticated shards. This establishes a
complete same-row diagnostic bank, not a trained corpus, a new value comparison
or playing strength. Existing four-shard scientific conclusions remain unchanged.

[Compact completion evidence](evidence/g10-ceres-complete32-20260912.json)
contains terminal records, counts, cumulative budget, preflight observations,
independent review and exact local artifact identities. Bulk shards remain under
`data/lc0/ceres_compact_sidecars/g10_run06_first4_v1/` and
`data/lc0/ceres_compact_sidecars/g10_run06_remaining28_v2/`; failed v1 artifacts
remain separate and are not admitted.

## September 12: complete next96 cohort collection launched

The existing collector driver launched at **2026-09-12T21:06:05.995196+00:00** for the complete,
already source-qualified run06 next96 cohort: **790,282 rows / 97 shards**.
This is a convenience cohort selected to expand the same-row teacher bank,
not a representative sample or another small teacher-agreement experiment.
The previous 262,079 G10 Ceres rows remain qualified. Successful collection and
saved-output qualification would raise that coverage to **1,052,361 rows**;
this launch does not establish the increase. The original 18,910,484-row Ceres
corpus is separate, and the diagnostic results above still concern first4 only.

Seven fresh chunks cover shards 0–15, 16–31, 32–47, 48–63, 64–79, 80–95 and 96.
The last shard contains 3,850 real rows and requires 22 padding rows: 790,304
fixed-batch input rows and 24,697 calls in total. Runtime `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`,
the C3 model, approximate fixed32 profile, ORT 1.29/NumPy 2.2.6/CUDA 13 libraries,
policy and both raw value-logit heads are unchanged. The newer source-cache
implementation was not adopted. No temperature fitting or backend-parity claim
is added.

The single 7,200-second allocation includes up to 6,600 seconds for the existing
owned collection driver and cleanup, then at most 600 seconds for saved-output
qualification. The latter also checks the original absolute deadline. Each chunk
retains its 1,200-second ceiling, clipped by driver time remaining. There is no
automatic retry. CPUs 2–3 and two numeric threads are used, with the existing GPU
lease, 8 GiB ORT allocator allowance, 12 GiB sampled device and per-process RSS
guards, a 32 GiB host MemAvailable floor and 150 GiB SSD reserve. Per-chunk output
and state remain bounded to 128 MiB. Memory checks occur every five seconds;
these are sampling guards, not peak measurements or hard process-RSS quotas.
The CPU-only qualifier has a 1 GiB address-space ceiling.

The parent reported 84 GiB available RAM, 12 GiB used, zero swap use and GPU 2,913 MiB/3%
at preflight, with no other GPU job. The actual driver start receipt is retained;
root 29510 is owned solely by parent completion observer 119. Existing CPU target
preparation and guarded SF generation are preserved. No collection outcome was
read for this publication.

The reboot changed the inode/ctime of the 942,048-byte WSL `libdxcore.so`; its
rehash matches the prior qualified digest. Current boot metadata is pinned,
and the unchanged collector still verifies actual mapped-library bytes and
provider execution. The final auditor retains all nine saved-array checks per
shard, call/padding accounting, row/source identity, provider and loaded-library
proofs. It uses the accepted next96 native-BT4 receipt and attributes for source
witnesses without rereading native arrays. A policy manifest will be assembled
from actual accepted Ceres attributes only after successful qualification;
training admission remains separate.

The prior 229,311-row collection took 985.2826 seconds, suggesting roughly 56.6 minutes
for the new cohort at unchanged throughput, before additional overhead. This is
an estimate, not a completion promise. Preparation passed ten metadata guard
cases, existing default validation and independent static review. A missing
fresh output parent in the first default attempt and a preparation timestamp
correction are recorded without implying data execution.

[Compact launch evidence](evidence/g10-ceres-next96-launched-20260912.json)
pins the actual driver receipt, reviewed plans and exact command. Local state is
`scratchpad/bt4_joint20/g10_ceres_run06_next96_v1/`; fresh outputs are under
`data/lc0/ceres_compact_sidecars/g10_run06_next96_v1/`.


## September 12: run06 next96 qualified; matching run07 collection launched

The run06 collection above completed at **2026-09-12T21:43:51.658683+00:00**
in **2,265.6635 seconds** (37.76 minutes), within its 6,600-second allocation.
All seven chunks exited zero. The saved-output qualifier passed, and the parent
observed exit zero for the complete collection-plus-qualification command.
The complete bank adds **790,282 rows / 97 shards**, bringing qualified G10 Ceres
coverage to **1,052,361 rows**, including the earlier 262,079. The original
18,910,484-row Ceres corpus remains separate.

The saved audit covers **873 arrays**, nine per shard, including compact policy
and both raw value-logit heads. Every chunk retains its first/final mapped-library
observations for 13 libraries, 403 CUDA neural-kernel events and four allowed CPU
shape events, with no missing final evidence or partial directories. The observed
24,697 fixed32 calls processed 790,304 input rows: 790,282 real and 22 padding.
Independent completion review checked the actual terminal records, manifest order,
head identities and saved proof summaries without repeating array audits or
reading source/model/library payloads.

The accepted policy-consumer manifest is
`scratchpad/bt4_joint20/g10_ceres_run06_next96_v1/post_collection/completed/ceres_policy_manifest.json`,
SHA-256 `a49e03dd01e53d1322f9d500e871591ffeed95b9442cc7673a1584daa67107fc`.
It identifies the qualified teacher bank; target materialization and training
admission remain separate. This completion adds no playing-strength result and
does not expand the earlier first4 diagnostic conclusions.

The matching run07 next96 collection launched at **2026-09-12T21:57:30.926921+00:00**
for **792,643 rows / 97 shards**. Six full 16-shard chunks plus one 6,211-row shard
require 29 padding rows: 792,672 fixed32 inputs and 24,771 calls. Its qualifier
selects the exact run07 cohort from the previously accepted 15-cohort native-BT4
qualification, preserving original row/source proofs and the same nine-array
Ceres audit. One copied run06 label in the prepared result limitations was
corrected to run07 before launch.

Runtime `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`, model, approximate fixed32
backend, heads and settings remain unchanged. The same 7,200-second total bound
applies: at most 6,600 for collection/cleanup and 600 for saved qualification,
with the original absolute deadline enforced. CPUs 2–3, two numeric threads,
8 GiB ORT allowance, 12 GiB sampled device/process guards, 32 GiB host-memory
floor and 150 GiB SSD reserve remain in force. These sampled guards are not
hard RAM quotas. The parent reported 83 GiB available RAM, zero swap use,
236 GiB free SSD and GPU 2,886 MiB/3% at preflight.

Run07's immutable driver start receipt records PID 70553. Parent exec session
46827 has sole completion observer 160; this publication reads its launch
receipt, not its evolving outcome. At that launch snapshot, run07 was not yet
included in qualified coverage. Successful completion and saved-output
qualification would raise G10 Ceres coverage to **1,845,004 rows**. No active observer was
polled for this publication.

[Compact completion and launch evidence](evidence/g10-ceres-run06-completed-run07-launched-20260912.json)
contains exact terminal/start records, per-chunk proof summaries, resource bounds
and artifact identities. Run07 state is
`scratchpad/bt4_joint20/g10_ceres_run07_next96_v1/`; outputs are under
`data/lc0/ceres_compact_sidecars/g10_run07_next96_v1/`.

## September 12: matching next96 complete; run06 large launched

Run07 next96 collection completed at **2026-09-12T22:36:36.462292+00:00**:
all seven chunks exited zero, with **792,643 rows / 97 shards** collected in
**2,345.535 seconds**. Parent root 46827 and sole observer 160 closed with exit 0.
Saved-output qualification passed the unchanged nine-array, provider/kernel,
loaded-library, source/row and call-count checks for all 97 shards. The final
6,211-row shard has 29 padding rows, yielding 24,771 fixed32 calls. The resulting
policy manifest has SHA-256
`f39b4ef89aab38bd2ae6068e55568b6a66b07195dd4a3c17c15e739a2cb9708e`.
Independent compact review checked exact completed layouts and proof records;
no arrays or model were reread.

Qualified G10 Ceres coverage is now **1,845,004 rows / 226 shards**:
262,079 original-increment rows + 790,282 run06 next96 + 792,643 run07 next96.
This remains separate from the original 18,910,484-row Ceres corpus. Both value
heads are retained; target materialization and training admission remain separate,
and the earlier first4 scientific results are unchanged.

The complete original `G10_common_large_v1/run06_g10` source collection then
launched at **2026-09-12T22:39:55.902706+00:00**, with actual driver PID 84779,
parent root 99754 and sole completion observer 172. It plans **2,013,019 rows /
246 shards** in fifteen 16-shard chunks plus a final 6-shard chunk. Last shard 245
contains 5,979 rows and requires 5 padding rows: **62,907 calls / 2,013,024 inputs**.
Only successful collection and saved qualification would raise G10 coverage to
**3,858,023 rows**. This launch adds no completed rows and does not repair the
separately failed adaptive-SF large derivation or relabel its frozen inputs.

The existing schema 2 native-WDL admission supplies an exact 246-shard mapping
across four genuine native output roots. The prepared qualifier preserves that
mapping and all saved Ceres checks; it creates no synthetic combined native
namespace. Runtime `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`, approximate fixed32
profile, model, policy and both value-logit outputs remain unchanged; no source
cache was adopted.

The **2h40 / 9,600-second inclusive allocation** permits 9,000 seconds for the
existing collection driver and cleanup, then at most 600 seconds for the CPU-only
saved audit, which also enforces the original absolute deadline. The measured
run06 next96 rate projects about 96.2 minutes for this cohort; that whole-run
extrapolation already includes its baseline pauses and is not a guarantee.
Existing per-chunk 1,200-second bounds, shared GPU lease, CPUs 2–3/two threads,
8 GiB ORT allowance, 12 GiB sampled device/per-process RSS guards, 32 GiB available
host-memory floor and 150 GiB SSD reserve remain in force. Output/state is capped
at 128 MiB per chunk; the final qualifier has 1 GiB address space. Samples do not
establish peaks or hard process-RSS quotas. No retry or automatic extension is
allocated. Parent preflight reported 84 GiB available RAM, zero swap use, 227 GiB
free SSD and GPU 2,933 MiB/9%. Existing CPU preparation and generation were preserved.

[Compact completed-run07 and large-launch evidence](evidence/g10-ceres-run07-completed-large-launched-20260912.json)
pins the terminal qualification, manifest, independent review, new plans and
actual start. Large state is `scratchpad/bt4_joint20/g10_ceres_run06_large_v1/`,
with fresh outputs under `data/lc0/ceres_compact_sidecars/g10_run06_large_v1/`.
No active observer was polled or new outcome read for this publication.


## 2026-09-13 UTC: run06 large collection and qualification complete

The complete `G10_common_large_v1/run06_g10` collection finished all **16 chunks,
246 shards and 2,013,019 real rows**. The driver reports **6,106.741 seconds
(1h41m46.741s)**, ending at **2026-09-13T00:21:42.643433+00:00**. This is the
collection-driver duration, including its inter-chunk pauses; a separate qualifier
duration is not recorded in the saved qualification JSON. Parent captured root
99754 / sole observer 172 closed with exit 0 after the automatic saved audit.
The registered allocation remained 9,600 seconds inclusive of collection, cleanup
and qualification; no retry or extension was used.

All **62,907 fixed32 calls** account for **2,013,024 input rows**, including exactly
**5 padding rows** in the final 5,979-row shard. The saved auditor passed all
**2,214 array proofs (nine arrays per shard)**, retaining policy and both raw value
heads, source/feed and storage bindings, provider placement, and first/final
mapped-library evidence. The existing schema 2 native admission retained the exact
246-shard mapping across four genuine output roots. No native arrays were reread
for this publication. The qualified policy manifest is
`74a0c76bc8b3fe0b20dc3ec77e259810fee0cae3ca55a21b3b3637f5efb6ed36`.

This raises qualified G10 Ceres coverage from **1,845,004 / 226** to
**3,858,023 rows / 472 shards**. The original 18,910,484-row Ceres corpus remains
separate. Runtime `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`, approximate fixed32
profile, model and both value-head settings are unchanged; no source cache was
adopted. Collection establishes saved teacher coverage, not native-backend parity,
training admission, representative G10 performance or a strength result. It does
not repair the earlier adaptive-SF derivation, and the first4 scientific readouts
remain unchanged. No additional Ceres collection is queued by this record.

[Compact completion evidence](evidence/g10-ceres-run06-large-completed-20260913.json)
pins the actual driver, qualification, policy manifest, source admission and
compact completion review. Review checked all 16 completion receipt hashes,
ordered shard/count agreement and the retained runtime proofs without repeating
array, model, library, test or active-job reads. The separately registered Downside
training proceeds under its own parent-owned launch; this collection record adds
no observation of that training.
