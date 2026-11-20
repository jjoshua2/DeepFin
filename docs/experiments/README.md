# Experiment records

- [2026-09-19: Workspace storage migration and queued-job adoption](2026-09-19-workspace-storage.md)

Bend engine records and their evidence live on the `bend` branch (see
[development](../development.md#bend-branch)); tag `pre-slim-20261113` holds the
last `main` that carried them.

Start here for experiment planning and readouts. [Project guidance](../../CLAUDE.md)
holds durable constraints; [evaluation](../eval_protocol.md) explains which
measurements support which claims.
Gaps that remain after a pull request merges are listed in [pull-request follow-ups](../pr_followups.md).
Evidence for records before 2026-10-25 that no staying code, current doc or later record reads (and that is not Bend-related) moved out of git on 2026-11-13: it lives in the `repo-slim-20261113` artifacts archive on the project host under the same relative path (`docs/experiments/evidence/...`, `docs/experiments/artifacts/...`, listed in its `MANIFEST.sha256`) and is recoverable from tag `pre-slim-20261113`, e.g. `git show pre-slim-20261113:docs/experiments/evidence/<path>`. A record link to such a file is therefore dangling in a fresh checkout by design.
One-off research drivers in `scripts/` that only pre-cut-off records or the frozen ledger cite (and their dedicated tests) moved out of git on 2026-11-14 to the same archive under `scripts/` and `tests/`; recover one with `git show pre-slim-20261113:scripts/<name>`. See [`scripts/README.md`](../../scripts/README.md).
For future bootstrap choices, see the [adaptive decision policy](2026-09-08-bootstrap-next-value-and-policy-contrasts.md#september-13-adaptive-decisions-for-the-remaining-bootstrap-research): registered result classification and the next research decision are separate.

## Records

- [2026-11-20: Loop experiment 1 — one self-play generation from the SL base (DRAFT, prospective)](2026-11-20-loop-exp1-one-generation-from-sl-base.md) — awaiting owner approval, nothing launched: one collection (16,000 games with snapshot gates, else 4,000) trained three ways on the same rows (policy only = default primary; search value; softened policy), gates before any arena, one deciding seat-balanced 3,200-game match vs the SL base read once (success = lower 95% bound > 0), unconditional positive control from Q50 c12, dispositions for every outcome, nine owner decisions with defaults

- [2026-11-20: Gumbel leaf buffer full — root-Q fallback replaced by flush-and-resume](2026-11-20-gumbel-buffer-full-fallback.md) — search-correctness fix (ABI 11): a leaf that found the capped encode buffer full was backed up with the root's Q (wrong-signed at odd depths); now flushed and evaluated, counted per search and banked by collect/arena. Reach audit: only the simple-RL smoke collect (cap 512, 30.6% of leaf visits absorbed) was contaminated; P1, head-swap, F12/Q50, rl-audit banks and production cannot bind (GPU rerun at P1 settings: 0 absorbed). Byte-identical when the cap does not bind.

- [2026-11-19: Twofold repetition as a proven draw — semantics, footprint, defects, plan](2026-11-19-twofold-repetition-as-proven-draw.md) — investigation only, no self-play code change: the C search proves a draw on the first repeat counting pre-root game history (lc0 does not); 3.9% of banked training rows have such a child and ~25% of them store q exactly 0 (~1% of all rows; calibrated vs outcome on average, partly self-fulfilling); UCI PUCT/walker reroot keeps a stale DRAW root (fix in #1421); the repeated position is never searched; B1 (root boundary) then B2 (no proof) proposed, each decided by a frozen deep-SF target audit, awaiting owner

- [2026-11-19: Winning-root draw pruning — no fake forced moves, below-zero draw fallback](2026-11-19-root-draw-pruning-singleton-fix.md) — search-correctness change; self-play footprint measured on 497,027 banked plies: 3.1 stored targets and 1.2 played moves per 10,000 plies change, all other roots bit-identical; no training experiment
- [2026-11-19: RL takeover — diagnostics and a same-data policy-only ablation](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md) — search gain measured (+849 Elo 64 vs 1 sim, +272 for 256 vs 64, +478 for 800 vs 64), no pipeline defect found in F12 cycle 1, policy-only ablation registered before any update — **P1 read: value loss implicated — policy-only +42.6 Elo (+30.1 to +55.2) over F12 cycle 12 on the same rows; −10.9 (−27.7 to +5.9) vs the SL base.** D4 round 2: no candidate-seat offset detected (pooled same-net +6.4, −4.2 to +17.1), so P1 stands against 0; Q50's loss is nearly all value, F12's splits about evenly; a post-hoc value recalibration (diagnostic, not pre-registered) recovers about 85% (0.50–1.18) of Q50's value-output play deficit.

- [2026-11-16: Existing RL failure evidence audit](2026-11-16-existing-rl-failure-evidence-audit.md) — completed existing-evidence synthesis: update-regime sensitivity and qualified conditional regressions; retained-search target negative in its tested use, causal head/mechanism unresolved; no new measurements or experiments selected.

- [2026-11-16: Integrated whole-game tools and original-A39 collector transport](2026-11-16-current-i-whole-game-tools-collector-engineering.md) — closed394-case software, current-source A39 CPU assets and17-attempt/two-ply public HOLD transport plus expected-refusal raw-history audit; no accepted bank, scientific cohort, strength or main-runtime adoption.

- [2026-11-16: Original A39/B32 public policy-anchor learner engineering](2026-11-16-a39-b32-public-anchor-engineering.md) — closed synthetic engineering: original A39, two public B32 updates per λ0/0.1 arm, own-live-to-saved CPU state and matched journals/RNG accepted; fictional bank/outcome authority, no causal strength, transport or research qualification.

- [2026-11-16: Fresh cache, complete-parent roles and starts preparation](2026-11-16-fresh-whole-game-preparation-readout.md) — accepted metadata preparation: fixed 16,384-entry cache, 16,368 viable; 192 TRAIN / 64 HOLD and six panels of 400 pairs and exact starts; no scientific collection, learning or playing.

- [2026-11-16: Fresh whole-game initial-policy anchor pilot](2026-11-16-fresh-whole-game-policy-anchor-pilot.md) — fixed three-arm design with selected 109-field source/profile and accepted roles/starts; final MAIN/phase authority still required, no scientific execution.

- [2026-11-16: Frozen-reference policy anchor software and asset qualification](2026-11-16-frozen-policy-anchor-software-qualification.md) — CPU software qualified with 38 fresh + 235 carried unique cases; full494 synthetic CUDA and genuine A39 CPU assets separately accepted by root and independent actual reviews. No real bank, collection, recipe, strength or adoption claim.

- [2026-11-15: Full494 accepted-bank learner and CUDA actor handoff](2026-11-15-full494-bank-handoff-software.md) — closed software qualification: one real public-CLI update, genuine closed-parent actor reload, both invalid-tag refusals and one BF16 inference; synthetic initial authority only, no real bank, sustained loop, adoption or playing-strength result.

- [2026-11-15: Fixed-bank SF current-P0 main-policy 512-update pilot](2026-11-15-sf-current-p0-main-policy-512-pilot.md#2026-11-15-accepted-screens-and-user-cancellation) — two accepted NONPASS screens (B/A 0.416; B/A39 0.4265); remaining diagnostic and planned audit cancelled by user; no confirmation, adoption or resume.

- [2026-11-15: A39/Lc0 joint GPU and timed-driver engineering](2026-11-15-a39-lc0-joint-timed-engineering.md#closed-readout--november-15-2026) — FAILED after 419.549/900 GPU seconds: 13 controls and reciprocal work passed, but only one game completed; Lc0 aborted with a native resource error. Separate CPU diagnosis accepted; cause unknown, no joint/timed or strength qualification and no automatic retry.

- [2026-11-15: Retained-search WDL target census of the fixed G1 bank](2026-11-15-retained-search-target-bank-census.md) — CLOSED QUALIFIED: all 35 local controls and 256 games / 24,827 rows passed in 36.32 CPU seconds, zero GPU. Archived-to-normalized aggregate residual changes are tiny; disagreement with recorded outcomes is much larger far from the stop. Finite-bank descriptions only; no causal, training, strength or adoption claim.

- [2026-11-15: Retained-search main-WDL fixed-G1 512-update pilot](2026-11-15-retained-search-mainwdl-512-pilot.md) — CLOSED NONPASS: B scored .453 against matched A and .4555 against A39, each over 1,000 games / 500 pairs with paired 95% upper bounds below .5. Both endpoints technically qualified; no confirmation or adoption, donor retained.

- [2026-11-14: A39 versus pinned eight-thread Stockfish timed baseline](2026-11-14-a39-sf-timed-baseline.md) — closed qualified descriptive 60+1 baseline: 0W/4D/4L over eight games/four pairs, score 25%; registered interval unavailable because all four pairs scored draw/loss. Separate engineering pair qualified. Donor retained; no learning gain, Elo, parity or milestone claim.

- [2026-11-14: Outcome-only A512 versus A39 control diagnostic](2026-11-14-a512-a39-control-diagnostic.md) — closed qualified UNRESOLVED: 300W/413D/287L over 1000 games / 500 pairs, 50.65% with paired approximate 95% interval [48.3480%, 52.9520%]. Donor retained; no confirmation, adoption, gain or equivalence claim.

- [2026-11-14: Fixed-G1 current-P0 native SF-WDL .2 pilot](2026-11-14-sf-current-p0-native020-pilot.md) — both fresh-A39 alpha0/.2 arms completed 512 updates with matching journals and accepted saved-state audits; common HOLD64 is descriptive only. Both qualified screens were NONPASS: 50.1953% versus A512 over 256 games and 51.1719% versus A39 over 512. Pilot closed; donor retained, unused confirmation protected, no gain, adoption or optimal-recipe claim.

- [2026-11-14: Actual A39/G1 SF-P0 consumer qualification](2026-11-14-sf-current-p0-a39-g1-qualification.md) — four real-network B256/BF16 engineering updates completed under matched structural controls; the positive branch exercised qualified, invalid and unavailable rows with measured health coverage. Root and independent actual acceptance; no research-dose, model-equivalence or strength claim.

- [2026-11-14: SF-P0 CUDA qualification and repeatability](2026-11-14-sf-current-p0-cuda-qualification.md) — original exact-zero GPU control failed; separate eight-update diagnostic found old-versus-old default-CUDA variation; two separately scoped positive teacher assertions passed. Original NONPASS retained; no model-equivalence tolerance, A39/G1/AMP qualification or strength claim.

- [2026-11-14: Fixed-G1 SF-P0 main-WDL finite-consumer qualification](2026-11-14-sf-current-p0-finite-consumer-qualification.md) — 22 new consumer and 21 legacy CPU controls passed with zero skips; exact zero bypass, masked outcome fallback and strict same-family continuation. The recorded cutoff does not qualify direct CLI, original-G1 model use or GPU behavior; no recipe, learning or strength admission.

- [2026-11-14: Current-P0 SF sidecar engineering qualification](2026-11-14-sf-current-p0-sidecar-qualification.md) — 33 focused tests, the four-row CLI chain and complete 256-game / 24,827-row original-G1 carrier passed; root data-only acceptance and independent actual review accepted. All rows/masks retained; no training recipe, model or playing-strength adoption.

- [2026-11-14: Banked original-G1 Stockfish P0 evidence census](2026-11-14-sf-banked-g1-p0-evidence-census.md) — completed data-only audit of all 256 games / 24,827 rows: 24,205 internally consistent source-context joins, 127 invalid joins and 495 unavailable. Recorded field presence and explicit regret are not teacher-target/training eligibility; independent actual census and metadata reviews accepted, no recipe or playing-strength adoption.

- [2026-11-14: Full-state lower-rate on-policy continuation](2026-11-14-low-lr-on-policy-continuation.md) — first leg closed at G8 / step 4,096: qualified terminal nonpass, 46.2890625% versus A39 over 512 games (paired 95% interval 43.1009–49.4772%). No C1 eligibility/allocation/consumption, conditional second leg, retry or adoption; study donor/incumbent unchanged and broader goal active.

- [2026-11-14: Fixed LOW512 versus original A39 successor](2026-11-14-low512-donor-screen.md) — completed qualified screen nonpass: 48.9258% versus A39 over 512 games (paired 95% interval 45.8184–52.0332%, includes parity). No confirmation eligible, allocated or consumed; donor retained, no playing gain or equivalence claim.

- [2026-11-13: Fixed G1 lower-LR probe and G2 WDL adjunct](2026-11-13-fixed-g1-lr-g2-wdl.md) — completed: LOW512 scored 57.03125% versus banked HIGH512 on reused diagnostic pairs (95% interval 52.0518–62.0107%); whole-G2 descriptive NLL/Brier order LOW512, A39, HIGH512. No A39 playing-gain, online recovery, optimal-LR or adoption claim; original baseline negative preserved.

- [2026-11-13: Core policy and resolved-outcome baseline](2026-11-13-core-outcome-policy-baseline.md) — completed leg-1 terminal nonpass: G8 scored 21.875% versus original A39 (paired 95% interval 19.0304–24.7196%) after 4,096 updates / 2,048 games. No eligible or consumed confirmation, C1, second leg or adoption; long-term strength goal remains unmet.

- [2026-11-13: Larger fresh-data outcome regime](2026-11-13-rl-larger-outcome-regime.md) — E4 completed: 192 games and 129 updates per arm; all three screens technically qualified, bundle contrast unresolved, neither arm donor-utility eligible. No confirmation or adoption; donor retained.

- [2026-11-13: TRAIN objective-gradient geometry](2026-11-13-rl-train-gradient-geometry.md) — completed E3 diagnostic on 32 fixed TRAIN roots; primary alignment effect unresolved, no playing or recipe claim.

- [2026-11-13: Resolved-outcome RL value bundle](2026-11-13-rl-resolved-outcome-bundle.md) — completed 40-update paired study with strict outcomes and three qualified playing screens; no registered gain, donor retained.

- [2026-11-12: E2 outcome-target bundle comparison](2026-11-12-e2-outcome-bundle.md) — preregistered FULLA search-only versus FULLB 20% genuine-outcome WDL/categorical bundle; scientific inputs frozen, execution separately gated, no research result or adoption.

- [2026-11-12: RL research program, first week](2026-11-12-rl-week-one.md) — ACTIVE November 12–19: establish a multicore external-engine baseline, reuse existing diagnostic ownership, and measure a justified controlled learning change; long-term strength objectives remain unmet.

- [2026-11-12: Epoch-two RL cap-40 exposure experiment](2026-11-12-epoch2-rl-cap40.md) — separate preregistered eight-window cap-40 test on fresh donor state and new source-ID-aware panels; closed: 320-update endpoint passed HEALTH but regressed on its independently validated 256-game screen (score 0.375); no confirmation or adoption. Accounting and resource closure recorded.

- [2026-11-11: Epoch-two RL transition](2026-11-11-epoch2-rl-transition.md) — qualified donor handoff and sequential RL loop; first experiment closed with an inconclusive 1,000-game confirmation and no confirmed gain. Compact evidence is recorded; the separate cap-40 follow-up has its own record.

- [2026-11-11: MultiPV rerank parser fix and bound clamping](2026-11-11-multipv-rerank-parser.md) — data-affecting SF label change (~15% of real MultiPV-6 searches in the review bank); not adopted, adoption condition stated.

- [2026-10-12: Frozen I-opt and completed-Q mechanism screen](2026-10-12-frozen-iopt-diagnostic.md) — opt-in native-root trace, counterfactual policy reconstruction, observed-label coverage and preregistration; no GPU readout or training.

- [2026-10-10: Shared teacher async plumbing and bounded first increment](2026-10-10-shared-teacher-async-increment.md)

- [2026-10-04: Generic arena resume result validation](2026-10-04-arena-resume-result-validation.md)

| Record | Scope |
| --- | --- |
| [Indexed bootstrap dose preparation](2026-10-04-indexed-bootstrap-dose-preparation.md) | Original-shard nested row loading, bounded metadata and CPU regression evidence; full scientific qualification pending |
| [Local arena adapter reuse source packet](2026-10-10-local-arena-adapter-reuse.md) | Reviewed operational adapter reuse, 11 isolated CPU contracts and future-launch preparation; no live adoption |
| [Audited 50M bootstrap expansion](2026-09-17-audited-scale-bootstrap.md) | Historical saved-audit V50 and explicit 50–60M contract; missing source binding and isolated scheduler recovered with both histories, no new experiment or training launch |
| [D-lite paired-pack auditor exit-aware wait](2026-10-01-sf-dlite-auditor-exit-wait.md) | One-hunk supervised-wait candidate and real-child CPU checks; no live audit, data, or admission credit |
| [Ceres direct-compact synthetic sink](2026-09-30-ceres-direct-compact-synthetic-sink.md) | 26-arm tiny full-head synthetic CPU screen: noisy +2.000% median sink-wall saving, semantic ZIP parity, no live adoption or measured production gain |
| [Ceres owner target and compact-order correction](2026-09-30-ceres-owner-target-and-compact-order.md) | CPU source-game target reconstruction, one real selected-row overlap and a zero-credit diagnostic legal-order fix; no saved-call or production gain claim |
| [Persistent Stockfish scalar cost](2026-09-29-sf-scalar-persistent-cost.md) | Frozen 10k-row persistent one-thread d6/d8/d10 cost/agreement screen with cold TT per row/depth and zero corpus/training credit |
| [Ceres stored-feed-once root preparation](2026-09-29-ceres-root-feedonce.md) | Tracked CPU root binder exercised by saved-game readback, exact 2,022-root replay parity and narrow 16.81% saved-root prep saving; fresh actor adoption pending |
| [Ceres saved19 negative-threefold CPU gate](2026-09-30-ceres-saved19-threefold-gate.md) | 19 saved full-history games / 2,022 roots: exact natural/strict outcome and root parity; 1.3706x narrow gate timing, no whole-root or live gain claim |
| [D-lite historical donor adaptation](2026-09-30-sf-dlite-historical-donor-adapter.md) | Main-archived patch against pinned historical game-epoch runtime; exact donor/paired-pack/consumed-batch gates CPU-reviewed, no training or strength credit |
| [Ceres fixed32 WSL recovery](2026-09-30-ceres-wsl-recovery.md) | Resumed exact chunks 0–6 after WSL crash; first new chunk 7 source, strict, ledger and archive PASS with zero admission; current boot uses bounded shared physical I/O telemetry |
| [Ceres run09 first durable chunk](2026-09-30-ceres-run09-first-durable-chunk.md) | Sealed 0–50 prefix reused; fixed32 chunk 51 CUDA and local durable markers pass, 125,489 unique feeds, zero admission; independent external archive audit pending |
| [Ceres fixed32 source timing observation](2026-09-30-ceres-source-phase-observation.md) | Closed run-8 phase evidence, one independently replayed ZIP shard, and a CPU-checked observation-only successor patch; no live change or corpus credit |
| [SF / BT4 / Ceres teacher adjudication](2026-09-11-teacher-adjudication-audit.md) | Broader saved-search audit of policy regret, ranking constraints, routing, optional Ceres complementarity/value calibration and a bounded Ceres-on-G10 selection; no inference/training result |
| [Corrected full58k BT4/Ceres selected targets](2026-09-30-full58820-corrected-selected-targets.md) | Banked CPU reconstruction and independent direct readback of all 58,820 targets; corrected Ceres batch-profile top-1, zero corpus or strength credit |
| [Ceres active128 frozen-source quality](2026-09-30-ceres-active128-frozen-quality.md) | 52,820 exact shared feeds, corrected FP16 target TV maximum 0.226679 fails the frozen 0.05 policy gate; source and CPU reader timings, no promotion |
| [Ceres live128 proof-writer A/B/B/A](2026-09-29-ceres-live128-writer-abba.md) | Four full-prefix 128-game arms, exact semantic/all-attempt parity and 8.75% source-stage saving; no full-pipeline or 500M gain claim |
| [Full 58,820-row BT4/Ceres batch profiles](2026-09-29-full58820-batch-profiles.md) | Separate audited BT4 and Ceres four-arm fixed-input screens: 28.558% and 74.070% call-core savings; numerical quality bounds, no integrated throughput or corpus credit |
| [D-lite scalar SF value plan](2026-09-29-dlite-scalar-value-plan.md) | Prospective separate value contrast, representative one/eight-engine cost screen and shared CPU/GPU month-budget limits |
| [Selected-E D-lite value preregistration](sf_dlite_selected_e_value_prereg_20260930.md) | Frozen 2.5M-row value-only contrast, direct depth-12 target audit, matched donor adaptation and 576-pair strength decision; no training or Elo result |
| [D-lite durable paired arena resume](2026-09-30-dlite-paired-arena-resume.md) | Opt-in source-bound, fsynced 576-pair resume with atomic pair receipts and an outer 55-minute attempt cap; CPU recovery proof only, no strength result |
| [D-lite depth-8 label operations](2026-09-30-sf-dlite-d8-label-operations.md) | Three bounded full-label attempts stopped at conservative RSS guards; fourth launch and its later parser failure are recorded in the successor amendment, with zero admitted credit |
| [Independent Selected-E/D-lite native-pair pack audit](2026-09-30-sf-dlite-native-pair-auditor.md) | 41-test integrated auditor with fresh exec guard, active-turn coverage proof and ≤30-minute resumable units; first builder shard sealed, complete-pack audit/training pending |
| [D-lite depth-8 value eligibility amendment](2026-09-30-sf-dlite-d8-value-eligibility-amendment.md) | Prospective identical-value UCI re-emission rule after full04 parser HOLD; full05 held on a genuine calibrated-value change and has zero admitted labels |
| [D-lite depth-8 first-target eligibility](2026-09-30-sf-dlite-d8-first-target-eligibility-amendment.md) | Historical prospective full06 rule keeps the first exact depth-8 value, validates later scored emissions and banks churn; full04/full05 zero credit, no admission from the amendment |
| [D-lite depth-8 full07 operational readout](2026-09-30-sf-dlite-d8-full07-operational-readout.md) | Historical full06 I/O device-set stop and fresh full07 launch with additive-device meter; zero admission, later recovery and audit recorded separately |
| [Full07 Stockfish CPU placement and guarded recovery](2026-09-30-sf-dlite-full07-cpu-placement.md) | Fsynced affinity amendment; three PSS stops followed by same-identity completion and independent source/roster/raw-D8 audit of all 2.5M rows; zero admission and training credit |
| [D-lite native first-three-pair cost and exit-aware wait](2026-10-01-sf-dlite-native-first3-and-exit-aware-wait.md) | Nine-JSON manifest arithmetic for 19,526 rows and independently reviewed CPU wait fixture; no pack qualification, live builder adoption or throughput claim |
| [D-lite small-bank scalar path](2026-09-29-sf-dlite-smallbank-path.md) | Audited 128-row d8 raw labels and 92-row B/C selected-value attachment smoke; no full-128 target, corpus, training or Elo credit |
| [Tracked Ceres saved-game source fixture](2026-09-29-ceres-saved-game-source-fixture.md) | Main CPU producer/readback entrypoint exercised on one 47-row saved natural game; complete feature/feed/legal/terminal parity, no generation or throughput credit |
| [Ceres stored-feed batch conversion](2026-09-29-ceres-stored-feed-batch.md) | CPU-only exact-byte feed proof on eight archived calls plus 512 stratified rows; 77.875% narrow conversion-stage reduction, no integrated throughput credit |
| [500M practical next tests and target storage](2026-09-29-500m-practical-next-tests.md) | Current-source cost pilot, larger-batch numerical variants, sparse-SF holdout and pack/trainer follow-through; exact old-cohort sparse storage screen |
| [Three-source selected-target diagnostic pipeline](2026-09-29-tri-source-selected-target-pipeline.md) | Audited 83,416-winner BT4/Ceres/SF-origin selected-target ZIP pack, missing-head cost and physical storage bounds; diagnostic zero corpus credit |
| [Tri-source replay storage checkpoints](2026-09-30-tri-source-storage-checkpoints.md) | Synthetic paired-wave restart fixture and independent 8,192-row BT4 native-byte compression audit; fixed-record/external-sort design remains unimplemented |
| [Lossless Stockfish raw-observation frames](2026-09-30-sf-raw-zstd-frames.md) | CPU-only Zstd checkpoint prototype and one failed full05 747-row byte-parity pilot; raw input retained, no corpus or external-drive throughput credit |
| [Bounded tri-source checkpointed sort foundation](2026-09-30-tri-source-checkpointed-sort-foundation.md) | CPU-only 100-byte candidate metadata and resumable sort parts, including SIGKILL tests; recursive verification and live source/target integration remain open |
| [Bounded tri-source shared verification](2026-09-30-tri-source-shared-verification.md) | CPU-only process-scoped capped part verification; synthetic no-op reads each distinct part once, while full-run validation and production source joins remain open |
| [Bounded tri-source verification rollover](2026-09-30-tri-source-verification-rollover.md) | CPU-only 129-run synthetic ancestry crosses the 128-summary cache limit and survives a final-merge SIGKILL/resume; rereads grow and 500M source integration remains open |
| [Literal-UID tri-source cursor diagnostic](2026-09-30-tri-source-literal-uid-cursor.md) | CPU-only bounded whole-game reader interface and paired segment/100-byte metadata/sort input; two crash boundaries and direct truncated-segment refusal, no archive admission |
| [BT4 one-game NPZ cursor binding](2026-10-01-bt4-one-game-npz-binding.md) | Pinned one-game replay/witness bridge with owned CPU deadline and complete original proof checks; no actual archive parity, target/trainer admission or corpus credit |
| [Full512 old-cohort selected/dual ABBA cost](2026-09-29-full512-old-cohort-abba-cost.md) | Four-arm same-row screen: 38.888684% selected complete-arm saving passes the 15% cost gate; zero corpus or 500M credit, representative pilot next |
| [Full512 58,820-row whole-bank selected/dual ABBA cost](2026-09-29-full512-whole-bank-abba-cost.md) | Independently audited 43.936202% mean complete-arm saving across two pairs; zero corpus or 500M credit |
| [E versus D on shared 576-opening strict bank](2026-09-29-e-vs-d-576-strict.md) | Direct original E−D: −12.37 Elo [−27.18, +2.39], preregistered result unresolved; 1,152 audited games and rule50-aware six-man replay |
| [Selected-E versus D direct strict arena](2026-09-28-selected-e-vs-d-direct-strict-readout.md) | Completed 576-pair strict six-man comparison: Selected-E −19.02 Elo [−33.43, −4.68] versus D; preregistered 30-Elo gate inconclusive |
| [Selected-E versus E seed-121 strict arena](2026-09-27-selected-e-vs-e-arena.md) | Preregistered 576-pair noninferiority gate passed; future one-teacher cost screen eligible, with no measured annotation saving or corpus credit |
| [500M source/target candidate after E–D](2026-09-29-500m-source-target-candidate.md) | Cheap BT4/Ceres targets, a separate SF-origin source test and sparse SF correction pilot; 37.045 ideal GPU-stage days before qualification and reuse |
| [Overlay validation reuse](2026-09-21-overlay-validation-reuse.md) | Future exact-epoch startup: nine-to-one semantic validations per shard; 5.42× faster constructor stages on eight real shards with exact plan/target parity; active runtime unchanged |
| [SF-free BT4+Ceres targets](2026-09-21-sffree-targets.md) | Reviewed E builder and three real source families qualified; bounded full 58M CPU preparation launched; training pending |
| [58M policy/value teacher factorial](2026-09-19-58m-policy-value-factorial.md) | Reviewed pipeline queued:22.78M missing Ceres labels, target overlays, four fresh58M arms and four direct matches; active continuation preserved; no factorial results |
| [Factorial58 CPU preparation overlap](2026-09-20-preparation-overlap.md) | Historical teacher-labeling/CPU-preparation overlap, inherited writer ownership and bounded probe evidence; publication recovery requires fresh qualification before adoption |
| [B100 Tactical300 policy transfer](2026-09-11-b100-tactical300-transfer.md) | Candidate sparse SF mass-transfer policy tooling with mandatory deeper-SF calibration before materialization or training |
| [Tactical300 saved-depth diagnostic](2026-09-11-tactical300-calibration.md) | Provenance-gated d9/BT4 disagreements against saved d10/d12; diagnostic tooling only, not a published-transfer calibration or training admission |
| [Ceres saved Syzygy source fixture](2026-09-29-ceres-saved-syzygy-source-fixture.md) | Tracked CPU producer/readback reproves two saved six-man WDL/DTZ game results; zero generated-row credit |
| [Ceres saved32 proof-reuse writer ABBA](2026-09-29-ceres-saved32-writer-abba.md) | Eight CPU arms: same 4,751 decoded rows and publication records, 34.48% faster saved32 writer interval; no live source-generation gain |
| [Ceres selected-label guard cadence](2026-09-27-ceres-selected-guard-cadence.md) | Four-arm selected-4K GPU A/B/A/B: exact target parity and 17.76% lower arm wall; no production or 500M throughput claim |
| [Ceres v8 feed-store CPU screen](2026-09-27-ceres-v8-feed-store-screen.md) | Bounded synthetic all-attempt feed Mapping/RSS screen, sparse and incompressible; frozen v8 unchanged, no full-path or adoption claim |
| [BT4/Ceres full-prefix source pilots and 500M capacity](2026-09-27-500m-source-capacity.md) | Strict complete-game source rates, corrected BT4 batching histogram, illustrative 30-day gap and zero-credit mixed-source admission gate |
| [Owner versus fair teacher route preregistration](2026-09-28-owner-vs-fair-teacher-route-prereg-v4.md) | Independently reviewed source-only matched-row protocol and conditional owner-reuse arithmetic; registered launch held, zero credit |
| [Ceres v11 sink 10% optimization screen](2026-09-28-ceres-sink-10pct-screen.md) | Read-only saved-receipt/source bound: no safe single low-risk verifier edit proves 10% source-wall saving; timer-only next step |
| [BT4 root worker CUDA qualification](2026-09-23-bt4-root-worker-gpu-qualification.md) | Preregistered tiny real-model 7-to-6 correctness and accepted-row writer-cost pilots under shared GPU lease; no GPU run yet |
| [BT4 opt-in root-policy worker](2026-09-23-bt4-root-policy-worker.md) | Experimental CPU-capable ONNX generator and atomic whole-game writer with strict rule50 six-man outcomes, raw teacher retention and >=7-piece rows; fake CPU checks only, no searched actor or replay consumer |
| [BT4 in-memory root-policy stepper](2026-09-23-bt4-root-policy-stepper.md) | Batch-friendly root-policy game state with explicit historical or rule50-aware six-man outcome mode, exact float32 source retention, outcome backfill and whole-game discard; CPU fakes only, no generator wiring |
| [AVI-style one-ply value backup](2026-09-11-avi-value-backup.md) | One-ply evaluator, provenance-gated sidecar and value-only rewrite implemented; no executed AVI corpus, training or strength result |
| [Exact gather CPU screen](2026-09-22-exact-gather-cpu-screen.md) | Exact preallocated gather CPU prototype/profile; isolated speedup measured, runtime adoption deferred pending end-to-end qualification |
| [Bound packed output accounting](2026-09-22-packed-output-accounting.md) | Removes repeated output walks during input verification; existing caps retained, no new GPU launch |
| [Storage observer retry and BT4 results](2026-09-22-packed-observer-retry.md) | First-batch observer schema failure preserved; corrected real-sampler regression and fresh retry; BT4 exact pipeline screen passed |
| [Same-ZIP external trainer screen](2026-09-29-zip128-external-trainer-screen.md) | One 128-shard actual-trainer pair passes the ≥0.95 short-path external/local rate rule at 0.9963; cache-affected, zero 500M capacity credit |
| [Host-overlap runner preparation](2026-09-22-host-overlap-runner.md) | Frozen-runtime CPU qualification and bounded OFF/ON GPU pair tooling; no full-stream or GPU launch yet |
| [Selected-E sparse source audit](2026-09-27-selected-e-sparse-source-audit.md) | NO-LAUNCH receipt map and game-cluster sample blocker; no authenticated sparse adapter or throughput result |
| [Selected-E complete-game index seam](2026-09-27-selected-e-game-index-seam.md) | CPU-only closed small-fixture identity reader and exact whole-game sampling; no registered E scan or row admission |
| [Selected-E one-teacher label-cost kernel](2026-09-27-selected-e-label-cost-kernel.md) | NO-LAUNCH CPU S/D kernel with exact selected and original-E blend byte pins; authenticated sparse-E adapter, postarena gate and full-wall screen remain |
| [SF-free E admission and paired seed replication](2026-09-22-sffree-E-admission.md) | Frozen five-job source and independent CPU admission review; no result claim. |
| [Bootstrap research toward RL restart](2026-09-06-bootstrap-research-roadmap.md) | Overall data, mixing, objective and search decisions toward a larger bootstrap and RL restart |
| [Graded TailRL bootstrap screen](2026-09-06-tailrl-bootstrap.md) | Exact finite-action objective and bounded CPU mechanism comparison; no training or strength result |
| [TailRL-inspired policy coverage](2026-09-05-tailrl-policy-coverage.md) | Frozen-cohort checkpoint diagnostic and isolated upper-tail loss prototype; no training readout |
| [Overlay hot-load diagnostic](2026-09-23-overlay-hotload-diagnostic.md) | Exact-parity ABBA on 65,536 B rows: 62.38% less measured loading time; trainer benefit remains unmeasured |
| [BT4 own-game teacher source](2026-09-23-bt4-own-game-source.md) | Full strict ordinary-bank replay and 752-row reusable policy/native-WDL source; no training admission |
| [SF-rooted rule-aware corpus outcome mode](2026-09-23-sf-rule50-corpus-mode.md) | Explicit theoretical versus strict six-man rule-aware labels, UCI request guard and derived provenance; synthetic tests plus one bounded real-Stockfish search fixture |
| [BT4 verified-session foundation](2026-09-23-bt4-verified-session-foundation.md) | CPU fake-session artifact, head, provider and exact-feed checks; no writer, real model run or throughput claim |
| [BT4 CPU prefetch](2026-09-21-bt4-cpu-prefetch.md) | Opt-in one-batch CPU preparation overlap with bounded cleanup and exact feed/output parity; no GPU throughput or live-adoption claim |
| [NumPy THP hot-load screen](2026-09-23-numpy-thp-hotload.md) | Bounded ABBA loader screen found no benefit from disabling current madvise huge-page advice; no production default change |
| [BT4 raw JSON projection](2026-09-21-bt4-raw-json-projection.md) | Guarded projected decoding preserves consumed values and hashes; 1.67x CPU preparation on one banked shard, no GPU/live adoption claim |
| [BT4 source read-ahead](2026-09-21-bt4-source-read.md) | Exact CPU input parity; chunk-aligned reads3.9–4.1x faster, measured CPU preparation1.31x; no end-to-end inference claim |
| [Downside policy/value combination](2026-09-14-downside-value-combination.md) | Prospective draft: prefer a single-copy matched 35M policy correction; value choice awaits active arena, integration and launch not yet qualified |
| [BT4 root-output reuse foundation](2026-09-23-bt4-root-output-reuse-foundation.md) | CPU-tested batched roots, explicit repetition-mode guard and optional C-search leaves; no game generator or throughput claim |
| [BT4 output-adapter foundation](2026-09-23-bt4-output-adapter-foundation.md) | Shared legal BT4-to-compact float32 conversion with caller-owned buffer parity; CPU contract only, no generator or speedup claim |
| [35M V50 training horizon](2026-09-18-v50-training-horizon.md) | Four complete 256-game matches: epochs 2/3/4 improve over epoch 1; latest increment and search-scaling interaction unresolved; first 50.55M pass +21.74 Elo [−9.66,+53.51] against 35M epoch 4; later expanded passes pending |
| [Ceres collector throughput: zero pauses and grouped sessions](2026-09-16-ceres-collector-throughput.md) | Historical September 16–17 fixed32/grouped and batch512 readouts, numerical-difference caveats and qualification recovery; seven original receipts, no new run or strength claim |
| [BT4 labeling throughput and scale costs](2026-09-17-bt4-label-throughput.md) | Verified1.527M fresh joint rows at578.76 outer rows/s;48h/100M and20days/1B linear estimates; bounded batch-tuning and metadata-cost priorities |
| [PFS-inspired frontier advancement](2026-09-14-pfs-frontier-advancement.md) | Proposed frozen-checkpoint search tests: diagnose admission, halving and interior bottlenecks; no runs or default changes |
| [BT4 generation outcome policy](2026-09-22-bt4-outcome-policy.md) | Standalone six-man theoretical adjudication and unresolved-game discard contract; fake-tablebase CPU tests only, no generator wiring or real games |
| [Full-width SF d8/d6 generation](2026-09-22-sf-shallow-generation.md) | Two equal-wall CPU generation arms; eligible rows, closed games, startup costs and limits; no strength or root-only generation claim |
| [Packed Zarr exact-sampler qualification](2026-09-20-packed-zarr-exact.md) | 131,072 identical ordered rows at batch256; external ZIP consumer12.537s vs NVMe directory10.953s; opt-in ordinary reader, no live adoption |
| [Ceres CPU semantics oracle](2026-09-13-ceres-cpu-semantics.md) | CPU oracle complete after comparator-only recovery: 144 histories match bytes; 131 value cases quantify FP16 getter differences, no neural parity claim |
| [Saved joint raw-label eligibility](2026-09-13-saved-joint-raw-eligibility.md) | 21.40M saved joint-label raw candidates outside covered rosters; 512-shard audit complete: 4,219,426 eligible / 4,249,935 raw rows; 88 exact exclusions qualified; 4,219,426-row / 516-shard derivation launched, no completion or training admission |
| [Bounded raw scale labels](2026-09-13-bounded-raw-scale-labels.md) | Completed 184 raw shards / 1,527,153 rows in 2638.682s; 12-group limit exhausted; baseline audit complete: 1,517,925 eligible rows, 30 phase-zero support failures and 9,198 no-result drops; exact filtered derivation complete: 1,517,925 rows / 186 derived shards; joint saved-policy/native-WDL adapter complete; B100 materialization complete after a preserved pre-writer command failure; metadata establishes 1,517,925 additional physical-source rows beyond existing 35M; potential 36,832,502-row union remains unadmitted |
| [Native BT4 value endpoint](2026-09-13-native-bt4-value-endpoint.md) | V100 seed-101 completed 35,314,577 rows / 68,974 updates; actual CPU preparation passed; fixed 512-game / 400-simulation V100-versus-V50 arena launched, no result yet |
| [Ceres100 policy endpoint preparation](2026-09-13-ceres100-policy-preparation.md) | Pure-Ceres endpoint completed: +2.04 Elo [−22.16, +26.26] versus B100 over 512 games at 400 simulations; unresolved, retain B100 policy |
| [Ceres materializer batch screen](2026-09-12-ceres-materializer-batch-benchmark.md) | Real 8,192-row outputs identical at 128/512; observed timings confounded by cache/order, no production speedup claim |
| [Ceres source-read cache](2026-09-12-ceres-source-block-cache.md) | CPU component benchmark: identical batches,1.74s→0.12s reads; future collector cache/timers, no GPU throughput claim |
| [Small G10 Ceres pilot](2026-09-11-g10-ceres-pilot.md) | 6,130,891 G10 Ceres rows / 751 shards qualified including complete batch-v3 run06; saved consumer qualification, not training admission; science remains first4 only |
| [Conservative research recovery](2026-09-12-conservative-research-recovery.md) | Guarded SF generation resumed; all 15 remaining native-WDL cohorts completed and qualified within two hours, covering the accepted 16.40M G10 positions; subsequent metadata consumer admission passed |
| [Saved SF negative constraints](2026-09-11-sf-negative-constraints-screen.md) | Completed 262,079-row SF/BT4 diagnostic: >300cp flagged mass mostly unscored; no blanket T300 promotion or Ceres strength claim |
| [Saved G10 value pilot and coverage](2026-09-11-g10-value-pilot-and-coverage.md) | 1,574,952 matched saved-SF rows; native BT4 WDL and metadata manifest admission complete for all 16,404,093 accepted common G10 positions; original B100 policy products now complete for all 20 (16,404,093 rows); fixed SF50/native BT4 WDL50 value products now also complete for all 20, including the four-root writer; [combined 35.3M value result supports V50 at +31.30 Elo](2026-09-13-combined35m-value-transfer.md) |
| [Weighted Ceres bootstrap preparation](2026-09-11-ceres-weighted-bootstrap.md) | Both CeresV25 fixed-400 matches complete: +6.79 Elo vs B100 and +25.83 vs B100V50, intervals include zero; retain B100 incumbent/Ceres alternative; no automatic extra games |
| [Combined 35.3M value transfer](2026-09-13-combined35m-value-transfer.md) | Completed 512-game/400-simulation comparison: V50 +31.30 Elo [+8.53, +54.34]; provisional 35M value recipe after reviewed read-only recovery |
| [V50 checkpoint continuation](2026-09-17-v50-checkpoint-continuation.md) | Strict LC0 bootstrap state continuation and retained epoch boundaries; archived 35M/50M operational evidence, no current launch claim |
| [All-move SF downside candidate](2026-09-12-sf-allmove-downside.md) | Completed 18.91M-row training at 2/2 workers; fixed 400-simulation comparison +17.66 Elo [−17.97, +53.66] versus B100, unresolved; selected G10 phase-zero/provenance pilot completed 8,192 rows in 9.889s, no full rewrite or training selected |
| [SF tactical guidance for BT4 policy](2026-09-10-bt4-sf-tactical-training.md) | Completed 400-simulation match: tactical recipe −13.6 Elo [−52.4,+24.9]; unresolved, retain B100 |
| [Deeper SF value census](2026-09-11-deeper-sf-value-census.md) | Complete: saved SF values pass label-change thresholds; 3,247 single-move exclusions and two malformed rosters explained; opt-in selector selected, no strength claim |
| [Saved adaptive SF value selector](2026-09-11-adaptive-sf-value-selector.md) | Opt-in deriver implementation and independent review passed; no real-corpus rewrite or training result |
| [BT4 target temperature and horizon: results](2026-09-10-bt4-target-temperature-horizon-results.md) | Complete: all three fixed deep matches; relative horizon change +5.664 points [−2.734, +13.867], direction and placement unresolved; [registration record](2026-09-09-bt4-target-temperature-horizon.md) retained |
| [B100 search-prior calibration](2026-09-09-b100-prior-calibration.md) | Complete: 0.7 versus 1.0 gives +16.30 Elo [−20.83, +53.80]; no follow-up selected, common prior 1.0 retained |
| [Bootstrap storage capacity](2026-09-08-bootstrap-storage-capacity.md) | Ongoing SSD-to-external transfer and verified reclamation for 100M; G20T1/G20T05/G50T05 reclamation complete; run04/run05 external copies verified and 870 closed shards reclaimed; metadata and unlisted tails retained; legacy E0 archive verified and 2,309 cold shard directories reclaimed; Tactical100 archive verified and 2,309 exact shards reclaimed; both summaries/checkpoints retained; Downside300 archive verified and exact 2,309 shards reclaimed; root summaries and all experiment artifacts retained; Ceres100 external copy verified and 2309 exact shards reclaimed, both summaries and all checkpoints retained; SoftSF10 archive verified and exact 2,309 shards reclaimed with both summaries retained; H20T05 copy-only archive launched, source retained and verification pending |
| [Value collection and horizon readiness](2026-09-08-value-collection-and-horizon-readiness.md) | Full 18.91M WDL bank and V50 training complete; V50 arena started, no playing result |
| [B100 tactical policy readiness](2026-09-08-b100-tactical-policy-readiness.md) | Producer/admission merged; corrected bounded preparation independently reviewed; V50-first schedule, no tactical rewrite/training |
| [Next value and policy contrasts](2026-09-08-bootstrap-next-value-and-policy-contrasts.md) | September 13 adaptive decision policy: retain registered result labels; estimates, uncertainty, mechanism and cost guide further tests or provisional use; V100 remains selected |
| [SF-anchored value bootstrap](2026-09-08-sf-anchored-value-bootstrap.md) | Matched 128-row WDL diagnostic complete; B100V50 is the first value contrast |
| [Ceres teacher readiness](2026-09-08-ceres-teacher-readiness.md) | C3 CPU/sample evidence retained; approximate fixed32 policy + primary-value collector implemented, 8,192-row integrated cost pilot prepared, not launched |
| [Faster recipe matches](2026-09-08-faster-recipe-matches.md) | First prospective SPRT plus fixed deep probe completed in46m35s; speculative scheduling bottleneck identified; no controlled speedup claim |
| [BT4 adapter identity collection](2026-09-08-bt4-adapter-single-pass.md) | Exact identity parity and 34.1% CPU component reduction in one closed-shard pair; full-pipeline benefit unmeasured |
| [Qualified Soft-SF training sample](2026-09-08-soft-sf-qualified-training-sample.md) | Complete: SoftSF10 loses first-look low SPRT and fixed400 probe (−162.99 Elo); no clear depth recovery |
| [BT4 target geometry](2026-09-08-bt4-target-geometry.md) | Fixed 128-row training sample: C support/maxima and raw-cp controls; no temperature or strength selection |
| [BT4 raw-label legal-move reuse](2026-09-07-bt4-label-legal-reuse.md) | Exact policy parity; first optimized group completed 132,467 rows, throughput descriptive only |
| [G10 adapter/rank overlap](2026-09-08-g10-stage-overlap.md) | Adopted in Worker01 attempt; cache failure, no measured speedup |
| [G10 qualified inventory](2026-09-11-g10-qualified-inventory.md) | 16,404,093 accepted common-input rows; native WDL coverage remains separate; transfer training/strength unestablished |
| [G10 transfer readiness and alignment](2026-09-07-g10-transfer-readiness.md) | Historical preparation and alignment protocol; later inventory linked above |
| [G50 versus B100 policy dose](2026-09-08-bt4-g50-b100-dose-comparison.md) | Complete: G50 loses shallow SPRT and fixed deep comparison (−50.57 Elo); B100 retained, depth interaction unresolved |
| [Pure BT4 policy endpoint](2026-09-08-bt4-pure-policy-endpoint.md) | B100 beats H20: shallow H1, deep +46.42 Elo [11.23,82.58]; search interaction unresolved, same-seed development |
| [Bootstrap training-horizon readiness](2026-09-08-bootstrap-training-horizon-readiness.md) | CPU and bounded compiled-CUDA checks passed; full SF/B100 planner passed; training comparison not launched |
| [BT4 hybrid and global endpoints](2026-09-07-bt4-hybrid-endpoints.md) | H20 package complete: C100 +45.07, G100 +85.78, C400 +34.86 Elo; aligned search interaction unresolved; no promotion |
| [External archive and restore record](2026-09-07-storage-archive.md) | 38 verified archives, qualified local cleanup, retained dependencies and restore instructions |
| [Fresh paired BT4 confirmation](2026-09-07-bt4-fresh-confirmation.md) | Completed fresh seed-one comparison: C beats E by 37.67 Elo [18.72, 56.85] at 100 sims |
| [Direct SF-close versus global sharpened BT4](2026-09-06-bt4-direct-close-global.md) | Completed 100-simulation comparison favors C over G20T05 |
| [Sharpened stored SF top ties versus SF-close](2026-09-06-bt4-sharpened-ties-screen.md) | Completed E0T05 seed-zero epoch and direct 100-simulation screen favoring C20T05 |
| [BT4 bootstrap results](2026-09-06-bt4-bootstrap-results.md) | Completed screens and confirmation; C is the selected next-stage baseline |
| [50% sharpened global BT4 preparation](2026-09-06-bt4-global-dose50-preparation.md) | Reviewed CPU target preparation for a larger teacher dose; no training or matches queued |
| [Global BT4 mixing and search scaling](2026-09-05-bt4-global-search-scaling.md) | Matched SF control, full-distribution mixtures and 25/100/400-simulation playing screen |
| [BT4 near-tie policy targets](2026-09-05-bt4-joint-targets.md) | Superseded six-arm plan and completed control diagnostic |
| [Varying-horizon online controller](varying_horizon_online_controller.md) | Preregistration and staged evaluation of online search continuation |
| [Value head architecture](value_head_arch.md) | Historical April 2026 experiments |
| [Historical ledger](../experiment_ledger.md) | Frozen July–September 2026 record, including yardsticks, gotchas and recovery snapshots |
| [Strict six-man factorial target readout](2026-09-27-factorial-strict-rule50-readout.md) | Four corrected rule50 A–D arenas: both policy edges +6.79 Elo, both value-recipe edges +25.83; shared-opening bootstrap and limits; separate E−D context |
| [Completed factorial and next-day queue](2026-09-22-factorial-readout-next24h.md) | Four conditional contrasts; SF-free value and matched second-seed preregistration; BT4 GPU optimization result and storage recovery |
| [Retained Syzygy option](2026-09-22-retained-syzygy-option.md) | Opt-in Stockfish SyzygyRetainOnNewGame forwarding with resume/provenance guards; no retention-performance claim |
| [500M continuation and partial factorial readout](2026-09-21-500m-continuation.md) | A/B/C exact epochs complete; policy +6.79 and value +21.74 Elo with intervals crossing zero; recovered arena bookkeeping, D ongoing; packed storage and bounded SF reset optimization |
| [BT4 complete pipeline screen](2026-09-21-bt4-pipeline-benchmark.md) | Fixed ABCCBA batch128 projected-decoder/prefetch comparison; exact inputs and outputs, complete labeling plus verification, reviewed and queued after factorial matches |
| [58M SF late-dose metadata and census design](2026-09-28-sf-late-dose-metadata.md) | 4.7 MB bounded metadata receipt for 58,090,688 retained rows; source-aware controls and capped three-shard pilot proposed, no corpus scan or training |
| [SF-free D/E phase-start arena](2026-09-27-factorial58-d-e-phase-arena.md) | Complete strict six-man 384-pair/768-game phase test: primary late-minus-early +1.758 points [−2.734, +6.445], inconclusive; secondary overall GSPRT H0 applies only to its declared hypotheses |
| [Saved Stockfish policy dose by ply and pieces](2026-09-28-sf-late-position-dose.md) | Registered read-only G10 Tactical300 cross-tab: 83.879% of moved mass at ply ≥80 with 7–13 pieces (55.462% at 8–13); no Elo or 58M distribution claim; bounded 58M screen remains next |
| [Zarr ZIP decoder pilot](2026-09-20-zarr-zip-pilot.md) | Byte-preserving packed chunks; validated external decoder 8.2x faster in cache-affected pilot, no sampler adoption |
| [External trainer throughput inventory](2026-09-28-500m-external-trainer-inventory.md) | Archived cold/warm packed trainer rates and local E epoch; historical 128-full-shard same-ZIP sizing, with the later completed screen linked separately |

These descriptions identify the records, not the state of a running experiment.
Check subsequent readouts and the actual process/artifacts before resuming work.

The [BT4 evidence catalog](evidence/bt4-bootstrap/README.md) provides published
registrations, machine-readable readouts, game banks and provenance receipts.

## New experiments and follow-ups

Create `YYYY-MM-DD-slug.md` here and add a link above. Keep the preregistration,
amendments and subsequent readouts in that record. Existing individual records keep
their filenames. For a follow-up to an archived experiment, create a new record
linking the original evidence; do not append to the frozen ledger.

Include the hypothesis, baseline/control, realized settings and revision, deciding
command/threshold, uncertainty method, budget, horizon, confounds, artifact identities
and recovery plan. Add the readout against the precommitted rule, with its limits and
next decision. Publish compact supporting evidence with the record: realized settings,
readouts, review receipts and manageable game banks, with a source/hash manifest.
Keep large corpora, checkpoints and transient logs external and document their
identities and retrieval or restore locations. Historical absolute paths inside
frozen evidence are provenance, not portable command defaults.

## Finding prior evidence

The old ledger remains large and unsplit to preserve its contents, anchors and evidence
references. Search it; read only the matching entries and their later corrections.

```bash
rg -n -i 'YOUR_TOPIC_OR_CONFIG_KEY' docs/experiments docs/experiment_ledger.md docs/rl_loop_audit.md
```

Search by experiment name, config key, artifact path or checkpoint identity. Historical
“LIVE” labels and operating instructions are not current authority. An artifact absent
from a worktree may still exist under its original run directory.

For durable knowledge, use [model heads](../model_heads.md),
[target rebuildability](../target_rebuildability.md),
[the loop audit](../rl_loop_audit.md) and [operations](../operations.md), following their
links to supporting experiments. The `experiment-readout` Skill covers the reusable
analysis workflow.

- [Ceres CPU history reductions](2026-09-21-ceres-history-reductions.md) — exact-byte CPU encoder comparison.
Current evidence correction and scale plan: [September 16 bootstrap audit](2026-09-16-bootstrap-status-audit.md).
- [2026-09-20: External storage and BT4 labeling efficiency](2026-09-20-storage-loader.md)

- [BT4 run11 variable-batch scheduling opportunity](2026-09-27-bt4-batching-opportunity.md): fixed-wave tail, conditional rolling-call arithmetic, and 128-game A/B compatibility HOLD.

- [2026-11-18 U2 fixed-union frozen-policy-anchor method and completed readout](2026-11-18-fixed-union-policy-anchor-u2.md) - Locally frozen KL1/outcome-only pass completed; public method record appeared after first launch. Fixed800-game reused-panel development scores49.75% vsF1 and45.375% vsA39; neither criterion passed, originalA39 retained. No demonstrated mitigation, fresh confirmation or general KL verdict.

- [2026-11-18 V10 fixed-union small-search WDL preregistration](2026-11-18-fixed-union-small-search-v10.md) - Prospectively test90% outcome/10% retained-search WDL with KL0 on one405,013-row global-uniform pass, then two fixed800-game reused-panel development comparisons; U2 preserved/unlaunched, no execution authority issued.

- [2026-11-17 U1 historical self-play uniform-once preregistration and readout](2026-11-17-historical-selfplay-uniform-once.md) - Fresh F1 completed405,013 views/12,657 updates; accepted fixed800 reused-panel DEVELOPMENT:212W/310D/278L,45.875% (paired95% CI43.28–48.47%); no donor-advantage selection, A39 retained; prior failures/costs preserved.

- [2026-11-17 D2 matched short-horizon policy-anchor controls](2026-11-17-short-horizon-policy-anchor-controls.md#november-17-completed-six-edge-readout) - Six accepted160-game screens selected S0 (55.9375% vsA39;57.5% vsL0); separate800-game confirmation scored48.75% and did not confirm donor advantage. A39 retained; no adoption or external strength claim.

- [2026-11-17 Fresh S0/A39 confirmation](2026-11-17-s0-a39-fresh-confirmation.md#november-17-completed-fresh-confirmation-nonconfirm--inconclusive) - Completed400pairs/800games:235W310D255L,48.75% [46.0267,51.4733]%; lower>.5 unmet, NONCONFIRM_INCONCLUSIVE. No harm/equivalence or adoption claim.

- [2026-11-17 Stored-row sampler reuse audit](2026-11-17-stored-row-sampler-reuse-audit.md) - Accepted read-only historical census: short/long maximum reuse8/20;1,160/17,210 rows exceed two presentations. Counts do not establish overfitting; exact-once software preparation does not clear the training hold.

- [2026-11-17 D2 metadata qualification recovery](2026-11-17-d2-metadata-qualification-recovery.md) - Closed: original and one bounded recovery metadata qualification both FAILED;45 local software cases and partial4+2+2 API reports do not complete the metadata qualification. All411 owners closed, costs charged; no D2 playing results or automatic retry. Path-normalized deployment-specific excerpt; not standalone.

- [2026-11-17 D2 retained-model metadata correction](2026-11-17-d2-retained-model-metadata-correction.md) - Prospective: genuine saved-only L0 compatibility fix and one separately admittedCPU290/260 qualification within original cumulative ceilings; both failed stops/costs retained, no new playing result or execution authority.

- [2026-11-18: Fresh current-policy A39 twelve-cycle baseline](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md) — completed 12 cycles /948 updates /241125 stored-row presentations; final 800-game reused-DEV score 47.125% versus fixed A39 (paired 95% CI 44.51–49.74%; conditional Elo-equivalent −20.00). No development gain established; A39 retained. M3/M6 and recovery history preserved.

- [2026-11-19: Q50 fresh current-policy A39 twelve-cycle value-blend sibling](2026-11-19-fresh-current-policy-a39-q50-twelve-cycle-baseline.md) — prospective search/outcome0.5/0.5 followup with distinct fixed seeds and fresh own-actor banks; final800-game consumed-DEV screen only, no interim adaptation, no execution authority or gain claim. Negative F12 and all prior costs/stops retained.
