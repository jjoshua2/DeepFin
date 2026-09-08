# Pure BT4 policy endpoint versus H20

**Selected September 8, 2026, after the [completed H20 package](2026-09-07-bt4-hybrid-endpoints.md#completed-c400-probe-and-package--september-8). Training and arena launch manifests remain pending qualification; this record does not launch either stage.**

## Question and selection

H20 is the development incumbent, without production promotion. Its registered
package favored H20 over C20T05 at 100 and 400 simulations and over G20T05 at 100.
The aligned higher-search contrast was unresolved. The highest-value next family
question is whether the pure BT4 policy endpoint improves on this hybrid, before
fine-tuning another intermediate mixture.

B100 uses the legal-normalized BT4 policy sharpened at **teacher temperature 0.5**
as the entire policy target. **SF-derived value targets and all other non-policy
fields remain unchanged.** This is not BT4-only value supervision, an unsharpened
teacher or a change to inference prior temperature. The common source remains
the original 18,910,484-row, 2,309-shard, 97,968-game corpus; no G10 transfer is
part of this comparison.

This dated selection supersedes the original *unselected* B100-versus-C fixed-bank
option. The completed H20 package and its original rules remain immutable.
G50 and other alternatives remain available, without an automatic queue.

## Training and controls

Train B100 once from scratch at seed zero, using the qualified one-full-game-epoch
schedule: batch 512, 16 planning/loading workers, 88-step windows, 36,935 batches
and the same optimizer, initialization and numerical semantics as H20. Require
completed finite diagnostics, zero nonfinite skips/CUDA retries and matching
prospective and realized source-normalized row/batch schedules. The canonical
schedule is `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Select the registered final checkpoint, with no intermediate-checkpoint search.

The H20 reference is its already qualified final seed-zero checkpoint, SHA256
`0a711fcf10ff87fc8360d3fd4b3035b170a7c15172616317c4a9687ae99d7017`.
The launch qualification must bind the B100 corpus lineage, policy normalization,
non-policy identity, actual runtime/native/configuration pins, schedules and final
checkpoint before arena execution. Training preserves the historical qualified
Python 3.10/Torch 2.11/CUDA 12.8 semantics; adopting newer arena control flow must
be qualified separately. Existing `valid_control:false` purity, committed-config
and historical-sampler limitations remain.

## Registered match allocation

The prospective allocation follows [PR #547](https://github.com/jjoshua2/DeepFin/pull/547).
It requires the reviewed ordered-prefix implementation and a qualified launcher
and reader before use. Existing fixed-N readers are not sufficient qualification
for a sequential result.

| Comparison | Registered allocation | Decision |
| --- | --- | --- |
| B100 versus H20, 100 simulations | Paired GSPRT: logistic H0=0/H1=+15 Elo, alpha=.05/beta=.10; first look at 128 pairs, then every 64; cap 500 pairs/1,000 games | First crossed boundary favors its separated hypothesis; cap without crossing is inconclusive |
| B100 versus H20, 400 simulations | Fixed 128 pairs/256 games, exactly the first 128 low-budget opening pairs | Report paired score/Elo and nominal 95% interval; retain this probe regardless of the shallow result |

The final 500-pair cap is a declared sequential look even though it falls between
64-pair steps. Only a complete canonical prefix can enter the sequential decision;
every released look is checked in order, and the first crossing is immutable.
Completed speculative suffix games remain banked separately from the deciding
sample. H1 is not a +15 Elo confidence lower bound; H0 is not equivalence. The
GSPRT's nominal error guarantees are asymptotic, and stopped ordinary Elo intervals
are descriptive. An operationally incomplete stage is not a negative result.

Use matched qualified training search shapes, prior temperature **1.0** on both
sides, move temperature 0.1, maximum 300 plies, original development book seed 42 and
16 opening plies. Bind exact full settings, opening histories/endpoints/colors,
execution capacity and implementation hashes before launch. Preserve fresh
confirmation seeds/openings. A separately bounded capacity component measurement
may inform execution capacity; it does not choose policy recipes from playing
outcomes or alter the registered scientific samples. No throughput gain is assumed.

A positive 400 result alone does not show a growing search advantage. If reporting
the secondary interaction, use the same fixed 128-pair core at both budgets and
label it exploratory; use the existing aligned-pair bootstrap convention of 10,000
PCG64 replicates, seed 20260903. Two budgets measure one contrast. Neither extra
games nor a sequential verdict captures training-seed variability. Review the
completed bundle before choosing further refinement or fresh confirmation.

## Resource and execution status

B100 has a **7.5 GPU-hour hard package cap**: training 4.5 hours and each arena 1.5
hours, including termination allowance. A separately registered capacity component
probe has its own **20-minute cap**, before training. The combined maximum is
therefore **7 hours 50 minutes**, not 7.5 hours including the probe. CPU-only metadata
and schedule qualification is outside these GPU allowances; each schedule stage
retains its 30-minute bound and releases the GPU lease.

Actual train/arena manifests, new sequential reader qualification and final runtime
pins remain pending. Preserve shared GPU leasing, surviving timeouts, STOP handling,
owned-process cleanup, durable failure/completion receipts and at least 150 GiB free.
No automatic retries or outcome-dependent extensions. Keep labeling available during
preparation and gaps. Unused allowance does not justify another arm. No production
adoption is implied.
