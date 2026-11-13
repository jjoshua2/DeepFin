# Resolved-outcome value targets at a fixed 40-update dose

Status: preregistration frozen before the 96 research games; independent review,
main publication and individual stage admissions remain required. No research
updates, playing result or candidate adoption have occurred.

The current RL transition has not demonstrated a repeatable strength gain.
Earlier endpoint results and the interrupted 53-game collection remain preserved
in the [week-one record](2026-11-12-rl-week-one.md). The donor's measured external
baseline was 1 win, 24 draws and 39 losses against Rofchade 3.1 with eight threads
at 30 seconds plus 0.1 seconds per move. That profile scored 20.3125%; it does not
establish the project's Rofchade milestone. The long-term objective remains
convincingly beating unhandicapped multicore Rofchade, then approaching strong
Stockfish and Lc0 deployments and investigating improvements beyond them.

This comparison asks whether adding 20% verified resolved-game outcome to both
value objectives improves playing strength over search-only targets at one fixed
dose. It tests a target bundle, not an isolated head mechanism or an optimal RL
formula. Teacher policy supervision and the rest of the original full-network
training configuration remain common to both arms.

## Frozen design

The machine-readable [preregistration](evidence/rl-e2tb-week-one/preregistration.json)
is authoritative. Its SHA256 is
`234e75ee42e174e20420165cae03d94def86c4ae0868e15f5c449f7c5b1f0d60`.
The [artifact manifest](evidence/rl-e2tb-week-one/artifact-manifest.json) records
the published evidence and external inputs. External artifacts live under
`/home/josh/chess-artifacts/operations/rl-week-one-20261112-v1`.

Both arms start from the exact epoch-two donor
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`,
with fresh optimizers and no SWA. The runtime remains frozen at
`ba9d6de1987902e1759d984532a4c63f091b9269`. The strict selfplay library is
published through [PR 1249](https://github.com/jjoshua2/DeepFin/pull/1249);
the final main version adds only two test metadata corrections to that runtime.
Other UCI changes on main are not deployed into this measured checkout.

| Property | Fixed choice |
| --- | --- |
| New study identity | `f1dadbe4-43e2-4cf3-9a93-70a288533b48` |
| Collection | 96 fixed physical attempts, 16 slots, 32 TRAIN and 64 HOLDOUT |
| Generation | One donor, pure selfplay, native Gumbel 64 simulations, 450 additional plies |
| Tablebases | Indexed three-through-six-piece WDL and DTZ; strict rule-50 and full-history semantics |
| FULLA | Main WDL search 1.0, outcome 0, SF 0; original categorical search target |
| FULLB | Main WDL search 0.8, resolved outcome 0.2, SF 0; matching categorical scalar blend |
| Delivery | 40 updates per arm, batch 256, one common uniform physical-row draw plan |
| Pairing | Common draw, mirror and forward/dropout RNG; original optimizer, losses and schedule |
| Snapshots | 0, 1, 8 and 40; only FULLB at step 40 is the candidate |

Categorical targets use signed Q in [-1, 1]. FULLB blends the original producer
float32 search Q with the verified outcome Q before HL-Gauss construction and the
production float16 cast. Every eligible FULLA row must reproduce the original
float32 search control and stored float16 categorical target exactly. This is
not a mixture of cross-entropy losses or gradients.

Natural outcomes, strict tablebase adjudications and unresolved games retain
distinct provenance classes. An independent owned tablebase verifier reconstructs
the legal full history, clocks, side to move and outcome witness. A positive-clock
decisive tablebase result can remain unresolved and play on. Censored games never
become draws or receive Stockfish outcome substitutes. Policy-only tablebase
rescoring remains enabled; theoretical whole-game outcome rescoring is disabled.

Exactly all 96 attempts must pass source, history, inventory and custody checks.
All 32 TRAIN games must resolve and have usable retained rows with original raw
controls. All 628 fixed playing starts and their production mirrors must be absent
from the retained raw/mirror inputs of all 96 games before learning. A collision,
missing attempt or unusable TRAIN game prevents learning. There are no favorable
subsets or replacements after collection starts.

All HOLDOUT games remain outside learning. Valid censoring, zero rows or failure
to provide an eligible held-out root makes the entire offline diagnostic
inconclusive without independently vetoing the primary playing endpoint. Missing
or invalid provenance is an all-96 integrity failure. Complete offline metrics
require all 64 games and one score-blind NET root per game at offsets 8 through
80, excluding known prior exposure and all new TRAIN inputs and histories.
The frozen hash and chronological tie rule choose roots before predictions.
Expected-score error, draw and three-class Brier scores, and legal-policy drift
use 10,000 whole-game bootstrap draws. They are diagnostics, never endpoint selectors.

## Playing decision and stopping rules

Run three fixed 256-game, 128-opening-pair comparisons: FULLB40 versus FULLA40,
FULLB40 versus donor, and descriptive FULLA40 versus donor. Run all three even
when a performance screen fails, unless a global control or technical failure
stops the protocol. FULLA cannot be promoted from its context result.

Both FULLB screens must score at least 0.53 with a paired normal 95% lower bound
strictly above 0.5. The sampling unit is the mean of the two color-swapped games
in an opening pair, with sample variance using n-1 and z=1.96. A zero-variance
interval is unavailable and cannot pass. Complete legal games, full histories,
counts, cap handling and process custody are required; a successful launcher
alone does not qualify a partial arena.

Only when both FULLB screens pass and all three screens are technically qualified
may the same FULLB40 checkpoint play one 1,000-game, 500-pair confirmation against
the donor on the preselected disjoint book. Apply the same point and interval
rule. There is no extension, retry, reseeding, alternate endpoint, FULLA fallback
or pooling of screening and confirmation games. A complete zero-variance context
result remains descriptive; a context technical failure prevents confirmation.

A primary upper confidence bound below 0.5 indicates harm relative to FULLA at
this dose. An eligibility upper bound below 0.5 indicates donor regression.
Other nonpasses are unread at this dose and training realization. Even a confirmed
gain would be conditional on this internal six-piece/64-simulation profile;
external wall-clock comparisons, training replication and adoption need their
own evidence. None of these outcomes completes the months-long research goal.

## Qualification, resources and recovery

The accepted engineering producer exercised a retained strict-tablebase row,
a natural zero-row draw and a censored zero-row game. It passed the raw-value
controls and the public shard format's exact canonical storage comparison.
Its final attempt took 17.352 seconds. Both earlier failed attempts remain failed;
all three attempts cost 41.874 seconds, charged against the original 180-second
producer reservation. Engineering-only retention conditioning is explicitly
different from research thinning, which remains enabled. These games are excluded
from research. The accepted known-exposure supplement preserves six deduplicated
raw/mirror identities and four history hashes, including failed-attempt lineage
and its remaining unknowns.

Actual CPU checks covered current native imports and all 756 selected encodings,
the production builder and final collector configuration, 29 TRAIN controls and
94 normalized Trainer arguments. Historical 4+4-update A/A and 32-game arena
evidence is reused only within the documented source compatibility scope;
neither is represented as a new runtime experiment or proof of 40-step parity.
The offline reader's 44 controlled wiring cases are synthetic evidence, not an
actual held-out or model-inference result.

The complete study has a hard 21,060-second GPU-intended owner-wall ceiling,
including failed and interrupted qualifications. Stage ceilings are 1,200 seconds
for collection, 1,200 for both learning arms, 300 for offline inference, 3,600 for
each screen and 7,200 for conditional confirmation. Qualification reservations
remain fixed and cannot be replenished from failed attempts. These charges also
count toward the week's 172,800-second ceiling and hard deadline
`2026-11-19T16:09:20Z`; the earlier 5,859.448 seconds remain charged to the week.

Use one GPU stage at a time, two CPU threads on physical cores 20 and 22, and all
three existing leases. The user explicitly permits bounded GPU experiments beside
Call of Duty. Leave that application untouched and record contention; latency and
cost are conditional on the resource window. Clear inherited math overrides in
the fixed commands, then verify the actual declared math context without changing
Torch settings to force agreement.

Every stage checks controls, remaining budgets, deadline, memory, disk and process
closure before admission. Honor STOP, PAUSE, CANCEL, YIELD and intentional-stop
markers; never clear them. Preserve the donor, both failed qualifications and the
stopped 53-game study. Immutable component templates and explicit identity-only
binding contracts prevent scientific changes during late binding of cohort,
draw, checkpoint and publication identities. Technical failure yields an
unqualified or inconclusive study and a separate diagnosis, not an automatic rerun.

## Readout

Research collection, updates, offline scores and playing results are unobserved
at preregistration. Append their actual provenance, costs and decision against
these rules here without rewriting the frozen preregistration.
