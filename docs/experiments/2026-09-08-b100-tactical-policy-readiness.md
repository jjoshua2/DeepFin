# B100Tactical100: selective SF policy correction with SF value retained

September 8, 2026 evening snapshot. **The producer and training-admission tools
are merged; the full rewrite and training have not launched.** V50 remains the
first selected contrast. Tactical preparation is a separate policy experiment,
not a combined value/policy recipe or a claim of stronger play.

## What this contrast tests

Keep B100's stored sharpened BT4 policy as the starting distribution, and retain
its original SF value targets. Use SF to reduce probability on moves with large
raw d9 deficits while preserving BT4's odds among acceptable moves. This addresses
a different question from V50, which holds B100 policy targets fixed and changes
only the main value target. Shared-trunk learning can affect both learned heads;
the supervision intervention is what remains isolated.

For ordinary scores, the fixed multiplier is
`max(0.1, exp(-max(0, best_cp - move_cp - 100) / 100))`. Multiply the actual stored
B100 probability by that value, then normalize. The gap100/decay100/floor0.1
recipe is a selected substantive contrast, not an optimum fitted to playing
results. The earlier [training-bank diagnostics](2026-09-08-bootstrap-next-value-and-policy-contrasts.md#next-policy-question-attenuate-large-sf-deficits)
found 13.43% weighted policy mass exposed to the gap100 correction, mean target
TV .05177 and top-move changes on 3.86% of rows. Those diagnostics motivate the
question; they do not establish tactical safety or strength.

Mates use categories, not differences between encoded mate scores. If winning
mates exist, their weights are 1 and all other weights .1. Otherwise ordinary
moves use the gap rule and losing-mate alternatives get .1. All-forced-loss rows
retain their original bytes. A .1 multiplier floor is not a minimum final
probability. **Stored zero mass stays zero**, even for a winning mate; the producer
reports rows whose whole winning-mate group has zero original B100 mass. It cannot
repair that missing support. Float16 support-loss and rounding diagnostics remain
visible; unchanged weights preserve the original policy bytes.

## Merged construction and admission

[PR595](https://github.com/jjoshua2/DeepFin/pull/595) adds the explicit tactical
route to the existing [SF policy writer](../../scripts/sf_policy_rewrite.py).
It consumes the original raw 20M prefix, original SF corpus and qualified B100
parent, preserving the original no-result filter and within-shard permutation.
Every accepted row requires the full legal d9 roster and exact reconstruction of
the original stored SF policy. The base is B100's actual stored float16 policy;
no new teacher inference or idealized pre-storage probabilities are substituted.
All 16 nonpolicy columns, including SF value and stored history, are independent
compressed-byte-verified copies. The genuine tactical summary records both source
lineages, consumed/output policy hashes, mate groups and storage diagnostics.

[PR596](https://github.com/jjoshua2/DeepFin/pull/596) adds the schema3-only
`B100Tactical100` route to the existing [original-epoch coordinator](../../scripts/bt4_one_epoch_screen.py).
It requires the genuine `bt4_sf_tactical_policy_summary.json`, exact recipe/source
identity, a separately qualified full output roster and prospective schedule.
It rejects renamed policy/value recipes. The original training command, seed 0,
batch 512, 16 planning/16 loading workers, 36,935 updates, 420 windows and completed
checkpoint/schedule interface remain unchanged. Later arenas use the existing
matched-original-epoch route and a separate actual registration.

History and invalid-control limitations are inherited, not retroactively repaired.
The original raw producer discarded UCI bound lines before storage; aggregate
bound-line counts are not a new independent exact-bound witness for each row.
SF agreement and deeper search share SF biases. Neither successful admission nor
this supervised correction proves calibration, generalization or playing strength.

The [implementation evidence](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/implementation.tar.gz)
preserves author validation and independent review. Producer validation covered
58 distinct cases, including actual tiny shuffled writer/loader and nonpolicy
byte parity, with a final whole-host static pass. Admission validation covered
131 focused cases; its whole-repository type check found only two fixture
annotations, followed by a final scoped Ruff/type/Vulture pass. These are separate
validation histories, not a single combined-suite claim. The [independent review](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/implementation.independent_review.json)
checked five additional scalar-oracle cases, all source identities and the
unchanged legacy paths. No full-source materialization or training was qualified
by these small tests.

## Concrete preparation and its remaining evidence

The inactive CPU runtime is pinned to merged `bbb4f6cac7dc914af772036745533b89d7bed345`.
Existing Python 3.13 CPU dependencies and four native binaries are reused after
source/import identity checks. A 6.70-second metadata admission inspected four
summaries, 4,618 SF/B100 shard attributes and 2,410 raw-file stat identities. It did
not read corpus payloads or establish a successful rewrite. The intended 20M raw
prefix preserves 18,910,484 derived rows in 2,309 shards; those are expected counts,
not a completed tactical corpus.

The [frozen preparation bundle](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/preparation.tar.gz)
contains exact command, runtime/source metadata, supervisor, prospective plan,
continuation specification and review. It proposes CPUs 0,1, two numeric/Torch/Blosc
threads, nice19/ionice3, GPU hidden, a 150 GiB free reserve and 32 GiB sampled
allocated-output cap including partials. These are limits, not measured output
size or reservations. There is no hard RSS quota. The related completed SoftSF
rewrite took 4.82h and peaked at650.45 MiB RSS; tactical additionally checks B100
and its nonpolicy copies, so that is not a tactical throughput forecast.

Independent preparation review caught a missing outer bound on the entry
operator's prelude. The corrected exact command wraps the whole invocation in
GNU timeout: 28,770 seconds to TERM plus 30 seconds to KILL. Internal deadlines
and owned-child cleanup remain in force. Original command/preparation evidence
and the [correction](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/entry_deadline_correction.json)
are preserved. The [independent final preparation review](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/preparation.independent_review.json)
closed the entry finding and checked all 29 frozen preparation members. Busy preparation lock, existing output/partial
or STOP refuse the attempt; no automatic retry is planned.

After successful production, the prepared metadata qualifier has a 600-second
inclusive bound and reuses actual producer byte proofs. Only its genuine output
can supply the missing dataset/summary hashes to one 1,800-second original-runtime
prospective schedule. Actual training and arena manifests remain unbound. The
[publication manifest](evidence/bt4-bootstrap/tactical-schedule-manifest.json)
records exact source and per-member hashes; archived scripts are inert data.

## Compute order and storage headroom

The [near-term schedule](../../scratchpad/bt4_joint20/publication_20260908_tactical_schedule_v1/schedule/README.md)
keeps V50 first after full native-WDL coverage and archive release of the shared
preparation lock. Both recipe rewrites and the archive use that same nonblocking
lock; different CPU affinities do not permit simultaneous materializations.
Tactical CPU work may overlap V50 GPU training only after V50's prospective
planning, with actual memory/I/O headroom for the original 16+16 workers and the
later realized verifier. Tactical still compares with unchanged B100, independently
of whether V50 wins; targets are not automatically combined.

A queued raw policy+WDL child already waits on the shared GPU lease. Preserve it.
The reviewed 900-second Ceres probe can use a genuinely idle boundary during V50's
CPU rewrite, after queued useful labels finish; it has no guaranteed priority.
The raw driver sleeps 15 seconds between groups, while a busy Ceres lease refuses.
No driver pause, inference or probe launch is part of this schedule.

At the parent's 02:55 UTC host observation, free space was **358 GiB SSD / 7.5 TiB
external**. Charging full planning allowances of 32+32 GiB for recipe outputs,
24 GiB archive staging,4 GiB estimated physical WDL storage and 0.125 GiB Ceres
output leaves **265.875 GiB**, 115.875 GiB above the 150 GiB reserve, before raw
growth, checkpoints, caches and other unbudgeted owners. Some bytes were already
present, making the additive calculation conservative; it is not a reservation
or 100M capacity proof. The WDL physical allowance is an estimate, while its actual
output guard measures logical bytes. Do not credit future archive reclamation or
count G20T1 recovery twice. Keep raw/SF/B100 dependencies and checkpoint/holdout
exclusions while continuing verified cold-policy transfers and reclamation.

After these two isolated contrasts, protect the broader questions: matched
fresh-seed replication, a same-runtime two-epoch comparison, targeted
same-checkpoint prior calibration, and an actual phase-qualified transfer recipe
from the 9,298,514 prepared G10 common rows. Existing compiled-fixture/planner
proofs do not establish full horizon training; common inputs are not already a
qualified finalist corpus. These are conditional useful choices, not another
mandatory queue of local target variants.
