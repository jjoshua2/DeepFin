# One all-move SF downside candidate

Status: the 8,192-row pilot passed, and full CPU corpus preparation launched September 12 at 12:32:49 UTC. Full completion and training admission are pending; no new playing result. The registered Ceres value comparisons retain priority.

For normalized stored B100 policy (BT4 temperature 0.5), multiply each legal move
with a raw d9 deficit strictly greater than 300 cp by 0.5, leave the others at 1,
then normalize. This halves flagged-versus-unflagged odds, not necessarily flagged
probability mass. The factor 0.5 is a prespecified moderate intervention, not an
optimized dose. Multiple near-best moves do not disable the correction. Within
either set, BT4 odds are retained. Any mate-domain row remains unchanged; missing,
duplicate, illegal or nonfinite retained rosters fail qualification. Scores come
from the original full legal roster, never a top-eight rank sidecar or reconstructed
quantized policy. A mathematically unchanged distribution preserves stored bytes,
including zero flagged mass and mass supported entirely inside one weight class.

The recipe identity is `stored-b100-allmove-sf-gapgt300-weight0.5-ordinary-v1`.
It reuses main's `sf_policy_rewrite` original raw/SF/B100 admission, shuffled-row
alignment, original q-policy reconstruction and sixteen nonpolicy-array copy checks.
These are also the source validations reused by the unmerged tactical-transfer
producer; importing that separate producer is unnecessary. Existing legacy targets
and full-corpus behavior remain the default.

## First real pilot preregistration

Use the original `run03_s3` raw source, original 18,910,484-row SF corpus and its
B100 T0.5 policy product. Their summary pins are respectively
`55a9cf043b9b90a005bd1adf1dc6d810cb282341bcc11a4eccf127c07c09d6af`,
`391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef`,
`47e0e0cca578a89278383d1faef70c5f1f8c45fbc5a256cb91400243315dbb43`,
and B100 mixture `221a8296608ee5c698a4d8bf59145208c409de43a2fc1427824c5ad5d453fd38`.

Invoke the existing producer with its pinned original/B100 inputs plus
`--tactical-recipe allmove-downside300 --pilot-shards 1 --pilot-max-raw-rows 16384`.
The proposed outer supervisor budget is 600 seconds inclusive, two CPU threads,
CUDA hidden, 8 GiB address space, 2 GiB fresh output and 150 GiB free-space reserve.
Finalize an exact source/runtime-pinned operator only after independent review;
this document is not an executable launch authorization.

The pilot stops raw iteration immediately after the first 8,192 retained rows,
with a hard physical-row cap that includes discarded no-result rows. It preserves
the original shard shuffle. It checks complete source membership metadata but
only inspects selected derived shard storage and joins the needed raw prefix.
Consumed raw files are still fully hashed for provenance; pilot mode rejects any
such file larger than 64 MiB before decoding. The first saved compressed raw file
is 15,091,686 bytes by metadata inspection, not a new payload scan. It can require
more than one raw file because physical rows and retained rows differ.

Prior G10 saved-source analyses took about 47 seconds for one 8,192-row shard,
and 377 seconds for four shards with Ceres comparison, but those are different
pipelines. They justify a bounded exploratory budget, not a runtime guarantee.
At registration, this producer's join/copy throughput was unmeasured. Stop on the first integrity,
resource or cap failure, preserve partial output, and diagnose before expanding.
No automatic full rewrite follows a passed pilot.

A pilot passes when it emits exactly the selected original rows with unchanged
nonpolicy bytes, valid normalized policy and explicit recipe identity. The output
is stamped `PILOT_COMPLETE_NOT_TRAINING`; it has no `derive_targets_summary.json`
and cannot masquerade as the full source. The record retains original summary
pins, actual consumed physical/dropped rows, raw-file proofs and selected shards.
Full execution retains the original complete-source count checks.

## What this can decide

A qualified pilot establishes producer correctness and measured preparation cost.
Only a later separately scheduled matched B100 training/playing comparison can
establish strength. Existing SF-bank agreement is a teacher-dependent diagnostic,
with substantial missing deeper scores, not evidence that every flagged move is
bad. This single candidate tests a different mechanism from isolated-best transfer;
it is not a threshold or dose sweep. It also changes entropy/support allocation,
so any eventual gain would not isolate tactical information from calibration.

## Completed first-shard producer pilot

The single pilot passed, including a separate saved-output check inside the same
bounded operator. It consumed 8,371 physical rows (179 without results), emitted
8,192 rows and changed 4,660 stored policies. There were 5,679 ordinary rows and
2,513 mate-domain rows; mate-domain targets were unchanged. No stored support
losses occurred. All sixteen nonpolicy arrays retained their source bytes, verified
through 273 copied-file hashes; the saved policy digest and normalized-mass checks
also passed. [Compact evidence](evidence/sf-allmove-downside-pilot-20260912.json)
contains all aggregate producer diagnostics and exact source/output/receipt pins.

The producer took 13.57 seconds with 518,204 KiB peak RSS; the complete operator,
including saved-output qualification, took 15.999 seconds. An initial attempt failed
before source admission because the CLI lacked the runtime `PYTHONPATH` (0.625
seconds). That failure remains intact. A fresh namespace with only the environment
correction and a 599-second remaining bound succeeded; no target or source contract
changed between attempts.

This establishes a working bounded producer on the selected original prefix. It
neither estimates playing strength nor proves whole-corpus throughput or validity.
The output remains explicitly pilot-only and has no trainable corpus summary.
No full rewrite or training is automatically launched from this result.

## Full CPU preparation registered and launched

Following the qualified pilot, the parent selected one complete preparation of this fixed recipe. The producer launched at 2026-09-12 12:32:49 UTC from isolated runtime `b39a5d589…`, with the same reviewed producer bytes `9c396c63…`. Only the pilot limits are removed from the target invocation. The intended corpus retains all 18,910,484 original rows across 2,309 shards, reconstructed from the original 20,000,000 physical rows with 1,089,516 no-result omissions. Row order, source identities, value targets and the sixteen nonpolicy arrays retain the original contract.

The allocation is 43,200 seconds inclusive: the enclosing timeout sends TERM at 43,170 seconds and KILL after a 30-second grace. The child receives TERM at 41,370 seconds with the same grace, leaving time for final qualification. It uses CPUs 4–5, two numeric threads, CUDA hidden, an 8 GiB address-space cap, a 150 GiB free-space reserve, and a 64 GiB output limit sampled every 60 seconds. The existing shared preparation lock, STOP checks and owned-process cleanup remain. Fresh launch preflight matched all ten small pins and observed 265.503 GiB free; the parent owns the sole completion observer.

The pilot's inner-producer and process-wall extrapolations suggested roughly 6.16–8.70 hours, but startup, raw-row mix, compression and I/O differ across the corpus. Twelve hours is an allocation, not a throughput guarantee. Stop on the first integrity or resource failure, preserve partial output, and do not automatically restart or extend the budget.

On successful producer exit, a bounded final check verifies complete counts, the original full layout, published recipe metadata and producer proof coverage. It inherits the producer's raw/q-policy, source-state and copied-byte verification rather than rescanning every payload. [Compact launch evidence](evidence/sf-allmove-downside-full-launch-20260912.json) retains the exact preregistration, command, plan, source review and actual launch snapshot. Full completion has not been observed for this record; a prepared corpus would still require a separately selected training comparison. The active Ceres value training is unchanged.

## Relationship to earlier SF targets

This is a gentler variant of an already tested all-move attenuation family, not
our first all-move SF experiment. Ordinary Tactical100 used
`max(0.1, exp(-max(0, deficit-100)/100))` on every move, followed by normalization;
it also changed winning/losing mate targets categorically. Its completed fixed-400
comparison against B100 was −13.58 Elo with nominal paired 95% interval
[−52.41, +24.91], an unresolved result. It did not establish that all SF downside
information is useless. See the [completed Tactical100 record](2026-09-10-bt4-sf-tactical-training.md).

Downside300 retains full weight through 300 cp, uses only a relative factor of 0.5
beyond that, and changes no mate-domain row. At a 200 cp deficit, Tactical100's
weight was about 0.368 while the new weight is 1; at 400 cp the old floor gives 0.1
and the new weight is 0.5. Both preserve conditional BT4 odds only where their
weights agree; the new step rule preserves odds throughout each of its two sets.
The separately considered Tactical300 transfer rule instead gates on the SF-best
versus next-lower score and transfers donor mass to the best set. That distinction
explains missed near-good-move cases, but does not make Downside300 novel relative
to Tactical100's all-move use.

## Prospective training comparison (not launched)

If the running complete producer succeeds and the actual completed source is
qualified, train `B100Downside300` for one original matched epoch against the
unchanged B100 control. Policy is the only target intervention; original SF values,
source rows/order, initialization seed zero, original historical trainer/runtime,
batch 512, 36,935 updates and 420 windows remain fixed. Keep the existing 16/16
plan/load workers and 16,200-second training cap, with 21,630 seconds inclusive for
the coordinator. Corpus completion, output summary hashes, qualification and
prospective/realized schedule receipts are unresolved prerequisites, not fabricated
manifest values. The schema-3 profile allows training only; arena preparation is
separate and requires the genuine completed checkpoint.

After a valid epoch, run one fixed comparison against B100 at 400 simulations,
128 swapped pairs / 256 games, priors 1.0, the existing development panel
`3c955d68…`, seed 20260909, 300 plies and no tablebases. Preserve rolling concurrency
128 and batch cap 4,096; retain the qualified arena and strict capture-corrected
coordinator. Arena stage cap is 5,400 seconds, with 10,230 seconds inclusive of
lease wait and cleanup. This matches the previous screens and tests a substantive
change in intervention strength without spending compute on another grid.

Read the candidate score and nominal paired 95% interval after all registered
pairs finish. An interval above 0.5 supports this package at this seed and search
setting; below 0.5 favors B100; crossing 0.5 remains unresolved. No automatic
extension, threshold/dose fit, promotion or seed replication follows. A positive
estimate alone does not establish a gain, and one reused development panel cannot
resolve training-seed variance. Preserve failed runs; invalid qualification or
incomplete training/matches do not provide a negative scientific verdict.
