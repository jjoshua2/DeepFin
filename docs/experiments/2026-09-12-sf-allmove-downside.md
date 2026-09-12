# One all-move SF downside candidate

Status: the one-shard saved-source producer pilot passed (8,192 rows). No full
corpus rewrite, inference, training or playing result. The registered Ceres value comparisons retain
priority.

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
