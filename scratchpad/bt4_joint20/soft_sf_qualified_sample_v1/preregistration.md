# Qualified Soft-SF training sample, 2026-09-08

Prepared before payload/target inspection. Bank 4,096 preselected rows: partition
all 2,309 original SF output shards into 64 contiguous strata, select one shard
per stratum using the smallest seed-labelled SHA256, and select 64 rows by the
same target-independent hash rule. The exact 64 shard locations and 4,096 row
indices are already frozen in plan.json. No eligibility prefilter or replacement.
The population is the original 18,910,484 training rows, not the 4K FEN audit set.
This is a stratified cluster probability design; rows in a shard/game are not iid.
Retain inverse inclusion weights and source-qualified game IDs. Descriptive mean
entropy uses self-normalized inverse-probability weights; raw row-level records
permit other weighted/cluster summaries without resampling or new corpus reads.

Join sampled stored game/ply identifiers against original run03_s3 summary game
membership, then scan only its implicated raw shards (at most192, at most2GiB
compressed). Duplicate or missing source-qualified joins are fatal. Reconstruct
only the4,096 selected histories/targets with the actual current deriver: original
input_key must validate and float16 x/SF policy must match stored rows. Compare
all16 non-policy SF/C columns, sidecar source_key against stored x, and actual C
against its original stored-SF/raw-BT4/rank3/20cp/T0.5 recipe. Preserve original raw
rows, effective d9 cp/mate scores, history keys, shard/physical/derived row IDs,
stored SF/C targets, BT4, and all16 source non-policy columns in a new bank.
Do not infer raw identity from bare FEN or accept a missing score/move as zero.

After every selected join passes, compare cp temperatures10/20/40/80, preserving
original effective-cp mate-distance encoding. Report both ideal normalized and
float32→float16 stored+renormalized entropy/support/top1/raw-best support mass.
Choose the candidate closest in weighted mean stored entropy to actual C;
exact ties choose cooler. This is a descriptive training-control parameter,
not a selected GPU experiment, strength claim, or full-corpus derivation proof.
The earlier128-row bank's10cp nearest entropy is not an input to this decision.

One execution, CPU6,7/two numerical and Blosc threads, nice19/ionice3, GPU hidden.
Outer timeout TERM570s + KILL30s; collector stops ordinary work at560s. Output cap
512MiB sampled between work units; no corpus writes, inference or model loads.
Partial bank/failure evidence remains; no retry/resampling or temperature choice
on failure. STOP is checked between bounded work units. Source/code metadata pins
are checked before/after; every consumed Zarr chunk retains byte hash/stable stat,
and each bounded raw file gets a compressed hash and before/after stat. These are
sample-specific checks, not a repeat full-corpus payload qualification.

Budget feasibility is prospective: selected arrays may decompress512-row chunks
for sparse samples, so x alone has a worst case~10.94GiB decoded per SF or C copy
across64 full shards, though reads are compressed and processed one shard at a
time. Raw scope becomes known after stored-ID reads; exceeding its cap is a
truthful failed attempt, not grounds for an automatic wider scan. Actual wall,
CPU, maximum RSS and consumed identities will be in the completion receipt.
