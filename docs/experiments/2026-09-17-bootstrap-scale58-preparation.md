# Prepare a larger matched-recipe bootstrap from saved labels

The next scale experiment should train the existing B100T0.5 policy/V50 value
recipe on more distinct positions. It does not require more teacher inference:
the selected raw shards already contain the SF observations and have saved,
one-evaluation BT4 policy and native WDL sidecars. B100 means 100% BT4 policy,
not 100 search nodes. Ceres coverage is optional for this comparison.

This record describes preparation, not a completed 50M or 58M training corpus.
The active 1,152-shard audit/derivation has a start receipt; the downstream chain
was prepared and validated but had not been launched at this publication snapshot.
The CPU operators use cores8–9/two numeric threads and hide CUDA. They preserve
all inputs and partial outputs, and do not restart or modify the active pipeline.

## Exact inventory and intended comparison

| Source contribution | Rows | State at preparation |
|---|---:|---|
| Existing matched 35M union | 35,314,577 | Previously admitted/trained |
| Earlier184 raw-shard extension | 1,517,925 | Baseline, adapter and B100 complete; V50 pending |
| Saved512 raw-shard extension | 4,219,426 | Baseline and adapter complete; B100/V50 pending |
| Next1,152 saved joint shards | 9,556,704 **raw** | Audit/derivation started; eligible count unknown |
| Remaining915 saved joint shards | 7,597,820 **raw** | Frozen receipt selection; not audited/derived |

The first three contributions total41,051,928 eligible baseline rows, but only
the35M union is ready for the matched V50 training recipe. The next1,152 shards
provide a raw upper bound of50,608,632 total rows. Adding the remaining915 gives
an upper bound of58,206,452; actual retained rows will be lower. These are source
identities, not an extrapolation of raw disk size. Exact source/shard subtraction
excludes the earlier saved512 selection and the next1,152 selection from the
915-shard remainder. Existing historical35M cohort identities remain unchanged;
the final union verifier also rejects overlapping raw source/shard identities.

Applying the prior512 retention fraction would suggest roughly50.54M then58.08M,
but these estimates are not eligibility or admission evidence. We will use the
actual successful audit counts. A one-pass larger-data run also receives more
training examples than a one-pass35M run; its result must not be described as a
pure effect of distinct data. The planned35M continuation provides useful
exposure context but does not by itself eliminate every schedule confound.

## Executable preparation sequence

1. The running pipeline audits1,152 shards in512/512/128 blocks. Each successful
   audit automatically freezes the exact diagnostic exclusions and derives both
   raw sources with the existing strict exclusion loader. Its SF contract remains
   `uniform-d9`, phase-zero policy, latest-phase search value, temperature0.0005,
   floor0, and explicit physical-row provenance. No label generation occurs.
2. The downstream operator waits for that successful terminal receipt, checks its
   completed process/summary bindings, and processes the existing and new cohorts
   sequentially. Missing adapters join the saved BT4 policy/WDL through exact
   physical-row provenance. Missing B100 copies use global alpha1 at teacherT0.5.
   V50 copies replace only `search_wdl` with an equal SF/native-BT4 mixture.
3. Source admission verifies the actual exclusion, derivation and adapter
   receipts. The union preserves the historical21 cohorts and appends only the
   completed source-qualified cohorts. Recipe, row layout, source disjointness
   and prospective whole-game schedule are checked. Its terminal status is
   explicitly **not training admission**; training's full checks still apply.
4. Once that first larger union completes, the same CPU chain processes the
   remaining915 shards in512/403 blocks, then their adapter/B100/V50 products and
   the cumulative union. This preserves the intermediate approximately50M result
   instead of requiring the entire approximately58M preparation to finish first.

The admission implementation is commit`a71c40c910ef082f27176ff1fe2ade939046a5a1`.
The audit/deriver remains the previously qualified saved512 runtime
`6b6893bd59628ce9d09bc0748baba3e1c36960dd`. All operator plans pin their code,
source manifests, saved receipt selections and required completed inputs. Future
results are pinned from their actual successful receipts, not invented in advance.
The chain never retries a partial output automatically.

## Time, disk and memory

Previous measured costs imply approximately2.37h audit plus3.47h derivation for
the first1,152 shards. Their adapter is approximately3.38h, B100 about0.73h, plus
about0.32h B100 for the existing4.219M extension. V50 writing and the final
schedule have not been timed for this exact expanded route. The remainder adds
about4.6h audit/derivation plus adapter and writing. Full58M preparation should
not be promised within12h; the chain is useful unattended work alongside GPU
labeling and the35M continuation.

The running audit/derive operator has an8h total bound,48GiB available RAM at
startup/32GiB while running,180GiB free disk at startup/150GiB floor and12GiB
allocated output cap. Downstream50 has a10h work bound after an up-to8h dependency
wait,32GiB output cap and190GiB startup reserve. The final downstream58 stage uses
an8h work bound,24GiB output cap and180GiB startup reserve. Each retained150GiB disk
floor protects the host; the startup reserves account for each stage's own bounded
writes. The CPU chain serializes these stages, so they do not compete on cores8–9.

Combined new copies across the whole plan may consume roughly55–65GiB; this is a
planning estimate, not measured allocation. Additional verified cold-data
reclamation may be necessary to complete the58M path. The operators stop safely
if a reserve fails. They contain no source deletion or archive action. GPU
training and labeling must continue to have their separate resource budget.

## Validation and remaining work

Metadata-only checks verified all frozen1,152 and915 receipts against the saved
source snapshots, exact row totals and uniqueness. The exclusion builder was
checked against the previous successful512 audit and reproduced its4,219,426
eligible rows. Actual source admission succeeded for all four existing extension
cohorts, totaling5,737,351 rows. The existing184-row-cohort path reached correctly
parsed V50 command construction without executing the writer. Operator syntax
checks passed. No new payload audit, teacher inference or target-writing result
is claimed by these preparation checks.

Independent launch review and actual downstream completion remain outstanding at
this snapshot. Even after the corpus/schedule receipt completes, full training
admission and the matched experiment launch remain separate concrete steps.
Frozen paths, hashes and exact selection counts are in the
[compact evidence record](evidence/2026-09-17-bootstrap-scale58-preparation.json).
