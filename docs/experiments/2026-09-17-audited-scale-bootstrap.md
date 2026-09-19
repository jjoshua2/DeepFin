# Audited expansion of the35M bootstrap

Status: implementation and CPU preparation plan; no new derivation, materialization,
prospective union scan or training launched by this change. Independent review pending.

## Question and unchanged recipe

The next substantive comparison is a larger bootstrap using the existing35M V50
recipe: 100% BT4 policy at teacher temperature0.5 and normalized arithmetic
50% stored SF /50% native BT4 WDL. B100 denotes a policy mixture weight, not100
search nodes. Ceres coverage is not required for this comparison.

A one-pass50M model versus the completed one-pass35M model changes both unique rows
and training exposure. It tests the larger training package, not the causal effect
of unique data alone. A matched-exposure control is needed for that narrower claim.
The deciding arena budget and rule must be recorded before that comparison; this
implementation does not impose the historical fixed512-game arena on it.

## Exact readiness snapshot

| Product | Rows | Remaining work |
| --- | ---: | --- |
| Trained/admitted35M V50 |35,314,577| Preserve unchanged |
| Additional earlier184 raw-shard selection |1,517,925| B100 complete; V50/admission pending |
| Additional saved512 raw-shard selection |4,219,426| SF derivation and joint adapter complete; B100/V50/admission pending |
| Total distinct baseline-eligible rows above |41,051,928| Only35,314,577 currently union-ready |
| Next1152 selected raw shards |9,556,704| Eligibility unknown; audit first |

The frozen next1152 selection comprises864 run06 and288 run07 closed shards.
Its physical source/shard roster is disjoint from the20 G10 cohorts in35M, the
additional184 selection and the completed512 selection. The original18,910,484-row
corpus has separate historical lineage. Physical-source disjointness is not global
FEN deduplication.

The next selection needs at least8,948,072 eligible rows to reach50M. Its raw upper
bound would produce50,608,632 rows. Applying the previous512 selection's retention
rate predicts50,540,027 total rows; this is only a planning estimate. The saved joint
pool remaining after this selection contains7,597,820 raw rows in915 shards.

Host-local evidence under `scratchpad/bt4_joint20/scale50_readiness_20260917/`:

- `readiness.json`: counts, missing stages and cost basis.
- `next1152_receipts.json`, SHA256
  `2d3ba422d5897812b850923b87b4b1552ea99b15e8fe75047b904c5508ca71ac`:
  frozen source-qualified selection, not an eligibility receipt.
- `run06_g10.admission.draft.json` and
  `run07_g10_companion4.admission.draft.json`: actual saved512 source receipts.
  Metadata admission passed3,165,724 rows/384 raw shards and1,053,702 rows/128 raw
  shards respectively. No payload scan or teacher evaluation was needed.

## CPU preparation order and costs

1. Materialize B100 for the completed4,219,426-row adapter and V50 for all5,737,351
   existing additional rows, preserving SF sources, B100 products and raw labels.
2. Audit the next1152 saved joint receipts in bounded blocks512+512+128. Freeze
   exact no-result drops and explicit baseline exclusions for each completed block.
   Select more rows only if the realized eligible total is below50M.
3. Derive SF targets from saved observations, adapt saved joint BT4 policy/native
   WDL through physical row offsets, then write B100 and V50 for each completed block.
4. Admit the append-only union, compute the frozen sampler's prospective schedule,
   and run the frozen trainer's selected-subset preflight. Pin these receipts before
   a training launch.

Linear extrapolations from completed local stages: next1152 audit2.37 CPU-hours,
SF derivation3.47h, joint adapter3.38h, B100 writer0.73h. Existing4.219M B100 adds
about0.32h. V50/admission costs remain unmeasured for this source route. The stages
therefore exceed10h sequentially; independent completed blocks may overlap within
available memory and CPU capacity. No new teacher inference is required.

The provisional additional disk budget is40GiB, with200GiB free at startup and a
150GiB reserve. Check actual output sizes and host headroom before launch. Preserve
all inputs and partial failures. Use bounded two-worker CPU stages alongside the
existing fleet only when resource guards allow them.

At the observed35M training rate,50M one pass is about5.63 GPU-hours; this is an
extrapolation. The explicit coordinator bounds are9h for training and12h overall,
with existing STOP, memory, disk and exclusive GPU guards. A bound is not a request
to wait or consume the full allocation.

## Admission implementation

`audited_source_admission.py` verifies saved eligibility/exclusion proof, exact
realized row counts, SF observation selectors, successful CPU derivation and adapter
process receipts, adapter manifest, shard order and physical-offset provenance.
It does not mark the growing raw corpus complete. `bt4_value_rewrite.py` exposes
this route only with the same pinned raw-adapted WDL input, records its provenance,
and retains the existing numeric rewrite and unchanged-column checks.

`combined_corpus_schedule.py` adds `audited-g10-selection` under the explicit
`audited-expanded-b100-native50-corpus-set` kind. Canonical source/config namespaces
and exact raw-shard rosters reject overlap with previous cohorts. The V50 product
must carry the same audited admission proof.

`combined_corpus_train.py` adds `audited50m_value_seed101` / `Expanded50M_V50`.
It accepts measured50–60M rows, requires the exact completed35M V50 predecessor
through its saved verifier, retains all first21 cohort records byte-for-byte at the
JSON level, and permits only audited additions. Historical35M dimensions and caps
remain unchanged. The frozen trainer, configuration, batch512, seed101, two loader
workers and one-pass game-epoch sampler remain unchanged. Completion verifies the
realized schedule and active value masks; this is not yet a training result.

Validation covers saved receipt rejection, foreign adapter/offset rejection,
actual raw-WDL-to-V50 array preservation, overlap rejection, exact historical
prefix preservation, measured-row admission and the old35M contracts. Focused CPU
tests, Ruff and type checks pass; no large-data writer or GPU stage has been run.
