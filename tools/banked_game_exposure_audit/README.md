# Banked game exposure audit

A pure-stdlib diagnostic for already accepted F12/Q50 TRAIN banks. It counts
physical, source-qualified game groups and retained rows, reports normalized row
exposure masses, and reconstructs batch game diversity only when both compiler
membership and permutation digests match. It reads no targets or payloads.

```sh
python tools/banked_game_exposure_audit/audit_exposure.py \
  --selection /absolute/selected-inputs.json \
  --selection-sha256 ACTUAL_SHA256 \
  --out /absolute/new-summary.json
```

This command is a reader, not an admission controller. For controlled study use,
the caller must separately select genuine bank/compiler/learner receipts and
admit the CPU work. Publication grants no actual inputs, budget or execution.
The output path must be new; existing files are never overwritten.

The selection JSON has exactly these top-level fields:

- `schema`: `ROOT_SELECTED_BANKED_GAME_EXPOSURE_AUDIT_INPUTS_V1`.
- `purpose`: `STRUCTURAL_EXPOSURE_ONLY_NO_SCIENTIFIC_SELECTION`.
- `limits`: the seven positive integer caps below, at or below their maxima.
- `banks`: an explicit list; no discovery or recursive reference traversal.

Each bank row has `study` (`F12` or `Q50`), `cycle` (1–12), `study_uuid`,
`expected_games` (192), and seven `{ "path": "/absolute/file.json",
"sha256": "64 lowercase hexadecimal characters" }` references:

| Reference key | Selected metadata |
| --- | --- |
| `bank_root` | Closed TRAIN bank Root8 acceptance |
| `bank_independent` | Its matching independent7 acceptance |
| `candidate` | The accepted `generation.json` game descriptors |
| `compiler_root` | Single-bank compiler Root9 acceptance |
| `compiler_plan` | Its one-bank, no-copy union plan |
| `contract` | The matching capped-cycle learner contract |
| `learner_technical` | The closed learner technical receipt |

References authenticate bytes; a status string alone does not authenticate an
issuer. The tool joins the selected receipts without repeating data admission.

| Limit key | Hard maximum |
| --- | ---: |
| `candidate_bytes` | 33,554,432 |
| `compact_bytes` | 4,194,304 |
| `bank_two_pass_bytes` | 134,217,728 |
| `total_two_pass_bytes` | 4,294,967,296 |
| `banks` | 24 |
| `games_per_bank` | 192 |
| `rows_per_bank` | 86,400 |

Both initial and final hash reads are reserved before opening each input. Files
must be canonical, sealed, regular JSON. One bank's decoded metadata is retained
at a time; the exclusive, fsynced summary is capped at 4 MiB. There is no budget
expansion, truncation, automatic retry, or partial-success output.

The output includes rows-per-game histograms and masses, exposure concentration,
and per-batch distinct-group counts including the final partial batch. Batch
diversity is **source-authenticated order reconstruction**, not journal replay.
An unsupported order or permutation mismatch reports diversity unavailable.
Unsupported historical technical schemas leave actual dose unavailable.
Referenced shards, proofs, models, books, rosters, quota ledgers, journals,
compiled manifests and saved order arrays are never opened or statted.

A physical group is qualified by run, collection, generation, source revision and
TRAIN role; `physical_game_id` is `collection_uuid:ordinal`. Separate origins are
never collapsed. New IDs do not prove independent trajectories or novel boards:
these studies reuse the same 192 TRAIN starts across cycles. Original and mirror
share one physical-row quota. Row fractions and the Kish-style reciprocal of
squared masses describe **exposure concentration**, not independent samples,
gradient ESS or measured gradient contribution. No board deduplication, semantic
trajectory equivalence, causality, optimum, strength or search-label independence
is inferred.
