# E4: outcome-bundle learning in a larger fresh-data regime

Status: **completed with no registered playing gain**. All 192 games qualified,
both arms completed 129 updates, and all three fixed 256-game screens passed
technical validation and independent review. The bundle contrast is unresolved;
neither arm qualifies for donor-utility confirmation. No candidate was selected,
no confirmation ran, and A39 remains retained. The
[completed E2TB study](2026-11-13-rl-resolved-outcome-bundle.md) established
no registered playing gain; the [E3 diagnostic](2026-11-13-rl-train-gradient-geometry.md)
left its primary geometry contrast unresolved. Those studies remain closed.
The first-week and long-term strength program remains active; E4 does not
establish an optimal RL formula or strength against Rofchade, Stockfish or Lc0.

## Question and controls

Does the same 20% verified-outcome WDL/categorical bundle improve playing over
the search-only control in a larger learning regime? Generate exactly 192 fresh
games with immutable A39, preassign 128 TRAIN and 64 HOLD by global identity-hash
rank, and train both arms for 129 matched updates of batch 256. Both start from
the same donor weights, empty original optimizer state, no SWA and step zero.
The model, heads, search settings, loss weights, optimizer, clipping and schedule
configuration remain the original recipe. Only the declared value-target bundle
differs between arms.

FULLA uses search-only main WDL and the original stored categorical control.
FULLB uses 80% search plus 20% genuine completed-outcome main WDL, with the
corresponding categorical target built from signed Q before the same F16
consumer boundary. Every usable original categorical control must bit-match
the qualified raw-F32-Q-to-HL-Gauss-to-F16 construction. This is a coupled
WDL/categorical intervention; policy and remaining targets stay common. Neither
arm introduces a Stockfish anchor or resets an auxiliary head.

This changes data size, update count, exposure and schedule phase relative to
E2TB. It does not isolate the effect of data quantity or dose. Each arm consumes
33,024 samples, averaging 258 sampled views per TRAIN game; actual per-row and
per-game exposure must be reported from the delivered cohort and draws.

## Fixed horizon and inputs

The original schedule has eight warmup updates and a 120-step square-root
release cycle. Update 129 is the final applied update before the restart at 130:
its applied Aurora LR is 7.136969439881711e-6 and auxiliary LR is
3.568484719940855e-7, or 11.894949% of peak. This is not the exact 10% floor.
The checkpoint after 129 already stores peak rates prepared for 130; those
stored rates must not be mistaken for the rates applied to update 129. The
endpoint was selected from source and the existing LR trace before E4 data or
learning. Snapshots 0/1/8 are diagnostic; only the fixed final 129 snapshots are
candidate endpoints.

Collection UUID is `cd47b230-f32f-4256-a51e-4869f5708bb6`. Fixed seeds are:

| Purpose | Seed |
| --- | ---: |
| Generation | 2026111340 |
| Ranking and splitting | 2026111341 |
| Draws | 2026111342 |
| Mirroring | 2026111343 |
| Common training randomness | 2026111344 |
| Screens | 2026111345 |
| Confirmation | 2026111346 |
| Offline bootstrap | 2026111347 |

Select 192 cohort starts, 128 screening pairs and 500 confirmation pairs from
the fixed preencoded 4,096-entry pool. Preserve the existing structural filters;
retire prior starts and known parent/history/normalized-position/raw-and-mirror
identities, including the previous 756 planned starts and all known E2 inputs.
Rank using the frozen new weighted-hash rule, never model scores or outcomes.
Insufficient eligible starts ends selection without enlargement or reseeding.
Actual R7 encoding of all 820 selected starts and their production mirrors must
qualify before generation. Broader family and pretraining overlap remain unknown.

Use at most 16 dynamic generation slots. Slot reuse follows completion order;
there are no per-slot TRAIN/HOLD quotas. Preserve full opening history, strict
tablebase/rule50 provenance, pure selfplay and the fixed 450-additional-ply cap.
All 192 games must pass integrity checks. All 128 preassigned TRAIN games must
be nonempty, usable, resolved and have verified outcome/raw-control lineage
before learning. There are no outcome-based replacements or completion quotas.
A valid censored HOLD game or missing eligible root makes the entire offline
diagnostic inconclusive; no subset is substituted and it does not veto an
otherwise valid TRAIN bank. Old HOLD 64 stays closed.

## Playing decisions

Run three fixed 256-game/128-opening-pair screens after technical qualification:
B129 versus A129, B129 versus donor, and A129 versus donor. All three are required
even if the first performance comparison fails; a technical failure or global
control stops the study. Preserve the prior internal matched-search profile:
64 simulations, temperature 0.1, noise enabled, eager serving with 32 slots and
maximum leaf batch 1,024, strict rule50 tablebases and 450 additional plies.
These internal comparisons have no per-game clock; the separate external
Rofchade baseline used 30+0.1. Paired means, sample variance with n-1 and z=1.96
intervals use whole opening pairs. Unavailable intervals cannot pass a gate.

For the scientific B-versus-A screen, a point score at least 53% and paired 95%
lower bound above 50% is screen support for the bundle; an upper bound below 50%
supports a disadvantage at this realization. Otherwise the comparison remains
unresolved. This screen alone does not confirm a general recipe benefit.

Both new 129 snapshots are prospectively eligible utility candidates. Each must
score at least 53% against the donor with paired 95% lower bound above 50%. If
exactly one qualifies, select it. If both qualify, select B when its B-versus-A
point score is at least 50%, including a tie; otherwise select A. If neither
qualifies or a required screen is technically invalid, no confirmation runs.
This selection does not establish superiority over the other trained arm.

Only the selected immutable candidate gets one fresh 1,000-game/500-pair
confirmation against the donor, with the same point/lower-bound criteria. No
other candidate replaces it after failure. Confirmation uses the preselected
fresh book and identical checkpoint hash. Do not pool studies or screens,
extend an endpoint after seeing a score, or reopen E2TB/E3. A confirmed result
would concern that candidate against the donor under this internal search
profile, not Elo, multicore Rofchade, Stockfish or Lc0 strength.

The new HOLD 64 diagnostic compares only donor/A129/B129 at one fixed root per
game. It cannot select a checkpoint, change a playing gate or reopen old HOLD.
Any selected-candidate confirmation is separate from the B-versus-A hypothesis.

## Resources and recovery

Proposed GPU-intended owner caps are 2,400 seconds for collection, 1,200 for paired
learning, 300 for offline scoring, 3,600 for each screen, 7,200 for the conditional
confirmation and 360 shared qualification seconds: 22,260 total. Each stage is
one-use and at most 7,200 seconds, within the original week-one deadline and
172,800-second week budget. CPU preparation/readouts have their own finite
owners. Use cores 20/22 and two Torch threads, shared leases and controls, one
compute stage at a time, no paid compute. These caps do not guarantee completion.

Preserve failures, partials and every stop marker. No automatic reseed, quota
replacement, training retry, alternate endpoint or budget extension follows a
failure. Existing immutable donor and prior negative results remain recovery
anchors. Source packets, actual immutable templates, config/profile identities,
fresh structural selection and independent review must be reconciled before
their corresponding execution. Preparation alone is not execution authority.

The separately proposed P0 Stockfish anchor, matched-SL optimizer restoration
and higher LR are distinct hypotheses. They are not E4 interventions. A passive
raw-SF observer is selected prospectively to preserve existing query evidence
before parser reduction and record thinning. Its source, dispatcher and eight
synthetic controls have received independent acceptance. It adds no queries
or training targets, and archive failure does not change core data eligibility.
Logging and the changed import order can affect timing and batching. The
synthetic checks do not establish actual worker concurrency, complete teacher
coverage, P0 label equivalence or which fallback result stamped a record.

Bulk artifacts stay under `~/chess-artifacts/operations/rl-week-one-20261112-v1/`.
The week-one and long-term strength program remains active; no strength
milestone or optimal RL formula is established by preparation alone.

## Actual readout: November 13

The [root readout](evidence/rl-e4-larger-regime-readout/root-readout.json) and
[fixed decision](evidence/rl-e4-larger-regime-readout/decision.json) record the
completed experiment. All three endpoints are technically qualified, but none
passes its registered positive-performance gate:

| Candidate | Reference | Wins–draws–losses | Score | Paired 95% interval | Registered result |
| --- | --- | ---: | ---: | ---: | --- |
| FULLB129 | FULLA129 | 80–99–77 | 50.59% | 46.32–54.85% | Bundle contrast unresolved |
| FULLB129 | A39 | 68–103–85 | 46.68% | 42.06–51.30% | Not utility eligible |
| FULLA129 | A39 | 71–105–80 | 48.24% | 43.47–53.02% | Not utility eligible |

The intervals use the precommitted 128 color-reversed opening-pair means,
sample variance with n-1 and z=1.96. Both donor differences also remain
unresolved for disadvantage: neither upper bound is below 50%. This is no
registered gain, not a finding of equivalence or proven harm. The separately
[reviewed decision](evidence/rl-e4-larger-regime-readout/decision-review.json)
is `NO_CONFIRMATION`, with both eligibility flags false, no selected candidate,
confirmation allowed/consumed both false and adoption false. The confirmation
stage and output remained absent. There is no intermediate or old-model fallback.

### Data and applied updates

The [cohort audit](evidence/rl-e4-larger-regime-readout/cohort-audit.json) accepted
all 192 planned attempts and integrity checks, all 128 TRAIN games, and the
all-628-playing-starts versus all-192-retained-inputs raw/mirror exclusion gate.
TRAIN contains 10,516 rows from 68 natural and 60 strict tablebase resolutions;
HOLD contains 5,284 rows from 33 natural and 31 strict tablebase resolutions.
There were no outcome-based replacements. All 128 fresh TRAIN proofs preceded
array consumption; actual raw-F32-Q categorical controls were validated before
the one fixed draw plan was generated.

The [actual checkpoint audit](evidence/rl-e4-larger-regime-readout/learner-cpu-readout.json)
and [independent review](evidence/rl-e4-larger-regime-readout/learner-review.json)
accepted 258 total applied updates, 33,024 samples per arm, matched physical
draws/host transformations/RNG/math and no skips or retries. Both step-zero
models equal all 494 donor state tensors and start with empty optimizers and
no SWA. Final model and stored optimizer tensors passed CPU finiteness and
ownership checks. Only final step 129 was evaluated for utility. Actual update
129 used the registered LR tail; the saved next-update peak was not applied.

The hard cap bound on 127/129 updates in FULLA and 128/129 in FULLB. These
rates concern the auxiliary AdamW parameter group; Aurora's matrix group is
outside that clipping scope. This is a conditional diagnostic, not evidence
that the whole trunk's learning rate was capped and not a recipe change.

This was one finite A39 collection followed by learning. The 129-step models
were not republished to generate another actor generation. Average sampled
views per retained TRAIN position were 3.1404, compared with 3.3997 in E2TB
(10,240/3,012). More independent games and total updates did not increase
average reuse or constitute a sustained closed-loop RL test.

### Whole-64 offline diagnostic

All 64 fixed HOLD roots qualified before branch predictions. Strict complete
donor/A129/B129 checkpoints were scored sequentially using the declared eager
BF16 production adapter, batch 64 and stored F16 inputs widened to F32. The
closed CPU reader recomputed metrics from the raw saved outputs, with 10,000
paired whole-game bootstrap samples and seed 2026111347.

Expected-score MSE means were 0.116145 for the donor, 0.107774 for FULLA129 and
0.106350 for FULLB129. B-minus-A was -0.001424 with percentile 95% interval
[-0.005151, 0.002263]. Every paired value-error interval includes zero. The
[accepted diagnostic](evidence/rl-e4-larger-regime-readout/offline-root-acceptance.json)
therefore remains unresolved; policy TV is drift, not playing quality. This
endpoint neither selected a model nor changed a playing gate. Old HOLD arrays
stayed closed.

### Cost, custody and interpretation

| GPU-intended stage | Charged owner-wall seconds |
| --- | ---: |
| Collection, including freezing/cleanup | 1,140.21 |
| Both 129-update arms | 125.18 |
| Whole-64 three-model inference | 13.33 |
| All three playing screens | 2,266.50 |
| Total | 3,545.22 |

The six GPU owners and all CPU readouts closed. Total GPU-intended owner time
was 59.09 minutes, within the 22,260-second E4 cap; learner time was 3.53% of
that total. Immutable checkpoint SHA/stat/read-only custody and actual loader
logs join the playing results to their models. The reader does not claim a
per-game served-state hash or per-forward finite-logit/precision instrumentation.
The intervals concern one training realization and this internal search profile;
they establish no Elo or external-engine result.

The next planning direction is a source-grounded LC0-style baseline adapted to
the available hardware, with a sustained preplanned dose, checkpoints and donor
comparisons. Departures from LC0 must be stated, and additional shortcuts or
auxiliary objectives tested individually. That plan is not an executed recipe
or a new training admission. The cumulative strength ambition remains open.

## Technical correction before draw planning

A source audit during collection found that the original draw planner expected
`science.components.draw_plan_binder`, which the registered JSON does not contain.
The protocol pins the same inputs through the final learning-template packet
and paired-protocol template. The original guard would raise `KeyError` before
creating output, importing Torch or loading arrays. Prepublication integration
review missed this check; its draw-guard compatibility claim
is superseded by the [finding](evidence/rl-e4-draw-contract-repair/original-finding.json).

The corrected consumer validates that exact frozen reference graph and requires
a separate published technical amendment authorizing its helper, entrypoint,
manifest and descriptor. Its five scientific/data identity inputs remain the
original inputs; a sixth input supplies that publication authority. The
original protocol and helper remain preserved. This amendment changes source,
entrypoint and manifest identities and is separate from collection's original
seven-field binding.

The draw generation, proof and loader logic, seed 2026111342, shape 129 by 256,
cohort, targets, learning schedule and playing decisions stayed fixed. Collection
completed under its original protocol. The correction passed independent
review and 22 focused CPU guard controls, then was published in
[PR #1257](https://github.com/jjoshua2/DeepFin/pull/1257) before the single actual
draw execution. Synthetic publication-predicate controls did not substitute
for the separately verified
[actual main publication](evidence/rl-e4-larger-regime-readout/technical-main-publication.json).
The complete-cohort and finite resource gates also passed. See the
[technical amendment](evidence/rl-e4-draw-contract-repair/amendment.json) and its
[source proof](evidence/rl-e4-draw-contract-repair/source-proof.json).

## Frozen record and supporting evidence

The authoritative [protocol](evidence/rl-e4-larger-regime/protocol.json) has SHA256
`7cfd6e45b565f39827a1801c828dc302fe5d79f176c7701b004fa7f1adab3478`.
The [publication manifest](evidence/rl-e4-larger-regime/publication-manifest.json)
identifies 99 exact source, template and CPU-evidence copies. The
[independent integration review](evidence/rl-e4-larger-regime/science-integration-review.json)
accepted the scientific rules and component bindings for publication; its
draw-guard compatibility claim is corrected above. [PR #1256](https://github.com/jjoshua2/DeepFin/pull/1256)
published the original record, followed by the canonical public CLI check and
finite admission for collection.

Actual CPU checks cover all 820 raw/mirror encodings and the effective config,
12 focused learner/offline boundaries, 15 decision cases and eight passive
observer cases. The [limited A/A reuse receipt](evidence/rl-e4-larger-regime/limited-AA-reuse.json)
preserves the distinction between historical four-update evidence and these
new synthetic checks. It does not claim a newly executed 129-update A/A test.

The [completed-readout publication manifest](evidence/rl-e4-larger-regime-readout/publication-manifest.json)
identifies 42 exact compact metadata copies. Bulk shards, draw arrays, raw
prediction arrays, checkpoints and game logs remain outside checkouts at the
paths and hashes recorded by those receipts. All scientific verdicts above
use the frozen rules; publication adds no new experiment or adoption.
