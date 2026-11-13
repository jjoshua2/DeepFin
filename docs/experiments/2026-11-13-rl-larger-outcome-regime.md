# E4: outcome-bundle learning in a larger fresh-data regime

Status: preregistered parameters; research execution pending. The fixed structural
selection has completed
and received independent acceptance: 192 cohort starts, 128 screening pairs and
500 confirmation pairs. All 820 actual network encodings and mirrors, the
canonical collection descriptor and the 129-step config have passed CPU
qualification and independent review. The public CLI check remains pending.
No E4 game generation, learning or playing has run. The
[completed E2TB study](2026-11-13-rl-resolved-outcome-bundle.md) established
no registered playing gain; the [E3 diagnostic](2026-11-13-rl-train-gradient-geometry.md)
left its primary geometry contrast unresolved. Those studies remain closed and
the A39 donor remains retained.

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
milestone or optimal RL formula is established by this preparation.

## Frozen record and supporting evidence

The authoritative [protocol](evidence/rl-e4-larger-regime/protocol.json) has SHA256
`7cfd6e45b565f39827a1801c828dc302fe5d79f176c7701b004fa7f1adab3478`.
The [publication manifest](evidence/rl-e4-larger-regime/publication-manifest.json)
identifies 99 exact source, template and CPU-evidence copies. The
[independent integration review](evidence/rl-e4-larger-regime/science-integration-review.json)
accepts the scientific rules and component bindings for publication. Main
publication, the actual canonical public CLI check and fresh finite resource
admission remain required before collection.

Actual CPU checks cover all 820 raw/mirror encodings and the effective config,
12 focused learner/offline boundaries, 15 decision cases and eight passive
observer cases. The [limited A/A reuse receipt](evidence/rl-e4-larger-regime/limited-AA-reuse.json)
preserves the distinction between historical four-update evidence and these
new synthetic checks. It does not claim a newly executed 129-update A/A test.
