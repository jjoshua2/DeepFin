# Bootstrap and evaluation Syzygy correctness audit — September 23

Status: IN PROGRESS. The user requires correct tablebase adjudication and search
support across bootstrap data and all evaluation paths. This record supersedes
any blanket assurance that existing matches already used six-man adjudication.
No active training runtime or existing data has been changed.

## Confirmed existing-experiment evidence

The [small-file audit](evidence/bootstrap-syzygy-audit-20260923/factorial-small-file-audit.json)
rechecks the four factorial game banks, their receipts, bound plans, initial-state
and summary files. All checked hashes match. Each match has 256 games with 128
paired openings, both colors, and the same opening identities across matches.
Recomputing scores reproduces B-A +6.7867, C-A +21.7431, D-C -16.2980 and
D-B +17.6584 Elo. These remain results of the actual recorded protocol, not a
corrected tablebase-aware protocol. This audit did not rehash model tensors or
all corpus data.

A-D training summaries agree on seed121, initial tensor hash, 58,090,688 rows,
113,459 updates and complete exact-epoch receipts. They explicitly mark the
historical control comparison invalid because of the changed sampler, no live
config comparison and no held-out purity receipt. The matched new factorial is
not the historical replacement-sampled control; held-out purity remains unknown.

## Confirmed gaps

1. Every actual factorial arena header records empty Syzygy paths and
   `syzygy_max_pieces=0`. E-D's queued plan had the same settings. The four
   completed arms were matched to each other, but did not meet the user's
   six-man evaluation requirement.
2. The four banks store starting positions and results, not move histories.
   They cannot be retrospectively re-adjudicated at the first tablebase position.
   Max-ply draw counts are B-A 7, C-A 4, D-C 0 and D-B 6; natural-result games
   may also have crossed tablebase positions, so these counts do not bound impact.
3. The exact arena runtime82298a5 routes its Syzygy argument to terminal
   adjudication only. Its match picker does not pass the existing tablebase probe
   into compiled Gumbel search. Enabling the CLI option alone cannot establish
   tablebase-guided move selection.
4. Existing theoretical adjudication treats cursed wins/blessed losses as
   decisive. The search probe maps those classes to draws but does not account
   for the current halfmove clock when marking other WDL hits solved. The
   intended game-rule contract and DTZ boundary behavior need qualification
   before those helpers are presented as exact fifty-move-aware support.

## Immediate containment and ongoing checks

The parent held only the queued E-D item under the scheduler's `gpu.lock`,
read back the result, and preserved every other item and all original plan files.
The [hold receipt](evidence/bootstrap-syzygy-audit-20260923/pending-eval-hold.json)
records before/after hashes. E training remains running; the held second seed
remains held. Corrected evaluation will use fresh identities and a preregistered
protocol rather than relabel old results.

Independent agents are tracing actual corpus generation/teacher materialization,
trainer target consumption and row alignment, and evaluation search/adjudication
entry points. Preliminary corpus metadata indicates all accepted A-E roots have
at least seven pieces; that finding and whether terminal outcomes affect the
trained targets are still being verified. No broad integrity PASS is claimed.

## Bootstrap data and value targets

The [teacher-path audit](evidence/bootstrap-syzygy-audit-20260923/teacher-path-audit.md)
traces source games, stored neural heads, materialization and actual trainer use.
All 35 accepted base summaries declare a seven-piece banking minimum, totaling
58,090,688 rows. This is source-code and receipt evidence, not a full decoded-row
census. The source generator adjudicates at six pieces; missing material can
continue play in the historical path. The derivations dropped 1,354,417 rows for
missing game results, without enough evidence to attribute those drops to missing
tablebases versus ordinary unresolved caps.

A–E train on `search_wdl`, with game-outcome fraction zero. Six-man game results
therefore affect source termination/retention and stored `wdl_target`, but do not
directly supply these experiments' value gradient. Raw BT4/Ceres inference is not
a tablebase search: its outputs remain raw teacher observations. Any exact-value
correction needs a separately declared target recipe and its own qualification.

A D/E compressed-policy hash discrepancy was investigated independently. The full
first shard (8,192 rows) has byte-identical decoded float16 policies; another
reviewer checked 3,584 rows across separated chunks in three cohorts with exact
parity. Replaying both frozen recipes on 16 physical rows reproduced stored
policies and intentionally different WDL mixtures exactly. The physical Blosc
payload differences do not establish a policy intervention. These bounded checks
are not a full-corpus logical-array equality proof.

The [bootstrap route matrix](evidence/bootstrap-syzygy-audit-20260923/bootstrap-route-matrix.md)
covers SF source generation, future BT4 root-only generation, optional search,
Ceres labeling/generation, and target consumption. Required future behavior is
explicit rule-aware outcomes, strict required-table failures, and observable
coverage; raw teacher heads and historical experiments are preserved. A future
BT4 outcome/stepper patch is in progress. SF persistent schema/resume integration
and Ceres generation remain unqualified; neither is covered by an arena-only fix.

A [bounded real-table smoke](evidence/bootstrap-syzygy-audit-20260923/real-tablebase-smoke.json)
opened the actual configured pair (1,000 WDL and 1,000 DTZ material entries) and
verified four valid three-man fixtures through the new shared helper: winning and
losing reset-clock KQK, drawn KBK, and a positive-clock KQK boundary that remains
unresolved. This confirms local real-file probing, not exhaustive six-man material
coverage, search integration, or a production-generation qualification.

## Bounded source-outcome check

The [raw-source prefix audit](evidence/bootstrap-syzygy-audit-20260923/bootstrap-terminal-sample.json)
read 6,153 rows from the first worker shard of each of run03/run06/run07, stopping
at 12 distinct games per source. Among these 36 games, 27 carry Syzygy terminal
FENs. All 27 have clock zero, raw WDL in {-2,0,+2}, and the stored game result
matches a fresh local tablebase probe under the corrected convention. No outcome
mismatch was found in this sample. This convenience sample does not estimate a
corpus-wide error rate, establish all material availability, or prove membership
of every sampled game in the accepted 58M rows. The bounded audit script is banked
beside the result; no source data was modified.

A separate [real six-man entry](evidence/bootstrap-syzygy-audit-20260923/real-sixman-entry.json)
checks a legal seven-piece b2xc3 capture from clock 70. It reaches six pieces at
clock zero and the shared helper agrees with the real WDL/DTZ loss verdict for
Black. This qualifies one actual capture boundary, not the whole generation path.

Independent helper review removed another overclaim: even a short DTZ plus a
positive halfmove clock does not alone prove a decisive match result against
possible repetition history. The reviewed helper certifies decisive positions only
at clock zero; positive-clock decisive WDL remains unresolved. Draw categories
and natural claimable outcomes retain their explicit handling. Future bootstrap
consumers must account for unresolved games rather than manufacture labels.

## E121–D121 evaluation amendment, before any result

The replacement evaluation retains the two trained checkpoints, 128 paired
openings/256 games, seed 2026092101, 400 simulations, training search shape,
policy prior temperature 1.0, and play temperature 0.1. Both sides use the same
reviewed strict six-man root/leaf and adjudication protocol, with the actual
WDL+DTZ path pair and saved PGNs. Its fresh plan/runtime/output identities will
be recorded before queue admission. The original disabled-tablebase plan remains
held evidence and will not be rewritten.

Raise the precommitted maximum game length from 300 to 1,000 plies to avoid
turning an arbitrary unfinished position into a draw. Any still-unresolved game
invalidates the fixed-size comparison; no selected game may be dropped to pass.
The new arena wall cap is 7,200 seconds with a 7,800-second supervising bound.
These are recovery limits, not duration forecasts. Require all 256 game records,
PGNs and the corrected protocol identity before a successful completion receipt.
A wall-cap partial bank is incomplete and cannot settle the contrast.

The first-seed result remains an opening-paired estimate conditional on these
checkpoints. It can inform the practical labeling-cost decision, but does not
prove seed robustness. The user's hold on the second seed remains unchanged.
Do not pool the corrected-protocol contrast as though it shared the old four
matches' disabled-tablebase protocol. No training recipe or active E process is
changed by this evaluation amendment.

Shared primitive publication: [PR #853](https://github.com/jjoshua2/DeepFin/pull/853),
reviewed remote head `0d089488955b3b34c4c535ad39c44967b12230e0`.
This alone is not a completed consumer integration or permission to claim all
bootstrap/evaluation routes have adopted it.

## Actual search and engine capability checks

A [16-simulation compiled Gumbel CPU fixture](evidence/bootstrap-syzygy-audit-20260923/actual-c-search-smoke.json)
used a synthetic zero evaluator and the actual local Syzygy pair. From the
seven-piece capture-boundary position it recorded eight eligible probes/eight
hits and a winning root value near 1. The test exercised the actual native tree
and shared rule-aware leaf probe, supplementing the fake-tree unit tests. It
used no GPU or production network; production arena qualification is separate.

The [installed Stockfish UCI advertisement](evidence/bootstrap-syzygy-audit-20260923/stockfish-uci-capabilities.json)
reports support for Syzygy50MoveRule (default true), ProbeDepth (default 1) and
ProbeLimit (default 7). This confirms the current binary's advertised capability
and defaults, not historical effective values or realized search hits. The future
strict bootstrap mode will request and stamp the six-man settings explicitly.

## Arena implementation review

The matched-simulation integration now forwards the same strict probe to both
sides' compiled Gumbel searches and uses the shared result rule for adjudication.
Unsupported time-mode/Python-search combinations are refused explicitly. PGN is
required, resume fingerprints include the new protocol, unresolved capped games
raise, and final results record invocation-scoped probe/hit and table-capacity
counts. A regression verifies a capture on the final permitted ply is adjudicated
before cap classification in both loop implementations.

The author ran 54 focused CPU cases including the Gumbel selfplay chain. Parent
review found and closed the last-ply omission, inspected the frozen diff, and
independently passed 36 helper/plumbing cases. The separate actual native-search
fixture above passed as well. Scoped Ruff/Basedpyright pass. Whole-repository lint
was attempted; local dependency resolution blocked whole Basedpyright. Shared
helper PR #853 subsequently passed hosted lint and PEXT checks (ordinary test
job still pending when checked). No actual GPU match or production-network result
is claimed by these checks. Other evaluation routes remain unqualified.

## Bootstrap consumer implementation and publication

The reviewed SF corpus implementation is published in
[PR #856](https://github.com/jjoshua2/DeepFin/pull/856), exact head
`ba5cbaa9a6c72ad1d0ccbc73e016ab346767f9e8`, stacked on shared helper
PR #853. New generation explicitly selects the outcome convention; strict
mode requests six-man, rule-aware Stockfish probing and uses the shared
adjudicator. The convention travels through raw rows, qualified manifests,
derived summaries and committed shard attributes. The raw-baseline audit,
ordinary and selected policy rewrites, and rank sidecar validate it against
qualified source provenance. Review caught and fixed both omitted consumer
arguments and a raw-versus-derived summary-shape mismatch.

Independent review passed. Four strict serialized-source consumer fixtures and
67 nearby selected tests passed, supplementing the earlier 481 selected cases.
Scoped lint passed; whole local type checking remained dependency-blocked.
Shared helper PR #853 subsequently passed all hosted lint, PEXT and test jobs.
The [SF implementation record](https://github.com/jjoshua2/DeepFin/blob/ba5cbaa9a6c72ad1d0ccbc73e016ab346767f9e8/docs/experiments/2026-09-23-sf-rule50-corpus-mode.md) is on
PR #856's branch until the stacked changes are merged. Its actual-engine
seven-to-six capture fixture observed 73 tablebase hits within 1,024 nodes;
this demonstrates one real search path, not historical corpus coverage.

The future BT4 outcome/stepper integration is separately published as
[PR #855](https://github.com/jjoshua2/DeepFin/pull/855), head
`85051aa078022c5a2a8b1833b9204236d3ccf80a`. These are reviewed changes,
not a claim of deployment. Historical source data is not rewritten. Raw neural
teacher outputs remain raw outputs; a tablebase-corrected training target is
an explicit different recipe. Existing A–E value recipes give terminal game
outcomes zero weight. Full BT4/Ceres generation/writer integration and every
evaluation route are not yet qualified.

A read-only entrypoint inventory further limits the coverage claim. BT4 raw,
derived-WDL and policy sidecars, and the Ceres derived sidecar, run direct ONNX
inference on saved positions; none performs a tablebase-guided search. The
reviewed BT4 root stepper is an in-memory actor with root adjudication, not a
runnable searched generator with a corpus writer. A future BT4 worker must wire
the strict probe into search leaves before claiming tablebase-guided move choice
from seven-piece roots; Ceres needs a live root/leaf adapter as well. The older
random bootstrap generators also lack this integration and are not qualified
alternatives for the planned corpus. None of these observations changes saved
teacher predictions or the running E recipe.


## Corrected evaluation registration

The separately reviewed operation is now queued as
`factorial58_sffree_20260923_E_D_rule50`, after the active E training job.
The [registration receipt](evidence/bootstrap-syzygy-audit-20260923/ed-rule50-registration.json)
records the atomic queue edit; every pre-existing item was preserved, including
the original held E_D and second-seed holds. The serial supervisor waits for E;
the operation additionally requires successful complete E/D donor receipts
before taking the GPU lease. The queue's dependency label alone is not the gate.

Runtime is the immutable arena commit
`6d22d1d6e3b8503144987bfda206559594af2039`. Final plan SHA256 is
`2e21158bb1dba36b2446d60d10f736ca89416328cf104e0649092ab959570d5a`;
registered descriptor SHA256 is
`5400361b4adeee9dd60178ea71191f9aff545b26b11df7353017c8594159ed03`.
The operation directory is
`/home/josh/chess-artifacts/operations/factorial58-sffree-training-20260922/E_D_rule50_20260923`.
The copied original D_C configuration is pinned via
`CHESS_ANTI_ENGINE_LIVE_CONFIG` in both supervisor and arena environments;
its complete realized search record is compared with the old D_C baseline.
This prevents newer runtime defaults from silently changing the contrast.

Independent review closed a completion-gate issue: the runner now retains its
unreaped session leader as an ownership anchor and verifies process-group
cleanup before PASS. Fourteen focused CPU tests and the parent's final static
check passed. Completion requires all 256 games/128 pairs, bank/result/PGN
consistency, strict Syzygy telemetry, exact donor binding and start/end input
integrity checks. The outer supervision limit is 7,860 seconds, allowing cleanup
beyond the preregistered internal bound; this is not an ETA.

Tablebase provenance checks inventory names/sizes, not file-content hashes.
PGNs start at opening FEN, so the independent verifier cannot reprove every
natural repetition draw involving pre-opening history; it admits only the
explicit limited draw case, never an unproved decisive result. No completed
GPU result is claimed at registration. Any later second-seed release must also
amend its dependency from the old held E_D item; it is not released here.

## September 23 reboot recovery and completed E–D readout

Post-reboot inspection found both experiments completed before the reboot:
E at 03:42:03 UTC, and the corrected match at 04:05:28 UTC. Their outer
scheduler terminal receipts both have return code zero; the supervisor logged
`QUEUE_IDLE` at 04:05:48 UTC. No interrupted epoch remains to resume.
E completed all 58,090,688 rows and 113,459 updates at seed 121. Preserve its
checkpoint rather than rerunning a completed epoch.

Independent receipt verification checked all 105 E-linked and 16 arena-linked
file hashes, including donor checkpoints and the match bank/PGN/results, with
no mismatch. The full 256-game/128-pair bank rederives a score of 133/256
(51.953125%), paired standard error 0.02049188, and **E–D +13.58 Elo,
95% interval [-14.35, +41.68]**. The strict arena recorded 164,767 search probes
and 120,037 hits, and completed all pairs without truncation. This qualifies
this particular arena execution beyond the earlier CPU fixtures.

The original SF-free preregistration states that a nonnegative point estimate
supports provisional E for its collection-cost benefit, while neither sign
establishes universal teacher usefulness. This result meets that screen's
provisional-E rule. It does not establish a strength gain, show SF never helps,
or decide whether cheap/targeted SF corrections are valuable. The first-seed
contrast uses the same registered policy recipe; bounded decoded checks found
policy parity, while the value mixture changed;
it is conditional on these checkpoints and the corrected Syzygy protocol.
The expensive second seed remains held, consistent with the user's instruction;
no sample extension is made to chase significance.

Recovery verified the external data mount and available storage. The four
replacement Sol 6 xhigh lanes cover receipt verification, a CPU-only NumPy THP
pilot and its independent review, and the experimental BT4 generation/writer
integration. New pilot launches require their own frozen admission; restoring
workers does not imply a new training run or deployment of unmerged fixes.

The [independent recovery receipt](evidence/bootstrap-syzygy-audit-20260923/reboot-factorial-readout.json) banks the exact preregistration quote, input hashes and recomputed statistics.
