# Frozen-reference policy anchor: software and asset qualification

Status: **closed CPU/GPU software qualification and genuine CPU asset readiness
PASS, with root and independent actual acceptance**. This is a
technical software readout. No research coefficient, real-data training run,
playing-strength result or live adoption follows from it. The
[first-week goal](2026-11-12-rl-week-one.md) remains active.

## Question and scope fixed before execution

The [opt-in frozen-reference family](../frozen_reference_policy_anchor.md) adds
forward KL(reference legal policy || student legal policy) to the actual Trainer
backward path. The reference is frozen, no-grad and outside the student's
optimizer. It receives the same eligible post-mirror/dropout inputs and legal
support. Eligibility is original network-turn × has-policy; the anchor contribution
to illegal and inactive logit gradients must remain zero. The coefficient-zero path
bypasses all reference work. Existing evaluation loss/ruler and old-family contracts remain unchanged.

The question here was whether that software path has the intended causal gradient,
state, identity and custody behavior. Separate root protocols, source/binding
reviews, bounded owners and fresh admissions preceded execution. **λ=1 was a
software contrast, not a research coefficient selection.** These checks establish
neither that anchoring improves chess nor that this coefficient is suitable for RL.

The full494 GPU comparison reused the earlier
[synthetic public-learner handoff](2026-11-15-full494-bank-handoff-software.md):
its genuinely saved warm student at step 1 and separately frozen initial random
reference at step 0. The population was one source-encoded standard-start row,
legal `e2e4` one-hot policy, explicit legal support and synthetic DRAW target.
The original bank/root/outcome authorities in that fixture graph remain fictional.
Neither original A39 weights nor a real accepted bank supplied this GPU comparison.

Both branches restored the same full CPU warm model, named optimizer, scheduler,
zclip, counters and banked RNG state. Each then used one draw, batch 1 and one
actual optimizer update, step 1 → 2, LR 1e-5 and mirror 0. A newly selected paired
CUDA seed 45 was explicit. This is not CPU-to-CUDA exact continuation. Reference
setup preserved the paired RNG state; restored-state and final five-stream RNG
fingerprints matched, and backend settings agreed. No loss
rescaling, transient retry or hidden numerical fallback was permitted.

## Source and CPU qualification

Frozen execution source was `b10c831e807dc57d0761fec9c39e3c9bdce1bc90`.
[PR 1350](https://github.com/jjoshua2/DeepFin/pull/1350) merged as
`066bad86b70b5c106814079294127abb4a45952e`; both trees are exactly
`e68d69c2e059302576bc7e8909256942a703d166`. All five required CI checks passed at
run `43786757848`. The three pytest command summaries within the primary CI test job
(capped, nonserial and serial) are separate populations; their counts are not
summed into a new unique-case claim.

| CPU owner | Actual test component | Overall result | Owner wall seconds |
| --- | --- | --- | ---: |
| Initial source `055a2ea…` | 273 cases: 269 passed, 4 failed | FAILED; fixture configuration and refusal-order fixes required | 66.513642882 |
| Corrected source `ec119720…` | All 273 unique cases passed, zero failures/errors/skips | FAILED; whole lint reported 1 type error and 1 warning; final native custody not reached | 178.496146256 |
| Final source `b10c831…` | 38 fresh cases in 17 functions, zero failures/errors/skips | PASS; whole lint and final source/native custody passed | 127.643743386 |

The final three-file correction made the fixture payload type explicit, replaced
the callback-mutated decode integer with a mutable count retaining the exact-one
pre/post-load guards, and detached the already-backpropagated loss only for its
scalar assertion. It changed no anchor formula or gradient predicate.

The final evidence is **38 fresh + 235 carried = 273 unique cases across source
snapshots**, not 273 freshly rerun cases. The eight carried legacy/ruler test files
and their relevant paths were unchanged; their exact case identities and source
blobs are pinned to the earlier completed JUnit. The changed positive constructor,
reference, public caller and actor paths were exercised by the fresh 38 cases.
Repeated cases earn no additive unique credit. Both earlier owners remain FAILED.
Final whole-repository lint reported Ruff PASS, Pyright zero errors/warnings/notes,
empty Vulture output and `lint: OK`.

Coverage includes actual Tiny/Transformer backward and analytic gradient controls,
coefficient-zero state/RNG equivalence, mask/refusal/alias/freeze/mirror/relations
checks, synthetic public anchored-bank caller integration and direct checkpoint
continuation references. The actor capture uses inert construction; it does not
establish full494 CUDA serving. Unit-fixture historical step 9000 is fictional and
must not be confused with the genuine donor's observed step below.

The final owner checked the interpreter and sole plain-pth content at entry, then
verified numerical import origins after loading the runtime. It checked six native
images/records and all 31 declared build inputs before fixture imports and repeated
the source/native custody checks at the end. Torch threads were 2 and CUDA remained
uninitialized. Its closure plus 235 prior component passes were independently
accepted. Total CPU software cost,
including both failures, was **372.653532524 owner-wall seconds**.

## Full494 CUDA software result

The first GPU owner remains **FAILED and charged 12.689394674 seconds**. It stopped
with `finite core update failed; retries forbidden`; its combined assertion did
not publish actual dtype evidence. Source analysis suggested an assertion/interface
mismatch, but the old actual dtype remains UNKNOWN. A separately reviewed and
admitted successor calibrated the oracle to the actual FP32 primary-policy node;
it did not change production loss math or refund the failed owner.

The successor closed PASS in **48.645387621 / 300 GPU-owner seconds**, with an
inner 270-second bound. Root and independent actual reviews accepted the completed GPU evidence with no
findings.
Both branches completed one actual `train_steps` update and optimizer posthook,
one draw/view and zero nonfinite skips/transient retries. All 438 active optimizer
slots ended at step 2. The full architecture measured 494 state entries, 175
input planes, **61,444,448 unique parameters** and 77,173,088 naive state elements.
The shared Smolgen alias group is counted once for unique parameters.

The reference made one no-grad forward for the one eligible row. Its diagnostics
recorded mean legal KL 0.0041531380. At the actual logit node, the measured gradient
difference matched F32(p−q): maximum absolute residual
`3.3527612686157227e-08`, within the fixed coordinate-wise bound
`2e-7 + 2e-5 × abs(expected)`. All 20 legal coordinates met the decisive-coordinate
criterion; illegal gradient differences were exactly zero. The reference stayed
frozen, separate from student storage and optimizer, and without parameter gradients.

The actual raw primary policy and its gradient were FP32 over 1,858 compact
outputs, consistent with the attention policy's FP32 temperature promotion.
CUDA/autocast was active, and observed policy query Linear and raw WDL outputs
were BF16 for student/reference paths. This does not establish that every operator
or intermediate was BF16. It is not a BF16-rounded logit-gradient oracle.

Updated model states differed: 430 of 479 unique parameter tensors, spanning
55,383,691 elements, changed between baseline and anchor, with maximum absolute
difference `2.0789913833141327e-05`. Optimizer-only differences would not satisfy
the criterion. The positive branch performed real `Trainer.save` and weights-only
saved-state verification: 722,011,824 bytes, SHA256
`d0d459951f4d566f4a4c0f3d6523e866eb5f374bb2f1f07a311e936766a769e4`.
That payload was inspected only inside the owner; this document copies metadata.

The RTX 5090 had 32,432,455,680 free bytes initially. The single cumulative
whole-phase peak was **1,529,196,544 allocated bytes** (1.424 GiB) and
1,581,252,608 reserved bytes, including student/reference setup, both updates,
positive save/verification and final observation. The 4 GiB PyTorch allocator
limit is not a GPU reservation, process-wide memory bound or neighbor guarantee.
Per-PID GPU memory was unavailable and not queried. Host 64 GiB and disk 20 GiB
were admission/major-boundary checks; R7's continuous RAM floor remained 8 GiB.

Observed synchronized train intervals were 5.3188 s baseline and 3.5725 s anchor,
with 1.6627 s separate positive setup. These are one ordered, instrumented warm
software pair, including synchronization and other work. They are not steady-state
throughput, isolated reference latency or a relative performance result.

This GPU phase exercised the real Trainer under no-retry instrumentation; it did
not execute the public anchored finite-bank CLI or an adjacent GPU resume, serve
an actor, or collect a game. Actual public CLI/data and saved-state auditing remain
necessary for real pipeline engineering.

## Genuine A39 CPU assets are a separate result

The asset owner closed PASS in **8.468794622 / 180 CPU seconds** (inner 150), with
root and independent actual acceptance. It used genuine original A39 checkpoint
SHA256 `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
The actual raw embedded step was **680276**, observed inside the owned decode.
Its strict complete CPU model-only load measured the same 494 states, 175 planes,
61,444,448 unique parameters and 77,173,088 naive elements. There was one model
construction, zero model forwards, no optimizer restoration/steps, zero games or
dataset labels, and no CUDA initialization. Original donor optimizer compatibility
is not established by this weights-only readiness result.

Five native images and their complete records were transferred and checked for
the current asset checkout: features, lc0, perft, mcts and nnue. This is the
collector's exact five-module subset, excluding optional TB C, with 23 used
dependencies and the unchanged full 31-input source proof. Actual read-only
image/record custody, current-source hash/stamp custody, ABI/symbol freshness and
import-origin checks passed.
No six-to-five rename or unqualified image was substituted.

Python tablebases mapped 1,000 WDL and 1,000 DTZ tables under inventory
`207f7bc99c9113b237bee40ef1dc4f64f29bd3524279b6637da31032a2305ae5`.
The identity covers resolved paths, sizes and mtimes, not content completeness.
Two public KQK checks exercised strict rule50 handling: halfmove 0 resolved to a
TB win with DTZ 13; halfmove 100 resolved to a natural draw. This is observed
availability on those fixtures, not proof for every table/position.

Pinned Stockfish SHA256
`2764e7ef36bee844347586c8025ba81464315352128ef250f8c609f76044a68e`
received requested Threads 1, Hash 16 MiB, MultiPV 1 and 64 nodes on the public
standard start with full-history arguments. Options/readiness and a legal bestmove
passed; it reported 68 nodes. Existing close forced shutdown with return code −9,
and the engine was reaped. This is not graceful shutdown, a hard 64-node cap,
effective-option introspection, labels, a timed match or a strength qualification.

The root subsequently issued source-compatible consumer envelopes for native,
TB and SF identities. They permit future consumers to authenticate readiness;
they do not authorize collection, learning or research. The helper child itself
never attested owner closure. Root/independent review joined the R7 closure with
all known births, including a separately captured fast Stockfish child.

## Evidence, cost and remaining decision

All paths in this table are relative to the external operation root
`~/chess-artifacts/operations/rl-week-one-20261112-v1`.
Raw checkpoints, native images/records and TB/SF payloads were not reopened to
write this document. The two owned anchor GPU attempts cost 61.334782295 seconds
including the failure. At this cutoff all 348 local owners were closed, and the
week's all-GPU owner-wall ledger was **76,318.80925468204 / 172,800 seconds**.
Failed execution and cleanup remain charged; no unused historical credit is reused.

| Artifact | External path | SHA256 |
| --- | --- | --- |
| CPU first failure | `frozen-reference-policy-anchor-unit-bindings-v2/root-failure-readout.json` | `4501d1f5288d7cb80b9b670dc1adb7a58f1faedd92966adacb650bbc8adffcea` |
| CPU lint failure | `frozen-reference-policy-anchor-unit-bindings-v3/root-lint-failure-readout.json` | `77a6851ddc79776a459dfce7504005e8420a8e4c98fb7e91ee4d354f031b6273` |
| CPU final acceptance | `frozen-reference-policy-anchor-unit-bindings-v4/root-acceptance.json` | `1fa9bf8a30dd3444c139d812e4ee530f7e6f2b203185017888978777e38eb8d3` |
| CPU independent actual | `frozen-reference-policy-anchor-CPU-independent-evidence-review-v1/review.json` | `91e21feb11cd56c89e4eec16aa87f3a7b4e509a348ce6cf2ae03872a7491345f` |
| GPU first failure | `frozen-reference-policy-anchor-full494-GPU-bindings-v1/root-failure-readout.json` | `a3c420875c1c6eecfb511af6255db907204e86c0cc5b009111a05fef3549d4e1` |
| GPU V3 helper | `frozen-reference-policy-anchor-full494-GPU-source-preparation-v3/qualify.py` | `34581f2e9fa3e1af974f67688587cd95dc258b4ef0285e31eccbaa8d966ffb79` |
| GPU final protocol | `frozen-reference-policy-anchor-full494-GPU-bindings-v2/protocol.json` | `07a1f09d6a594fdfac67053382fa18a071c9ba362e30fa026fb797a5c2036c9d` |
| GPU root acceptance | `frozen-reference-policy-anchor-full494-GPU-bindings-v2/root-acceptance.json` | `9f6c9aa82d7e87acbff769109d2dfef04d30bb49f4da7edc1cf516989afa092f` |
| GPU actual readout | `frozen-reference-policy-anchor-full494-GPU-results-v2/readout.json` | `a89581390141ad5afb41689b617fcaa64bf434f8ca82730791f869f47c8a2a0c` |
| GPU independent actual | `frozen-reference-policy-anchor-full494-GPU-independent-actual-review-v1/review.json` | `17fc46e122169ed8df31127b2614108724702a57f813d5d655f0db0848701824` |
| Asset root acceptance | `collector-asset-qualification-bindings-v1/root-acceptance.json` | `ef60540da200be45c9cf303bee9b769803bb17cf0f13165529b92e889d733c6c` |
| Asset independent actual | `collector-asset-qualification-independent-actual-review-v1/review.json` | `179d937352b36ad06bc9341040f66326f29ffe8840633d73e9b151c6e69f3e99` |
| Asset actual readout | `collector-asset-qualification-results-v1/readout.json` | `9429630a09c7292f76b5a1f04a053f0891c9bd8aa7e2492f7a69a007df870539` |
| Asset freshness/symbol gate log | `collector-asset-qualification-results-v1/native-freshness.log` | `df271326a623ae5696d6d00bf8116dae07003faa6eaff819d24d8d81a16d089e` |
| Asset consumer envelopes | `collector-asset-qualification-bindings-v1/consumer-envelopes-v1/manifest.json` | `2725a19d84c058d8f3df003b7c9c6cf6b1c129b905b08677a8d1a62e1553e728` |
| Source main publication | `frozen-reference-policy-anchor-main-publication-v1/publication.json` | `a4dea746e24f90b4aa154347532f4cc17b6d79f7aebefc0a9995530416403c25` |

Document preparation performed no numerical execution or payload reopening.
Independent helper reviews came from other actors where this document's author
had authored a CPU driver or asset helper; the actual receipts disclose those
foundation limits. CI and local software owners are separate evidence populations.

The next research question is still unselected. Fresh whole-game collection,
explicit data split/exclusions, genuine closed audit/promotion, controls, a public
preregistration and independent confirmation are needed before any playing-strength
claim. Neither the λ=1 software effect nor asset availability selects a recipe or
establishes Elo gain. No live runtime adoption or goal completion is recorded.
The cancelled 5k study, its successors and 150k proposal remain cancelled;
the old retained queue and THREE remain STOPPED.
