# Original A39, batch 32: public policy-anchor learner engineering

Status: **closed synthetic CPU/GPU engineering PASS, with root and independent
actual acceptance for preparation, both public learner arms and the saved-state
CPU audit**. The work used genuine original A39 weights. The one-row bank and
its outcome and acceptance authorities were fictional software fixtures. This is
not a real-data training result, recipe selection, playing-strength result or
live runtime adoption. The [first-week goal](2026-11-12-rl-week-one.md) remains
active.

## Question and fixed settings

The earlier [anchor software qualification](2026-11-16-frozen-policy-anchor-software-qualification.md)
checked the gradient and identity mechanisms on synthetic models and separately
qualified genuine A39 CPU assets. This engineering step asked whether the actual
public anchored-bank learner, at the intended full architecture and batch 32,
could bootstrap original A39 model-only, perform two updates, save, and reconcile
the complete saved state and sampling history without observer side effects.
It did not repeat the earlier isolated gradient oracle.

Execution source was frozen H revision
`31ba591bbda95e1530ce995a07bdbce24f00fb73`. Later main changes, including unrelated
UCI/tablebase work, were not adopted by that runtime. The original checkpoint was
SHA256 `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`,
with historical embedded step **680276**. Each arm started from that same complete
model with a **fresh optimizer at learner step 0**; neither restored the original
optimizer. The measured architecture had 494 state entries, 175 input planes,
**61,444,448 unique parameters** and 77,173,088 naive state elements. The tied
Smolgen storage is counted once for unique parameters.

| Setting | Both arms unless stated |
| --- | --- |
| Initial model | Original A39, strict complete model-only bootstrap |
| Arm A | λ=0, null reference and complete reference-work bypass |
| Arm C | λ=0.1, frozen original A39 reference at historical step 680276 |
| Anchor | Forward KL(reference legal policy ∥ student legal policy), on eligible original network-turn/has-policy rows with explicit legal support |
| Dose | Two applied updates, batch 32, 64 sampled views per arm |
| Learning rate | Constant 3e-6; peak/floor/warmup-start equal, warmup 0, scheduler first period 2048 |
| Seed and mirror | 2026111681; mirror probability 0.5 |
| CUDA | Eager public learner with BF16 autocast; one visible device |
| CPU execution | Affinity 20,22; Torch threads 2; inherited/observed interop 2 |

The population was one actual source-encoded and F16-serialized standard-start
row: 20 legal moves, one-hot `e2e4` policy and an explicitly invented DRAW target.
All nested bank/producer/outcome/audit/root/independent/closure authorities were
marked **FICTIONAL**. The unchanged public validators and replay decoder accepted
that software fixture for both arms. No whole game, real HOLD/TRAIN cohort or
scientific learner dose was produced.

## Initial preparation failure and selected correction

The first CPU preparation remained **FAILED**, charged **7.445893738 seconds**.
It stopped before GPU training with `KeyError` for
`chess_anti_engine/replay/generation.py` in the fixture's required-source map.
The 77 pinned files had been authenticated for identity without establishing
complete coverage of the APIs' required source sets. Seven required entries were
missing. The failed output and partial fixture were preserved without preparation
qualification or reuse.

Root selected a specific source correction before another attempt. V3 took the
union of the previously qualified H209 closure and the prior 77 pins: **210
immutable source pins**. Static resolution confirmed complete core, admission,
bank and anchor source rosters. Frozen project code and all helper function bodies
were unchanged; the prepare caps changed from 180/150 to 170/140 seconds. All
execution/output/binding names and five UUIDs were fresh. The corrected prepare
plus saved-audit ceilings and prior failed time fit the original CPU ceiling:
`7.445893738 + 170 + 420 = 597.445893738 <= 600` seconds.

The corrected CPU preparation closed PASS in **7.465713851 seconds**, with one
complete A39 CPU load, zero model forwards, optimizer actions or replay draws,
CUDA uninitialized and the CPU model released. No additional initial checkpoint
was manufactured. The actual generated contracts and stages, their source
identities and public command suffixes were banked before GPU admission.

## Two genuine public learner executions

Each GPU owner ran a source-pinned passive, in-process `runpy` observer that
delegated the actual `scripts.core_generation_bank_policy_anchor` public module
`main`, with the exact generated contract/stage suffix. This was **not a literal
bare `python -m` invocation**. Hooks observed existing student/reference forwards
and the real save boundary; they added no forward, replay draw, loss wrapper,
gradient mutation, optimizer update, reseed or retry.

| Observation | A: λ=0 | C: λ=0.1 |
| --- | ---: | ---: |
| Closed GPU owner wall seconds | 18.890893148 | 22.097915897 |
| Applied updates / sampled views | 2 / 64 | 2 / 64 |
| Existing student / reference forwards | 2 / 0 | 2 / 2 |
| Actual `Trainer.save` calls | 1 | 1 |
| Peak PyTorch allocated bytes | 1,887,301,632 (1.758 GiB) | 2,159,113,216 (2.011 GiB) |
| Peak PyTorch reserved bytes | 1,988,100,096 | 2,258,632,704 |
| Reported saved checkpoint bytes | 721,941,536 | 721,945,760 |

Both started with the exact A39 model state and an empty fresh optimizer. Both
completed learner step **0 → 2**, with zero nonfinite skips, transient retries or
clipping. Lambda zero made no reference forward and published null anchor
diagnostics. C made two no-grad reference forwards, with 64 eligible of 64 total
rows. Its finite mean legal KL was **0.0011541841086000204**, row sum
0.0738677829504013 and coefficient-weighted row sum 0.007386778295040131.
These are observed software diagnostics, not evidence of useful regularization.

The actual raw primary policy was CUDA FP32 `[32,1858]`; WDL was BF16 `[32,3]`
and the observed policy-query Linear output was BF16 `[32,64,512]`. BF16 autocast
was active for student and reference. This does not mean every operator or tensor
was BF16, and it does not qualify actor inference or F32 transport.

Reference setup preserved RNG. The complete observed live state, storage aliases,
gradients and RNG snapshots immediately before and after each actual save were
identical. Each public boundary matched the save's four global RNG streams.
The separate CPU audit below checked the bytes that were actually saved.

## Closed saved-state and sampling audit

The CPU audit closed PASS in **9.633832934 seconds**, with CUDA uninitialized.
Inside that owner, both saved checkpoints were authenticated before and after
weights-only CPU decode. Each complete saved Trainer payload matched its **own
actual live pre-save fingerprint**: model, optimizer, scheduler, step/LR, donor
provenance, zclip, architecture, parameter names and anchor metadata. The exact
public-family tag and original A39 reference identity were checked separately.

Both saved models were finite and complete across all 494 entries, with exact
shape/dtype and the tied storage aliases. Each optimizer had **479 named parameter
entries and 438 active slots**, with every active slot counter at step 2. The
438 count is an observation, not a selected architecture rule. Both saved
schedulers retained constant 3e-6; clipping mode was `none` and zclip remained
uninitialized. This audited the newly produced optimizer state; it did not restore
or qualify the original donor's optimizer or an adjacent GPU resume.

The audit read the actual physical journals inside its owned phase: two batches
of 32 draws of the synthetic row per arm. Journals, bank membership, all five final
RNG streams and backend settings matched across arms, as did their initial
model/optimizer/scheduler/zclip/RNG observations. Earlier arm receipts correctly
left payload and journal auditing pending; this later accepted audit resolves
those pending checks without rewriting the earlier cutoff.

Both saved model states changed from original A39, and the A/C states differed.
**Those differences are descriptive. GPU deterministic algorithms were disabled,
so this single ordered pair does not isolate a causal anchor effect.** It supplies
no strength, WDL-health, generalization or coefficient-optimality evidence. Anchor
host intervals were unsynchronized enqueue-time measurements, not CUDA latency;
arm wall times do not establish a relative performance result.

## Custody, resources and evidence

Before/final custody matched for all 210 source pins and the six existing native
images/records with 31 build inputs in frozen H. Exact interpreter, plain-pth,
numerical versions/origins, native freshness/ABI and final custody were guarded
inside each owned phase. This is a source/native custody bracket, not continuous
mapped-byte identity or adoption of newer main source.

Initial CUDA free memory was 32,432,455,680 bytes in each arm. The selected 8 GiB
PyTorch allocator ceiling was not a reservation or whole-process memory bound.
Host 64 GiB and disk 20 GiB were admission/boundary observations; R7's continuous
host floor remained 8 GiB. The original plus both saved checkpoints totalled
2,109,552,333 observed bytes, within the selected 1 GiB per-file and 4 GiB combined
acceptance limits. These were not hard write limits or unobserved-peak guarantees.

Total charged owner-wall cost was **24.545440523 CPU seconds**, including the
failed preparation, and **40.988809045 GPU seconds**, each within its 600-second
program ceiling. All four successful owners had supported EOF0 and root closure
of the supervisor and known births, plus independent actual acceptance. At the
final audit cutoff all **356 local owners were closed**. There was no new GPU
phase after the two arms.

The earlier anchor CPU qualification (38 fresh + 235 carried = 273 unique cases)
and explicit-HOLD CPU qualification (276 cases) were separate historical software
evidence. They were not rerun or added to a new test count here. Separate scorer
and extension source/CPU qualification remained pending; this engineering record
does not qualify their unrun test population. No tests, probes, numerical jobs or
payload reopening were performed to write this document.

All table paths are relative to the external operation root
`~/chess-artifacts/operations/rl-week-one-20261112-v1`. Large checkpoint, native,
shard and journal payloads remain external; only sealed aggregate metadata and
previous independent receipts were used to prepare this readout.

| Artifact | External path | SHA256 |
| --- | --- | --- |
| Final engineering package | `A39-B32-public-anchor-engineering-bindings-v2/final-engineering-package.json` | `58223f141ea04a2afa75395533d920405bbbe73ec4c6e4639d92968a87348e01` |
| Corrective selection | `A39-B32-public-anchor-engineering-corrective-selection-v1/selection.json` | `c1360e1e4bb3cad52ebeb7e97b46de5f735de79761d7aaa4bb09030ca57dd685` |
| Preserved prepare failure | `A39-B32-public-anchor-engineering-bindings-v1/prepare.root-failure-readout.json` | `f7cb380483a9a3fa9c82773f108652f8dcf2f8eea5631a0e13bcccd888f78b99` |
| Source V3 | `A39-B32-public-anchor-engineering-source-v3/manifest.json` | `236b5d1f223fc6cc388ff213b5756e50d446393cb4d5671b17dffa5037d81120` |
| Source V3 independent | `A39-B32-public-anchor-engineering-independent-source-review-v3/review.json` | `270c81b39283c505d0006296270460e0f8b7f0fc31b8795fccf9b84fbe97c006` |
| prepare root | `A39-B32-public-anchor-engineering-bindings-v2/prepare-root-acceptance.json` | `1b192836109ef67d7a46262764870ac188817a2f69319c24c510147638b0aab8` |
| prepare independent | `A39-B32-public-anchor-engineering-bindings-v2/prepare-independent-review.json` | `a7fbad8d8e1ac078349f21b7ea9b45ab3dce05ba023fd253a179dbb74d2e1f90` |
| prepare readout | `A39-B32-public-anchor-engineering-prepare-results-v2/readout.json` | `3952e5f0578070bb1c10ce2b289d593f7043e338b734a1cf925e4acd5083e382` |
| coefficient0 root | `A39-B32-public-anchor-engineering-bindings-v2/coefficient0-root-acceptance.json` | `2ffea798c351003a8a932c740558bc4b4da64429785bd3541b98f00647c66843` |
| coefficient0 independent | `A39-B32-public-anchor-engineering-bindings-v2/coefficient0-independent-review.json` | `2a18640a0831dcee584c970c0e115ca8ed9b9ce9bec93baa7fa6dfc601eeb62a` |
| coefficient0 readout | `A39-B32-public-anchor-engineering-coefficient0-results-v2/readout.json` | `99be566ab2e2f5ea5f5218808d79dc179ff07a9f1a70c2db2e2679f600c39268` |
| coefficient01 root | `A39-B32-public-anchor-engineering-bindings-v2/coefficient01-root-acceptance.json` | `3c61da6058cb9dad8a67394ab96db310736651341ae1d15cad229c11a0a7cfa2` |
| coefficient01 independent | `A39-B32-public-anchor-engineering-bindings-v2/coefficient01-independent-review.json` | `c0122318a9e0429a3d7840a5c90d3f188260e99ff8a06f81566b8f24aa01e015` |
| coefficient01 readout | `A39-B32-public-anchor-engineering-coefficient01-results-v2/readout.json` | `0c154c4c79a46c277aec595252f60026139a6258117fd83e713a6da53016f532` |
| audit root | `A39-B32-public-anchor-engineering-bindings-v2/audit-root-acceptance.json` | `e05da8a6832f707f7c975afe761b5a8b83c01c0ab9da0acd7779afd7996e5f3a` |
| audit independent | `A39-B32-public-anchor-engineering-bindings-v2/audit-independent-review.json` | `8cc25299a2b7c3c3c898c62bf98d2640a3a8b4f2a251ea4356bfd192a0aefb20` |
| audit readout | `A39-B32-public-anchor-engineering-audit-results-v2/readout.json` | `d4543065db9169c4b8f0b746f662b9fedc00acea67faba67630545b06f0e8423` |

The [prospective whole-game pilot](2026-11-16-fresh-whole-game-policy-anchor-pilot.md)
still requires its genuine collection, splits, accepted data, final phase bindings
and scientific gates. This engineering PASS does not admit that work or adopt a
runtime. It does not complete the first-week learning/strength goal. The cancelled
5k study, its successors and 150k proposal remain cancelled; the old retained queue
and THREE remain STOPPED.
