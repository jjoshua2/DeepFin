# Full494 accepted-bank learner and CUDA actor handoff

Status: **closed software qualification PASS, with root and independent actual
acceptance**. A real public learner completed one synthetic update; its genuinely
closed checkpoint passed the actual collector actor loader and one CUDA BF16
inference. This establishes a bounded software handoff. Playing strength and the
[first-week research goal](2026-11-12-rl-week-one.md) remain unproven and active.

## Criterion recorded before execution

The question was whether the published bank-acceptance, new-family learner and
collector-lineage interfaces could carry the full architecture through a real
update and reload. The sealed external design V2 and each phase's concrete plan,
independent binding review and root admission preceded that phase. Their exact
identities are in the [compact evidence](evidence/2026-11-15-full494-bank-handoff-software.json).
This document publishes the subsequent technical readout.

The selected architecture was fixed: a transformer with embedding width 512, 16 layers and 16 heads,
split QKV, per-layer Smolgen, a relation basis and shared generator weights, coupled categorical
head, linear policy embedding, auxiliary policy width 128, replication-history
fix and `v2_threats`. The complete schema 20 manifest is banked in the evidence.
Counts, keys, shapes, dtypes and shared aliases had to match; padding keys,
trimming the model or silently changing configuration was forbidden.

The synthetic fixture used the standard start position, the shared encoder and
compact policy mapping, a legal `e2e4` one-hot target, all 20 legal moves, and a
synthetic DRAW label. Actual `ReplaySample` serialization and Zarr decoding were
required. Its initial model was freshly randomized at step 0; original A39
weights and research banks were not inputs.

**All initial producer, auditor, root/independent, outcome and serving authority
claims in the fixture graph are fictional.** They are labelled in the external
`FICTIONAL-SCOPE.json`. They exercise caller-trusted metadata schemas; they are
not evidence that a real game, collection or data audit occurred. The subsequent
learner execution, saved checkpoint and supervisor closure are genuine.

The learner settings were software-only: logical first generation 2, segment 1,
one update, batch 1, seed 45, mirror 0, fresh optimizer and constant LR 1e-5
(floor/peak/warmup-start equal, warmup 0, period 2). Success required one optimizer
posthook, one draw/view/sample, step 0 → 1, zero nonfinite skips or transient retries,
complete finite saved state and changed model weights. No recipe comparison was
selected. Any identity, count, finite-state, custody, capacity, control or deadline
failure stopped the qualification without automatic retry or widened limits.

Source review caught a native-custody gap before execution: the first helper
compared hashes without retaining observed image modes/stamps. That immutable
packet remains unrun. V2 requires read-only images and banks actual non-atime
image and record stamps; its correction and source review are separately pinned.

## Actual execution

The frozen child source was `4d2e1a984525bd053b10fed7085adeb6ecba2c5a`, tree
`c270475db5da51a143ed542b14a2408f5e9be3c3`, identical to published main
`36f85105f8d747f1c89089987fe0813e06b5bd87`. This includes
[bank acceptance](https://github.com/jjoshua2/DeepFin/pull/1344), the
[public learner](https://github.com/jjoshua2/DeepFin/pull/1346) and
[collector lineage bridge](https://github.com/jjoshua2/DeepFin/pull/1347).
The R7 supervisor remained at `ba9d6de1987902e1759d984532a4c63f091b9269`;
owner and learner source identities are distinct.

| Phase | Actual result | Owner wall seconds | Outer / inner cap |
| --- | --- | ---: | ---: |
| CPU prepare | Full model, real serializer/replay decode; zero draws/updates/CUDA | 5.376039559 | 300 / 270 |
| Public learner CPU | One counted update and view, step 0 → 1, zero skips/retries | 8.503416504 | 600 / 570 |
| CUDA loader/inference | Two tag refusals and one genuine positive | 12.639142276 | 300 / 270 |

The public learner used the unchanged `-m scripts.core_generation_bank_learner`
entry and exact contract/stage suffix. GNU timeout supplied 570 seconds plus a
15-second kill backstop. `env -C` selected the frozen source, cleared inherited
Python/backend overrides, and invoked the qualified plain-pth Python with `-B`
and without `-I`. The owner supplied affinity 20,22 and CUDA was hidden in CPU
phases. Actual interpreter/version/origin guards passed: Python 3.13.15,
Torch 2.14.0+cu130 and NumPy 2.2.6; observed Torch intra/inter-op counts were 2/2.
This is not a universal library-thread claim or a fresh package-directory census.

The full model measured 494 state entries, 175 planes, **61,444,448 unique
parameters** and 77,173,088 naive state elements. One shared alias group has 16
names; the naive sum counts shared Smolgen weights repeatedly.

The actual boundary recorded logical generation 2, segment 1, initial actor step 0,
start step 0 and final step 1. Its optimizer posthook, draw, training-step and sample
counts were each 1. All five RNG streams were banked in that boundary; continuation
restoration was not exercised. The supervisor then closed before the actor
predecessor envelope was issued. The serving descriptor denotes producer logical
generation 3 while its loaded learner tag remains generation 2/segment 1/step 1;
no generation 3 collection was run.

Inside the separately admitted GPU owner, actual payload inspection verified
complete finite model/optimizer/scheduler/zclip state, donor provenance and the
flattened optimizer manifest of 479 names. All 438 active optimizer slots had counter 1.
The model-state digest changed from the initial random model. Saved keys, shapes,
dtypes, aliases and unique tensor count matched before positive construction.

| Actual actor-loader attempt | Real weights-only decodes | Constructors | Result |
| --- | ---: | ---: | --- |
| Missing family tag | 1 | 0 | Specific loaded-tag refusal |
| Incorrect family tag | 1 | 0 | Specific loaded-tag refusal |
| Genuine saved tag and closed parent | 1 | 1 | Complete CUDA actor and one forward |

Faults were injected only in memory after the real decode. Genuine checkpoint
bytes and parent metadata remained unchanged. The positive used actual
`core_finite_collect.load_actor` and `DirectGPUEvaluator`, with no predicate mock.

The forward observed `cuda:0` input `(1,175,8,8)`, no-grad and CUDA bfloat16 autocast.
Raw WDL output and 204 observed Linear outputs were BF16. Several policy outputs
were float32: this does not establish that every operator was BF16. The actual
float32 transport was finite, with policy shape `(1,4672)` and WDL shape `(1,3)`;
the existing finite policy padding was preserved. `_inference_only` stayed false,
served tensor/file custody passed, and the model was released.

## Resources, custody and interpretation

The RTX 5090 observation had 32,432,455,680 free bytes before decode. Peak Torch
allocated/reserved memory was 974,753,792 / 1,029,701,632 bytes under the4 GiB
allocator cap. This excludes a claim about total process/driver/cuBLAS residency;
per-PID memory was explicitly unavailable. Initial and trained checkpoint sizes
combined were 967,880,929 bytes, within the observed1 GiB single/2 GiB combined
bounds, with no remaining serialization temporary at the closed-output check.

Host64 GiB and disk20 GiB were admission/boundary observations. R7's continuous RAM
floor was 8 GiB. The root's attempted in-phase CLI metadata observation arrived
after termination and queried no raw files/resources. Thus there is no claim of
continuous64 GiB availability, an enforced hard write limit or unobserved temporary
peak bytes. Actual saved-size acceptance ran inside the GPU owner.

Six read-only native images, their records and 31 source inputs passed actual
checks. Prepare-final, GPU-first and GPU-final observed snapshots matched under
frozen no-writer custody, bracketing the closed public CLI. The CLI also passed
its source/import guards. This is not CLI-local image hashing, mapped-byte
attestation or resistance to an undetected replace-and-restore. Native records
retain their explicitly allowed writable-mode contract while exact SHA/stamps
are compared. The selected `.pth` content/origins were checked; sole-pth
provenance remains the earlier static observation plus no-writer custody.

All three supported sessions ended with exit0, closed supervisors and known
births, and free custody leases. Cost is 13.879456063 CPU-owner wall seconds and
12.639142276 GPU-owner wall seconds, including decode/checks/cleanup. These are
not kernel timings, a throughput benchmark or evidence of sustained readiness.
Root and independent actual reviews accepted each phase with no findings.

This qualifies the exercised first-segment software handoff. Real accepted-bank
collection, a further learner segment, a sustained RL loop and research/runtime
adoption require separate work. No games or external-engine matches were played;
there is no Elo result, proof against multicore Rofchade, Stockfish or Lc0, or
evidence of an optimum RL formula. The cancelled study and stopped retained
queue remain stopped. The first-week and long-term strength objectives remain
active.
