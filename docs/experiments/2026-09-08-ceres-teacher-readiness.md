# Ceres as an additional bootstrap policy teacher

Research direction and results, September 8, 2026. C3 passes CPU execution and
byte-adapter checks. A completed 128-position training sample finds fairly similar
Ceres/BT4 policies at temperature 1, with 88.28% weighted top-move agreement.
No bulk Ceres labeling, training or playing comparison has launched.

## Decision and hypothesis

The user prefers a bootstrap that retains useful Stockfish supervision and explores
ideas beyond reproducing LC0. Keep SF-derived value targets in these policy
comparisons, and prioritize a retained-SF policy candidate when exploring Ceres.
This preference does not erase the [B100 result](2026-09-08-bt4-pure-policy-endpoint.md):
pure sharpened BT4 policy is a useful diagnostic endpoint and development baseline.
The [running G50 comparison](2026-09-08-bt4-g50-b100-dose-comparison.md) directly tests
retaining half of the stored SF policy. Soft-SF preparation continues separately.

Hypothesis: Ceres supplies useful differences in move ranking beyond BT4, so replacing
some of BT4's contribution can improve a retained-SF bootstrap. Different architecture
or weights do not establish independent errors or removal of LC0 bias. CeresTrain
[supports LC0 and TPG training sources](https://github.com/dje-dev/CeresTrain/blob/main/src/Trainer/TrainingConfig/ConfigData.cs);
the inspected release descriptions do not establish the exact training-data lineage
of the C3 checkpoints. Treat that provenance as unresolved.

## Upstream candidates and local evidence

The public release API, checked September 8, reports:

| Candidate | Published | Release description | Role in initial qualification |
| --- | --- | --- | --- |
| [C3-768-30-pre8-I8](https://github.com/dje-dev/CeresNets/releases/tag/C3-768-30-pre8-I8) | 2026-05-19 | 300M parameters, byte input | Primary current-generation candidate |
| [C3-512-34-pre8-I8](https://github.com/dje-dev/CeresNets/releases/tag/C3-512-34-pre8-I8) | 2026-05-19 | 170M parameters, byte input | Smaller throughput alternative if needed |
| [C1-640-34-LEPNED-I8](https://github.com/dje-dev/CeresNets/releases/tag/C1-640-34-LEPNED-I8) | 2026-06-28 | Lepned tune of C1-640-34-I8 | Later publication date does not imply a newer architecture |

These descriptions do not establish the strongest one-node policy teacher. The
[CeresNets README](https://github.com/dje-dev/CeresNets) strength table covers older
networks and uses 1,000-node searches, not raw-policy ranking. Parameter count,
release ordering and the GitHub Latest badge cannot settle the choice.

The existing local Ceres inventory includes float-input C1-384-12, C1-512-15,
C1-640-25 and C1-640-34. Historical Ceres teacher audit artifacts exist; this is
not a completed C3 integration or a new strength result.

The repository already implements [Ceres TPG encoding](../../chess_anti_engine/encoding/ceres_tpg.py):
64 square records with 137 features, including a raw-byte primitive. The float
input is those bytes divided by 100. The audit found that the [ONNX adapter](../../chess_anti_engine/onnx/load.py)
and [foreign-net audit](../../scripts/foreign_net_audit.py) used only floating-point
input paths. They now share explicit byte-input handling, including the startup
WDL probe. Casting normalized input to uint8 would lose the intended byte values;
the adapter rejects that misuse. Also, the existing BT4 raw-label pipeline uses LC0 planes; changing only the
network filename cannot make it a Ceres labeler. Verify the downloaded graph's actual
inputs, outputs and operators before deciding that byte-input wiring is the only gap.

## Initial investigation plan

1. Inspect the selected release graph and pin its identity. Prefer a narrow extension
   of our existing ONNX/TPG path if it can execute the graph faithfully. Use Ceres's
   evaluator API if its additional inputs or operators make that the simpler route.
   Extract the full legal-move policy from one forward evaluation; a UCI bestmove
   or a one-visit search histogram is not an equivalent teacher distribution.
   Native `GetPolicy` also returns a [compressed representation](https://github.com/dje-dev/Ceres/blob/64558176ad4933d3cd85b5604133d6576de921d2/src/Ceres.Chess/LC0/Positions/Policy/CompressedPolicyVector.cs#L59-L129)
   with at most 80 move slots and quantized probabilities. Capture policy logits before that
   compression for full-distribution labels; preserve legal masking and normalize
   explicitly. The upstream [TensorRT evaluator](https://github.com/dje-dev/Ceres/blob/64558176ad4933d3cd85b5604133d6576de921d2/src/Ceres.Chess/NNEvaluators/Base/TensorRT/NNEvaluatorTensorRT.cs#L2453-L2486)
   demonstrates this softmax-to-compression boundary. A native evaluator API alone
   does not guarantee lossless labels.
2. Qualify history, side-to-move orientation, castling, promotions, legal-move mapping,
   input scaling and output normalization against the native evaluator or an
   independently established reference. Check precision and batch consistency.
   Keep raw distributions so later temperature or mixture changes reuse inference.
3. Use a small frozen training-only sample to measure ranking disagreements,
   entropy and measured inference cost. Include positions where SF and BT4 disagree.
   SF agreement is a diagnostic, not a veto on a Ceres playing test. Qualify on a
   reserved GPU slot; preserve the current G50 training and data preparation.
4. If valid and affordable, compare a retained-SF control with the same SF weight
   and a partial replacement of BT4 by Ceres. For example, at SF weight 0.5:
   control `0.5 S + 0.5 B`, challenger `0.5 S + 0.25 B + 0.25 Ceres`.
   This is an illustrative substantive comparison, not a launched recipe or a
   claim that these weights are optimal. Do not assume BT4's temperature 0.5 is
   appropriate for Ceres. Choose and record the Ceres temperature from training-only
   calibration before playing results, holding BT4's treatment fixed.

Before compute is committed, publish the exact network/runtime, sample, calibration
rule, baseline, budget and stopping rule. Routine qualification and experiment
choices remain within the autonomous research scope; no per-test user approval is
needed. If implementation cost grows substantially, record the blocker and preserve
compute for the existing broad target, horizon and scale comparisons.

A subsequent playing screen should use the existing economical shallow sequential
comparison plus protected deeper probe, with matched rows, initialization, training
budget and non-policy targets. Fresh-seed confirmation and larger-data transfer
remain necessary before choosing the bootstrap. Ceres is an additional substantive
family, not a reason to postpone those comparisons for a large mixture grid.


## Completed CPU graph and adapter checks

The [registered check](https://github.com/jjoshua2/DeepFin/pull/574#issuecomment-5587004050)
used eight fixed valid positions covering opening history, both sides to move,
castling, promotions and history-bearing en passant. The selected release has one
`squares_byte` UINT8 input of shape `[batch,64,137]`, a raw `policy` FLOAT16 head
with 1,858 entries, standard-domain opset 23, and no external initializers or
additional state input. The policy output ends in a linear layer, not a softmax.
Model SHA256: `44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3`.

ONNX Runtime 1.23.2 with extended optimization and CPUExecutionProvider successfully
executed the graph without the registered disabled-optimization fallback. All
policy logits were finite. Batch-one versus batch-four maximum logit difference was
0.001953125 and maximum legal-probability difference was 0.0003513172; all eight
top moves agreed. An independent reviewer reproduced the saved bank's results using
a separate direct UCI-table mapping. These are numerical consistency observations,
not probabilities of a strength improvement.

The [subsequent adapter check](https://github.com/jjoshua2/DeepFin/pull/574#issuecomment-5587118150)
reused the saved raw logits as its reference. The new original-byte adapter produced
identical input bytes, correctly classified the primary WDL head as logits, and
matched all **178 legal policy logits exactly** across the eight positions. This
checks the actual C3 adapter path as well as the synthetic regression fixtures.
No direct-session qualification was repeated for that integration check.

The implementation passed 125 distinct affected CPU test cases and whole lint.
Synthetic tests used ONNX Runtime 1.29.0; the real C3 checks above used 1.23.2.
Both float Ceres and LC0 regression coverage passed. The implementation validates
input arity/name/type and refuses normalized floating input for a byte graph.

The [compact readout](evidence/ceres-bootstrap/c3-pre8-cpu-readout.json) includes
model identity, fixed positions, source hashes, per-position results and the
[original input/logit bank](evidence/ceres-bootstrap/c3-pre8-cpu-policy-bank.npz).
The initial download-inspection script failed after extraction because it used
`hashlib.file_digest` under Python 3.10. Recovery hashed the existing archive/model
and read the graph; download and extraction were not repeated. That inspection
failure did not involve inference or change the model.

This establishes CPU compatibility with the selected release and our adapter.
It does not establish complete native Ceres input parity, CUDA compatibility,
labeling throughput or move-ranking quality. Source review supports the fixed
history-eight/fill/default-Q=0.03/zero-ply contract, but repetition-key edge cases
remain unverified. A separate nondefault Q rounding/saturation discrepancy was
found; the tested default is unaffected and the encoder was not changed here.

Next, bank a bounded training-only policy sample using the qualified path and source
history, and measure CUDA labeling cost in a reserved slot. Qualify a distinct Ceres
sidecar identity before bulk labeling; the BT4 labeler remains LC0-specific. Retain
SF in the proposed mixture comparison and preserve compute for horizon and scale
transfer rather than expanding into a large teacher grid.

## Training-only policy sample protocol

The registered diagnostic adds Ceres outputs to **128 positions** from the existing
[qualified 4,096-row training sample](2026-09-08-soft-sf-qualified-training-sample.md).
Within each of its 64 sampled shard groups, take the two lowest salted SHA256 ranks
of the source/shard/row identifiers. The selection is fixed before inspecting Ceres
outputs; no position is replaced because of its policy or result. Selection SHA256:
`2e24425b7ef5e9a84520b45a51c2dac81cc84bdc47a9c2eee2f52e74eac79d4f`.
An independent recomputation confirmed the 128 identities and conditional weight
factor 32. Actual source/shard/game clusters must be counted after joining the raw
records; a derived-shard/game tuple alone does not establish independent games.

Reuse the existing bank's raw history records and SF/C/BT4 targets. Reconstruct the
original available move history, check legal replay and source identity, and require
the reconstructed LC0 input key and float16 input tensor to match the bank. Feed
Ceres the corresponding original TPG bytes at the fixed defaults. This uses banked
data only: no original training-corpus scan or repeated BT4 inference.

Run one C3 CPU session with the same model identity as above, extended optimization,
primary `policy` output only, and batches of four. The collector uses the frozen
Python 3.13 / ONNX Runtime 1.29 environment: importing shared data helpers under the
older Python environment failed before payload access because its native extension
ABI differed. That preparation failure is not a Ceres model result. Actual graph
execution in the chosen environment is part of this diagnostic, not assumed from
the previous ONNX Runtime 1.23.2 check.

The complete attempt has a ten-minute hard ceiling and two CPU threads. Stop on an
identity, legality, finiteness, resource or runtime failure; preserve completed
batches and the failure, with no replacement rows or automatic retry. Save all 1,858
raw graph-output logits per selected position, along with legal-move mapping and
reference targets. Full distribution here means the graph's float16 output precision,
not unquantized training weights. No GPU, training or playing games are included.

Report descriptive weighted entropy/support, pairwise Jensen-Shannon divergence,
total variation, top-move agreement and mass assigned to the other teacher's leading
moves. Ceres and BT4 at temperature 1 are the primary distribution comparison;
BT4 at the historical temperature 0.5 is separately named recipe context. SF score
agreement and disagreement strata are diagnostics, not a teacher acceptance gate.
Use the parent sampling weights with the conditional subsampling factor; do not
attach ordinary iid confidence intervals to this stratified training subset.

No Ceres temperature, mixing weight or training winner is selected from this sample.
Its purpose is to reveal distribution differences and implementation problems before
bulk labeling and matched training. Favor a retained-SF comparison with SF's weight
held fixed when introducing Ceres; pure BT4 remains a useful diagnostic baseline.

## Completed training-sample readout

The [registered attempt](https://github.com/jjoshua2/DeepFin/pull/575#issuecomment-5587623614)
completed all **128 rows in 32 batches** under its original budget. History replay,
input keys, stored float16 tensors, all-legal support and final input/model checks
passed. The rows cover 64 derived shard groups and 90 raw shards; both recorded
source/worker/game and source/raw-shard/game groupings contain 128 distinct tuples.
This remains a stratified training sample, not independent confirmation.

Independent review recomputed the fixed selection and weights, reconstructed
histories, checked the original SF/C/BT4 bank, used an independent move-table mapping,
and reproduced the saved Ceres probabilities and aggregate metrics.

| Distribution | Weighted mean entropy (nats) | Weighted mean maximum probability |
| --- | ---: | ---: |
| Stored SF | 0.5407 | 79.51% |
| C20T05 | 0.4288 | 85.14% |
| BT4, T=1 | 2.0743 | 40.73% |
| BT4, historical T=0.5 | 1.1967 | 63.60% |
| Ceres C3, T=1 | 2.1627 | 37.88% |

Ceres and BT4 at the common temperature 1 have weighted mean **Jensen-Shannon
divergence 0.00946 nats**, **total variation 0.09220**, and **top-move agreement
88.28%**. They disagree on 15 of the 128 unweighted positions. Ceres is slightly
softer on this sample, and the two raw policies are fairly similar. Their Jensen-Shannon divergence
and total variation are smaller than those produced by sharpening BT4 to
temperature 0.5. Sharpening preserves BT4's top-move ordering; Ceres changes the
top move on the 15 observed disagreements. This does not establish equivalent rankings, a shared
training source, independent errors, or which teacher makes a stronger bootstrap.

The [readout and manifest](evidence/ceres-bootstrap/c3-training-sample-readout.json),
[per-position distributions and histories](evidence/ceres-bootstrap/c3-training-sample-rows.json),
[raw byte/logit bank](evidence/ceres-bootstrap/c3-training-sample-bank.npz), and
[frozen collector source](evidence/ceres-bootstrap/c3-training-sample-collector-source.txt)
are published. The bank concatenates the original batch arrays losslessly; the
manifest preserves each original batch hash. Source paths are logical repository
paths. The original source-bank qualification and host execution records remain
external, with their identities retained. The exported histories and policy arrays
support recomputing the main distribution comparisons without another model run.

Preparation encountered two corrected issues before the sample ran: the older
Python environment lacked a compatible native decoder import, and the first source
binding loop included virtualenv packages. The actual run used the recorded 3.13 /
ORT 1.29 environment and restricted source binding to repository modules. No sample
inference was repeated, no position was replaced, and no failed batch was converted
into successful evidence.

**Next decision:** keep Ceres as a candidate for partial BT4 replacement with a
retained SF contribution. The observed differences justify that option, but this
sample does not demonstrate better rankings or reduced shared bias. Finish the G50
and softened-SF comparisons and preserve the training-horizon/scale work before
committing to bulk Ceres labeling. A bounded CUDA compatibility/cost check and a
source-qualified Ceres sidecar path remain useful preparation for a later matched
training comparison. Temperature and mixture choices remain prospective; no fit or
winner was selected from these 128 positions.
