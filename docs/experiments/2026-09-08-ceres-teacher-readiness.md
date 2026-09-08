# Ceres as an additional bootstrap policy teacher

Research direction and compatibility audit, September 8, 2026. The selected C3
graph now passes a bounded CPU execution and byte-adapter check. No Ceres corpus
labeling, training or playing comparison has launched.

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
