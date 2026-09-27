# Selected-E one-teacher cost kernel (NO-LAUNCH)

This patch is a CPU-testable kernel, not a labeling launch or a throughput
measurement. `python3 -m scripts.selected_e_label_cost_screen` always reports
`NO-LAUNCH`. No entry point opens registered E payload, an ONNX model, a GPU
provider, or an output bank. The caller supplies already admitted rows and
sessions to `matched_cpu_kernel`; its diagnostic inference timings exclude
source read, session open, qualification and sealed readback. Do not use them
as a full-wall rate or owner credit.

The kernel applies the reviewed `factorial58-E-teacher-row-v0` hash to a
cohort-manifest digest, shard ordinal, row offset and routing-seed digest. It
checks row and target byte pins, distinct exact inputs, compact legal masks,
and the declared selected teacher before any session call. Selected calls
precede the dual arm's extra unselected calls, so the selected physical
batch32 roster is identical in both arms. BT4 uses the existing
`bt4_derived_wdl_sidecar.stored_feed`, `bt4_policy_dump.compact_legal_policy`,
`bt4_policy_mix._tempered_bt4_policy`, and native WDL validator. Ceres uses
the existing stored-x-to-TPG conversion, Leela gather map,
`ceres_target_mix.policy_target` at BT4 weight zero and
`ceres_value_mix.softmax`; every Ceres call is fixed32 with repeat-last-real
padding and named `policy`, `value`, `value2` FP16 heads. Both heads from one
routed teacher must match the independently pinned target bytes and the dual
arm component bitwise. A changed raw head or selected call feed also fails.

The D counterfactual gets both teacher observations for each same row and
builds the original E recipe **before** FP16 storage: policy is half the
normalized, *already stored FP16* BT4 T=0.5 policy plus half the Ceres legal
T=0.5 policy; WDL is half normalized native BT4 probabilities plus half
`0.6*softmax(value/0.55) + 0.4*softmax(value2/1.5)`. The implementation uses
the same underlying `ceres_target_mix.policy_target`,
`ceres_value_mix.copies.normalized`, and `ceres_value_mix.softmax` calls as
the frozen `bootstrap_sffree_targets.mixed_targets` builder. It pins both
FP16 D output heads per row, returns their byte arrays and aggregate hashes,
and refuses changed dual-only head bytes through the D blend pins. Synthetic
nonuniform policy and WDL fixtures check calibration, rounding and both S/D
bytes; their D hashes also matched the frozen E builder in a separate CPU
process. This does not constitute saved output/readback or full-wall timing.

The historical 58,090,688-row E bank is a **lower-assurance legacy source** for
this cost-only comparison. Its historical WDL-only tablebase check and lack of
a new full native all-row qualification must remain visible. A future adapter
must freeze a 12,288-row, 35-cohort, game-clustered sample, authenticate its
existing E receipts and selected source-shard bytes with bounded reads, pin
input/legal/independent component-target bytes, and supply source identity and
objective-mask evidence. It must refuse duplicate exact input/context keys
and incomplete rows. The existing BT4 raw sidecar writer requires a closed
raw JSON shard, while the Ceres derived writer requires a contiguous derived
shard and matching summary. A sparse E sample cannot be passed to either
writer by fabricating a source shard. The adapter should feed the kernel's
actual producer functions and separately bind the registered ONNX/runtime,
provider proof, 8 GiB per-provider arena cap, sole-GPU lease, process/RAM and
output bounds.

The hypothesis is that one teacher for both main heads reduces qualified
labeling wall without a material strength loss against original E. Original
E's completed seed-121 one-epoch result is a same-recipe reference, not an
authenticated production control. The decision gate is a new strict rule50
six-man Selected-E-versus-E arena at the exact matched training horizon:
**576 pairs/1,152 games**, color-swapped, all source/training/terminal gates
clean, and paired 95% candidate-minus-E Elo **lower bound > -20**. A point
estimate or incomplete bank does not advance. Only then may the wrapper run
the cost screen in a sole-GPU slot, without concurrent training or arena.

The wrapper must write sealed per-call feeds/heads and saved S/D-target
readback, independently rebuild both selected targets, and time cold session
open, input conversion, inference, target math, write/fsync, qualification
and teardown. The same 12,288 qualified unique rows are the S and D numerator
only if every row passes. Report inference rows/s separately from qualified
unique rows divided by full process wall and the measured `(W_D-W_S)/W_D`;
timeouts and failures are incomplete, not a smaller passing sample. Pin
models/runtime/provider/driver and enforce <=8 GiB per provider arena,
<=12 GiB process RSS, >=32 GiB available RAM, <=2 GiB output, two CPU numeric
threads, <=30 minutes per job and <=90 minutes through final readback. The
gross Ceres run06 329.64 rows/s is not an end-to-end Selected-E rate. A clean
small screen would diagnose label economics on legacy E rows, not grant source
reuse or extrapolate to 500M. It ranks before the longer zero-credit Ceres v8
source pilot only after the arena gate clears.

Historical planning provenance, supplementary to this self-contained record:
`/tmp/selected-e-one-teacher-cost-screen-plan-20260927/MEMO.md` (SHA-256
`59b5cc7867b6802aecdd8dd6296b4c128f971c13af62778c7d73bc7c14fabaab`)
and `/tmp/teacher-e-selected-vs-e-rule50-20260926/PREREGISTRATION.md`
(SHA-256 `29e264e76e5cc961144f42f8fc854bacddaea0e8447016a880a5289060f87e2c`).
