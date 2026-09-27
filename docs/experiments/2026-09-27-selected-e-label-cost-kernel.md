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

That wrapper may launch only after the exact Selected-E-versus-E seed-121
strict rule50 six-man arena completes 576 pairs/1,152 games with all
source/training/terminal gates and a paired 95% candidate-minus-E Elo **lower
bound > -20**. It must write sealed per-call feeds/heads and saved-target
readback, independently rebuild both selected targets, time cold open,
conversion, inference, target math, write/fsync, qualification and teardown,
and report qualified unique rows divided by full process wall. The gross
Ceres run06 329.64 rows/s is not such a rate. The source and decision gates
are in `/tmp/selected-e-one-teacher-cost-screen-plan-20260927/MEMO.md` (SHA-256
`59b5cc7867b6802aecdd8dd6296b4c128f971c13af62778c7d73bc7c14fabaab`)
and `/tmp/teacher-e-selected-vs-e-rule50-20260926/PREREGISTRATION.md`
(SHA-256 `29e264e76e5cc961144f42f8fc854bacddaea0e8447016a880a5289060f87e2c`).
