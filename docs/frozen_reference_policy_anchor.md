# Explicit finite frozen-reference policy anchor

This is a new opt-in software family, not a production/YAML default or a selected
training study. Existing CORE and accepted-bank entry schemas stay strict. No
actor, reference, coefficient, dose, playing protocol or scientific budget is
selected by this implementation. Cancelled studies and intentional stops remain
unchanged. This is forward KL(reference || student), not GRPO, reverse KL, the
P1 `policy_sf` auxiliary, or SF-shape conditional KL. It does not protect WDL.

`scripts.core_generation_bank_policy_anchor` accepts the same CLI flags as the
existing bank learner: `--contract`, `--contract-sha256`, `--stage`,
`--stage-sha256`. It calls an explicit private family branch of the shared actual
executor; no injected validator, fake old manifest, or duplicate learning loop.
Contract schema is `core_generation_bank_policy_anchor_v1`, with all existing bank
contract fields plus exactly `policy_anchor`. Stage schema is
`core_generation_bank_policy_anchor_stage_v1`; its remaining fields are unchanged.
Root still authenticates actual bank root+independent pins before the public API.
Actual payload checks, generation-aware replay, journal, real Trainer, posthooks,
finite/no-retry checks, and atomic boundary publication are shared.

`policy_anchor` has exactly these fields:

```json
{
  "schema": "frozen_reference_policy_anchor_v1",
  "direction": "reference_to_student_forward_KL",
  "mask": "original_network_turn_has_policy_with_explicit_legal_mask_v1",
  "coefficient": 0,
  "reference": null
}
```

Zero requires null reference and is an exact numerical bypass: no reference file
I/O, constructor, forward, RNG capture, state check, loss addition or final hash.
The explicit disabled identity is saved only when this new opt-in is configured;
legacy Trainer defaults do not gain a checkpoint field or numerical operation.

For a strictly positive finite coefficient, reference is exactly
`{checkpoint:{path,sha256}, architecture, checkpoint_step,
historical_source_revision}`. The reference architecture must equal the complete
normalized student ModelConfig manifest. The actual immutable canonical external
file's SHA, read-only custody, embedded architecture and integer step are checked;
the real weights-only public loader constructs a complete separate model. Its
current loader/caller/math source is pinned by the contract source revision and source_files.
Historical source may be null (unknown) or a contract-declared40-hex provenance
value; the code does not infer it from an old payload or authenticate a historical
training process. There is no fabricated mandatory attestation framework.

Reference construction preserves Python, global NumPy, Torch CPU and already
initialized CUDA RNG streams. It shares neither parameter nor buffer storage with
the student; alias checks happen before loading/freezing a constructed alias.
It is eval/no-grad, frozen, outside student modules/optimizer/state, and never
serialized inside the student checkpoint. Lightweight parameter/buffer identity
and version checks plus RNG checks guard forwards; no full checkpoint hash or
full reference tensor clone/CPU digest occurs per microbatch. Full reference
identity is rechecked at save/boundary. Reference setup/extra forward costs are
real; separate diagnostics record eligible/total rows, reference-forward count,
KL row sum/mean and host interval, not a fabricated zero cost or GPU kernel timer.

The Trainer hook runs only in training, after the actual shared mirror/collate and
in-place classical-feature dropout and after the student forward. Reference gets
the eligible slice of that same `x` and `relations`; it does not sample, mirror,
drop out or query a teacher again. Eligibility is original binary network-turn ×
has-policy. Active rows require present, flagged, finite, binary, width-matched,
nonempty legal masks. No full-action-space fallback is allowed. An empty eligible
batch skips reference forward and addition; diagnostics have mean=null, rows0.
Stable F32 legal log-softmax computes one KL per eligible row and averages those
rows. Illegal terms are made safe before multiplication, avoiding0×(-inf).
The logit gradient is coefficient×(p−q)/eligible_count, with exactly zero illegal
and inactive gradients and no reference gradient. Nonfinite active inputs fail.

First supported positive scope is eager single-microbatch replacement replay;
compiled student, prefetch, residual-channel dropout and exact-epoch replay are
refused. Existing evaluation `compute_loss`, evaluation ruler closure and primary
TrainMetrics are unchanged. Primary training metrics continue to report base
policy/WDL loss; anchor diagnostics are separate and explicitly row-normalized.

Completed boundary status is
`COMPLETED_CORE_GENERATION_BANK_POLICY_ANCHOR_SEGMENT_V1`; its separate checkpoint
tag is `core_generation_bank_policy_anchor_v1`. Both bind the contract/run/source,
segment, logical generation and exact policy-anchor identity. Trainer also saves
the standalone identity under `frozen_reference_policy_anchor_v1`, with no copy of
reference weights. Adjacent resume requires the same source/contract/reference,
closed own-family owner/command/stage/history and matching saved identity; it
restores model/named optimizer/scheduler/zclip and then all five boundary RNG
streams after student/replay/reference reconstruction. Model-only first bootstrap
still starts optimizer0 independently of the actor's historical checkpoint step.

The explicit adjacent collector kind is `bank_policy_anchor`, with lineage schema
`core_generation_bank_policy_anchor_actor_lineage_v1`. Its lineage contains the
old bank fields plus policy_anchor. It requires its own closed anchored parent,
exact new CLI suffix, contract/stage/boundary/reference identity and positive
completed step. The actor is the actual student's checkpoint/path/SHA/step.
The actual loaded tag and saved anchor identity are checked before construction;
mixed old/new family tags are refused. Serving does not reopen/load the frozen
reference. Producer logical generation, parent logical generation, learner
segment/cumulative optimizer step and owner/producer/learner sources stay distinct.
Data-only auditing accepts only the matching new serving tag and still emits
unpromoted candidates; the legacy recipe-bound auditor refuses this actor kind.

New tests are synthetic source controls, not real-data or GPU evidence. They
cover analytic and actual-backward nonzero gradients, illegal/inactive masks,
reference freeze/no alias/RNG/mirror/relations, zero state equivalence, actual
public caller and full-state continuation, strict schema/refusal, actual tag
capture before an inert CPU constructor, and the real raw/Zarr/FakeTB audit path.
Fictional root/owner/bank fixtures are labelled as such. The serving capture test
does a real CPU decode with an inert constructor; it is not a CUDA/full494 serving
qualification. Numerical tests and whole-repository lint must be run by the
separately admitted root owner; source authors do not claim them passed.
