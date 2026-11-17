# One pass over stored training rows

The finite entry point `scripts.core_generation_bank_stored_row_once` consumes a fixed population of accepted bank rows once, without replacement. The row identity is `(generation, source_sha256, row_index)` before augmentation. An original row and its mirrored view share one reservation; naturally repeated board or opening contents in different rows stay distinct. This is a stored-row guarantee, not a claim of unique chess positions.

The ordinary and policy-anchor contract schemas are `core_generation_bank_stored_row_once_v1` and `core_generation_bank_policy_anchor_stored_row_once_v1`. Their stage schema is `core_generation_bank_stored_row_once_stage_v1`. They reuse the existing [finite bank learner](core_generation_bank_learner.md) architecture, objective, optimizer, LR and bank validators, with a separate sampling contract and explicit closed-family consumer. The historical entry points and exact-epoch loss behavior are preserved.

## Sampling contract and dose

The additional `sampling` object has these exact fields. The following values illustrate the shape; they are not a bound recipe or a selected training bank:

```json
{
  "schema": "STORED_ROW_EXACT_ONCE_PASS_V2",
  "order_seed": 7,
  "expected_rows": 17568,
  "presentations_per_row": 1,
  "mirror_shares_quota": true,
  "partial_batch": "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING",
  "order": "python_random_mt19937_shuffle_v1",
  "permutation_sha256": "<SHA256 of the selected ordered permutation>",
  "lineage_uuid": "<same UUID as run_uuid>",
  "quota_ledger": "/external/lineage/reservations.jsonl"
}
```

The complete selected population stays fixed across closed segments. An isolated `random.Random(order_seed).shuffle` creates a uniform random ordering without replacement and consumes no mirroring RNG. The contract pins the versioned order identifier and the canonical permutation SHA256; `permutation_for(N, seed)` and `digest(...)` in the sampler implement that definition. Checkpoint state `STORED_ROW_EXACT_ONCE_CURSOR_V2` retains the complete permutation, digest, population, cursor and ordered-row prefix. A different runtime permutation refuses. There is no priority weighting, difficulty selection or board deduplication. The strict V2 sampling/state schemas reject earlier hash-order/refusal contracts and states; the outer finite-family names stay unchanged.

For population N and nominal batch size B, emit N//B full batches and one real final N%B-row batch when needed: ceil(N/B) optimizer updates and exactly N row presentations. Never drop, pad, repeat or wrap a row. For an illustrative 17,568-row population and B32, this remains 549 full updates. An exhausted population refuses another draw, including phantom empty updates. This pass uses each stored row once within the user's upper limit of two total deliberate presentations; it grants no second pass.

The final partial-batch objective, including any policy-anchor penalty, is multiplied by actual_rows/B before backpropagation. This retains the nominal 1/B coefficient per returned row. It is one real adaptive optimizer, scheduler and weight-decay step; no equivalence to padding or a different batch recipe is claimed. Base loss metrics are weighted by actual rows and continue to exclude the anchor penalty from logged base TOTAL. The separate exact-epoch masked objective and its anchor refusal stay unchanged. Sampling remains serialized, with no prefetch overshoot, discarded nonfinite update or automatic forward retry.

## Durable reservation and continuation

The contract fixes one canonical external quota-ledger path outside every segment output. Segment one creates it exclusively; each segment holds its exclusive lock. Before a local journal, mirroring or forward call, each batch reservation is appended and fsynced to this ledger. Output publication cannot seal the ledger read-only because ledger-inside-output placement is refused upfront.

Checkpoints retain the existing complete model, optimizer, scheduler, clipper and five-RNG handoff plus quota state and the immutable journal chain. Cursor equals min(completed_batches*B,N), completed batches cannot exceed ceil(N/B), and last_batch_rows records the real tail size. Journal and ledger checks retain actual row counts before and after the tail; an interrupted tail cannot be replayed from an older checkpoint. A continuation must match the same contract, full checkpoint, population, permutation, cursor, journal prefix and entire persistent ledger. Changed seeds or fresh output directories cannot hide a failed sibling tail. Stale checkpoints, journal gaps or tails, tainted consumption, concurrent owners and failed-output reuse refuse. Do not reset or truncate a failed lineage and call it a continuation.

The closed-family consumer `validate_closed_segment(..., require_complete=True)` validates this family and its bounded actual journal chain. It is not the payload-free D2 metadata consumer, and it does not turn legacy checkpoints or proofs into once-sampled endpoints.

## Use and scope

```bash
python -m scripts.core_generation_bank_stored_row_once \
  --contract CONTRACT.json --contract-sha256 CONTRACT_SHA \
  --stage STAGE.json --stage-sha256 STAGE_SHA
```

This family still admits its fixed consecutive bank roster through the existing bank validators; it does not implement the larger branched historical union or bypass the four-generation loader. That union needs a separate fixed-population adapter.

Publication supplies software preparation only. It starts no training, selects no scientific recipe, opens no holdout and updates no live runtime. One contract-fixed lineage is enforced by this code. Historical or cross-lineage presentations elsewhere are not certified. Future admission must bind the actual student lineage and reject a new-UUID/new-ledger reset when continuing trained state. Independent fresh model-only A39 students are distinct lineages; historical use of a bank in other experiments does not impose a worldwide unused-data requirement. Existing D1/D2 overuse remains recorded; an old bank is not declared unseen. The training hold remains until the selected runtime, sampler manifest and cap accounting are verified.
