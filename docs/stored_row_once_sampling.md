# One pass over stored training rows

The finite entry point `scripts.core_generation_bank_stored_row_once` consumes a fixed population of accepted bank rows once, without replacement. The row identity is `(generation, source_sha256, row_index)` before augmentation. An original row and its mirrored view share one reservation; naturally repeated board or opening contents in different rows stay distinct. This is a stored-row guarantee, not a claim of unique chess positions.

The ordinary and policy-anchor contract schemas are `core_generation_bank_stored_row_once_v1` and `core_generation_bank_policy_anchor_stored_row_once_v1`. Their stage schema is `core_generation_bank_stored_row_once_stage_v1`. They reuse the existing [finite bank learner](core_generation_bank_learner.md) architecture, objective, optimizer, LR and bank validators, with a separate sampling contract and explicit closed-family consumer. The historical entry points and exact-epoch loss behavior are preserved.

## Sampling contract and dose

The additional `sampling` object has these exact fields. The following values illustrate the shape; they are not a bound recipe or a selected training bank:

```json
{
  "schema": "STORED_ROW_EXACT_ONCE_PASS_V1",
  "order_seed": 7,
  "expected_rows": 17568,
  "presentations_per_row": 1,
  "mirror_shares_quota": true,
  "partial_batch": "REFUSE_BEFORE_MODEL_CONSTRUCTION",
  "lineage_uuid": "<same UUID as run_uuid>",
  "quota_ledger": "/external/lineage/reservations.jsonl"
}
```

The complete selected population stays fixed across closed segments. A deterministic SHA256-key ordering is generated once; the saved state carries the permutation, population identity, cursor and ordered-row prefix. It does not consume the shared mirroring RNG. Sampling is serialized, with no prefetch overshoot, discarded nonfinite update or automatic forward retry. The separate quota marker preserves existing loss scaling and policy-anchor behavior.

Only full batches are supported in this initial family. Reject `N % batch_size != 0` before model construction; never drop, pad, repeat or wrap rows to fill a batch. For an illustrative 17,568-row population and batch size 32, the full pass is 549 optimizer updates and 17,568 views. An exhausted population refuses another draw. A complete pass uses each stored row once, within the user's upper limit of two total deliberate presentations; it grants no second pass.

## Durable reservation and continuation

The contract fixes one canonical external quota-ledger path outside every segment output. Segment one creates it exclusively; each segment holds its exclusive lock. Before a local journal, mirroring or forward call, each batch reservation is appended and fsynced to this ledger. Output publication cannot seal the ledger read-only because ledger-inside-output placement is refused upfront.

Checkpoints retain the existing complete model, optimizer, scheduler, clipper and five-RNG handoff plus quota state and the immutable journal chain. A continuation must match the same contract, full checkpoint, population, permutation, cursor, journal prefix and entire persistent ledger. Changed seeds or fresh output directories cannot hide a failed sibling tail. Stale checkpoints, journal gaps or tails, tainted consumption, concurrent owners and failed-output reuse refuse. Do not reset or truncate a failed lineage and call it a continuation.

The closed-family consumer `validate_closed_segment(..., require_complete=True)` validates this family and its bounded actual journal chain. It is not the payload-free D2 metadata consumer, and it does not turn legacy checkpoints or proofs into once-sampled endpoints.

## Use and scope

```bash
python -m scripts.core_generation_bank_stored_row_once \
  --contract CONTRACT.json --contract-sha256 CONTRACT_SHA \
  --stage STAGE.json --stage-sha256 STAGE_SHA
```

Publication supplies software preparation only. It starts no training, selects no scientific recipe, opens no holdout and updates no live runtime. One contract-fixed lineage is enforced by this code. Historical or cross-lineage presentations elsewhere are not certified: future admission must bind controlled prior-usage accounting and reject a new-UUID/new-ledger quota reset. Existing D1/D2 overuse remains recorded; an old bank is not declared unseen. The training hold remains until the selected runtime, sampler manifest and cap accounting are verified.
