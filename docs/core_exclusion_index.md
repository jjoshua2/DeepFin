# Bounded prior-exposure metadata

`core_finite_collect.exclusion_inputs` keeps the existing flat `{path,sha256}`
route unchanged, including its existing32MiB reader and collision decisions.
TRAIN/HOLD schemas, actor selection, raw/history/outcome audits, candidate
statuses and learning authority are unchanged. No global metadata cap is raised.

The explicit opt-in `exclusions.prior` envelope is:

```json
{"schema":"core_exclusion_index_v1","manifest":{"path":"/absolute/readonly/index.json","sha256":"..."}}
```

The manifest has exactly `schema,chunks,unique_counts,serialized_chunk_bytes,
coverage_scope,donor_pretraining_exposure`. Its schema is
`core_exclusion_index_v1`; coverage is a nonempty explicit scope and pretraining
is literally `UNKNOWN`. `unique_counts` has exactly nonnegative Python integers
for `history_sha256` and `stored_input_float16_sha256`.

Each ordered chunk descriptor has exactly `kind,path,sha256,bytes,count`.
Kinds are `history_sha256` first, then `stored_input_float16_sha256`.
Each chunk object has exactly `schema,kind,values`, schema
`core_exclusion_chunk_v1`, matching kind and a nonempty list of lowercase64hex
identities. Values must be strictly increasing within each kind across chunks;
no duplicate identity is accepted anywhere, including across the two domains.
Nothing is deduplicated, truncated, replaced or recursively expanded. The
explicit empty manifest is allowed for honestly declared empty software fixtures;
it is not an assertion of universal unseen exposure.

Limits are4MiB for the manifest,32MiB per chunk,128 descriptors and512MiB for
the **sum of authenticated serialized chunk file bytes**. These are not a
Python-object deep-size bound, aggregate I/O bound or continuous RSS promise.
All declared counts/bytes are checked before chunks are opened; each actual
canonical external readonly file is size-capped before the genuine existing
collector hash/duplicate-key/descriptor reader is used. Actual sizes, hashes,
non-atime stamps, source order and unique totals must match. Every bounded
input is rechecked before any complete flat membership view is returned.
Missing, mutable, linked, noncanonical, inconsistent or excessive inputs fail;
there is no partial result or fallback to flat metadata. Root-owned phases
retain their separate resource limits; this source makes no fit guarantee.

The result has the existing `history_sha256,stored_input_float16_sha256,
coverage_scope,donor_pretraining_exposure` membership fields. Existing report vs
refuse policy and playing-start collision logic are unchanged. New study
protocols must bind the selected refuse policy explicitly. The collector's CLI
precheck and direct execute path authenticate all chunks before actor/engine
work. No file read creates a collection or execution admission.

## Source closure

When the new envelope is used, producer source pins must include
`scripts/core_exclusion_index.py`. The current helper digest is checked before
import; current source/origin guards still apply. Data-only TRAIN/HOLD and the
legacy recipe-bound auditor additionally require the same producer-helper pin
in their contract/spec source set. The audit still repeats the genuine collision
checks against the unchanged producer route.

The metadata-only bank reader checks the historical producer/spec helper-pin
join and binds the historical manifest without reopening any index/chunk.
Its readonly evidence adds `exclusion_index:{manifest,source_sha256}` only for
this explicit route. Bank learning, including the anchored public family,
requires its **current** helper source pin when that evidence is present; it
never compares historical helper bytes to the later consumer checkout. Legacy
flat evidence and identity schemas/statuses stay unchanged. The helper is not
an accepted-bank token, parent proof, protected-role selector or learning waiver.

Source qualification uses tiny fictional metadata, explicit private smaller
limits, genuine `checked`, actual exclusion decisions and public source guards.
No real exposure index or bank has been read/audited by the source author.
