# Incremental common-input qualification contract

Prospective metadata only. No raw payload was scanned, no new teacher inference ran,
and no output corpus or runtime was created. The pinned selections contain exactly
w00-00032..00063 from each original source: 532,389 physical rows. The completed
v3 prefix and all its files stay unchanged. Source-qualified game IDs in the new
closed-progress entries have zero intersection with the v3 prefix for each source.

## Selection and derivation

Use the shared `--source-shards` schema agreed with the implementation author:
`schema:1`, resolved original `source_dir`, `source_config_sha256`,
`source_manifest_sha256`, and `shards:[{source_shard,rows,source_sha256}]`.
Each list is a validated subset of the genuine closed inventory; retain its
canonical order, original paths and source namespace. `--limit` is that selected
source's complete row total, not a prefix of the full live corpus. Both derivation
and rank must use the same manifest/hash. No alias corpus or substituted manifest.

The source generator may append unselected shards/progress records during the run.
That is permitted. Pin the immutable source launch configuration and selected raw
identities, not the live whole-progress-file bytes. Selection admission performs
its ordinary selected-payload verification. Do not add the old v3 `source_check`
whole-payload pass on top of that merely because v3 had it. Preserve selected
storage identity checks across the stages and final publication.

Keep phase0 complete d9 policy, latest-phase d9 search value, q temperature0.0005,
floor0, full history/provenance, seed0 and 8192 output rows/shard. Emit actual
support exclusions under the existing bounded opt-in; no preliminary census.
Per-source ceilings: 64 support exclusions; no-result drops at most floor(rows/50)
(5329 run06,5317 run07); zero envelope exclusions. Other malformed observations
remain fatal. These are admission ceilings, not expected counts or a license to
relax observation semantics. Latest-phase values remain unchanged; the old
64-shard value diagnostic does not certify the unseen subset's observations.

## Survivor proof, without another raw census

Let U be the registered source-qualified physical row universe, S the support
exclusion set and N the missing-result set. The v3 operational wrapper hardcodes
prior census files and exact observed counts; that is not a reusable tool
requirement. Replace only that pilot-specific expectation in a new bounded wrapper.

1. Preserve the emitted derivation summary and actual support JSONL. Require exact
   inline/JSONL equality, original source/config/shard membership, unique in-range
   physical references and the registered cap. The existing rank consumer rereads
   selected raw rows once and independently checks each support defect's full-width
   metadata, ranks, finite scores, legal missing/duplicate roster and history key.
   It must encounter every S reference and accept no derived reference to S.
2. Rank already counts missing results from raw observations independently of the
   deriver and compares that count to the derivation counter. It constructs valid
   raw identity records only for result-bearing, unexcluded rows. Each emitted
   reference must join a valid record with exact game/ply/worker/original-history/
   quantized-history identity. A disk bitmap rejects duplicates across output
   shards. This supplies an injective map of emitted rows into eligible U, rather
   than a comparison of counts alone.
3. Require zero envelope drops and emitted count = |U| - |S| - |N|, with independent
   rank and derivative accounting agreeing and both drop caps respected. Alongside
   the injective eligibility proof, cardinality establishes the complete survivor
   complement: no eligible row is silently lost and no excluded row survives.
   Keep per-source populations separate. The existing qualifier's emitted-row
   bitmap can optionally export omitted physical IDs; U minus emitted refs already
   makes them reconstructible without another raw pass.
4. Reuse ordinary adapter/rank source-bound admission, per-output correspondence,
   policy/legal normalization, original/quantized input-key alignment, and unchanged
   derived storage proof. Retain the compact actual summaries and source-selection
   hashes. Do not repeat x/raw corpus validation merely to recreate a previous
   proof. Fully qualified output is shared input for later recipes, not a training
   or game-strength result.

The result-present support-exclusion guard is already present on current main
91ef31f54 (introduced by0fad0721b). The initial inspection used the older frozen
v3 runtime, which lacked it. The newly qualified runtime must retain this existing
guard, so rank independently enforces the disjoint missing-result/support sets.
No additional policy implementation is required.

## Cost, integration and remaining blockers

The comparable v3 batch took3396.99s, with maximum reported per-stage RSS1279244KiB;
it passed8GiB output/cache and150GiB free-space guards but did not bank actual peak
disk consumption. Roughly one hour is a planning reference for this similarly
sized increment, not a throughput guarantee. Cap the new run at7200s including
termination, two CPUs2–3, numeric threads2, nice19/ionice3, GPUhidden,8GiB
output/cache and150GiB reserve. Both per-source identity-cache ceilings remain64MiB.
Original selected raw compressed files total1,357,158,582bytes and are retained,
not recopied. No extra target recipes are materialized by this common preparation.

Source-preserving selection CLI implementation/review and a separately frozen
bounded launch wrapper are pending. The old v3 runner cannot be reused unchanged:
it selects a32-shard prefix, requires old census identities and pins the old CPUs.
Keep each source/increment in distinct physical output directories; existing
training loaders namespace resolved parents. Preserve source-qualified provenance
and the demonstrated nonoverlapping games rather than flattening directories.
No new preparation is queued by these records, and this is not a mandatory next
experiment. Parent owns resource admission and the actual launch decision.
