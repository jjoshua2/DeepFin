# Retained-search main-WDL finite capability

**Synthetic CPU software qualification accepted; full A39/G1/CUDA qualification
and scientific use remain UNRUN.** No bank, dose, scientific seed, learning rate or experiment is selected by
this capability. It exposes one existing target-source control through a new
strict family; it does not restore a historical multitask recipe or select a winner.

Run the separate public entry point with authenticated, read-only external JSON:

```bash
python -B -m scripts.retained_search_finite_learner \
  --contract /external/contract.json --contract-sha256 SHA256 \
  --stage /external/segment.json --stage-sha256 SHA256
```

The learning contract has the exact fields of `core_finite_learning_v1`, with
`schema: retained_search_main_wdl_learning_v1`, `updates_per_segment` replacing
`updates_per_generation`, and one required `retained_search` profile:

```json
{
  "schema": "retained_search_main_wdl_profile_v1",
  "scope": "ROOT_ADMITTED_FIXED_TRAIN_BANK",
  "search_share": 1,
  "data_identity": {"run_uuid": "ORIGINAL_PRODUCING_UUID", "source_revision": "ORIGINAL_SOURCE_SHA40"},
  "generation_manifest": {"path": "/external/accepted-generation.json", "sha256": "SHA256"}
}
```

`search_share` accepts numeric 0 or 1, rejecting booleans, intermediate values,
nonfinite values and unknown keys. `SYNTHETIC_ONLY` is a distinct qualification
scope requiring the explicit `SYNTHETIC_NOT_REAL_AUTHORITY` generation marker;
that fixture cannot pass the actual-data scope. The producing run remains distinct
from the learner run. The bank is a single original generation 1 with its original
actor, TRAIN inventories, physical game/ply identities, genuine resolved outcomes
and producing evidence. Root's accepted generation interface remains the authority;
this caller issues no data acceptance. Its first actor must match the declared
fresh donor. No HOLD, renamed cohort or on-policy collection integration is added.

Stage fields are exactly `schema`, `contract_sha256`, `segment`, `generations`,
`parent`, `output`. Schema is `retained_search_main_wdl_segment_v1`; `generations`
is the one-element list containing the profile's exact manifest reference. Each
segment reuses that fixed bank and applies its explicit contract dose. Draw journal
generation/source/row IDs remain original data IDs, never segment IDs. Existing
uniform sampling, one callback, one mirror and all-row denominators are reused.
No sidecar, wrapper, target reconstruction, extra random draw or source-array
mutation is introduced.

Share 0 uses the original constructor and unwrapped replay, with no additional
search-target validation branch. Share 1 supplies only `search_wdl_frac=1` to the
existing Trainer; main SF remains 0, policy and main WDL remain 1, all twelve
auxiliary weights remain 0, and optimizer, scheduler, clipping, architecture and
other core controls are unchanged. Existing loss normalizes the stored search
triple when its U8 availability mask is 1; unavailable labels use the unchanged
genuine integer-outcome one-hot. The positive path checks F16 `(rows,3)` storage,
U8 binary availability and finite nonnegative qualified simplex rows (0.002 storage
rounding tolerance) before admitting each shard to replay. Missing both fields or
an absent vector with an all-zero mask takes the existing fallback. A present
vector without its mask, an absent claimed vector, or invalid qualified row fails.
Unclaimed nonfinite rows retain the existing loss's explicit fallback accounting.
Left/right augmentation does not exchange W and L.

Continuation requires the exact same family, contract hash, source, profile,
manifest, consecutive segment, prior stage and complete closed PASS owner command.
External failed/pending publication markers refuse continuation. The original
strict full-state restore still checks architecture, parameter names, optimizer,
scheduler, clipping state and step. Before that restore, the new family checks the
checkpoint's `retained_search_learning_contract` metadata against the exact family,
contract SHA, run/source and profile. Changed shares and cross-family continuation
are refused; using weights as a separately declared fresh donor is a different
operation, not optimizer continuation.

After Trainer saves its own new checkpoint, this family's same owned phase loads
it with `weights_only=True` on CPU, appends only that built-in metadata, and writes
fresh staging bytes followed by atomic replacement and fsync before publication.
No original donor or prior-family checkpoint is rewritten. This packaging pass
preserves tensor values, shapes, dtypes and storage sharing, and does not alter the
live Trainer or its RNG. CUDA archive device tags become CPU tags; exact archive
bytes/device tags are not promised. Existing restore maps trained state to the
configured device. Root qualification must verify this packaging/restore path and
charge its serialization wall time; source inspection is not numerical proof.

The closed boundary uses `COMPLETED_RETAINED_SEARCH_MAIN_WDL_SEGMENT_V1`, includes
`segment`, exact profile, checkpoint identity, full state/RNG lineage and descriptive
search/outcome target masses derived from the loss availability metric over all
drawn rows. Availability can be positive at share 0; it is not proof of an active
objective or target quality. Closed producer output is not model promotion,
checkpoint adoption or strength evidence.

Source pins include the core required set plus this entry point, replay buffer,
dataset and augmentation. The focused tests, old core/SF guard regressions,
scoped static checks, native/runtime/source custody and synthetic CPU
packaging/continuation controls passed as recorded below. A real-model GPU/data
qualification and a separate prospective protocol remain required before scientific
learning admission. The authored tests exercise the real finite worker, host and
loss: CE and gradient changes, zero-state/RNG equivalence, mirror and physical-ID
alignment, outcome fallback, invalid storage, strict share/family guards and
same-family continuation. No global YAML/default,
`CORE_KWARGS`, shared loss or legacy CLI contract was changed.


## Accepted synthetic CPU qualification — November 14, 2026

At source `1b2de99f6fdb4c55c220295c9a4ae295224fe060`, scoped Ruff,
basedpyright and Vulture passed, and all **66 tests passed with no skips/errors**:
23 retained-search cases, 21 legacy core cases and 22 SF-current-P0 cases. The
actual finite worker and host loss were exercised, including analytic CE/gradient,
outcome fallback, unchanged share-zero state/RNG/journal and same-family restore.
Torch used two threads; CUDA remained uninitialized. Final source/native/plan and
admission checks completed; all owned processes closed. Independent actual review
accepted the result with no findings.

The six native images were built at original source
`67022011ecb0082de0948bce2dd5cc4cbfd30a60`. An explicit reviewed transfer retained
that build identity while authenticating the later test-only corrections and all
unchanged native inputs. No image was relabeled as newly built. One static lint
failure and a later 64-pass/two-failure fixture run remain preserved. The final
passing owner used 19.791205023997463 seconds; total CPU owner wall, including the
cold build and both failures, was **70.7656186169479 / 600 seconds**.

This is tiny synthetic CPU software evidence. It does not qualify full A39/G1,
GPU precision/backward/serialization/restore, a real learning recipe, model delivery,
playing strength or adoption. No scientific bank, dose, seed or budget is selected.
Documentation publication does not change the tested Python/native code.

Evidence uses `~/chess-artifacts/operations/rl-week-one-20261112-v1/`; bulk fixtures
and output stay outside Git.

| Artifact | SHA-256 |
| --- | --- |
| `core-retained-search-mainWDL-consumer-source-preparation-v4/manifest.json` | `23f7500cd3416013e6ca7eff8cfb6327ba063a7e3f96df19a51f6586cb882475` |
| `core-retained-search-mainWDL-consumer-independent-source-review-v4/review.json` | `e4f2c3ed50ee36e4f3f6ebf99b439a87cbe86f9c270781b4022e1fed9b46abe2` |
| `core-retained-search-mainWDL-consumer-CPU-qualification-preparation-v3/manifest.json` | `74b6db7f4b0d570e62760c2ecb979463e1b075154a149336ba0765db0d90d8ca` |
| `core-retained-search-mainWDL-consumer-CPU-qualification-independent-source-review-v3/review.json` | `04f9d750c746898521169f0aac2e65ce3c5a1fdf3ff0df358aff2f63c6ff33db` |
| `core-retained-search-mainWDL-consumer-CPU-qualification-bindings-v1/test-only-source-transfer-v3.json` | `94f243f261469c6497151b8f98f8c671ca509951df1c79870ee6846c05a40844` |
| `core-retained-search-mainWDL-consumer-CPU-qualification-bindings-v1/focused-root-acceptance-v3.json` | `7c6f43dbbc74b70133eadb84a7f0b5a53c992d9d4a2e799db0f2111c4cdc4b3e` |
| `core-retained-search-mainWDL-consumer-independent-actual-review-v1/review.json` | `4d9da02ffa1208a0fda9c291b031a29c56e200b34a5083c33b074059de509e8a` |
