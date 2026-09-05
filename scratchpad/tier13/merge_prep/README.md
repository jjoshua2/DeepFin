# Prepared resolution for `configs/pbt2_small.yaml` (Tier-13 pre-arena merge)

Valid ONLY for this exact pair. Re-do the resolution if either side moves.

    live  ops/live-20260725 = c082808fbc2aebb3f58c5cae7606f076ed280c01
    main  origin/main       = 2c700dd703f2ca10f66e4eaf540b010af614cdb8
    resolved sha256         = b588d276f0626dc70b462aa5d54b7081868ad6907bae4d2b560e6ef491211acc

## What was resolved, and why neither side alone is correct

Hunk 1 — `diff_focus_norm_enabled`. Live has `true` (DEPLOYED, tasks #171/#173);
main has the PR's default-off `false`. **Kept live's `true`** plus its #171/#173
rationale, then appended main's comment describing which keys the distributed
worker actually reads — the two comments describe DIFFERENT key groups (live's
covers the five below, main's the five above and the six below), so both are kept.

Hunk 2 — `diff_focus_norm_shared`. New key from #408, absent on live. **Took
main's key and comment**, and kept live's "PIN (both)" comment, which documents
`categorical_bins`/`hlgauss_sigma` and which main never had.

## Verification (run in the worktree, against MAIN's schema)

    flatten_run_config_defaults  -> 318 keys, OK
    TrialConfig.from_dict        -> OK

Flattened diff against the LIVE yaml:

    added   1  (diff_focus_norm_shared = False)
    removed 0
    changed 0

Zero changed keys is the proof the resolution reverts nothing. Any other result
means the resolution is wrong — re-check before writing it live.

## Not yet done

Writing this file into the live tree, the merge commit itself, and
`python3 scripts/build_production_extensions.py` (MANDATORY: #409 raises
_REQUIRED_MCTS_ABI 3 -> 4, so every arena and training start dies at import until
the extension is rebuilt). All three are gated on arm C reaching iteration 100.
