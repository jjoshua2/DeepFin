# B100/H20 recipe-screen readout

`scripts/bt4_recipe_readout.py` certifies completed evidence from the separately
registered B100-candidate versus H20-reference development screen. It does not
launch matches or select training. The historical `bt4_joint_readout.read_arm`
continues to require its original fixed-size banks and reject SPRT.

The new reader has two explicit modes:

- `low_sprt`: 100 simulations, at most 1,000 games / 500 canonical opening pairs;
  logistic-Elo GSPRT 0 versus +15, alpha .05, beta .10, first look at 128 pairs,
  subsequent looks every 64 pairs, including the final cap.
- `high_fixed128`: 400 simulations, exactly 256 games / 128 pairs, using the same
  first 128 canonical openings. It requires a pinned low manifest and recertifies
  that low cell before reporting the fixed-core cross-budget contrast.

A valid low cell needs all first 128 pairs. Boundary crossings can end it early;
uncrossed cap or normal deadline endings are valid `INCONCLUSIVE` results. Process
failure, a torn bank, missing core, incorrect mapping or inconsistent terminal
accounting is `OPERATIONALLY_INVALID`, not an unfavorable recipe verdict. This
profile permits no resume or duplicate finished half. Final rolling orphan halves
and speculative completed pairs are retained and checked separately.

## Pinned input interface

```bash
PYTHONPATH=. python scripts/bt4_recipe_readout.py \
  --manifest /absolute/completed-cell.json --expected-manifest-sha256 SHA256
```

Each pin is `{ "path": "/absolute/file", "sha256": "..." }`. The manifest contains:

| Field | Meaning |
| --- | --- |
| `schema` | `1` |
| `mode` | `low_sprt` or `high_fixed128` |
| `bank`, `result` | Finished-game JSONL and exactly one terminal arena-result JSONL |
| `process` | Successful existing owned-stage receipt: `process_complete:true`, `exit_code:0`, exact `command`, `hard_seconds`, `started_unix`, `ended_unix`, `stage_seconds`, `gpu_seconds` |
| `launch` | Pinned input/command proof described below |
| `opening_panel` | Exactly 500 ordered `{root_fen,moves,fen}` objects, with 16 legal UCI history moves each |
| `expected_settings` | Entire expected arena game-log settings dictionary, including both full realized search records |
| `expected_execution` | `loop` (`rolling`/`chunked`), `compile:"on"`, string `eval_hoist`, integer `eval_max_batch`, `eval_leaf_cap_uncapped`, `max_concurrent_games`, `arena_pool_size`, `max_seconds`, `hard_seconds` |
| `low_manifest` | High mode only: pinned low-mode input manifest; no recursive high manifests |

The launch proof has `settings`, `execution` and `opening_panel` equal to the
manifest, `candidate_role:"B100"`, `reference_role:"H20"`, and the exact stage
`command`. Its `identities` maps `candidate`, `reference`, `book`, `runtime`, and
`preregistration` to path/SHA pins; `runtime` additionally carries the producing
`git_sha`. Candidate/reference/book paths match the header. Runtime and
preregistration refer to compact receipts/documents. Checkpoint/book payload
hashes were checked by the launch producer; this reader does not reread those
large files or repeat training/dataset qualification.

Commands must explicitly provide the registered candidates, book, games,
simulations, seed42, opening-plies16, max-plies300, move temperature .1, training
search shape, both `policy_temp=1.0` overrides, compile on, evaluator/concurrency
limits, deadline and exact output paths. `--no-rolling` selects chunked execution;
its absence means rolling, matching the actual arena CLI. The low command must
contain the exact declared SPRT spec. Resume, conflicting per-side search overrides
and disabling Gumbel noise are refused. The supported CLI is parsed without
abbreviations; unknown options and duplicate options (including `--flag=value`)
are refused. Device must remain CUDA. Terminal argv,
header settings, execution tags, runtime commit and pinned process command agree.

Process wall and stage charges must be finite, positive and within the registered
hard cap. GPU charge equals the owned stage charge; wall and stage elapsed times
must agree within one second for receipt bookkeeping. An uncrossed deadline stop
also requires stage elapsed time to reach `max_seconds`. Arena `duration_s` starts
after loading, so it cannot alone prove that earlier deadline was reached.

Use the actual post-load uncapped leaf requirement and realized pool size in the
execution proof; this reader checks those registered values against the result,
not a new model load or guessed capacity. Preserve JSON number representation
between expected settings, launch proof and result (for example, deadline seconds
are floats in the producing arena result).

## What is reconstructed

Every banked game is checked for canonical pair/half, candidate color, opening
endpoint, score/result, seed, loop and compile/evaluator tags. Duplicate halves
are refused. The monitor rebuilds every declared look from all complete banked
pairs in canonical order, reproducing the **first** crossing even when a delayed
pair released several looks together. Terminal spec, trajectory, prefix,
pentanomial counts, rounded score/Elo, speculative identities and caps must match.
Computed `llr`, `llr_first` and trajectory likelihoods allow an absolute difference
of at most `1e-12`, with no relative tolerance. Python 3.10 and 3.13 can differ
by a few ulps because their floating-point `sum` implementations differ. Values
must be finite JSON floats and remain on the same side of both inclusive stopping
boundaries; even an adjacent-float change that alters a crossing is rejected.
Trajectory look indices, verdicts, counts, IDs and all protocol fields remain
exact. `llr_first` records the regularized initial sample, not the first scheduled
look. This compatibility check changes neither producer arithmetic nor sampling.
Finished halves, unique inflight IDs and the unstarted complement account for the
whole schedule. Actual inflight membership and repeated consultation count remain
producer telemetry; the final bank can validate consistency, not observe missing
runtime events.

The reader replays each panel's root/history to its endpoint and matches the
banked endpoints. Game logs do not contain the full initial move stack: actual
history consumption still depends on qualified frozen launcher/input-generation
proof. High and low must share the identical pinned panel, model/content identities
and header protocol, apart from game count and simulation budget.

Stopped low Elo/ordinary intervals are descriptive and selection-biased. H1 is
not a +15 lower confidence bound; H0 is not equivalence. Fixed first128-core
summaries and the paired 400-minus-100 score contrast retain their ordinary
fixed-N interpretation. The exploratory cross-budget interval uses the registered
aligned-pair percentile bootstrap: 10,000 PCG64 resamples, seed 20260903,
preserving low/high covariance within each of the 128 pairs. Neither mode establishes an optimal recipe, independent
confirmation or automatic promotion.
