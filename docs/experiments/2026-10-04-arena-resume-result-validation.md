# Generic arena resume result validation

A syntactically valid game log could contain a complete pair with an unknown
result token. On main `0386c2c2986385eb7109fb5f0a7072870b5076bc`, the production
`run_arena(..., resume=True)` path called `load_arena_resume`, which converted any
nonwinning, nondraw string to a candidate loss. A declared `score_candidate`
conflicting with the valid PGN result was ignored. This changes pair bins and
reported score/Elo, and can affect a sequential decision, without playing more games.
It requires corrupted or inconsistent local input; this is a correctness fix.

The fix rejects unknown or nonstring results and checks any present candidate
score against the PGN result after color conversion. Legacy rows without a declared
score remain accepted. Generic `*` as draw compatibility remains deliberate;
strict panel/history consumers reject invalid results and censoring independently.
This reproduction does not establish a frozen DeepFin arena bypass. The active
frozen runtime and its artifact pins are unchanged and do not adopt this patch.

## Validation and review

The regression uses the actual `run_arena` resume entry point with a completed
four-pair log and fails before dispatch or log mutation. On the unmodified base,
eight corruption cases (both colors; malformed result, conflicting numeric score,
boolean score and string score) fail the expected-refusal test; both valid legacy
cases pass. With validation, all 192 tests across `test_match_resume.py`,
`test_arena_standard.py` and `test_arena_sprt.py` pass on CPU with two Torch threads.
Independent Sol source/test review reports no actionable finding. Full repository
Ruff/basedpyright/vulture validation passes with zero findings.

Open-fix inspection on 2026-10-04 found no equivalent resume-result fix. PR #1001
owns paired uncertainty/degenerate statistics and does not modify this loader.
The change is submitted as [draft PR #1020](https://github.com/jjoshua2/DeepFin/pull/1020) against main; no merge, deployment,
GPU work or scientific-strength claim is made.
