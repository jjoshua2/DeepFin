# Explicit checkpoint/prior package readout

`python scripts/bt4_package_readout.py --contract FILE` reads one fixed paired
match between two distinct checkpoints with independently specified search-prior
temperatures. For example, a pure-BT4 T=1 checkpoint at search prior 0.5 can be
compared with a T=0.5 checkpoint at search prior 1.0. This is a package comparison;
it does not change the same-checkpoint calibration or equal-search recipe readers.
The existing arena runs the match; this command launches nothing.

The JSON contract has exactly these fields:

- `schema: 1`, `profile: explicit_checkpoint_prior_packages`.
- `candidate` and `reference`: `{role, path, sha256}`. Roles, resolved checkpoint
  paths and content hashes must be distinct. Both files are streamed through
  SHA256 now and their metadata must remain stable through reading.
- Positive finite `candidate_prior_temperature`, `reference_prior_temperature`;
  positive integer `sims`, integer `pairs` of at least two, and integer `seed`.
- `settings`: the complete actual `arena_game_log_settings` header from preparation.
  It must describe matched simulations, a book with 16-ply openings, 300 maximum
  plies, move temperature 0.1, noise on, training search shape, no candidate
  volatility, no UCI or tablebase opponent. Per-side search settings may differ
  only in their declared prior and descriptive `source` field.
- `execution`: `{loop, compile, eval_max_batch, max_concurrent_games, max_seconds}`.
  Loop is `rolling` or `chunked`, compile is `on`, sizes are positive integers and
  the arena deadline is positive and finite.
- `opening_panel`, `bank`, `process`: `{path, sha256}` pins. The panel has exactly
  `pairs` entries `{root_fen, moves, fen}`, each with 16 legal history moves and
  unique nonterminal endpoints. A forced single legal move is valid.
- Absolute `results_path`: the arena command's results output path. The per-game
  bank is the scoring source; a claimed result total cannot replace its games.

The process record must report `exit_code: 0`, `process_complete: true` and the
actual `command` argument vector. If `arena_cmdline` is present it must match.
The supported command is a direct interpreter plus `arena_standard.py` and explicit
candidate/reference, games, matched-simulation mode, sims, seed, book, opening
plies, maximum plies, move temperature, training search shape, compilation,
`--syzygy-max-pieces 0`, output paths, evaluator size, concurrency and deadline.
`--cand-gumbel` and `--ref-gumbel` may contain only the respective `policy_temp`.
The strict non-abbreviating parser refuses duplicates, unsupported overrides,
resume and SPRT. Optional device must be CUDA; label, report interval, compile
cache and PGN output are permitted. `--no-rolling` must agree with the declared loop.

Reading requires exactly both halves of every canonical pair, correct colors,
seed, endpoint order, realized loop/compile/evaluator tags, and scores matching
finished results. Torn tails, duplicate halves, orphans and incomplete fixed-N
banks are refused even after process exit zero. The report retains all paired
scores, pentanomial counts and the existing nominal paired Elo interval.

This verifies current checkpoint contents, legal panel histories and the supplied
recorded process/settings/command. External launch evidence must still establish
runtime provenance, checkpoint/book bytes and actual full-history consumption at
launch. Roles do not prove training lineage or recipe qualification. A selected
package contrast does not establish optimal temperatures, seed variability or
asymptotic target quality. Use separate development and confirmation allocations;
this reader does not choose or promote a package.
