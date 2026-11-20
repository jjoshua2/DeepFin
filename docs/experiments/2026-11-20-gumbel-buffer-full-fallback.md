# 2026-11-20: Gumbel leaf buffer full: root-Q fallback replaced by flush-and-resume

This is a search-correctness fix with a reach audit. No training compute was used
and no live yaml changed. The fix lands in the main `_mcts_tree.c` (ABI 10 → 11).
The pinned live checkout runs its own `.so` and is not redeployed by this merge.

## Mechanism

`run_gumbel_root_many_c` sizes the C leaf encode buffer at
`leaf_buffer_rows(n_boards, topk)`. On the in-place path it caps that size at the
evaluator's `_max_batch`; the copying path caps it at `search_leaf_row_cap` when
the evaluator sets one. In `gss_prepare_batch`, a leaf that found the buffer full
was not evaluated:

```c
if (s->n_leaves >= g->enc_capacity) {
    stored_append_terminal(s, path_buf, path_len, g->root_qs[bi], SOLVED_UNKNOWN);
    continue;
}
```

`tree_backprop(path, value)` reads `value` from the leaf's side to move, and the
root's Q is from the root's side to move. As a result:

- every starved leaf at an odd depth (a root child, a great-grandchild, ...) was
  backed up with the wrong sign: for a losing root those leaves looked as good
  as the root is bad;
- every starved leaf at an even depth got the right sign, but the value was the
  root's, not its own: fabricated.

The accumulate guard in `gss_step` stops before a rep whose 2 × queries would not
fit, and later reps are no larger than the first. So the fallback fires exactly
when one rep's leaf requests exceed the buffer: when the cap is below boards per
call × topk (per group when pipelined).

The root-only site (`path_len <= 1`) is correctly signed and is unchanged. No
other search uses this fallback: PUCT, the walker and root-parallel Gumbel do not.

**Fix.** A full buffer now flushes and resumes, exactly like the C17
duplicate-leaf flush. The query cursor (`q_resume`) is set at the leaf that did
not fit and the pending batch is returned for evaluation. Nothing had been
committed for that query: no leaf appended, no virtual loss applied. Every leaf
is therefore evaluated. Later leaves see a fresher tree than they would in one
big batch, so a binding cap changes the search the way a smaller `target_batch`
does. A buffer with zero rows is refused at `start_gumbel_sims`.

Each flush is counted:

- in C, by `MCTSTree.gumbel_buffer_full_flushes()`, per search and reset by
  `start_gumbel_sims`;
- in Python, by the process total `gumbel_c.buffer_full_flushes()`, read off the
  trees after every search, with a one-time stderr line.

ABI 11 makes a stale extension fail at import instead of absorbing leaves
silently.

**Where the count is banked:**

| Consumer | What it records | What it does when the count is > 0 |
|---|---|---|
| `simple_rl` collect | `stats.json` `buffer_full_flushes`, the delta around the generation | Refuses to seal: the generation is redone whole, because its rows were made by a different search than the pinned settings |
| Arena | Result record `buffer_full_flushes`, plus `eval_max_batch_realised` read back off the built evaluators; `eval_leaf_cap_bound` is now derived from the realised cap, not the requested one | Records the count and keeps the warning. It does not fail, because since ABI 11 a binding cap re-batches the search instead of fabricating values, the same class of change as a smaller `target_batch`, which the arena also only records |
| Production self-play and UCI | The one-time stderr line, plus `buffer_full_flushes()` | Nothing; the reach audit shows the cap cannot bind there |

## Reach audit (before the fix)

**Instrument:** a temporary per-search counter on the parent code in a worktree
build, plus CPU replays with a deterministic hash evaluator that exposes the
in-place slot API at a chosen `_max_batch`. The GPU run used the SL base. The
live tree was not touched.

| Consumer | Boards per call × topk | Buffer (cap) | Can bind? | Observed |
|---|---|---|---|---|
| Arena matched_sims (P1's deciding match, P1 gates, head-swap rounds 1–2, loop-smoke and optimizer-start arenas, search-gain) | ≤ 128 × 16 = 2,048 | 4,096 (`--eval-max-batch` default) | No | Every banked log has max_batch 4096 and no cap warning. All 35 result rows have `eval_leaf_cap_bound` false. GPU rerun at the P1 / head-swap settings (SL base vs SL base, 128 games, pool 128, seed 20261119): 430 searches, 708,381 leaf evaluations, **0 absorbed** |
| F12 / Q50 collection (16 slots, serving max_batch 512, persistent tree) | 16 × 16 = 256 | 512 | No | CPU replay: 0 of 3,071 leaves absorbed |
| rl-audit search bank (64 and 800 sims, `per_call` 16, F12 serving) | 256 | 512 | No | CPU replay at 800 sims: 0 of 37,767 |
| Production distributed self-play (broker on, 32 threads, ≤ 512 games per batch split evenly, topk 16) | ≤ 16 × 16 = 256 per thread | 512, uncapped: the slot client has no in-place API; the broker slot max is 1190 | No | CPU replay: 0 |
| UCI classic Gumbel (1 board) | topk (≤ 256) | 512 floor, min'd with MaxBatch (default 1024) | Only if MaxBatch < topk | MaxBatch 16 at topk 32: 10% absorbed (not a shipped setting) |
| UCI root-parallel Gumbel, PUCT / walker | — | — | No such path | — |
| `audit_targets.py`, `search_gain_probe.py` (LocalModelEvaluator) | — | Uncapped (copying path) | No | — |
| `value_regret.py` | — | No search | — | — |
| **`simple_rl` loop smoke collect** (`--eval-max-batch` 512, concurrency 128 / 256 / 64) | 128 × 16 = 2,048 | 512 | **Yes** | CPU replay at 128 boards: **30.6% of leaf visits absorbed** (7,483 of 24,475) |

**Contaminated:** only the simple-RL loop smoke.

- `loop-smoke-gpu/run`, generations 1–3, and `run_priormix`, generation 1, under
  the operations bank `rl-claude-20261119/`. Both directories now carry
  `CORRUPTED_TARGETS.txt`.
- The smoke's probe drift (|ΔQ| 0.30) and its reply-side target damage were
  traced to this on 2026-11-20; see `docs/simple_rl_loop.md`, "Evaluator batch
  cap".
- PR #1427 makes the collect default to an uncapped evaluator and makes train
  refuse rows (and descendants of rows) from capped collects.

**Not contaminated, no rerun needed:**

- P1's +42.6 Elo verdict and its gates;
- head-swap rounds 1 and 2;
- the F12 / Q50 collections and everything trained on them;
- the rl-audit 64- and 800-sim banks, including the D3 800-sim bank;
- production self-play;
- every arena run at the default `--eval-max-batch`.

In each case the cap could not bind, so the parent's searches were identical to
uncapped ones.

## Measured footprint

`scripts/bench_selfplay_identity.py` with the SL base: F12 protocol configs, 128
games, 128 slots, 64 sims, Syzygy pair.

| Build | max_batch | Digest | Leaf rows evaluated | Plies | Leaf rows per ply | Forwards | Flushes |
|---|---|---|---|---|---|---|---|
| parent | 4096 | `8d377744e79a73e3` | 831,658 | 14,139 | 58.8 | 2,443 | — |
| fix | 4096 | `8d377744e79a73e3` | 831,658 | 14,139 | 58.8 | 2,443 | 0 |
| parent | 512 | `d14368767a848d38` | 408,932 | 9,613 | 42.5 | 2,225 | — (absorbing) |
| fix | 512 | `e4551f0f2b70cdfd` | 799,794 | 13,822 | 57.9 | 3,598 | 408 |

At cap 512 the parent evaluated 28% fewer leaves per ply. Its games were also a
third shorter (9,613 vs 14,139 plies), because corrupted searches play decisive,
lopsided games. The fix at cap 512 evaluates 98% of the uncapped per-ply leaves,
at the cost of 47% more forward calls.

## Identity when the cap never binds

The branch is not taken, so outputs are byte-identical to the parent. Identity is
established by the deterministic instruments:

- the reviewer's CPU fuzz: 150 parent-vs-head cases compared node for node, all
  identical, 0 flushes;
- `bench_arena_identity.py`, 64 pairs, pool 128, 64 sims, max_batch 4096: digest
  `45bdc247…20adaa` on both builds, 0 flushes (the reviewer's repeats agree);
- `tests/test_uci_search_identity_pin.py` passes;
- `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke` passes.

`bench_selfplay_identity.py` at max_batch 4096 also gave equal digests
(`8d377744e79a73e3`) on one run per build. That bench is not deterministic run to
run, though: both builds produce the same two digests across repeats, and
`sf_block_starved` varies, which suggests Stockfish-pool timing. A single equal
pair is therefore consistent with identity but does not prove it. The footprint
rows above are single runs. The bench's nondeterminism is a follow-up and is not
fixed here.

## Tests and mutants

`tests/test_gumbel_buffer_full_flush.py` drives the real C search on CPU, with
the cap imposed through `search_leaf_row_cap`. It covers:

- a lost root (Q ≈ −1) with draw-valued leaves: every visited root child must
  back up exactly 0;
- the leaf count matches an uncapped run: at least 90% of the uncapped leaves,
  no batch above the cap, and 64 root visits in both runs;
- the counter: 0 when nothing binds, and the per-tree count equals the process
  delta;
- an empty buffer is refused;
- the arena record and the realised-cap bound;
- the collect's record-and-refuse.
- no node carries virtual loss after a capped search.

Mutants:
- Parent semantics restored in C (absorb with the root's Q, counter kept): the
  sign test and the leaf-count test fail.
- Counter never incremented in C: 3 tests fail.
- Virtual loss applied at the flush and never undone: the virtual-loss test fails.
- Six Python-side mutants: all killed. They cover:
  - banking removed, on the single path or entirely;
  - the arena observation zeroed;
  - the arena bound taken from the requested cap;
  - the collect sealing despite flushes;
  - the collect recording the process total instead of the delta.

## Follow-ups

- `scripts/bench_selfplay_identity.py` is nondeterministic run to run (two
  digests per build across repeats; Stockfish-pool timing suspected), so it cannot
  certify identity from one run per build.

- Results from capped runs on builds before ABI 11 are suspect wherever
  `eval_leaf_cap_bound` was true or a collect cap sat below boards × topk. The
  audit above found only the smoke.
- #1427 (merged) defaults the collect cap to the search's own leaf buffer and
  makes train refuse rows from capped collects, so a default collect never
  flushes; an explicit smaller cap is refused before pinning.
