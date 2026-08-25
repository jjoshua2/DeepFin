# Native NNUE value head

A big-net Stockfish-NNUE evaluator written in our own C, behind a pluggable
eval seam in the C Gumbel tree. **No Stockfish code is imported, linked or
copied.** The Stockfish sources were read as a format and algorithm reference;
the implementation is ours, and the thing that proves it is right is Stockfish
itself, not a second implementation of ours.

## Why native, and why inside the tree

Measured during scoping (`scratchpad/native_gen_scoping/`): the existing
UCI-driven generator's cost decomposes as generator-side Python 46.3%,
python-chess driver 34.6%, Stockfish compute+setup 15.8%, IPC floor 3.3%.

⚑ **"The UCI round trip dominates" is refuted as stated.** Swapping the
evaluator behind the existing Python per-leaf driver caps at ~2.2× and fails the
≥5× gate. The evaluator has to be called from **inside the C tree**. That is an
architectural requirement, not an optimisation, and it is why this lands as a
seam in `_mcts_tree.c` rather than as a faster Python helper.

## The three artefacts

| What | Where | Committed? |
|---|---|---|
| `.nnue` weight file | wherever you put it | **No** — runtime artifact |
| `.pack` mmap-able weight pack | wherever you put it | **No** — runtime artifact |
| Evaluator + seam + tooling | this repo | Yes |

Nothing in the repo hardcodes a weights path; every entry point takes one.

### Getting the weights

The big net is **embedded in the Stockfish binary**, so no download is needed —
any Stockfish build will write its own copy back out:

```bash
printf 'export_net big.nnue\nquit\n' | stockfish
```

`export_net` reproduces the canonical file byte for byte: the SHA-256 prefix of
what it writes is the `nn-<hex>.nnue` name Stockfish reports as its `EvalFile`
UCI option default. (It also writes the small net into the current directory
under its own name — run it somewhere disposable.) The same file can be
downloaded from `https://tests.stockfishchess.org/api/nn/<filename>` if you
prefer.

Then build the pack:

```bash
PYTHONPATH=. python3 scripts/nnue_pack.py big.nnue big.pack
```

`scripts/nnue_parse.py` prints the SHA-256 and the canonical name; the converter
records the SHA-256 in the pack header, and `scripts/nnue_parity.py` refuses to
run if it does not match the net the engine under test is loading. That check
exists because comparing two different networks is the failure that looks
exactly like a passing gate.

## What the evaluator computes

`psqt / 16 + positional / 16`, side-to-move POV — Stockfish's *internal units*,
the number its `eval` command prints as:

```
(Big net) NNUE evaluation          -71 (side to move, internal units)
```

We deliberately do **not** reproduce `Eval::evaluate()`'s post-processing
(optimism blending, complexity damping, material scaling, rule50 damping).
Matching a number four scalings downstream would let an eval defect hide inside
them.

**Always the big net.** Stockfish's small net is a speed optimisation for
material-imbalanced positions, not an accuracy win; always-big means one format,
one code path, and no net-selection rule to keep in parity. Documented
consequence: our labels differ from Stockfish's `eval` exactly where Stockfish
would have picked the small net.

### Architecture (verified against both shipped nets, exact-EOF parse)

The big net is **not** classic HalfKAv2. It carries two feature sets:

- `HalfKAv2_hm` — 22,528 features, int16 weights, plus an int32 PSQT table
- `FullThreats` — 60,720 attack-relation features (attacker, from, to,
  attacked), int8 weights, 62 MB, plus its own int32 PSQT table

Both accumulate into the same 1024-wide int16 accumulator (Stockfish keeps two
and adds them at transform time; int16 addition is modular, so summing into one
is bit-identical and halves the traffic). Then a pairwise-product transform to
1024 uint8, then a per-bucket layer stack — `bucket = (piece_count - 1) / 4`,
eight of them.

Measured active threat features: ~24–40 typical, well under Stockfish's own
128-feature cap, which is what makes a **full refresh** viable. Incremental
accumulator updates are not implemented and are not needed yet.

## ⚑⚑ The in-check contract

**NNUE is undefined in check.** Stockfish asserts `!pos.checkers()` before
evaluating and its `eval` command refuses outright, printing
`Final evaluation: none (in check)` with no network lines at all.

- The evaluator **refuses** an in-check position: a negative status, and the
  out-parameter is left untouched. There is no sentinel value a caller could
  read as an evaluation.
- **Callers must resolve check nodes recursively before calling eval.** An
  evasion can itself give check, so the resolution is recursive: minimax backup
  over the evasions, repetition and 50-move terminals handled inside the
  resolver, mate when there are no evasions — continuing until a non-check
  position or a terminal is reached.
- The refusal is the **enforcement backstop** for that invariant, not a
  substitute for it. A caller that leans on the refusal has a hole in its
  search. The resolver itself is tree-side search infrastructure — a shared
  mandatory layer, not part of any one arm. It is described below.

The parity sample excludes in-check FENs by construction and **reports the
excluded fraction** rather than dropping them silently.

## The check resolver

`chess_anti_engine/mcts/_check_resolver.h`. Recursive, minimax, integer, and
**parameterised by what happens at a resolved non-check node** (`CaeLeafEvalFn`)
— that parameter is the only thing the two race arms differ in.

- **In check** → generate evasions and recurse on each; back up by **maximising
  for the side to move**, negating each child (a child's value is child-STM POV).
  Never an average: the side to move has no choice about being in check and must
  survive every reply.
- **No evasions** → mate. ⚑ **Tested BEFORE the draw rules**, because
  `cboard_is_game_over` checks the fifty-move clock and "no legal moves" without
  asking whether the side to move is in check — so consulting it first would
  score a checkmate at clock ≥ 100 as a draw, and a draw is a plausible 0 that
  nothing downstream can distinguish from a real one.
- **Terminal** (stalemate, three-fold, fifty-move, insufficient material, and
  LC0-style **two-fold as draw**) → 0. The decision comes from
  `cboard_search_terminal`, moved into `mcts/_search_terminal.h` so the tree and
  the resolver answer it with one piece of code rather than two that drift.
- **Not in check, not terminal** → call the leaf hook.

**Why it terminates.** Every step is a move, so it either resets the halfmove
clock or repeats a position within the current reversible run; a perpetual check
therefore hits the two-fold terminal. The depth cap (32) is a backstop for the
pathological remainder and is **counted** (`depth_cutoffs`), not assumed away —
0 on every position in the bench, and the counter is pinned as *capable* of
firing by a test that caps the recursion at 1 ply.

### Score scale

⚑ **Search-internal, and NOT centipawns.** Values are in the evaluator's own
internal units so a resolved value is a drop-in replacement for the static
evaluation it stands in for. It shares numeric constants with `stockfish/wdl.py`
by deliberate analogy, **not** by shared definition; a resolver score must never
reach the training-target pipeline as cp, and `mate_to_effective_cp` remains the
single home for the mate→cp mapping.

| | value | why |
|---|---|---|
| `CAE_RESOLVER_EVAL_CLAMP` | 32000 | every raw evaluation is clamped into ±this |
| `CAE_RESOLVER_MATE_BASE` | 100000 | mate at depth 0 |
| `CAE_RESOLVER_MATE_PLY_STEP` | 100 | deeper mates are worth less, so a shorter mate is preferred |
| mate band floor | 74400 | `100000 − 256×100` |

**The mate band cannot overlap the evaluation band** — 74400 > 32000, pinned by
test. This is audit N1's defect class in a new scale: the codebase once carried
two mate→cp formulas, one of which folded mates into a range real evaluations
reach, so on 1.34% of live rows a mate scored *below* a plain evaluation. The
clamp is measured rather than guessed: over the 49,619-position parity pool the
largest |evaluation| is 14,295, so it never binds and its job is to make the
separation total rather than empirical.

⚑ A **mate-band value from the leaf hook passes through unclamped**; only an
evaluation is clamped. Quiescence can legitimately return a mate it found below
the node, and an unconditional clamp would silently demote a forced mate to a
merely good position. A hook returning a raw evaluation clamps it itself, where
"this is not a mate score" is known.

## The two race arms

Both are providers on the same seam, published from `_nnue_ext` as
`static_arm_capsule` / `qsearch_arm_capsule` and installable by name.

| provider | leaf at a resolved non-check node |
|---|---|
| `nnue` | *(refuses in check — the backstop, unchanged)* |
| `nnue-static` | the static NNUE evaluation |
| `nnue-qsearch` | stand-pat quiescence over captures, promotions, and checks |

⚑⚑ **Arm fairness is structural.** Check resolution is mandatory correctness
work, so it must not appear as a cost of the qsearch arm — and the guarantee is
that it is *one component both arms call*, not a promise. The qsearch arm cannot
bypass it: its leaf hook is reachable only from inside the resolver, and every
child it generates that is in check goes back through `cae_resolve_node` rather
than to the evaluator. With quiescence off (`qsearch_max_ply=0`) the two arms
agree exactly, node for node — that is the arm's own negative control and it is
asserted.

⚑ The two arms' *total* resolver counts differ, and that is correct: quiescence
walks into check nodes of its own, and resolving those is quiescence's cost. The
counters are comparable because they are defined identically, not because they
are equal.

### Quiescence budget — measured, not chosen

This quiescence has no SEE or delta pruning, so its cost is close to exponential
in the ply budget while its value saturates early. Over 600 stratified positions
with the real big net, each budget scored against the deepest available:

| `max_ply` | qnodes/eval | mean \|v − v@8\| | positions differing (of 600) |
|---|---|---|---|
| 0 | 1.0 | 985.7 | 201 |
| 1 | 10.2 | 201.5 | 101 |
| 2 | 12.0 | 182.3 | 62 |
| 3 | 58.7 | 14.0 | 43 |
| **4** | **72.4** | **6.5** | **20** |
| 6 | 345.2 | 4.2 | 10 |
| 8 | 1796.5 | 0.0 | 0 |

4 to 8 is 25× the work to close a mean 6.5 internal units (~0.4 cp) on 20 of 600
positions, so **4 is the default**. ⚑ The tail is not closed: max |v − v@8| is
2425 units at both 3 and 4 plies, so a handful of positions really do need
depth — which is why it is a knob and not a constant.

### ⚑⚑ Two counters, and conflating them was a real bug

The mutual recursion carries **two** numbers, and they are not interchangeable:

| | counts | bounded by |
|---|---|---|
| `depth` | plies below the **resolution root**, through both directions | `resolver_max_depth`; also what mate distance is measured in |
| `qply` | plies of **quiescence** | `qsearch_max_ply`, `qsearch_check_plies` |

The first version passed `depth` as quiescence's ply. The resolver only calls its
leaf hook at a **non-check** node, so an in-check root entered quiescence at
`depth >= 1` — and `try_checks = (ply < check_plies)` at the default
`check_plies = 1` was therefore **false on exactly the positions the check budget
exists for**. A check chain 4 deep entered quiescence at ply 4 and stood pat
without looking at a single capture. Both knobs were accepted, consumed, and
silently meant something else.

⚑ The obvious fix — reset `qply` to 0 at the leaf hook — is also wrong, and worse:
it **refunds** the budget. Quiescence plays a check, the resolver resolves it, and
the resolver's leaves start quiescence again with a full allowance. Measured, it
does not finish. So the count is **carried** across the excursion: check
resolution is mandatory and free, quiescence moves are what the budget buys.

What separates carried from refunded is that quiescence work **saturates** as the
recursion cap rises, because quiescence is bounded by its own budget and not the
resolver's. At `max_ply=2, check_plies=2` on the four chain positions:

| `resolver_max_depth` | carried (shipped) | refunded |
|---|---|---|
| 6 | 5,468 qnodes | 8,378 |
| 12 | 8,446 | 15,327,851 |
| 32 | 8,446 | *(does not finish)* |

Every counter still looks obedient under a refund — `qmax_ply_seen` caps at
`max_ply` because each restart caps at `max_ply` all over again — so the test that
catches it asserts saturation, not a counter.

### The knobs, and where they take effect

`_nnue_ext.set_arm_config(resolver_max_depth, qsearch_max_ply,
qsearch_check_plies)` sets what **new** contexts are built with. ⚑⚑ **It is read
at `init()`, not at `eval()`**: a context snapshots its configuration and keeps
it for life, so a setter that appeared to retune a running provider would be
this repo's signature defect wearing a fix's clothes. The numbers a batch
actually ran under are therefore reported **out of the context that ran them**
(`arm_eval`'s stats dict, `arm_stats(handle)`), never restated from the globals.
The triple is read and written under a mutex, so a context can never be built
from a half-updated configuration.

### Overhead

```bash
PYTHONPATH=. nice -n 19 python3 scripts/nnue_resolver_bench.py \
    --pack big.pack --quiet-n 2000 --check-n 500 --bank runs/resolver_bench.jsonl
```

⚑ **The overhead is not one number**, and quoting it as one hides the only thing
it depends on: `blended = (1-f)*quiet + f*in_check`, where `f` is the in-check
rate of the position stream — a property of the corpus generator, not of the
resolver.

⚑⚑ **Both conditions are sampled from ONE stream, with matched weights.** An
earlier version blended the *stratified* pool's rate (round-robin over
(bucket, threat-bin) cells, which deliberately over-represents thin ones) with a
first-N in-check pool from raw playouts. Weighting two differently-drawn
conditional samples by the natural in-check fraction estimates neither
distribution — it describes a population nobody generated. Both pools are now
reservoir-sampled over the same fixed set of playouts, and `f` is measured on
that same stream. The stratified pool is still reported, labelled **COVERAGE**,
and is never blended.

⚑ **Positions carry their move history.** Rebuilding a board from its FEN throws
away the move stack, and the resolver consults it — two-fold repetition is what
makes a perpetual check terminate. A FEN-only pool measures a resolver that can
never see a repetition.

⚑ **The dedup key is resolver-complete**, not the parity sampler's
placement + side-to-move: castling and en passant change the evasions, the
halfmove clock decides the fifty-move terminal, and the history decides the
two-fold one.

⚑ It reports each arm against **its own quiet rate**, not only against the raw
evaluator: that baseline runs through a different harness
(`_nnue_ext.benchmark`), and the static arm has measured *faster* than it, which
is a harness difference and certainly not a speedup from adding work.

⚑ `--repeats` uses **one long-lived context** across all passes. Re-creating it
per pass reset the counters while time and eval count accumulated, understating
every counter in the table by the repeat factor.

⚑ `--bank` writes **one JSONL row per position** — FEN, value, the arm settings
as the context reported them, and the game/ply cluster key — staged and
`os.replace`d. An aggregate cannot be re-stratified, and a rerun happens on a box
whose load has moved on.

Measured on the big net, AVX2, 400 playouts → 72,430 positions, f = 0.0472,
2000 quiet + 500 in-check:

| arm | quiet | in-check | blended @ f=0.0472 |
|---|---|---|---|
| `nnue-static` | 165,688/s | 40,633/s | **144,654/s — 12.7% below its own quiet rate** |
| `nnue-qsearch` | 5,321/s | 468/s | 3,572/s |

An in-check evaluation costs **4.08×** a quiet one through the static arm and
**11.36×** through the qsearch arm; the recursion reached depth 3 (static) / 14
(qsearch) and `depth_cutoffs` was **0**.

⚑ **The qsearch arm's in-check rate is 17.7× lower than before the ply fix**
(8,268/s → 468/s), and the old number was the bug showing: quiescence stood pat on
in-check roots without searching anything, so the arm looked *faster* on in-check
positions than on quiet ones (0.69×). An arm that costs less where it is supposed
to do more was the anomaly; 11.36× is the mechanism working.

## Generating shards with an arm

`scripts/gen_random_selfplay_shards.py` grows two value sources,
`--value-source nnue-static` and `--value-source nnue-qsearch`, so a full
corpus can be generated with native NNUE values and **no Stockfish process**:

```bash
PYTHONPATH=. nice -n 10 python3 scripts/gen_random_selfplay_shards.py \
    --out-dir data/gen0_nnue --games 500 --workers 4 \
    --value-source nnue-static --nnue-pack big.pack --sims 32 \
    --run-id gen0_nnue_static
```

The shards are written by the same code path as every other gen-0 shard, so
they are schema-identical to the banked UCI anchor sets and the deep-SF ruler
scores both in one pass. The prior stays uniform; no net is involved.

### ⚑ The arm reads the POSITION, not the planes

`MCTSTree.pending_leaf_cboards()` returns the whole pending leaf batch in
encoded-planes row order, and that is what the arm evaluates. It has to: check
resolution needs the legal evasions (castling rights, the en-passant square),
the fifty-move terminal needs the halfmove clock, and the two-fold terminal that
makes a perpetual check finite needs the repetition history. **None of those
survive the 175-plane encoding**, so a board rebuilt from planes would be a
position that merely looks right and the resolver would score a different tree.

⚑ Not `get_pending_tb_leaves(32)`: that returns only the Syzygy-eligible subset
(`castling == 0` and few enough pieces), which is empty through most of the
opening and middlegame.

The generator re-encodes the tree's own first leaf once per worker and requires
the planes to match; `tests/test_gen_nnue_selfplay_shards.py` does it for every
row of every batch.

⚑ **Roots are counted apart from leaves.** One root position per ply also goes
through the arm (the search is handed pre-computed root logits so the leaf
contract stays exact). The C context cannot tell the two apart, so every rate
derived from it is over CALLS, is named `in_check_call_frac` rather than
"leaf fraction", and the sidecar reports `roots` next to `leaves` so a reader
can price the difference — pooling them would put a per-ply population inside a
per-leaf rate at a ratio that moves with the sim budget.

⚑ **This is a Python round trip per leaf BATCH, not per leaf.** The scoping
result that ruled out a Python driver was about a per-LEAF `python-chess` + UCI
round trip; here one batch of ~8-11 leaves crosses the boundary and the
evaluation loop itself is C with the GIL released. Measured below, because
that argument is not the evidence.

### ⚑ The value scale is a free parameter

The arms return Stockfish **internal units**; the banked UCI anchors were
labelled from Stockfish **search cp** through production's cp-logistic.
`--nnue-cp-per-unit` bridges the two and its default was measured, not assumed:
over 786 quiet positions from natural playouts, least squares of SF `go nodes
512` cp on our static value gives **slope 0.2784, R² 0.877** over
|internal| ≤ 3000 (median ratio 0.2877; whole range 0.2168, lower because SF's
cp normalisation compresses large evaluations). The default is **0.28**, which
also happens to be Stockfish's own `100 / NormalizeToPawnValue`.

⚑ Getting it wrong makes every stored policy target uniformly sharper or
flatter **with no other symptom** — which is exactly what a label-quality
readout measures. It is recorded in the sidecar and it must be pinned (or swept
as its own axis) before cells are compared. Mate scores never go through this
slope: they are folded through `mate_to_effective_cp`, the single mate→score
home.

### ⚑ `--all-root-moves` is ON by default for the arms

Under a uniform prior, sequential halving picks its root candidates by the
Gumbel draw alone — a random subset. With an uninformative value that is
harmless; with a real one the search ranks a random subset sharply and never
looks at the rest. The flag raises `topk` to the 218-move ceiling and makes the
per-root budget `max(--sims, 2 * n_legal)`; a `--topk` that would drop moves is
**refused**, not quietly raised. The banked UCI anchors were generated with it.

It also turns **off** `allow_terminal_root_shortcuts`. That shortcut prunes
every immediately-drawing legal move from a root whose value is positive,
*before* candidate selection — so the move is never evaluated, gets zero target
probability, and the stored legal mask advertises it anyway. On a root with a
repetition or a drawing capture available that is exactly the coverage hole the
flag exists to close, and it is silent (the target is dense over the legal set
either way). Production's topk-capped path keeps the shortcut, where it is a
real saving and no coverage is promised.

### ⚑ The C sampler's ranking ceiling, and what it actually costs

`gss_score_and_halve` writes into a `double scores_buf[GSS_MAX_CANDS]` (64) and
clamps `n_cands` to fit it, so `--topk 218` cannot be honoured in full on a root
with more legal moves than that. **Measured, because the obvious reading is
wrong**: on the 218-legal-move position
`R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1` at `--all-root-moves`
(realized budget 436), every one of the 218 root children is expanded and
visited and the stored target has non-zero mass on all 218. Budget is deducted
for every candidate before the clamp. What the surplus loses is a place in the
**ranking**: candidates past the ceiling are dropped from the halving unscored
and are never deepened. The cap costs ranking resolution on wide roots, not
coverage.

The C limit stays at 64 and the generator **reports** every root that exceeded
it (`root_budget.roots_over_candidate_cap`, `legal_moves_over_cap_max`, and a
one-shot warning). Over 500 measured games the widest root had **61** legal
moves and the counter read 0; the banked UCI anchors' widest was 65.

### ⚑ Banking the raw leaf observations

`--bank-leaf-observations` writes one JSONL row per arm evaluation beside the
shards (`leaf_observations_w<id>.jsonl`): the **raw internal score**, the FEN,
the pack's file digest, the cp settings in force, and the `(game, ply)` cluster
key. `--nnue-cp-per-unit` and the cp-logistic are a pure function of that raw
value, so with the bank a scale correction or a different estimator is a
**reanalysis**; without it the arithmetic is baked into `policy_target` and the
only way back is to replay the games.

It is **off by default** because it is one row per evaluated position: measured
at ~399 bytes/row, which is ~11 MB for an 8-game qsearch run and would be
~113 GB/h at the static arm's rate. **Every preregistered readout cell must be
generated with it on**; exploratory and throughput runs need not be.

### Throughput, measured

500 games, 4 workers, `nice -n 10`, `--sims 32` (realized mean 45.0 under
all-root-moves), concurrent with other GPU work on the box:

| arm | rows/h | vs UCI@32 anchor (162,667) | vs the ≥5× gate (813,000) |
|---|---|---|---|
| `nnue-static` | **6,345,273** | 39.0× | **7.8×** |
| `nnue-qsearch` | 36,609 † | 0.22× | 0.05× |

† 8 games, 4 workers, `--max-plies 60`, **and with `--bank-leaf-observations`
on** — the qsearch arm is far too slow for a 500-game point at this budget, and
because every game is truncated at 60 plies this is **not** a like-for-like
number: truncated games are the cheap prefix of a game, so it flatters the arm
rather than understating it. Read it as an order of magnitude, not a rate.

`wall_seconds` (and therefore `rows_per_hour`) **includes shard writing**: the
workers write shards inside the timed region, which is the honest accounting for
a corpus generator whose product is the files.

⚑ **The qsearch arm's cost is heavy-tailed on a real game stream and the
resolver-bench figure does not predict it.** The bench's stratified pool costs
~2 quiescence nodes per resolved leaf; inside a game the search steers into
forcing positions and the same arm spends **~3,264 resolver nodes and ~5,200
quiescence nodes per leaf**, with 2.4 % of leaves returning mate scores. The
positions themselves are ordinary — evaluating the same leaves rebuilt from
their FENs costs the same, so the tree's boards are not pathological.

`--nnue-qsearch-check-plies` is the lever: at 0 the same run does 2.8M
quiescence nodes instead of 45.9M and delivers **22,228 rows/h instead of
1,287** (single worker, `--max-plies 60`, same seed) — 17×, and the corpus
changes with it (mate-band leaves 0.13 % → 2.39 %).

## The eval seam

`chess_anti_engine/mcts/_value_provider.h` defines:

```c
typedef struct CaeValueProvider {
    const char *name;
    void *(*init)(const char *weights_path, char *err, size_t errlen);
    int   (*eval)(void *ctx, const CBoard *board, int32_t *out_value);
    void *(*retain)(void *ctx);
    void  (*destroy)(void *ctx);
    const char *(*kernel_name)(void);
} CaeValueProvider;
```

The tree holds a **pointer**, not a hard-wired call. `eval` returns a status and
writes through an out-parameter rather than returning the value directly —
required, not ceremony, because the in-check refusal must not be expressible as
an in-band int32.

`retain`/`destroy` are a **refcount pair**, not alloc/free. A caller about to
release the GIL and evaluate holds its own reference for the duration, so
another thread clearing or replacing the provider cannot unmap the weights
mid-evaluation. That failure is worth naming: a read of a freed read-only
mapping usually returns *data* rather than crashing, so the symptom is a wrong
evaluation, not a signal.

Composition is recursion through the same interface: a provider that wants an
inner evaluator (a leaf qsearch, a per-node mate extension) keeps the inner
`{provider, ctx}` pair in its own context and calls `cae_value_eval()` on it.

### ⚑⚑ Providers arrive by capsule, never by `#include`

The evaluator is header-only statics, so **every extension that includes it gets
its own copy of its code and of its static state.** A tree that obtained the
NNUE provider by including `_nnue_provider.h` would hold a second
kernel-selection flag and a second weight cache: `_nnue_ext.set_simd(False)`
would not change what the tree ran, and one weight file would be mapped twice at
62 MB. Neither is visible from outside — the tree keeps returning plausible
evaluations from the copy nobody configured.

So the module that *implements* a provider publishes it as a PyCapsule named
`cae.value_provider.v1`, wrapping a `CaeValueProviderExport` (ABI version, the
vtable, and the typed in-check exception it raises). The tree imports that
capsule. One copy of the code, one copy of its state.

```python
tree.set_value_provider("nnue", "/path/to/big.pack")   # known name, or …
tree.set_value_provider(_nnue_ext.value_provider_capsule, path)   # … any capsule
tree.value_provider_name()      # read off the tree's STORED pointer
tree.value_provider_kernel()    # "avx2"/"scalar", asked of that same pointer
tree.value_provider_eval(cboard)   # raises _nnue_ext.InCheckError in check
tree.clear_value_provider()
```

A **future provider needs no edit to the tree**: it publishes the same capsule
shape from its own module and is installed by passing that capsule.

`value_provider_name()` and `value_provider_kernel()` both report off the
pointer the consumer is holding, not off the argument that was passed in — the
producer's copy of a setting is not evidence the consumer received it. The two
observations that pin this down, and fail if the evaluator is ever compiled into
the tree again:

* `_nnue_ext.weight_cache_size()` counts *that module's* cache, so it rises when
  a pack is loaded through the **tree** only;
* `_nnue_ext.set_simd(False)` changes what `tree.value_provider_kernel()`
  reports.

Weights are mapped `PROT_READ | MAP_PRIVATE`, so every worker on the box shares
the same physical pages through the page cache; within a process a small
refcounted cache hands out one mapping per file, double-checked so two threads
racing on the first load still end up with one mapping.

⚑ **The cache key is file identity — `dev`+`inode`+`size`+`mtime` — not the
pathname.** `nnue_pack.convert()` publishes atomically, so a repacked net arrives
at the *same* path as a different file. Keyed by path, a long-lived process goes
on serving the mapping it already had: `set_value_provider(..., same_path)`
returns success and the tree keeps evaluating with the previous weights, with no
error and no log line. The identity is taken from the `fstat` of the descriptor
we actually map, so there is no window in which the file changes between the
lookup and the mapping. Handles already issued keep their own mapping — that is
what makes a swap safe rather than a use-after-free; only *new* loads observe
the replacement.

### The pack is validated relationally, not just by range

`cae_nnue_bind()` checks the header fields against **each other**, not only
against their limits: `fc0_outputs == l2 + 1`, `fc1_outputs == l3`, and every
padded width wide enough for its consumer. Each of those can be violated by a
pack whose every individual field is in range, and the consequence is not a
crash — `propagate()` reads uninitialised stack or walks off a row and returns a
plausible-looking evaluation. Our own packer cannot emit such a pack, which is
precisely why `bind()` has to reject one: it is the only thing between a
hand-made or corrupted pack and a silently wrong number.

## Running the gate

```bash
# development
PYTHONPATH=. python3 scripts/nnue_parity.py --pack big.pack --nnue big.nnue --n 5000

# the gate: 50k FENs, both kernel paths
PYTHONPATH=. nice -n 19 python3 scripts/nnue_parity.py \
    --pack big.pack --nnue big.nnue --n 50000 --simd avx2
PYTHONPATH=. nice -n 19 python3 scripts/nnue_parity.py \
    --pack big.pack --nnue big.nnue --n 50000 --simd scalar
```

**Any mismatch on a non-check FEN is a bug. There is no tolerance band.**

The AVX2 kernels are selected at *runtime*, not only at compile time, so both
paths live in one binary and each can be run against the engine. A SIMD path
that has never been compared against Stockfish is a SIMD path nobody has
checked.

⚑ **The default (portable) build is scalar-only.** The AVX2 kernels compile in
only under `-march=native` — i.e. when `CAE_EXT_NATIVE` is set, as
`scripts/build_production_extensions.py` does and CI does not. On a portable
build `_nnue_ext.HAVE_AVX2` is 0 and `set_simd(True)` raises, so anything that
flips kernels must branch on `HAVE_AVX2`, and `--simd avx2` is not runnable
there. The numbers are identical either way; that is what the gate exists to
keep true.

The gate refuses to report anything unless it actually compared:

* an empty sample, `--n 0`, or an all-refused set exits non-zero rather than
  printing a pass — "we checked nothing" must not be reportable in the same
  words as "we agree on fifty thousand";
* an engine whose `EvalFile` cannot be read is a hard failure, not a skipped
  check. It used to be `if sf.eval_file and …`, which is a provenance gate that
  cannot fire in exactly the drifted case it exists for;
* every `(fen, ours, stockfish)` triple is banked to a gzipped JSONL artifact by
  default, not just the mismatches — re-analysing from a summary means re-running
  the engine, which re-rolls the sample and the engine build along with whatever
  is being changed;
* **every banked row carries its resampling unit** (`playout`, `ply`). The
  sampler walks whole random games, so positions from one playout are
  correlated; a row without its cluster key can only be re-analysed as an
  independent draw, which it is not. The key travels with the FENs in the
  sample file, so a run driven from `--fens-in` keeps it;
* **the bank is never destroyed by a failed run.** The gate refuses to overwrite
  an existing bank without `--overwrite`, and stages into a temp file that is
  `os.replace`d into position only once the comparison has actually run. It
  previously opened the destination `"wt"` *before* validating the engine, so a
  run that then refused had already truncated the previous artifact.

⚑ **The sample is deduplicated on the state the evaluator can see** — piece
placement plus side to move, and nothing else. Traced through
`cae_nnue_pos_from_cboard`: no castling right, en-passant square, halfmove clock
or fullmove number reaches the feature computation, and
`test_ep_castling_and_clocks_do_not_change_the_evaluation` asserts that against
the compiled evaluator rather than leaving it as a claim. Deduplicating on the
full FEN instead lets one evaluator input recur with different clocks — routine
in an endgame shuffle — and counts each copy as another distinct position
covered. One full FEN per class is retained, so Stockfish still sees a real
position. The harness reports distinct states and distinct playouts alongside the
raw FEN count, because "50,000 FENs" is neither of those numbers and it is the
one that flatters.

On failure the harness localises in three layers — per-bucket PSQT, per-bucket
positional, then the total — against `scripts/nnue_reference.py`'s numpy
implementation, and reports whether that reference *also* disagrees with
Stockfish. ⚑ The reference is a **bisector, never the gate**: internal
Python-vs-C parity cannot find a rule that is wrong in both, so its only job is
to say which of the two is wrong once Stockfish has already said that one is.

## Throughput

```bash
PYTHONPATH=. nice -n 19 python3 scripts/nnue_bench.py --pack big.pack --n 5000
```

⚑ The reported rate **includes feature-index computation** (attack graph, threat
relations, index mapping). The scoping projection measured the accumulator
gather alone and said so; this closes that caveat.

⚑ The **position reuse factor is printed with the result**, because it is part
of the reading: every pass after the first re-evaluates the same positions with
their weight rows already hot, so a high `--repeats` number is a warm-cache
number. `--repeats 1` evaluates each position exactly once and is the
conservative figure to quote when the question is working-set reuse.

The benchmark reports two position sets, because one number here misleads in
both directions: the *stratified* set spans all eight layer stacks but is
dominated by sparse endgames where there is little to gather, while the
*middlegame* set (≥24 pieces) is the regime a game-playing generator actually
spends its time in and is the slower one.

## Files

| File | Role |
|---|---|
| `scripts/nnue_parse.py` | `.nnue` parser; version check is FATAL; must land on exact EOF |
| `scripts/nnue_pack.py` | `.nnue` → mmap-able little-endian pack |
| `scripts/nnue_reference.py` | numpy forward pass — debugging bisector, not a gate |
| `scripts/nnue_fens.py` | stratified, in-check-excluding FEN sampler |
| `scripts/nnue_parity.py` | the gate: exact integer equality against Stockfish |
| `scripts/nnue_bench.py` | throughput, index computation included |
| `scripts/nnue_resolver_bench.py` | resolver overhead, blended at the measured in-check rate |
| `chess_anti_engine/nnue/_nnue_impl.h` | the evaluator |
| `chess_anti_engine/nnue/_nnue_provider.h` | the evaluator as a value provider |
| `chess_anti_engine/nnue/_arm_providers.h` | the two race arms + the provider registry |
| `chess_anti_engine/mcts/_value_provider.h` | the seam and its status contract |
| `chess_anti_engine/mcts/_check_resolver.h` | recursive check resolution, shared by both arms |
| `chess_anti_engine/mcts/_search_terminal.h` | the terminal decision, shared with the tree |
| `chess_anti_engine/nnue/_nnue_ext.c` | Python surface for parity, bench, tests |
| `scripts/gen_random_selfplay_shards.py` | the shard generator; `--value-source nnue-static` / `nnue-qsearch` |
| `tests/test_nnue_native_eval.py` | parser, converter, indices, refusal, seam |
| `tests/test_check_resolver.py` | recursion, terminals, minimax, arm fairness, the knobs |
| `tests/test_gen_nnue_selfplay_shards.py` | the arms inside the generator: schema, binding, knob liveness |
