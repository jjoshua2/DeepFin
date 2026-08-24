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
  search. The resolver itself is tree-side search infrastructure and lands
  separately; it is a shared mandatory layer, not part of any one arm.

The parity sample excludes in-check FENs by construction and **reports the
excluded fraction** rather than dropping them silently.

## The eval seam

`chess_anti_engine/mcts/_value_provider.h` defines:

```c
typedef struct CaeValueProvider {
    const char *name;
    void *(*init)(const char *weights_path, char *err, size_t errlen);
    int   (*eval)(void *ctx, const CBoard *board, int32_t *out_value);
    void  (*destroy)(void *ctx);
} CaeValueProvider;
```

The tree holds a **pointer**, not a hard-wired call. `eval` returns a status and
writes through an out-parameter rather than returning the value directly —
required, not ceremony, because the in-check refusal must not be expressible as
an in-band int32.

Composition is recursion through the same interface: a provider that wants an
inner evaluator (a leaf qsearch, a per-node mate extension) keeps the inner
`{provider, ctx}` pair in its own context and calls `cae_value_eval()` on it. A
second provider is an added entry in `CAE_VALUE_PROVIDERS`, not a change to the
call site.

Python surface on the tree:

```python
tree.set_value_provider("nnue", "/path/to/big.pack")
tree.value_provider_name()      # read off the tree's STORED pointer
tree.value_provider_eval(cboard)
tree.clear_value_provider()
```

`value_provider_name()` reports the name off the pointer the consumer is
holding, not off the argument that was passed in — the producer's copy of a
setting is not evidence the consumer received it.

Weights are mapped `PROT_READ | MAP_PRIVATE`, so every worker on the box shares
the same physical pages through the page cache; within a process a small
refcounted cache hands out one mapping per path.

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
| `chess_anti_engine/nnue/_nnue_impl.h` | the evaluator |
| `chess_anti_engine/nnue/_nnue_provider.h` | the evaluator as a value provider |
| `chess_anti_engine/mcts/_value_provider.h` | the seam and its status contract |
| `chess_anti_engine/nnue/_nnue_ext.c` | Python surface for parity, bench, tests |
| `tests/test_nnue_native_eval.py` | parser, converter, indices, refusal, seam |
