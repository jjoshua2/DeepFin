# Stored table headers through all later writes

This is a proof-only continuation of #865. It promotes its checked one-block
header composition and proves selected headers remain correct through every
later block and the final extras stage. Production code and existing accepted
laws are unchanged.

## Public contracts

`own_block_header` is the previously supplementary result, now an explicit
LAWS/PROOF obligation. It observes the actual mask or widened prefix after one
real `Tables.tables` iteration, including that block's complete fill.

`stored_header_after_tables` selects a block by `before`, followed by `after`
later blocks. The actual call starts at key `k` and certified `prefix(k)`, executes
`before + 1 + after` blocks, and the queried key is `before + k`. The only numeric
budget is `before + 1 + after + k <= 128`. The input is an arbitrary complete
depth-17 affine array. No initial header values or per-write preservation theorem
is supplied by the caller. The result is the complete final read pair: the actual
updated array plus the intended actual mask or widened certified prefix.

`stored_header_after_extras` composes that result with actual initialization and
actual extras. Allocation depth stays symbolic with an explicit equality to 17;
the proof supplies the complete-shape certificate. The extras square range is
bounded by 64. The seed remains arbitrary. This includes all selected positions
in a full 128-block run and a complete 64-square extras pass, as parameter
instances; no separate normalization of a closed `Tables.build()` is claimed.

The mask specification is the actual production relevant mask, not an independent
source theorem of blocker-ray geometry. Correct headers do not by themselves prove
all final slider-data values survive later blocks or that lookup is geometrically
correct. These remain separate P2 obligations.

## Proof structure

Facts/Header/OneBlock retain the exact previously checked supplementary bytes.
Addresses relates header slots to their mathematical indices and proves disjointness
between either earlier header and both later header writes. Preserve combines the
actual public-array frame lemmas, bounded-fill framing and the exact-prefix schedule.
Stored inducts over the selected position and later writes, then uses the existing
checked `tables_follow_prefix` theorem to return to the actual builder. Pipeline
uses the already-qualified extras frame and derives initialization shape.

The explicit schedule retains actual masks, public writes and actual `Tables.fill`.
It is not an independent implementation supplying candidate answers. The existing
prefix size certificates are discharged by their original imported producers.

## Commands and evidence labels

```bash
# New source contracts and all new rejection controls only:
bun native/bend_engine/standalone/proofs/headers/focused.js /path/to/pinned/bend --report /tmp/header-focused.json
# Development controls only: reports focused_gate and consumer as NOT_RUN:
bun native/bend_engine/standalone/proofs/headers/focused.js /path/to/pinned/bend --controls-only --report /tmp/header-controls.json
# Full opt-in aggregate, including the unchanged 100-law/226-control parent:
bun native/bend_engine/standalone/proofs/headers/verify.js /path/to/pinned/bend --report /tmp/header-aggregate.json
# Independent complete-buffer native reference:
bun native/bend_engine/standalone/proofs/headers/verify_native.js /path/to/pinned/bend --report /tmp/header-native.json
```

The expected full aggregate is 103 laws and 243 rejection controls. It must not be
called executed merely because exact-source parent evidence is reused with the new
focused gate. The dated record identifies which commands actually ran.

The source gate requires zero exit and exactly `All terms check.`. Semantic controls
require ordinary expected/observed errors at intended locations, not crashes,
timeouts, missing imports or linearity errors. Policy checks reject omitted laws,
missing proof imports, holes, foreign/unsafe dependencies and symlinks. The one
explicit external-to-standalone proof dependency allowed is the unchanged, inherited
`bitboard_probe/Sliders.bend`, which supplies existing public-operation proofs.

The native candidate creates fresh storage with a different XOR-derived value in
every cell, computes its own start offset from actual production masks and executes
actual table/extras loops. Nine cases cover prefix, suffix and full runs, both sides
of the rook/bishop boundary, terminal zero count and full extras. Every one of the
131072 public reads per case is compared with an independent flat-array reference.
The reference uses signed-coordinate rays, direct bit deposition and mathematical
prefixes. Expected data never enters the candidate. Four modes repeat fixtures;
there is no exhaustive native arbitrary-input claim or physical lifetime theorem.

All new checks are bounded and opt-in. No routine perft, model/GPU, full-engine,
training or performance workload is added. Source proofs trust the pinned checker
and Base semantics; native lowering, allocation/lifetime, ABI, toolchain and hardware
remain separate trust boundaries. Self-review is not independent review.
