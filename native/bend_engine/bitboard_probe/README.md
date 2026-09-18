# Pinned-U64 sliding-attack integration gate

This is the first DeepFin consumer of the U64 compiler fork, independent of the
CUDA/search stack. It leaves training, search, the existing release-toolchain
probes, and the regular perft suite unchanged. It is **not a Bend chess engine**:
legal move generation and perft have not been replaced.

## Boundary

`fixtures.c` includes DeepFin's real `_cboard_impl.h` and fast-slider header.
It supplies each square's real masks, magic multiplier and production magic
attack table. A PEXT-order table is built independently with a carry-rippler
occupancy enumeration and the retained C ray walker; the generator does not
use PEXT or PDEP. Expected attacks are computed by that ray walker, not by the
Bend index function or a second copy of its formulas.

`main.bend` loads one fixture per square/piece, then performs **all index
arithmetic, Array<U64> lookups, bounds checks and full-width attack comparisons
in pure Bend**. There is no C callback per lookup. `Sliders.bend` holds reusable
PEXT/magic indexing helpers. The two table layouts are distinct and must never
be mixed. Bounds are checked explicitly because Bend Array.get wraps indices.

The foreign-effect marshaler is tied to the pinned compiler's concrete
`Fixture` and `Array<U64>` layouts. Each U64 is stored as two U32 limbs, not an
arbitrary raw word in a tagged Term slot; the fixture arity is checked at runtime.
A future compiler/layout change requires requalifying this boundary.

## Run (Bun, Clang and Python 3 required)

From the DeepFin root, in a disposable development checkout:

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python native/bend_engine/bitboard_probe/run_probe.py \
  --report artifacts/bend-u64-sliders.json
```

The installer reads `toolchain.json`, fetches the exact fork commit into
`build/bend_u64_toolchain/source`, and verifies the compiler/prelude/effect
sources by SHA-256. It never replaces the existing `build/bend_toolchain` or
installs an unpinned latest release. Existing source directories are not reset.
With a source checkout already available:

```sh
python native/bend_engine/bitboard_probe/run_probe.py \
  --compiler-root /path/to/pinned/bend --modes generic portable native ubsan
```

`--bun` and `--cc` select executables; temporary build directories are removed.
The test uses one CPU thread, no GPU, no Torch and no Python extension build.
Native-target executables run only on the machine that built them.

## Coverage and what failure means

For both rook and bishop on all 64 squares, the probe checks:

- all 107,648 relevant-occupancy subsets;
- the same subsets with **all off-mask bits set** (including edge squares);
- 64 seeded full-board/sparse/dense occupancies per square/piece.

That is **223,488 occupancy cases**, each checked using PEXT and magic indexing:
**446,976 exact attack comparisons per build**. Both index bounds and both
64-bit attack results must match. A deliberately corrupted table must make the
executable exit with code 1 and an identifying square/case message. The runner
also rejects incomplete/duplicate report rows and compares full reports across
all build variants. Generic, forced-portable, native and UBSan builds are tested.

`verification_wall_seconds` includes fixture generation, FFI copying, checks
and printing. It is not an attack-generation benchmark or a C-versus-Bend speed
claim. This gate establishes the actual table-lookup integration before perft
or a separate controlled throughput measurement.

Only the cheap report-parser tests run in ordinary pytest. Native compilation
runs in the path-scoped `Bend U64 sliders` workflow or by the explicit command
above. No perft depth is increased. The fork's upstream source-token budget
issue remains separate; no compiler source, reference semantics, tests or size
limits are changed here.
