# Vendored: Fathom (Syzygy tablebase prober)

- Upstream: https://github.com/jdart1/Fathom
- Commit: `c9c6fef0dddc05d2e242c183acf5833149ab676d` (2025-12-23, "Fix possible
  integer alignment issues")
- Licence: MIT, see `LICENSE` beside this file (copyright 2013-2018 Ronald de
  Man, 2015 basil00, 2016-2025 Jon Dart). The MIT terms require the copyright
  notice and permission notice to accompany all copies or substantial portions,
  so `LICENSE` and the per-file headers are kept verbatim.

`src/stdendian.h` carries no licence header of its own. Its first lines name
its origin as `https://gist.github.com/michaeljclark/3b4fd912f6fa8bb598b3`,
"modified" by Fathom. That gist could not be fetched (the GitHub API returns
"Not Found", checked 2026-11-14), so its original terms are unrecorded. It is
vendored as part of upstream Fathom under Fathom's `LICENSE`.

The notice ships with the package: `pyproject.toml` lists `fathom/LICENSE`,
`fathom/VENDORED.md` and `fathom/README.md` as package data, so they are in
the sdist and the wheel.

## Files

Copied byte-for-byte from the upstream tree at that commit (`src/` keeps the
upstream layout; `src/apps/` and the Makefiles are not vendored), except
`src/tbprobe.c`, which carries the one local patch below. The wrapper reaches
Fathom's static `probe_wdl`/`probe_dtz` by including `src/tbprobe.c` into
`chess_anti_engine/tb/_tb_fathom.c`.

## Local patch (`src/tbprobe.c`)

Every changed line is marked `DEEPFIN-LOCAL`. Upstream sha256 of the file at
the commit above: `7ed0cc80626271342bfe413d6c421756b6851f977a064461f5f555e1b598df35`.
Purpose: a corrupt table must fail at load (the engine refuses `SyzygyPath`,
naming the file), never crash the engine on the first probe of that material
mid-game. It is deliberately NOT a general hardening of the decoder:

- a registry of each table's name and entry (`dfTables`, filled in `init_tb`,
  reset in `tb_init`), so `_tb_fathom.c` can run `init_table` for every table
  eagerly (`df_tb_validate_table`);
- `init_table`: a file shorter than its 16-byte header is corrupt; a failed
  `setup_pairs` fails the table; the DTM/DTZ map walks stay inside the file;
  the index, size and data sections the header describes must fit in the file
  (overflow-safe replay of the pointer walk). Failure unmaps the file and
  returns false, which marks the table unusable exactly like upstream's own
  bad-magic path;
- `setup_pairs`: returns NULL (instead of dereferencing a NULL `malloc`) on an
  allocation failure, on a header no real table has (`h` outside 1..64,
  `minLen` 0, `blockSize` > 31, `idxBits` outside 1..63, `numSyms` >= 4096,
  which upstream only `assert`s) and when the symbol table runs past the file;
- `calc_symLen`: a symbol that references a symbol index out of range, or a
  reference cycle, marks the table corrupt instead of reading out of bounds or
  recursing without end.

On Windows the file-size bound is not available here and those checks are
inert (Windows is untested anyway). Corruption inside the compressed data is
not detected; see docs/tablebase_prober.md, "Eager validation at load".

| file | sha256 (`src/tbprobe.c`: of the locally patched file) |
|---|---|
| `LICENSE` | `c8038055839bd02f995cd7d1bba19657720097f5604411a7977ebcb5b4a37759` |
| `README.md` | `8a44c2a068f97fb237cea30f8039f4942950668c97b8f37092d1065b26cd074b` |
| `src/stdendian.h` | `e5b8187eec89ef83e731ea92be8ca0df4b533d8d21075fc4e2c86b4b00d7b7c3` |
| `src/tbchess.c` | `119f19fa8714686798ca7be011e5b234f99c93aec751e9492f2664757d6c029d` |
| `src/tbconfig.h` | `69fe1821b471164bd759cf46c486d55b7fa727e348a5ee306b18d19537ac3a03` |
| `src/tbprobe.c` | `206fe66758935b4665facef459e08a4e241de6635e9e43fc25492b0384d9d80d` |
| `src/tbprobe.h` | `dc7808dc58c2a1af921a612f94041b917dfac94d3215a76a2b2b0c53c94ea561` |

`tests/test_tb_prober.py::test_vendored_fathom_files_match_recorded_hashes`
re-hashes these files, so an edit here fails CI until this table (and the
reason for the edit) is updated.

## Upstream behaviours a caller should know

- `map_tb()` calls `exit(EXIT_FAILURE)` if `mmap` of a table that passed the
  size check fails (e.g. address-space exhaustion). Tables are mapped lazily,
  on first probe of that material.
- `test_tb()` prints `info string Incomplete tablebase file ...` to stdout and
  stderr for a file whose size is not 16 mod 64, and skips it.
- Tables are opened `O_RDONLY` and mapped `PROT_READ`/`MAP_SHARED`; nothing is
  ever written.
