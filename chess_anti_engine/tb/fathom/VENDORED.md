# Vendored: Fathom (Syzygy tablebase prober)

- Upstream: https://github.com/jdart1/Fathom
- Commit: `c9c6fef0dddc05d2e242c183acf5833149ab676d` (2025-12-23, "Fix possible
  integer alignment issues")
- Licence: MIT, see `LICENSE` beside this file (copyright 2013-2018 Ronald de
  Man, 2015 basil00, 2016-2025 Jon Dart). The MIT terms require the copyright
  notice and permission notice to accompany all copies or substantial portions,
  so `LICENSE` and the per-file headers are kept verbatim.

## Files

Copied byte-for-byte from the upstream tree at that commit (`src/` keeps the
upstream layout; `src/apps/` and the Makefiles are not vendored). No patch is
applied: the wrapper reaches Fathom's static `probe_wdl`/`probe_dtz` by
including `src/tbprobe.c` into `chess_anti_engine/tb/_tb_fathom.c` instead of
editing it.

| file | sha256 |
|---|---|
| `LICENSE` | `c8038055839bd02f995cd7d1bba19657720097f5604411a7977ebcb5b4a37759` |
| `README.md` | `8a44c2a068f97fb237cea30f8039f4942950668c97b8f37092d1065b26cd074b` |
| `src/stdendian.h` | `e5b8187eec89ef83e731ea92be8ca0df4b533d8d21075fc4e2c86b4b00d7b7c3` |
| `src/tbchess.c` | `119f19fa8714686798ca7be011e5b234f99c93aec751e9492f2664757d6c029d` |
| `src/tbconfig.h` | `69fe1821b471164bd759cf46c486d55b7fa727e348a5ee306b18d19537ac3a03` |
| `src/tbprobe.c` | `7ed0cc80626271342bfe413d6c421756b6851f977a064461f5f555e1b598df35` |
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
