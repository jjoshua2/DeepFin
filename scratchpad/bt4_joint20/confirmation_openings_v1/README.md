# Fresh confirmation openings v1

Prepared with seed **20260906** for later independent-training-seed confirmation.
No candidate, dose, training seed or winner is selected here; no arena was launched.

`openings.fen` contains exactly **500** ordered history-bearing seed lines:
`<standard start FEN> | <16 legal UCI moves>`. The existing frozen
`arena_standard.load_fen_openings` loader reproduced every starting board,
terminal board and all 16 moves, with zero repeated terminal EPDs and zero overlap
with the 500 terminal EPDs exposed by the completed global-screen banks.
EPD exclusion ignores clocks and excludes transpositions even when history differs.

The weighted sampler produced 1,000 unique terminal positions from 1,006 draws.
Six unique positions had already been exposed; one of those was also unusable.
All 1,000 had 16 plies. The first 500 of 994 eligible fresh positions were selected.
`sampled_pool.jsonl` preserves every draw, including six rejected duplicate draws,
all histories, exclusion flags and selection indices (in the receipt).
There was no duplicate fallback. Generation took 124.99 seconds at nice 19 with
numeric thread caps of two; total bank/provenance files are under 800 KiB.

## Provenance

`generation_receipt.json` pins the book, frozen sampler/arena sources, generator,
NumPy/python-chess runtime, RNG states, all three completed E0 screen bank hashes
and their completion receipts. Those three budgets use the same 500 seed-42
opening pairs; no global-treatment arena had completed when this bank was made.
`excluded_epds.json` banks their union. The source book's prefix-frequency sampling
is retained, conditioned on fresh legal 16-ply positions with at least two legal
moves, without repeated terminal positions. This is not uniform sampling of PGNs.

SHA256:

- `openings.fen`: `e0d13b2ea70c0ac278570a0e463c3c1c3030a18256522bcba864db23cdc07c98`
- `generation_receipt.json`: `8b781aeb5612e4ab2051f07d3ed80a032463f80eb0d69d688c00390b73909252`
- `generate.py`: `86362a65e786b03749cde79934210f648b11e8e68efc6eaab7ab1a5e53789094`

The exact completed generation command was:

```bash
nice -n 19 /usr/bin/python3 /home/josh/projects/chess/scratchpad/bt4_joint20/confirmation_openings_v1/generate.py
```

Do not rerun in this populated directory: the generator refuses to overwrite its
artifacts. The script and complete input receipts are retained for review.

## Later confirmation use

Use this same immutable bank and pair order across all registered confirmation
budgets. Add the following to the separately registered arena invocation:

```text
--openings-fen /home/josh/projects/chess/scratchpad/bt4_joint20/confirmation_openings_v1/openings.fen
--games 1000
```

Omit both `--openings` and `--opening-plies`; the latter is explicitly rejected
with `--openings-fen`. Exactly 500 usable rows means no loader subsampling or
shrinking. Preserve the full history-bearing lines; bare terminal FENs would
change model inputs. Preregister the separate arena/search and training seeds.

Before launch/resume, verify the bank SHA. The frozen arena fingerprint stores
its path, not a file-content or history hash, and resume checks terminal FENs;
those checks alone do not protect against changed histories at the same position.
The confirmation readout must explicitly validate this bank and the FEN-list
protocol; the old global-screen readout requires `openings_kind=book` and is not
appropriate unchanged.

Before confirmation, check that every additional selection-screen bank's opening
EPDs remain inside the frozen exclusion set (the registered global arms share
seed 42 and the same source book). If some later screen used other openings,
check their intersection with this bank explicitly before calling it fresh.
Disjoint terminal opening positions do not imply disjoint training data, unrelated
opening families or absence of later-game transpositions. This bank establishes
opening freshness; it does not establish a strength result.
