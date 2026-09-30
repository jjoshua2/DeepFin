# Strict factorial readout: compact evidence

These files are source-only copies of the frozen analysis protocol and numerical
receipts. They contain no checkpoints, registered corpus, opening book, PGNs or
bulk game banks. `SHA256SUMS` pins the exact published bytes.

| File | Role |
| --- | --- |
| `protocol.md` | Preregistered four-edge estimators, validation gate and limits; source SHA-256 `746f78e57289b922be772cdf9d6e6eb2c738c72ffdf41269d3e8380407abc238` |
| `four-edge-readout.json` | Frozen strict result and plan, receipt, game-log, PGN and result hashes; source SHA-256 `1e923d16ae317e5b2aeaadd27490d248b57681ee7fc6d08fe3d786e3a32594db` |
| `independent-recompute.json` | Independent game-score and paired-bootstrap recomputation from the small logs |
| `independent-validation.json` | Scope and PASS status of that limited recomputation |
| `ed-context.json` | Independently recomputed separate E−D score and source hashes |

The external four-edge banks are identified by logical operation IDs below.
Every edge has 256 games and 128 two-color opening pairs. The frozen readout
contains each plan, completion receipt and game-log SHA-256; it also pins a
common opening book SHA-256 of
`70d0dfa50a6b1191f1db702a093a4911b7433c612a30599517c2c2d92f0cea7c`.

| Edge | Retained operation ID | Game-log SHA-256 |
| --- | --- | --- |
| B−A | `factorial58-ba-rule50-20260924` | `b1bbceff6b808acec097add54e62203f52fde671ab09a3b37f021da903a36e33` |
| D−C | `factorial58-dc-rule50-20260923` | `4c314b64867d57c31a371478518c2dcea03ad6ebb0a25a2e4a69d08a93e8e1ec` |
| C−A | `factorial58-ca-rule50-20260924` | `1fe46d4b3c4389103e6526171263b33209dee09133afd5535c907e31fb5311c0` |
| D−B | `factorial58-db-rule50-20260924` | `7f98e29e5cad2721debb5819cd3b5756e86104f3320f0b75e235f7f8bd86498b` |

The E−D context game log is pinned in `ed-context.json` and is discussed in
the earlier [Syzygy correctness audit](../../2026-09-23-bootstrap-syzygy-correctness-audit.md).
The recomputation verified plan/receipt/log hashes, 256 game records per edge,
result/color scoring, both color halves and opening alignment, then matched
the frozen means, paired standard errors and 20,000 shared-pair bootstrap draws.
It did not independently replay moves, verify every tablebase file, hash model
checkpoints or validate training-source integrity. The frozen strict arena
verifiers provide those game-level checks, subject to their own recorded limits.
