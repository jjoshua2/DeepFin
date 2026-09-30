# Explicit retained-Syzygy corpus option — September 22

The corpus generator can now forward an explicit retained-tablebase request to
an engine that supports it. `--sf-retain-syzygy-on-new-game` passes
`SyzygyRetainOnNewGame=true` through the frozen worker specification to both the
initial Stockfish process and every replacement. This is a dedicated boolean
option; it does not open a general search-setting override channel.
Six-man adjudication and per-game TT clearing remain unchanged: the generator
still sends `ucinewgame`; the separately qualified engine option retains only
its tablebase initialization across that call.

The default remains off. Without the flag, the wrapper sends its previous UCI
handshake and the requested configuration stamp keeps its previous keys/hash.
With the flag, the engine must advertise exactly the standard check option;
missing, differently named or wrong-type advertisements fail before search and
the process is reaped. The new `setoption` line precedes readiness. Arbitrary
truthy strings are rejected by the Python API.

The requested flag participates in configuration identity. Resume refuses both
changing an old/default-off run to on and changing an on run to off; absent legacy
keys mean the historical off default. A successfully forwarded request is banked
as `sf_syzygy_retain_option_sent` in the worker's realized stamp. UCI provides no
effective-option readback: that stamp reports the sent command, not measured
mapping retention or a claim that search became faster.

Example for a separately qualified engine build:

```bash
python -m scripts.gen_sf_rooted_corpus \
  --out-dir /path/to/fresh-corpus --stockfish /path/to/qualified-stockfish \
  --sf-retain-syzygy-on-new-game
```

This enables later experiments with the already banked optional retention build;
it does not replace any deployed binary, change active corpus workers, adopt a
new generator schedule, or qualify label equivalence for a new run. The existing
retention evidence remains under
`~/chess-artifacts/operations/sf-retain-qualification-20260921/` and is conditional
on its frozen engine/protocol/fixtures. No new generation or engine benchmark was
launched for this plumbing change.

Tests exercise the CLI-to-worker setting, actual PTY command delivery on initial
and replacement engines, exact default setoption sequence, unsupported engine
cleanup, bool validation, legacy Namespace compatibility and both resume changes.
The existing generator suite was also exercised; focused final static and
independent review results are recorded in the PR.

Independent parent review approved the frozen source: [receipt](evidence/retained-syzygy-option-20260922/retain-option-parent-review-20260922.json).
Its protocol/thread/reaping suite passed 81 tests, with one pre-existing
PDEATHSIG expectation failure reproduced on unchanged main in this host environment.
Author focused tests passed 14. Final scoped static checks pass with zero findings.
Whole-repository Ruff/Vulture passed and 14 type errors match the established
baseline; an additional bool-guard warning sampled during the whole check was
removed by the subsequently validated object-typed helper. Complete logs and
[validation accounting](evidence/retained-syzygy-option-20260922/validation.json)
are banked; whole-repository lint is not represented as green.
