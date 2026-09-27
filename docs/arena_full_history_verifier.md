# Future matched-sims full-history verifier

`chess_anti_engine.eval.strict_arena_history.verify_strict_arena_history` is a
CPU-only verifier of **supplied immutable JSONL and PGN bytes**. It is separate
from the arena writer and from the running Selected-E packet. The default
opening contract is a standard-start, 16-ply stack. Tests can supply a smaller
root/ply contract for tiny rule-50 and WDL+DTZ examples; a production caller
must pin those arguments in its plan.

The caller supplies exact expected JSONL settings, the independently
reconstructed opening board for every scored pair ID, exact PGN source and
protocol tags, engine names/search descriptions, and an already opened strict
six-man WDL+DTZ probe. The verifier requires both halves for each supplied
pair, with no extra scored pair/half, checks JSONL settings fingerprint and
matched-sims six-man protocol, and cross-checks PGN `PairId`/`PairHalf`, colors,
search descriptions, results, plies, opening root/UCIs, start FEN, and other
source tags. It reports the observed compile and evaluator-hoist tags because
these can vary across a resume. It legally replays the book stack and PGN mainline moves on one
board, refusing a move after an earlier natural or rule-50/Syzygy adjudication.
At the final position it verifies natural repetition/rule-50 results or probes
both WDL and DTZ with the match adjudication rules. It rejects max-ply draws,
`*`, PGN parser errors, unparsed population, and missing history.

The append-only resume case is deliberately narrow. Each `(PairId, PairHalf)`
can have at most one replacement; JSONL and PGN attempt counts and visible
fields must match in order. A duplicate PGN key must have exactly one later
`ResumeReplay=1` replacement, and a replayed pair must have marked selected
halves of both colors. Identical JSONL attempts are refused because the row
does not carry PGN move bytes or an attempt digest. Stale attempts are checked
for cross-file source, opening and record identity but are not counted as
scored games or independently terminal-certified. A future writer should add a
per-attempt ID and movetext digest for stronger duplicate custody.

This function opens no paths. An admitting caller still needs bounded
no-follow same-byte capture of the two files; a trusted bootstrap and pinned
source/parser closure; authenticated final arena population/settings and
schedule; source and health identity for the separately opened six-man
tablebase; and independent run exit/terminal readback. The PGN and settings
`SyzygyProtocol` tags record a claim: replay checks the archived game result,
but cannot prove which root/leaf probes the playing engine privately used.
The verifier does not grant current Selected-E or legacy matched-time games
retroactive opening-history credit. No GPU, tablebase payload, model, registered
book, or registered arena data was used to build or test this branch.

Source basis: prospective contract
`/tmp/future-strict-arena-full-history-verifier-contract-20260927.md`
(SHA-256 `f7be05a7a4e0a21041e92d2229113693bc45b4cc669d458733cb6ea3c0489d3f`)
and reviewed arena opening-stack commit
`e9a66f57bf665e491461b328df3ff167188a1c46`. This verifier branch is
based separately on `origin/main` `38bcfa49a6ce2fab583fffc020ca8610c082794a`;
it does not contain the arena writer patch.
