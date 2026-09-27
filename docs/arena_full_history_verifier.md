# Future matched-sims full-history verifier

`chess_anti_engine.eval.strict_arena_history.verify_strict_arena_history` is a
CPU-only verifier of **supplied immutable JSONL and PGN bytes**. It is separate
from the arena writer and from the running Selected-E packet. The default
opening contract is a standard-start, 16-ply stack. Tests can supply a smaller
root/ply contract for tiny rule-50 and WDL+DTZ examples; a production caller
must pin those arguments in its plan.

**No current full-history PASS:** PR #903's arena writer has no played-move
digest in its JSONL/PGN schema. A different legal PGN mainline can have the
same opening, result and ply count while the JSONL stays unchanged. This
verifier therefore requires a future writer to record `played_uci_sha256` in
each JSONL game row and `PlayedUCISHA256` in each PGN game. Both must equal
SHA-256 of `b'arena-played-uci-v1\0'` followed by compact ASCII JSON of the
ordered **actual played** UCI moves, excluding the pre-play opening stack.
Existing PR #903 logs fail this gate. The verifier also binds the recorded
`opening_plies` to the expected stack length and caps every row's played plies
at recorded `max_plies`.

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
halves of both colors. Exactly one half may have a committed stale JSONL row:
the writer never replays a complete pair. Identical attempt signatures are
refused. Stale attempts are checked
for cross-file source, opening and record identity but are not counted as
scored games or independently terminal-certified. A future writer should add a
per-attempt ID for stronger duplicate custody.

**Normal crash limitation:** the writer writes PGN before JSONL. A crash
between them may leave a PGN-only orphan, even without a committed stale
JSONL row or a `ResumeReplay` marker on restart. This verifier currently
refuses extra PGNs rather than choosing one by guesswork. Thus the future
digest schema alone does not guarantee that every normal resumed arena can
pass; an attempt ID in both records or an independently reviewed unique-digest
selection rule is needed. No production resume population is certified here.

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
The later PR #903 head inspected for this fail-closed correction is
`0a733b27c07bc29586c40a735d8b5e7bd7a19af8` (tree
`da8556e406b25eebae188846f143e75967fd07a2`). Its writer source
`scripts/arena_standard.py` has SHA-256
`a295fc15cce80f731cd3310d9d94fc076bfaf2df4668542d2f2d93e5a3822a6a`;
the unchanged PGN writer and game-log utility are
`b8b6eb84d34389ad8387c334304144c4e605fd5613c1adc0a039d0dff152aae8`
and `87de3afe424507c488d879897b0cc77a47397bb79fd4b96927f3f57a3fb121`.
