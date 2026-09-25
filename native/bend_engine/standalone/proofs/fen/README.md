# Public FEN result composition

This opt-in suite follows the qualified `frontier/` suite on PR #882. It reuses
its actual initialized-placement consistency result rather than introducing a
second frontier implementation. No production Position/Text/Chess code changes.

```sh
bun native/bend_engine/standalone/proofs/fen/focused.js /path/to/pinned/bend --report /tmp/fen-focused.json
bun native/bend_engine/standalone/proofs/fen/verify_native.js /path/to/pinned/bend --report /tmp/fen-native.json
# The expensive full chain is separate from focused or modular evidence:
bun native/bend_engine/standalone/proofs/fen/verify.js /path/to/pinned/bend --report /tmp/fen-aggregate.json
```

## Three source contracts

`fen_result_is_consistent` states that, for all six input Strings, the actual
optional `Position.fen` result is None or its Board satisfies the accepted
independent partition invariant. No caller-provided placement, freshness or
consistency certificate is required. `Consistency` adapts the monadic composition
from the saved local freshness candidate to the published frontier producer;
it does not claim new parser-wide freshness mathematics.

`rejected_placement_rejects_fen` states that an actual failed initialized
placement produces None regardless of the five remaining input Strings. The
premise is the actual optional placement result, not a guessed malformed-input
classification. Incomplete prefixes that might later be extended are not covered
by a claim of permanent lexical invalidity.

`validated_fields_construct_exact_game` returns the entire exact Some Game,
including metadata, halfmove/fullmove counts and empty history. It explicitly
requires that actual placement, side, rights, EP and decimal parsers return the
specified values, and that the resulting counts satisfy the actual 1,000,000
limit with positive fullmove count. These are actual subparser certificates,
not an assumed desired Game or an independently proved six-field grammar.
`Assembly.actual` links the proof-only monadic assembly to imported Position.fen.
None of these assembly helpers executes in the native candidate.

`Witness.bend` supplies closed nonempty acceptance and invalid-placement examples.
The consumer additionally uses the universal contracts with symbolic inputs.
The consistency predicate permits None, so the exact-result law and concrete
acceptance witness are important: an always-rejecting parser is not correct merely
because it never returns an inconsistent Board.

## Explicit limits

Representation consistency is not legal chess. King counts, legal pawn ranks,
meaningful rights/EP, king safety and reachable positions remain separate. The
actual FEN stage permits an empty Board and syntactically valid EP coordinates
that may not describe a legal position. The exact-field theorem does not prove
the validators' entire lexical soundness/completeness or the intended placement
contents for arbitrary input. Raw layout rollback is not claimed.

The saved local `freshness/` candidate remains preserved separately; this suite
uses the published `frontier/` implementation and recovers its missing public-FEN
result connection without copying the other frontier proofs or weakening them.
The earlier parser-rejection variant is likewise not overwritten.

## Evidence categories

The focused gate checks the three public laws and importing consumer plus the
closed output witness. Controls distinguish direct universal/refinement failures
from actual-implementation closed-output failures, import policies, and a synthetic
warning-output unit. Exact safe output and status zero are required. Crashes,
missing files, timeouts and malformed terms are not successful semantic rejection.

The native probe calls only actual Position/Text/Chess functions. A separate
rank-token grammar and square-array reference checks acceptance and all returned
Board/clock/history-length fields. Keys, offsets or expected results are not
supplied to the candidate. Bounded raw input Strings are the only inputs.
The four modes repeat the same fixtures; they are not exhaustive String/Board or
legal-game coverage. Static source laws, native agreement and performance remain
different claims.
