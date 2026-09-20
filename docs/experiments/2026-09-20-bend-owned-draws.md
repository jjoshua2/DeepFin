# Bend-owned history rules and automatic search draws

## Preregistration

Base: #799, `496284b233de9cc3727ddd4b79dc5720d6ed945e`.
Keep today's source-verified U64 compiler `fd1df81707fd758f749a9570ccb5b12b1bb2fea3`.
Question: can the standalone executable own history-based rule decisions, rather
than querying Python, while retaining its directly launched no-interpreter runtime?

Implement automatic fivefold/75-move and a conservative insufficient-material
subset in Bend. Use full piece/color/turn/castling identity and only legally
available en-passant rights. Reconstruct each selected search leaf from the
Bend-owned played history and actual tree ancestors; validate its full board.
Use the existing terminal-zero reply and cache. Mate/stalemate remain the native
search's first terminal checks. Do not force optional threefold/fifty-move claims.
No compiler, search-core, C transport, neural model or production changes.

Deciding checks, before running new runtime tests: compare current rule facts and
complete unchanged histories with external python-chess across clock boundaries,
repetition histories, lost rights, legal/pinned EP, material and terminal cases.
Test rules below root, cached revisits, identical-board/historyless controls,
invalid-command transactionality, rewinds, and unchanged perft semantics. Preserve
the original standalone verifier unmodified. Run generic/portable/native/UBSan and
static-only empty chroot; only external tests use Python. Do not claim exhaustive
dead-position solving, claim-choice migration, neural inference or performance.

Budget: isolated CPU development, at most two compiler processes, one runtime
thread, one bounded hosted confirmation (10 minutes). Additional runs only repair
concrete failures. Native tests are opt-in; no recurring perft-depth/test expansion.
No live process, training, GPU, merge or deployment. Recovery: discard isolated
branch. Self-review; independent review is not currently available.

Rules source: FIDE Laws articles 9.2.3 and 9.6 (and 5.2 for terminal outcomes),
https://handbook.fide.com/chapter/E012023 . The implemented material detector is
only a sufficient subset of the general dead-position rule.

## Readout

Pending execution. Store actual source/build identities and observed coverage,
including limitations and failures, before publishing the feature.
