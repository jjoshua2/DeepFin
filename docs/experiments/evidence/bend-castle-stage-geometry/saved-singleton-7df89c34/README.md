# Saved singleton proof implementation

All 14 primary files from the saved local castle_singleton suite at `7df89c34d03149b6ff00ebd5456ef96be2d901ee` are retained as UTF-8 bytes with a .txt suffix, alongside the original readout, publication metadata and selected execution receipts. This is an archive, not an additional active five-law suite.

The published implementation in PR #898 is castle_kings: four laws and 19 controls with explicit Boolean side/valid-turn conditions. The saved version has five laws and 18 controls and also follows raw U32 side behavior. The statement domains and implementations are not identical. Neither is silently replaced or counted twice. No formal equivalence theorem between implementations is claimed.

The saved guard-supplied singleton law is explicit rather than folded into a final-stage theorem. Saved check routing uses the original raw turn, not the child turn. Both versions retain input consistency and singleton premises; neither derives legal king counts from an arbitrary accepted FEN.

## Restore the saved proof/test suite

In a disposable checkout of exact #897 head `1b27a4d75744167fc0e365f7f331d2cd80881b21`, copy source/ into native/bend_engine/standalone/proofs/castle_singleton/, removing only the final .txt suffix. Follow the original source README with the pinned compiler. The manifest records every file hash; the saved focused report records 13 suite dependencies and 38 inherited files, all checked by this archive job. README is the fourteenth primary file, outside the focused manifest.

This archive job does not rerun the saved consumer or native tests. The original local lint failure remains a failure. The preserved 191/599 total is historical for the alternate stack, not the active published total. Original JSON objects are retained semantically; archive formatting is not claimed to match every external log byte. All 14 primary source/test files are byte-identical.

The original complete review ZIP, patch and Git bundle remain in the conversation attachments. Original publication metadata records their identities and the 31-file commit. This 20-entry selected archive does not pretend to reproduce that complete saved commit tree. Active initialized-stage composition has its own qualification; no proof pass follows merely from these integrity checks.
