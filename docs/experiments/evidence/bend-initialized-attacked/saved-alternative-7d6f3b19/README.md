# Preserved alternative initialized-attack proof

These are the exact 15 primary source/test files from saved local commit
7d6f3b19c3cd5d377d058c0ee2af927f9df97a63, with its original readout,
publication manifest and selected execution receipts. The source snapshot was
af9f7c6322fe724721b72b6203b86b59abe98c8c, based on PR #896.

The active initialized_attacks suite in PR #897 is a different implementation
of the same three contract domains. Neither implementation is overwritten.
The source/ files here have .txt appended to prevent duplicate gate discovery.
Mapping and SHA-256 identities are recorded in reconciliation.json. These are
archived local proofs, not three extra registered laws or an independent review.
No formal equivalence theorem between the two proof implementations is claimed.

To restore for review, copy each source/NAME.txt to
native/bend_engine/standalone/proofs/attack_composition/NAME in a disposable
checkout of exact base 68fee266244d7b6210d004a78c865a29266b76e3, then use
the commands in the archived source/README.md.txt. Verify archive-manifest.json.
Original full logs, patch and Git bundle remain in the conversation review ZIP,
SHA-256 d051d5b9f0e634958008009382b739effe801b421cdb22a2c95faf8f759b8ceb.
The unchanged local lint failure in the readout remains historical, not replaced
by this reconciliation's separate hosted result.
