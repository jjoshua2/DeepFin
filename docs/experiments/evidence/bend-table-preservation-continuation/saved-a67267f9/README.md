# Saved three-law implementation, not a second active suite

All14 primary files of the saved local `table_preservation` implementation at
`a67267f9832abb843e1230fc619533122833e718` are retained byte-for-byte in `source/`,
with only a `.txt` filename suffix. The active implementation in PR#910 has five
public laws, different helper bodies, and broader explicit raw-kind query coverage.
Neither implementation is silently overwritten or treated as formally equivalent.
The three archived laws are not added again to active totals.

The original readout, source identity, focused receipt, qualification, lint failure,
self-review, recovery verification and final identity are also retained exactly.
`selected-native-record.json` is explicitly a selection, not a complete copy of the
original native summary. The complete original review ZIP, patch and bundle remain
attached to the conversation. This selected archive does not reproduce the whole
saved28-file commit. Original local196/640 accounting is historical for that stack,
not the active#910 baseline198/640.

To restore, use a disposable checkout of exact#904 base
`1915e4f283b1e3d5daf7e26683d5a6c2ef48a121`, copy `source/` into
`native/bend_engine/standalone/proofs/table_preservation/`, and remove only the final
`.txt` suffix. Follow the original README with the pinned compiler. Do not overlay
it on the active five-law suite: the paths intentionally overlap.

Archive integrity is separate from proof execution. Any fresh restored source gate
is reported separately; there is no fresh archived-native pass unless explicitly
recorded. The original local lint failure is not relabeled successful. The archive
manifest records exact primary/evidence identities and excludes itself and this new
README to avoid self-referential hashes.
