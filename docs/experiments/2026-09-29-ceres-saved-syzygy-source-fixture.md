# Tracked Ceres saved Syzygy fixture, September 29, 2026

The tracked saved-game producer now accepts completed `rule50_match_v1` Syzygy
games through the same pinned ZIP, replay and unqualified publication path used
for natural games. The CLI requires the exact saved Syzygy path pair and opens
an owned strict six-man handle. The public readback CLI opens a fresh handle;
neither command accepts the saved result merely because it appears in metadata.

For every saved root, replay checks the 175 stored planes, 137 feed bytes,
history identity and legal maps, then asks the main BT4 outcome policy whether
the game would already have ended. At the terminal board it requires a Syzygy
decision with the recorded result and detail. The main policy probes both WDL
and DTZ, treats cursed wins and blessed losses as draws, and rejects an
unresolved positive-clock decisive WDL. Missing eligible WDL or DTZ fails the
run. The saved path, six-man profile and table counts must match the opened
handle. The source ZIP remains physically SHA-pinned and the output ZIP is
fully reopened before and after no-replace publication and again by the
separate public readback command.

Two actual saved adjudicated games from source ZIP SHA-256
`09bd17dd73a7adaefc9c2fbde289174eceba365c835b50cd9a069f4c15d94e67`
passed producer and separate readback under the local 3–6-man Syzygy pair:

| Game | Rows | Reproved result | Output ZIP SHA-256 |
| --- | ---: | --- | --- |
| 29 | 115 | `0-1` | `e35c2b156289693e8efe3d3646ceff5f24dbf6e6bbaea19420f8cec7c01e7865` |
| 12 | 118 | `1-0` | `f049acd7e2c588823bf41ef890b47300227be0ed5407a301961cb6ad4ae97a67` |

Changing the actual saved game 29 result to `1-0` was rejected before an output
directory was created. Focused synthetic tests also reject a reversed winner,
a cursed-win draw mislabeled decisive, an unresolved positive-clock position,
missing DTZ, and a wrong Syzygy path; a strict Syzygy fixture cannot be read
back without its tablebase handle. The source games' saved
`outcome_reproven_from_replay=false` provenance field is retained verbatim;
this tracked validation is demonstrated by the producer and independent
readback, not retroactively attributed to the legacy source.

This still grants zero generated-row or corpus credit. It does not run actor
inference, retime the rolling scheduler, or establish sustained generation
throughput. The result is a real CPU publication and readback boundary for
saved strict six-man games; source selection and tablebase availability remain
explicit prerequisites.
