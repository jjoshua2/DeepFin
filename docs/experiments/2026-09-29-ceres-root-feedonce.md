# Ceres stored-feed-once root preparation

The Ceres source actor built each root's 137-byte feed twice: once from its
full-history board and once from the same float16 tensor saved for the student.
The per-root comparison was a diagnostic duplicate. The frozen full-prefix
reader separately reconstructs direct-board feeds on accepted and discarded
attempts. This change exposes a tracked root-input binder that constructs the
actor feed once from the stored tensor. The tracked saved-game producer/readback
calls the binder after its outcome check; it independently re-encodes the full
history board tensor, compares that tensor to the archived float16 input, and
checks the stored-derived feed hash. Its preexisting readback was also
stored-derived. Strict six-man WDL/DTZ/rule50 adjudication, full-history
snapshots, legal maps and physical archive proof remain in place.

The archived control/candidate actor screen used the 19-game, 2,022-row saved
source ZIP with SHA-256 `09bd17dd73a7adaefc9c2fbde289174eceba365c835b50cd9a069f4c15d94e67`.
Both produced the same ordered root receipt SHA-256
`62161ded5a834349ba577bbad28208ec0aac57ee9c926b8fda577dd7096e4524`.
The candidate made zero direct-board feed constructions in the actor loop
versus 2,022 for control. Serial CPU A/B/B/A saved-root replay averaged
2.940726 seconds for control and 2.446278 seconds for candidate, a 16.81%
reduction in that narrow replay loop. The tracked binder separately reproduced
all 2,022 saved row feed and stored-input hashes, the ordered root receipt,
and direct-board feed bytes for every root. It also exercises the public
saved-game writer/readback in focused tests.

After removing the duplicate, a bounded timer on the same saved roots measured
0.137 seconds for full-stack snapshots, 0.414 seconds for strict outcomes
(0.372 seconds of that in threefold-claim checks), 0.098 seconds for compact
and Leela legal maps, 0.425 seconds for two position fingerprints, and
0.828 seconds for stored-feed conversion. These are nested CPU component times
with timer overhead, not an additive source wall decomposition. A separate
CPU A/B/B/A screen of the existing converter on the 2,022 stored tensors
found exact feed/fingerprint parity, including the six-row tail: per-row
conversion plus fingerprint averaged 0.698 seconds; groups of 32 averaged
0.214 seconds. This suggests batching prepared roots as the next actor change.
It does not establish integrated source throughput.

The tracked binder validates the `lc0_root_legacy_meta`, `v2_threats`,
`history_rep_fix=True` profile and live/stored position fingerprints. The
caller remains responsible for encoding its supplied float32 tensor from the
full-history board, applying the strict outcome gate before the binder, and
independent whole-game readback. The fresh chunked producer has a concrete
CPU-tested adapter but has not yet pinned this tracked binder or run a GPU
qualification. This readout claims no new generated or corpus-admitted rows,
no live throughput gain and no 500M completion estimate.

The supporting CPU artifacts were private source-host review inputs rather than
portable repository artifacts: `BENCHMARK.json`, the independent source review,
the saved32 proof, and the batch-conversion screen. Their former machine-specific
temporary paths are deliberately not published here. The durable public claim is
bounded by the pinned source/test identities, archived source ZIP and ordered-root
receipt above; this record does not imply those private artifacts are retrievable
from another checkout.
