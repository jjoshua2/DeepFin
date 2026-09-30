# D-lite scalar value: small-bank executable path

Status: source implementation and synthetic tests only. No mixed-source scalar
label job, selected-target attachment, pack, training, corpus admission or Elo
result has been run by this path.

The [small-bank adapter](../../scripts/sf_dlite_smallbank.py) accepts at most
512 uniquely keyed winners and their compact, authenticated tri-source v6
per-game `history_chain` proof lines. It rebuilds each row from the original
root FEN, 16-ply opening and complete played UCI prefix. It checks exact
winner UID/input digest, per-row full-stack SHA, side to move and rule50 clock,
then uses `gen_sf_rooted_corpus.history_for` for the Stockfish position. A
canonical tensor or history SHA alone cannot supply this move history.

The label step owns one persistent single-thread engine and closes it on every
path. It pins the binary, requests the existing strict six-man Syzygy option
profile (`Syzygy50MoveRule=true`, `SyzygyProbeLimit=6`, retained tablebase
cache), checks the observed UCI request flags, and uses Hash8 with a cold
`ucinewgame` plus serialized `readyok` for every row. Fixed d6/d8/d10 is
chosen before a run. The sidecar retains raw CP or mate, native UCI WDL,
nodes, move and **separate historical D-calibrated** WDL. Missing or malformed
scores fail the complete small-bank attempt; each row also retains its raw
UCI `info` lines, including the PV/node/WDL emission used for the score,
so an independent parser can revisit it without re-searching. They never become zero CP or a
neural-only fallback. The executable uses pinned inputs of at most 16 MiB
each, a cooperative one-hour cap, a 15-second per-search tripwire and a
64-MiB output cap. It requires a fresh one-shot output directory, writes a
claim, reopens written bytes and publishes `COMPLETE` only after all rows.
This is not an independent physical resource supervisor or attestation of
effective engine options. The claim and completion record the requested
Syzygy path/rule50/probe-limit/retention profile, qualified binary SHA and
runtime source-code SHAs.

The separate attachment step requires the *corrected and independently
audited* selected-neural target route, exact target-byte SHA, matching UID,
input digest and side-to-move identity, and a legal policy mask. It replaces
only the three float16 main `search_wdl` values with the float32 blend
`(SF_D_calibrated + 2 * selected_neural) / 3` rounded once to float16. All
1,858 policy bytes are copied exactly, and no SF policy/value auxiliary is
populated. The chosen target and route need their own source authentication;
the adapter recomputes the frozen source-independent fair BT4/Ceres route
from the UID and refuses a conflicting chosen teacher. It still cannot certify
that an upstream producer used the right model or legal-head alignment. A recently found
legal-order permutation in an untrained full-bank Ceres target readback must
be corrected before physical attachment. The full D-lite experiment also
needs a frozen matched training/arena comparison against selected-neural
control; this slice grants neither training nor playing-strength credit.

`inspect` is read-free. A future bounded physical invocation can run `label`
with pinned small winner/proof JSONL, qualified Stockfish binary and strict
Syzygy directories, then `attach` with pinned label and selected-target
JSONL. Each selected-target line supplies `uid`, `input_digest`, `pov_white`,
`teacher`, `target_sha256`, `target_hex` (float16 policy then WDL) and
`legal_mask_hex`. The operator must first extract the small winner/proof bank
from a successfully audited v6 replay receipt; source proof lines and selected
targets cannot be substituted from the old tensor-only spool. Synthetic tests
cover B/C/S ply offsets, history/identity/STM/rule50 mismatch, missing raw
scores, legal-mask and policy-byte preservation, fake-engine option profile,
ownership/failure cleanup, complete small-bank output and resume refusal.
