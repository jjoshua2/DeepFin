# Actual prepared king selection

PR1030 still required a caller square q, q<64, and an equation identifying q with the king selected by production. This suite derives all three from the actual `filter_checked(False)` expression:
`U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))`.
The only presence premise is that this intersected plane is nonzero. Multiple kings and arbitrary U32 turns are allowed: actual color selects white at 1 and black otherwise.

`Selection.bound/chosen/present` compose the existing typed all-U64 ctz range and selected-bit theorems with an exact U32 reconstruction. `king_bit/owned` project the selected bit to the king and moving-color planes; the right projection is structural over both limbs. `kind` additionally uses `board.Spec.valid` and the actual decoder row to prove `Chess.piece(b,selected)==5`. This partition invariant excludes a pawn or other earlier decoder kind colliding at the selected king square. It is neither a king-count invariant nor chess legality.

`consumer.use_coverage` instantiates PR1030 at this selected index without a caller square/bound/equality. `use_filter_checked` rewrites the whole actual False-dispatch output through the separate rook/bishop exact-table-preserving unmasked-slider contracts, including the exact returned table. `use_ray` connects the derived range and equality to PR1030's actual pre-castling full-Ply candidate consumer, seven fuel, actual False flags, shared accumulator, and actual ordinary make_move occupancy. Generation and filtering tables remain distinct.

`use_required` proves that a move sourced at the actual selected king always triggers actual `filter_requires`, for arbitrary destination/promotion/flag/rays, by deriving its owned-king certificate. No board consistency or lookup contract is needed for this guard theorem.

Presence remains explicit. A consistent empty board has ctz(zero)=64; a board with only the opposite side's king also has selected square 64. Presence alone permits pawn/king collisions, where the actual decoder returns pawn 0. Concrete typed witnesses also check multiple moving kings (32 and 63 selects 32), opposite-colored lower king exclusion, black high bit 63, and noncanonical turn 2 selecting black.

The supplied-table lookup contracts remain assumptions. This suite proves neither production table-builder validity nor P2 initialization, postmove king-plane preservation, unconditional arbitrary-table coverage, attack safety, fast/full filter equivalence, full legal-move correctness, canonical turns, uniqueness, or reachability.

## Qualification

The intended base is exact PR1030 `ef601b76201f181a19217b2d17dc61825f3b4cf4`, tree `b8e92f1efb3dc6a5f55a9b74e864a8aed09f1a9e`. The new four-file suite and PR1030's portable README command correction change; both inherited Bend theorem and consumer files remain exact-base. The harness verifies every qualified file against clean HEAD Git blobs and every reused dependency against that base. It verifies all 84 pinned checker files before and after, with Bend 2.0.21+U64 `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, source fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`, Bun 1.4.2.

Run from the project root, using a fresh external report and evidence directory:

```bash
PYTHONDONTWRITEBYTECODE=1 BUN="$HOME/.bun/bin/bun" python3 -m \
  native.bend_engine.standalone.proofs.prepared_king_selection.qualify_prepared_king_selection \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /path/to/checker-tree.json \
  --report /external/fresh/qualification.json --evidence-dir /external/fresh/checks
```

Each checker call allows 86400 seconds, at most two CPUs, 6 GiB address-space/RSS, and 16 MiB per output file. Pass requires exit zero, exactly `All terms check.`, and resource compliance. Controls run only in isolated external copies. There are 18 specified controls: 11 contract-coupling mutations and seven concrete false witnesses. Contract coupling shows that the existing proof body does not fit an altered premise/conclusion; it does not independently establish semantic falsity or necessity. The concrete false witnesses mutate actual computed multi-king, black/high, noncanonical-turn, kingless-range, opposite-side, colliding-decoder, and selected-king guard facts. Rejections count only at the named typed obligation with different expected/observed types; parser/import/linearity/resource/timeout failures do not count.

Durable commands, outputs, timings, hashes, CI checkout evidence, independent review and manifests live outside the checkout at `~/chess-artifacts/deepfin-prepared-king-selection-20261004/evidence/2026-10-04/`.
