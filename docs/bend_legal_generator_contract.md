# Complete Bend legal-generator contract

Status: **target specification and qualification work, not a completed proof**.
The implementation under discussion is the actual
[`Chess.legal_moves`](../native/bend_engine/legal_probe/Chess.bend), including its
optimized unchecked branch, called with actual
[`Tables.build()`](../native/bend_engine/standalone/Tables.bend).
Training may remain in Python. This contract is one correctness boundary of the
complete Bend UCI engine; UCI, search, model execution and native qualification
have separate acceptance conditions.

## The top-level claim

For every finite independently valid orthodox move history H and every concrete
Board B representing its last state, evaluate

```text
(T, moves) = Chess.legal_moves(Tables.build(), B)
```

The requested theorem establishes all of the following together:

1. **Soundness:** every occurrence m in moves decodes to an independently legal
   move from the last state of H.
2. **Completeness:** every independently legal move from that state has an
   occurrence in moves with its unique canonical encoding.
3. **No duplicates:** that canonical encoding occurs exactly once. Equality is
   the complete (source, destination, promotion, flag) tuple; distinct promotion
   choices remain distinct moves. Soundness also excludes alternate encodings
   of the same semantic move.
4. **State threading:** T equals the complete array returned by Tables.build(),
   including unused slots. The next call can reuse T without assuming that it
   remains initialized.
5. **Successor representation:** applying actual Chess.make_move(B,m) to any
   returned m represents the independent successor, including the next side,
   lost castling rights and en-passant metadata. The independent history can be
   extended by that move, so validity is available for the next invocation.

Equivalently, canonical-encoding multiplicity in moves is one exactly for legal
moves and zero otherwise, with the two state equalities above. List order is not
specified. Proving equality to an audit implementation alone does not establish
these obligations.

These are source-level statements for the pinned checker and Base. No axiom,
caller-supplied move list, initialized-table answer, per-move safety certificate,
correctness assumption about the optimized branch, or production-generator-based
definition of legal history may discharge them.

## Independent state and history

The semantic anchor remains the repository's pinned [English FIDE Laws applied
from 1 January 2023](https://handbook.fide.com/chapter/e012023), specifically the
move rules in Article 3 and the initial position. This is a fixed edition, not a
claim about whichever edition is current.
This is orthodox chess, not Chess960. The independent specification must use
bounded file/rank coordinates and a 64-cell piece map, not attack-table lookups
or Chess.legal_moves.

A semantic state contains the cell map, side to move, four historical castling
entitlements and the immediately preceding move needed for en passant.
H starts at the orthodox initial position and each later state is produced by
the independent legal transition relation below. Thus history validity is
defined before, and independently of, the implementation theorem.

Represent(B,last(H)) requires:

- Each of the 64 cells corresponds exactly to one of the six piece-kind bits
  and exactly one color bit, or to no kind/color bits for an empty square
- B.turn is exactly 1 for White or 0 for Black
- B.rights has no bits outside K=1,Q=2,k=4,q=8 and equals the four entitlements
- B.ep is 64 unless the immediately previous move was the opponent's legal
  initial two-square pawn advance, in which case it equals the crossed square,
  even if no current pawn can capture there

Reachability supplies one king per side, no unpromoted pawn on the first/eighth
rank, original-king/rook provenance for remaining rights, and safety of the side
that just moved. These consequences must be derived where used, not added as
unexplained output premises. A promoted rook or a rook returning to its corner
does not recreate castling entitlement. No-rights and no-en-passant states remain
ordinary cases, not excluded domains.

This theorem deliberately assumes a represented independent history. It does
not certify arbitrary raw Boards, malformed FEN, parser correctness, fabricated
rights or fabricated en-passant metadata. A later parser/history theorem can
establish its premises for UCI inputs. A separate local-validity generalization
may widen the domain, but must state exactly which history obligations it replaces.

## Independent legal transition

A move has bounded, different source/destination squares, an owned source piece,
no friendly target and no capture of the opposing king.

- Rook, bishop and queen motion follows rank/file/diagonal alignment with every
  strict intervening square empty
- Knight motion has absolute coordinate differences (1,2) or (2,1)
- A king's ordinary move has maximum absolute coordinate difference one
- A pawn moves one forward into an empty square, or two from its initial rank
  with both intermediate and target squares empty; a diagonal move captures an
  opposing non-king piece
- A pawn arriving on its final rank must promote to exactly knight, bishop,
  rook or queen; other moves have no promotion
- En passant requires a diagonal one-step pawn move onto the empty crossed
  square of the immediately preceding opposing two-square pawn advance. The
  victim is removed from its actual landing square before king safety is tested
- Castling requires the proper original home king and corner rook, the
  historical entitlement, and all intervening squares empty. The initial king
  square, transit square with the old king square vacated, and destination in
  the completed king-and-rook position must be unattacked by the opposing side

For every move class, independent simultaneous piece-map update produces the
child; the original moving side's king must be unattacked there. Attacks use
piece geometry and blockers, including opposing king adjacency and pawn capture
directions. They are attacks even when the attacking piece cannot legally move
because it would expose its own king. This is not recursively defined by legal
move generation.

The transition flips side, removes rights after the relevant king/rook move or
original-rook capture, and updates en-passant metadata as above. The machine
encoding uses promotion 0 or piece IDs 1..4, and flag 0 for ordinary/promotions,
1 for en passant, 2 for castling. Those tags and bounds must be proved for outputs.

Draw claims, automatic draws, clocks, resignation and game termination do not
prune this move relation. They are separate game/controller rules; standard
legal-move generation and perft can still describe movement options on a drawn
position. Checkmate/stalemate correspond to absence of legal moves plus the
independent current-check predicate. No perft count or finite test is this theorem.

## Three compositional milestones

### 1. Actual initialization and complete castling contract

Close the checked Tables.build boundary, compose initialized forward attack
semantics, then prove the exact castling subset of the actual full optimized
generator: both wings, all three safety stages, canonical encoding, completeness
and multiplicity at most one. Historical rights are supplied by Represent and
history, not invented by a guard. Keep ordinary moves in the actual scan/filter
pipeline when proving castling membership.

Existing pieces: builder/initialized_attacks, attack_reversal/slider_reversal,
castle_sequence/castle_checked_pipeline, castle_chain/castle_safety,
table_preservation and accepted_castle. The latter proves accepted destination
safety; it does not yet prove this complete milestone.

### 2. Ordinary, promotion and en-passant contract

Connect actual bit_squares, destinations, scan, put_move and piece_targets to
independent candidate geometry with exact coverage, bounded indices and
nonduplication. Prove all child representation/history updates. Establish the
optimized skip-check theorem: when not initially in check, a generated ordinary
non-king non-EP move whose source is outside the first-blocker ray set preserves
king safety. This must derive safety from board/move geometry; assuming the
child is safe or merely testing an audit generator is insufficient.

Compose full-check and skipped-check paths to obtain soundness, completeness
and uniqueness for these move classes. Preserve actual table threading and
all special-move interactions, including EP discoveries and promotion captures.

### 3. Whole generator and induction-ready consumer

Combine the disjoint move classes into the single top-level theorem with actual
Tables.build and optimized legal_moves. The importing consumer must request all
five obligations with only independent history and representation premises.
Run a complete dependency qualification on the final integrated sources, retain
semantic corruption controls and bounded independent native references, and
obtain an independent review. Modular historical counts are not a substitute
for this final composition.

This is a bounded set of finish lines, not a promise that the current backend
can normalize every needed proposition. If a blocker is reproduced, record the
exact theorem, source/checker identities, command and observed failure. Do not
weaken the target or call the target complete because a smaller lemma passes.

## Existing boundary and current first attempt

The qualified symbolic initialized recipe is
Init.run(depth,seed,blocks,extras), with depth=17, blocks=128 and extras=64.
The production public wrapper is presently connected by an exact source-token
guard and native mutation test. A direct closed equality previously exceeded
30 seconds; the guard is not that equality.

The new [closed-builder candidate](../native/bend_engine/standalone/proofs/generator_contract/Closed.bend)
and its importing consumer test the exact equality on the unchanged compiler
aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae. This candidate has no accepted-law
credit until its complete fail-closed source/control gate succeeds. Even then,
all remaining milestones above stay open.

The first hosted run exceeded the 600-second positive-consumer bound with no
checker output. Its [dated record](experiments/2026-10-01-bend-generator-contract.md)
retains the exact identities and NOT_COMPLETED outcome. Zero accepted laws and
zero executed semantic controls are credited. The limitation is being diagnosed;
it does not show that the equality is false.

## Residual trust and completion language

Trusted components remain the pinned checker/normalizer and Base. A completed
source theorem does not verify C lowering, ownership/lifetime, allocation
success, ABI, native compiler, libraries, operating system or hardware. Actual
compiled-engine differential, corruption and sanitizer tests remain separate
evidence, as do the trained-model/GPU/UCI qualification track.

The proof does not establish playing strength, model quality, performance,
search correctness, complete UCI behavior, Python-free training, or freedom from
all engine bugs. Final reports must name the actual theorem, premises, source
head, passing checks, independent review and residual trusted components.

## Qualified both-wing component, 1 October 2026

The [castling projection composition](../native/bend_engine/standalone/proofs/generator_contract/CASTLING.md)
now has a complete passing source/control gate at a2824a0d586a1ca17bb279538866872edb4e03df
([run 36814349505](https://github.com/jjoshua2/DeepFin/actions/runs/36814349505)).
It proves exact whole-array and tag-2 output-list equality for actual optimized
legal_moves, plus complete-Ply Boolean uniqueness. Both wing guards and the three
independent target-centred geometry checks determine the canonical output list.
The ordinary scan remains in the actual implementation before projection.

The premises remain symbolic initialized construction, valid Board, Boolean side
and a home-king singleton. Historical entitlement and derivation of these premises,
general king-away positions, forward/reverse geometry composition and the closed
public-builder bridge remain separate. This is genuine component closure inside
milestone 1, not completion of that milestone or of the five-part top-level claim.
