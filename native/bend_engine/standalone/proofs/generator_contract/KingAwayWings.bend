# Recreate constructor images for erased proofs; never duplicate affine storage.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Flow.bend as Flow
import ../storage/Observe.bend as O
import ./KingAwayGuard.bend as Guard

def cells(+c: O.Cells,+b: Chess.Board,+white: Bool,+sq: Nat,
  +turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(sq) : U64},
  +bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  +away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool},
  +xs: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),xs))) ==
    (O.pack(c),xs) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.castle_side(b,False{},Chess.castle_side(b,True{},(O.pack(c),xs))),
    Chess.castle_side(b,False{},(O.pack(c),xs)),(O.pack(c),xs),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
      z => Chess.castle_side(b,False{},z),Chess.castle_side(b,True{},(O.pack(c),xs)),(O.pack(c),xs),
      Flow.rejected(b,True{},O.pack(c),xs,Guard.guard_false(b,white,sq,True{},turn,one,bound,away))),
    Flow.rejected(b,False{},O.pack(c),xs,Guard.guard_false(b,white,sq,False{},turn,one,bound,away)))
