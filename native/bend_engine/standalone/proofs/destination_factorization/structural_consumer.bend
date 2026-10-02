import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Factor.bend as F

def use_structural(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,
  +keys: List<&2,U32>,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,F.emit(keys,src,pawn,ep,tail)) ==
    F.emit(Chess.bit_squares(n,empty,bb,keys),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  F.structural(n,empty,bb,src,pawn,ep,keys,tail)
