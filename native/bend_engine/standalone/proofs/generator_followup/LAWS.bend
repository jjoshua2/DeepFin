# Follow-up query semantics after complete actual generation, not move-set safety.
import Base
import ./Runtime.bend as Run
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

law generation_then_check_matches_direct_check:
  for a: Array<U64>
  for +b: Chess.Board
  for +q: Chess.Board
  for +side: U32
  {Run.run(a,b,q,side) == Run.attach(Run.moves(Chess.legal_moves(a,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}

law initialized_generation_then_check_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +q: Chess.Board
  for +sq: Nat
  for +side: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  for one: {U64.and(Chess.get_kings(q),Chess.color(q,Bool.to_u32(side))) == U64.bit(sq) : U64}
  {Run.run(Init.run(d,seed,n,extra),b,q,Bool.to_u32(side)) ==
    (Init.run(d,seed,n,extra),(Run.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),Geometry.attacked(q,sq,Bool.not(side)))) : Array<U64> & (List<&2,Chess.Ply> & Bool)}
