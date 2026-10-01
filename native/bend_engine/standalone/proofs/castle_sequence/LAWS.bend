# Complete table threading and exact decisions; guard/invariants are explicit.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S
import ./Runtime.bend as Runtime

law initialized_castle_producer_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  for +tail: List<&2,Chess.Ply>
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) == (Init.run(d,seed,n,extra),S.before(b,white,ks,tail)) : Array<U64> & List<&2,Chess.Ply>}

law initialized_filtered_castle_matches_three_stage_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {Runtime.run(Init.run(d,seed,n,extra),b,ks) == (Init.run(d,seed,n,extra),S.after(b,white,ks)) : Array<U64> & List<&2,Chess.Ply>}
