# Initialized geometric result for each actual castling stage; not a safety verdict.
import Base
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

law initialized_castling_stage_check_matches_geometry:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +extra: Nat
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for +stage: K.Stage
  for depth: {d == 17n : Nat}
  for tables: {n == 128n : Nat}
  for full: {extra == 64n : Nat}
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),Bool.to_u32(white)) ==
    (Init.run(d,seed,n,extra),Geometry.attacked(K.stage_board(b,white,ks,stage),
      U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white))) : Array<U64> & Bool}
