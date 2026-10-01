import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def use_stage(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),Bool.to_u32(white)) ==
    (Init.run(d,seed,n,extra),Geometry.attacked(K.stage_board(b,white,ks,stage),
      U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white))) : Array<U64> & Bool}:
  Laws.initialized_castling_stage_check_matches_geometry(d,seed,n,extra,b,white,ks,stage,
    depth,tables,full,good,turn,one,guard)
