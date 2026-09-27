import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S
import ../castle_stage_geometry/PROOF.bend as StageProof
import ../castle_stage_geometry/LAWS.bend as Stage

def at(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),Chess.get_turn(b)) ==
   (Init.run(d,seed,n,extra),S.check(b,white,ks,stage)) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),Chess.get_turn(b)),
    Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),Bool.to_u32(white)),
    (Init.run(d,seed,n,extra),S.check(b,white,ks,stage)),
    Equal.cong(U32,Array<U64> & Bool,side => Chess.in_check(Init.run(d,seed,n,extra),K.stage_board(b,white,ks,stage),side),
      Chess.get_turn(b),Bool.to_u32(white),turn),
    Stage.initialized_castling_stage_check_matches_geometry(d,seed,n,extra,b,white,ks,stage,
      depth,tables,full,good,turn,one,guard))
