# Derive the exact selected-king premise consumed by initialized attack semantics.
import Base
import ../castle_kings/Spec.bend as K
import ../castle_kings/Check.bend as Check
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def singleton(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage,
  good: {B.valid(b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},
  guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.and(Chess.get_kings(K.stage_board(b,white,ks,stage)),
    Chess.color(K.stage_board(b,white,ks,stage),Bool.to_u32(white))) ==
    U64.bit(U32.to_nat(K.stage_square(white,ks,stage))) : U64}:
  Equal.trans(U64,
    U64.and(Chess.get_kings(K.stage_board(b,white,ks,stage)),
      Chess.color(K.stage_board(b,white,ks,stage),Bool.to_u32(white))),
    K.kings(K.stage_board(b,white,ks,stage),white),
    U64.bit(U32.to_nat(K.stage_square(white,ks,stage))),
    Check.color(K.stage_board(b,white,ks,stage),white),
    Check.single(b,white,ks,stage,good,turn,one,guard))
