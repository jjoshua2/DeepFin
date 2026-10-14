# MUST reject at reject_wrong_white: fixed white output cannot use the black model.
import Base
import ./CastleFrame.bend as C
import ./Frame.bend as F
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def reject_wrong_white(+b: Chess.Board,+ks: Bool,
  rows: {B.valid(b) == True{} : Bool},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,E.move(b,ks))),Chess.get_white(Chess.make_move(b,E.move(b,ks)))) ==
   F.model(b,E.move(b,ks),Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  C.white_actual(b,E.move(b,ks),C.guarded_clear(b,ks,rows,guard))
