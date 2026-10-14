import Base
import ./CastleCount.bend as C
import ./Actual.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def use_castle(+b: Chess.Board,+ks: Bool,
  rows: {B.valid(b) == True{} : Bool},guard: {E.guard(b,ks) == True{} : Bool}) ->
  K.preserved(b,E.move(b,ks)):
  C.preserved(b,ks,rows,guard)
