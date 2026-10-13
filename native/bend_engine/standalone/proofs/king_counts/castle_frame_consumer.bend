import Base
import ./CastleFrame.bend as C
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def use_guarded(+b: Chess.Board,+ks: Bool,
  rows: {B.valid(b) == True{} : Bool},guard: {E.guard(b,ks) == True{} : Bool}) -> C.both(b,E.move(b,ks)):
  C.guarded(b,ks,rows,guard)
