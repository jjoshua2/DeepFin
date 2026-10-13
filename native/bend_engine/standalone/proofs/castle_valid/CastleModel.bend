import Base
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def model(white: Bool,+ks: Bool) -> Chess.Ply:
  +home=Bool.pick(U32,white,0,56)
  Chess.Ply{U32.add(home,4),U32.add(home,Bool.pick(U32,ks,6,2)),0,2}
def move_same(+b: Chess.Board,+ks: Bool,+white: Bool,same: {U32.is_eq(Chess.get_turn(b),1) == white : Bool}) ->
  {E.move(b,ks) == model(white,ks) : Chess.Ply}:
  Equal.cong(Bool,Chess.Ply,x => model(x,ks),U32.is_eq(Chess.get_turn(b),1),white,same)
