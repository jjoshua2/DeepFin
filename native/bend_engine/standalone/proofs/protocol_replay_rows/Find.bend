# Transport a property of every full Ply to the exact first text match.
import Base
import ./Spec.bend as S
import ../castle_chain/Spec.bend as C
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def chosen(-P: Chess.Ply -> Type,equal: Bool,+m: Chess.Ply,
  +rest: Maybe<&2,Chess.Ply>,head: P(m),tail: S.selected(P,rest)) ->
  S.selected(P,Pos.chosen(equal,m,rest)):
  match equal:
    case True{}: (head,Unit{})
    case False{}: tail

def all(-P: Chess.Ply -> Type,+xs: List<&2,Chess.Ply>,+text: String,
  good: C.every(P,xs)) -> S.selected(P,Pos.find_move(xs,text)):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,+rest}:
      (head,tail)=good
      chosen(P,String.eq(Pos.move_text(m),text),m,Pos.find_move(rest,text),
        head,all(P,rest,text,tail))
