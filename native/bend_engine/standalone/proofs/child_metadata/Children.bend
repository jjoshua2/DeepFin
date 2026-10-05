import Base
import ./Post.bend as Post
import ../castle_chain/Spec.bend as C
import ../../../legal_probe/Chess.bend as Chess

def every(bs: List<&2,Chess.Board>,-white: Bool) -> Type:
  match bs:
    case Nil{}: Unit
    case Con{head,tail}: Post.ready(head,white) & every(tail,white)

def lift(+b: Chess.Board,+xs: List<&2,Chess.Ply>,+white: Bool,
  good: C.every(m => Post.child(b,m,white),xs)) ->
  every(Chess.children(xs,b),Bool.not(white)):
  match xs:
    case Nil{}: Unit{}
    case Con{m,+rest}:
      (head,tail)=good
      (head,lift(b,rest,white,tail))
