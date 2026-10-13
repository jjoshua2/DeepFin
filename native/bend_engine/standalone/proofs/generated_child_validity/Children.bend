# Carry occurrence certificates through the actual order-preserving child mapper.
import Base
import ./Post.bend as Post
import ../board/Spec.bend as B
import ../castle_chain/Spec.bend as C
import ../../../legal_probe/Chess.bend as Chess

def every(bs: List<&2,Chess.Board>) -> Type:
  match bs:
    case Nil{}: Unit
    case Con{b,rest}: {B.valid(b) == True{} : Bool} & every(rest)

def lift(+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  good: C.every(m => Post.valid(b,m),xs)) -> every(Chess.children(xs,b)):
  match xs:
    case Nil{}: Unit{}
    case Con{m,+rest}:
      (head,tail)=good
      (head,lift(b,rest,tail))

