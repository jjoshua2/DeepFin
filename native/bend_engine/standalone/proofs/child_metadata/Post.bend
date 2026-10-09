import Base
import ./Range.bend as R
import ./Source.bend as Source
import ../board/Spec.bend as B
import ../castle_kings/Stages.bend as Turn
import ../generated_child_validity/Post.bend as Prior
import ../castle_chain/Spec.bend as C
import ../../../legal_probe/Chess.bend as Chess

def ready(-b: Chess.Board,-white: Bool) -> Type:
  {B.valid(b) == True{} : Bool} &
  ({Chess.get_turn(b) == Bool.to_u32(white) : U32} & R.ep(Chess.get_ep(b)))

def child(-b: Chess.Board,-m: Chess.Ply,-white: Bool) -> Type:
  ready(Chess.make_move(b,m),Bool.not(white))

def generated(+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  good: {B.valid(b) == True{} : Bool},canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  fields: Source.generated(b,m)) -> child(b,m,white):
  (bound,shape)=fields
  (Prior.generated(b,m,good,shape),(Turn.flip(b,m,white,canonical),R.post(b,m,bound)))

def occurrences(+b: Chess.Board,+xs: List<&2,Chess.Ply>,+white: Bool,
  +good: {B.valid(b) == True{} : Bool},+canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  fields: C.every(m => Source.generated(b,m),xs)) -> C.every(m => child(b,m,white),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,+rest}:
      (head,tail)=fields
      (generated(b,m,white,good,canonical,head),occurrences(b,rest,white,good,canonical,tail))
