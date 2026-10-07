# Produce the remaining row certificate from the actual initialized parser.
import Base
import ../frontier/consumer.bend as Frontier
import ../frontier/Spec.bend as FS
import ../board/Spec.bend as B
import ../board/Actual.bend as Board
import ../fen_validated_full/Root.bend as Root
import ../../Position.bend as Pos
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def layout_output(out: Maybe<&2,Chess.Board>) -> Type:
  match out:
    case None{}: Unit
    case Some{b}: {B.valid(b) == True{} : Bool}

def layout_certificate(+out: Maybe<&2,Chess.Board>,
  good: {FS.result_valid(out) == True{} : Bool}) -> layout_output(out):
  match out:
    case None{}: Unit{}
    case Some{b}: good

def output(out: Maybe<&2,Pos.Game>) -> Type:
  Root.rows(out)

def metadata(+b: Chess.Board,+t: U32,r: U32,e: U32,
  good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Pos.metadata(b,t,r,e)) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Pos.metadata(b,t,r,e)),B.valid(b),True{},
    Board.metadata(b,t,r,e),good)

# These are the exact Maybe binds and numeric guards of Position.fen.
def full_bound(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  final: Maybe<&2,U32>,good: {B.valid(b) == True{} : Bool}) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- final
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match final:
    case None{}: Unit{}
    case Some{f}: metadata(b,t,r,e,good)

def positive(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  result: Maybe<&2,U32>,good: {B.valid(b) == True{} : Bool}) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- result
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match result:
    case None{}: Unit{}
    case Some{+f}: full_bound(b,t,r,e,h,Text.bounded(f),good)

def number(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  result: Maybe<&2,U32>,good: {B.valid(b) == True{} : Bool}) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- result
    f : U32 <- Text.positive(f)
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match result:
    case None{}: Unit{}
    case Some{+f}: positive(b,t,r,e,h,Text.positive(f),good)

def half_bound(+b: Chess.Board,+t: U32,+r: U32,+e: U32,
  result: Maybe<&2,U32>,+full: Maybe<&2,U32>,good: {B.valid(b) == True{} : Bool}) -> output(do Maybe<&2,Pos.Game>:
    h : U32 <- result
    f : U32 <- full
    f : U32 <- Text.positive(f)
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match result:
    case None{}: Unit{}
    case Some{+h}: number(b,t,r,e,h,full,good)

def cases(+layout: Maybe<&2,Chess.Board>,+turn: Maybe<&2,U32>,
  +rights: Maybe<&2,U32>,+ep: Maybe<&2,U32>,+half: Maybe<&2,U32>,+full: Maybe<&2,U32>,
  good: layout_output(layout)) -> output(do Maybe<&2,Pos.Game>:
    b : Chess.Board <- layout
    t : U32 <- turn
    r : U32 <- rights
    e : U32 <- ep
    h : U32 <- half
    h : U32 <- Text.bounded(h)
    f : U32 <- full
    f : U32 <- Text.positive(f)
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match layout:
    case None{}: Unit{}
    case Some{+b}:
      match turn:
        case None{}: Unit{}
        case Some{+t}:
          match rights:
            case None{}: Unit{}
            case Some{+r}:
              match ep:
                case None{}: Unit{}
                case Some{+e}:
                  match half:
                    case None{}: Unit{}
                    case Some{+h}:
                      half_bound(b,t,r,e,Text.bounded(h),full,good)

def fen(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  Root.rows(Pos.fen(pieces,turn,castle,target,half,full)):
  cases(Pos.layout_result(Pos.layout(pieces,Pos.Layout{Pos.empty(),0,7,True{}})),
    Pos.side(turn),Pos.rights(castle),Pos.ep(target),Text.number(half),Text.number(full),
    layout_certificate(Pos.layout_result(Pos.layout(pieces,Pos.Layout{Pos.empty(),0,7,True{}})),
      Frontier.arbitrary_placement(pieces)))
