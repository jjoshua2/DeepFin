# Canonical turn comes from the actual FEN side parser, not raw validation.
import Base
import ../../Position.bend as Pos
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def canonical(-t: U32) -> Type:
  {t == Bool.to_u32(U32.is_eq(t,1)) : U32}
def side_result(out: Maybe<&2,U32>) -> Type:
  match out:
    case None{}: Unit
    case Some{t}: canonical(t)
def output(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: canonical(Chess.get_turn(Pos.board(g)))

def boolean(white: Bool) -> canonical(Bool.to_u32(white)):
  match white:
    case False{}: {==}
    case True{}: {==}

def only(ok: Bool,+white: Bool) -> side_result(Text.only(ok,Bool.to_u32(white))):
  match ok:
    case False{}: Unit{}
    case True{}: boolean(white)

def side(+s: String) -> side_result(Pos.side(s)):
  only(Bool.or(String.eq(s,"w"),String.eq(s,"b")),String.eq(s,"w"))

def metadata(b: Chess.Board,+t: U32,r: U32,e: U32,good: canonical(t)) ->
  canonical(Chess.get_turn(Pos.metadata(b,t,r,e))):
  match b:
    case Chess.Board{p,n,bi,ro,q,k,w,bl,oldturn,oldrights,oldep}: good

# Match the exact Maybe binds in Position.fen, including all numeric guards.
def full_bound(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  final: Maybe<&2,U32>,good: canonical(t)) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- final
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match final:
    case None{}: Unit{}
    case Some{f}: metadata(b,t,r,e,good)

def positive(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  result: Maybe<&2,U32>,good: canonical(t)) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- result
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match result:
    case None{}: Unit{}
    case Some{+f}: full_bound(b,t,r,e,h,Text.bounded(f),good)

def number(+b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,
  result: Maybe<&2,U32>,good: canonical(t)) -> output(do Maybe<&2,Pos.Game>:
    f : U32 <- result
    f : U32 <- Text.positive(f)
    f : U32 <- Text.bounded(f)
    return Pos.Game{Pos.metadata(b,t,r,e),h,f,Nil{}}):
  match result:
    case None{}: Unit{}
    case Some{+f}: positive(b,t,r,e,h,Text.positive(f),good)

def half_bound(+b: Chess.Board,+t: U32,+r: U32,+e: U32,
  result: Maybe<&2,U32>,+full: Maybe<&2,U32>,good: canonical(t)) -> output(do Maybe<&2,Pos.Game>:
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
  good: side_result(turn)) -> output(do Maybe<&2,Pos.Game>:
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
  output(Pos.fen(pieces,turn,castle,target,half,full)):
  cases(Pos.layout_result(Pos.layout(pieces,Pos.Layout{Pos.empty(),0,7,True{}})),
    Pos.side(turn),Pos.rights(castle),Pos.ep(target),Text.number(half),Text.number(full),side(turn))
