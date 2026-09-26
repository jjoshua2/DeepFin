import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E

def transport(-P: @-m: Chess.Ply -> Type,-xs: List<&2,Chess.Ply>,-ys: List<&2,Chess.Ply>,e: {xs == ys : List<&2,Chess.Ply>},p: S.all(P,xs)) -> S.all(P,ys):
  %e : S.all(P,_)
  p

def member(+xs: List<&2,Chess.Ply>,-P: @-m: Chess.Ply -> Type,-m: Chess.Ply,p: S.all(P,xs),here: E.member(m,xs)) -> P(m):
  match xs:
    case Nil{}: Empty.absurd(P(m),here)
    case Con{h,rest}:
      (head,tail) = p
      match here:
        case Inl{same}:
          %Equal.sym(Chess.Ply,m,h,same) : P(_)
          head
        case Inr{later}: member(rest,P,m,tail,later)

def weaken(+xs: List<&2,Chess.Ply>,-head: Chess.Ply,-tail: List<&2,Chess.Ply>,p: S.all(m => E.member(m,tail),xs)) -> S.all(m => E.member(m,Con{head,tail}),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,rest}:
      (here,later) = p
      (Inr{here},weaken(rest,head,tail,later))

def reflexive(+xs: List<&2,Chess.Ply>) -> S.all(m => E.member(m,xs),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+h,+rest}:
      (Inl{{==}},weaken(rest,h,rest,reflexive(rest)))

def as_origin(+xs: List<&2,Chess.Ply>,-b: Chess.Board,-tail: List<&2,Chess.Ply>,p: S.all(m => E.member(m,tail),xs)) -> S.all(m => S.origin(b,tail,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,rest}:
      (head,later) = p
      (Inl{head},as_origin(rest,b,tail,later))

def as_classified(+xs: List<&2,Chess.Ply>,-b: Chess.Board,p: S.all(S.not_castle,xs)) -> S.all(m => S.classified(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,rest}:
      (head,later) = p
      (Inl{head},as_classified(rest,b,later))
