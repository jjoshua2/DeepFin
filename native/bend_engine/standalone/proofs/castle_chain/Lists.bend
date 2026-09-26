import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S
import ../u64/Words.bend as Old

def at(-P: Chess.Ply -> Type,xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  good: S.every(P,xs),here: E.member(m,xs)) -> P(m):
  match xs:
    case Nil{}: Empty.absurd(P(m),here)
    case Con{+h,t}:
      (ph,pt) = good
      match here:
        case Inl{same}:
          %Equal.sym(Chess.Ply,m,h,same) : P(_)
          ph
        case Inr{later}: at(P,t,m,pt,later)

def lift(-b: Chess.Board,xs: List<&2,Chess.Ply>,good: S.every(S.ordinary,xs)) ->
  S.every(m => S.classified(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      (ph,pt) = good
      (Inl{ph},lift(b,t,pt))

def weaken(-h: Chess.Ply,-base: List<&2,Chess.Ply>,xs: List<&2,Chess.Ply>,good: S.every(m => E.member(m,base),xs)) ->
  S.every(m => E.member(m,Con{h,base}),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,t}:
      (pm,pt) = good
      (Inr{pm},weaken(h,base,t,pt))

def self(+xs: List<&2,Chess.Ply>) -> S.every(m => E.member(m,xs),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+h,+t}: (Inl{{==}},weaken(h,t,t,self(t)))

def extract(-P: Chess.Ply -> Type,r: Array<U64> & List<&2,Chess.Ply>,+m: Chess.Ply,good: S.every(P,E.moves(r))) ->
  E.member(m,E.moves(r)) -> P(m):
  (table,xs) = r
  here => at(P,xs,m,good,here)

def exclude(+m: Chess.Ply,plain: S.ordinary(m),castle: {S.flag(m) == 2 : U32}) -> Empty:
  Old.false_true(Empty,Equal.trans(Bool,False{},U32.is_eq(S.flag(m),2),True{},
    Equal.sym(Bool,U32.is_eq(S.flag(m),2),False{},plain),
    Equal.cong(U32,Bool,f => U32.is_eq(f,2),S.flag(m),2,castle)))

def old_tail(-b: Chess.Board,-base: List<&2,Chess.Ply>,xs: List<&2,Chess.Ply>,good: S.every(m => E.member(m,base),xs)) ->
  S.every(m => E.member(m,base) | S.castle(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,t}:
      (pm,pt) = good
      (Inl{pm},old_tail(b,base,t,pt))

def castling(-b: Chess.Board,+m: Chess.Ply,tag: {S.flag(m) == 2 : U32},p: S.classified(b,m)) -> S.castle(b,m):
  match p:
    case Inl{plain}: Empty.absurd(S.castle(b,m),exclude(m,plain,tag))
    case Inr{castle}: castle
