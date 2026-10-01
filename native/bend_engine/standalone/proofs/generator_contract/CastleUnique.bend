# Complete-Ply Boolean uniqueness, derived from the exact two-wing list.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_sequence/Spec.bend as S
import ../castle_emission/Spec.bend as E
import ../castle_kings/Spec.bend as K
import ./CastleSpec.bend as P

def same(m: Chess.Ply,n: Chess.Ply) -> Bool:
  Chess.Ply{ms,md,mp,mf} = m
  Chess.Ply{ns,nd,np,nf} = n
  Bool.and(U32.is_eq(ms,ns),Bool.and(U32.is_eq(md,nd),
    Bool.and(U32.is_eq(mp,np),U32.is_eq(mf,nf))))

def absent(+m: Chess.Ply,xs: List<&2,Chess.Ply>) -> Bool:
  match xs:
    case Nil{}: True{}
    case Con{n,rest}: Bool.and(Bool.not(same(m,n)),absent(m,rest))

def unique(xs: List<&2,Chess.Ply>) -> Bool:
  match xs:
    case Nil{}: True{}
    case Con{+m,+rest}: Bool.and(absent(m,rest),unique(rest))

def rejected(+b: Chess.Board,+white: Bool,+ks: Bool) -> Bool:
  Bool.or(Bool.not(E.guard(b,ks)),Bool.or(S.checked(b,white,ks,K.Initial{}),
    Bool.or(S.checked(b,white,ks,K.Transit{}),S.checked(b,white,ks,K.Completed{}))))

def total_case(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.after_guard(ok,b,white,ks,tail) ==
    S.select(Bool.or(Bool.not(ok),Bool.or(S.checked(b,white,ks,K.Initial{}),
      Bool.or(S.checked(b,white,ks,K.Transit{}),S.checked(b,white,ks,K.Completed{})))),
      S.move(white,ks),tail) : List<&2,Chess.Ply>}:
  match ok:
    case False{}: {==}
    case True{}: {==}

def total(+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.after(b,white,ks,tail) == S.select(rejected(b,white,ks),S.move(white,ks),tail) : List<&2,Chess.Ply>}:
  total_case(E.guard(b,ks),b,white,ks,tail)

def selected(king_reject: Bool,queen_reject: Bool,white: Bool) ->
  {unique(S.select(king_reject,S.move(white,True{}),
    S.select(queen_reject,S.move(white,False{}),Nil{}))) == True{} : Bool}:
  match king_reject queen_reject white:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def expected(+b: Chess.Board,+white: Bool) ->
  {unique(P.expected(b,white)) == True{} : Bool}:
  Equal.trans(Bool,unique(P.expected(b,white)),
    unique(S.select(rejected(b,white,True{}),S.move(white,True{}),
      S.select(rejected(b,white,False{}),S.move(white,False{}),Nil{}))),True{},
    Equal.cong(List<&2,Chess.Ply>,Bool,xs => unique(xs),
      P.expected(b,white),
      S.select(rejected(b,white,True{}),S.move(white,True{}),
        S.select(rejected(b,white,False{}),S.move(white,False{}),Nil{})),
      Equal.trans(List<&2,Chess.Ply>,P.expected(b,white),
        S.select(rejected(b,white,True{}),S.move(white,True{}),P.after(b,white,False{},Nil{})),
        S.select(rejected(b,white,True{}),S.move(white,True{}),
          S.select(rejected(b,white,False{}),S.move(white,False{}),Nil{})),
        total(b,white,True{},P.after(b,white,False{},Nil{})),
        Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
          xs => S.select(rejected(b,white,True{}),S.move(white,True{}),xs),
          P.after(b,white,False{},Nil{}),
          S.select(rejected(b,white,False{}),S.move(white,False{}),Nil{}),
          total(b,white,False{},Nil{})))),
    selected(rejected(b,white,True{}),rejected(b,white,False{}),white))
