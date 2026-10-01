# Exact-list retention implies every one of the three checked values is false.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S

def all_false(+a: Bool,+b: Bool,+c: Bool,-m: Chess.Ply,-h: Chess.Ply,
  here: E.member(m,S.keep(Bool.or(Bool.or(a,b),c),h,Nil{}))) ->
  {a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}):
  match a b c:
    case False{} False{} False{}: ({==},({==},{==}))
    case False{} False{} True{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case False{} True{} False{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case False{} True{} True{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case True{} False{} False{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case True{} False{} True{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case True{} True{} False{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)
    case True{} True{} True{}: Empty.absurd({a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}),here)

def transport(-m: Chess.Ply,-a: List<&2,Chess.Ply>,-b: List<&2,Chess.Ply>,
  e: {a == b : List<&2,Chess.Ply>},p: E.member(m,a)) -> E.member(m,b):
  %e : E.member(m,_)
  p

def result(-out: Array<U64> & List<&2,Chess.Ply>,-table: Array<U64>,+a: Bool,+b: Bool,+c: Bool,-m: Chess.Ply,-h: Chess.Ply,
  e: {out == (table,S.keep(Bool.or(Bool.or(a,b),c),h,Nil{})) : Array<U64> & List<&2,Chess.Ply>},
  here: E.member(m,E.moves(out))) ->
  {a == False{} : Bool} & ({b == False{} : Bool} & {c == False{} : Bool}):
  all_false(a,b,c,m,h,transport(m,E.moves(out),S.keep(Bool.or(Bool.or(a,b),c),h,Nil{}),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,List<&2,Chess.Ply>,x => E.moves(x),out,(table,S.keep(Bool.or(Bool.or(a,b),c),h,Nil{})),e),here))
