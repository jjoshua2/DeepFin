# Equal complete ownership-threaded results, not just Boolean route metadata.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S
import ./Required.bend as R

def step(+required: Bool,+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>,e: {required == True{} : Bool}) ->
  {Chess.filter_fast_step(required,b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Bool,required,True{},e) : {Chess.filter_fast_step(_,b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def certified_step(+b: Chess.Board,+m: Chess.Ply,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>,p: C.castle(b,m)) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r,R.certificate(b,m,rays,p))

def all(+xs: List<&2,Chess.Ply>,+b: Chess.Board,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>,good: C.every(m => C.castle(b,m),xs)) ->
  {Chess.filter_fast(xs,b,S.sensitive(b,rays),r) == Chess.filter_legal(xs,b,r) : Array<U64> & List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      (pm,pt) = good
      %Equal.sym(Bool,Chess.filter_requires(S.sensitive(b,rays),m),True{},R.certificate(b,m,rays,pm)) :
        {Chess.filter_fast(rest,b,S.sensitive(b,rays),Chess.filter_fast_step(_,b,m,r)) == Chess.filter_legal(Con{m,rest},b,r) : Array<U64> & List<&2,Chess.Ply>}
      all(rest,b,rays,Chess.filter_step(b,m,r),pt)

def blockers(+b: Chess.Board,+xs: List<&2,Chess.Ply>,table: Array<U64>,+rays: U64,good: C.every(m => C.castle(b,m),xs)) ->
  {Chess.filter_blockers(b,xs,(table,rays)) == Chess.filter_legal(xs,b,(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  all(xs,b,rays,(table,Nil{}),good)
