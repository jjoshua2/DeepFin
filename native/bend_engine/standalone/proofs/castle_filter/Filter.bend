import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E

# Continuations thread the actual affine table once; no cloned-array assumption.
def retain(+reject: Bool,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,-P: @-m: Chess.Ply -> Type,head: P(m),tail: S.all(P,acc)) -> S.all(P,Chess.retain_move(reject,m,acc)):
  match reject:
    case True{}: tail
    case False{}: (head,tail)

def after_then(+m: Chess.Ply,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: P(m),tail: S.all(P,acc),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_after(m,acc,r)):
  (table,+check) = r
  next((table,Chess.retain_move(check,m,acc)),retain(check,m,acc,P,head,tail))

def step_then(+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: P(m),tail: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_step(b,m,r)):
  (table,acc) = r
  after_then(m,acc,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)),P,Q,head,tail,next)

def legal_then(+xs: List<&2,Chess.Ply>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  ps: S.all(P,xs),pa: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_legal(xs,b,r)):
  match xs:
    case Nil{}: next(r,pa)
    case Con{+m,+rest}:
      (head,tail) = ps
      step_then(b,m,r,P,out => Q(Chess.filter_legal(rest,b,out)),head,pa,
        out => p => legal_then(rest,b,out,P,Q,tail,p,next))

def fast_step_then(+required: Bool,+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: P(m),tail: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_fast_step(required,b,m,r)):
  match required:
    case True{}: step_then(b,m,r,P,Q,head,tail,next)
    case False{}:
      (table,acc) = r
      next((table,Con{m,acc}),(head,tail))

def fast_then(+xs: List<&2,Chess.Ply>,+b: Chess.Board,+sensitive: U64,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  ps: S.all(P,xs),pa: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_fast(xs,b,sensitive,r)):
  match xs:
    case Nil{}: next(r,pa)
    case Con{+m,+rest}:
      (head,tail) = ps
      fast_step_then(Chess.filter_requires(sensitive,m),b,m,r,P,
        out => Q(Chess.filter_fast(rest,b,sensitive,out)),head,pa,
        out => p => fast_then(rest,b,sensitive,out,P,Q,tail,p,next))

def blockers_then(+b: Chess.Board,+xs: List<&2,Chess.Ply>,r: Array<U64> & U64,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  ps: S.all(P,xs),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_blockers(b,xs,r)):
  (table,rays) = r
  fast_then(xs,b,U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),(table,Nil{}),P,Q,ps,Unit{},next)

def checked_then(+b: Chess.Board,+xs: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  ps: S.all(P,xs),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_checked(b,xs,r)):
  (table,check) = r
  match check:
    case True{}: legal_then(xs,b,(table,Nil{}),P,Q,ps,Unit{},next)
    case False{}:
      +sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
      blockers_then(b,xs,Chess.attack(4,table,sq,Chess.get_turn(b),Chess.occupied(b)),P,Q,ps,next)

def prepare_then(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  ps: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.filter_prepare(b,r)):
  (table,xs) = r
  checked_then(b,xs,Chess.in_check(table,b,Chess.get_turn(b)),P,Q,ps,next)

def prepare(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,-P: @-m: Chess.Ply -> Type,ps: S.all(P,E.moves(r))) -> S.all(P,E.moves(Chess.filter_prepare(b,r))):
  prepare_then(b,r,P,out => S.all(P,E.moves(out)),ps,out => p => p)
