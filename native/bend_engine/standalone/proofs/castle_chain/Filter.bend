# Preserve any pointwise property through the actual full/fast filter.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S

# Dependent continuation keeps the actual array single-owned across the check.
def after(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +m: Chess.Ply,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  pm: P(m),pa: S.every(P,acc),next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_after(m,acc,r)):
  (table,check) = r
  match check:
    case True{}: next((table,acc),pa)
    case False{}: next((table,Con{m,acc}),(pm,pa))

def step(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>,
  pm: P(m),pa: S.every(P,E.moves(r)),next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_step(b,m,r)):
  (table,acc) = r
  after(P,Goal,m,acc,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)),pm,pa,next)

def fast_step(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  required: Bool,+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>,
  pm: P(m),pa: S.every(P,E.moves(r)),next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_fast_step(required,b,m,r)):
  match required:
    case True{}: step(P,Goal,b,m,r,pm,pa,next)
    case False{}:
      (table,acc) = r
      next((table,Con{m,acc}),(pm,pa))

def legal(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  xs: List<&2,Chess.Ply>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(P,xs),acc_good: S.every(P,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_legal(xs,b,r)):
  match xs:
    case Nil{}: next(r,acc_good)
    case Con{+m,+rest}:
      (pm,prest) = good
      step(P,z => Goal(Chess.filter_legal(rest,b,z)),b,m,r,pm,acc_good,
        z => pz => legal(P,Goal,rest,b,z,prest,pz,next))

def fast(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  xs: List<&2,Chess.Ply>,+b: Chess.Board,+sensitive: U64,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(P,xs),acc_good: S.every(P,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_fast(xs,b,sensitive,r)):
  match xs:
    case Nil{}: next(r,acc_good)
    case Con{+m,+rest}:
      (pm,prest) = good
      fast_step(P,z => Goal(Chess.filter_fast(rest,b,sensitive,z)),Chess.filter_requires(sensitive,m),b,m,r,pm,acc_good,
        z => pz => fast(P,Goal,rest,b,sensitive,z,prest,pz,next))

def blockers(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+xs: List<&2,Chess.Ply>,r: Array<U64> & U64,good: S.every(P,xs),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_blockers(b,xs,r)):
  (table,rays) = r
  fast(P,Goal,xs,b,U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),(table,Nil{}),good,Unit{},next)

def checked(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+xs: List<&2,Chess.Ply>,r: Array<U64> & Bool,good: S.every(P,xs),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_checked(b,xs,r)):
  (table,check) = r
  match check:
    case True{}: legal(P,Goal,xs,b,(table,Nil{}),good,Unit{},next)
    case False{}:
      sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
      blockers(P,Goal,b,xs,Chess.attack(4,table,sq,Chess.get_turn(b),Chess.occupied(b)),good,next)

def prepare(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: S.every(P,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.filter_prepare(b,r)):
  (table,+xs) = r
  checked(P,Goal,b,xs,Chess.in_check(table,b,Chess.get_turn(b)),good,next)
