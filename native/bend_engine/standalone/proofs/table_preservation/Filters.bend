# Actual full, fast and prepared filters retain every source-table value.
import Base
import ./Core.bend as C
import ./Checks.bend as K
import ./Queries.bend as Q
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def after(+m: Chess.Ply,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_after(m,acc,r)) == C.table(Bool,r) : Array<U64>}:
  (table,check) = r
  {==}

def step(+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_step(b,m,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,+acc) = r
  C.apply(List<&2,Chess.Ply>,a => Chess.filter_step(b,m,(a,acc)),table,+c =>
    Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.filter_step(b,m,(O.pack(c),acc))),
      C.table(Bool,Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b))),O.pack(c),
      after(m,acc,Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b))),
      K.check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b))))

def full(xs: List<&2,Chess.Ply>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_legal(xs,b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  match xs:
    case Nil{}:
      (table,acc) = r
      {==}
    case Con{+m,+rest}:
      (table,+acc) = r
      C.apply(List<&2,Chess.Ply>,a => Chess.filter_legal(Con{m,rest},b,(a,acc)),table,+c =>
        Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.filter_legal(Con{m,rest},b,(O.pack(c),acc))),
          C.table(List<&2,Chess.Ply>,Chess.filter_step(b,m,(O.pack(c),acc))),O.pack(c),
          full(rest,b,Chess.filter_step(b,m,(O.pack(c),acc))),step(b,m,(O.pack(c),acc))))

def fast_step(required: Bool,+b: Chess.Board,+m: Chess.Ply,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_fast_step(required,b,m,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  match required:
    case True{}: step(b,m,r)
    case False{}:
      (table,acc) = r
      {==}

def fast(xs: List<&2,Chess.Ply>,+b: Chess.Board,+sensitive: U64,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_fast(xs,b,sensitive,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  match xs:
    case Nil{}:
      (table,acc) = r
      {==}
    case Con{+m,+rest}:
      (table,+acc) = r
      C.apply(List<&2,Chess.Ply>,a => Chess.filter_fast(Con{m,rest},b,sensitive,(a,acc)),table,+c =>
        Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.filter_fast(Con{m,rest},b,sensitive,(O.pack(c),acc))),
          C.table(List<&2,Chess.Ply>,Chess.filter_fast_step(Chess.filter_requires(sensitive,m),b,m,(O.pack(c),acc))),O.pack(c),
          fast(rest,b,sensitive,Chess.filter_fast_step(Chess.filter_requires(sensitive,m),b,m,(O.pack(c),acc))),
          fast_step(Chess.filter_requires(sensitive,m),b,m,(O.pack(c),acc))))

def blockers(+b: Chess.Board,+moves: List<&2,Chess.Ply>,r: Array<U64> & U64) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_blockers(b,moves,r)) == C.table(U64,r) : Array<U64>}:
  (table,rays) = r
  fast(moves,b,U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),(table,Nil{}))

def checked(+b: Chess.Board,+moves: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_checked(b,moves,r)) == C.table(Bool,r) : Array<U64>}:
  (table,check) = r
  match check:
    case True{}: full(moves,b,(table,Nil{}))
    case False{}:
      +sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
      C.apply(List<&2,Chess.Ply>,a => Chess.filter_checked(b,moves,(a,False{})),table,+c =>
        Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.filter_checked(b,moves,(O.pack(c),False{}))),
          C.table(U64,Chess.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b))),O.pack(c),
          blockers(b,moves,Chess.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b))),
          Q.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b))))

def prepare(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,+moves) = r
  C.apply(List<&2,Chess.Ply>,a => Chess.filter_prepare(b,(a,moves)),table,+c =>
    Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,(O.pack(c),moves))),
      C.table(Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b))),O.pack(c),
      checked(b,moves,Chess.in_check(O.pack(c),b,Chess.get_turn(b))),
      K.check(O.pack(c),b,Chess.get_turn(b))))
