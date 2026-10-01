# Relate actual state-threading filter calls to a fixed-table decision fold.
import Base
import ./Spec.bend as S
import ./Pair.bend as Pair
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def after(table: Array<U64>,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,reject: Bool) ->
  {Chess.filter_after(m,acc,(table,reject)) == (table,S.retain(reject,m,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  match reject:
    case True{}: {==}
    case False{}: {==}

def step(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_step(b,m,(O.pack(c),acc)) == (O.pack(c),S.retain(S.checked(c,b,m),m,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)),(O.pack(c),S.checked(c,b,m)),Pair.check(c,Chess.make_move(b,m),Chess.get_turn(b))) :
    {Chess.filter_after(m,acc,_) == (O.pack(c),S.retain(S.checked(c,b,m),m,acc)) : Array<U64> & List<&2,Chess.Ply>}
  after(O.pack(c),m,acc,S.checked(c,b,m))

def full(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_legal(xs,b,(O.pack(c),acc)) == (O.pack(c),S.full(xs,c,b,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      %Equal.sym(Array<U64> & List<&2,Chess.Ply>,Chess.filter_step(b,m,(O.pack(c),acc)),(O.pack(c),S.retain(S.checked(c,b,m),m,acc)),step(c,b,m,acc)) :
        {Chess.filter_legal(rest,b,_) == (O.pack(c),S.full(rest,c,b,S.retain(S.checked(c,b,m),m,acc))) : Array<U64> & List<&2,Chess.Ply>}
      full(rest,c,b,S.retain(S.checked(c,b,m),m,acc))

def fast_step(required: Bool,+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(required,b,m,(O.pack(c),acc)) == (O.pack(c),S.fast_step(required,c,b,m,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  match required:
    case True{}: step(c,b,m,acc)
    case False{}: {==}

def fast(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,+sensitive: U64,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_fast(xs,b,sensitive,(O.pack(c),acc)) == (O.pack(c),S.fast(xs,c,b,sensitive,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      %Equal.sym(Array<U64> & List<&2,Chess.Ply>,Chess.filter_fast_step(Chess.filter_requires(sensitive,m),b,m,(O.pack(c),acc)),(O.pack(c),S.fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc)),fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc)) :
        {Chess.filter_fast(rest,b,sensitive,_) == (O.pack(c),S.fast(rest,c,b,sensitive,S.fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc))) : Array<U64> & List<&2,Chess.Ply>}
      fast(rest,c,b,sensitive,S.fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc))

def blockers(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+rays: U64) ->
  {Chess.filter_blockers(b,xs,(O.pack(c),rays)) == (O.pack(c),S.choose(False{},c,b,xs,rays)) : Array<U64> & List<&2,Chess.Ply>}:
  fast(xs,c,b,U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),Nil{})

def checked(check: Bool,+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>) ->
  {Chess.filter_checked(b,xs,(O.pack(c),check)) == (O.pack(c),S.choose(check,c,b,xs,S.rays(c,b))) : Array<U64> & List<&2,Chess.Ply>}:
  match check:
    case True{}: full(xs,c,b,Nil{})
    case False{}:
      %Equal.sym(Array<U64> & U64,Chess.attack(4,O.pack(c),U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)))),Chess.get_turn(b),Chess.occupied(b)),(O.pack(c),S.rays(c,b)),Pair.attack(c,b)) :
        {Chess.filter_blockers(b,xs,_) == (O.pack(c),S.choose(False{},c,b,xs,S.rays(c,b))) : Array<U64> & List<&2,Chess.Ply>}
      blockers(c,b,xs,S.rays(c,b))

def prepare(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>) ->
  {Chess.filter_prepare(b,(O.pack(c),xs)) == (O.pack(c),S.prepared(c,b,xs)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b)),(O.pack(c),S.initial(c,b)),Pair.check(c,b,Chess.get_turn(b))) :
    {Chess.filter_checked(b,xs,_) == (O.pack(c),S.prepared(c,b,xs)) : Array<U64> & List<&2,Chess.Ply>}
  checked(S.initial(c,b),c,b,xs)
