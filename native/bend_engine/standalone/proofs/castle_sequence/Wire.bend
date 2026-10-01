# State-threading bridge. Public proofs derive all internal query certificates.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Runtime.bend as Runtime
import ./Spec.bend as S

def finish(table: Array<U64>,+m: Chess.Ply,+tail: List<&2,Chess.Ply>,+a: Bool,+c: Bool) ->
  {Chess.castle_finish(m,a,tail,(table,c)) == (table,S.select(Bool.or(a,c),m,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  match a c:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def producer(table: Array<U64>,+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+tail: List<&2,Chess.Ply>,+a: Bool,+c: Bool,
  initial: {Chess.in_check(table,b,Chess.get_turn(b)) == (table,a) : Array<U64> & Bool},
  mid: {Chess.in_check(table,Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b)) == (table,c) : Array<U64> & Bool}) ->
  {Chess.castle_checked(True{},table,b,src,dst,transit,tail) == (table,S.select(Bool.or(a,c),Chess.Ply{src,dst,0,2},tail)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(table,b,Chess.get_turn(b)),(table,a),initial) :
    {Chess.castle_start(b,src,dst,transit,tail,_) == (table,S.select(Bool.or(a,c),Chess.Ply{src,dst,0,2},tail)) : Array<U64> & List<&2,Chess.Ply>}
  %Equal.sym(Array<U64> & Bool,Chess.in_check(table,Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b)),(table,c),mid) :
    {Chess.castle_finish(Chess.Ply{src,dst,0,2},a,tail,_) == (table,S.select(Bool.or(a,c),Chess.Ply{src,dst,0,2},tail)) : Array<U64> & List<&2,Chess.Ply>}
  finish(table,Chess.Ply{src,dst,0,2},tail,a,c)

def after(table: Array<U64>,+m: Chess.Ply,+f: Bool) ->
  {Chess.filter_after(m,Nil{},(table,f)) == (table,S.select(f,m,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  match f:
    case True{}: {==}
    case False{}: {==}

def filter_one(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+a: Bool,+c: Bool,+f: Bool,
  final: {Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)) == (table,f) : Array<U64> & Bool}) ->
  {Runtime.finish(b,(table,S.select(Bool.or(a,c),m,Nil{}))) == (table,S.select(Bool.or(a,Bool.or(c,f)),m,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  match a c:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}:
      %Equal.sym(Array<U64> & Bool,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)),(table,f),final) :
        {Chess.filter_after(m,Nil{},_) == (table,S.select(f,m,Nil{})) : Array<U64> & List<&2,Chess.Ply>}
      after(table,m,f)
