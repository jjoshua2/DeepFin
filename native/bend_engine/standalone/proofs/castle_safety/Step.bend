import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

def blockers(+b: Chess.Board,+moves: List<&2,Chess.Ply>,table: Array<U64>,+rays: U64) ->
  {Chess.filter_blockers(b,moves,(table,rays)) == Chess.filter_fast(moves,b,S.sensitive(b,rays),(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def checked(+b: Chess.Board,+m: Chess.Ply,+rays: U64,table: Array<U64>,+acc: List<&2,Chess.Ply>,
  e: {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,(table,acc)) ==
    Chess.filter_after(m,acc,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b))) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Bool,Chess.filter_requires(S.sensitive(b,rays),m),True{},e) :
    {Chess.filter_fast_step(_,b,m,(table,acc)) == Chess.filter_after(m,acc,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b))) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def rejected(+m: Chess.Ply,+acc: List<&2,Chess.Ply>,table: Array<U64>) ->
  {Chess.filter_after(m,acc,(table,True{})) == (table,acc) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
