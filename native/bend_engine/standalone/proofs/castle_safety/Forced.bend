# Audit-only alternative: explicitly force flag2 through the full check.
# Other generation/filtering behavior remains imported actual Chess operations.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_chain/Spec.bend as ChainSpec
import ./Spec.bend as S

def required(+b: Chess.Board,+m: Chess.Ply,rays: U64) -> Bool:
  Bool.or(U32.is_eq(ChainSpec.flag(m),2),Chess.filter_requires(S.sensitive(b,rays),m))

def fast(xs: List<&2,Chess.Ply>,+b: Chess.Board,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  match xs:
    case Nil{}: r
    case Con{+m,rest}: fast(rest,b,rays,Chess.filter_fast_step(required(b,m,rays),b,m,r))

def blockers(+b: Chess.Board,moves: List<&2,Chess.Ply>,r: Array<U64> & U64) -> Array<U64> & List<&2,Chess.Ply>:
  (table,rays) = r
  fast(moves,b,rays,(table,Nil{}))

def checked(+b: Chess.Board,moves: List<&2,Chess.Ply>,r: Array<U64> & Bool) -> Array<U64> & List<&2,Chess.Ply>:
  (table,check) = r
  match check:
    case True{}: Chess.filter_legal(moves,b,(table,Nil{}))
    case False{}:
      sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
      blockers(b,moves,Chess.attack(4,table,sq,Chess.get_turn(b),Chess.occupied(b)))

def prepare(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,moves) = r
  checked(b,moves,Chess.in_check(table,b,Chess.get_turn(b)))

def suffix(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},r)))

def legal_moves(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  +own = Chess.color(b,Chess.get_turn(b))
  suffix(b,Chess.scan(Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{})))
