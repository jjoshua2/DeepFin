# Proof/test adapter: actual generation and a following query on its returned table.
import Base
import ../../../legal_probe/Chess.bend as Chess

def moves(r: Array<U64> & List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  (a,ms) = r
  ms

def attach(+ms: List<&2,Chess.Ply>,r: Array<U64> & Bool) -> Array<U64> & (List<&2,Chess.Ply> & Bool):
  (a,answer) = r
  (a,(ms,answer))

def follow(+q: Chess.Board,+side: U32,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & (List<&2,Chess.Ply> & Bool):
  (a,ms) = r
  attach(ms,Chess.in_check(a,q,side))

def run(a: Array<U64>,+b: Chess.Board,q: Chess.Board,side: U32) -> Array<U64> & (List<&2,Chess.Ply> & Bool):
  follow(q,side,Chess.legal_moves(a,b))
