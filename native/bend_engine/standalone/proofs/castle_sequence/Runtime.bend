# Proof/test adapter: actual single producer followed by actual full filter.
# This is not a new production generator or the optimized whole legal_moves path.
import Base
import ../../../legal_probe/Chess.bend as Chess

def finish(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,moves) = r
  Chess.filter_legal(moves,b,(table,Nil{}))

def run(table: Array<U64>,+b: Chess.Board,ks: Bool) -> Array<U64> & List<&2,Chess.Ply>:
  finish(b,Chess.castle_side(b,ks,(table,Nil{})))
