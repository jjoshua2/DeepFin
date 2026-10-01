# Exact production wiring lemma; no initialization or legality assumption.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./CastleSpec.bend as P

def unfold(table: Array<U64>,+b: Chess.Board) ->
  {Chess.legal_moves(table,b) ==
    Chess.filter_prepare(b,P.prefix(table,b)) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
