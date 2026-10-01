# Exact actual reverse-pawn query routing; no table-content or output assumption.
import Base
import ../../../legal_probe/Chess.bend as Chess

def route(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) -> Array<U64> & Bool:
  Chess.attacked_pawn(b,sq,by,Chess.attack(0,table,sq,U32.xor(by,1),Chess.occupied(b)))

def unfold(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) ->
  {Chess.attacked(table,b,sq,by) == route(table,b,sq,by) : Array<U64> & Bool}:
  {==}
