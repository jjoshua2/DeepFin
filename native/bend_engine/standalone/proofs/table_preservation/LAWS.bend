# Whole-value storage identity only: no move/attack correctness inferred.
import Base
import ./Core.bend as C
import ../../../legal_probe/Chess.bend as Chess

law attack_query_preserves_table:
  for +kind: U32
  for table: Array<U64>
  for +sq: U32
  for +side: U32
  for +occ: U64
  {C.table(U64,Chess.attack(kind,table,sq,side,occ)) == table : Array<U64>}

law ordinary_scan_preserves_table:
  for +squares: List<&2,U32>
  for +b: Chess.Board
  for r: Array<U64> & List<&2,Chess.Ply>
  {C.table(List<&2,Chess.Ply>,Chess.scan(squares,b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}

law castling_side_preserves_table:
  for +b: Chess.Board
  for +king_side: Bool
  for r: Array<U64> & List<&2,Chess.Ply>
  {C.table(List<&2,Chess.Ply>,Chess.castle_side(b,king_side,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}

law prepared_filter_preserves_table:
  for +b: Chess.Board
  for r: Array<U64> & List<&2,Chess.Ply>
  {C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}

law legal_moves_preserves_table:
  for table: Array<U64>
  for +b: Chess.Board
  {C.table(List<&2,Chess.Ply>,Chess.legal_moves(table,b)) == table : Array<U64>}
