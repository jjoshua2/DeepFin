import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Core.bend as C
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def use_attack_query_preserves_table(+kind: U32,table: Array<U64>,+sq: U32,+side: U32,+occ: U64) ->
  {C.table(U64,Chess.attack(kind,table,sq,side,occ)) == table : Array<U64>}:
  Laws.attack_query_preserves_table(kind,table,sq,side,occ)

def use_ordinary_scan_preserves_table(+squares: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.scan(squares,b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Laws.ordinary_scan_preserves_table(squares,b,r)

def use_castling_side_preserves_table(+b: Chess.Board,+king_side: Bool,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_side(b,king_side,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Laws.castling_side_preserves_table(b,king_side,r)

def use_prepared_filter_preserves_table(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Laws.prepared_filter_preserves_table(b,r)

def use_legal_moves_preserves_table(table: Array<U64>,+b: Chess.Board) ->
  {C.table(List<&2,Chess.Ply>,Chess.legal_moves(table,b)) == table : Array<U64>}:
  Laws.legal_moves_preserves_table(table,b)

# This nonuniform source tree is deliberately not a complete initialized chess table.
# Legal-move correctness is not asserted for it; table identity still holds.
def ragged(+x: U64,+y: U64,+z: U64,+b: Chess.Board) ->
  {C.table(List<&2,Chess.Ply>,Chess.legal_moves(O.pack(O.Branch{O.Cell{x},O.Branch{O.Cell{y},O.Cell{z}}}),b)) ==
    O.pack(O.Branch{O.Cell{x},O.Branch{O.Cell{y},O.Cell{z}}}) : Array<U64>}:
  Laws.legal_moves_preserves_table(O.pack(O.Branch{O.Cell{x},O.Branch{O.Cell{y},O.Cell{z}}}),b)
