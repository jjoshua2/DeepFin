# Full actual generator provenance for flag-two results, not semantic chess safety.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ./Spec.bend as S

law ordinary_scan_excludes_castling:
  for keys: List<&2,U32>
  for +b: Chess.Board
  for table: Array<U64>
  S.every(S.ordinary,E.moves(Chess.scan(keys,b,(table,Nil{}))))

law final_filter_preserves_member_origin:
  for +b: Chess.Board
  for table: Array<U64>
  for +xs: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for here: E.member(m,E.moves(Chess.filter_prepare(b,(table,xs))))
  E.member(m,xs)

law filtered_castling_suffix_origin:
  for +b: Chess.Board
  for table: Array<U64>
  for +xs: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for here: E.member(m,E.moves(S.suffix(b,(table,xs))))
  E.member(m,xs) | S.castle(b,m)

law legal_castle_member_has_producer:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for tag: {S.flag(m) == 2 : U32}
  S.castle(b,m)

law legal_castle_member_preserves_representation:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for good: {B.valid(b) == True{} : Bool}
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for tag: {S.flag(m) == 2 : U32}
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}
