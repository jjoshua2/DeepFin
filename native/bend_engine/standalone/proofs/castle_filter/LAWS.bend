# Whole legal-move provenance, not independent attack correctness or legal completeness.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as Board
import ./Spec.bend as S

law scan_excludes_castling:
  for +xs: List<&2,U32>
  for +b: Chess.Board
  for r: Array<U64> & List<&2,Chess.Ply>
  for tail: S.all(S.not_castle,E.moves(r))
  S.all(S.not_castle,E.moves(Chess.scan(xs,b,r)))

law final_filter_preserves_predicate:
  for +b: Chess.Board
  for r: Array<U64> & List<&2,Chess.Ply>
  for -P: @-m: Chess.Ply -> Type
  for before: S.all(P,E.moves(r))
  S.all(P,E.moves(Chess.filter_prepare(b,r)))

law both_castling_sides_provenance:
  for +b: Chess.Board
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  S.all(m => S.origin(b,tail,m),E.moves(S.both(b,(table,tail))))

law filtered_castling_provenance:
  for +b: Chess.Board
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  S.all(m => S.origin(b,tail,m),E.moves(S.filtered(b,(table,tail))))

law legal_castle_guard_and_route:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for is_castle: {S.flag(m) == 2 : U32}
  S.certified(b,m)

law legal_castle_preserves_representation:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for good: {Board.valid(b) == True{} : Bool}
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for is_castle: {S.flag(m) == 2 : U32}
  {Board.valid(Chess.make_move(b,m)) == True{} : Bool}
