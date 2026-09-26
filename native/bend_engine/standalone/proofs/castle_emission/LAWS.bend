# Actual producer output provenance; no attack-correctness or whole legal_moves claim.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ../castling/Route.bend as Route
import ./Spec.bend as S

law castle_side_guarded_extension:
  for +b: Chess.Board
  for +ks: Bool
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  S.growth(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail)))

law new_castle_member_has_guard:
  for +b: Chess.Board
  for +ks: Bool
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail))))
  for absent: S.member(m,tail) -> Empty
  {S.guard(b,ks) == True{} : Bool}

law new_castle_member_matches_route:
  for +b: Chess.Board
  for +ks: Bool
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail))))
  for absent: S.member(m,tail) -> Empty
  {m == S.move(b,ks) : Chess.Ply}

law producer_guard_supplies_rook_freshness:
  for +b: Chess.Board
  for +ks: Bool
  for guard: {S.guard(b,ks) == True{} : Bool}
  {Board.fresh_mask(b,Route.rook(S.route(b,ks))) == True{} : Bool}

law new_castle_member_preserves_representation:
  for +b: Chess.Board
  for +ks: Bool
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  for +m: Chess.Ply
  for good: {Board.valid(b) == True{} : Bool}
  for here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail))))
  for absent: S.member(m,tail) -> Empty
  {Board.valid(Chess.make_move(b,m)) == True{} : Bool}

law rejected_guard_preserves_pair:
  for +b: Chess.Board
  for +ks: Bool
  for table: Array<U64>
  for +tail: List<&2,Chess.Ply>
  for rejected: {S.guard(b,ks) == False{} : Bool}
  {Chess.castle_side(b,ks,(table,tail)) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}
