# Operational king-safety path, not independent semantic attack correctness.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ./Spec.bend as S

law guarded_castle_requires_full_check:
  for +b: Chess.Board
  for +side: Bool
  for +rays: U64
  for ok: {E.guard(b,side) == True{} : Bool}
  {Chess.filter_requires(S.sensitive(b,rays),E.move(b,side)) == True{} : Bool}

law generated_castle_requires_full_check:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for +rays: U64
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for tag: {C.flag(m) == 2 : U32}
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}

law generated_castle_fast_step_equals_checked_step:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for +rays: U64
  for r: Array<U64> & List<&2,Chess.Ply>
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for tag: {C.flag(m) == 2 : U32}
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}

law guarded_castling_blocker_filter_equals_full_filter:
  for +b: Chess.Board
  for +xs: List<&2,Chess.Ply>
  for table: Array<U64>
  for +rays: U64
  for good: C.every(m => C.castle(b,m),xs)
  {Chess.filter_blockers(b,xs,(table,rays)) == Chess.filter_legal(xs,b,(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}
