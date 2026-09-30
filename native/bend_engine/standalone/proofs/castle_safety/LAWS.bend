# Mandatory actual check path, not independent semantic attack correctness.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ./Spec.bend as S
import ./Forced.bend as Force

law owned_king_source_requires_check:
  for +b: Chess.Board
  for +rays: U64
  for +src: U32
  for +dst: U32
  for +promotion: U32
  for +flag: U32
  for bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool}
  for present: {S.owned_king(b,src) == True{} : Bool}
  {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,promotion,flag}) == True{} : Bool}

law blocker_stage_uses_sensitive_mask:
  for +b: Chess.Board
  for +moves: List<&2,Chess.Ply>
  for table: Array<U64>
  for +rays: U64
  {Chess.filter_blockers(b,moves,(table,rays)) == Chess.filter_fast(moves,b,S.sensitive(b,rays),(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}

law generated_castle_requires_check:
  for table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for +rays: U64
  for here: E.member(m,E.moves(Chess.legal_moves(table,b)))
  for tag: {ChainSpec.flag(m) == 2 : U32}
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}

law generated_castle_takes_checked_step:
  for generation_table: Array<U64>
  for +b: Chess.Board
  for +m: Chess.Ply
  for +rays: U64
  for here: E.member(m,E.moves(Chess.legal_moves(generation_table,b)))
  for tag: {ChainSpec.flag(m) == 2 : U32}
  for filter_table: Array<U64>
  for +acc: List<&2,Chess.Ply>
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,(filter_table,acc)) ==
    Chess.filter_after(m,acc,Chess.in_check(filter_table,Chess.make_move(b,m),Chess.get_turn(b))) : Array<U64> & List<&2,Chess.Ply>}

law positive_check_rejects_current_move:
  for +m: Chess.Ply
  for +acc: List<&2,Chess.Ply>
  for table: Array<U64>
  {Chess.filter_after(m,acc,(table,True{})) == (table,acc) : Array<U64> & List<&2,Chess.Ply>}

law complete_generator_equals_forced_castle_checks:
  for table: Array<U64>
  for +b: Chess.Board
  {Chess.legal_moves(table,b) == Force.legal_moves(table,b) : Array<U64> & List<&2,Chess.Ply>}
