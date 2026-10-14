import Base
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as Spec

def source_count(+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,
  piece_result: Array<U64> & U64,+tail_count: Nat) -> Nat:
  (_, +targets) = piece_result
  Spec.tally(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),
    src,pawn,ep,query,tail_count)

def after_count(+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  +query: Chess.Ply,piece_result: Array<U64> & U64) ->
  {After.pair_count(Chess.scan_after(src,pawn,ep,tail,piece_result),query) ==
    source_count(src,pawn,ep,query,piece_result,Spec.count(tail,query)) : Nat}:
  (table,+targets) = piece_result
  After.scan_after_count(table,targets,src,pawn,ep,query,tail)

def actual_step_count(+b: Chess.Board,+src: U32,table: Array<U64>,
  +tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.scan_step(b,src,(table,tail)),query) ==
    source_count(src,U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),
      Chess.get_ep(b),query,Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),
        table,b,src),Spec.count(tail,query)) : Nat}:
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  after_count(src,pawn,Chess.get_ep(b),tail,query,Chess.piece_targets(pawn,table,b,src))
