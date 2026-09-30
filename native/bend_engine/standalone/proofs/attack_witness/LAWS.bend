# Exact reduction and query composition. Mask geometry is an explicit remaining boundary.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Reduction.bend as R
import ./Queries.bend as Q

law mask_reduction_has_square_witness:
  for +b: Chess.Board
  for +by: U32
  for +m: S.Masks
  {Bool.not(U64.is_zero(U64.and(R.hits(b,m),Chess.color(b,by)))) == S.present(b,by,m) : Bool}

law actual_attacked_uses_witness_reduction:
  for table: Array<U64>
  for +b: Chess.Board
  for +sq: U32
  for +by: U32
  {Chess.attacked(table,b,sq,by) == Q.run(table,b,sq,by) : Array<U64> & Bool}

law single_king_in_check_uses_witness_reduction:
  for table: Array<U64>
  for +b: Chess.Board
  for +side: U32
  for +sq: Nat
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  for one: {U64.and(Chess.get_kings(b),Chess.color(b,side)) == U64.bit(sq) : U64}
  {Chess.in_check(table,b,side) == Q.run(table,b,U32.from_nat(sq),U32.xor(side,1)) : Array<U64> & Bool}
