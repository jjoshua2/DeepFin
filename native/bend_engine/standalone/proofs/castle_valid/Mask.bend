import Base
import ./Frame.bend as F
import ../king_counts/Actual.bend as K
import ../color_counts/Actual.bend as Fields
import ../backrank_pawns/Mask.bend as M
import ../child_ep_coherence/Bits.bend as B
import ../initialized_pawn_double/Algebra.bend as A
import ../../../legal_probe/Chess.bend as Chess

def no_rook(+m: Chess.Ply,+i: Nat,no: {U32.is_eq(Fields.flag(m),2) == False{} : Bool}) ->
  {U64.test_bit(F.rook_to(m),i) == False{} : Bool}:
  Chess.Ply{+s,+d,p,+f}=m
  %Equal.sym(Bool,U32.is_eq(f,2),False{},no) :
    {U64.test_bit(Chess.select_u64(_,U64.bit(U32.to_nat(U32.div(U32.add(s,d),2))),U64.zero()),i) == False{} : Bool}
  A.zero(i)
def covered(+m: Chess.Ply,+i: Nat,+no: {U32.is_eq(Fields.flag(m),2) == False{} : Bool},
  source: {U64.test_bit(U64.bit(U32.to_nat(Fields.source(m))),i) == False{} : Bool},
  +target: {U64.test_bit(M.target(m),i) == False{} : Bool},
  capture: {U64.test_bit(U64.bit(U32.to_nat(K.cap(m))),i) == False{} : Bool}) ->
  {U64.test_bit(M.remove(m),i) == False{} : Bool} &
    ({U64.test_bit(F.rook_to(m),i) == False{} : Bool} & {U64.test_bit(F.added(m),i) == False{} : Bool}):
  Chess.Ply{+s,+d,+p,+f}=m
  +rook=no_rook(m,i,no)
  (Equal.trans(Bool,U64.test_bit(M.remove(m),i),
    U64.test_bit(U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(K.cap(m)))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),i),False{},
    Equal.cong(U64,Bool,x => U64.test_bit(x,i),M.remove(m),U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(K.cap(m)))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),K.simple_remove(m,no)),
    B.or_false(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(K.cap(m)))),U64.or(U64.bit(U32.to_nat(d)),U64.zero()),i,
      B.or_false(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(K.cap(m))),i,source,capture),
      B.or_false(U64.bit(U32.to_nat(d)),U64.zero(),i,target,A.zero(i)))),
   (rook,B.or_false(M.target(m),F.rook_to(m),i,target,rook)))
