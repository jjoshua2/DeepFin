import Base
import ../../../legal_probe/Chess.bend as Chess
import ./OneBit.bend as OneBit
import ./ClearStep.bend as ClearStep
import ./Inventory.bend as Inventory
import ../destination_factorization/Spec.bend as S

def use_singleton_inventory(+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Chess.bit_squares(64n,U64.is_zero(U64.bit(sq)),U64.bit(sq),Nil{}) == U32.from_nat(sq) <> Nil{} : List<&2,U32>}:
  Inventory.singleton_indices(sq,e)

def use_actual_scan(table: Array<U64>,+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool},+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,U64.bit(sq))) ==
    (table,Chess.put_move(Bool.and(pawn,Bool.or(U32.is_zero(U32.div(U32.from_nat(sq),8)),U32.is_eq(U32.div(U32.from_nat(sq),8),7))),
      src,U32.from_nat(sq),Bool.to_u32(Bool.and(pawn,U32.is_eq(U32.from_nat(sq),ep))),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  OneBit.scan_one_bit(table,sq,e,src,pawn,ep,tail)

def use_actual_scan_step_onehot(+b: Chess.Board,+src: U32,table: Array<U64>,+tail: List<&2,Chess.Ply>,+sq: Nat,
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),table,b,src)
    == (table,U64.bit(sq)) : Array<U64> & U64}) ->
  {Chess.scan_step(b,src,(table,tail)) ==
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),U32.from_nat(sq)),src,
      U32.from_nat(sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),U32.from_nat(sq),Chess.get_ep(b)),tail))
    : Array<U64> & List<&2,Chess.Ply>}:
  OneBit.scan_step_onehot(b,src,table,tail,sq,bound,target)

def use_actual_scan_step_empty(+b: Chess.Board,+src: U32,table: Array<U64>,+tail: List<&2,Chess.Ply>,
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(src)),table,b,src)
    == (table,U64.zero()) : Array<U64> & U64}) ->
  {Chess.scan_step(b,src,(table,tail)) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}:
  OneBit.scan_step_empty(b,src,table,tail,target)


def use_actual_scan_source(+b: Chess.Board,table: Array<U64>,+src_sq: Nat,
  src_bound: {Nat.is_lt(src_sq,64n) == True{} : Bool},
  source: {Chess.color(b,Chess.get_turn(b)) == U64.bit(src_sq) : U64},+dst_sq: Nat,
  dst_bound: {Nat.is_lt(dst_sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),table,b,U32.from_nat(src_sq))
    == (table,U64.bit(dst_sq)) : Array<U64> & U64}) ->
  {Chess.scan(Chess.bit_squares(64n,U64.is_zero(Chess.color(b,Chess.get_turn(b))),Chess.color(b,Chess.get_turn(b)),Nil{}),b,(table,Nil{})) ==
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
      U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
        U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  OneBit.scan_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)

def use_actual_legal_moves_one_source(+b: Chess.Board,table: Array<U64>,+src_sq: Nat,
  src_bound: {Nat.is_lt(src_sq,64n) == True{} : Bool},
  source: {Chess.color(b,Chess.get_turn(b)) == U64.bit(src_sq) : U64},+dst_sq: Nat,
  dst_bound: {Nat.is_lt(dst_sq,64n) == True{} : Bool},
  target: {Chess.piece_targets(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),table,b,U32.from_nat(src_sq))
    == (table,U64.bit(dst_sq)) : Array<U64> & U64}) ->
  {Chess.legal_moves(table,b) == Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},
    (table,Chess.put_move(S.promotes(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),U32.from_nat(dst_sq)),
      U32.from_nat(src_sq),U32.from_nat(dst_sq),S.flag(U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(src_sq))),
        U32.from_nat(dst_sq),Chess.get_ep(b)),Nil{}))))) : Array<U64> & List<&2,Chess.Ply>}:
  OneBit.legal_moves_one_source(b,table,src_sq,src_bound,source,dst_sq,dst_bound,target)


def use_clear_lsb_popcount_step(+a: U64) ->
  {U64.popcount(a) == U32.from_nat(Nat.add(Bool.pick(Nat,ClearStep.is_zero_word(64n,U64.to_word(a)),0n,1n),
    Word.count(64n,U64.to_word(U64.clear_lsb(a))))) : U32}:
  ClearStep.clear_lsb_popcount_step(a)
