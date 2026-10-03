# First nonzero step of the actual Chess.bit_squares enumeration.
# Scratch-only: imports actual Chess and checked zero/ctz/clear bridges.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as CtzBit

def bit_squares_step(+bb: U64,+e: {U64.is_zero(bb) == False{} : Bool}) ->
  {Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}) ==
    Chess.bit_squares(63n,U64.is_zero(U64.clear_lsb(bb)),U64.clear_lsb(bb),U64.ctz(bb) <> Nil{}) : List<&2,U32>}:
  Equal.trans(List<&2,U32>,
    Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
    Chess.bit_squares(64n,False{},bb,Nil{}),
    Chess.bit_squares(63n,U64.is_zero(U64.clear_lsb(bb)),U64.clear_lsb(bb),U64.ctz(bb) <> Nil{}),
    Equal.cong(Bool,List<&2,U32>,empty => Chess.bit_squares(64n,empty,bb,Nil{}),
      U64.is_zero(bb),False{},e),
    {==})

def bit_squares_step_head_is_set(+bb: U64,+e: {U64.is_zero(bb) == False{} : Bool}) ->
  {U64.test_bit(bb,U32.to_nat(U64.ctz(bb))) == True{} : Bool}:
  CtzBit.actual_ctz_selects_bit(bb,e)

def nonzero_clear_count(+bb: U64,+e: {U64.is_zero(bb) == False{} : Bool}) ->
  {U64.popcount(bb) == U32.from_nat(Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(bb))))) : U32}:
  Equal.trans(U32,
    U64.popcount(bb),
    U32.from_nat(Nat.add(Bool.pick(Nat,ClearStep.is_zero_word(64n,U64.to_word(bb)),0n,1n),
      Word.count(64n,U64.to_word(U64.clear_lsb(bb))))),
    U32.from_nat(Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(bb))))),
    ClearStep.clear_lsb_popcount_step(bb),
    Equal.cong(Bool,U32,
      z => U32.from_nat(Nat.add(Bool.pick(Nat,z,0n,1n),Word.count(64n,U64.to_word(U64.clear_lsb(bb))))),
      ClearStep.is_zero_word(64n,U64.to_word(bb)),False{},
      Equal.trans(Bool,
        ClearStep.is_zero_word(64n,U64.to_word(bb)),
        U64.is_zero(bb),
        False{},
        Equal.sym(Bool,U64.is_zero(bb),ClearStep.is_zero_word(64n,U64.to_word(bb)),
          ZeroBridge.u64_is_zero_word(bb)),
        e)))
