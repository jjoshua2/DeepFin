# Frequency-level support for actual Chess.bit_squares membership.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../u64/Words.bend as Old
import ../layout/Domain.bend as Domain
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge
import ../ray/Traversal.bend as NatEq
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as Bit
import ./ActualClearBit.bend as ActualClear
import ./RemainingBudget.bend as Budget
import ./PrefixDecomposition.bend as Prefix
import ./Cardinality.bend as Cardinality
import ./IndexEquality.bend as IndexEquality

def count_zero_word_is_zero(+n: Nat,+w: Word(n),+e: {Word.count(n,w) == 0n : Nat}) ->
  {ClearStep.is_zero_word(n,w) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},tail}:
          Equal.trans(Bool,ClearStep.is_zero_word(1n+p,w),ClearStep.is_zero_word(p,tail),True{},
            {==},count_zero_word_is_zero(p,tail,e))
        case WCon{True{},tail}:
          Old.false_true({ClearStep.is_zero_word(1n+p,w) == True{} : Bool},
            Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),Word.count(1n+p,w),0n,e))

def zero_word_testbit_false(+n: Nat,+w: Word(n),+i: Nat,
  +bound: {Nat.is_lt(i,n) == True{} : Bool},
  +zero: {ClearStep.is_zero_word(n,w) == True{} : Bool}) ->
  {Ctz.word_bit(n,w,i) == False{} : Bool}:
  match n:
    case 0n: Domain.impossible({Ctz.word_bit(0n,w,i) == False{} : Bool},i,bound)
    case 1n+p:
      match w:
        case WCon{False{},tail}:
          match i:
            case 0n: {==}
            case 1n+q: zero_word_testbit_false(p,tail,q,bound,zero)
        case WCon{True{},tail}:
          Old.false_true({Ctz.word_bit(1n+p,w,i) == False{} : Bool},zero)

def zero_u64_testbit_false(+bb: U64,+i: Nat,
  +bound: {Nat.is_lt(i,64n) == True{} : Bool},
  +zero: {U64.is_zero(bb) == True{} : Bool}) ->
  {U64.test_bit(bb,i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(bb,i),Ctz.word_bit(64n,U64.to_word(bb),i),False{},
    Bit.actual_testbit_word(bb,i,bound),
    zero_word_testbit_false(64n,U64.to_word(bb),i,bound,
      Equal.trans(Bool,ClearStep.is_zero_word(64n,U64.to_word(bb)),U64.is_zero(bb),True{},
        Equal.sym(Bool,U64.is_zero(bb),ClearStep.is_zero_word(64n,U64.to_word(bb)),ZeroBridge.u64_is_zero_word(bb)),zero)))

def count_zero_u64_testbit_false(+bb: U64,+i: Nat,
  +bound: {Nat.is_lt(i,64n) == True{} : Bool},
  +zero_count: {Word.count(64n,U64.to_word(bb)) == 0n : Nat}) ->
  {U64.test_bit(bb,i) == False{} : Bool}:
  zero_u64_testbit_false(bb,i,bound,
    Equal.trans(Bool,U64.is_zero(bb),ClearStep.is_zero_word(64n,U64.to_word(bb)),True{},
      ZeroBridge.u64_is_zero_word(bb),
      count_zero_word_is_zero(64n,U64.to_word(bb),zero_count)))

def empty_fuel_frequency(+bb: U64,+query: Nat,+bound: {Nat.is_lt(query,64n) == True{} : Bool},
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),1n) == True{} : Bool}) ->
  {Prefix.occurrences(Chess.bit_squares(0n,U64.is_zero(bb),bb,Nil{}),U32.from_nat(query)) ==
    Bool.pick(Nat,U64.test_bit(bb,query),1n,0n) : Nat}:
  Equal.trans(Nat,Prefix.occurrences(Chess.bit_squares(0n,U64.is_zero(bb),bb,Nil{}),U32.from_nat(query)),
    0n,Bool.pick(Nat,U64.test_bit(bb,query),1n,0n),{==},
    Equal.trans(Nat,0n,Bool.pick(Nat,False{},1n,0n),
      Bool.pick(Nat,U64.test_bit(bb,query),1n,0n),
      {==},
      Equal.sym(Nat,Bool.pick(Nat,U64.test_bit(bb,query),1n,0n),
        Bool.pick(Nat,False{},1n,0n),
        Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),U64.test_bit(bb,query),False{},
          count_zero_u64_testbit_false(bb,query,bound,
            Cardinality.count_lt_one_zero(Word.count(64n,U64.to_word(bb)),budget))))))

def occurrence_one(+key: U32,+query: U32) ->
  {Prefix.occurrences(key <> Nil{},query) == Bool.pick(Nat,U32.is_eq(key,query),1n,0n) : Nat}:
  Equal.trans(Nat,
    Prefix.occurrences(key <> Nil{},query),
    Nat.add(Bool.pick(Nat,U32.is_eq(key,query),1n,0n),0n),
    Bool.pick(Nat,U32.is_eq(key,query),1n,0n),
    {==},Cardinality.nat_add_zero_right(Bool.pick(Nat,U32.is_eq(key,query),1n,0n)))
