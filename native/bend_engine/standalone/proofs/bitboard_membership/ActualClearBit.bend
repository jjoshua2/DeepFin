# Actual bounded U64 test_bit behavior under clear_lsb.
import Base
import ./ClearModelBit.bend as Model
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as CtzBit
import ../bitboard_inventory/ClearStep.bend as ClearStep

def actual_clear_lsb_bit(+a: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  +nonzero: {U64.is_zero(a) == False{} : Bool}) ->
  {U64.test_bit(U64.clear_lsb(a),i) ==
    Bool.and(U64.test_bit(a,i),Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))) : Bool}:
  Equal.trans(Bool,
    U64.test_bit(U64.clear_lsb(a),i),
    Ctz.word_bit(64n,U64.to_word(U64.clear_lsb(a)),i),
    Bool.and(U64.test_bit(a,i),Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
    CtzBit.actual_testbit_word(U64.clear_lsb(a),i,bound),
    Equal.trans(Bool,
      Ctz.word_bit(64n,U64.to_word(U64.clear_lsb(a)),i),
      Ctz.word_bit(64n,ClearStep.clear_model(64n,U64.to_word(a)),i),
      Bool.and(U64.test_bit(a,i),Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
      Equal.cong(Word(64n),Bool,w => Ctz.word_bit(64n,w,i),
        U64.to_word(U64.clear_lsb(a)),ClearStep.clear_model(64n,U64.to_word(a)),
        ClearStep.clear_actual(a)),
      Equal.trans(Bool,
        Ctz.word_bit(64n,ClearStep.clear_model(64n,U64.to_word(a)),i),
        Bool.and(Ctz.word_bit(64n,U64.to_word(a),i),
          Bool.not(Nat.is_eq(i,Ctz.first_set(64n,U64.to_word(a))))),
        Bool.and(U64.test_bit(a,i),Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
        Model.clear_model_bit(64n,U64.to_word(a),i),
        Equal.trans(Bool,
          Bool.and(Ctz.word_bit(64n,U64.to_word(a),i),
            Bool.not(Nat.is_eq(i,Ctz.first_set(64n,U64.to_word(a))))),
          Bool.and(Ctz.word_bit(64n,U64.to_word(a),i),
            Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
          Bool.and(U64.test_bit(a,i),Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
          Equal.cong(Nat,Bool,k => Bool.and(Ctz.word_bit(64n,U64.to_word(a),i),
              Bool.not(Nat.is_eq(i,k))),
            Ctz.first_set(64n,U64.to_word(a)),U32.to_nat(U64.ctz(a)),
            Equal.sym(Nat,U32.to_nat(U64.ctz(a)),Ctz.first_set(64n,U64.to_word(a)),
              Ctz.actual_ctz_index_nat(a,nonzero))),
          Equal.cong(Bool,Bool,b => Bool.and(b,Bool.not(Nat.is_eq(i,U32.to_nat(U64.ctz(a))))),
            Ctz.word_bit(64n,U64.to_word(a),i),U64.test_bit(a,i),
            Equal.sym(Bool,U64.test_bit(a,i),Ctz.word_bit(64n,U64.to_word(a),i),
              CtzBit.actual_testbit_word(a,i,bound)))))))
