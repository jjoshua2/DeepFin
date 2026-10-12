# Remaining-index count budget transition for actual U64.clear_lsb.
import Base
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge

def lt_succ_cancel(+a: Nat,+b: Nat) ->
  {Nat.is_lt(Nat.add(1n,a),Nat.add(1n,b)) == Nat.is_lt(a,b) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: lt_succ_cancel(p,q)

def nonzero_clear_word_count(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {Word.count(64n,U64.to_word(a)) ==
    Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))) : Nat}:
  Equal.trans(Nat,
    Word.count(64n,U64.to_word(a)),
    Nat.add(Bool.pick(Nat,ClearStep.is_zero_word(64n,U64.to_word(a)),0n,1n),
      Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
    Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
    ClearStep.clear_lsb_count_step(a),
    Equal.cong(Bool,Nat,
      z => Nat.add(Bool.pick(Nat,z,0n,1n),Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
      ClearStep.is_zero_word(64n,U64.to_word(a)),False{},
      Equal.trans(Bool,
        ClearStep.is_zero_word(64n,U64.to_word(a)),
        U64.is_zero(a),
        False{},
        Equal.sym(Bool,U64.is_zero(a),ClearStep.is_zero_word(64n,U64.to_word(a)),
          ZeroBridge.u64_is_zero_word(a)),
        e)))

def clear_count_budget(+a: U64,+n: Nat,
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(a)),Nat.add(1n,n)) == True{} : Bool},
  +nonzero: {U64.is_zero(a) == False{} : Bool}) ->
  {Nat.is_lt(Word.count(64n,U64.to_word(U64.clear_lsb(a))),n) == True{} : Bool}:
  Equal.trans(Bool,
    Nat.is_lt(Word.count(64n,U64.to_word(U64.clear_lsb(a))),n),
    Nat.is_lt(Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))),Nat.add(1n,n)),
    True{},
    lt_succ_cancel(Word.count(64n,U64.to_word(U64.clear_lsb(a))),n),
    Equal.trans(Bool,
      Nat.is_lt(Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))),Nat.add(1n,n)),
      Nat.is_lt(Word.count(64n,U64.to_word(a)),Nat.add(1n,n)),
      True{},
      Equal.cong(Nat,Bool,k => Nat.is_lt(k,Nat.add(1n,n)),
        Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
        Word.count(64n,U64.to_word(a)),
        Equal.sym(Nat,
          Word.count(64n,U64.to_word(a)),
          Nat.add(1n,Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
          nonzero_clear_word_count(a,nonzero))),
      budget))
