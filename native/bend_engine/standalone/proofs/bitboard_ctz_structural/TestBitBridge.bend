import Base
import ./WordCtzCore.bend as Ctz
import ./U64BitProbe.bend as Bit
import ./EqZeroProbe.bend as EqZero
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge
import ../successor/Step.bend as Step
import ../bitboard_inventory/ClearStep.bend as ClearStep

def zero_word_is_zero(+n: Nat) -> {ClearStep.is_zero_word(n,Word.zero(n)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: zero_word_is_zero(p)

def and_zero_right(+n: Nat,+w: Word(n)) ->
  {Word.and(n,w,Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
            Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero_right(p,t))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
            Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero_right(p,t))

def mask_bit_nonzero(+n: Nat,+w: Word(n),+i: Nat) ->
  {Bool.not(ClearStep.is_zero_word(n,Word.and(n,w,Bit.word_singleton(n,i)))) == Ctz.word_bit(n,w,i) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          match i:
            case 0n:
              Equal.trans(Bool,
                Bool.not(ClearStep.is_zero_word(1n+p,Word.and(1n+p,w,Bit.word_singleton(1n+p,0n)))),
                Bool.not(ClearStep.is_zero_word(p,Word.zero(p))),
                False{},
                Equal.cong(Word(1n+p),Bool,x => Bool.not(ClearStep.is_zero_word(1n+p,x)),
                  Word.and(1n+p,w,Bit.word_singleton(1n+p,0n)),WCon{False{},Word.zero(p)},
                  Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
                    Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero_right(p,t))),
                Equal.cong(Bool,Bool,x => Bool.not(x),ClearStep.is_zero_word(p,Word.zero(p)),True{},zero_word_is_zero(p)))
            case 1n+q:
              mask_bit_nonzero(p,t,q)
        case WCon{True{},t}:
          match i:
            case 0n: {==}
            case 1n+q: mask_bit_nonzero(p,t,q)

def actual_and_word(+a: U64,+i: Nat,bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.to_word(U64.and(a,U64.bit(i))) == Word.and(64n,U64.to_word(a),Bit.word_singleton(64n,i)) : Word(64n)}:
  Equal.trans(Word(64n),
    U64.to_word(U64.and(a,U64.bit(i))),
    Word.and(64n,U64.to_word(a),U64.to_word(U64.bit(i))),
    Word.and(64n,U64.to_word(a),Bit.word_singleton(64n,i)),
    Step.and_word(a,U64.bit(i)),
    Equal.cong(Word(64n),Word(64n),x => Word.and(64n,U64.to_word(a),x),
      U64.to_word(U64.bit(i)),Bit.word_singleton(64n,i),Bit.bit_word_exact(i,bound)))

def testbit_zero_struct(+a: U64,+i: Nat) ->
  {U64.test_bit(a,i) == Bool.not(ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i))))) : Bool}:
  Equal.trans(Bool,
    U64.test_bit(a,i),
    Bool.not(U64.is_eq(U64.and(a,U64.bit(i)),U64.zero())),
    Bool.not(ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i))))),
    {==},
    Equal.trans(Bool,
      Bool.not(U64.is_eq(U64.and(a,U64.bit(i)),U64.zero())),
      Bool.not(U64.is_zero(U64.and(a,U64.bit(i)))),
      Bool.not(ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i))))),
      Equal.cong(Bool,Bool,x => Bool.not(x),
        U64.is_eq(U64.and(a,U64.bit(i)),U64.zero()),
        U64.is_zero(U64.and(a,U64.bit(i))),EqZero.eq_zero_is_zero(U64.and(a,U64.bit(i)))),
      Equal.cong(Bool,Bool,x => Bool.not(x),
        U64.is_zero(U64.and(a,U64.bit(i))),
        ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i)))),
        ZeroBridge.u64_is_zero_word(U64.and(a,U64.bit(i))))))

def actual_testbit_word(+a: U64,+i: Nat,bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(a,i) == Ctz.word_bit(64n,U64.to_word(a),i) : Bool}:
  Equal.trans(Bool,
    U64.test_bit(a,i),
    Bool.not(ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i))))),
    Ctz.word_bit(64n,U64.to_word(a),i),
    testbit_zero_struct(a,i),
    Equal.trans(Bool,
      Bool.not(ClearStep.is_zero_word(64n,U64.to_word(U64.and(a,U64.bit(i))))),
      Bool.not(ClearStep.is_zero_word(64n,Word.and(64n,U64.to_word(a),Bit.word_singleton(64n,i)))),
      Ctz.word_bit(64n,U64.to_word(a),i),
      Equal.cong(Word(64n),Bool,x => Bool.not(ClearStep.is_zero_word(64n,x)),
        U64.to_word(U64.and(a,U64.bit(i))),Word.and(64n,U64.to_word(a),Bit.word_singleton(64n,i)),
        actual_and_word(a,i,bound)),
      mask_bit_nonzero(64n,U64.to_word(a),i)))

def actual_ctz_selects_bit(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {U64.test_bit(a,U32.to_nat(U64.ctz(a))) == True{} : Bool}:
  Equal.trans(Bool,
    U64.test_bit(a,U32.to_nat(U64.ctz(a))),
    Ctz.word_bit(64n,U64.to_word(a),U32.to_nat(U64.ctz(a))),
    True{},
    actual_testbit_word(a,U32.to_nat(U64.ctz(a)),Ctz.actual_ctz_range(a,e)),
    Equal.trans(Bool,
      Ctz.word_bit(64n,U64.to_word(a),U32.to_nat(U64.ctz(a))),
      Ctz.word_bit(64n,U64.to_word(a),Ctz.first_set(64n,U64.to_word(a))),
      True{},
      Equal.cong(Nat,Bool,i => Ctz.word_bit(64n,U64.to_word(a),i),
        U32.to_nat(U64.ctz(a)),Ctz.first_set(64n,U64.to_word(a)),Ctz.actual_ctz_index_nat(a,e)),
      Equal.trans(Bool,
        Ctz.word_bit(64n,U64.to_word(a),Ctz.first_set(64n,U64.to_word(a))),
        Bool.not(ClearStep.is_zero_word(64n,U64.to_word(a))),
        True{},
        Ctz.first_set_bit(64n,U64.to_word(a)),
        Equal.cong(Bool,Bool,x => Bool.not(x),
          ClearStep.is_zero_word(64n,U64.to_word(a)),False{},Ctz.actual_word_nonzero(a,e)))))
