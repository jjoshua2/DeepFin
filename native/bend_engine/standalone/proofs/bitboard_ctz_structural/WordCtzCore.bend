import Base
import ../bitboard_zero_bridge/Bridge.bend as ZeroBridge
import ../layout/Domain.bend as Domain
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../successor/Borrow.bend as Borrow
import ../successor/Step.bend as Step

def first_set(+n: Nat,+w: Word(n)) -> Nat:
  match n:
    case 0n: 0n
    case 1n+p:
      match w:
        case WCon{False{},t}: 1n+first_set(p,t)
        case WCon{True{},t}: 0n

def lowbit_model(+n: Nat,+w: Word(n)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match w:
        case WCon{False{},t}: WCon{False{},lowbit_model(p,t)}
        case WCon{True{},t}: WCon{True{},Word.zero(p)}

def full_add_not(+b: Bool) ->
  {Bool.full_add(False{},Bool.not(b),False{}) == (Bool.not(b),False{}) : Bool & Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def adc_zero_not(+n: Nat,+w: Word(n)) ->
  {Word.adc(n,Word.zero(n),w,True{},False{}) == Word.not(n,w) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{b,t}:
          Equal.trans(Word(1n+p),
            Word.adc.con(p,Word.zero(p),t,True{},Bool.full_add(False{},Bool.not(b),False{})),
            WCon{Bool.not(b),Word.adc(p,Word.zero(p),t,True{},False{})},
            Word.not(1n+p,w),
            Equal.cong(Bool & Bool,Word(1n+p),sk => Word.adc.con(p,Word.zero(p),t,True{},sk),
              Bool.full_add(False{},Bool.not(b),False{}),(Bool.not(b),False{}),full_add_not(b)),
            Equal.cong(Word(p),Word(1n+p),x => WCon{Bool.not(b),x},
              Word.adc(p,Word.zero(p),t,True{},False{}),Word.not(p,t),adc_zero_not(p,t)))

def neg_zero_false(+p: Nat,+t: Word(p)) ->
  {Word.sub(1n+p,Word.zero(1n+p),WCon{False{},t}) == WCon{False{},Word.sub(p,Word.zero(p),t)} : Word(1n+p)}:
  {==}

def neg_zero_true(+p: Nat,+t: Word(p)) ->
  {Word.sub(1n+p,Word.zero(1n+p),WCon{True{},t}) == WCon{True{},Word.not(p,t)} : Word(1n+p)}:
  Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},
    Word.adc(p,Word.zero(p),t,True{},False{}),Word.not(p,t),adc_zero_not(p,t))

def and_not_zero(+n: Nat,+w: Word(n)) ->
  {Word.and(n,w,Word.not(n,w)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
            Word.and(p,t,Word.not(p,t)),Word.zero(p),and_not_zero(p,t))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
            Word.and(p,t,Word.not(p,t)),Word.zero(p),and_not_zero(p,t))

def lsb_model(+n: Nat,+w: Word(n)) ->
  {Word.and(n,w,Word.sub(n,Word.zero(n),w)) == lowbit_model(n,w) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.trans(Word(1n+p),
            Word.and(1n+p,w,Word.sub(1n+p,Word.zero(1n+p),w)),
            WCon{False{},Word.and(p,t,Word.sub(p,Word.zero(p),t))},
            lowbit_model(1n+p,w),
            Equal.cong(Word(1n+p),Word(1n+p),x => Word.and(1n+p,w,x),
              Word.sub(1n+p,Word.zero(1n+p),w),WCon{False{},Word.sub(p,Word.zero(p),t)},neg_zero_false(p,t)),
            Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
              Word.and(p,t,Word.sub(p,Word.zero(p),t)),lowbit_model(p,t),lsb_model(p,t)))
        case WCon{True{},t}:
          Equal.trans(Word(1n+p),
            Word.and(1n+p,w,Word.sub(1n+p,Word.zero(1n+p),w)),
            WCon{True{},Word.and(p,t,Word.not(p,t))},
            lowbit_model(1n+p,w),
            Equal.cong(Word(1n+p),Word(1n+p),x => Word.and(1n+p,w,x),
              Word.sub(1n+p,Word.zero(1n+p),w),WCon{True{},Word.not(p,t)},neg_zero_true(p,t)),
            Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},
              Word.and(p,t,Word.not(p,t)),Word.zero(p),and_not_zero(p,t)))


def zero_count(+n: Nat) -> {Word.count(n,Word.zero(n)) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p: zero_count(p)

def first_count_index(+n: Nat,+w: Word(n)) ->
  {Word.count(n,Borrow.dec(n,lowbit_model(n,w))) == first_set(n,w) : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Nat,Nat,x => 1n+x,
            Word.count(p,Borrow.dec(p,lowbit_model(p,t))),first_set(p,t),
            first_count_index(p,t))
        case WCon{True{},t}:
          Equal.trans(Nat,Word.count(p,Word.zero(p)),0n,first_set(1n+p,WCon{True{},t}),
            zero_count(p),{==})

def popcount_lowbit_index(+n: Nat,+w: Word(n)) ->
  {Word.count(n,Word.sub(n,lowbit_model(n,w),ClearStep.one(n))) == first_set(n,w) : Nat}:
  Equal.trans(Nat,
    Word.count(n,Word.sub(n,lowbit_model(n,w),ClearStep.one(n))),
    Word.count(n,Borrow.dec(n,lowbit_model(n,w))),
    first_set(n,w),
    Equal.cong(Word(n),Nat,x => Word.count(n,x),
      Word.sub(n,lowbit_model(n,w),ClearStep.one(n)),Borrow.dec(n,lowbit_model(n,w)),
      ClearStep.sub_one(n,lowbit_model(n,w))),
    first_count_index(n,w))


def zero_word() -> {U64.to_word(U64.zero()) == Word.zero(64n) : Word(64n)}:
  {==}

def lsb_word(+a: U64) ->
  {U64.to_word(U64.lsb(a)) == Word.and(64n,U64.to_word(a),Word.sub(64n,Word.zero(64n),U64.to_word(a))) : Word(64n)}:
  Equal.trans(Word(64n),
    U64.to_word(U64.lsb(a)),
    Word.and(64n,U64.to_word(a),U64.to_word(U64.sub(U64.zero(),a))),
    Word.and(64n,U64.to_word(a),Word.sub(64n,U64.to_word(U64.zero()),U64.to_word(a))),
    Step.and_word(a,U64.sub(U64.zero(),a)),
    Equal.cong(Word(64n),Word(64n),x => Word.and(64n,U64.to_word(a),x),
      U64.to_word(U64.sub(U64.zero(),a)),Word.sub(64n,U64.to_word(U64.zero()),U64.to_word(a)),
      Borrow.u64_sub(U64.zero(),a)))

def actual_lsb_model(+a: U64) ->
  {U64.to_word(U64.lsb(a)) == lowbit_model(64n,U64.to_word(a)) : Word(64n)}:
  Equal.trans(Word(64n),
    U64.to_word(U64.lsb(a)),
    Word.and(64n,U64.to_word(a),Word.sub(64n,Word.zero(64n),U64.to_word(a))),
    lowbit_model(64n,U64.to_word(a)),
    lsb_word(a),lsb_model(64n,U64.to_word(a)))

def actual_count_index(+a: U64) ->
  {Word.count(64n,U64.to_word(U64.sub(U64.lsb(a),U64.one()))) == first_set(64n,U64.to_word(a)) : Nat}:
  Equal.trans(Nat,
    Word.count(64n,U64.to_word(U64.sub(U64.lsb(a),U64.one()))),
    Word.count(64n,Word.sub(64n,U64.to_word(U64.lsb(a)),U64.to_word(U64.one()))),
    first_set(64n,U64.to_word(a)),
    Equal.cong(Word(64n),Nat,w => Word.count(64n,w),
      U64.to_word(U64.sub(U64.lsb(a),U64.one())),
      Word.sub(64n,U64.to_word(U64.lsb(a)),U64.to_word(U64.one())),
      Borrow.u64_sub(U64.lsb(a),U64.one())),
    Equal.trans(Nat,
      Word.count(64n,Word.sub(64n,U64.to_word(U64.lsb(a)),U64.to_word(U64.one()))),
      Word.count(64n,Word.sub(64n,lowbit_model(64n,U64.to_word(a)),ClearStep.one(64n))),
      first_set(64n,U64.to_word(a)),
      Equal.cong(Word(64n),Nat,w => Word.count(64n,w),
        Word.sub(64n,U64.to_word(U64.lsb(a)),U64.to_word(U64.one())),
        Word.sub(64n,lowbit_model(64n,U64.to_word(a)),ClearStep.one(64n)),
        Equal.trans(Word(64n),
          Word.sub(64n,U64.to_word(U64.lsb(a)),U64.to_word(U64.one())),
          Word.sub(64n,U64.to_word(U64.lsb(a)),ClearStep.one(64n)),
          Word.sub(64n,lowbit_model(64n,U64.to_word(a)),ClearStep.one(64n)),
          Equal.cong(Word(64n),Word(64n),x => Word.sub(64n,U64.to_word(U64.lsb(a)),x),
            U64.to_word(U64.one()),ClearStep.one(64n),ClearStep.one_word()),
          Equal.cong(Word(64n),Word(64n),x => Word.sub(64n,x,ClearStep.one(64n)),
            U64.to_word(U64.lsb(a)),lowbit_model(64n,U64.to_word(a)),actual_lsb_model(a)))),
      popcount_lowbit_index(64n,U64.to_word(a))))

def actual_ctz_index(+a: U64) ->
  {U64.ctz(a) == U32.from_nat(first_set(64n,U64.to_word(a))) : U32}:
  Equal.trans(U32,
    U64.ctz(a),
    U32.from_nat(Word.count(64n,U64.to_word(U64.sub(U64.lsb(a),U64.one())))),
    U32.from_nat(first_set(64n,U64.to_word(a))),
    {==},
    Equal.cong(Nat,U32,n => U32.from_nat(n),
      Word.count(64n,U64.to_word(U64.sub(U64.lsb(a),U64.one()))),
      first_set(64n,U64.to_word(a)),actual_count_index(a)))


def first_set_lt_nonzero(+n: Nat,+w: Word(n)) ->
  {Nat.is_lt(first_set(n,w),n) == Bool.not(ClearStep.is_zero_word(n,w)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}: first_set_lt_nonzero(p,t)
        case WCon{True{},t}: {==}

def first_set_roundtrip(+n: Nat,+w: Word(n), e: {Nat.is_lt(first_set(n,w),128n) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(first_set(n,w))) == first_set(n,w) : Nat}:
  Domain.exact(first_set(n,w),e)


def exact_u32_below64(+n: Nat,e: {Nat.is_lt(n,64n) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(n)) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: Domain.impossible({U32.to_nat(U32.from_nat(64n+p)) == 64n+p : Nat},p,e)

def actual_word_nonzero(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {ClearStep.is_zero_word(64n,U64.to_word(a)) == False{} : Bool}:
  Equal.trans(Bool,
    ClearStep.is_zero_word(64n,U64.to_word(a)),U64.is_zero(a),False{},
    Equal.sym(Bool,U64.is_zero(a),ClearStep.is_zero_word(64n,U64.to_word(a)),ZeroBridge.u64_is_zero_word(a)),e)

def actual_first_nonzero_bound(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {Nat.is_lt(first_set(64n,U64.to_word(a)),64n) == True{} : Bool}:
  Equal.trans(Bool,
    Nat.is_lt(first_set(64n,U64.to_word(a)),64n),
    Bool.not(ClearStep.is_zero_word(64n,U64.to_word(a))),True{},
    first_set_lt_nonzero(64n,U64.to_word(a)),
    Equal.cong(Bool,Bool,x => Bool.not(x),ClearStep.is_zero_word(64n,U64.to_word(a)),False{},actual_word_nonzero(a,e)))

def actual_ctz_range(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U64.ctz(a)),64n) == True{} : Bool}:
  Equal.trans(Bool,
    Nat.is_lt(U32.to_nat(U64.ctz(a)),64n),
    Nat.is_lt(U32.to_nat(U32.from_nat(first_set(64n,U64.to_word(a)))),64n),
    True{},
    Equal.cong(U32,Bool,x => Nat.is_lt(U32.to_nat(x),64n),
      U64.ctz(a),U32.from_nat(first_set(64n,U64.to_word(a))),actual_ctz_index(a)),
    Equal.trans(Bool,
      Nat.is_lt(U32.to_nat(U32.from_nat(first_set(64n,U64.to_word(a)))),64n),
      Nat.is_lt(first_set(64n,U64.to_word(a)),64n),
      True{},
      Equal.cong(Nat,Bool,x => Nat.is_lt(x,64n),
        U32.to_nat(U32.from_nat(first_set(64n,U64.to_word(a)))),
        first_set(64n,U64.to_word(a)),
        exact_u32_below64(first_set(64n,U64.to_word(a)),actual_first_nonzero_bound(a,e))),
      actual_first_nonzero_bound(a,e)))

def word_bit(+n: Nat,+w: Word(n),+i: Nat) -> Bool:
  match n:
    case 0n: False{}
    case 1n+p:
      match w:
        case WCon{b,t}:
          match i:
            case 0n: b
            case 1n+q: word_bit(p,t,q)

def first_set_bit(+n: Nat,+w: Word(n)) ->
  {word_bit(n,w,first_set(n,w)) == Bool.not(ClearStep.is_zero_word(n,w)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}: first_set_bit(p,t)
        case WCon{True{},t}: {==}


def actual_ctz_index_nat(+a: U64,+e: {U64.is_zero(a) == False{} : Bool}) ->
  {U32.to_nat(U64.ctz(a)) == first_set(64n,U64.to_word(a)) : Nat}:
  Equal.trans(Nat,
    U32.to_nat(U64.ctz(a)),
    U32.to_nat(U32.from_nat(first_set(64n,U64.to_word(a)))),
    first_set(64n,U64.to_word(a)),
    Equal.cong(U32,Nat,x => U32.to_nat(x),
      U64.ctz(a),U32.from_nat(first_set(64n,U64.to_word(a))),actual_ctz_index(a)),
    exact_u32_below64(first_set(64n,U64.to_word(a)),actual_first_nonzero_bound(a,e)))
