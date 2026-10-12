import Base
import ../ordinary_king_frame/Algebra.bend as OldFrame
import ../u64/Words.bend as Old
import ../ray_clear_disjoint/Algebra.bend as D
import ../bitboard_inventory/ClearStep.bend as Zero
import ../bitboard_zero_bridge/Bridge.bend as Bridge
import ../bitboard_ctz_structural/EqZeroProbe.bend as EqZero

def and_commute_bool(+a: Bool,+b: Bool) -> {Bool.and(a,b) == Bool.and(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def and_commute_word(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,a,b) == Word.and(n,b,a) : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{+ah,+at} WCon{+bh,+bt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(ah,bh),Word.and(p,at,bt)},WCon{Bool.and(bh,ah),Word.and(p,at,bt)},WCon{Bool.and(bh,ah),Word.and(p,bt,at)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,at,bt)},Bool.and(ah,bh),Bool.and(bh,ah),and_commute_bool(ah,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(bh,ah),t},Word.and(p,at,bt),Word.and(p,bt,at),and_commute_word(p,at,bt)))

def and_commute(+a: U64,+b: U64) -> {U64.and(a,b) == U64.and(b,a) : U64}:
  match a b:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,al,bl)},U32{Word.and(32n,ah,bh)}},U64{U32{Word.and(32n,bl,al)},U32{Word.and(32n,ah,bh)}},U64{U32{Word.and(32n,bl,al)},U32{Word.and(32n,bh,ah)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,ah,bh)}},Word.and(32n,al,bl),Word.and(32n,bl,al),and_commute_word(32n,al,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,bl,al)},U32{h}},Word.and(32n,ah,bh),Word.and(32n,bh,ah),and_commute_word(32n,ah,bh)))

def excluded_bool(+a: Bool,+own: Bool,+k: Bool) -> {Bool.and(Bool.and(a,Bool.not(Bool.or(own,k))),k) == False{} : Bool}:
  match a own k:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def excluded_word(+n: Nat,+a: Word(n),+own: Word(n),+k: Word(n)) -> {Word.and(n,Word.and(n,a,Word.not(n,Word.or(n,own,k))),k) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match a own k:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a own k:
        case WCon{+ah,+at} WCon{+ownh,+ownt} WCon{+kh,+kt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.and(ah,Bool.not(Bool.or(ownh,kh))),kh),Word.and(p,Word.and(p,at,Word.not(p,Word.or(p,ownt,kt))),kt)},WCon{False{},Word.and(p,Word.and(p,at,Word.not(p,Word.or(p,ownt,kt))),kt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.and(p,at,Word.not(p,Word.or(p,ownt,kt))),kt)},Bool.and(Bool.and(ah,Bool.not(Bool.or(ownh,kh))),kh),False{},excluded_bool(ah,ownh,kh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.and(p,at,Word.not(p,Word.or(p,ownt,kt))),kt),Word.zero(p),excluded_word(p,at,ownt,kt)))

def excluded(+a: U64,+own: U64,+k: U64) -> {U64.and(U64.and(a,U64.not(U64.or(own,k))),k) == U64.zero() : U64}:
  match a own k:
    case U64{U32{+al},U32{+ah}} U64{U32{+ownl},U32{+ownh}} U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,ownl,kl))),kl)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,ownh,kh))),kh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,ownh,kh))),kh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,ownh,kh))),kh)}},Word.and(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,ownl,kl))),kl),Word.zero(32n),excluded_word(32n,al,ownl,kl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,ownh,kh))),kh),Word.zero(32n),excluded_word(32n,ah,ownh,kh)))

def capturable_bool(+a: Bool,+k: Bool) -> {Bool.and(Bool.and(a,Bool.not(k)),k) == False{} : Bool}:
  match a k:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def capturable_word(+n: Nat,+a: Word(n),+k: Word(n)) -> {Word.and(n,Word.and(n,a,Word.not(n,k)),k) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match a k:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a k:
        case WCon{+ah,+at} WCon{+kh,+kt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.and(ah,Bool.not(kh)),kh),Word.and(p,Word.and(p,at,Word.not(p,kt)),kt)},WCon{False{},Word.and(p,Word.and(p,at,Word.not(p,kt)),kt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.and(p,at,Word.not(p,kt)),kt)},Bool.and(Bool.and(ah,Bool.not(kh)),kh),False{},capturable_bool(ah,kh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.and(p,at,Word.not(p,kt)),kt),Word.zero(p),capturable_word(p,at,kt)))

def capturable(+a: U64,+k: U64) -> {U64.and(U64.and(a,U64.not(k)),k) == U64.zero() : U64}:
  match a k:
    case U64{U32{+al},U32{+ah}} U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.and(32n,al,Word.not(32n,kl)),kl)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,kh)),kh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,kh)),kh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,kh)),kh)}},Word.and(32n,Word.and(32n,al,Word.not(32n,kl)),kl),Word.zero(32n),capturable_word(32n,al,kl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.and(32n,ah,Word.not(32n,kh)),kh),Word.zero(32n),capturable_word(32n,ah,kh)))

def union_clear_bool(+a: Bool,+b: Bool,+k: Bool,+e0: {Bool.and(a,k) == False{} : Bool},+e1: {Bool.and(b,k) == False{} : Bool}) -> {Bool.and(Bool.or(a,b),k) == False{} : Bool}:
  match a b k:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: Old.false_true({Bool.and(Bool.or(False{},True{}),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e1))
    case True{} False{} False{}: {==}
    case True{} False{} True{}: Old.false_true({Bool.and(Bool.or(True{},False{}),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e0))
    case True{} True{} False{}: {==}
    case True{} True{} True{}: Old.false_true({Bool.and(Bool.or(True{},True{}),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e0))

def union_clear_word(+n: Nat,+a: Word(n),+b: Word(n),+k: Word(n),+e0: {Word.and(n,a,k) == Word.zero(n) : Word(n)},+e1: {Word.and(n,b,k) == Word.zero(n) : Word(n)}) -> {Word.and(n,Word.or(n,a,b),k) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match a b k:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a b k:
        case WCon{+ah,+at} WCon{+bh,+bt} WCon{+kh,+kt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),kh),Word.and(p,Word.or(p,at,bt),kt)},WCon{False{},Word.and(p,Word.or(p,at,bt),kt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),kt)},Bool.and(Bool.or(ah,bh),kh),False{},union_clear_bool(ah,bh,kh,Equal.cong(Word(1n+p),Bool,w => OldFrame.head(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Bool,w => OldFrame.head(p,w),Word.and(1n+p,b,k),Word.zero(1n+p),e1))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.or(p,at,bt),kt),Word.zero(p),union_clear_word(p,at,bt,kt,Equal.cong(Word(1n+p),Word(p),w => OldFrame.tail(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Word(p),w => OldFrame.tail(p,w),Word.and(1n+p,b,k),Word.zero(1n+p),e1))))

def union_clear(+a: U64,+b: U64,+k: U64,+e0: {U64.and(a,k) == U64.zero() : U64},+e1: {U64.and(b,k) == U64.zero() : U64}) -> {U64.and(U64.or(a,b),k) == U64.zero() : U64}:
  match a b k:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}} U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),kl)},U32{Word.and(32n,Word.or(32n,ah,bh),kh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.or(32n,ah,bh),kh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),kh)}},Word.and(32n,Word.or(32n,al,bl),kl),Word.zero(32n),union_clear_word(32n,al,bl,kl,Equal.cong(U64,Word(32n),OldFrame.low,U64.and(a,k),U64.zero(),e0),Equal.cong(U64,Word(32n),OldFrame.low,U64.and(b,k),U64.zero(),e1))),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.or(32n,ah,bh),kh),Word.zero(32n),union_clear_word(32n,ah,bh,kh,Equal.cong(U64,Word(32n),OldFrame.high,U64.and(a,k),U64.zero(),e0),Equal.cong(U64,Word(32n),OldFrame.high,U64.and(b,k),U64.zero(),e1))))

def gate_clear_bool(+a: Bool,+b: Bool,+k: Bool,+e0: {Bool.and(a,k) == False{} : Bool}) -> {Bool.and(Bool.and(a,b),k) == False{} : Bool}:
  match a b k:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: Old.false_true({Bool.and(Bool.and(True{},True{}),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e0))

def gate_clear_word(+n: Nat,+a: Word(n),+b: Word(n),+k: Word(n),+e0: {Word.and(n,a,k) == Word.zero(n) : Word(n)}) -> {Word.and(n,Word.and(n,a,b),k) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match a b k:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a b k:
        case WCon{+ah,+at} WCon{+bh,+bt} WCon{+kh,+kt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.and(ah,bh),kh),Word.and(p,Word.and(p,at,bt),kt)},WCon{False{},Word.and(p,Word.and(p,at,bt),kt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.and(p,at,bt),kt)},Bool.and(Bool.and(ah,bh),kh),False{},gate_clear_bool(ah,bh,kh,Equal.cong(Word(1n+p),Bool,w => OldFrame.head(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),e0))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.and(p,at,bt),kt),Word.zero(p),gate_clear_word(p,at,bt,kt,Equal.cong(Word(1n+p),Word(p),w => OldFrame.tail(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),e0))))

def gate_clear(+a: U64,+b: U64,+k: U64,+e0: {U64.and(a,k) == U64.zero() : U64}) -> {U64.and(U64.and(a,b),k) == U64.zero() : U64}:
  match a b k:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}} U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.and(32n,al,bl),kl)},U32{Word.and(32n,Word.and(32n,ah,bh),kh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.and(32n,ah,bh),kh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.and(32n,ah,bh),kh)}},Word.and(32n,Word.and(32n,al,bl),kl),Word.zero(32n),gate_clear_word(32n,al,bl,kl,Equal.cong(U64,Word(32n),OldFrame.low,U64.and(a,k),U64.zero(),e0))),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.and(32n,ah,bh),kh),Word.zero(32n),gate_clear_word(32n,ah,bh,kh,Equal.cong(U64,Word(32n),OldFrame.high,U64.and(a,k),U64.zero(),e0))))

def empty_guard_bool(+a: Bool,+b: Bool) -> {Bool.and(Bool.and(a,Bool.not(b)),b) == False{} : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def empty_guard_word(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,Word.and(n,a,Word.not(n,b)),b) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{+ah,+at} WCon{+bh,+bt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.and(ah,Bool.not(bh)),bh),Word.and(p,Word.and(p,at,Word.not(p,bt)),bt)},WCon{False{},Word.and(p,Word.and(p,at,Word.not(p,bt)),bt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.and(p,at,Word.not(p,bt)),bt)},Bool.and(Bool.and(ah,Bool.not(bh)),bh),False{},empty_guard_bool(ah,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.and(p,at,Word.not(p,bt)),bt),Word.zero(p),empty_guard_word(p,at,bt)))

def empty_guard(+a: U64,+b: U64) -> {U64.and(U64.and(a,U64.not(b)),b) == U64.zero() : U64}:
  match a b:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.and(32n,al,Word.not(32n,bl)),bl)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,bh)),bh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,bh)),bh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.and(32n,ah,Word.not(32n,bh)),bh)}},Word.and(32n,Word.and(32n,al,Word.not(32n,bl)),bl),Word.zero(32n),empty_guard_word(32n,al,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.and(32n,ah,Word.not(32n,bh)),bh),Word.zero(32n),empty_guard_word(32n,ah,bh)))

def flip(+a: U64,+b: U64,e: D.disjoint(a,b)) -> D.disjoint(b,a):
  Equal.trans(U64,U64.and(b,a),U64.and(a,b),U64.zero(),and_commute(b,a),e)

def word_zero_sound(+n: Nat,+w: Word(n),e: {Zero.is_zero_word(n,w) == True{} : Bool}) ->
  {w == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},t,Word.zero(p),word_zero_sound(p,t,e))
        case WCon{True{},t}: Old.false_true({w == Word.zero(1n+p) : Word(1n+p)},e)

def zero_sound(+a: U64,e: {U64.is_zero(a) == True{} : Bool}) -> {a == U64.zero() : U64}:
  Equal.trans(U64,a,U64.from_word(U64.to_word(a)),U64.zero(),
    Equal.sym(U64,U64.from_word(U64.to_word(a)),a,Old.from_to(a)),
    Equal.cong(Word(64n),U64,w => U64.from_word(w),U64.to_word(a),Word.zero(64n),
      word_zero_sound(64n,U64.to_word(a),Equal.trans(Bool,Zero.is_zero_word(64n,U64.to_word(a)),U64.is_zero(a),True{},
        Equal.sym(Bool,U64.is_zero(a),Zero.is_zero_word(64n,U64.to_word(a)),Bridge.u64_is_zero_word(a)),e))))

def not_false(+b: Bool,e: {Bool.not(b) == False{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: Old.false_true({False{} == True{} : Bool},Equal.sym(Bool,True{},False{},e))

# No range restriction: actual out-of-range U64.bit is zero.
def test_zero(+a: U64,+i: Nat,e: {U64.test_bit(a,i) == False{} : Bool}) -> D.disjoint(U64.bit(i),a):
  flip(a,U64.bit(i),zero_sound(U64.and(a,U64.bit(i)),
    Equal.trans(Bool,U64.is_zero(U64.and(a,U64.bit(i))),U64.is_eq(U64.and(a,U64.bit(i)),U64.zero()),True{},
      Equal.sym(Bool,U64.is_eq(U64.and(a,U64.bit(i)),U64.zero()),U64.is_zero(U64.and(a,U64.bit(i))),EqZero.eq_zero_is_zero(U64.and(a,U64.bit(i)))),
      not_false(U64.is_eq(U64.and(a,U64.bit(i)),U64.zero()),e))))
