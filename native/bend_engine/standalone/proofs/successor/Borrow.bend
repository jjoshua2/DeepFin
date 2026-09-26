# Full-width subtraction refinement. All arithmetic observations use Base.
import Base

# c is the two's-complement carry-in: True means no incoming borrow.
def carry_pair(pair: Bool & Bool) -> Bool:
  (sum, carry) = pair
  carry

def carry_bit(a: Bool, b: Bool, c: Bool) -> Bool:
  carry_pair(Bool.full_add(a, Bool.not(b), c))

def carry(n: Nat, a: Word(n), b: Word(n), c: Bool) -> Bool:
  match n:
    case 0n: c
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          carry(p, at, bt, carry_bit(ab, bb, c))

def cmp_carry(t: Cmp, c: Bool) -> Bool:
  match t:
    case LT{}: False{}
    case EQ{}: c
    case GT{}: True{}

def cmp_step(+t: Cmp, +a: Bool, +b: Bool, +c: Bool) ->
  {cmp_carry(t, carry_bit(a, b, c)) == cmp_carry(Word.cmp.fin(a, b, t), c) : Bool}:
  match t:
    case LT{}: {==}
    case GT{}: {==}
    case EQ{}:
      match a b c:
        case False{} False{} False{}: {==}
        case False{} False{} True{}: {==}
        case False{} True{} False{}: {==}
        case False{} True{} True{}: {==}
        case True{} False{} False{}: {==}
        case True{} False{} True{}: {==}
        case True{} True{} False{}: {==}
        case True{} True{} True{}: {==}

def carry_cmp(+n: Nat, +a: Word(n), +b: Word(n), +c: Bool) ->
  {carry(n, a, b, c) == cmp_carry(Word.cmp(n, a, b), c) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          Equal.trans(Bool, carry(p, at, bt, carry_bit(ab, bb, c)),
            cmp_carry(Word.cmp(p, at, bt), carry_bit(ab, bb, c)),
            cmp_carry(Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)), c),
            carry_cmp(p, at, bt, carry_bit(ab, bb, c)),
            cmp_step(Word.cmp(p, at, bt), ab, bb, c))

def cmp_no_borrow(t: Cmp) -> {cmp_carry(t, True{}) == Bool.not(Cmp.is_lt(t)) : Bool}:
  match t:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

def outgoing(+n: Nat, +a: Word(n), +b: Word(n)) ->
  {carry(n, a, b, True{}) == Bool.not(Cmp.is_lt(Word.cmp(n, a, b))) : Bool}:
  Equal.trans(Bool, carry(n, a, b, True{}), cmp_carry(Word.cmp(n, a, b), True{}),
    Bool.not(Cmp.is_lt(Word.cmp(n, a, b))), carry_cmp(n, a, b, True{}),
    cmp_no_borrow(Word.cmp(n, a, b)))

# The whole-word ADC splits at an arbitrary bit, carrying into the high part.
def split(+n: Nat, +m: Nat, +al: Word(n), +ah: Word(m), +bl: Word(n), +bh: Word(m), +c: Bool) ->
  {Word.adc(Nat.add(n,m), Word.join(n,m,al,ah), Word.join(n,m,bl,bh), True{}, c)
    == Word.join(n,m,Word.adc(n,al,bl,True{},c), Word.adc(m,ah,bh,True{},carry(n,al,bl,c))) : Word(Nat.add(n,m))}:
  match n:
    case 0n: {==}
    case 1n+p:
      match al bl c:
        case WCon{False{}, at} WCon{False{}, bt} False{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{True{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},False{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},False{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,False{}))),
            split(p,m,at,ah,bt,bh,False{}))
        case WCon{False{}, at} WCon{False{}, bt} True{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{False{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},True{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},True{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,True{}))),
            split(p,m,at,ah,bt,bh,True{}))
        case WCon{False{}, at} WCon{True{}, bt} False{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{False{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},False{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},False{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,False{}))),
            split(p,m,at,ah,bt,bh,False{}))
        case WCon{False{}, at} WCon{True{}, bt} True{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{True{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},False{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},False{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,False{}))),
            split(p,m,at,ah,bt,bh,False{}))
        case WCon{True{}, at} WCon{False{}, bt} False{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{False{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},True{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},True{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,True{}))),
            split(p,m,at,ah,bt,bh,True{}))
        case WCon{True{}, at} WCon{False{}, bt} True{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{True{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},True{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},True{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,True{}))),
            split(p,m,at,ah,bt,bh,True{}))
        case WCon{True{}, at} WCon{True{}, bt} False{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{True{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},False{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},False{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,False{}))),
            split(p,m,at,ah,bt,bh,False{}))
        case WCon{True{}, at} WCon{True{}, bt} True{}:
          Equal.cong(Word(Nat.add(p,m)), Word(Nat.add(1n+p,m)), w => WCon{False{},w},
            Word.adc(Nat.add(p,m),Word.join(p,m,at,ah),Word.join(p,m,bt,bh),True{},True{}),
            Word.join(p,m,Word.adc(p,at,bt,True{},True{}),Word.adc(m,ah,bh,True{},carry(p,at,bt,True{}))),
            split(p,m,at,ah,bt,bh,True{}))

# A proof-only decrement observation, connected below to actual Word.sub.
def dec(n: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match w:
        case WCon{False{},t}: WCon{True{},dec(p,t)}
        case WCon{True{},t}: WCon{False{},t}

def sub_zero(+n: Nat, +w: Word(n)) -> {Word.sub(n,w,Word.zero(n)) == w : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          %sub_zero(p,t) : {WCon{False{},Word.sub(p,t,Word.zero(p))} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t}:
          %sub_zero(p,t) : {WCon{True{},Word.sub(p,t,Word.zero(p))} == WCon{True{},_} : Word(1n+p)}
          {==}

def adc_dec(+n: Nat, +a: Word(n), +b: Word(n)) ->
  {dec(n,Word.adc(n,a,b,True{},True{})) == Word.adc(n,a,b,True{},False{}) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{False{},at} WCon{False{},bt}:
          %adc_dec(p,at,bt) : {WCon{True{},dec(p,Word.adc(p,at,bt,True{},True{}))} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{False{},at} WCon{True{},bt}:
          {==}
        case WCon{True{},at} WCon{False{},bt}:
          {==}
        case WCon{True{},at} WCon{True{},bt}:
          %adc_dec(p,at,bt) : {WCon{True{},dec(p,Word.adc(p,at,bt,True{},True{}))} == WCon{True{},_} : Word(1n+p)}
          {==}

def borrow_zero(+n: Nat, +w: Word(n)) ->
  {Word.adc(n,w,Word.zero(n),True{},False{}) == dec(n,w) : Word(n)}:
  %sub_zero(n,w) : {Word.adc(n,w,Word.zero(n),True{},False{}) == dec(n,_) : Word(n)}
  Equal.sym(Word(n), dec(n,Word.adc(n,w,Word.zero(n),True{},True{})),
    Word.adc(n,w,Word.zero(n),True{},False{}), adc_dec(n,w,Word.zero(n)))

def sub_one(+n: Nat, +w: Word(1n+n)) ->
  {Word.sub(1n+n,w,WCon{True{},Word.zero(n)}) == dec(1n+n,w) : Word(1n+n)}:
  match w:
    case WCon{False{},t}:
      %borrow_zero(n,t) : {WCon{True{},Word.adc(n,t,Word.zero(n),True{},False{})} == WCon{True{},_} : Word(1n+n)}
      {==}
    case WCon{True{},t}:
      %sub_zero(n,t) : {WCon{False{},Word.sub(n,t,Word.zero(n))} == WCon{False{},_} : Word(1n+n)}
      {==}

def high_borrow(+a: Word(32n), +b: Word(32n), +borrow: Bool) ->
  {U32.sub(U32.sub(U32{a},U32{b}),Bool.to_u32(borrow))
    == U32{Word.adc(32n,a,b,True{},Bool.not(borrow))} : U32}:
  match borrow:
    case False{}:
      %sub_zero(32n,Word.sub(32n,a,b)) : {U32{Word.sub(32n,Word.sub(32n,a,b),Word.zero(32n))} == U32{_} : U32}
      {==}
    case True{}:
      Equal.cong(Word(32n),U32,w => U32{w},
        Word.sub(32n,Word.sub(32n,a,b),WCon{True{},Word.zero(31n)}),
        Word.adc(32n,a,b,True{},False{}),
        Equal.trans(Word(32n),Word.sub(32n,Word.sub(32n,a,b),WCon{True{},Word.zero(31n)}),
          dec(32n,Word.sub(32n,a,b)),Word.adc(32n,a,b,True{},False{}),
          sub_one(31n,Word.sub(32n,a,b)),adc_dec(32n,a,b)))

def u64_sub(+a: U64, +b: U64) ->
  {U64.to_word(U64.sub(a,b)) == Word.sub(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      %Equal.sym(U32, U32.sub(U32.sub(U32{ah},U32{bh}),Bool.to_u32(Cmp.is_lt(Word.cmp(32n,al,bl)))),
        U32{Word.adc(32n,ah,bh,True{},Bool.not(Cmp.is_lt(Word.cmp(32n,al,bl))))},
        high_borrow(ah,bh,Cmp.is_lt(Word.cmp(32n,al,bl)))) :
        {U64.to_word(U64{U32{Word.sub(32n,al,bl)},_})
          == Word.sub(64n,Word.join(32n,32n,al,ah),Word.join(32n,32n,bl,bh)) : Word(64n)}
      %outgoing(32n,al,bl) :
        {Word.join(32n,32n,Word.sub(32n,al,bl),Word.adc(32n,ah,bh,True{},_))
          == Word.sub(64n,Word.join(32n,32n,al,ah),Word.join(32n,32n,bl,bh)) : Word(64n)}
      Equal.sym(Word(64n),
        Word.sub(64n,Word.join(32n,32n,al,ah),Word.join(32n,32n,bl,bh)),
        Word.join(32n,32n,Word.sub(32n,al,bl),Word.adc(32n,ah,bh,True{},carry(32n,al,bl,True{}))),
        split(32n,32n,al,ah,bl,bh,True{}))
