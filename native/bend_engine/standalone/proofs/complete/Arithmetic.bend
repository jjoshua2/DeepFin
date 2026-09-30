# Unsigned half-interval and subtraction cancellation lemmas for real U32 routes.
import Base
import ../normalization/Bound.bend as Bound
import ../normalization/Compare.bend as Compare
import ../u64/Words.bend as Old

def cap(d: Nat) -> U32:
  U32{Bound.limit(32n,d)}

def one(+n: Nat, +w: Word(n), +e: {Cmp.is_lt(Word.cmp(n,w,Bound.limit(n,0n))) == True{} : Bool}) ->
  {w == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{b,t}:
          Compare.below_one_finish(p,b,t,
            Compare.le_zero(p,t,Compare.tail_of_below_one(b,Word.cmp(p,t,Word.zero(p)),e)),e)

def not_low(+b: Bool, c: Cmp,
  e: {Cmp.is_lt(Word.cmp.fin(b,False{},c)) == False{} : Bool}) ->
  {Cmp.is_lt(c) == False{} : Bool}:
  match c:
    case LT{}: e
    case EQ{}: {==}
    case GT{}: {==}

def lift_low(+b: Bool,c: Cmp,e: {Cmp.is_lt(c) == True{} : Bool}) ->
  {Cmp.is_lt(Word.cmp.fin(b,False{},c)) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: Old.false_true({Cmp.is_lt(Word.cmp.fin(b,False{},EQ{})) == True{} : Bool},e)
    case GT{}: e

def sub_self_zero(n: Nat) -> {Word.sub(n,Word.zero(n),Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.sub(p,Word.zero(p),Word.zero(p)),Word.zero(p),sub_self_zero(p))


def right_one(+p: Nat, +b: Bool, +t: Word(p),
  +same: {t == Word.zero(p) : Word(p)},
  e: {Cmp.is_lt(Word.cmp(1n+p,WCon{b,t},Bound.limit(1n+p,0n))) == False{} : Bool}) ->
  {Cmp.is_lt(Word.cmp(1n+p,Word.sub(1n+p,WCon{b,t},Bound.limit(1n+p,0n)),Bound.limit(1n+p,0n))) == True{} : Bool}:
  match b:
    case False{}:
      Old.false_true({Cmp.is_lt(Word.cmp(1n+p,Word.sub(1n+p,WCon{False{},t},Bound.limit(1n+p,0n)),Bound.limit(1n+p,0n))) == True{} : Bool},
        Equal.sym(Bool,True{},False{},
          Equal.trans(Bool,True{},Cmp.is_lt(Word.cmp(1n+p,WCon{False{},t},Bound.limit(1n+p,0n))),False{},
            Equal.sym(Bool,Cmp.is_lt(Word.cmp(1n+p,WCon{False{},t},Bound.limit(1n+p,0n))),True{},
              Equal.cong(Cmp,Bool,c => Cmp.is_lt(Word.cmp.fin(False{},True{},c)),Word.cmp(p,t,Word.zero(p)),EQ{},Compare.zero_comparison(p,t,same))),e)))
    case True{}:
      %Equal.sym(Word(p),t,Word.zero(p),same) :
        {Cmp.is_lt(Word.cmp(1n+p,Word.sub(1n+p,WCon{True{},_},Bound.limit(1n+p,0n)),Bound.limit(1n+p,0n))) == True{} : Bool}
      Equal.cong(Cmp,Bool,c => Cmp.is_lt(Word.cmp.fin(False{},True{},c)),
        Word.cmp(p,Word.sub(p,Word.zero(p),Word.zero(p)),Word.zero(p)),EQ{},
        Compare.zero_comparison(p,Word.sub(p,Word.zero(p),Word.zero(p)),sub_self_zero(p)))


def right(+n: Nat, +k: Nat, +w: Word(n),
  upper: {Cmp.is_lt(Word.cmp(n,w,Bound.limit(n,1n+k))) == True{} : Bool},
  lower: {Cmp.is_lt(Word.cmp(n,w,Bound.limit(n,k))) == False{} : Bool}) ->
  {Cmp.is_lt(Word.cmp(n,Word.sub(n,w,Bound.limit(n,k)),Bound.limit(n,k))) == True{} : Bool}:
  match n k:
    case 0n _: upper
    case 1n+p 0n:
      match w:
        case WCon{b,t}:
          right_one(p,b,t,one(p,t,Compare.strip_low(b,Word.cmp(p,t,Bound.limit(p,0n)),upper)),lower)
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}:
          lift_low(False{},Word.cmp(p,Word.sub(p,t,Bound.limit(p,q)),Bound.limit(p,q)),
            right(p,q,t,Compare.strip_low(False{},Word.cmp(p,t,Bound.limit(p,1n+q)),upper),
              not_low(False{},Word.cmp(p,t,Bound.limit(p,q)),lower)))
        case WCon{True{},t}:
          lift_low(True{},Word.cmp(p,Word.sub(p,t,Bound.limit(p,q)),Bound.limit(p,q)),
            right(p,q,t,Compare.strip_low(True{},Word.cmp(p,t,Bound.limit(p,1n+q)),upper),
              not_low(True{},Word.cmp(p,t,Bound.limit(p,q)),lower)))

def right32(+k: Nat, +i: U32,
  upper: {U32.is_lt(i,cap(1n+k)) == True{} : Bool},
  lower: {U32.is_lt(i,cap(k)) == False{} : Bool}) ->
  {U32.is_lt(U32.sub(i,cap(k)),cap(k)) == True{} : Bool}:
  match i:
    case U32{w}: right(32n,k,w,upper,lower)


def head(-p: Nat, w: Word(1n+p)) -> Bool:
  match w:
    case WCon{b,t}: b

def tail(-p: Nat, w: Word(1n+p)) -> Word(p):
  match w:
    case WCon{b,t}: t

# Ripple subtraction is injective for a fixed subtrahend and carry-in.
# Only the local full-adder truth table is split; whole words remain arbitrary.
def cancel(+n: Nat, +a: Word(n), +b: Word(n), +z: Word(n), +c: Bool,
  eq: {Word.adc(n,a,z,True{},c) == Word.adc(n,b,z,True{},c) : Word(n)}) ->
  {a == b : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b z c:
        case WCon{False{},at} WCon{False{},bt} WCon{False{},zt} False{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},at,bt,
            cancel(p,at,bt,zt,False{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{False{},zt},True{},False{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{False{},zt},True{},False{}),eq)))
        case WCon{False{},at} WCon{False{},bt} WCon{False{},zt} True{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},at,bt,
            cancel(p,at,bt,zt,True{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{False{},zt},True{},True{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{False{},zt},True{},True{}),eq)))
        case WCon{False{},at} WCon{False{},bt} WCon{True{},zt} False{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},at,bt,
            cancel(p,at,bt,zt,False{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{True{},zt},True{},False{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{True{},zt},True{},False{}),eq)))
        case WCon{False{},at} WCon{False{},bt} WCon{True{},zt} True{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},at,bt,
            cancel(p,at,bt,zt,False{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{True{},zt},True{},True{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{True{},zt},True{},True{}),eq)))
        case WCon{False{},at} WCon{True{},bt} WCon{False{},zt} False{}:
          Old.false_true({WCon{False{},at} == WCon{True{},bt} : Word(1n+p)},
            Equal.sym(Bool,True{},False{},Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{False{},zt},True{},False{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{False{},zt},True{},False{}),eq)))
        case WCon{False{},at} WCon{True{},bt} WCon{False{},zt} True{}:
          Old.false_true({WCon{False{},at} == WCon{True{},bt} : Word(1n+p)},
            Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{False{},zt},True{},True{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{False{},zt},True{},True{}),eq))
        case WCon{False{},at} WCon{True{},bt} WCon{True{},zt} False{}:
          Old.false_true({WCon{False{},at} == WCon{True{},bt} : Word(1n+p)},
            Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{True{},zt},True{},False{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{True{},zt},True{},False{}),eq))
        case WCon{False{},at} WCon{True{},bt} WCon{True{},zt} True{}:
          Old.false_true({WCon{False{},at} == WCon{True{},bt} : Word(1n+p)},
            Equal.sym(Bool,True{},False{},Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{False{},at},WCon{True{},zt},True{},True{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{True{},zt},True{},True{}),eq)))
        case WCon{True{},at} WCon{False{},bt} WCon{False{},zt} False{}:
          Old.false_true({WCon{True{},at} == WCon{False{},bt} : Word(1n+p)},
            Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{False{},zt},True{},False{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{False{},zt},True{},False{}),eq))
        case WCon{True{},at} WCon{False{},bt} WCon{False{},zt} True{}:
          Old.false_true({WCon{True{},at} == WCon{False{},bt} : Word(1n+p)},
            Equal.sym(Bool,True{},False{},Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{False{},zt},True{},True{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{False{},zt},True{},True{}),eq)))
        case WCon{True{},at} WCon{False{},bt} WCon{True{},zt} False{}:
          Old.false_true({WCon{True{},at} == WCon{False{},bt} : Word(1n+p)},
            Equal.sym(Bool,True{},False{},Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{True{},zt},True{},False{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{True{},zt},True{},False{}),eq)))
        case WCon{True{},at} WCon{False{},bt} WCon{True{},zt} True{}:
          Old.false_true({WCon{True{},at} == WCon{False{},bt} : Word(1n+p)},
            Equal.cong(Word(1n+p),Bool,w => head(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{True{},zt},True{},True{}),
              Word.adc(1n+p,WCon{False{},bt},WCon{True{},zt},True{},True{}),eq))
        case WCon{True{},at} WCon{True{},bt} WCon{False{},zt} False{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},at,bt,
            cancel(p,at,bt,zt,True{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{False{},zt},True{},False{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{False{},zt},True{},False{}),eq)))
        case WCon{True{},at} WCon{True{},bt} WCon{False{},zt} True{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},at,bt,
            cancel(p,at,bt,zt,True{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{False{},zt},True{},True{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{False{},zt},True{},True{}),eq)))
        case WCon{True{},at} WCon{True{},bt} WCon{True{},zt} False{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},at,bt,
            cancel(p,at,bt,zt,False{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{True{},zt},True{},False{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{True{},zt},True{},False{}),eq)))
        case WCon{True{},at} WCon{True{},bt} WCon{True{},zt} True{}:
          Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},at,bt,
            cancel(p,at,bt,zt,True{},Equal.cong(Word(1n+p),Word(p),w => tail(p,w),
              Word.adc(1n+p,WCon{True{},at},WCon{True{},zt},True{},True{}),
              Word.adc(1n+p,WCon{True{},bt},WCon{True{},zt},True{},True{}),eq)))

def word32(i: U32) -> Word(32n):
  match i:
    case U32{w}: w

def subtract_injective(+i: U32, +j: U32, +h: U32,
  eq: {U32.sub(i,h) == U32.sub(j,h) : U32}) -> {i == j : U32}:
  match i j h:
    case U32{x} U32{y} U32{z}:
      Equal.cong(Word(32n),U32,w => U32{w},x,y,
        cancel(32n,x,y,z,True{},Equal.cong(U32,Word(32n),u => word32(u),U32.sub(U32{x},U32{z}),U32.sub(U32{y},U32{z}),eq)))

def one32(+i: U32, bound: {U32.is_lt(i,1) == True{} : Bool}) -> {i == 0 : U32}:
  match i:
    case U32{w}: Equal.cong(Word(32n),U32,v => U32{v},w,Word.zero(32n),one(32n,w,bound))

def eq_refl(+i: U32) -> {U32.is_eq(i,i) == True{} : Bool}:
  match i:
    case U32{w}: Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),Word.cmp(32n,w,w),EQ{},Compare.reflexive(32n,w))

def different(+i: U32, +j: U32, ne: {U32.is_eq(i,j) == False{} : Bool}, eq: {i == j : U32}) ->
  {False{} == True{} : Bool}:
  Equal.trans(Bool,False{},U32.is_eq(i,j),True{},Equal.sym(Bool,U32.is_eq(i,j),False{},ne),
    Equal.trans(Bool,U32.is_eq(i,j),U32.is_eq(j,j),True{},Equal.cong(U32,Bool,k => U32.is_eq(k,j),i,j,eq),eq_refl(j)))
