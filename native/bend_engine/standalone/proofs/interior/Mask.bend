# Structural bound-to-mask bridge for interior addresses, not an enumerated oracle.
import Base
import ../u64/Words.bend as Old
import ../Mask.bend as Absorb

def low(n: Nat, k: Nat) -> Word(n):
  match n k:
    case 0n _: WNil{}
    case 1n+p 0n: Word.zero(1n+p)
    case 1n+p 1n+q: WCon{True{},low(p,q)}

def power(n: Nat, k: Nat) -> Word(n):
  match n k:
    case 0n _: WNil{}
    case 1n+p 0n: WCon{True{},Word.zero(p)}
    case 1n+p 1n+q: WCon{False{},power(p,q)}

def tail_le(+a: Bool, +b: Bool, +c: Cmp,
  e: {Cmp.is_le(Word.cmp.fin(a,b,c)) == True{} : Bool}) ->
  {Cmp.is_le(c) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: e

def tail_lt(+a: Bool, +c: Cmp,
  e: {Cmp.is_lt(Word.cmp.fin(a,False{},c)) == True{} : Bool}) ->
  {Cmp.is_lt(c) == True{} : Bool}:
  match a c:
    case _ LT{}: {==}
    case _ GT{}: e
    case False{} EQ{}: e
    case True{} EQ{}: e

def weak(c: Cmp, e: {Cmp.is_lt(c) == True{} : Bool}) ->
  {Cmp.is_le(c) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: e

def zero_cmp(n: Nat) -> {Word.cmp(n,Word.zero(n),Word.zero(n)) == EQ{} : Cmp}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Cmp,Cmp,c => Word.cmp.fin(False{},False{},c),
        Word.cmp(p,Word.zero(p),Word.zero(p)),EQ{},zero_cmp(p))

def cmp_tail_zero(+n: Nat, +a: Bool, +b: Bool, +t: Word(n), same: {t == Word.zero(n) : Word(n)}) ->
  {Word.cmp(1n+n,WCon{a,t},WCon{b,Word.zero(n)}) == Bool.cmp(a,b) : Cmp}:
  Equal.trans(Cmp,Word.cmp(1n+n,WCon{a,t},WCon{b,Word.zero(n)}),
    Word.cmp.fin(a,b,Word.cmp(n,Word.zero(n),Word.zero(n))),Bool.cmp(a,b),
    Equal.cong(Word(n),Cmp,x => Word.cmp(1n+n,WCon{a,x},WCon{b,Word.zero(n)}),t,Word.zero(n),same),
    Equal.cong(Cmp,Cmp,c => Word.cmp.fin(a,b,c),Word.cmp(n,Word.zero(n),Word.zero(n)),EQ{},zero_cmp(n)))

def zero(+n: Nat, +w: Word(n), +e: {Cmp.is_le(Word.cmp(n,w,Word.zero(n))) == True{} : Bool}) ->
  {w == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},t,Word.zero(p),
            zero(p,t,tail_le(False{},False{},Word.cmp(p,t,Word.zero(p)),e)))
        case WCon{True{},+t}:
          +same = zero(p,t,tail_le(True{},False{},Word.cmp(p,t,Word.zero(p)),e))
          Old.false_true({WCon{True{},t} == Word.zero(1n+p) : Word(1n+p)},
            Equal.trans(Bool,False{},Cmp.is_le(Word.cmp(1n+p,WCon{True{},t},Word.zero(1n+p))),True{},
              Equal.cong(Cmp,Bool,c => Cmp.is_le(c),GT{},Word.cmp(1n+p,WCon{True{},t},Word.zero(1n+p)),
                Equal.sym(Cmp,Word.cmp(1n+p,WCon{True{},t},Word.zero(1n+p)),GT{},cmp_tail_zero(p,True{},False{},t,same))),e))

def one(+n: Nat, +w: Word(1n+n), +e: {Cmp.is_lt(Word.cmp(1n+n,w,power(1n+n,0n))) == True{} : Bool}) ->
  {w == Word.zero(1n+n) : Word(1n+n)}:
  match w:
    case WCon{False{},t}:
      Equal.cong(Word(n),Word(1n+n),x => WCon{False{},x},t,Word.zero(n),
        zero(n,t,tail_le(False{},True{},Word.cmp(n,t,Word.zero(n)),weak(Word.cmp(1n+n,WCon{False{},t},power(1n+n,0n)),e))))
    case WCon{True{},+t}:
      +same = zero(n,t,tail_le(True{},True{},Word.cmp(n,t,Word.zero(n)),weak(Word.cmp(1n+n,WCon{True{},t},power(1n+n,0n)),e)))
      Old.false_true({WCon{True{},t} == Word.zero(1n+n) : Word(1n+n)},
        Equal.trans(Bool,False{},Cmp.is_lt(Word.cmp(1n+n,WCon{True{},t},power(1n+n,0n))),True{},
          Equal.cong(Cmp,Bool,c => Cmp.is_lt(c),EQ{},Word.cmp(1n+n,WCon{True{},t},power(1n+n,0n)),
            Equal.sym(Cmp,Word.cmp(1n+n,WCon{True{},t},power(1n+n,0n)),EQ{},cmp_tail_zero(n,True{},True{},t,same))),e))

def masked(+n: Nat, +k: Nat, +w: Word(n), e: {Cmp.is_lt(Word.cmp(n,w,power(n,k))) == True{} : Bool}) ->
  {Word.and(n,w,low(n,k)) == w : Word(n)}:
  match n k:
    case 0n _:
      match w:
        case WNil{}: {==}
    case 1n+p 0n:
      +same = one(p,w,e)
      %Equal.sym(Word(1n+p),w,Word.zero(1n+p),same) :
        {Word.and(1n+p,_,Word.zero(1n+p)) == _ : Word(1n+p)}
      Absorb.zero_left(1n+p,Word.zero(1n+p))
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,low(p,q)),t,
            masked(p,q,t,tail_lt(False{},Word.cmp(p,t,power(p,q)),e)))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},Word.and(p,t,low(p,q)),t,
            masked(p,q,t,tail_lt(True{},Word.cmp(p,t,power(p,q)),e)))

def allocation(+x: U32, e: {U32.is_lt(x,131072) == True{} : Bool}) ->
  {U32.and(x,131071) == x : U32}:
  match x:
    case U32{w}:
      Equal.cong(Word(32n),U32,v => U32{v},Word.and(32n,w,low(32n,17n)),w,masked(32n,17n,w,e))
