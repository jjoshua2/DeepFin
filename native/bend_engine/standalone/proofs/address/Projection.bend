# Exact low-half observation by retained-width induction. Keep the bound symbolic:
# eagerly expanding a concrete mathematical 2^32 exceeds this checker's stack.
import Base
import ../index/Bits.bend as Bits
import ../index/Nats.bend as Nats
import ../index/Words.bend as IndexWords
import ../u64/Words.bend as Old
import ../successor/Ordinal.bend as Ordinal
import ../layout/Low.bend as Low

def lt_le(+a: Nat, +b: Nat, e: {Nat.is_lt(a,b) == True{} : Bool}) ->
  {Nat.is_le(a,b) == True{} : Bool}:
  match a b:
    case 0n _: Low.zero_le(b)
    case 1n+p 0n: e
    case 1n+p 1n+q: lt_le(p,q,e)

def lt_trans(+a: Nat, +b: Nat, +c: Nat,
  ab: {Nat.is_lt(a,b) == True{} : Bool}, bc: {Nat.is_lt(b,c) == True{} : Bool}) ->
  {Nat.is_lt(a,c) == True{} : Bool}:
  Low.le_lt(a,b,c,lt_le(a,b,ab),bc)

def take_zero(+n: Nat, +m: Nat) ->
  {Word.take(n,m,Word.zero(Nat.add(n,m))) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},
        Word.take(p,m,Word.zero(Nat.add(p,m))),Word.zero(p),take_zero(p,m))

def zero_values(+n: Nat, +m: Nat) ->
  {Word.to_nat(n,Word.take(n,m,Word.zero(Nat.add(n,m)))) == Word.to_nat(Nat.add(n,m),Word.zero(Nat.add(n,m))) : Nat}:
  %Equal.sym(Word(n),Word.take(n,m,Word.zero(Nat.add(n,m))),Word.zero(n),take_zero(n,m)) :
    {Word.to_nat(n,_) == Word.to_nat(Nat.add(n,m),Word.zero(Nat.add(n,m))) : Nat}
  Equal.trans(Nat,Word.to_nat(n,Word.zero(n)),0n,Word.to_nat(Nat.add(n,m),Word.zero(Nat.add(n,m))),
    Ordinal.zero_value(n),Equal.sym(Nat,Word.to_nat(Nat.add(n,m),Word.zero(Nat.add(n,m))),0n,Ordinal.zero_value(Nat.add(n,m))))

# k is the symbolic numerical width; n is the number of retained low bits.
def take_exact(+n: Nat, +m: Nat, +k: Nat, +w: Word(Nat.add(n,m)),
  retained: {Nat.is_le(k,n) == True{} : Bool},
  bound: {Nat.is_lt(Word.to_nat(Nat.add(n,m),w),Bits.power(k)) == True{} : Bool}) ->
  {Word.to_nat(n,Word.take(n,m,w)) == Word.to_nat(Nat.add(n,m),w) : Nat}:
  match n k:
    case 0n 0n:
      %Equal.sym(Word(m),w,Word.zero(m),IndexWords.zero_small(m,w,bound)) :
        {Word.to_nat(0n,Word.take(0n,m,_)) == Word.to_nat(m,_) : Nat}
      zero_values(0n,m)
    case 0n 1n+q:
      Old.false_true({Word.to_nat(0n,Word.take(0n,m,w)) == Word.to_nat(m,w) : Nat},retained)
    case 1n+p 0n:
      %Equal.sym(Word(Nat.add(1n+p,m)),w,Word.zero(Nat.add(1n+p,m)),IndexWords.zero_small(Nat.add(1n+p,m),w,bound)) :
        {Word.to_nat(1n+p,Word.take(1n+p,m,_)) == Word.to_nat(Nat.add(1n+p,m),_) : Nat}
      zero_values(1n+p,m)
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}:
          Equal.cong(Nat,Nat,v => Nat.double(v),
            Word.to_nat(p,Word.take(p,m,t)),Word.to_nat(Nat.add(p,m),t),
            take_exact(p,m,q,t,retained,Nats.half_even(Word.to_nat(Nat.add(p,m),t),Bits.power(q),bound)))
        case WCon{True{},t}:
          Equal.cong(Nat,Nat,v => 1n+Nat.double(v),
            Word.to_nat(p,Word.take(p,m,t)),Word.to_nat(Nat.add(p,m),t),
            take_exact(p,m,q,t,retained,Nats.half_odd(Word.to_nat(Nat.add(p,m),t),Bits.power(q),bound)))


def low_exact(+x: U64, +k: Nat,
  retained: {Nat.is_le(k,32n) == True{} : Bool},
  bound: {Nat.is_lt(Bits.value(x),Bits.power(k)) == True{} : Bool}) ->
  {U32.to_nat(U64.low(x)) == Bits.value(x) : Nat}:
  match x:
    case U64{U32{lo},U32{hi}}:
      %Old.take_join(32n,32n,lo,hi) :
        {Word.to_nat(32n,_) == Word.to_nat(64n,Word.join(32n,32n,lo,hi)) : Nat}
      take_exact(32n,32n,k,Word.join(32n,32n,lo,hi),retained,bound)
