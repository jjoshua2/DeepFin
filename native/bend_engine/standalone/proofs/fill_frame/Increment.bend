# Numeric successor for bounded actual U32 addresses; no overflowing power literal.
import Base
import ../normalization/Bound.bend as Bound
import ../normalization/Compare.bend as Compare
import ../complete/Arithmetic.bend as Complete
import ../successor/Ordinal.bend as Ordinal
import ../u64/Words.bend as Old

def zero_inc(+n: Nat) ->
  {Word.to_nat(1n+n,Word.inc(1n+n,Word.zero(1n+n))) == 1n : Nat}:
  Equal.cong(Nat,Nat,x => 1n+Nat.double(x),Word.to_nat(n,Word.zero(n)),0n,Ordinal.zero_value(n))

def word(+n: Nat,+k: Nat,+w: Word(n),
  e: {Cmp.is_lt(Word.cmp(n,w,Bound.limit(n,k))) == True{} : Bool}) ->
  {Word.to_nat(n,Word.inc(n,w)) == 1n+Word.to_nat(n,w) : Nat}:
  match n k:
    case 0n _:
      Old.false_true({Word.to_nat(0n,Word.inc(0n,w)) == 1n+Word.to_nat(0n,w) : Nat},e)
    case 1n+p 0n:
      +z = Complete.one(1n+p,w,e)
      %Equal.sym(Word(1n+p),w,Word.zero(1n+p),z) :
        {Word.to_nat(1n+p,Word.inc(1n+p,_)) == 1n+Word.to_nat(1n+p,w) : Nat}
      %Equal.sym(Word(1n+p),w,Word.zero(1n+p),z) :
        {Word.to_nat(1n+p,Word.inc(1n+p,Word.zero(1n+p))) == 1n+Word.to_nat(1n+p,_) : Nat}
      Equal.trans(Nat,Word.to_nat(1n+p,Word.inc(1n+p,Word.zero(1n+p))),1n,
        1n+Word.to_nat(1n+p,Word.zero(1n+p)),zero_inc(p),
        Equal.sym(Nat,1n+Word.to_nat(1n+p,Word.zero(1n+p)),1n,
          Equal.cong(Nat,Nat,x=>1n+x,Word.to_nat(1n+p,Word.zero(1n+p)),0n,Ordinal.zero_value(1n+p))))
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}: {==}
        case WCon{True{},t}:
          Equal.cong(Nat,Nat,x=>Nat.double(x),Word.to_nat(p,Word.inc(p,t)),1n+Word.to_nat(p,t),
            word(p,q,t,Compare.strip_low(True{},Word.cmp(p,t,Bound.limit(p,q)),e)))

def exact(+i: U32,e: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {U32.to_nat(U32.inc(i)) == 1n+U32.to_nat(i) : Nat}:
  match i:
    case U32{w}: word(32n,17n,w,e)
