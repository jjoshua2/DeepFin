# Lossless low projection under a symbolic compact-width bound.
# Keep powers symbolic: no huge unary 2^32 literal is normalized by the checker.
import Base
import ../index/Bits.bend as Bits
import ../index/Nats.bend as Nats
import ../u64/Words.bend as Old
import ../layout/Low.bend as Low
import ../layout/Domain.bend as Domain

def lt_one(+n: Nat, e: {Nat.is_lt(n,1n) == True{} : Bool}) -> {0n == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p: Domain.impossible({0n == 1n+p : Nat},p,e)

def take_bounded(+n: Nat, +m: Nat, +k: Nat, +w: Word(Nat.add(n,m)),
  kn: {Nat.is_le(k,n) == True{} : Bool},
  e: {Nat.is_lt(Word.to_nat(Nat.add(n,m),w),Bits.power(k)) == True{} : Bool}) ->
  {Word.to_nat(n,Word.take(n,m,w)) == Word.to_nat(Nat.add(n,m),w) : Nat}:
  match n k:
    case 0n 0n: lt_one(Word.to_nat(m,w),e)
    case 0n 1n+q:
      Old.false_true({Word.to_nat(0n,Word.take(0n,m,w)) == Word.to_nat(m,w) : Nat},kn)
    case 1n+p 0n:
      match w:
        case WCon{False{},t}:
          Equal.cong(Nat,Nat,v => Nat.double(v),Word.to_nat(p,Word.take(p,m,t)),
            Word.to_nat(Nat.add(p,m),t),take_bounded(p,m,0n,t,Low.zero_le(p),
              Nats.double_lt_one(Word.to_nat(Nat.add(p,m),t),e)))
        case WCon{True{},t}:
          Empty.absurd({Word.to_nat(1n+p,Word.take(1n+p,m,WCon{True{},t}))
            == Word.to_nat(Nat.add(1n+p,m),WCon{True{},t}) : Nat},
            Nats.odd_not_one(Word.to_nat(Nat.add(p,m),t),e))
    case 1n+p 1n+q:
      match w:
        case WCon{False{},t}:
          Equal.cong(Nat,Nat,v => Nat.double(v),Word.to_nat(p,Word.take(p,m,t)),
            Word.to_nat(Nat.add(p,m),t),take_bounded(p,m,q,t,kn,
              Nats.half_even(Word.to_nat(Nat.add(p,m),t),Bits.power(q),e)))
        case WCon{True{},t}:
          Equal.cong(Nat,Nat,v => 1n+Nat.double(v),Word.to_nat(p,Word.take(p,m,t)),
            Word.to_nat(Nat.add(p,m),t),take_bounded(p,m,q,t,kn,
              Nats.half_odd(Word.to_nat(Nat.add(p,m),t),Bits.power(q),e)))

def low_exact(+x: U64, +k: Nat, kn: {Nat.is_le(k,32n) == True{} : Bool},
  e: {Nat.is_lt(Bits.value(x),Bits.power(k)) == True{} : Bool}) ->
  {U32.to_nat(U64.low(x)) == Bits.value(x) : Nat}:
  match x:
    case U64{U32{lo},U32{hi}}:
      %Old.take_join(32n,32n,lo,hi) :
        {Word.to_nat(32n,_) == Word.to_nat(64n,Word.join(32n,32n,lo,hi)) : Nat}
      take_bounded(32n,32n,k,Word.join(32n,32n,lo,hi),kn,e)
