# The actual low-half projection cannot increase a word's Nat value.
import Base
import ../index/Bits.bend as Bits
import ../u64/Words.bend as Old
import ./Domain.bend as Domain

def zero_le(n: Nat) -> {Nat.is_le(0n,n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: {==}

def double_le(+a: Nat, +b: Nat, e: {Nat.is_le(a,b) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(a),Nat.double(b)) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n:
      Old.false_true({Nat.is_le(Nat.double(1n+p),0n) == True{} : Bool},e)
    case 1n+p 1n+q: double_le(p,q,e)

def take_le(+n: Nat, +m: Nat, +w: Word(Nat.add(n,m))) ->
  {Nat.is_le(Word.to_nat(n,Word.take(n,m,w)),Word.to_nat(Nat.add(n,m),w)) == True{} : Bool}:
  match n:
    case 0n: zero_le(Word.to_nat(m,w))
    case 1n+p:
      match w:
        case WCon{False{},t}:
          double_le(Word.to_nat(p,Word.take(p,m,t)),Word.to_nat(Nat.add(p,m),t),take_le(p,m,t))
        case WCon{True{},t}:
          double_le(Word.to_nat(p,Word.take(p,m,t)),Word.to_nat(Nat.add(p,m),t),take_le(p,m,t))

def low_le(+x: U64) -> {Nat.is_le(U32.to_nat(U64.low(x)),Bits.value(x)) == True{} : Bool}:
  match x:
    case U64{U32{lo},U32{hi}}:
      %Old.take_join(32n,32n,lo,hi) :
        {Nat.is_le(Word.to_nat(32n,_),Word.to_nat(64n,Word.join(32n,32n,lo,hi))) == True{} : Bool}
      take_le(32n,32n,Word.join(32n,32n,lo,hi))

def le_lt(+a: Nat, +b: Nat, +c: Nat,
  ab: {Nat.is_le(a,b) == True{} : Bool}, bc: {Nat.is_lt(b,c) == True{} : Bool}) ->
  {Nat.is_lt(a,c) == True{} : Bool}:
  match a b c:
    case 0n _ 0n: Domain.impossible({Nat.is_lt(0n,0n) == True{} : Bool},b,bc)
    case 0n _ 1n+r: {==}
    case 1n+p 0n _:
      Old.false_true({Nat.is_lt(1n+p,c) == True{} : Bool},ab)
    case 1n+p 1n+q 0n:
      Old.false_true({Nat.is_lt(1n+p,0n) == True{} : Bool},bc)
    case 1n+p 1n+q 1n+r: le_lt(p,q,r,ab,bc)
