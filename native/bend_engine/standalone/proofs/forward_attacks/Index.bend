# A decreasing limb budget supplies every source-square bound.
import Base
import ../decoder/Bits.bend as B
import ../prefix/Nats.bend as N

def source(high: Bool,+i: Nat) -> Nat:
  Bool.pick(Nat,high,32n+i,i)

def limb(a: U64,high: Bool) -> Word(32n):
  match a high:
    case U64{U32{lo},U32{hi}} False{}: lo
    case U64{U32{lo},U32{hi}} True{}: hi

def get(+a: U64,+high: Bool,+i: Nat,bound: {Nat.is_lt(i,32n) == True{} : Bool}) ->
  {B.get(32n,limb(a,high),i) == B.get64(a,source(high,i)) : Bool}:
  match a high:
    case U64{U32{+lo},U32{+hi}} False{}:
      Equal.sym(Bool,B.get64(U64{U32{lo},U32{hi}},i),B.get(32n,lo,i),
        Equal.cong(Bool,Bool,c => Bool.pick(Bool,c,B.get(32n,lo,i),B.get(32n,hi,Nat.sub(i,32n))),
          Nat.is_lt(i,32n),True{},bound))
    case U64{U32{+lo},U32{+hi}} True{}:
      match i:
        case 0n: {==}
        case 1n+j: {==}

def bounded(high: Bool,+i: Nat,bound: {Nat.is_lt(i,32n) == True{} : Bool}) ->
  {Nat.is_lt(source(high,i),64n) == True{} : Bool}:
  match high:
    case False{}: N.strip(32n,i,64n,bound)
    case True{}: bound

def advance(high: Bool,+i: Nat) -> {source(high,1n+i) == 1n+source(high,i) : Nat}:
  match high:
    case False{}: {==}
    case True{}: {==}

def next_bound(+n: Nat,+i: Nat,bound: {Nat.is_lt(Nat.add(1n+n,i),33n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(n,1n+i),33n) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Nat.add(n,1n+i),33n),Nat.is_lt(Nat.add(1n+n,i),33n),True{},
    Equal.cong(Nat,Bool,j => Nat.is_lt(j,33n),Nat.add(n,1n+i),1n+Nat.add(n,i),N.add_suc(n,i)),bound)
