import Base
import ./Increment.bend as Inc
import ../prefix/Nats.bend as P
import ../layout/Low.bend as Low
import ../relative/Order.bend as Order
import ../u64/Words.bend as Old

def le_self(n: Nat) -> {Nat.is_le(n,n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: le_self(p)

def le_trans(+a: Nat,+b: Nat,+c: Nat,
  ab: {Nat.is_le(a,b) == True{} : Bool},bc: {Nat.is_le(b,c) == True{} : Bool}) ->
  {Nat.is_le(a,c) == True{} : Bool}:
  match a b c:
    case 0n _ _: Low.zero_le(c)
    case 1n+p 0n _: Old.false_true({Nat.is_le(1n+p,c) == True{} : Bool},ab)
    case 1n+p 1n+q 0n: bc
    case 1n+p 1n+q 1n+r: le_trans(p,q,r,ab,bc)

def suc_le_lt(+a: Nat,+b: Nat,e: {Nat.is_le(1n+a,b) == True{} : Bool}) ->
  {Nat.is_lt(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n: e
    case 0n 1n+q: {==}
    case 1n+p 0n: e
    case 1n+p 1n+q: suc_le_lt(p,q,e)

def comm(+a: Nat,+b: Nat) -> {Nat.add(a,b) == Nat.add(b,a) : Nat}:
  match a:
    case 0n: Equal.sym(Nat,Nat.add(b,0n),b,P.right_zero(b))
    case 1n+p:
      Equal.trans(Nat,1n+Nat.add(p,b),1n+Nat.add(b,p),Nat.add(b,1n+p),
        Equal.cong(Nat,Nat,n=>1n+n,Nat.add(p,b),Nat.add(b,p),comm(p,b)),
        Equal.sym(Nat,Nat.add(b,1n+p),1n+Nat.add(b,p),P.add_suc(b,p)))

def to_lt(+a: U32,+b: U32,e: {U32.is_lt(a,b) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,Nat.is_lt(Word.to_nat(32n,x),Word.to_nat(32n,y)),U32.is_lt(U32{x},U32{y}),True{},
        Equal.sym(Bool,U32.is_lt(U32{x},U32{y}),Nat.is_lt(Word.to_nat(32n,x),Word.to_nat(32n,y)),
          Equal.cong(Cmp,Bool,c=>Cmp.is_lt(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))),e)

def to_le(+a: U32,+b: U32,e: {U32.is_le(a,b) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,Nat.is_le(Word.to_nat(32n,x),Word.to_nat(32n,y)),U32.is_le(U32{x},U32{y}),True{},
        Equal.sym(Bool,U32.is_le(U32{x},U32{y}),Nat.is_le(Word.to_nat(32n,x),Word.to_nat(32n,y)),
          Equal.cong(Cmp,Bool,c=>Cmp.is_le(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))),e)

def room_head(+n: Nat,+at: U32,+end: U32,
  e: {Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {U32.is_lt(at,end) == True{} : Bool}:
  Order.from_nat_lt(at,end,P.strip(n,U32.to_nat(at),U32.to_nat(end),suc_le_lt(Nat.add(n,U32.to_nat(at)),U32.to_nat(end),e)))

def next_room(+n: Nat,+at: U32,+end: U32,
  safe: {U32.is_lt(at,131072) == True{} : Bool},
  e: {Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(n,U32.to_nat(U32.inc(at))),U32.to_nat(end)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(Nat.add(n,U32.to_nat(U32.inc(at))),U32.to_nat(end)),
    Nat.is_le(Nat.add(n,1n+U32.to_nat(at)),U32.to_nat(end)),True{},
    Equal.cong(Nat,Bool,x=>Nat.is_le(Nat.add(n,x),U32.to_nat(end)),U32.to_nat(U32.inc(at)),1n+U32.to_nat(at),Inc.exact(at,safe)),
    Equal.trans(Bool,Nat.is_le(Nat.add(n,1n+U32.to_nat(at)),U32.to_nat(end)),
      Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)),True{},
      Equal.cong(Nat,Bool,x=>Nat.is_le(x,U32.to_nat(end)),Nat.add(n,1n+U32.to_nat(at)),1n+Nat.add(n,U32.to_nat(at)),P.add_suc(n,U32.to_nat(at))),e))

def before_next(+j: U32,+at: U32,
  safe: {U32.is_lt(at,131072) == True{} : Bool},e: {U32.is_lt(j,at) == True{} : Bool}) ->
  {U32.is_lt(j,U32.inc(at)) == True{} : Bool}:
  Order.from_nat_lt(j,U32.inc(at),Equal.trans(Bool,Nat.is_lt(U32.to_nat(j),U32.to_nat(U32.inc(at))),
    Nat.is_lt(U32.to_nat(j),1n+U32.to_nat(at)),True{},
    Equal.cong(Nat,Bool,x=>Nat.is_lt(U32.to_nat(j),x),U32.to_nat(U32.inc(at)),1n+U32.to_nat(at),Inc.exact(at,safe)),
    P.weaken(U32.to_nat(j),U32.to_nat(at),to_lt(j,at,e))))

def reverse_nat_ne(+a: Nat,+b: Nat,e: {Nat.is_lt(b,a) == True{} : Bool}) -> {Nat.is_eq(a,b) == False{} : Bool}:
  match a b:
    case 0n 0n: Equal.sym(Bool,False{},True{},e)
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: reverse_nat_ne(p,q,e)

def reverse_ne(+a: U32,+b: U32,e: {U32.is_lt(b,a) == True{} : Bool}) -> {U32.is_eq(a,b) == False{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,U32.is_eq(U32{x},U32{y}),Nat.is_eq(Word.to_nat(32n,x),Word.to_nat(32n,y)),False{},
        Equal.cong(Cmp,Bool,c=>Cmp.is_eq(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y)),
        reverse_nat_ne(Word.to_nat(32n,x),Word.to_nat(32n,y),to_lt(U32{y},U32{x},e)))
