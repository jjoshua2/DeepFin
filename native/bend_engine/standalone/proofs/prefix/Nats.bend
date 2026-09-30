import Base
import ../u64/Words.bend as Old
import ../layout/Low.bend as Low

# Strip a nonnegative left summand from a strict bound.
def weaken(+n: Nat, +c: Nat, e: {Nat.is_lt(n,c) == True{} : Bool}) ->
  {Nat.is_lt(n,1n+c) == True{} : Bool}:
  match n c:
    case 0n _: {==}
    case 1n+p 0n: Old.false_true({Nat.is_lt(1n+p,1n) == True{} : Bool},e)
    case 1n+p 1n+q: weaken(p,q,e)

def strip(+p: Nat, +k: Nat, +cap: Nat, e: {Nat.is_lt(Nat.add(p,k),cap) == True{} : Bool}) ->
  {Nat.is_lt(k,cap) == True{} : Bool}:
  match p:
    case 0n: e
    case 1n+r:
      match cap:
        case 0n: Old.false_true({Nat.is_lt(k,0n) == True{} : Bool},e)
        case 1n+c: weaken(k,c,strip(r,k,c,e))

def add_suc(+p: Nat, +k: Nat) -> {Nat.add(p,1n+k) == 1n+Nat.add(p,k) : Nat}:
  match p:
    case 0n: {==}
    case 1n+r: Equal.cong(Nat,Nat,n => 1n+n,Nat.add(r,1n+k),1n+Nat.add(r,k),add_suc(r,k))

def next_bound(+p: Nat, +k: Nat, e: {Nat.is_lt(Nat.add(1n+p,k),129n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(p,1n+k),129n) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Nat.add(p,1n+k),129n),Nat.is_lt(Nat.add(1n+p,k),129n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,129n),Nat.add(p,1n+k),1n+Nat.add(p,k),add_suc(p,k)),e)

def right_zero(+n: Nat) -> {Nat.add(n,0n) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p: Equal.cong(Nat,Nat,x => 1n+x,Nat.add(p,0n),p,right_zero(p))

def subtract_back(+i: Nat, +j: Nat, e: {Nat.is_le(i,j) == True{} : Bool}) ->
  {Nat.add(Nat.sub(j,i),i) == j : Nat}:
  match i j:
    case 0n 0n: {==}
    case 0n 1n+q: right_zero(1n+q)
    case 1n+p 0n: Old.false_true({Nat.add(Nat.sub(0n,1n+p),1n+p) == 0n : Nat},e)
    case 1n+p 1n+q:
      Equal.trans(Nat,Nat.add(Nat.sub(q,p),1n+p),1n+Nat.add(Nat.sub(q,p),p),1n+q,
        add_suc(Nat.sub(q,p),p),Equal.cong(Nat,Nat,x => 1n+x,Nat.add(Nat.sub(q,p),p),q,subtract_back(p,q,e)))

def lt_to_le(+i: Nat, +j: Nat, e: {Nat.is_lt(i,j) == True{} : Bool}) ->
  {Nat.is_le(i,j) == True{} : Bool}:
  match i j:
    case 0n _: Low.zero_le(j)
    case 1n+p 0n: e
    case 1n+p 1n+q: lt_to_le(p,q,e)

def lt_suc(+i: Nat, +j: Nat, e: {Nat.is_lt(i,j) == True{} : Bool}) ->
  {Nat.is_le(1n+i,j) == True{} : Bool}:
  match i j:
    case 0n 0n: e
    case 0n 1n+q: Low.zero_le(q)
    case 1n+p 0n: e
    case 1n+p 1n+q: lt_suc(p,q,e)
