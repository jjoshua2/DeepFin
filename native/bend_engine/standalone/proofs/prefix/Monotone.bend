import Base
import ./Spec.bend as S
import ./Facts.bend as Facts
import ./Nats.bend as Nats
import ./Order.bend as Order

def walk(+n: Nat, +k: Nat, +valid: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool}) ->
  {U32.is_le(S.prefix(k),S.prefix(Nat.add(n,k))) == True{} : Bool}:
  match n:
    case 0n: Order.refl(S.prefix(k))
    case 1n+p:
      +mid = Nat.add(p,k)
      Order.trans(S.prefix(k),S.prefix(mid),S.prefix(1n+mid),
        walk(p,k,Nats.weaken(mid,128n,valid)),
        Order.lt_le(S.prefix(mid),S.prefix(1n+mid),Facts.increasing(mid,Facts.transitions(mid,valid))))

def ordered(+i: Nat, +j: Nat, +le: {Nat.is_le(i,j) == True{} : Bool}, valid: {Nat.is_lt(j,129n) == True{} : Bool}) ->
  {U32.is_le(S.prefix(i),S.prefix(j)) == True{} : Bool}:
  +same = Nats.subtract_back(i,j,le)
  %same : {U32.is_le(S.prefix(i),S.prefix(_)) == True{} : Bool}
  walk(Nat.sub(j,i),i,Equal.trans(Bool,Nat.is_lt(Nat.add(Nat.sub(j,i),i),129n),Nat.is_lt(j,129n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,129n),Nat.add(Nat.sub(j,i),i),j,same),valid))
