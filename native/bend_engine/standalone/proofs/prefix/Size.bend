# Uses existing certified population equality, not a new assumed size oracle.
import Base
import ./Spec.bend as S
import ../layout/Spec.bend as Layout
import ../layout/Geometry.bend as Geometry
import ../layout/Domain.bend as Domain
import ../layout/Consequences.bend as Consequences

def size(+k: Nat, +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  count: Consequences.count(U32.from_nat(k))) ->
  {Layout.size(U32.from_nat(k)) == S.size(k) : U32}:
  +key = U32.from_nat(k)
  +exact = Domain.exact(k,valid)
  +bounded = Equal.trans(Bool,Nat.is_lt(U32.to_nat(key),128n),Nat.is_lt(k,128n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,128n),U32.to_nat(key),k,exact),valid)
  Equal.cong(Nat,U32,n => U32.shln(1,n),U32.to_nat(U64.popcount(Layout.mask(key))),Geometry.population(k),
    Equal.trans(Nat,U32.to_nat(U64.popcount(Layout.mask(key))),Geometry.population(U32.to_nat(key)),Geometry.population(k),
      Consequences.observed(key,bounded,count),Equal.cong(Nat,Nat,n => Geometry.population(n),U32.to_nat(key),k,exact)))

def step(+k: Nat, +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  count: Consequences.count(U32.from_nat(k))) ->
  {U32.add(S.prefix(k),Layout.size(U32.from_nat(k))) == S.prefix(1n+k) : U32}:
  Equal.cong(U32,U32,n => U32.add(S.prefix(k),n),Layout.size(U32.from_nat(k)),S.size(k),size(k,valid,count))
