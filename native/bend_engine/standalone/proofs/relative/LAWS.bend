import Base
import ./Spec.bend as S
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as Separation

law relative_address_bounds:
  for +k: Nat
  for +r: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for rb: {U32.is_lt(r,Layout.size(U32.from_nat(k))) == True{} : Bool}
  S.interval(k,r)

law relative_addition_exact:
  for +k: Nat
  for +r: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for rb: {U32.is_lt(r,Layout.size(U32.from_nat(k))) == True{} : Bool}
  {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}

law lookup_address_bounds:
  for +k: Nat
  for +occ: U64
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  S.interval(k,S.index(k,occ))

law ordered_block_addresses:
  for +i: Nat
  for +j: Nat
  for +ri: U32
  for +rj: U32
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  for rib: {U32.is_lt(ri,Layout.size(U32.from_nat(i))) == True{} : Bool}
  for rjb: {U32.is_lt(rj,Layout.size(U32.from_nat(j))) == True{} : Bool}
  {U32.is_lt(S.address(i,ri),S.address(j,rj)) == True{} : Bool}

law ordered_block_address_paths:
  for a: Array<U64>
  for +i: Nat
  for +j: Nat
  for +ri: U32
  for +rj: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  for +rib: {U32.is_lt(ri,Layout.size(U32.from_nat(i))) == True{} : Bool}
  for +rjb: {U32.is_lt(rj,Layout.size(U32.from_nat(j))) == True{} : Bool}
  {Separation.separate(a,S.address(i,ri),S.address(j,rj)) == True{} : Bool}

law lookup_other_block_write:
  for a: Array<U64>
  for +i: Nat
  for +j: Nat
  for +occi: U64
  for +occj: U64
  for +v: U64
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  {Array.get(U64,Array.set(U64,a,S.address(i,S.index(i,occi)),v),S.address(j,S.index(j,occj))) ==
    (Array.set(U64,a,S.address(i,S.index(i,occi)),v),
      Separation.value(Array.get(U64,a,S.address(j,S.index(j,occj))))) : Array<U64> & U64}
