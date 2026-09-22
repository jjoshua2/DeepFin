import Base
import ./Spec.bend as S
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

law prefix_widening:
  for +k: Nat
  for +valid: {Nat.is_lt(k,129n) == True{} : Bool}
  {U64.from_u32(S.prefix(k)) == S.wide(k) : U64}

law prefix_in_allocation:
  for +k: Nat
  for +valid: {Nat.is_lt(k,129n) == True{} : Bool}
  {U32.is_le(512,S.prefix(k)) == True{} : Bool}
  & {U32.is_lt(S.prefix(k),131072) == True{} : Bool}

law step_no_overflow:
  for +k: Nat
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  {U64.add(U64.from_u32(S.prefix(k)),U64.from_u32(Layout.size(U32.from_nat(k)))) == U64.from_u32(S.prefix(1n+k)) : U64}

law blocks_ordered:
  for +i: Nat
  for +j: Nat
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  {U32.is_le(U32.add(S.prefix(i),Layout.size(U32.from_nat(i))),S.prefix(j)) == True{} : Bool}

law tables_follow_prefix:
  for +n: Nat
  for +k: Nat
  for a: Array<U64>
  for +valid: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool}
  {Tables.tables(n,U32.from_nat(k),S.prefix(k),a) == S.scheduled(n,k,a) : Array<U64>}

law logical_end:
  {S.prefix(128n) == 108160 : U32}
