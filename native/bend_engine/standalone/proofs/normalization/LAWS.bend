import Base
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

# Every numeric address in the actual depth-17 allocation, not just block endpoints.
law bounded_mask_identity:
  for +i: U32
  for bound: {U32.is_lt(i,131072) == True{} : Bool}
  {U32.and(i,U32.sub(131072,1)) == i : U32}

law bounded_mask_injective:
  for +i: U32
  for +j: U32
  for ib: {U32.is_lt(i,131072) == True{} : Bool}
  for jb: {U32.is_lt(j,131072) == True{} : Bool}
  for same: {U32.and(i,U32.sub(131072,1)) == U32.and(j,U32.sub(131072,1)) : U32}
  {i == j : U32}

# Capacity is an observation of the actual affine array, not an assumed read result.
law bounded_read_direct:
  for a: Array<U64>
  for +i: U32
  for capacity: {S.size(O.shape(a)) == 131072 : U32}
  for bound: {U32.is_lt(i,131072) == True{} : Bool}
  {Array.get(U64,a,i) == Array.get.go(U64,a,131072,i) : Array<U64> & U64}

law bounded_write_direct:
  for a: Array<U64>
  for +i: U32
  for +v: U64
  for capacity: {S.size(O.shape(a)) == 131072 : U32}
  for bound: {U32.is_lt(i,131072) == True{} : Bool}
  {Array.set(U64,a,i,v) == Array.set.fin(U64,Array.swap.go(U64,a,131072,i,v)) : Array<U64>}

law bounded_separation_direct:
  for a: Array<U64>
  for +i: U32
  for +j: U32
  for capacity: {S.size(O.shape(a)) == 131072 : U32}
  for ib: {U32.is_lt(i,131072) == True{} : Bool}
  for jb: {U32.is_lt(j,131072) == True{} : Bool}
  {S.separate(a,i,j) == S.apart(O.shape(a),131072,i,j) : Bool}
