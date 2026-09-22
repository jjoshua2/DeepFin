# Reuse proved actual-size/endpoints; no new endpoint table or assumed carry certificate.
import Base
import ./Spec.bend as S
import ./Add.bend as Add
import ./Order.bend as Order
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout

def endpoint(+a: U32,+size: U32,+end: U32,
  wide: {U64.add(U64.from_u32(a),U64.from_u32(size)) == U64.from_u32(end) : U64}) ->
  {U32.add(a,size) == end : U32}:
  Equal.cong(U64,U32,w => U64.low(w),U64.add(U64.from_u32(a),U64.from_u32(size)),U64.from_u32(end),wide)

def upper(+k: Nat,pair: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}
  & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) -> {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}:
  (lo,hi) = pair
  hi

def finish(+k: Nat,+r: U32,
  wide: {U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(Layout.size(U32.from_nat(k)))) == U64.from_u32(Prefix.prefix(1n+k)) : U64},
  limit: {U32.is_lt(Prefix.prefix(1n+k),131072) == True{} : Bool},
  pair: {Add.carry32(Prefix.prefix(k),r) == False{} : Bool}
    & {U32.is_lt(S.address(k,r),U32.add(Prefix.prefix(k),Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  S.interval(k,r) & {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}:
  (+safe,inside) = pair
  +end = endpoint(Prefix.prefix(k),Layout.size(U32.from_nat(k)),Prefix.prefix(1n+k),wide)
  +bounded = Equal.trans(Bool,U32.is_lt(S.address(k,r),Prefix.prefix(1n+k)),
    U32.is_lt(S.address(k,r),U32.add(Prefix.prefix(k),Layout.size(U32.from_nat(k)))),True{},
    Equal.cong(U32,Bool,b => U32.is_lt(S.address(k,r),b),Prefix.prefix(1n+k),
      U32.add(Prefix.prefix(k),Layout.size(U32.from_nat(k))),
      Equal.sym(U32,U32.add(Prefix.prefix(k),Layout.size(U32.from_nat(k))),Prefix.prefix(1n+k),end)),inside)
  ((Order.reverse_le(Prefix.prefix(k),S.address(k,r),Add.no_wrap(Prefix.prefix(k),r,safe)),
    (bounded,Order.lt_lt(S.address(k,r),Prefix.prefix(1n+k),131072,bounded,limit))),Add.widening(Prefix.prefix(k),r,safe))


def interval(+k: Nat,+r: U32,pair: S.interval(k,r)
  & {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}) -> S.interval(k,r):
  (bounds,wide) = pair
  bounds

def widening(+k: Nat,+r: U32,pair: S.interval(k,r)
  & {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}) ->
  {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}:
  (bounds,wide) = pair
  wide

def lower(+k: Nat,+r: U32,bounds: S.interval(k,r)) -> {U32.is_le(Prefix.prefix(k),S.address(k,r)) == True{} : Bool}:
  (lo,rest) = bounds
  lo

def inside(+k: Nat,+r: U32,bounds: S.interval(k,r)) -> {U32.is_lt(S.address(k,r),Prefix.prefix(1n+k)) == True{} : Bool}:
  (lo,(hi,cap)) = bounds
  hi

def allocation(+k: Nat,+r: U32,bounds: S.interval(k,r)) -> {U32.is_lt(S.address(k,r),131072) == True{} : Bool}:
  (lo,(hi,cap)) = bounds
  cap
