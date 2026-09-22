# Turn certified numeric intervals into actual complete-tree route/write contracts.
# Prefix ordering is supplied by the public proof, never by its public caller.
import Base
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ./Order.bend as Order
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as Complete
import ../complete/Actual.bend as Actual
import ../storage/Observe.bend as O
import ../separation/Spec.bend as Separation

def ordered(+i: Nat,+j: Nat,+ri: U32,+rj: U32,
  bi: S.interval(i,ri),bj: S.interval(j,rj),
  end: {U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))) == Prefix.prefix(1n+i) : U32},
  blocks: {U32.is_le(U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(j)) == True{} : Bool}) ->
  {U32.is_lt(S.address(i,ri),S.address(j,rj)) == True{} : Bool}:
  +ends = Equal.trans(Bool,U32.is_le(Prefix.prefix(1n+i),Prefix.prefix(j)),
    U32.is_le(U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(j)),True{},
    Equal.cong(U32,Bool,k => U32.is_le(k,Prefix.prefix(j)),Prefix.prefix(1n+i),
      U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),
      Equal.sym(U32,U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(1n+i),end)),blocks)
  Order.lt_le(S.address(i,ri),Prefix.prefix(j),S.address(j,rj),
    Order.lt_le(S.address(i,ri),Prefix.prefix(1n+i),Prefix.prefix(j),Bounds.inside(i,ri,bi),ends),Bounds.lower(j,rj,bj))

def routes(a: Array<U64>,+i: Nat,+j: Nat,+ri: U32,+rj: U32,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  bi: S.interval(i,ri),bj: S.interval(j,rj),
  order: {U32.is_lt(S.address(i,ri),S.address(j,rj)) == True{} : Bool}) ->
  {Separation.separate(a,S.address(i,ri),S.address(j,rj)) == True{} : Bool}:
  Actual.paths(a,S.address(i,ri),S.address(j,rj),shape,Bounds.allocation(i,ri,bi),Bounds.allocation(j,rj,bj),
    Order.lt_ne(S.address(i,ri),S.address(j,rj),order))

def write(a: Array<U64>,+i: Nat,+j: Nat,+ri: U32,+rj: U32,+v: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  bi: S.interval(i,ri),bj: S.interval(j,rj),
  order: {U32.is_lt(S.address(i,ri),S.address(j,rj)) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,S.address(i,ri),v),S.address(j,rj)) ==
    (Array.set(U64,a,S.address(i,ri),v),Separation.value(Array.get(U64,a,S.address(j,rj)))) : Array<U64> & U64}:
  Actual.frame(a,S.address(i,ri),S.address(j,rj),v,shape,Bounds.allocation(i,ri,bi),Bounds.allocation(j,rj,bj),
    Order.lt_ne(S.address(i,ri),S.address(j,rj),order))
