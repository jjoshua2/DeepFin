# Construct endpoint and order premises using the already discharged prefix laws.
import Base
import ./Spec.bend as S
import ./Interval.bend as Interval
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Certified.bend as Sizes
import ../prefix/Nats.bend as Nats
import ../layout/Spec.bend as Layout
import ../layout/Low.bend as Low

def first(+k: Nat, r: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool} & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  (lo,hi) = r
  lo

def last(+k: Nat, r: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool} & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}:
  (lo,hi) = r
  hi

def finish(+k: Nat, valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {S.finish(k) == Prefix.prefix(1n+k) : U32}:
  Equal.cong(U32,U32,s => U32.add(Prefix.prefix(k),s),Layout.size(U32.from_nat(k)),Prefix.size(k),Sizes.size(k,valid))

def certificate(+k: Nat, +x: U32, +valid: {Nat.is_lt(k,128n) == True{} : Bool}, within: S.member(k,x)) -> S.interior(x):
  +start_lower = first(k,PrefixLaws.prefix_in_allocation(k,Nats.weaken(k,128n,valid)))
  +end_upper = last(1n+k,PrefixLaws.prefix_in_allocation(1n+k,valid))
  +actual_upper = Equal.trans(Bool,U32.is_lt(S.finish(k),131072),U32.is_lt(Prefix.prefix(1n+k),131072),True{},
    Equal.cong(U32,Bool,t => U32.is_lt(t,131072),S.finish(k),Prefix.prefix(1n+k),finish(k,valid)),end_upper)
  Interval.member(k,x,within,start_lower,actual_upper)

def nonalias(+i: Nat, +j: Nat, +x: U32, +y: U32,
  +before: {Nat.is_lt(i,j) == True{} : Bool}, +valid: {Nat.is_lt(j,128n) == True{} : Bool},
  ix: S.member(i,x), jy: S.member(j,y)) ->
  {U32.is_ne(S.normalize(x),S.normalize(y)) == True{} : Bool}:
  (+xl,+xu) = ix
  (+yl,+yu) = jy
  +ivalid = Low.le_lt(i,j,128n,Nats.lt_to_le(i,j,before),valid)
  +xnorm = S.normalized(x,certificate(i,x,ivalid,(xl,xu)))
  +ynorm = S.normalized(y,certificate(j,y,valid,(yl,yu)))
  Interval.ordered(S.finish(i),Prefix.prefix(j),x,y,xu,PrefixLaws.blocks_ordered(i,j,before,valid),yl,xnorm,ynorm)
