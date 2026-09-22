# Numeric bridge; the public module constructs endpoint certificates from prior proofs.
import Base
import ./Spec.bend as S
import ./Mask.bend as Mask
import ./Order.bend as Order
import ../prefix/Order.bend as Earlier
import ../prefix/Spec.bend as Prefix

def certify(+start: U32, +end: U32, +x: U32,
  lower: {U32.is_le(start,x) == True{} : Bool}, upper: {U32.is_lt(x,end) == True{} : Bool},
  reserved: {U32.is_le(512,start) == True{} : Bool}, capacity: {U32.is_lt(end,131072) == True{} : Bool}) -> S.interior(x):
  +bounded = Order.le_lt(x,end,131072,Earlier.lt_le(x,end,upper),capacity)
  (Earlier.trans(512,start,x,reserved,lower),(bounded,Mask.allocation(x,bounded)))

def injective(+x: U32, +y: U32,
  bx: {U32.is_lt(x,131072) == True{} : Bool}, by: {U32.is_lt(y,131072) == True{} : Bool},
  same: {S.normalize(x) == S.normalize(y) : U32}) -> {x == y : U32}:
  Equal.trans(U32,x,S.normalize(x),y,Equal.sym(U32,S.normalize(x),x,Mask.allocation(x,bx)),
    Equal.trans(U32,S.normalize(x),S.normalize(y),y,same,Mask.allocation(y,by)))

def ordered(+end_i: U32, +start_j: U32, +x: U32, +y: U32,
  xu: {U32.is_lt(x,end_i) == True{} : Bool}, order: {U32.is_le(end_i,start_j) == True{} : Bool},
  yl: {U32.is_le(start_j,y) == True{} : Bool},
  nx: {S.normalize(x) == x : U32}, ny: {S.normalize(y) == y : U32}) ->
  {U32.is_ne(S.normalize(x),S.normalize(y)) == True{} : Bool}:
  %Equal.sym(U32,S.normalize(x),x,nx) : {U32.is_ne(_,S.normalize(y)) == True{} : Bool}
  %Equal.sym(U32,S.normalize(y),y,ny) : {U32.is_ne(x,_) == True{} : Bool}
  Order.not_equal(U32.cmp(x,y),Order.lt_le(x,start_j,y,Order.lt_le(x,end_i,start_j,xu,order),yl))

# The interval premise is an ordinary pair of endpoint inequalities.
def member(+k: Nat, +x: U32, within: S.member(k,x),
  reserved: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool},
  capacity: {U32.is_lt(S.finish(k),131072) == True{} : Bool}) -> S.interior(x):
  (lo,hi) = within
  certify(Prefix.prefix(k),S.finish(k),x,lo,hi,reserved,capacity)
