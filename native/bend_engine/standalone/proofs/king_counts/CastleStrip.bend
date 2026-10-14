# Strip destination and rook-source erasures; no count or child equality is assumed.
import Base
import ./Algebra.bend as A
import ../ray/Algebra.bend as Ray
import ../candidate_king_destination/Algebra.bend as Clear

def strip(+k: U64,+c: U64,+s: U64,+d: U64,+r: U64,
  +dc: {U64.and(d,k) == U64.zero() : U64},
  rc: {U64.and(r,k) == U64.zero() : U64}) ->
  {U64.and_not(U64.and(k,c),U64.or(U64.or(s,d),U64.or(d,r))) ==
   U64.and_not(U64.and(k,c),s) : U64}:
  Equal.trans(U64,
    U64.and_not(U64.and(k,c),U64.or(U64.or(s,d),U64.or(d,r))),
    A.erase(k,c,s,d,U64.or(d,r)),A.single(k,c,s),
    Equal.cong(U64,U64,x => U64.and_not(U64.and(k,c),U64.or(U64.or(s,d),x)),
      U64.or(d,r),U64.or(U64.or(d,r),U64.zero()),
      Equal.sym(U64,U64.or(U64.or(d,r),U64.zero()),U64.or(d,r),Ray.zero(U64.or(d,r)))),
    A.strip(k,c,s,d,U64.or(d,r),dc,Clear.union_clear(d,r,k,dc,rc)))
