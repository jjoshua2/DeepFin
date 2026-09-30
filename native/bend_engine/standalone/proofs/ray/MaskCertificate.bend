import Base
import ./Masking.bend as M
import ./Rays.bend as R
import ../layout/Domain.bend as D

def four(+k: Nat) -> Type:
  +s = Nat.mod(k,64n)
  +d = Bool.pick(Nat,Nat.is_ge(k,64n),4n,0n)
  M.keeps(R.path(7n,s,d),R.mask(k)) &
    (M.keeps(R.path(7n,s,1n+d),R.mask(k)) &
      (M.keeps(R.path(7n,s,2n+d),R.mask(k)) & M.keeps(R.path(7n,s,3n+d),R.mask(k))))

# Structural certificate for every key's generated paths; no key/occupancy case table.
def certificate(+k: Nat) -> four(k):
  +s = Nat.mod(k,64n)
  +d = Bool.pick(Nat,Nat.is_ge(k,64n),4n,0n)
  +x0 = R.path(7n,s,d)
  +x1 = R.path(7n,s,1n+d)
  +x2 = R.path(7n,s,2n+d)
  +x3 = R.path(7n,s,3n+d)
  +m0 = R.bits(R.interior(x0))
  +m1 = R.bits(R.interior(x1))
  +m2 = R.bits(R.interior(x2))
  +m3 = R.bits(R.interior(x3))
  (M.add_right(x0,U64.or(m0,m1),U64.or(m2,m3),M.add_right(x0,m0,m1,M.own(x0))),
    (M.add_right(x1,U64.or(m0,m1),U64.or(m2,m3),M.add_left(x1,m1,m0,M.own(x1))),
      (M.add_left(x2,U64.or(m2,m3),U64.or(m0,m1),M.add_right(x2,m2,m3,M.own(x2))),
        M.add_left(x3,U64.or(m2,m3),U64.or(m0,m1),M.add_left(x3,m3,m2,M.own(x3))))))

def finite(+k: Nat,e: {Nat.is_lt(k,128n) == True{} : Bool}) -> four(k):
  certificate(k)
