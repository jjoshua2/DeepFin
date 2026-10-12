import Base
import ../ray/Algebra.bend as A
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Path.bend as P
import ../ray/Rays.bend as R
import ../ray/Spec.bend as S
import ../ray/Slider.bend as Slider
import ../layout/Domain.bend as Domain
import ../../Tables.bend as Tables

def lift_left(+small: U64,+a: U64,+b: U64,e: L.subset(small,a)) ->
  L.subset(small,U64.or(a,b)):
  +same: {U64.and(a,small) == small : U64} = e
  Equal.trans(U64,U64.and(U64.or(a,b),small),
    U64.and(U64.or(a,b),U64.and(a,small)),small,
    Equal.cong(U64,U64,w => U64.and(U64.or(a,b),w),small,U64.and(a,small),
      Equal.sym(U64,U64.and(a,small),small,same)),
    Equal.trans(U64,U64.and(U64.or(a,b),U64.and(a,small)),U64.and(a,small),small,
      A.extend_right(a,b,small),same))

def lift_right(+small: U64,+a: U64,+b: U64,e: L.subset(small,b)) ->
  L.subset(small,U64.or(a,b)):
  +same: {U64.and(b,small) == small : U64} = e
  Equal.trans(U64,U64.and(U64.or(a,b),small),
    U64.and(U64.or(a,b),U64.and(b,small)),small,
    Equal.cong(U64,U64,w => U64.and(U64.or(a,b),w),small,U64.and(b,small),
      Equal.sym(U64,U64.and(b,small),small,same)),
    Equal.trans(U64,U64.and(U64.or(a,b),U64.and(b,small)),U64.and(b,small),small,
      A.extend_left(a,b,small),same))

def first(+a: U64,+b: U64,+c: U64,+d: U64) ->
  L.subset(a,U64.or(U64.or(a,b),U64.or(c,d))):
  lift_left(a,U64.or(a,b),U64.or(c,d),A.absorb_left(a,b))
def second(+a: U64,+b: U64,+c: U64,+d: U64) ->
  L.subset(b,U64.or(U64.or(a,b),U64.or(c,d))):
  lift_left(b,U64.or(a,b),U64.or(c,d),D.right_member(a,b))
def third(+a: U64,+b: U64,+c: U64,+d: U64) ->
  L.subset(c,U64.or(U64.or(a,b),U64.or(c,d))):
  lift_right(c,U64.or(a,b),U64.or(c,d),A.absorb_left(c,d))
def fourth(+a: U64,+b: U64,+c: U64,+d: U64) ->
  L.subset(d,U64.or(U64.or(a,b),U64.or(c,d))):
  lift_right(d,U64.or(a,b),U64.or(c,d),D.right_member(c,d))

def path(+q: Nat,dir: Nat,occ: U64) -> U64:
  P.attack(R.path(7n,q,dir),occ)
def geometry(+q: Nat,+occ: U64) -> U64:
  U64.or(S.directional(q,0n,occ),S.directional(q,4n,occ))
def actual(+q: Nat,+occ: U64) -> U64:
  U64.or(Tables.slider(U32.from_nat(q),False{},occ,False{}),
    Tables.slider(U32.from_nat(q),True{},occ,False{}))

# All eight input directions; occupancy is arbitrary and includes the first blocker.
def geometric(+q: Nat,+dir: Nat,+occ: U64,db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  L.subset(path(q,dir,occ),geometry(q,occ)):
  match dir:
    case 0n:
      lift_left(path(q,0n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        first(path(q,0n,occ),path(q,1n,occ),path(q,2n,occ),path(q,3n,occ)))
    case 1n:
      lift_left(path(q,1n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        second(path(q,0n,occ),path(q,1n,occ),path(q,2n,occ),path(q,3n,occ)))
    case 2n:
      lift_left(path(q,2n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        third(path(q,0n,occ),path(q,1n,occ),path(q,2n,occ),path(q,3n,occ)))
    case 3n:
      lift_left(path(q,3n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        fourth(path(q,0n,occ),path(q,1n,occ),path(q,2n,occ),path(q,3n,occ)))
    case 4n:
      lift_right(path(q,4n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        first(path(q,4n,occ),path(q,5n,occ),path(q,6n,occ),path(q,7n,occ)))
    case 5n:
      lift_right(path(q,5n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        second(path(q,4n,occ),path(q,5n,occ),path(q,6n,occ),path(q,7n,occ)))
    case 6n:
      lift_right(path(q,6n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        third(path(q,4n,occ),path(q,5n,occ),path(q,6n,occ),path(q,7n,occ)))
    case 7n:
      lift_right(path(q,7n,occ),S.directional(q,0n,occ),S.directional(q,4n,occ),
        fourth(path(q,4n,occ),path(q,5n,occ),path(q,6n,occ),path(q,7n,occ)))
    case 8n+p: Domain.impossible(L.subset(path(q,8n+p,occ),geometry(q,occ)),p,db)

def actual_geometry(+q: Nat,+occ: U64,+qb: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {actual(q,occ) == geometry(q,occ) : U64}:
  Slider.combine(Tables.slider(U32.from_nat(q),False{},occ,False{}),S.directional(q,0n,occ),
    Tables.slider(U32.from_nat(q),True{},occ,False{}),S.directional(q,4n,occ),
    Slider.square(q,False{},occ,qb),Slider.square(q,True{},occ,qb))

def actual_coverage(+q: Nat,+dir: Nat,+occ: U64,
  qb: {Nat.is_lt(q,64n) == True{} : Bool},db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  L.subset(path(q,dir,occ),actual(q,occ)):
  L.rewrite(path(q,dir,occ),geometry(q,occ),path(q,dir,occ),actual(q,occ),
    {==},actual_geometry(q,occ,qb),geometric(q,dir,occ,db))
