import Base
import ./Algebra.bend as D
import ../ray/Algebra.bend as A
import ../ray/Path.bend as P
import ../ray/Masking.bend as M
import ../ray/Spec.bend as S
import ../blocker_frame/Blocker.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../occupancy_insertion/Insertion.bend as I
import ../../Tables.bend as Tables

def clear(occ: U64,removed: U64) -> U64:
  U64.and(occ,U64.not(removed))

def head_kept(+removed: U64,+bit: U64,+tail: U64,+stop: Bool,
  empty: D.disjoint(removed,U64.or(bit,Bool.pick(U64,stop,U64.zero(),tail)))) ->
  {U64.and(U64.not(removed),bit) == bit : U64}:
  D.keep_complement(removed,bit,D.disjoint_subset(removed,bit,U64.or(bit,Bool.pick(U64,stop,U64.zero(),tail)),
    A.absorb_left(bit,Bool.pick(U64,stop,U64.zero(),tail)),empty))

# Only an old clear observation allows the tail-disjoint certificate to recurse.
def tail_disjoint(+removed: U64,+bit: U64,+tail: U64,+stop: Bool,
  empty: D.disjoint(removed,U64.or(bit,Bool.pick(U64,stop,U64.zero(),tail))),
  continued: {stop == False{} : Bool}) -> D.disjoint(removed,tail):
  D.disjoint_subset(removed,tail,U64.or(bit,tail),D.right_member(bit,tail),
    Equal.trans(U64,U64.and(removed,U64.or(bit,tail)),
      U64.and(removed,U64.or(bit,Bool.pick(U64,stop,U64.zero(),tail))),U64.zero(),
      Equal.cong(Bool,U64,b => U64.and(removed,U64.or(bit,Bool.pick(U64,b,U64.zero(),tail))),
        False{},stop,Equal.sym(Bool,stop,False{},continued)),empty))

def step_from_same(+old: Bool,+new: Bool,+later: Bool,
  same: {new == old : Bool},
  next: {old == False{} : Bool} -> {later == True{} : Bool}) ->
  {B.agree_step(old,new,u => later) == True{} : Bool}:
  match old:
    case False{}:
      Equal.trans(Bool,B.agree_step(False{},new,u => later),later,True{},
        Equal.cong(Bool,Bool,b => B.agree_step(False{},b,u => later),new,False{},same),next({==}))
    case True{}:
      Equal.cong(Bool,Bool,b => B.agree_step(True{},b,u => later),new,True{},same)

# Derive observations from the OLD imported attack, never from attack equality.
def agreement(+xs: List<&2,Nat>,+occ: U64,+removed: U64,
  +empty: {U64.and(removed,P.attack(xs,occ)) == U64.zero() : U64}) -> {B.agree(xs,occ,clear(occ,removed)) == True{} : Bool}:
  match xs:
    case Nil{}: {==}
    case +x <> tail:
      match tail:
        case Nil{}: {==}
        case +y <> +rest:
          step_from_same(U64.test_bit(occ,x),U64.test_bit(clear(occ,removed),x),
            B.agree(y <> rest,occ,clear(occ,removed)),
            M.bit_preserved(occ,U64.not(removed),x,
              head_kept(removed,U64.bit(x),P.attack(y <> rest,occ),U64.test_bit(occ,x),empty)),
            certificate => agreement(y <> rest,occ,removed,
              tail_disjoint(removed,U64.bit(x),P.attack(y <> rest,occ),U64.test_bit(occ,x),empty,certificate)))

def attack(+xs: List<&2,Nat>,+occ: U64,+removed: U64,
  empty: D.disjoint(removed,P.attack(xs,occ))) -> {P.attack(xs,clear(occ,removed)) == P.attack(xs,occ) : U64}:
  Equal.sym(U64,P.attack(xs,occ),P.attack(xs,clear(occ,removed)),
    B.attack_equal(xs,occ,clear(occ,removed),agreement(xs,occ,removed,empty)))

def scan(+xs: List<&2,Nat>,+occ: U64,+removed: U64,+acc: U64,
  empty: D.disjoint(removed,P.attack(xs,occ))) -> {S.scan(xs,clear(occ,removed),acc) == S.scan(xs,occ,acc) : U64}:
  Equal.sym(U64,S.scan(xs,occ,acc),S.scan(xs,clear(occ,removed),acc),
    B.scan_equal(xs,occ,clear(occ,removed),acc,agreement(xs,occ,removed,empty)))

def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+removed: U64,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(removed,P.attack(xs,occ))) ->
  {Tables.ray(n,False{},sq,dir,clear(occ,removed),False{},acc) == Tables.ray(n,False{},sq,dir,occ,False{},acc) : U64}:
  Equal.sym(U64,Tables.ray(n,False{},sq,dir,occ,False{},acc),Tables.ray(n,False{},sq,dir,clear(occ,removed),False{},acc),
    B.actual_ray(n,sq,dir,xs,occ,clear(occ,removed),acc,route,agreement(xs,occ,removed,empty)))

# Bounded source-bit specialization records its range; the general result is range-free.
def actual_source(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+src: U32,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},source_bound: {U32.is_lt(src,64) == True{} : Bool},
  empty: D.disjoint(U64.bit(U32.to_nat(src)),P.attack(xs,occ))) ->
  {Tables.ray(n,False{},sq,dir,clear(occ,U64.bit(U32.to_nat(src))),False{},acc) == Tables.ray(n,False{},sq,dir,occ,False{},acc) : U64}:
  actual_ray(n,sq,dir,xs,occ,U64.bit(U32.to_nat(src)),acc,route,empty)

def clear_then_insert(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+removed: U64,+added: U64,+acc: U64,
  +route: {S.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(removed,P.attack(xs,occ))) ->
  L.subset(Tables.ray(n,False{},sq,dir,U64.or(clear(occ,removed),added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc)):
  L.rewrite(Tables.ray(n,False{},sq,dir,U64.or(clear(occ,removed),added),False{},acc),
    Tables.ray(n,False{},sq,dir,clear(occ,removed),False{},acc),
    Tables.ray(n,False{},sq,dir,U64.or(clear(occ,removed),added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc),{==},
    Equal.sym(U64,Tables.ray(n,False{},sq,dir,clear(occ,removed),False{},acc),
      Tables.ray(n,False{},sq,dir,occ,False{},acc),actual_ray(n,sq,dir,xs,occ,removed,acc,route,empty)),
    I.actual_ray(n,sq,dir,xs,clear(occ,removed),added,acc,route))
