import Base
import ./Lattice.bend as L
import ../ray/Path.bend as P
import ../ray/Algebra.bend as A
import ../ray/Spec.bend as S
import ../ray/Fold.bend as Fold
import ../ray/Traversal.bend as T
import ../../Tables.bend as Tables
import ../u64/Words.bend as W

def head(+occupied: Bool,+bit: U64,+tail: U64) -> U64:
  U64.or(bit,Bool.pick(U64,occupied,U64.zero(),tail))

# Insertion keeps every old blocker. A new blocker can only discard the suffix.
def head_subset(+old: Bool,+new: Bool,+bit: U64,+small: U64,+big: U64,
  present: {old == True{} : Bool} -> {new == True{} : Bool},
  tail: L.subset(small,big)) -> L.subset(head(new,bit,small),head(old,bit,big)):
  match old new:
    case False{} False{}: L.or_subset(bit,small,big,tail)
    case False{} True{}:
      L.rewrite(bit,U64.or(bit,big),U64.or(bit,U64.zero()),U64.or(bit,big),
        A.zero(bit),{==},A.absorb_left(bit,big))
    case True{} True{}: L.reflexive(U64.or(bit,U64.zero()))
    case True{} False{}:
      W.false_true(L.subset(head(False{},bit,small),head(True{},bit,big)),present({==}))

def attack_subset(+xs: List<&2,Nat>,+occ: U64,+added: U64) ->
  L.subset(P.attack(xs,U64.or(occ,added)),P.attack(xs,occ)):
  match xs:
    case Nil{}: {==}
    case +x <> +tail:
      head_subset(U64.test_bit(occ,x),U64.test_bit(U64.or(occ,added),x),U64.bit(x),
        P.attack(tail,U64.or(occ,added)),P.attack(tail,occ),
        certificate => L.insert_preserves(occ,added,x,certificate),attack_subset(tail,occ,added))

def scan_subset(+xs: List<&2,Nat>,+occ: U64,+added: U64,+acc: U64) ->
  L.subset(S.scan(xs,U64.or(occ,added),acc),S.scan(xs,occ,acc)):
  L.rewrite(U64.or(acc,P.attack(xs,U64.or(occ,added))),U64.or(acc,P.attack(xs,occ)),
    S.scan(xs,U64.or(occ,added),acc),S.scan(xs,occ,acc),
    Fold.accumulator(xs,U64.or(occ,added),acc),Fold.accumulator(xs,occ,acc),
    L.or_subset(acc,P.attack(xs,U64.or(occ,added)),P.attack(xs,occ),attack_subset(xs,occ,added)))

# Input-only route, actual fuel, stopped=False, mask=False, one shared accumulator.
def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+added: U64,+acc: U64,
  +route: {S.trace(n,sq,dir,xs) == True{} : Bool}) ->
  L.subset(Tables.ray(n,False{},sq,dir,U64.or(occ,added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc)):
  L.rewrite(S.scan(xs,U64.or(occ,added),acc),S.scan(xs,occ,acc),
    Tables.ray(n,False{},sq,dir,U64.or(occ,added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc),
    T.actual(n,sq,dir,xs,U64.or(occ,added),acc,route),T.actual(n,sq,dir,xs,occ,acc,route),
    scan_subset(xs,occ,added,acc))
