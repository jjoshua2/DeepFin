import Base
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ./Certified.bend as Certified
import ./Frame.bend as Frame
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Nats.bend as Nats
import ../layout/Low.bend as Low
import ../layout/Spec.bend as Layout

def first_valid(+i: Nat,+j: Nat,before: {Nat.is_lt(i,j) == True{} : Bool},valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {Nat.is_lt(i,128n) == True{} : Bool}:
  Low.le_lt(i,j,128n,Nats.lt_to_le(i,j,before),valid)

def Laws.relative_address_bounds(k,r,valid,rb):
  Bounds.interval(k,r,Certified.certificate(k,r,valid,rb))

def Laws.relative_addition_exact(k,r,valid,rb):
  Bounds.widening(k,r,Certified.certificate(k,r,valid,rb))

def Laws.lookup_address_bounds(k,occ,valid):
  Laws.relative_address_bounds(k,S.index(k,occ),valid,Certified.lookup_index(k,occ,valid))

def Laws.ordered_block_addresses(i,j,ri,rj,before,valid,rib,rjb):
  +vi = first_valid(i,j,before,valid)
  Frame.ordered(i,j,ri,rj,Laws.relative_address_bounds(i,ri,vi,rib),Laws.relative_address_bounds(j,rj,valid,rjb),
    Bounds.endpoint(Prefix.prefix(i),Layout.size(U32.from_nat(i)),Prefix.prefix(1n+i),PrefixLaws.step_no_overflow(i,vi)),
    PrefixLaws.blocks_ordered(i,j,before,valid))

def Laws.ordered_block_address_paths(a,i,j,ri,rj,shape,before,valid,rib,rjb):
  Frame.routes(a,i,j,ri,rj,shape,
    Laws.relative_address_bounds(i,ri,first_valid(i,j,before,valid),rib),Laws.relative_address_bounds(j,rj,valid,rjb),
    Laws.ordered_block_addresses(i,j,ri,rj,before,valid,rib,rjb))

def Laws.lookup_other_block_write(a,i,j,occi,occj,v,shape,before,valid):
  +vi = first_valid(i,j,before,valid)
  +ri = S.index(i,occi)
  +rj = S.index(j,occj)
  Frame.write(a,i,j,ri,rj,v,shape,Laws.lookup_address_bounds(i,occi,vi),Laws.lookup_address_bounds(j,occj,valid),
    Laws.ordered_block_addresses(i,j,ri,rj,before,valid,Certified.lookup_index(i,occi,vi),Certified.lookup_index(j,occj,valid)))
