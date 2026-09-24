import Base
import ./LAWS.bend as Laws
import ./Boundary.bend as Boundary
import ./Spec.bend as B
import ../ray/Spec.bend as S
import ../layout/Domain.bend as Domain
import ../../../legal_probe/Chess.bend as Chess

def Laws.full_table_lookup(d,seed,n,key,extra,start,occ,valid,depth,full,eb):
  Boundary.actual(d,seed,n,key,extra,start,occ,valid,depth,full,eb)

def Laws.u32_full_table_lookup(d,seed,n,key,extra,start,occ,valid,depth,full,eb):
  Boundary.u32(d,seed,n,key,extra,start,occ,valid,depth,full,eb)
