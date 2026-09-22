import Base
import ./LAWS.bend as Laws
import ./Mask.bend as Mask
import ./Interval.bend as Interval
import ./Certified.bend as Certified
import ./Read.bend as Read
import ./Spec.bend as S

def Laws.allocation_mask_exact(x,bounded):
  Mask.allocation(x,bounded)

def Laws.allocation_mask_injective(x,y,bx,by,same):
  Interval.injective(x,y,bx,by,same)

def Laws.prefix_interior_certificate(k,x,valid,within):
  Certified.certificate(k,x,valid,within)

def Laws.ordered_normalized_addresses_distinct(i,j,x,y,before,valid,ix,jy):
  Certified.nonalias(i,j,x,y,before,valid,ix,jy)

def Laws.interior_read_uses_unmasked_address(a,k,x,valid,within,size):
  Read.actual(a,x,size,S.bound(x,Certified.certificate(k,x,valid,within)))
