import Base
import ./LAWS.bend as Laws
import ./Actual.bend as Actual
import ./Certified.bend as Certified

def Laws.bounded_clear_before(a,n,at,end,q,shape,eb,qb,budget):
  Actual.before(a,n,at,end,q,shape,eb,qb,budget)

def Laws.bounded_clear_after(a,n,at,q,shape,qb,budget):
  Actual.after(a,n,at,q,shape,qb,budget)

def Laws.bounded_fill_preserves_before(a,n,at,end,q,sq,bishop,mask,subset,shape,eb,qb,budget):
  Actual.frame_before(a,n,at,end,q,sq,bishop,mask,subset,shape,eb,qb,budget)

def Laws.bounded_fill_preserves_after(a,n,at,q,sq,bishop,mask,subset,shape,qb,budget):
  Actual.frame_after(a,n,at,q,sq,bishop,mask,subset,shape,qb,budget)

def Laws.block_fill_preserves_before(a,k,q,valid,shape,qb):
  Certified.before(a,k,q,valid,shape,qb)

def Laws.block_fill_preserves_after(a,k,q,valid,shape,start,qb):
  Certified.after(a,k,q,valid,shape,start,qb)
