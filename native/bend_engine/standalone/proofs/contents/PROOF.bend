import Base
import ./LAWS.bend as Laws
import ./Actual.bend as Actual
import ./Certified.bend as Certified

def Laws.bounded_fill_entry(a,n,i,at,end,sq,bishop,mask,subset,shape,ib,eb,budget):
  Actual.entry(a,n,i,at,end,sq,bishop,mask,subset,shape,ib,eb,budget)

def Laws.bounded_zero_fill_read(a,n,i,at,end,sq,bishop,mask,shape,ib,eb,budget):
  Actual.complete_pair(a,n,i,at,end,sq,bishop,mask,shape,ib,eb,budget)

def Laws.full_block_entry(a,k,i,valid,shape,ib):
  Certified.block(a,k,i,valid,shape,ib)
