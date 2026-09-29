import Base
import ./LAWS.bend as Laws
import ./Bound.bend as Bound
import ./Actual.bend as Actual

def Laws.bounded_mask_identity(i,bound):
  Bound.table_address(i,bound)

def Laws.bounded_mask_injective(i,j,ib,jb,same):
  Equal.trans(U32,i,U32.and(i,U32.sub(131072,1)),j,
    Equal.sym(U32,U32.and(i,U32.sub(131072,1)),i,Bound.table_address(i,ib)),
    Equal.trans(U32,U32.and(i,U32.sub(131072,1)),U32.and(j,U32.sub(131072,1)),j,same,Bound.table_address(j,jb)))

def Laws.bounded_read_direct(a,i,capacity,bound):
  Actual.read(a,i,capacity,bound)

def Laws.bounded_write_direct(a,i,v,capacity,bound):
  Actual.write(a,i,v,capacity,bound)

def Laws.bounded_separation_direct(a,i,j,capacity,ib,jb):
  Actual.paths(a,i,j,capacity,ib,jb)
