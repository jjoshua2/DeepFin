import Base
import ./LAWS.bend as Laws
import ./Actual.bend as Actual

def Laws.allocation_complete(d,seed):
  Actual.allocation(d,seed)

def Laws.table_pipeline_complete(d,n,extra,key,at,sq,seed):
  Actual.pipeline(d,n,extra,key,at,sq,seed)

def Laws.bounded_complete_routes(a,i,j,shape,ib,jb,ne):
  Actual.paths(a,i,j,shape,ib,jb,ne)

def Laws.bounded_other_index_write(a,i,j,v,shape,ib,jb,ne):
  Actual.frame(a,i,j,v,shape,ib,jb,ne)
