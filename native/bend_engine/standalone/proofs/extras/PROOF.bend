import Base
import ./LAWS.bend as Laws
import ./Actual.bend as Actual
import ./Certified.bend as Certified

def Laws.bounded_extras_read(a,n,k,q,shape,budget,qb,outside):
  Actual.pair(a,n,k,q,shape,budget,qb,outside)

def Laws.table_pipeline_extras_read(d,n,key,at,seed,extra,k,q,depth,budget,qb,outside):
  Certified.pipeline(d,n,key,at,seed,extra,k,q,depth,budget,qb,outside)
