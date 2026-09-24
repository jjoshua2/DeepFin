import Base
import ./LAWS.bend as Laws
import ./Frame.bend as Frame
import ./Stored.bend as Stored
import ./Pipeline.bend as Pipeline

def Laws.later_tables_preserve_data(a,n,k,q,shape,budget,lower,qb,before):
  Frame.actual(a,n,k,q,shape,budget,lower,qb,before)

def Laws.stored_data_after_tables(a,before,after,k,i,shape,budget,ib):
  Stored.actual(a,before,after,k,i,shape,budget,ib)

def Laws.stored_data_after_extras(d,seed,before,after,k,i,extra,start,depth,budget,ib,ebudget):
  Pipeline.initialized(d,seed,before,after,k,i,extra,start,depth,budget,ib,ebudget)
