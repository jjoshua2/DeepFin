import Base
import ./LAWS.bend as Laws
import ./OneBlock.bend as One
import ./Stored.bend as Stored
import ./Pipeline.bend as Pipeline

def Laws.own_block_header(a,k,which,valid,shape):
  One.header_after_own_fill(a,k,which,valid,shape)

def Laws.stored_header_after_tables(a,before,after,k,which,shape,budget):
  Stored.actual(a,before,after,k,which,shape,budget)

def Laws.stored_header_after_extras(d,seed,before,after,k,which,extra,start,depth,budget,ebudget):
  Pipeline.initialized(d,seed,before,after,k,which,extra,start,depth,budget,ebudget)
