import Base
import ./LAWS.bend as Laws
import ./Route.bend as Route
import ./Canonical.bend as Canonical
import ./Pipeline.bend as Pipeline
import ./MaskedLookup.bend as Masked

def Laws.certified_header_route(a,key,occ,mask,at,mh,oh):
  Route.actual(a,key,occ,mask,at,mh,oh)

def Laws.selected_state_is_masked(key,occ,valid):
  Canonical.state(key,occ,valid)

def Laws.initialized_indexed_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget):
  Pipeline.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)

def Laws.initialized_masked_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget):
  Masked.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)
