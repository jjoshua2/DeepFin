import Base
import ./LAWS.bend as Laws
import ./Route.bend as Route
import ./Canonical.bend as Canonical
import ./MaskedLookup.bend as Result

def Laws.lookup_from_headers(a,key,occ,mask,at,mh,oh):
  Route.actual(a,key,occ,mask,at,mh,oh)

def Laws.selected_state_is_masked(key,occ,valid):
  Canonical.state(key,occ,valid)

def Laws.initialized_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget):
  Result.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)
