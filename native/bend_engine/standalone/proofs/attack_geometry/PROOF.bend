import Base
import ./LAWS.bend as Laws
import ./Computed.bend as Computed
import ./Full.bend as Full
import ./Initialized.bend as I

def Laws.computed_leaper_matches_coordinates(kind,sq,bound):
  Computed.actual(kind,sq,bound)

def Laws.extras_store_coordinate_mask(extra,a,kind,sq,shape,bound,full):
  Full.pair(extra,a,kind,sq,shape,bound,full)

def Laws.initialized_leaper_query_matches_coordinates(d,seed,n,key,at,extra,kind,sq,occ,depth,full,bound):
  I.query(d,seed,n,key,at,extra,kind,sq,occ,depth,full,bound)
