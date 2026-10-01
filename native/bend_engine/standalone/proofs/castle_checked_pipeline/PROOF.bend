import Base
import ./LAWS.bend as Laws
import ./Composition.bend as C

def Laws.initialized_castle_producer_matches_checks(d,seed,n,extra,b,white,ks,tail,depth,tables,full,good,turn,one,guard):
  C.producer(d,seed,n,extra,b,white,ks,tail,depth,tables,full,good,turn,one,guard)

def Laws.initialized_castle_pipeline_matches_three_checks(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard):
  C.pipeline(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard)

def Laws.accepted_castle_pipeline_has_safe_stages(d,seed,n,extra,b,white,ks,m,depth,tables,full,good,turn,one,guard,here):
  C.accepted(d,seed,n,extra,b,white,ks,m,depth,tables,full,good,turn,one,guard,here)
