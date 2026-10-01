import Base
import ./LAWS.bend as Laws
import ../castle_stage_geometry/PROOF.bend as StageProof
import ./Composition.bend as C

def Laws.initialized_castle_producer_matches_geometry(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard,tail):
  C.producer(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard,tail)

def Laws.initialized_filtered_castle_matches_three_stage_geometry(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard):
  C.filtered(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard)
