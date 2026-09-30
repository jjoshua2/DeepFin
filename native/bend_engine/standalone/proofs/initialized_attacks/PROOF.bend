import Base
import ./LAWS.bend as Laws
import ./Masks.bend as Masks
import ./Routing.bend as Routing
import ./Composition.bend as Composition

def Laws.initialized_piece_attack_matches_geometry(d,seed,n,extra,kind,sq,white,occ,depth,tables,full,bound):
  Masks.actual(d,seed,n,extra,kind,sq,white,occ,depth,tables,full,bound)

def Laws.initialized_attacked_matches_geometry(d,seed,n,extra,b,sq,by,depth,tables,full,bound):
  Composition.attacked(d,seed,n,extra,b,sq,by,depth,tables,full,bound)

def Laws.initialized_single_king_check_matches_geometry(d,seed,n,extra,b,sq,side,depth,tables,full,bound,one):
  Composition.checked(d,seed,n,extra,b,sq,side,depth,tables,full,bound,one)
