import Base
import ./LAWS.bend as Laws
import ./Termination.bend as End
import ./Refinement.bend as Ref
import ./Slider.bend as Slider
import ./Geometry.bend as Geometry
import ./Lookup.bend as Lookup

def Laws.seven_steps_reach_edge(sq,dir,sb,db):
  End.finite(sq,dir,sb,db)

def Laws.ray_matches_independent_path(sq,dir,occ,acc,sb,db):
  Ref.ray(sq,dir,occ,acc,sb,db)

def Laws.slider_matches_independent_rays(key,occ,valid):
  Slider.actual(key,occ,valid)

def Laws.relevant_mask_preserves_attacks(key,occ,valid):
  Geometry.masked_slider(key,occ,valid)

def Laws.initialized_lookup_matches_independent_rays(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget):
  Lookup.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)
