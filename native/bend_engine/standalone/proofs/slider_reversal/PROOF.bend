import Base
import ./LAWS.bend as Laws
import ./Reversal.bend as R

def Laws.slider_geometric_membership_reverses(kind,src,dst,occ,source_bound,target_bound):
  R.geometry(kind,src,dst,occ,source_bound,target_bound)

def Laws.computed_slider_mask_reverses(kind,src,dst,occ,source_bound,target_bound):
  R.computed(kind,src,dst,occ,source_bound,target_bound)
