import Base
import ./LAWS.bend as Laws
import ./ClearActual.bend as Clear
import ./Ordinary.bend as Ordinary

def Laws.clear_mask_exact_update(b,mask):
  Clear.exact(b,mask)

def Laws.clear_mask_preserves_representation(b,mask,good):
  Clear.preserves(b,mask,good)

def Laws.cleared_target_is_fresh(b,source,target):
  Clear.fresh(b,source,target)

def Laws.ordinary_make_move_exact_update(b,src,dst):
  Ordinary.actual(b,src,dst)

def Laws.ordinary_make_move_preserves_representation(b,src,dst,good):
  Ordinary.preserves(b,src,dst,good)
