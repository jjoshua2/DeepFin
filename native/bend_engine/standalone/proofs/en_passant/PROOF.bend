import Base
import ./LAWS.bend as Laws
import ./Actual.bend as EP

def Laws.en_passant_make_move_exact_update(b,src,dst):
  EP.actual(b,src,dst)

def Laws.en_passant_make_move_preserves_representation(b,src,dst,good):
  EP.preserves(b,src,dst,good)
