import Base
import ./LAWS.bend as Laws
import ./Actual.bend as C
import ./Preserve.bend as Preserve

def Laws.castling_make_move_exact_update(b,src,dst):
  C.actual(b,src,dst)

def Laws.castling_make_move_preserves_representation(b,route,good,fresh):
  Preserve.actual(b,route,good,fresh)
