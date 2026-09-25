import Base
import ./LAWS.bend as Laws
import ./Actual.bend as A

def Laws.empty_representation(): {==}
def Laws.initial_representation(): {==}
def Laws.fresh_put_preserves_representation(b,sq,piece,white,good,fresh):
  A.insert(b,sq,piece,white,good,fresh)
def Laws.put_preserves_metadata(b,kind,white,sq):
  A.put_metadata(b,kind,white,sq)
def Laws.metadata_preserves_representation(b,turn,rights,ep):
  A.metadata(b,turn,rights,ep)
def Laws.put_exact_bitboard_update(b,sq,piece,white):
  A.bridge(b,sq,piece,white)
