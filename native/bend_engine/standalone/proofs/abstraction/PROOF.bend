import Base
import ./Projection.bend as Projection
import ./LAWS.bend as Laws
import ./Roundtrip.bend as R
import ./Properties.bend as P

def Laws.whole_board_roundtrip(b,good): R.board(b,good)
def Laws.snapshot_has_64_squares(b): P.lengths(b)
def Laws.equal_snapshots_identify_board(a,b,ga,gb,same): P.injective(a,b,ga,gb,same)
def Laws.empty_square_has_empty_observation(b,sq,good,bound,absent): P.empty(b,sq,good,bound,absent)


def Laws.snapshot_square_correspondence(b,sq,bound): Projection.actual(b,sq,bound)
