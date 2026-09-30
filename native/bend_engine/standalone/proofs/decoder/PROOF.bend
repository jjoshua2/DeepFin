import Base
import ./LAWS.bend as Laws
import ./Actual.bend as A
import ./Projection.bend as P

def Laws.square_projection_valid(b,sq,good,bound): P.valid(b,sq,good,bound)
def Laws.abstract_square_roundtrip(b,sq,good,bound): A.roundtrip(b,sq,good,bound)
def Laws.guarded_decoder_matches_square(b,sq,good,bound): A.decode(b,sq,good,bound)
def Laws.occupied_decoder_matches_square(b,sq,good,bound,present): A.occupied(b,sq,good,bound,present)
