import Base
import ./LAWS.bend as Laws
import ./Reduction.bend as R
import ./Flow.bend as F
import ./King.bend as K

def Laws.mask_reduction_has_square_witness(b,by,m):
  R.decision(b,by,m)

def Laws.actual_attacked_uses_witness_reduction(table,b,sq,by):
  F.attacked(table,b,sq,by)

def Laws.single_king_in_check_uses_witness_reduction(table,b,side,sq,bound,one):
  K.checked(table,b,side,sq,bound,one)
