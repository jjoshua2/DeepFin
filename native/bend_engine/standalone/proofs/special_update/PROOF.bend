import Base
import ./LAWS.bend as Laws
import ./Promotion.bend as Promotion
import ./EnPassant.bend as EnPassant

def Laws.typed_promotion_exact_update(b,src,dst,p):
  Promotion.actual(b,src,dst,p)

def Laws.typed_promotion_preserves_representation(b,src,dst,p,good):
  Promotion.preserves(b,src,dst,p,good)

def Laws.en_passant_exact_update(b,src,dst):
  EnPassant.actual(b,src,dst)

def Laws.en_passant_preserves_representation(b,src,dst,good):
  EnPassant.preserves(b,src,dst,good)
