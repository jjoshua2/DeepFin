import Base
import ./LAWS.bend as Laws
import ./Projection.bend as Projection
import ./Correspondence.bend as Correspondence
import ../layout/LAWS.bend as LayoutLaws
import ../layout/PROOF.bend as LayoutProof

# Helpers receive a count certificate only to avoid repeatedly checking the
# expensive finite geometry proof while developing structural lemmas. Every
# accepted chess law below obtains that certificate from the actual prior proof.
def Laws.low_projection_exact(x,k,retained,bound):
  Projection.low_exact(x,k,retained,bound)

def Laws.compact_index_exact(occ,mask,small):
  Correspondence.compact_exact(occ,mask,small)

def Laws.chess_lookup_exact(key,occ,valid):
  Correspondence.chess_exact(key,occ,valid,LayoutLaws.mask_population(key,valid))

def Laws.chess_lookup_word(key,occ,valid):
  Correspondence.chess_word(key,occ,valid,LayoutLaws.mask_population(key,valid))

def Laws.chess_sequence_ordinal(key,i,valid,bounded):
  Correspondence.ordinal(key,i,valid,bounded,LayoutLaws.mask_population(key,valid))

def Laws.chess_sequence_coverage(key,s,valid,member):
  Correspondence.covering(key,s,valid,member,LayoutLaws.mask_population(key,valid))

def Laws.chess_lookup_injective(key,a,b,valid,am,bm,same):
  Correspondence.injective(key,a,b,valid,am,bm,same,LayoutLaws.mask_population(key,valid))

def Laws.chess_lookup_redeposit(key,occ,valid):
  Correspondence.redeposit(key,occ,valid,LayoutLaws.mask_population(key,valid))
