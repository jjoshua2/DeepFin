import Base
import ../layout/LAWS.bend as Layout
import ../layout/PROOF.bend as LayoutProof
import ../successor/PROOF.bend as SuccessorProof
import ./LAWS.bend as Laws
import ./Projection.bend as Projection
import ./Correspondence.bend as Correspondence

def Laws.low_projection_exact(x,k,width,bounded):
  Projection.low_exact(x,k,width,bounded)

def Laws.lookup_index_exact(key,occ,valid):
  Correspondence.exact(key,occ,valid,Layout.mask_population(key,valid))

def Laws.lookup_sequence_ordinal(key,i,valid,bounded):
  Correspondence.ordinal(key,i,valid,bounded,Layout.mask_population(key,valid))

def Laws.lookup_collision_sound(key,a,b,valid,same):
  Correspondence.collision_sound(key,a,b,valid,same,Layout.mask_population(key,valid))

def Laws.lookup_collision_complete(a,b,mask,same):
  Correspondence.collision_complete(a,b,mask,same)

def Laws.lookup_state_recovery(key,occ,valid):
  Correspondence.recovery(key,occ,valid,Layout.mask_population(key,valid))
