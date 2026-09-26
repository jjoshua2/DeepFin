import Base
import ./LAWS.bend as Laws
import ./Borrow.bend as Borrow
import ./Step.bend as Step
import ./Ordinal.bend as Ordinal
import ../index/Bits.bend as Bits
import ../index/LAWS.bend as IndexLaws
import ../index/PROOF.bend as IndexProof

def Laws.subtraction_refinement(a,b):
  Borrow.u64_sub(a,b)

def Laws.step_successor(s,mask,member):
  Step.actual_value(s,mask,member)

def Laws.sequence_ordinal(i,mask,bounded):
  Ordinal.ordinal(i,mask,bounded)

def Laws.sequence_nonduplicating(i,j,mask,ib,jb,same):
  Ordinal.nonduplicating(i,j,mask,ib,jb,same)

def Laws.sequence_coverage(s,mask,member):
  (Bits.value(U64.pext(s,mask)), (IndexLaws.extraction_bound(s,mask),Ordinal.covering(s,mask,member)))

def Laws.cycle_endpoint(mask):
  Ordinal.wrap(mask)

def Laws.sequence_periodic(i,mask):
  Ordinal.periodic(i,mask)
