import Base
import ./LAWS.bend as Laws
import ./Mask.bend as Mask
import ./Step.bend as Step
import ./Generated.bend as Generated
import ./Pipeline.bend as Pipeline
import ./Forced.bend as Force
import ../../../legal_probe/Chess.bend as Chess

def Laws.owned_king_source_requires_check(b,rays,src,dst,promotion,flag,bound,present):
  Mask.owned(b,rays,src,dst,promotion,flag,bound,present)

def Laws.blocker_stage_uses_sensitive_mask(b,moves,table,rays):
  Step.blockers(b,moves,table,rays)

def Laws.generated_castle_requires_check(table,b,m,rays,here,tag):
  Generated.generated(table,b,m,rays,here,tag)

def Laws.generated_castle_takes_checked_step(generation_table,b,m,rays,here,tag,filter_table,acc):
  Generated.generated_step(generation_table,b,m,rays,here,tag,filter_table,acc)

def Laws.positive_check_rejects_current_move(m,acc,table):
  Step.rejected(m,acc,table)

def Laws.complete_generator_equals_forced_castle_checks(table,b):
  Equal.sym(Array<U64> & List<&2,Chess.Ply>,Force.legal_moves(table,b),Chess.legal_moves(table,b),Pipeline.legal(table,b))
