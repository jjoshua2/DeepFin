import Base
import ./LAWS.bend as Laws
import ./Stages.bend as Stages
import ./Check.bend as Check

def Laws.transit_preserves_single_moving_king(b,white,ks,good,turn,one):
  Stages.mid(b,white,ks,good,turn,one)

def Laws.castling_preserves_single_moving_king(b,white,ks,good,turn,one,guard):
  Stages.final(b,white,ks,good,turn,one,guard)

def Laws.move_flips_valid_side(b,move,white,turn):
  Stages.flip(b,move,white,turn)

def Laws.castling_stage_check_targets_moving_king(table,b,white,ks,stage,good,turn,one,guard):
  Check.stage(table,b,white,ks,stage,good,turn,one,guard)
