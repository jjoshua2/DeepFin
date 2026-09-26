import Base
import ./LAWS.bend as Laws
import ./Required.bend as R
import ./Filter.bend as F
import ./Connect.bend as Connect

def Laws.guarded_castle_requires_full_check(b,side,rays,ok):
  R.guard(b,side,rays,ok)

def Laws.generated_castle_requires_full_check(table,b,m,rays,here,tag):
  Connect.generated(table,b,m,rays,here,tag)

def Laws.generated_castle_fast_step_equals_checked_step(table,b,m,rays,r,here,tag):
  Connect.step(table,b,m,rays,r,here,tag)

def Laws.guarded_castling_blocker_filter_equals_full_filter(b,xs,table,rays,good):
  F.blockers(b,xs,table,rays,good)
