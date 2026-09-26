import Base
import ./LAWS.bend as Laws
import ./Flow.bend as Flow
import ./Membership.bend as M
import ./Guard.bend as Guard

def Laws.castle_side_guarded_extension(b,ks,table,tail):
  Flow.side(b,ks,table,tail)

def Laws.new_castle_member_has_guard(b,ks,table,tail,m,here,absent):
  M.guard(b,ks,table,tail,m,here,absent)

def Laws.new_castle_member_matches_route(b,ks,table,tail,m,here,absent):
  M.exact(b,ks,table,tail,m,here,absent)

def Laws.producer_guard_supplies_rook_freshness(b,ks,guard):
  Guard.fresh(b,ks,guard)

def Laws.new_castle_member_preserves_representation(b,ks,table,tail,m,good,here,absent):
  Guard.emitted(b,ks,table,tail,m,good,here,absent)

def Laws.rejected_guard_preserves_pair(b,ks,table,tail,rejected):
  Flow.rejected(b,ks,table,tail,rejected)
