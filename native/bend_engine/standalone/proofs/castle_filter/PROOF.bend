import Base
import ./LAWS.bend as Laws
import ./Scan.bend as Scan
import ./Filter.bend as Filter
import ./Both.bend as Both
import ./Legal.bend as Legal

def Laws.scan_excludes_castling(xs,b,r,tail):
  Scan.no_castle(xs,b,r,tail)

def Laws.final_filter_preserves_predicate(b,r,P,before):
  Filter.prepare(b,r,P,before)

def Laws.both_castling_sides_provenance(b,table,tail):
  Both.origin(b,table,tail)

def Laws.filtered_castling_provenance(b,table,tail):
  Both.filtered_origin(b,table,tail)

def Laws.legal_castle_guard_and_route(table,b,m,here,is_castle):
  Legal.castle(table,b,m,here,is_castle)

def Laws.legal_castle_preserves_representation(table,b,m,good,here,is_castle):
  Legal.legal_preserves(table,b,m,good,here,is_castle)
