import Base
import ./LAWS.bend as Laws
import ./Queries.bend as Q
import ./Targets.bend as Targets
import ./Castles.bend as Castles
import ./Filters.bend as Filters
import ./Generation.bend as Generation

def Laws.attack_query_preserves_table(kind,table,sq,side,occ):
  Q.attack(kind,table,sq,side,occ)

def Laws.ordinary_scan_preserves_table(squares,b,r):
  Targets.scan(squares,b,r)

def Laws.castling_side_preserves_table(b,king_side,r):
  Castles.side(b,king_side,r)

def Laws.prepared_filter_preserves_table(b,r):
  Filters.prepare(b,r)

def Laws.legal_moves_preserves_table(table,b):
  Generation.legal(table,b)
