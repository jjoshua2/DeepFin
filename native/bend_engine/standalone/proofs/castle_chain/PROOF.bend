import Base
import ./LAWS.bend as Laws
import ./Chain.bend as C

def Laws.ordinary_scan_excludes_castling(keys,b,table):
  C.scan_plain(keys,b,table)

def Laws.final_filter_preserves_member_origin(b,table,xs,m,here):
  C.filter_member(b,table,xs,m,here)

def Laws.filtered_castling_suffix_origin(b,table,xs,m,here):
  C.suffix_member(b,table,xs,m,here)

def Laws.legal_castle_member_has_producer(table,b,m,here,tag):
  C.legal_member(table,b,m,here,tag)

def Laws.legal_castle_member_preserves_representation(table,b,m,good,here,tag):
  C.legal_preserves(table,b,m,good,here,tag)
