import Base
import ./LAWS.bend as Laws
import ./Public.bend as Public
import ./Geometry.bend as Geometry

def Laws.prepared_castle_check_returns_false(a,b,xs,m,castle,here):
  Public.prepared(a,b,xs,m,castle,here)

def Laws.generated_castle_check_returns_false(a,b,m,here,tag):
  Public.legal(a,b,m,here,tag)

def Laws.initialized_generated_castle_destination_is_unattacked(d,seed,n,extra,b,white,m,depth,tables,full,good,turn,one,here,tag):
  Geometry.actual(d,seed,n,extra,b,white,m,depth,tables,full,good,turn,one,here,tag)
