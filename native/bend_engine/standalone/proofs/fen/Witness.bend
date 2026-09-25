# Closed actual-output controls, distinct from the three universal public laws.
import Base
import ../../Position.bend as P

def nonempty_exact() ->
  {P.fen("P7/8/8/8/8/8/8/7n","b","Kq","h8","0017","00042") ==
    Some{P.Game{P.metadata(P.put(1,False{},7,P.put(0,True{},56,P.empty())),0,9,63),17,42,Nil{}}} : Maybe<&2,P.Game>}:
  {==}

def invalid_placement() ->
  {P.fen("x7/8/8/8/8/8/8/8","w","-","-","0","1") == None{} : Maybe<&2,P.Game>}:
  {==}
