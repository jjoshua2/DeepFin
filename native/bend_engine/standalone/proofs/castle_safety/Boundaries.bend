# Closed raw-filter boundaries; not additional registered universal laws.
import Base
import ./Spec.bend as S
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def empty_source() ->
  {Chess.filter_requires(S.sensitive(Position.empty(),U64.zero()),Chess.Ply{4,6,0,2}) == False{} : Bool}:
  {==}

def king_source() ->
  {Chess.filter_requires(S.sensitive(Position.put(5,True{},4,Position.empty()),U64.zero()),Chess.Ply{4,6,0,2}) == True{} : Bool}:
  {==}
