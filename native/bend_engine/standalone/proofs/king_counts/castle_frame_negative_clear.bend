# MUST reject at reject_missing_clear: the importing premise is false.
import Base
import ./CastleFusion.bend as X
def reject_missing_clear() ->
  {Bool.and(True{},True{}) == Bool.and(True{},False{}) : Bool}:
  X.extra_bool(True{},False{},False{},False{},True{},False{},True{},{==})
