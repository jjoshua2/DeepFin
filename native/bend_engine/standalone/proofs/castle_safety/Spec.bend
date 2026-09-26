# Predicates and the exact sensitive-mask expression used by actual filter_blockers.
import Base
import ../../../legal_probe/Chess.bend as Chess

def sensitive(+b: Chess.Board,rays: U64) -> U64:
  U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b)))
