# Match the actual filter_blockers mask; Step.blockers checks this connection.
import Base
import ../../../legal_probe/Chess.bend as Chess

def sensitive(+b: Chess.Board,rays: U64) -> U64:
  U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b)))

def source(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f} = m
  s

def owned_king(+b: Chess.Board,sq: U32) -> Bool:
  U64.test_bit(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),U32.to_nat(sq))
