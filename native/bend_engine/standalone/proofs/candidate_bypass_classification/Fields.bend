# Exact full-Ply alternatives, distinct from castle_chain's flag != 2 predicate.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../promotion/Spec.bend as Promotion
import ../castle_chain/Spec.bend as Chain

def plain(-m: Chess.Ply) -> Type:
  Exists(U32,src => Exists(U32,dst => {m == Chess.Ply{src,dst,0,0} : Chess.Ply}))

def ep(-m: Chess.Ply) -> Type:
  Exists(U32,src => Exists(U32,dst => {m == Chess.Ply{src,dst,0,1} : Chess.Ply}))

def promoted(-m: Chess.Ply) -> Type:
  Exists(U32,src => Exists(U32,dst => Exists(Promotion.Choice,choice =>
    {m == Promotion.ply(src,dst,choice) : Chess.Ply})))

def scanned(-m: Chess.Ply) -> Type:
  plain(m) | (ep(m) | promoted(m))

def generated(-b: Chess.Board,-m: Chess.Ply) -> Type:
  scanned(m) | Chain.castle(b,m)

def bypass_shape(-m: Chess.Ply) -> Type:
  plain(m) | promoted(m)
