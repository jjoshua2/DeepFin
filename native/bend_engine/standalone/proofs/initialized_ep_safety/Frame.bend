# Derive the child's singleton at the original king square.
import Base
import ../ep_king_frame/Frame.bend as F
import ../../../legal_probe/Chess.bend as Chess

def preservation(+b: Chess.Board,-m: Chess.Ply) -> Type:
  {Chess.get_kings(Chess.make_move(b,m)) == Chess.get_kings(b) : U64} &
  {F.plane(Chess.make_move(b,m),Chess.get_turn(b)) == F.plane(b,Chess.get_turn(b)) : U64}

def singleton(+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  +turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  cert: preservation(b,m)) ->
  {F.plane(Chess.make_move(b,m),Bool.to_u32(side)) == U64.bit(q) : U64}:
  (kings,kept) = cert
  Equal.trans(U64,F.plane(Chess.make_move(b,m),Bool.to_u32(side)),
    F.plane(Chess.make_move(b,m),Chess.get_turn(b)),U64.bit(q),
    Equal.sym(U64,F.plane(Chess.make_move(b,m),Chess.get_turn(b)),
      F.plane(Chess.make_move(b,m),Bool.to_u32(side)),
      Equal.cong(U32,U64,t => F.plane(Chess.make_move(b,m),t),
        Chess.get_turn(b),Bool.to_u32(side),turn)),
    Equal.trans(U64,F.plane(Chess.make_move(b,m),Chess.get_turn(b)),
      F.plane(b,Chess.get_turn(b)),U64.bit(q),kept,
      Equal.trans(U64,F.plane(b,Chess.get_turn(b)),F.plane(b,Bool.to_u32(side)),U64.bit(q),
        Equal.cong(U32,U64,t => F.plane(b,t),Chess.get_turn(b),Bool.to_u32(side),turn),one)))
