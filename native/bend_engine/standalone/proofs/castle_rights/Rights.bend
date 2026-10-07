import Base
import ./Words.bend as U
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

# These expressions mirror the actual update, whose exact projection is proved below.
def king_loss(+b: Chess.Board,+src: U32) -> U32:
  Bool.pick(U32,U32.is_eq(Chess.piece(b,src),5),
    Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12),0)
def lost(+b: Chess.Board,m: Chess.Ply) -> U32:
  Chess.Ply{+src,dst,promotion,flag}=m
  U32.or(U32.or(Chess.corner_right(src),Chess.corner_right(dst)),king_loss(b,src))
def exact(+b: Chess.Board,+m: Chess.Ply) ->
  {Chess.get_rights(Chess.make_move(b,m)) == U32.and(Chess.get_rights(b),U32.not(lost(b,m))) : U32}:
  Chess.Ply{src,dst,promotion,flag}=m
  {==}
def subset(+b: Chess.Board,+m: Chess.Ply) ->
  U.subset(Chess.get_rights(b),Chess.get_rights(Chess.make_move(b,m))):
  %Equal.sym(U32,Chess.get_rights(Chess.make_move(b,m)),U32.and(Chess.get_rights(b),U32.not(lost(b,m))),exact(b,m)) :
    U.subset(Chess.get_rights(b),_)
  U.absorb(Chess.get_rights(b),U32.not(lost(b,m)))
def zero(+b: Chess.Board,+m: Chess.Ply,same: {Chess.get_rights(b) == 0 : U32}) ->
  {Chess.get_rights(Chess.make_move(b,m)) == 0 : U32}:
  %Equal.sym(U32,Chess.get_rights(Chess.make_move(b,m)),U32.and(Chess.get_rights(b),U32.not(lost(b,m))),exact(b,m)) : {_ == 0 : U32}
  %Equal.sym(U32,Chess.get_rights(b),0,same) : {U32.and(_,U32.not(lost(b,m))) == 0 : U32}
  U.zero(U32.not(lost(b,m)))
def revoke(+b: Chess.Board,+m: Chess.Ply,+bit: U32,included: U.subset(lost(b,m),bit)) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,m)),bit) == 0 : U32}:
  %Equal.sym(U32,Chess.get_rights(Chess.make_move(b,m)),U32.and(Chess.get_rights(b),U32.not(lost(b,m))),exact(b,m)) :
    {U32.and(_,bit) == 0 : U32}
  U.disjoint_subset(Chess.get_rights(b),lost(b,m),bit,included)
def source(+b: Chess.Board,+src: U32,+dst: U32,+promotion: U32,+flag: U32) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{src,dst,promotion,flag})),Chess.corner_right(src)) == 0 : U32}:
  revoke(b,Chess.Ply{src,dst,promotion,flag},Chess.corner_right(src),
    U.trans(lost(b,Chess.Ply{src,dst,promotion,flag}),U32.or(Chess.corner_right(src),Chess.corner_right(dst)),Chess.corner_right(src),
      U.union_left(U32.or(Chess.corner_right(src),Chess.corner_right(dst)),king_loss(b,src)),
      U.union_left(Chess.corner_right(src),Chess.corner_right(dst))))
def destination(+b: Chess.Board,+src: U32,+dst: U32,+promotion: U32,+flag: U32) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{src,dst,promotion,flag})),Chess.corner_right(dst)) == 0 : U32}:
  revoke(b,Chess.Ply{src,dst,promotion,flag},Chess.corner_right(dst),
    U.trans(lost(b,Chess.Ply{src,dst,promotion,flag}),U32.or(Chess.corner_right(src),Chess.corner_right(dst)),Chess.corner_right(dst),
      U.union_left(U32.or(Chess.corner_right(src),Chess.corner_right(dst)),king_loss(b,src)),
      U.union_right(Chess.corner_right(src),Chess.corner_right(dst))))
def king(+b: Chess.Board,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  decoded: {Chess.piece(b,src) == 5 : U32}) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{src,dst,promotion,flag})),
    Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12)) == 0 : U32}:
  included: U.subset(lost(b,Chess.Ply{src,dst,promotion,flag}),king_loss(b,src))=
    U.union_right(U32.or(Chess.corner_right(src),Chess.corner_right(dst)),king_loss(b,src))
  %decoded :
    {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{src,dst,promotion,flag})),
      Bool.pick(U32,U32.is_eq(_,5),Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12),0)) == 0 : U32}
  revoke(b,Chess.Ply{src,dst,promotion,flag},king_loss(b,src),included)
def valid_right(+b: Chess.Board,+bit: U32,+rook: U32,+king: U32,+side: U32,
  same: {Chess.get_rights(b) == 0 : U32}) -> {Pos.valid_right(b,bit,rook,king,side) == True{} : Bool}:
  %Equal.sym(U32,Chess.get_rights(b),0,same) :
    {Bool.or(U32.is_zero(U32.and(_,bit)),Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side))) == True{} : Bool}
  %Equal.sym(U32,U32.and(0,bit),0,U.zero(bit)) :
    {Bool.or(U32.is_zero(_),Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side))) == True{} : Bool}
  {==}
