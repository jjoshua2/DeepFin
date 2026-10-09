# Parameterized home occupants survive every ordinary generated full Ply
# whose child right is still set. No assumptions about child rows are used.
import Base
import ./Right.bend as A
import ./Rights.bend as R
import ./GeneralHome.bend as H
import ./GeneralGeometry.bend as G
import ./Mask.bend as Mask
import ./Rows.bend as Rows
import ./Frame.bend as F
import ../king_counts/Actual.bend as K
import ../color_counts/Actual.bend as Fields
import ../backrank_pawns/Mask.bend as M
import ../board/Spec.bend as Board
import ../ep_occurrence_tag/Spec.bend as EP
import ../bitboard_membership/IndexEquality.bend as Index
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def or_true(a: Bool) -> {Bool.or(a,True{}) == True{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}
def misses(+right: A.Right,-s: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.king(right))) == False{} : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.rook(right))) == False{} : Bool}
def singleton(+right: A.Right,+s: U32,tests: G.home_tests(right,s),four: {U32.is_eq(s,A.king(right)) == False{} : Bool},seven: {U32.is_eq(s,A.rook(right)) == False{} : Bool}) -> misses(right,s):
  (a,b)=tests
  (Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.king(right))),U32.is_eq(s,A.king(right)),False{},a,four),
   Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.rook(right))),U32.is_eq(s,A.rook(right)),False{},b,seven))
def king_miss(+right: A.Right,+s: U32,tests: G.home_tests(right,s),no: {U32.is_eq(s,A.king(right)) == False{} : Bool}) -> {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.king(right))) == False{} : Bool}:
  (a,b)=tests
  Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.king(right))),U32.is_eq(s,A.king(right)),False{},a,no)
def rook_miss(+right: A.Right,+s: U32,tests: G.home_tests(right,s),no: {U32.is_eq(s,A.rook(right)) == False{} : Bool}) -> {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.rook(right))) == False{} : Bool}:
  (a,b)=tests
  Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(A.rook(right))),U32.is_eq(s,A.rook(right)),False{},b,no)
def capture_when(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,+yes: Bool,
  chosen: {U32.is_eq(f,1) == yes : Bool},coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: EP.gate(yes,b,Chess.Ply{s,d,p,f}),target: misses(right,d)) -> misses(right,K.cap(Chess.Ply{s,d,p,f})):
  match yes:
    case False{}:
      %Equal.sym(Bool,U32.is_eq(f,1),False{},chosen) : misses(right,Bool.pick(U32,_,U32.xor(d,8),d))
      target
    case True{}:
      (promotion,(pawn,(same,below)))=tagged
      %Equal.sym(Bool,U32.is_eq(f,1),True{},chosen) : misses(right,Bool.pick(U32,_,U32.xor(d,8),d))
      G.generated_ep_victim(right,b,d,below,coherent,same)
def capture(+right: A.Right,+b: Chess.Board,+m: Chess.Ply,coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: EP.property(b,m),target: misses(right,Fields.destination(m))) -> misses(right,K.cap(m)):
  Chess.Ply{+s,+d,+p,+f}=m
  capture_when(right,b,s,d,p,f,U32.is_eq(f,1),{==},coherent,tagged,target)
def keep(+b: Chess.Board,+m: Chess.Ply,+sq: U32,+piece: U32,+side: U32,
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  target: {U64.test_bit(M.target(m),U32.to_nat(sq)) == False{} : Bool},
  covered: {U64.test_bit(M.remove(m),U32.to_nat(sq)) == False{} : Bool} &
    ({U64.test_bit(F.rook_to(m),U32.to_nat(sq)) == False{} : Bool} & {U64.test_bit(F.added(m),U32.to_nat(sq)) == False{} : Bool}),
  old: {Pos.occupied_by(b,sq,piece,side) == True{} : Bool}) -> {Pos.occupied_by(Chess.make_move(b,m),sq,piece,side) == True{} : Bool}:
  (removed,(rook,added))=covered
  Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,m),sq,piece,side),Pos.occupied_by(b,sq,piece,side),True{},
    Rows.occupant(b,m,sq,piece,side,Rows.row(b,m,sq,bound,removed,target,rook,added)),old)
def finish(+right: A.Right,+b: Chess.Board,+m: Chess.Ply,+no: {U32.is_eq(Fields.flag(m),2) == False{} : Bool},
  source: misses(right,Fields.source(m)),target: misses(right,Fields.destination(m)),capture: misses(right,K.cap(m)),
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  Chess.Ply{+s,+d,+p,+f}=m
  (s4,s7)=source
  (+d4,+d7)=target
  (c4,c7)=capture
  (r,k)=homes
  +rook=keep(b,m,A.rook(right),3,A.side(right),A.rook_bound(right),d7,Mask.covered(m,U32.to_nat(A.rook(right)),no,s7,d7,c7),r)
  +king=keep(b,m,A.king(right),5,A.side(right),A.king_bound(right),d4,Mask.covered(m,U32.to_nat(A.king(right)),no,s4,d4,c4),k)
  Equal.trans(Bool,Pos.valid_right(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right)),
    Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right))),True{}),True{},
    Equal.cong(Bool,Bool,x => Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right))),x),
      Bool.and(Pos.occupied_by(Chess.make_move(b,m),A.rook(right),3,A.side(right)),Pos.occupied_by(Chess.make_move(b,m),A.king(right),5,A.side(right))),True{},
      Equal.trans(Bool,Bool.and(Pos.occupied_by(Chess.make_move(b,m),A.rook(right),3,A.side(right)),Pos.occupied_by(Chess.make_move(b,m),A.king(right),5,A.side(right))),
        Pos.occupied_by(Chess.make_move(b,m),A.king(right),5,A.side(right)),True{},
        Equal.cong(Bool,Bool,x => Bool.and(x,Pos.occupied_by(Chess.make_move(b,m),A.king(right),5,A.side(right))),Pos.occupied_by(Chess.make_move(b,m),A.rook(right),3,A.side(right)),True{},rook),king)),or_true(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right)))))
def active_homes(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,+rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,Chess.Ply{s,d,p,f}),
  +active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right))) == False{} : Bool},
  cert: K.emitted(b,Chess.Ply{s,d,p,f}),
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,Chess.Ply{s,d,p,f}),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  ((+sb,(owned,(+db,no))),(selector,(clear,capclear)))=cert
  (rook,+king)=homes
  +d4=king_miss(right,d,G.singleton_home_tests(right,d,db),H.destination_king(right,b,d,U32.is_eq(d,A.king(right)),{==},clear,H.king_test(right,b,rows,king)))
  +d7=rook_miss(right,d,G.singleton_home_tests(right,d,db),H.destination_corner(right,b,s,d,p,f,U32.is_eq(d,A.rook(right)),{==},active))
  finish(right,b,Chess.Ply{s,d,p,f},no,
    singleton(right,s,G.singleton_home_tests(right,s,sb),H.source_king(right,b,s,d,p,f,U32.is_eq(s,A.king(right)),{==},rows,owned,king,active),
      H.source_corner(right,b,s,d,p,f,U32.is_eq(s,A.rook(right)),{==},active)),
    (d4,d7),capture(right,b,Chess.Ply{s,d,p,f},coherent,tagged,(d4,d7)),(rook,king))
def active(+right: A.Right,+b: Chess.Board,m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  +survives: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right))) == False{} : Bool},
  parent: {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  Chess.Ply{+s,+d,+p,+f}=m
  active_homes(right,b,s,d,p,f,rows,coherent,tagged,survives,cert,R.parents(b,Chess.Ply{s,d,p,f},A.bit(right),A.rook(right),A.king(right),A.side(right),survives,parent))
def when(+right: A.Right,+b: Chess.Board,+m: Chess.Ply,empty: Bool,
  same: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right))) == empty : Bool},
  rows: {Board.valid(b) == True{} : Bool},coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  parent: {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  match empty:
    case True{}: R.empty_valid(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right),Index.u32_eq_true_same(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right)),0,same))
    case False{}: active(right,b,m,rows,coherent,tagged,same,parent,cert)
def post(+right: A.Right,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  parent: {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  when(right,b,m,U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right))),{==},rows,coherent,tagged,parent,cert)
