# White kingside home occupants survive every ordinary generated full Ply
# whose child right is still set. No assumptions about child rows are used.
import Base
import ./Rights.bend as R
import ./Home.bend as H
import ./Geometry.bend as G
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
def misses(-s: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(s)),4n) == False{} : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(s)),7n) == False{} : Bool}
def singleton(+s: U32,tests: G.home_tests(s),four: {U32.is_eq(s,4) == False{} : Bool},seven: {U32.is_eq(s,7) == False{} : Bool}) -> misses(s):
  (a,b)=tests
  (Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),4n),U32.is_eq(s,4),False{},a,four),
   Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),7n),U32.is_eq(s,7),False{},b,seven))
def four(+s: U32,tests: G.home_tests(s),no: {U32.is_eq(s,4) == False{} : Bool}) -> {U64.test_bit(U64.bit(U32.to_nat(s)),4n) == False{} : Bool}:
  (a,b)=tests
  Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),4n),U32.is_eq(s,4),False{},a,no)
def seven(+s: U32,tests: G.home_tests(s),no: {U32.is_eq(s,7) == False{} : Bool}) -> {U64.test_bit(U64.bit(U32.to_nat(s)),7n) == False{} : Bool}:
  (a,b)=tests
  Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(s)),7n),U32.is_eq(s,7),False{},b,no)
def capture_when(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,+yes: Bool,
  chosen: {U32.is_eq(f,1) == yes : Bool},coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: EP.gate(yes,b,Chess.Ply{s,d,p,f}),target: misses(d)) -> misses(K.cap(Chess.Ply{s,d,p,f})):
  match yes:
    case False{}:
      %Equal.sym(Bool,U32.is_eq(f,1),False{},chosen) : misses(Bool.pick(U32,_,U32.xor(d,8),d))
      target
    case True{}:
      (promotion,(pawn,(same,below)))=tagged
      %Equal.sym(Bool,U32.is_eq(f,1),True{},chosen) : misses(Bool.pick(U32,_,U32.xor(d,8),d))
      G.generated_ep_victim(b,d,below,coherent,same)
def capture(+b: Chess.Board,+m: Chess.Ply,coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: EP.property(b,m),target: misses(Fields.destination(m))) -> misses(K.cap(m)):
  Chess.Ply{+s,+d,+p,+f}=m
  capture_when(b,s,d,p,f,U32.is_eq(f,1),{==},coherent,tagged,target)
def keep(+b: Chess.Board,+m: Chess.Ply,+sq: U32,+piece: U32,
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  target: {U64.test_bit(M.target(m),U32.to_nat(sq)) == False{} : Bool},
  covered: {U64.test_bit(M.remove(m),U32.to_nat(sq)) == False{} : Bool} &
    ({U64.test_bit(F.rook_to(m),U32.to_nat(sq)) == False{} : Bool} & {U64.test_bit(F.added(m),U32.to_nat(sq)) == False{} : Bool}),
  old: {Pos.occupied_by(b,sq,piece,1) == True{} : Bool}) -> {Pos.occupied_by(Chess.make_move(b,m),sq,piece,1) == True{} : Bool}:
  (removed,(rook,added))=covered
  Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,m),sq,piece,1),Pos.occupied_by(b,sq,piece,1),True{},
    Rows.occupant(b,m,sq,piece,1,Rows.row(b,m,sq,bound,removed,target,rook,added)),old)
def finish(+b: Chess.Board,+m: Chess.Ply,+no: {U32.is_eq(Fields.flag(m),2) == False{} : Bool},
  source: misses(Fields.source(m)),target: misses(Fields.destination(m)),capture: misses(K.cap(m)),
  homes: {Pos.occupied_by(b,7,3,1) == True{} : Bool} & {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool}:
  Chess.Ply{+s,+d,+p,+f}=m
  (s4,s7)=source
  (+d4,+d7)=target
  (c4,c7)=capture
  (r,k)=homes
  +rook=keep(b,m,7,3,{==},d7,Mask.covered(m,7n,no,s7,d7,c7),r)
  +king=keep(b,m,4,5,{==},d4,Mask.covered(m,4n,no,s4,d4,c4),k)
  Equal.trans(Bool,Pos.valid_right(Chess.make_move(b,m),1,7,4,1),
    Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1)),True{}),True{},
    Equal.cong(Bool,Bool,x => Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1)),x),
      Bool.and(Pos.occupied_by(Chess.make_move(b,m),7,3,1),Pos.occupied_by(Chess.make_move(b,m),4,5,1)),True{},
      Equal.trans(Bool,Bool.and(Pos.occupied_by(Chess.make_move(b,m),7,3,1),Pos.occupied_by(Chess.make_move(b,m),4,5,1)),
        Pos.occupied_by(Chess.make_move(b,m),4,5,1),True{},
        Equal.cong(Bool,Bool,x => Bool.and(x,Pos.occupied_by(Chess.make_move(b,m),4,5,1)),Pos.occupied_by(Chess.make_move(b,m),7,3,1),True{},rook),king)),or_true(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1))))
def active_homes(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,+rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,Chess.Ply{s,d,p,f}),
  +active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1)) == False{} : Bool},
  cert: K.emitted(b,Chess.Ply{s,d,p,f}),
  homes: {Pos.occupied_by(b,7,3,1) == True{} : Bool} & {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,Chess.Ply{s,d,p,f}),1,7,4,1) == True{} : Bool}:
  ((+sb,(owned,(+db,no))),(selector,(clear,capclear)))=cert
  (rook,+king)=homes
  +d4=four(d,G.singleton_home_tests(d,db),H.destination_king(b,d,U32.is_eq(d,4),{==},clear,H.king_test(b,rows,king)))
  +d7=seven(d,G.singleton_home_tests(d,db),H.destination_corner(b,s,d,p,f,U32.is_eq(d,7),{==},active))
  finish(b,Chess.Ply{s,d,p,f},no,
    singleton(s,G.singleton_home_tests(s,sb),H.source_king(b,s,d,p,f,U32.is_eq(s,4),{==},rows,owned,king,active),
      H.source_corner(b,s,d,p,f,U32.is_eq(s,7),{==},active)),
    (d4,d7),capture(b,Chess.Ply{s,d,p,f},coherent,tagged,(d4,d7)),(rook,king))
def active(+b: Chess.Board,m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  +survives: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1)) == False{} : Bool},
  parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool}:
  Chess.Ply{+s,+d,+p,+f}=m
  active_homes(b,s,d,p,f,rows,coherent,tagged,survives,cert,R.parents(b,Chess.Ply{s,d,p,f},1,7,4,1,survives,parent))
def when(+b: Chess.Board,+m: Chess.Ply,empty: Bool,
  same: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1)) == empty : Bool},
  rows: {Board.valid(b) == True{} : Bool},coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool}:
  match empty:
    case True{}: R.empty_valid(Chess.make_move(b,m),1,7,4,1,Index.u32_eq_true_same(U32.and(Chess.get_rights(Chess.make_move(b,m)),1),0,same))
    case False{}: active(b,m,rows,coherent,tagged,same,parent,cert)
def post(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},tagged: EP.property(b,m),
  parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},cert: K.emitted(b,m)) ->
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool}:
  when(b,m,U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),1)),{==},rows,coherent,tagged,parent,cert)
