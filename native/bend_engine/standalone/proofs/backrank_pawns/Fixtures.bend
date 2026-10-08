import Base
import ./Mask.bend as M
import ./Emission.bend as Emit
import ./Spec.bend as S
import ./Root.bend as Root
import ./consumer.bend as Consumer
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def pawn(+src: U32,turn: U32,ep: U32) -> Chess.Board:
  Chess.Board{U64.bit(U32.to_nat(src)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    U64.bit(U32.to_nat(src)),U64.zero(),turn,4294967295,ep}
def parent() -> Chess.Board: pawn(48,2,16)
def targets() -> U64: U64.from_parts(2147483648,65537)
def tail() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,0,1,0},Nil{}}}}
def actual() -> List<&2,Chess.Ply>: Chess.destinations(64n,False{},targets(),48,True{},16,tail())
def expected() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},
    Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,0,1,0},Con{Chess.Ply{48,0,2,0},Con{Chess.Ply{48,0,3,0},Con{Chess.Ply{48,0,4,0},tail()}}}}}}}}}
def full_fields() -> {actual() == expected() : List<&2,Chess.Ply>}: {==}
def twelve_occurrences() -> {List.length(&2,Chess.Ply,actual()) == 12n : Nat}: {==}
def tail_good() -> C.every(m => M.emitted(parent(),m),tail()): ({==},({==},({==},Unit{})))
def multi_tail() -> C.every(m => M.emitted(parent(),m),actual()):
  Emit.destinations(parent(),64n,False{},targets(),48,True{},16,tail(),{==},{==},tail_good())
def actual_children() -> C.every(m => M.safe(Chess.make_move(parent(),m)),actual()):
  Emit.children(parent(),actual(),{==},multi_tail())
def ragged() -> Array<U64>: ANode{ALeaf{U64.from_u32(17)},ANode{ALeaf{U64.from_u32(23)},ALeaf{U64.from_u32(29)}}}
def pair_fields() -> {Chess.scan_after(48,True{},16,tail(),(ragged(),targets())) == (ragged(),expected()) : Array<U64> & List<&2,Chess.Ply>}: {==}
def pair_table() -> {T.table(List<&2,Chess.Ply>,Chess.scan_after(48,True{},16,tail(),(ragged(),targets()))) == ragged() : Array<U64>}: {==}
def pair_proof() -> C.every(m => M.emitted(parent(),m),E.moves(Chess.scan_after(48,True{},16,tail(),(ragged(),targets())))):
  Emit.after(r => C.every(m => M.emitted(parent(),m),E.moves(r)),parent(),48,True{},16,tail(),(ragged(),targets()),{==},tail_good(),r => good => good)
def exhausted() -> {Chess.destinations(0n,False{},targets(),48,True{},16,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def empty() -> {Chess.destinations(64n,True{},U64.zero(),48,True{},16,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def lower_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(0n),8,True{},0,Nil{}) ==
  Con{Chess.Ply{8,0,1,0},Con{Chess.Ply{8,0,2,0},Con{Chess.Ply{8,0,3,0},Con{Chess.Ply{8,0,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def upper_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(63n),48,True{},63,Nil{}) ==
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def nonpawn_ep() -> {Chess.destinations(64n,False{},U64.bit(0n),8,False{},0,Nil{}) == Con{Chess.Ply{8,0,0,0},Nil{}} : List<&2,Chess.Ply>}: {==}
def ordinary_ep() -> {Chess.destinations(64n,False{},U64.bit(16n),8,True{},16,Nil{}) == Con{Chess.Ply{8,16,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def raw_ungenerated() -> {U64.is_zero(U64.and(Chess.get_pawns(Chess.make_move(pawn(8,1,64),Chess.Ply{8,0,0,0})),M.mask())) == False{} : Bool}: {==}
def raw_bad_parent() -> {U64.is_zero(U64.and(Chess.get_pawns(Chess.make_move(pawn(0,1,64),Chess.Ply{8,16,0,0})),M.mask())) == False{} : Bool}: {==}
def castle_raw_turn(+b: Chess.Board,good: M.safe(b)) -> M.emitted(b,E.move(b,True{})):
  M.castle(b,True{},good)
def first_full_ply() -> {Pos.find_move(Con{Chess.Ply{8,16,0,1},Con{Chess.Ply{8,16,0,0},Nil{}}},"a2a3") == Some{Chess.Ply{8,16,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def duplicates(a: Array<U64>,+g: Pos.Game,+text: String,good: M.safe(Pos.board(g))) -> S.replay(a,Some{g},Con{text,Con{text,Nil{}}}):
  Consumer.use_replay(a,Some{g},Con{text,Con{text,Nil{}}},good)
def none_tail(a: Array<U64>,+text: String) -> S.replay(a,None{},Con{text,Con{text,Nil{}}}):
  Consumer.use_replay(a,None{},Con{text,Con{text,Nil{}}},Unit{})
def empty_tail(a: Array<U64>,+g: Pos.Game,good: M.safe(Pos.board(g))) -> S.replay(a,Some{g},Nil{}):
  Consumer.use_replay(a,Some{g},Nil{},good)
def validated_parent(a: Array<U64>,+g: Pos.Game,+xs: List<&2,String>,good: {Pos.valid(Pos.board(g)) == True{} : Bool}) -> S.replay(a,Some{g},xs):
  Consumer.use_replay(a,Some{g},xs,Root.valid(Pos.board(g),good))
