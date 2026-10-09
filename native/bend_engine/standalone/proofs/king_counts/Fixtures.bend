import Base
import ./Actual.bend as K
import ./Emission.bend as Emit
import ./consumer.bend as Consumer
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def parent() -> Chess.Board:
  Chess.Board{U64.bit(48n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(4n),U64.bit(48n)),U64.bit(60n),1,0,64}
def targets() -> U64: U64.from_parts(2147483648,65537)
def tail() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,16,0,0},Con{Chess.Ply{48,16,0,0},Con{Chess.Ply{48,0,1,0},Nil{}}}}
def actual() -> List<&2,Chess.Ply>: Chess.destinations(64n,False{},targets(),48,True{},64,tail())
def expected() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},
    Con{Chess.Ply{48,16,0,0},Con{Chess.Ply{48,0,1,0},Con{Chess.Ply{48,0,2,0},Con{Chess.Ply{48,0,3,0},Con{Chess.Ply{48,0,4,0},tail()}}}}}}}}}
def full_fields() -> {actual() == expected() : List<&2,Chess.Ply>}: {==}
def twelve_occurrences() -> {List.length(&2,Chess.Ply,actual()) == 12n : Nat}: {==}
def tail_good() -> C.every(m => K.emitted(parent(),m),tail()):
  ((({==},({==},({==},{==}))),({==},({==},{==}))),
   ((({==},({==},({==},{==}))),({==},({==},{==}))),
    ((({==},({==},({==},{==}))),({==},({==},{==}))),Unit{})))
def multi_tail() -> C.every(m => K.emitted(parent(),m),actual()):
  Emit.destinations(parent(),64n,False{},targets(),48,True{},tail(),{==},{==},{==},{==},{==},{==},{==},tail_good())
def children(+b: Chess.Board,xs: List<&2,Chess.Ply>,+rows: {Board.valid(b) == True{} : Bool},cert: C.every(m => K.emitted(b,m),xs)) ->
  C.every(m => K.preserved(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,t}:
      (head,tail)=cert
      (K.post(b,m,rows,head),children(b,t,rows,tail))
def actual_children() -> C.every(m => K.preserved(parent(),m),actual()): children(parent(),actual(),{==},multi_tail())
def ragged() -> Array<U64>: ANode{ALeaf{U64.from_u32(17)},ANode{ALeaf{U64.from_u32(23)},ALeaf{U64.from_u32(29)}}}
def pair_fields() -> {Chess.scan_after(48,True{},64,tail(),(ragged(),targets())) == (ragged(),expected()) : Array<U64> & List<&2,Chess.Ply>}: {==}
def pair_table() -> {T.table(List<&2,Chess.Ply>,Chess.scan_after(48,True{},64,tail(),(ragged(),targets()))) == ragged() : Array<U64>}: {==}
def pair_proof() -> C.every(m => K.emitted(parent(),m),E.moves(Chess.scan_after(48,True{},64,tail(),(ragged(),targets())))):
  Emit.after(r => C.every(m => K.emitted(parent(),m),E.moves(r)),parent(),48,True{},tail(),(ragged(),targets()),{==},{==},{==},{==},{==},{==},tail_good(),r => good => good)
def exhausted() -> {Chess.destinations(0n,False{},targets(),48,True{},64,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def empty() -> {Chess.destinations(64n,True{},U64.zero(),48,True{},64,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def lower_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(0n),8,True{},0,Nil{}) ==
  Con{Chess.Ply{8,0,1,0},Con{Chess.Ply{8,0,2,0},Con{Chess.Ply{8,0,3,0},Con{Chess.Ply{8,0,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def upper_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(63n),48,True{},63,Nil{}) ==
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def nonpawn_ep() -> {Chess.destinations(64n,False{},U64.bit(0n),8,False{},0,Nil{}) == Con{Chess.Ply{8,0,0,0},Nil{}} : List<&2,Chess.Ply>}: {==}
def ordinary_ep() -> {Chess.destinations(64n,False{},U64.bit(16n),8,True{},16,Nil{}) == Con{Chess.Ply{8,16,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def raw_turn() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),U64.bit(4n),U64.bit(60n),2,0,64}
def raw_turn_white() -> {K.white(Chess.make_move(raw_turn(),Chess.Ply{60,59,0,0})) == 1n : Nat}: {==}
def raw_turn_black() -> {K.black(Chess.make_move(raw_turn(),Chess.Ply{60,59,0,0})) == 1n : Nat}: {==}
def malformed_ep() -> Chess.Board:
  Chess.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(24n),U64.bit(60n)),U64.or(U64.bit(8n),U64.bit(24n)),U64.bit(60n),1,0,16}
def malformed_ep_rows() -> {Board.valid(malformed_ep()) == True{} : Bool}: {==}
def malformed_ep_rejected() -> {Pos.valid_ep(16,malformed_ep()) == False{} : Bool}: {==}
def malformed_ep_loses_king() -> {K.white(Chess.make_move(malformed_ep(),Chess.Ply{8,16,0,1})) == 0n : Nat}: {==}
def first_full_ply() -> {Pos.find_move(Con{Chess.Ply{8,16,0,1},Con{Chess.Ply{8,16,0,0},Nil{}}},"a2a3") == Some{Chess.Ply{8,16,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def duplicates(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+g: Pos.Game,+text: String,
  rows: {Board.valid(Pos.board(g)) == True{} : Bool},canonical: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(U32.is_eq(Chess.get_turn(Pos.board(g)),1)) : U32},
  coherent: {Pos.valid_ep(Chess.get_ep(Pos.board(g)),Pos.board(g)) == True{} : Bool},zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.replay(a,d,seed,n,extra,Some{g},Con{text,Con{text,Nil{}}}):
  Consumer.use_replay(a,d,seed,n,extra,Some{g},Con{text,Con{text,Nil{}}},(rows,canonical),coherent,zero,built,depth,tables,full)
