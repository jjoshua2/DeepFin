import Base
import ./Actual.bend as A
import ../color_counts/Emission.bend as Emit
import ../color_counts/Actual.bend as Sources
import ./Spec.bend as S
import ./Root.bend as Root
import ./consumer.bend as Consumer
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

# Raw turn 2 selects black for both the generator and actual make_move.
# The other color occupies both the EP destination and the XOR-8 victim.
def parent() -> Chess.Board:
  Chess.Board{U64.or(U64.bit(48n),U64.or(U64.bit(16n),U64.bit(24n))),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    U64.or(U64.bit(16n),U64.bit(24n)),U64.bit(48n),2,0,16}
def targets() -> U64: U64.from_parts(2147483648,65537)
def tail() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,0,1,0},Nil{}}}}
def actual() -> List<&2,Chess.Ply>: Chess.destinations(64n,False{},targets(),48,True{},16,tail())
def expected() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},
    Con{Chess.Ply{48,16,0,1},Con{Chess.Ply{48,0,1,0},Con{Chess.Ply{48,0,2,0},Con{Chess.Ply{48,0,3,0},Con{Chess.Ply{48,0,4,0},tail()}}}}}}}}}
def full_fields() -> {actual() == expected() : List<&2,Chess.Ply>}: {==}
def twelve_occurrences() -> {List.length(&2,Chess.Ply,actual()) == 12n : Nat}: {==}
def tail_good() -> C.every(m => Sources.emitted(parent(),m),tail()):
  (({==},({==},({==},{==}))),(({==},({==},({==},{==}))),(({==},({==},({==},{==}))),Unit{})))
def multi_tail() -> C.every(m => Sources.emitted(parent(),m),actual()):
  Emit.destinations(parent(),64n,False{},targets(),48,True{},16,tail(),{==},{==},{==},tail_good())
def children(+b: Chess.Board,xs: List<&2,Chess.Ply>,cert: C.every(m => Sources.emitted(b,m),xs)) ->
  C.every(m => A.decreases(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,+rest}:
      (head,good)=cert
      (A.post(b,m,head),children(b,rest,good))
def actual_children() -> C.every(m => A.decreases(parent(),m),actual()): children(parent(),actual(),multi_tail())
def ragged() -> Array<U64>: ANode{ALeaf{U64.from_u32(17)},ANode{ALeaf{U64.from_u32(23)},ALeaf{U64.from_u32(29)}}}
def pair_fields() -> {Chess.scan_after(48,True{},16,tail(),(ragged(),targets())) == (ragged(),expected()) : Array<U64> & List<&2,Chess.Ply>}: {==}
def pair_table() -> {T.table(List<&2,Chess.Ply>,Chess.scan_after(48,True{},16,tail(),(ragged(),targets()))) == ragged() : Array<U64>}: {==}
def pair_proof() -> C.every(m => Sources.emitted(parent(),m),E.moves(Chess.scan_after(48,True{},16,tail(),(ragged(),targets())))):
  Emit.after(r => C.every(m => Sources.emitted(parent(),m),E.moves(r)),parent(),48,True{},16,tail(),(ragged(),targets()),{==},{==},tail_good(),r => good => good)
def exhausted() -> {Chess.destinations(0n,False{},targets(),48,True{},16,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def empty() -> {Chess.destinations(64n,True{},U64.zero(),48,True{},16,tail()) == tail() : List<&2,Chess.Ply>}: {==}
def lower_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(0n),8,True{},0,Nil{}) ==
  Con{Chess.Ply{8,0,1,0},Con{Chess.Ply{8,0,2,0},Con{Chess.Ply{8,0,3,0},Con{Chess.Ply{8,0,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def upper_ep_promotion() -> {Chess.destinations(64n,False{},U64.bit(63n),48,True{},63,Nil{}) ==
  Con{Chess.Ply{48,63,1,0},Con{Chess.Ply{48,63,2,0},Con{Chess.Ply{48,63,3,0},Con{Chess.Ply{48,63,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def nonpawn_ep() -> {Chess.destinations(64n,False{},U64.bit(0n),8,False{},0,Nil{}) == Con{Chess.Ply{8,0,0,0},Nil{}} : List<&2,Chess.Ply>}: {==}
def ordinary_ep() -> {Chess.destinations(64n,False{},U64.bit(16n),8,True{},16,Nil{}) == Con{Chess.Ply{8,16,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def ep_removes_both() -> {A.white(Chess.make_move(parent(),Chess.Ply{48,16,0,1})) == 0n : Nat}: {==}
def raw_turn_moves_black() -> {A.black(Chess.make_move(parent(),Chess.Ply{48,16,0,1})) == 1n : Nat}: {==}
def coalesced_source_destination() -> {A.black(Chess.make_move(parent(),Chess.Ply{48,48,0,0})) == 1n : Nat}: {==}
def promotion_removes_pawn() -> {A.black(Chess.make_move(parent(),Chess.Ply{48,63,4,0})) == 0n : Nat}: {==}
def unowned() -> Chess.Board:
  Chess.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),1,0,64}
def raw_unowned_increases() -> {Nat.is_le(A.white(Chess.make_move(unowned(),Chess.Ply{8,16,0,0})),A.white(unowned())) == False{} : Bool}: {==}
def missing_rook() -> Chess.Board:
  Chess.Board{U64.bit(5n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(4n),U64.bit(4n),U64.zero(),1,0,64}
def raw_castle_increases() -> {Nat.is_le(A.white(Chess.make_move(missing_rook(),Chess.Ply{4,6,0,2})),A.white(missing_rook())) == False{} : Bool}: {==}
def first_full_ply() -> {Pos.find_move(Con{Chess.Ply{8,16,0,1},Con{Chess.Ply{8,16,0,0},Nil{}}},"a2a3") == Some{Chess.Ply{8,16,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def duplicates(a: Array<U64>,+g: Pos.Game,+text: String,good: S.board_good(Pos.board(g))) -> S.replay(a,Some{g},Con{text,Con{text,Nil{}}}):
  Consumer.use_replay(a,Some{g},Con{text,Con{text,Nil{}}},good)
def none_tail(a: Array<U64>,+text: String) -> S.replay(a,None{},Con{text,Con{text,Nil{}}}):
  Consumer.use_replay(a,None{},Con{text,Con{text,Nil{}}},Unit{})
def empty_tail(a: Array<U64>,+g: Pos.Game,good: S.board_good(Pos.board(g))) -> S.replay(a,Some{g},Nil{}):
  Consumer.use_replay(a,Some{g},Nil{},good)
def validated_parent(a: Array<U64>,+g: Pos.Game,+xs: List<&2,String>,zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},good: {Pos.valid(Pos.board(g)) == True{} : Bool}) -> S.replay(a,Some{g},xs):
  Consumer.use_replay(a,Some{g},xs,(zero,Root.valid(Pos.board(g),good)))

# A nonpawn move clears an unowned pawn bit at its destination before repainting.
def ghost_destination() -> Chess.Board:
  Chess.Board{U64.from_u32(130816),U64.bit(0n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.from_u32(65281),U64.zero(),1,0,64}
def ghost_cleared() -> {A.white(Chess.make_move(ghost_destination(),Chess.Ply{0,16,0,0})) == 8n : Nat}: {==}
def ghost_exact_plane() -> {Chess.get_pawns(Chess.make_move(ghost_destination(),Chess.Ply{0,16,0,0})) == U64.from_u32(65280) : U64}: {==}
def overlapping() -> Chess.Board:
  Chess.Board{U64.from_u32(65280),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.from_u32(65280),U64.from_u32(65280),1,0,64}
def overlapping_active() -> {A.white(Chess.make_move(overlapping(),Chess.Ply{8,16,0,0})) == 8n : Nat}: {==}
def overlapping_inactive() -> {A.black(Chess.make_move(overlapping(),Chess.Ply{8,16,0,0})) == 7n : Nat}: {==}
