import Base
import ./Spec.bend as S
import ./consumer.bend as Consumer
import ./Rights.bend as Rights
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def pawn(+src: U32,rights: U32,turn: U32) -> Chess.Board:
  Chess.Board{U64.bit(U32.to_nat(src)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    U64.bit(U32.to_nat(src)),U64.zero(),turn,rights,64}
def empty(rights: U32,turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),turn,rights,64}
def game(b: Chess.Board) -> Pos.Game: Pos.Game{b,0,1,Nil{}}

def high_bits() -> {Chess.get_rights(Chess.make_move(pawn(8,4294967295,1),Chess.Ply{8,9,4,2})) == 4294967295 : U32}: {==}
def a1_source() -> {Chess.get_rights(Chess.make_move(pawn(0,15,1),Chess.Ply{0,1,0,0})) == 13 : U32}: {==}
def h1_destination() -> {Chess.get_rights(Chess.make_move(pawn(8,15,1),Chess.Ply{8,7,2,1})) == 14 : U32}: {==}
def a8_source() -> {Chess.get_rights(Chess.make_move(pawn(56,15,0),Chess.Ply{56,57,3,2})) == 7 : U32}: {==}
def h8_destination() -> {Chess.get_rights(Chess.make_move(pawn(8,15,1),Chess.Ply{8,63,4,0})) == 11 : U32}: {==}
def white_decoded() -> {Chess.get_rights(Chess.make_move(empty(15,1),Chess.Ply{4,6,1,2})) == 12 : U32}: {==}
def black_decoded() -> {Chess.get_rights(Chess.make_move(empty(15,0),Chess.Ply{60,62,0,0})) == 3 : U32}: {==}
def raw_turn() -> {Chess.get_rights(Chess.make_move(empty(15,2),Chess.Ply{4,5,0,1})) == 3 : U32}: {==}
def decoded_fallback() -> {Chess.piece(empty(15,1),4) == 5 : U32}: {==}
def promotion_is_not_kind() -> {Chess.get_rights(Chess.make_move(pawn(8,15,1),Chess.Ply{8,9,5,0})) == 15 : U32}: {==}

# Actual raw EP can remove a home rook without revoking its right.
# This witnesses why irreversibility alone does not prove validity of surviving bits.
def ep_parent() -> Chess.Board:
  Chess.Board{U64.bit(16n),U64.zero(),U64.zero(),U64.bit(0n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(16n),U64.or(U64.bit(0n),U64.bit(4n))),U64.bit(60n),1,2,8}
def surviving_ep_right() -> {Chess.get_rights(Chess.make_move(ep_parent(),Chess.Ply{16,8,0,1})) == 2 : U32}: {==}
def surviving_ep_invalid() -> {Pos.valid_right(Chess.make_move(ep_parent(),Chess.Ply{16,8,0,1}),2,0,4,1) == False{} : Bool}: {==}

def shared_text() -> {Pos.move_text(Chess.Ply{4,6,0,0}) == Pos.move_text(Chess.Ply{4,6,0,2}) : String}: {==}
def first_full_ply() -> {Pos.find_move(Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{4,6,0,0},Nil{}}},"e1g1") == Some{Chess.Ply{4,6,0,2}} : Maybe<&2,Chess.Ply>}: {==}
def reverse_full_ply() -> {Pos.find_move(Con{Chess.Ply{4,6,0,0},Con{Chess.Ply{4,6,0,2},Nil{}}},"e1g1") == Some{Chess.Ply{4,6,0,0}} : Maybe<&2,Chess.Ply>}: {==}
def duplicates(a: Array<U64>,+g: Pos.Game,+text: String) ->
  S.replay(a,Chess.get_rights(Pos.board(g)),Some{g},Con{text,Con{text,Nil{}}}):
  Consumer.use_game(a,g,Con{text,Con{text,Nil{}}})
def zero_duplicates(a: Array<U64>,+g: Pos.Game,+text: String,good: {Chess.get_rights(Pos.board(g)) == 0 : U32}) ->
  S.zero_replay(a,Some{g},Con{text,Con{text,Nil{}}}):
  Consumer.use_zero(a,Some{g},Con{text,Con{text,Nil{}}},good)
def none_tail(a: Array<U64>,+text: String) -> S.zero_replay(a,None{},Con{text,Con{text,Nil{}}}):
  Consumer.use_zero(a,None{},Con{text,Con{text,Nil{}}},Unit{})
def empty_tail(a: Array<U64>,+g: Pos.Game) -> S.replay(a,Chess.get_rights(Pos.board(g)),Some{g},Nil{}):
  Consumer.use_game(a,g,Nil{})
