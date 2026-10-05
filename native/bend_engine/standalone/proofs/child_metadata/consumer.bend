import Base
import ./Range.bend as R
import ./Source.bend as Source
import ./Post.bend as Post
import ./Children.bend as Children
import ../board/Spec.bend as B
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

# Parent EP is unrestricted; only the actual output field is bounded.
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  good: {B.valid(b) == True{} : Bool},canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> Post.child(b,m,white):
  Post.generated(b,m,white,good,canonical,Source.member(table,b,m,here))

def children_result(+b: Chess.Board,+white: Bool,
  +good: {B.valid(b) == True{} : Bool},+canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  r: Array<U64> & List<&2,Chess.Ply>,fields: C.every(m => Source.generated(b,m),E.moves(r))) ->
  Children.every(Chess.children(E.moves(r),b),Bool.not(white)):
  (table,+xs)=r
  Children.lift(b,xs,white,Post.occurrences(b,xs,white,good,canonical,fields))

def legal_children(table: Array<U64>,+b: Chess.Board,+white: Bool,
  +good: {B.valid(b) == True{} : Bool},+canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  Children.every(Chess.children(E.moves(Chess.legal_moves(table,b)),b),Bool.not(white)):
  Source.legal(r => Children.every(Chess.children(E.moves(r),b),Bool.not(white)),table,b,
    r => fields => children_result(b,white,good,canonical,r,fields))
