import Base
import ./Post.bend as Post
import ./Children.bend as Children
import ../board/Spec.bend as B
import ../candidate_bypass_classification/Generated.bend as G
import ../candidate_bypass_classification/Fields.bend as Fields
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

# An arbitrary actual input array is consumed by the real classifier continuation.
def prefilter_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},
  here: E.member(m,E.moves(G.produced(table,b)))) -> Post.valid(b,m):
  Post.generated(b,m,good,G.member_prefilter(table,b,m,here))

def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> Post.valid(b,m):
  Post.generated(b,m,good,G.member_legal(table,b,m,here))

def prefilter_all(table: Array<U64>,+b: Chess.Board,
  +good: {B.valid(b) == True{} : Bool}) ->
  C.every(m => Post.valid(b,m),E.moves(G.produced(table,b))):
  G.prefilter(r => C.every(m => Post.valid(b,m),E.moves(r)),table,b,
    r => fields => Post.occurrences(b,E.moves(r),good,fields))

def legal_all(table: Array<U64>,+b: Chess.Board,
  +good: {B.valid(b) == True{} : Bool}) ->
  C.every(m => Post.valid(b,m),E.moves(Chess.legal_moves(table,b))):
  G.legal(r => C.every(m => Post.valid(b,m),E.moves(r)),table,b,
    r => fields => Post.occurrences(b,E.moves(r),good,fields))

# No post-board validity, tag bounds or castle target freshness is supplied.
def children_result(+b: Chess.Board,+good: {B.valid(b) == True{} : Bool},
  r: Array<U64> & List<&2,Chess.Ply>,
  fields: C.every(m => Fields.generated(b,m),E.moves(r))) ->
  Children.every(Chess.children(E.moves(r),b)):
  (table,+xs)=r
  Children.lift(b,xs,Post.occurrences(b,xs,good,fields))

def legal_children(table: Array<U64>,+b: Chess.Board,
  +good: {B.valid(b) == True{} : Bool}) ->
  Children.every(Chess.children(E.moves(Chess.legal_moves(table,b)),b)):
  G.legal(r => Children.every(Chess.children(E.moves(r),b)),table,b,
    r => fields => children_result(b,good,r,fields))
