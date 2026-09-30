import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../board/PROOF.bend as BoardProof

def split(+prefix: String,+suffix: String,+l: Position.Layout) ->
  {Position.layout(String.append(prefix,suffix),l) == Position.layout(suffix,Position.layout(prefix,l)) : Position.Layout}:
  Laws.layout_append_exact(prefix,suffix,l)

def invalid_suffix(+s: String,+b: Chess.Board,+f: U32,+r: U32) ->
  {Position.layout_result(Position.layout(s,Position.Layout{b,f,r,False{}})) == None{} : Maybe<&2,Chess.Board>}:
  Laws.invalid_layout_rejected(s,Position.Layout{b,f,r,False{}},{==})

def accepted_prefix(+a: String,+b: String,+l: Position.Layout,
  e: {S.accepted(Position.layout_result(Position.layout(String.append(a,b),l))) == True{} : Bool}) ->
  {S.live(Position.layout(a,l)) == True{} : Bool}:
  Laws.accepted_layout_has_valid_prefix(a,b,l,e)

def metadata(+s: String,+b: Chess.Board,+f: U32,+r: U32,+v: Bool) ->
  {S.metadata(Position.layout(s,Position.Layout{b,f,r,v})) == B.metadata(b) : U32 & (U32 & U32)}:
  Laws.layout_preserves_metadata(s,Position.Layout{b,f,r,v})

def insert(+p: B.Piece,+w: Bool,+b: Chess.Board,+f: U32,+r: U32,+v: Bool) ->
  {Position.layout(SCon{Chr{S.letter(p,w)},SNil{}},Position.Layout{b,f,r,v}) ==
    Position.Layout{Position.put(B.tag(p),w,U32.add(f,U32.mul(r,8)),b),U32.inc(f),r,Bool.and(v,U32.is_lt(f,8))} : Position.Layout}:
  Laws.typed_piece_transition(p,w,b,f,r,v)

# Acceptance can include an empty board: this is placement syntax, not chess legality.
def accepted_empty_prefix() ->
  {S.live(Position.layout("8/8/",Position.Layout{Position.empty(),0,7,True{}})) == True{} : Bool}:
  Laws.accepted_layout_has_valid_prefix("8/8/","8/8/8/8/8/8",Position.Layout{Position.empty(),0,7,True{}},{==})

# A bad digit overflow cannot be repaired by later well-formed ranks.
def overflow_never_recovers() ->
  {Position.layout_result(Position.layout("8/8/8/8/8/8/8",Position.layout("88/",Position.Layout{Position.empty(),0,7,True{}}))) == None{} : Maybe<&2,Chess.Board>}:
  Laws.invalid_layout_rejected("8/8/8/8/8/8/8",Position.layout("88/",Position.Layout{Position.empty(),0,7,True{}}),{==})

# Raw traversal is not rollback: invalid nonpiece 'x' still changes a color plane.
def rejection_is_not_board_rollback() ->
  {Chess.get_black(S.board(Position.layout("x",Position.Layout{Position.empty(),0,7,True{}}))) == U64.bit(56n) : U64}:
  {==}

def same_invalid_step_returns_none() ->
  {Position.layout_result(Position.layout("x",Position.Layout{Position.empty(),0,7,True{}})) == None{} : Maybe<&2,Chess.Board>}:
  {==}
