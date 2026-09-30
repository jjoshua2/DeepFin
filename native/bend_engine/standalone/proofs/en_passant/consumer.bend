# Public flag-one contracts, both colors and excluded-legality boundary.
import Base
import ./Actual.bend as EP
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../board/Spec.bend as B
import ../board/LAWS.bend as BL
import ../board/PROOF.bend as BP
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def exact(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(b,Chess.Ply{src,dst,0,1}) == EP.expected(b,src,dst) : Chess.Board}:
  Laws.en_passant_make_move_exact_update(b,src,dst)

def consistent(+b: Chess.Board,+src: U32,+dst: U32,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}:
  Laws.en_passant_make_move_preserves_representation(b,src,dst,good)

# e5xd6 removes d5 while leaving a distant black rook unchanged.
def white_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},36,Position.put(0,False{},35,Position.put(3,False{},56,Position.empty()))),1,15,43)

def white_capture_exact() ->
  {Chess.make_move(white_board(),Chess.Ply{36,43,0,1}) ==
    Position.metadata(Position.put(0,True{},43,Position.put(3,False{},56,Position.empty())),0,15,64) : Chess.Board}:
  {==}

# e4xd3 removes d4; the other side and rank arithmetic are not inferred from white.
def black_capture_exact() ->
  {Chess.make_move(Position.metadata(Position.put(0,False{},28,Position.put(0,True{},27,Position.empty())),0,15,19),Chess.Ply{28,19,0,1}) ==
    Position.metadata(Position.put(0,False{},19,Position.empty()),1,15,64) : Chess.Board}:
  {==}

# The raw function does not validate a source pawn or EP rights.
def empty_source_is_not_legality() ->
  {B.valid(Chess.make_move(Position.empty(),Chess.Ply{0,1,0,1})) == True{} : Bool}:
  Laws.en_passant_make_move_preserves_representation(Position.empty(),0,1,BL.empty_representation())
