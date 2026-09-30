import Base
import ./Actual.bend as C
import ./Preserve.bend as P
import ./Route.bend as R
import ../board/Spec.bend as B
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def exact(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(b,Chess.Ply{src,dst,0,2}) == C.expected(b,src,dst) : Chess.Board}:
  C.actual(b,src,dst)

def conditional(+b: Chess.Board,+route: R.Route,good: {B.valid(b) == True{} : Bool},
  fresh: {B.fresh_mask(b,R.rook(route)) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{R.src(route),R.dst(route),0,2})) == True{} : Bool}:
  P.actual(b,route,good,fresh)

def input(src: U32,rook: U32,+white: Bool,turn: U32) -> Chess.Board:
  Position.metadata(Position.put(5,white,src,Position.put(3,white,rook,Position.empty())),turn,15,64)

def white_king() ->
  {Chess.make_move(input(4,7,True{},1),Chess.Ply{4,6,0,2}) ==
    Position.metadata(Position.put(5,True{},6,Position.put(3,True{},5,Position.empty())),0,12,64) : Chess.Board}: {==}

def white_queen() ->
  {Chess.make_move(input(4,0,True{},1),Chess.Ply{4,2,0,2}) ==
    Position.metadata(Position.put(5,True{},2,Position.put(3,True{},3,Position.empty())),0,12,64) : Chess.Board}: {==}

def black_king() ->
  {Chess.make_move(input(60,63,False{},0),Chess.Ply{60,62,0,2}) ==
    Position.metadata(Position.put(5,False{},62,Position.put(3,False{},61,Position.empty())),1,3,64) : Chess.Board}: {==}

def black_queen() ->
  {Chess.make_move(input(60,56,False{},0),Chess.Ply{60,58,0,2}) ==
    Position.metadata(Position.put(5,False{},58,Position.put(3,False{},59,Position.empty())),1,3,64) : Chess.Board}: {==}

# A representation-consistent but illegal raw request can overlap at the rook target.
def blocked() -> Chess.Board:
  Position.put(2,False{},5,input(4,7,True{},1))

def blocked_input_consistent() -> {B.valid(blocked()) == True{} : Bool}: {==}
def blocked_rook_target() -> {B.fresh_mask(blocked(),R.rook(R.WhiteKing{})) == False{} : Bool}: {==}
def blocked_output_inconsistent() ->
  {B.valid(Chess.make_move(blocked(),Chess.Ply{4,6,0,2})) == False{} : Bool}: {==}

# Conditions sufficient for partition preservation are not sufficient for legality.
def absent_king_rook_still_consistent() ->
  {B.valid(Chess.make_move(Position.empty(),Chess.Ply{4,6,0,2})) == True{} : Bool}:
  P.actual(Position.empty(),R.WhiteKing{},{==},{==})
