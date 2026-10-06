# Independent child geometry and inhabited parent premises in both directions.
import Base
import ../accepted_ep/Boards.bend as B
import ../board/Spec.bend as Board
import ../initialized_attacks/Spec.bend as G
import ../ep_king_frame/Frame.bend as F
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def white_parent() -> {Board.valid(B.white()) == True{} : Bool} &
  ({Position.valid_ep(C.get_ep(B.white()),B.white()) == True{} : Bool} &
  {F.plane(B.white(),1) == U64.bit(0n) : U64}): ({==},({==},{==}))
def black_parent() -> {Board.valid(B.black()) == True{} : Bool} &
  ({Position.valid_ep(C.get_ep(B.black()),B.black()) == True{} : Bool} &
  {F.plane(B.black(),0) == U64.bit(63n) : U64}): ({==},({==},{==}))
def white_child_clear() ->
  {G.attacked(C.make_move(B.white(),C.Ply{36,43,0,1}),0n,False{}) == False{} : Bool}: {==}
def black_child_clear() ->
  {G.attacked(C.make_move(B.black(),C.Ply{27,20,0,1}),63n,True{}) == False{} : Bool}: {==}
def discovered_parent() -> {Board.valid(B.discovered()) == True{} : Bool} &
  ({Position.valid_ep(C.get_ep(B.discovered()),B.discovered()) == True{} : Bool} &
  ({C.get_turn(B.discovered()) == Bool.to_u32(True{}) : U32} &
  {F.plane(B.discovered(),Bool.to_u32(True{})) == U64.bit(32n) : U64})):
  ({==},({==},({==},{==})))
def discovered_child_attacked() ->
  {G.attacked(C.make_move(B.discovered(),C.Ply{36,43,0,1}),32n,False{}) == True{} : Bool}: {==}
def discovered_child_other_side_clear() ->
  {G.attacked(C.make_move(B.discovered(),C.Ply{36,43,0,1}),63n,True{}) == False{} : Bool}: {==}
def wrong_square_clear() ->
  {G.attacked(C.make_move(B.discovered(),C.Ply{36,43,0,1}),43n,False{}) == False{} : Bool}: {==}
def configuration() -> {17n == 17n : Nat} & ({128n == 128n : Nat} & {64n == 64n : Nat}):
  ({==},({==},{==}))
