# Local source-centred rook observations, without expanding a whole closed scan.
import Base
import ../forward_attacks/Spec.bend as Forward
import ../accepted_ep/Boards.bend as B
import ../../../legal_probe/Chess.bend as Chess

def discovered_rook_edge() ->
  {Forward.edge(Forward.Rook{},39n,32n,False{},
    Chess.occupied(Chess.make_move(B.discovered(),Chess.Ply{36,43,0,1}))) == True{} : Bool}:
  {==}

def parent_blocked() ->
  {Forward.edge(Forward.Rook{},39n,32n,False{},Chess.occupied(B.discovered())) == False{} : Bool}:
  {==}

def discovered_rook_owned() ->
  {Bool.and(U64.test_bit(Chess.get_rooks(Chess.make_move(B.discovered(),Chess.Ply{36,43,0,1})),39n),
    U64.test_bit(Chess.color(Chess.make_move(B.discovered(),Chess.Ply{36,43,0,1}),0),39n)) == True{} : Bool}:
  {==}

def diagonal_kind_clear() ->
  {Forward.edge(Forward.Bishop{},39n,32n,False{},
    Chess.occupied(Chess.make_move(B.discovered(),Chess.Ply{36,43,0,1}))) == False{} : Bool}:
  {==}
