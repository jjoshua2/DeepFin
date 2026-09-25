# The public optional result may reject; consistency is not chess legality.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position
import ../board/Spec.bend as B
import ../frontier/Spec.bend as Frontier

def placement(pieces: String) -> Maybe<&2,Chess.Board>:
  Position.layout_result(Position.layout(pieces,Frontier.initial()))


def game_consistent(result: Maybe<&2,Position.Game>) -> Bool:
  match result:
    case None{}: True{}
    case Some{Position.Game{b,h,f,hist}}: B.valid(b)
