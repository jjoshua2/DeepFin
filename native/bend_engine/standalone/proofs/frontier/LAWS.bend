# Actual initialized parser trajectories, not arbitrary true-flag caller states.
import Base
import ./Spec.bend as Spec
import ../../Position.bend as Position
import ../parser/Spec.bend as Parser
import ../board/Spec.bend as Board

law layout_preserves_frontier:
  for +text: String
  for +state: Position.Layout
  for good: {Spec.safe(state) == True{} : Bool}
  {Spec.safe(Position.layout(text,state)) == True{} : Bool}

law accepted_prefix_has_frontier:
  for +prefix: String
  for +suffix: String
  for accepted: {Parser.accepted(Position.layout_result(Position.layout(String.append(prefix,suffix),Spec.initial()))) == True{} : Bool}
  {Spec.ready(Position.layout(prefix,Spec.initial())) == True{} : Bool}

law live_typed_insertion_is_fresh:
  for +prefix: String
  for +piece: Board.Piece
  for +white: Bool
  for live: {Parser.live(Position.layout(Spec.typed(piece,white),Position.layout(prefix,Spec.initial()))) == True{} : Bool}
  {Spec.fresh_target(Position.layout(prefix,Spec.initial())) == True{} : Bool}

law accepted_placement_has_consistent_board:
  for +text: String
  {Spec.result_valid(Position.layout_result(Position.layout(text,Spec.initial()))) == True{} : Bool}
