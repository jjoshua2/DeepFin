# Four promotion choices and complete updates; no legal-move claim.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ./Spec.bend as Spec

law promotion_choice_roundtrip:
  for +choice: Spec.Choice
  {Spec.from_tag(Spec.tag(choice)) == Some{choice} : Maybe<&2,Spec.Choice>}

law promotion_choices_exact:
  for +src: U32
  for +dst: U32
  for flag: U32
  for +tail: List<&2,Chess.Ply>
  {Chess.put_move(True{},src,dst,flag,tail) == List.append(&2,Chess.Ply,Spec.choices(src,dst),tail) : List<&2,Chess.Ply>}

law promotion_make_move_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for +choice: Spec.Choice
  {Chess.make_move(b,Spec.ply(src,dst,choice)) == Spec.expected(b,src,dst,choice) : Chess.Board}

law promotion_make_move_preserves_representation:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for +choice: Spec.Choice
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Spec.ply(src,dst,choice))) == True{} : Bool}
