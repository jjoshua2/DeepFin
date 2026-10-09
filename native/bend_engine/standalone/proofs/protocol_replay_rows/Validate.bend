# Validation can reject, but successful output retains its input row certificate.
import Base
import ./Spec.bend as S
import ../fen_validated_full/Root.bend as Root
import ../board/Spec.bend as B
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../table_preservation/Checks.bend as Checks
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def checked(+c: O.Cells,+g: Pos.Game,r: Array<U64> & Bool,
  kept: {T.table(Bool,r) == O.pack(c) : Array<U64>},
  good: {B.valid(Pos.board(g)) == True{} : Bool}) ->
  S.root(c,Protocol.king_checked(g,r)):
  (a,illegal)=r
  match illegal:
    case True{}: (kept,Unit{})
    case False{}: (kept,good)

def board(+c: O.Cells,+g: Pos.Game,valid: Bool,
  good: {B.valid(Pos.board(g)) == True{} : Bool}) ->
  S.root(c,Protocol.valid_board(valid,g,O.pack(c))):
  match valid:
    case False{}: ({==},Unit{})
    case True{}:
      checked(c,g,Chess.in_check(O.pack(c),Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1)),
        Checks.check(O.pack(c),Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1)),good)

def known(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,good: Root.rows(maybe)) ->
  S.root(c,Protocol.validate(maybe,O.pack(c))):
  match maybe:
    case None{}: ({==},Unit{})
    case Some{+g}: board(c,g,Pos.valid(Pos.board(g)),good)
