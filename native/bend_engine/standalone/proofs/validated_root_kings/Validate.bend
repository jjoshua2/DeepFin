import Base
import ./Kings.bend as Kings
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def root(-g: Pos.Game) -> Type:
  {Pos.valid(Pos.board(g)) == True{} : Bool} &
  (Kings.selected(Pos.board(g)) & Kings.safe_query(Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1)))

def accepted(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: root(g)

def value(r: Array<U64> & Maybe<&2,Pos.Game>) -> Maybe<&2,Pos.Game>:
  (table,out)=r
  out

def output(r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  accepted(value(r))

def certify(+g: Pos.Game,+good: {Pos.valid(Pos.board(g)) == True{} : Bool}) -> root(g):
  (good,(Kings.selection(Pos.board(g),good),Kings.query_bound(Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1),good)))

def checked(-Goal: @-r: Array<U64> & Maybe<&2,Pos.Game> -> Type,
  +g: Pos.Game,r: Array<U64> & Bool,good: {Pos.valid(Pos.board(g)) == True{} : Bool},
  next: @r: Array<U64> & Maybe<&2,Pos.Game> -> accepted(value(r)) -> Goal(r)) ->
  Goal(Protocol.king_checked(g,r)):
  (table,illegal)=r
  match illegal:
    case True{}: next((table,None{}),Unit{})
    case False{}: next((table,Some{g}),certify(g,good))

def board(-Goal: @-r: Array<U64> & Maybe<&2,Pos.Game> -> Type,
  +g: Pos.Game,table: Array<U64>,valid: Bool,
  same: {Pos.valid(Pos.board(g)) == valid : Bool},
  next: @r: Array<U64> & Maybe<&2,Pos.Game> -> accepted(value(r)) -> Goal(r)) -> Goal(Protocol.valid_board(valid,g,table)):
  match valid:
    case False{}: next((table,None{}),Unit{})
    case True{}:
      checked(Goal,g,Chess.in_check(table,Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1)),same,next)

# The branch certificate comes from the actual validator, not from the caller.
def validate(-Goal: @-r: Array<U64> & Maybe<&2,Pos.Game> -> Type,
  table: Array<U64>,maybe: Maybe<&2,Pos.Game>,
  next: @r: Array<U64> & Maybe<&2,Pos.Game> -> accepted(value(r)) -> Goal(r)) -> Goal(Protocol.validate(maybe,table)):
  match maybe:
    case None{}: next((table,None{}),Unit{})
    case Some{+g}: board(Goal,g,table,Pos.valid(Pos.board(g)),{==},next)

def before_query(-g: Pos.Game,valid: Bool) -> Type:
  match valid:
    case False{}: Unit
    case True{}: Kings.safe_query(Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1))

def precheck_cases(+g: Pos.Game,valid: Bool,same: {Pos.valid(Pos.board(g)) == valid : Bool}) ->
  before_query(g,valid):
  match valid:
    case False{}: Unit{}
    case True{}: Kings.query_bound(Pos.board(g),U32.xor(Chess.get_turn(Pos.board(g)),1),same)

def precheck(+g: Pos.Game) -> before_query(g,Pos.valid(Pos.board(g))):
  precheck_cases(g,Pos.valid(Pos.board(g)),{==})
