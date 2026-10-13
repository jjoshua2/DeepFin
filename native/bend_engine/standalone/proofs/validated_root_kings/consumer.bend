import Base
import ./Kings.bend as Kings
import ./Validate.bend as V
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess
import ../prepared_ray_coverage/Query.bend as Q

# No caller king-presence, selected-square, partition, canonical-turn or table premise.
def validated_root(table: Array<U64>,maybe: Maybe<&2,Pos.Game>) -> V.output(Protocol.validate(maybe,table)):
  V.validate(r => V.output(r),table,maybe,r => good => good)

def validator_lookup(+g: Pos.Game) -> V.before_query(g,Pos.valid(Pos.board(g))):
  V.precheck(g)

def filter_output(out: Maybe<&2,Pos.Game>,-rays: U64,-dst: U32,-promotion: U32,-flag: U32) -> Type:
  match out:
    case None{}: Unit
    case Some{g}:
      {Chess.filter_requires(U64.and(Chess.color(Pos.board(g),Chess.get_turn(Pos.board(g))),U64.or(rays,Chess.get_kings(Pos.board(g)))),
        Chess.Ply{Q.king(Pos.board(g)),dst,promotion,flag}) == True{} : Bool}

def required_result(+out: Maybe<&2,Pos.Game>,+rays: U64,+dst: U32,+promotion: U32,+flag: U32,
  good: V.accepted(out)) -> filter_output(out,rays,dst,promotion,flag):
  match out:
    case None{}: Unit{}
    case Some{+g}:
      (valid,rest)=good
      Kings.required(Pos.board(g),rays,dst,promotion,flag,valid)

# One actual validator result is threaded through this dependent continuation.
def required(table: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+rays: U64,+dst: U32,+promotion: U32,+flag: U32) ->
  filter_output(V.value(Protocol.validate(maybe,table)),rays,dst,promotion,flag):
  V.validate(r => filter_output(V.value(r),rays,dst,promotion,flag),table,maybe,
    r => good => required_result(V.value(r),rays,dst,promotion,flag,good))
