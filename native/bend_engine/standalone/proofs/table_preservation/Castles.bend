# Both branches of the actual castling producer preserve arbitrary table values.
import Base
import ./Core.bend as C
import ./Checks.bend as K
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def finish(+m: Chess.Ply,+start: Bool,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_finish(m,start,acc,r)) == C.table(Bool,r) : Array<U64>}:
  (table,transit) = r
  {==}

def start(+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_start(b,src,dst,transit,acc,r)) == C.table(Bool,r) : Array<U64>}:
  (table,+initial) = r
  C.apply(List<&2,Chess.Ply>,a => Chess.castle_start(b,src,dst,transit,acc,(a,initial)),table,+c =>
    Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.castle_start(b,src,dst,transit,acc,(O.pack(c),initial))),
      C.table(Bool,Chess.in_check(O.pack(c),Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b))),O.pack(c),
      finish(Chess.Ply{src,dst,0,2},initial,acc,Chess.in_check(O.pack(c),Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b))),
      K.check(O.pack(c),Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b))))

def checked(ok: Bool,table: Array<U64>,+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+acc: List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_checked(ok,table,b,src,dst,transit,acc)) == table : Array<U64>}:
  match ok:
    case False{}: {==}
    case True{}:
      C.apply(List<&2,Chess.Ply>,a => Chess.castle_checked(True{},a,b,src,dst,transit,acc),table,+c =>
        Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.castle_checked(True{},O.pack(c),b,src,dst,transit,acc)),
          C.table(Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b))),O.pack(c),
          start(b,src,dst,transit,acc,Chess.in_check(O.pack(c),b,Chess.get_turn(b))),
          K.check(O.pack(c),b,Chess.get_turn(b))))

def side(+b: Chess.Board,+king_side: Bool,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.castle_side(b,king_side,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,acc) = r
  +home = Bool.pick(U32, U32.is_eq(Chess.get_turn(b), 1), 0, 56)
  +src = U32.add(home, 4)
  rook = U32.add(home, Bool.pick(U32, king_side, 7, 0))
  dst = U32.add(home, Bool.pick(U32, king_side, 6, 2))
  transit = U32.add(home, Bool.pick(U32, king_side, 5, 3))
  right = U32.mul(Bool.pick(U32, U32.is_eq(Chess.get_turn(b), 1), 1, 4), Bool.pick(U32, king_side, 1, 2))
  between = U64.shln(U64.from_u32(Bool.pick(U32, king_side, 96, 14)), U32.to_nat(home))
  +own = Chess.color(b, Chess.get_turn(b))
  ok = Bool.and(Bool.not(U32.is_zero(U32.and(Chess.get_rights(b), right))),
    Bool.and(U64.test_bit(U64.and(Chess.get_kings(b), own), U32.to_nat(src)),
      Bool.and(U64.test_bit(U64.and(Chess.get_rooks(b), own), U32.to_nat(rook)), U64.is_zero(U64.and(Chess.occupied(b), between)))))
  checked(ok,table,b,src,dst,transit,acc)
