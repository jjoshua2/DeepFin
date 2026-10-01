# The real ordinary piece scan only observes its threaded array.
import Base
import ./Core.bend as C
import ./Queries.bend as Q
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def pawn_after(+advances: U64,+capturable: U64,r: Array<U64> & U64) ->
  {C.table(U64,Chess.pawn_after(advances,capturable,r)) == C.table(U64,r) : Array<U64>}:
  (table,pa) = r
  {==}

def pawn(table: Array<U64>,+b: Chess.Board,+src: U32) ->
  {C.table(U64,Chess.pawn_targets(table,b,src)) == table : Array<U64>}:
  +w = U32.is_eq(Chess.get_turn(b), 1)
  +step = Bool.pick(U32, w, 8, 4294967288)
  +first = U32.add(src, step)
  +second = U32.add(first, step)
  +occ = Chess.occupied(b)
  +empty1 = Bool.and(U32.is_lt(first, 64), Bool.not(U64.test_bit(occ, U32.to_nat(first))))
  start_rank = Bool.pick(U32, w, 1, 6)
  empty2 = Bool.and(empty1, Bool.and(U32.is_eq(U32.div(src, 8), start_rank),
    Bool.not(U64.test_bit(occ, U32.to_nat(second)))))
  +advances = U64.or(Chess.select_u64(empty1, U64.bit(U32.to_nat(first)), U64.zero()),
    Chess.select_u64(empty2, U64.bit(U32.to_nat(second)), U64.zero()))
  +enemies = U64.and_not(Chess.color(b, U32.xor(Chess.get_turn(b), 1)), Chess.get_kings(b))
  C.apply(U64,a => Chess.pawn_targets(a,b,src),table,+c =>
    Equal.trans(Array<U64>,C.table(U64,Chess.pawn_targets(O.pack(c),b,src)),
      C.table(U64,Chess.attack(0,O.pack(c),src,Chess.get_turn(b),occ)),O.pack(c),
      pawn_after(advances,U64.or(enemies,Chess.ep_target(b)),Chess.attack(0,O.pack(c),src,Chess.get_turn(b),occ)),
      Q.attack(0,O.pack(c),src,Chess.get_turn(b),occ)))

def exclude(+excluded: U64,r: Array<U64> & U64) ->
  {C.table(U64,Chess.exclude_targets(excluded,r)) == C.table(U64,r) : Array<U64>}:
  (table,targets) = r
  {==}

def piece(is_pawn: Bool,table: Array<U64>,+b: Chess.Board,+src: U32) ->
  {C.table(U64,Chess.piece_targets(is_pawn,table,b,src)) == table : Array<U64>}:
  match is_pawn:
    case True{}: pawn(table,b,src)
    case False{}:
      C.apply(U64,a => Chess.piece_targets(False{},a,b,src),table,+c =>
        Equal.trans(Array<U64>,C.table(U64,Chess.piece_targets(False{},O.pack(c),b,src)),
          C.table(U64,Chess.attack(Chess.piece(b,src),O.pack(c),src,Chess.get_turn(b),Chess.occupied(b))),O.pack(c),
          exclude(U64.or(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b)),
            Chess.attack(Chess.piece(b,src),O.pack(c),src,Chess.get_turn(b),Chess.occupied(b))),
          Q.attack(Chess.piece(b,src),O.pack(c),src,Chess.get_turn(b),Chess.occupied(b))))

def after(+src: U32,+is_pawn: Bool,+ep: U32,+acc: List<&2,Chess.Ply>,r: Array<U64> & U64) ->
  {C.table(List<&2,Chess.Ply>,Chess.scan_after(src,is_pawn,ep,acc,r)) == C.table(U64,r) : Array<U64>}:
  (table,targets) = r
  {==}

def step(+b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.scan_step(b,src,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  (table,+acc) = r
  +is_pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  C.apply(List<&2,Chess.Ply>,a => Chess.scan_step(b,src,(a,acc)),table,+c =>
    Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.scan_step(b,src,(O.pack(c),acc))),
      C.table(U64,Chess.piece_targets(is_pawn,O.pack(c),b,src)),O.pack(c),
      after(src,is_pawn,Chess.get_ep(b),acc,Chess.piece_targets(is_pawn,O.pack(c),b,src)),
      piece(is_pawn,O.pack(c),b,src)))

def scan(xs: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {C.table(List<&2,Chess.Ply>,Chess.scan(xs,b,r)) == C.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  match xs:
    case Nil{}:
      (table,acc) = r
      {==}
    case Con{+src,+rest}:
      (table,+acc) = r
      C.apply(List<&2,Chess.Ply>,a => Chess.scan(Con{src,rest},b,(a,acc)),table,+c =>
        Equal.trans(Array<U64>,C.table(List<&2,Chess.Ply>,Chess.scan(Con{src,rest},b,(O.pack(c),acc))),
          C.table(List<&2,Chess.Ply>,Chess.scan_step(b,src,(O.pack(c),acc))),O.pack(c),
          scan(rest,b,Chess.scan_step(b,src,(O.pack(c),acc))),step(b,src,(O.pack(c),acc))))
