import Base
import ./Algebra.bend as A
import ./Occupied.bend as O
import ../board/Spec.bend as B
import ../ray/Frame.bend as F
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as Chess

def mask(r: Array<U64> & U64) -> U64:
  (table,targets) = r
  targets

def empty_square(+b: Chess.Board,+i: Nat,good: {B.valid(b) == True{} : Bool},
  empty: {U64.test_bit(Chess.occupied(b),i) == False{} : Bool}) -> D.disjoint(U64.bit(i),Chess.get_kings(b)):
  D.disjoint_subset(U64.bit(i),Chess.get_kings(b),Chess.occupied(b),O.kings_occupied(b,good),A.test_zero(Chess.occupied(b),i,empty))

def selected_empty(+b: Chess.Board,+i: Nat,ok: Bool,good: {B.valid(b) == True{} : Bool},
  guarded: {Bool.and(ok,U64.test_bit(Chess.occupied(b),i)) == False{} : Bool}) ->
  D.disjoint(Chess.select_u64(ok,U64.bit(i),U64.zero()),Chess.get_kings(b)):
  match ok:
    case False{}: D.left_zero(Chess.get_kings(b))
    case True{}: empty_square(b,i,good,guarded)

def nested_empty(+a: Bool,+b: Bool,+t: Bool) -> {Bool.and(Bool.and(a,Bool.and(b,Bool.not(t))),t) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: A.empty_guard_bool(b,t)

def ep_guard(+a: Bool,+b: Bool,+empty: Bool,+victim: Bool) ->
  {Bool.and(Bool.and(a,Bool.and(b,Bool.and(Bool.not(empty),victim))),empty) == False{} : Bool}:
  match a b empty victim:
    case False{} _ _ _: {==}
    case True{} False{} _ _: {==}
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} True{} _: {==}

def ep(+b: Chess.Board,good: {B.valid(b) == True{} : Bool}) -> D.disjoint(Chess.ep_target(b),Chess.get_kings(b)):
  +target = Chess.get_ep(b)
  +w = U32.is_eq(Chess.get_turn(b),1)
  +rank = Bool.pick(U32,w,5,2)
  +victim = U32.xor(target,8)
  +a = U32.is_lt(target,64)
  +rank_ok = U32.is_eq(U32.div(target,8),rank)
  +empty = U64.test_bit(Chess.occupied(b),U32.to_nat(target))
  +present = U64.test_bit(U64.and(Chess.get_pawns(b),Chess.color(b,U32.xor(Chess.get_turn(b),1))),U32.to_nat(victim))
  +ok = Bool.and(a,Bool.and(rank_ok,Bool.and(Bool.not(empty),present)))
  selected_empty(b,U32.to_nat(target),ok,good,ep_guard(a,rank_ok,empty,present))

def gate_right(+a: U64,+capturable: U64,+k: U64,clear: D.disjoint(capturable,k)) -> D.disjoint(U64.and(a,capturable),k):
  Equal.trans(U64,U64.and(U64.and(a,capturable),k),U64.and(U64.and(capturable,a),k),U64.zero(),
    Equal.cong(U64,U64,x => U64.and(x,k),U64.and(a,capturable),U64.and(capturable,a),A.and_commute(a,capturable)),
    A.gate_clear(capturable,a,k,clear))

def pawn_after(-Goal: @-r: Array<U64> & U64 -> Type,+advances: U64,+capturable: U64,+k: U64,r: Array<U64> & U64,
  advance_clear: D.disjoint(advances,k),capture_clear: D.disjoint(capturable,k),
  next: @r: Array<U64> & U64 -> D.disjoint(mask(r),k) -> Goal(r)) -> Goal(Chess.pawn_after(advances,capturable,r)):
  (table,+pa) = r
  next((table,U64.or(advances,U64.and(pa,capturable))),A.union_clear(advances,U64.and(pa,capturable),k,advance_clear,gate_right(pa,capturable,k,capture_clear)))

def pawn_result(-Goal: @-r: Array<U64> & U64 -> Type,table: Array<U64>,+b: Chess.Board,+src: U32,+good: {B.valid(b) == True{} : Bool},
  next: @r: Array<U64> & U64 -> D.disjoint(mask(r),Chess.get_kings(b)) -> Goal(r)) -> Goal(Chess.pawn_targets(table,b,src)):
  +w = U32.is_eq(Chess.get_turn(b),1)
  +step = Bool.pick(U32,w,8,4294967288)
  +first = U32.add(src,step)
  +second = U32.add(first,step)
  +occ = Chess.occupied(b)
  +t1 = U64.test_bit(occ,U32.to_nat(first))
  +t2 = U64.test_bit(occ,U32.to_nat(second))
  +empty1 = Bool.and(U32.is_lt(first,64),Bool.not(t1))
  +start = U32.is_eq(U32.div(src,8),Bool.pick(U32,w,1,6))
  +empty2 = Bool.and(empty1,Bool.and(start,Bool.not(t2)))
  +one = Chess.select_u64(empty1,U64.bit(U32.to_nat(first)),U64.zero())
  +two = Chess.select_u64(empty2,U64.bit(U32.to_nat(second)),U64.zero())
  +enemies = U64.and_not(Chess.color(b,U32.xor(Chess.get_turn(b),1)),Chess.get_kings(b))
  pawn_after(Goal,U64.or(one,two),U64.or(enemies,Chess.ep_target(b)),Chess.get_kings(b),Chess.attack(0,table,src,Chess.get_turn(b),occ),
    A.union_clear(one,two,Chess.get_kings(b),
      selected_empty(b,U32.to_nat(first),empty1,good,A.empty_guard_bool(U32.is_lt(first,64),t1)),
      selected_empty(b,U32.to_nat(second),empty2,good,nested_empty(empty1,start,t2))),
    A.union_clear(enemies,Chess.ep_target(b),Chess.get_kings(b),A.capturable(Chess.color(b,U32.xor(Chess.get_turn(b),1)),Chess.get_kings(b)),ep(b,good)),next)

def excluded(-Goal: @-r: Array<U64> & U64 -> Type,+own: U64,+k: U64,r: Array<U64> & U64,
  next: @r: Array<U64> & U64 -> D.disjoint(mask(r),k) -> Goal(r)) -> Goal(Chess.exclude_targets(U64.or(own,k),r)):
  (table,+targets) = r
  next((table,U64.and_not(targets,U64.or(own,k))),A.excluded(targets,own,k))

def piece(-Goal: @-r: Array<U64> & U64 -> Type,pawn: Bool,table: Array<U64>,+b: Chess.Board,+src: U32,good: {B.valid(b) == True{} : Bool},
  next: @r: Array<U64> & U64 -> D.disjoint(mask(r),Chess.get_kings(b)) -> Goal(r)) -> Goal(Chess.piece_targets(pawn,table,b,src)):
  match pawn:
    case True{}: pawn_result(Goal,table,b,src,good,next)
    case False{}: excluded(Goal,Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b),Chess.attack(Chess.piece(b,src),table,src,Chess.get_turn(b),Chess.occupied(b)),next)
