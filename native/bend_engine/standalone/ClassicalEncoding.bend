# DeepFin v1 (34) and v2_threats (63) planes. No foreign feature computation.
# Pseudo-attacks and pawn mobility intentionally match the training encoders,
# not legal move counts. All output squares use the current side's perspective.
import Base
import ../legal_probe/Chess.bend as Chess
import ./Tables.bend as Tables
import ./HistoryEncoding.bend as H

# Attack unions by piece type plus 3-bit saturating attacker counts (0..7).
type Maps is Data:
  Maps{p: U64, n: U64, b: U64, r: U64, q: U64, k: U64, one: U64, two: U64, four: U64}

def empty() -> Maps:
  Maps{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero()}

def select_mask(kind: U32, p: U64, n: U64, b: U64, r: U64, q: U64, k: U64) -> U64:
  match kind:
    case 0: p
    case 1: n
    case 2: b
    case 3: r
    case 4: q
    case _: k

def typed(kind: U32, m: Maps) -> U64:
  Maps{p, n, b, r, q, k, one, two, four} = m
  select_mask(kind, p, n, b, r, q, k)

def union(m: Maps) -> U64:
  Maps{p, n, b, r, q, k, one, two, four} = m
  U64.or(U64.or(U64.or(p, n), U64.or(b, r)), U64.or(q, k))

def add(+kind: U32, +att: U64, m: Maps) -> Maps:
  Maps{p, n, b, r, q, k, +one, +two, +four} = m
  +inc = U64.and(att, U64.not(U64.and(one, U64.and(two, four))))
  +carry = U64.and(one, inc)
  Maps{U64.or(p, Chess.select_u64(U32.is_eq(kind, 0), att, U64.zero())),
    U64.or(n, Chess.select_u64(U32.is_eq(kind, 1), att, U64.zero())),
    U64.or(b, Chess.select_u64(U32.is_eq(kind, 2), att, U64.zero())),
    U64.or(r, Chess.select_u64(U32.is_eq(kind, 3), att, U64.zero())),
    U64.or(q, Chess.select_u64(U32.is_eq(kind, 4), att, U64.zero())),
    U64.or(k, Chess.select_u64(U32.is_eq(kind, 5), att, U64.zero())),
    U64.xor(one, inc), U64.xor(two, carry), U64.xor(four, U64.and(two, carry))}

def count(m: Maps, +sq: U32) -> U32:
  Maps{p, n, b, r, q, k, one, two, four} = m
  U32.add(Bool.to_u32(U64.test_bit(one, U32.to_nat(sq))),
    U32.add(Bool.pick(U32, U64.test_bit(two, U32.to_nat(sq)), 2, 0),
      Bool.pick(U32, U64.test_bit(four, U32.to_nat(sq)), 4, 0)))

def number(n: U32) -> F32:
  F32.from_nat(U32.to_nat(n))

def dy(side: U32) -> U32:
  Bool.pick(U32, U32.is_eq(side, 1), 1, 4294967295)

def flip(side: U32) -> U32:
  Bool.pick(U32, U32.is_zero(side), 56, 0)

def plane(index: U32, turn: U32, mask: U64, out: Array<F32>) -> Array<F32>:
  H.plane(U32.add(112, index), flip(turn), mask, 1.0, out)

def at(index: U32, turn: U32, sq: U32, v: F32, out: Array<F32>) -> Array<F32>:
  Array.set(F32, out, U32.add(U32.mul(U32.add(112, index), 64), U32.xor(sq, flip(turn))), v)

def pieces(kind: U32, +b: Chess.Board, side: U32) -> U64:
  U64.and(H.piece_mask(kind, b), Chess.color(b, side))

def king(+b: Chess.Board, side: U32) -> U32:
  U64.ctz(U64.and(Chess.get_kings(b), Chess.color(b, side)))

def pawn_moves(+b: Chess.Board, +sq: U32, +side: U32, +att: U64) -> U32:
  +occ = Chess.occupied(b)
  +single = U64.bit(U32.to_nat(Tables.jump(sq, (0, dy(side)))))
  +one = Bool.and(Bool.not(U64.is_zero(single)), U64.is_zero(U64.and(single, occ)))
  two = Bool.and(one, Bool.and(U32.is_eq(U32.div(sq, 8), Bool.pick(U32, U32.is_eq(side, 1), 1, 6)),
    U64.is_zero(U64.and(U64.bit(U32.to_nat(Tables.jump(sq, (0, U32.mul(dy(side), 2))))), occ))))
  U32.add(U32.add(Bool.to_u32(one), Bool.to_u32(two)), U32.add(U64.popcount(U64.and(att, Chess.color(b, U32.xor(side, 1)))),
    Bool.to_u32(U64.test_bit(att, U32.to_nat(Chess.get_ep(b))))))

def mobility(kind: U32, +b: Chess.Board, sq: U32, +side: U32, +att: U64) -> U32:
  match kind:
    case 0: pawn_moves(b, sq, side, att)
    case _: U64.popcount(U64.and(att, U64.not(Chess.color(b, side))))

def maximum(kind: U32) -> F32:
  match kind:
    case 0: 4.0
    case 1: 8.0
    case 2: 13.0
    case 3: 14.0
    case 4: 27.0
    case _: 8.0

type Collected is Type:
  Collected{table: Array<U64>, out: Array<F32>, white: Maps, black: Maps}

def collected_side(side: U32, kind: U32, att: U64, table: Array<U64>, out: Array<F32>, white: Maps, black: Maps) -> Collected:
  match side:
    case 1: Collected{table, out, add(kind, att, white), black}
    case _: Collected{table, out, white, add(kind, att, black)}

def collected(+b: Chess.Board, +sq: U32, +kind: U32, +side: U32,
  out: Array<F32>, white: Maps, black: Maps, a: Array<U64> & U64) -> Collected:
  (table, +att) = a
  index = U32.add(24, U32.mod(U32.add(kind, 5), 6))
  out = at(index, Chess.get_turn(b), sq, F32.div(number(mobility(kind, b, sq, side, att)), maximum(kind)), out)
  collected_side(side, kind, att, table, out, white, black)

def collect_square(occupied: Bool, +sq: U32, +b: Chess.Board, c: Collected) -> Collected:
  match occupied:
    case False{}: c
    case True{}:
      Collected{table, out, white, black} = c
      +kind = Chess.piece(b, sq)
      +side = Bool.to_u32(U64.test_bit(Chess.get_white(b), U32.to_nat(sq)))
      collected(b, sq, kind, side, out, white, black, Chess.attack(kind, table, sq, side, Chess.occupied(b)))

def collect(fuel: Nat, +sq: U32, +b: Chess.Board, c: Collected) -> Collected:
  match fuel:
    case 0n: c
    case 1n+p:
      collect(p, U32.inc(sq), b, collect_square(U64.test_bit(Chess.occupied(b), U32.to_nat(sq)), sq, b, c))

# King zone extends one and two ranks towards the opponent on three files.
def front_zone(fuel: Nat, +i: U32, +sq: U32, +side: U32, mask: U64) -> U64:
  match fuel:
    case 0n: mask
    case 1n+p:
      x = U32.sub(U32.mod(i, 3), 1)
      y = U32.mul(dy(side), U32.inc(U32.div(i, 3)))
      front_zone(p, U32.inc(i), sq, side, U64.or(mask, U64.bit(U32.to_nat(Tables.jump(sq, (x, y))))))

def zone(+sq: U32, side: U32) -> U64:
  front_zone(6n, 0, sq, side, U64.or(U64.bit(U32.to_nat(sq)), Tables.leaps(8n, sq, 0, True{}, U64.zero())))

def safety(+base: U32, +turn: U32, +z: U64, m: Maps, out: Array<F32>) -> Array<F32>:
  Maps{p, n, b, r, q, k, one, two, four} = m
  out = plane(base, turn, z, out)
  out = plane(U32.inc(base), turn, U64.and(z, n), out)
  out = plane(U32.add(base, 2), turn, U64.and(z, b), out)
  out = plane(U32.add(base, 3), turn, U64.and(z, r), out)
  plane(U32.add(base, 4), turn, U64.and(z, q), out)

# Complete pin lines include both directions through the friendly king.
def nearest(+mask: U64, dir: U32) -> U32:
  Bool.pick(U32, U64.is_zero(mask), 64, Bool.pick(U32, U32.is_zero(U32.mod(dir, 2)), U64.ctz(mask), U32.sub(63, U64.clz(mask))))

def pin_dir(+b: Chess.Board, +side: U32, +sq: U32, +dir: U32, r: U64 & U64) -> U64 & U64:
  (pinned, rays) = r
  +ray = Tables.ray(7n, False{}, sq, dir, U64.zero(), False{}, U64.zero())
  +hits = U64.and(ray, Chess.occupied(b))
  +first = U64.bit(U32.to_nat(nearest(hits, dir)))
  next = U64.bit(U32.to_nat(nearest(U64.and(hits, U64.not(first)), dir)))
  sliders = U64.or(Chess.get_queens(b), Chess.select_u64(U32.is_lt(dir, 4), Chess.get_rooks(b), Chess.get_bishops(b)))
  +ok = Bool.and(Bool.not(U64.is_zero(U64.and(first, Chess.color(b, side)))),
    Bool.not(U64.is_zero(U64.and(next, U64.and(sliders, Chess.color(b, U32.xor(side, 1)))))))
  opposite = Bool.pick(U32, U32.is_lt(dir, 4), U32.xor(dir, 1), U32.sub(11, dir))
  line = U64.or(U64.bit(U32.to_nat(sq)), U64.or(ray, Tables.ray(7n, False{}, sq, opposite, U64.zero(), False{}, U64.zero())))
  (U64.or(pinned, Chess.select_u64(ok, first, U64.zero())), U64.or(rays, Chess.select_u64(ok, line, U64.zero())))

def pins(fuel: Nat, +dir: U32, +b: Chess.Board, +side: U32, +sq: U32, r: U64 & U64) -> U64 & U64:
  match fuel:
    case 0n: r
    case 1n+p: pins(p, U32.inc(dir), b, side, sq, pin_dir(b, side, sq, dir, r))

def remove(b: Chess.Board, sq: U32) -> Chess.Board:
  Chess.Board{p, n, bi, r, q, k, w, bl, turn, rights, ep} = b
  +keep = U64.not(U64.bit(U32.to_nat(sq)))
  Chess.Board{U64.and(p, keep), U64.and(n, keep), U64.and(bi, keep), U64.and(r, keep),
    U64.and(q, keep), U64.and(k, keep), U64.and(w, keep), U64.and(bl, keep), turn, rights, ep}

# This definition intentionally also handles already-checked enemy kings:
# removal of any own piece which leaves a check is the training signal.
def discovered_result(sq: U32, mask: U64, r: Array<U64> & Bool) -> Array<U64> & U64:
  (table, ok) = r
  (table, U64.or(mask, Chess.select_u64(ok, U64.bit(U32.to_nat(sq)), U64.zero())))

def discovered_square(own: Bool, +sq: U32, +b: Chess.Board, side: U32, king_sq: U32,
  r: Array<U64> & U64) -> Array<U64> & U64:
  match own:
    case False{}: r
    case True{}:
      (table, mask) = r
      discovered_result(sq, mask, Chess.attacked(table, remove(b, sq), king_sq, side))

def discovered(fuel: Nat, +sq: U32, +b: Chess.Board, +side: U32, +king_sq: U32,
  r: Array<U64> & U64) -> Array<U64> & U64:
  match fuel:
    case 0n: r
    case 1n+p:
      discovered(p, U32.inc(sq), b, side, king_sq,
        discovered_square(U64.test_bit(Chess.color(b, side), U32.to_nat(sq)), sq, b, side, king_sq, r))

def pin_output(+base: U32, +turn: U32, out: Array<F32>, masks: U64 & U64,
  result: Array<U64> & U64) -> H.Encoded:
  (pinned, rays) = masks
  (table, disc) = result
  out = plane(base, turn, pinned, out)
  out = plane(U32.inc(base), turn, rays, out)
  H.Encoded{table, plane(U32.add(base, 2), turn, disc, out)}

def pin_planes(base: U32, +b: Chess.Board, +side: U32, e: H.Encoded) -> H.Encoded:
  H.Encoded{table, out} = e
  pin_output(base, Chess.get_turn(b), out, pins(8n, 0, b, side, king(b, side), (U64.zero(), U64.zero())),
    discovered(64n, 0, b, side, king(b, U32.xor(side, 1)), (table, U64.zero())))

def file_valid(ok: Bool, f: U32) -> U64:
  match ok:
    case False{}: U64.zero()
    case True{}: U64.shln(U64.from_parts(16843009, 16843009), U32.to_nat(f))

def file_mask(+f: U32) -> U64:
  file_valid(U32.is_lt(f, 8), f)

def adjacent(+f: U32) -> U64:
  U64.or(file_mask(U32.sub(f, 1)), file_mask(U32.inc(f)))

def ahead(+rank: U32, +side: U32, inclusive: Bool) -> U64:
  extra = Bool.pick(U32, inclusive, 0, 1)
  +n = U32.mul(8, U32.add(Bool.pick(U32, U32.is_eq(side, 1), rank, U32.sub(7, rank)), extra))
  Chess.select_u64(U32.is_eq(side, 1), U64.shln(U64.not(U64.zero()), U32.to_nat(n)), U64.shrn(U64.not(U64.zero()), U32.to_nat(n)))

type Pawns is Data:
  Pawns{passed: U64, isolated: U64, backward: U64, connected: U64, space: U64, tension: U64}

def pawn_square(own: Bool, +sq: U32, +side: U32, +ours: U64, +theirs: U64, ps: Pawns) -> Pawns:
  match own:
    case False{}: ps
    case True{}:
      Pawns{passed, isolated, backward, connected, space, tension} = ps
      +f = U32.mod(sq, 8)
      +r = U32.div(sq, 8)
      +adj = adjacent(f)
      +bit = U64.bit(U32.to_nat(sq))
      +front = Tables.jump(sq, (0, dy(side)))
      +isol = U64.is_zero(U64.and(adj, ours))
      pass = U64.is_zero(U64.and(U64.and(U64.or(adj, file_mask(f)), ahead(r, side, False{})), theirs))
      back = Bool.and(Bool.not(isol), Bool.and(U32.is_lt(front, 64),
        Bool.and(Bool.not(U64.is_zero(U64.and(Tables.pawns(front, dy(side)), theirs))),
          U64.is_zero(U64.and(U64.and(adj, ahead(r, side, True{})), ours)))))
      +rank_mask = U64.shln(U64.from_u32(255), U32.to_nat(U32.mul(r, 8)))
      conn = Bool.not(U64.is_zero(U64.and(ours, U64.and(adj, U64.or(rank_mask,
        U64.or(U64.shln(rank_mask, 8n), U64.shrn(rank_mask, 8n)))))))
      +behind = U32.sub(0, dy(side))
      space_bits = U64.or(U64.bit(U32.to_nat(Tables.jump(sq, (0, behind)))),
        U64.bit(U32.to_nat(Tables.jump(sq, (0, U32.mul(behind, 2))))))
      tens = Bool.not(U64.is_zero(U64.and(Tables.pawns(sq, dy(side)), theirs)))
      Pawns{U64.or(passed, Chess.select_u64(pass, bit, U64.zero())),
        U64.or(isolated, Chess.select_u64(isol, bit, U64.zero())),
        U64.or(backward, Chess.select_u64(back, bit, U64.zero())),
        U64.or(connected, Chess.select_u64(conn, bit, U64.zero())),
        U64.or(space, Chess.select_u64(Bool.and(U32.is_ge(f, 2), U32.is_le(f, 5)), space_bits, U64.zero())),
        U64.or(tension, Chess.select_u64(tens, bit, U64.zero()))}

def pawns(fuel: Nat, +sq: U32, +side: U32, +ours: U64, +theirs: U64, ps: Pawns) -> Pawns:
  match fuel:
    case 0n: ps
    case 1n+p: pawns(p, U32.inc(sq), side, ours, theirs,
      pawn_square(U64.test_bit(ours, U32.to_nat(sq)), sq, side, ours, theirs, ps))

def pawn_facts(+b: Chess.Board, +side: U32) -> Pawns:
  pawns(64n, 0, side, pieces(0, b, side), pieces(0, b, U32.xor(side, 1)),
    Pawns{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero()})

def pawn_planes(+base: U32, +outpost: U32, +turn: U32, ps: Pawns, own_att: U64, enemy_att: U64, out: Array<F32>) -> Array<F32>:
  Pawns{passed, isolated, backward, connected, space, tension} = ps
  out = plane(base, turn, passed, out)
  out = plane(U32.inc(base), turn, isolated, out)
  out = plane(U32.add(base, 2), turn, backward, out)
  out = plane(U32.add(base, 3), turn, connected, out)
  out = plane(outpost, turn, U64.and(own_att, U64.not(enemy_att)), out)
  plane(U32.inc(outpost), turn, space, out)

def attack_planes(fuel: Nat, +kind: U32, +base: U32, +turn: U32, +m: Maps, out: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: out
    case 1n+p: attack_planes(p, U32.inc(kind), base, turn, m, plane(U32.add(base, kind), turn, typed(kind, m), out))

def cheaper(+b: Chess.Board, side: U32, m: Maps) -> U64:
  Maps{p, n, bi, r, q, k, one, two, four} = m
  +rooks = Chess.get_rooks(b)
  +queens = Chess.get_queens(b)
  U64.and(Chess.color(b, side), U64.or(U64.and(p, U64.or(U64.or(Chess.get_knights(b), Chess.get_bishops(b)), U64.or(rooks, queens))),
    U64.or(U64.and(U64.or(n, bi), U64.or(rooks, queens)), U64.and(r, queens))))

def distance(+a: U32, +b: U32) -> U32:
  U32.sub(U32.max(a, b), U32.min(a, b))

def storm(active: Bool, sq: U32, king_sq: U32) -> F32:
  match active:
    case False{}: 0.0
    case True{}: F32.sub(1.0, F32.div(number(distance(U32.div(sq, 8), U32.div(king_sq, 8))), 7.0))

def graded(fuel: Nat, +sq: U32, +b: Chess.Board, +us: Maps, +them: Maps, out: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: out
    case 1n+p:
      +turn = Chess.get_turn(b)
      +ku = king(b, turn)
      +kt = king(b, U32.xor(turn, 1))
      +cu = count(us, sq)
      +ct = count(them, sq)
      out = at(48, turn, sq, F32.div(number(U32.min(cu, 4)), 4.0), out)
      out = at(49, turn, sq, F32.div(number(U32.min(ct, 4)), 4.0), out)
      out = at(58, turn, sq, F32.max(F32.sub(0.0, 1.0), F32.min(1.0, F32.div(F32.sub(number(cu), number(ct)), 4.0))), out)
      out = at(61, turn, sq, storm(Bool.and(U64.test_bit(pieces(0, b, U32.xor(turn, 1)), U32.to_nat(sq)),
        U32.is_le(distance(U32.mod(sq, 8), U32.mod(ku, 8)), 1)), sq, ku), out)
      out = at(62, turn, sq, storm(Bool.and(U64.test_bit(pieces(0, b, turn), U32.to_nat(sq)),
        U32.is_le(distance(U32.mod(sq, 8), U32.mod(kt, 8)), 1)), sq, kt), out)
      graded(p, U32.inc(sq), b, us, them, out)

def safe_after(kind: U32, turn: U32, safe: U64, out: Array<F32>, r: Array<U64> & U64) -> H.Encoded:
  (table, att) = r
  H.Encoded{table, plane(U32.add(53, kind), turn, U64.and(att, safe), out)}

def safe_checks(fuel: Nat, +kind: U32, +turn: U32, +king_sq: U32, +occ: U64, +safe: U64, e: H.Encoded) -> H.Encoded:
  match fuel:
    case 0n: e
    case 1n+p:
      H.Encoded{table, out} = e
      safe_checks(p, U32.inc(kind), turn, king_sq, occ, safe,
        safe_after(kind, turn, safe, out, Chess.attack(kind, table, king_sq, turn, occ)))

def tension(ps: Pawns) -> U64:
  Pawns{passed, isolated, backward, connected, space, result} = ps
  result

def threats(enabled: Bool, +b: Chess.Board, +us: Maps, +them: Maps, pu: Pawns, pt: Pawns, e: H.Encoded) -> H.Encoded:
  match enabled:
    case False{}: e
    case True{}:
      H.Encoded{table, out} = e
      +turn = Chess.get_turn(b)
      +au = union(us)
      +athem = union(them)
      +ours = Chess.color(b, turn)
      +theirs = Chess.color(b, U32.xor(turn, 1))
      out = plane(34, turn, au, out)
      out = plane(35, turn, athem, out)
      out = attack_planes(6n, 0, 36, turn, us, out)
      out = attack_planes(6n, 0, 42, turn, them, out)
      out = plane(50, turn, U64.and(ours, U64.and(athem, U64.not(au))), out)
      out = plane(51, turn, U64.and(theirs, U64.and(au, U64.not(athem))), out)
      out = plane(52, turn, cheaper(b, turn, them), out)
      out = plane(53, turn, cheaper(b, U32.xor(turn, 1), us), out)
      out = plane(59, turn, tension(pu), out)
      out = plane(60, turn, tension(pt), out)
      out = graded(64n, 0, b, us, them, out)
      safe_checks(4n, 1, turn, king(b, U32.xor(turn, 1)), Chess.occupied(b),
        U64.not(U64.or(athem, ours)), H.Encoded{table, out})

def oriented(enabled: Bool, +b: Chess.Board, +us: Maps, +them: Maps, table: Array<U64>, out: Array<F32>) -> H.Encoded:
  +turn = Chess.get_turn(b)
  +other = U32.xor(turn, 1)
  out = safety(0, turn, zone(king(b, turn), turn), them, out)
  out = safety(5, turn, zone(king(b, other), other), us, out)
  +pu = pawn_facts(b, turn)
  +pt = pawn_facts(b, other)
  out = pawn_planes(16, 30, turn, pu, typed(0, us), typed(0, them), out)
  out = pawn_planes(20, 32, turn, pt, typed(0, them), typed(0, us), out)
  e = pin_planes(13, b, other, pin_planes(10, b, turn, H.Encoded{table, out}))
  threats(enabled, b, us, them, pu, pt, e)

def finish(turn: U32, enabled: Bool, b: Chess.Board, table: Array<U64>, out: Array<F32>, white: Maps, black: Maps) -> H.Encoded:
  match turn:
    case 1: oriented(enabled, b, white, black, table, out)
    case _: oriented(enabled, b, black, white, table, out)

def collected_input(enabled: Bool, +b: Chess.Board, c: Collected) -> H.Encoded:
  Collected{table, out, white, black} = c
  finish(Chess.get_turn(b), enabled, b, table, out, white, black)

def encode(enabled: Bool, +b: Chess.Board, table: Array<U64>, out: Array<F32>) -> H.Encoded:
  collected_input(enabled, b, collect(64n, 0, b, Collected{table, out, empty(), empty()}))
