# Orthodox-chess rules in Bend. C is used only to initialize attack tables/input.
# Copy/make + king-safety filtering intentionally favors an auditable first core.
import Base
import ../bitboard_probe/Sliders.bend as Sliders

type Board is Data:
  Board{pawns: U64, knights: U64, bishops: U64, rooks: U64, queens: U64,
    kings: U64, white: U64, black: U64, turn: U32, rights: U32, ep: U32}

type Ply is Data:
  Ply{src: U32, dst: U32, promotion: U32, flag: U32}

# Squares: a1=0. Turn: white=1. Rights: K=1,Q=2,k=4,q=8. No ep: 64.
# Piece ids: pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5.
# Ply promotion: 0=none, otherwise piece id. Flag: 0=ordinary,1=ep,2=castle.

def get_pawns(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  pawns

def get_knights(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  knights

def get_bishops(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  bishops

def get_rooks(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  rooks

def get_queens(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  queens

def get_kings(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  kings

def get_white(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  white

def get_black(b: Board) -> U64:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  black

def get_turn(b: Board) -> U32:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  turn

def get_rights(b: Board) -> U32:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  rights

def get_ep(b: Board) -> U32:
  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b
  ep

def color(+b: Board, turn: U32) -> U64:
  Bool.pick(U64, U32.is_eq(turn, 1), get_white(b), get_black(b))

def occupied(+b: Board) -> U64:
  U64.or(get_white(b), get_black(b))

def piece(+b: Board, +sq: U32) -> U32:
  Bool.pick(U32, U64.test_bit(get_pawns(b), U32.to_nat(sq)), 0,
    Bool.pick(U32, U64.test_bit(get_knights(b), U32.to_nat(sq)), 1,
      Bool.pick(U32, U64.test_bit(get_bishops(b), U32.to_nat(sq)), 2,
        Bool.pick(U32, U64.test_bit(get_rooks(b), U32.to_nat(sq)), 3,
          Bool.pick(U32, U64.test_bit(get_queens(b), U32.to_nat(sq)), 4, 5)))))

# All callers supply sq<64. Metadata/table sizes are checked on input.
# Slots 0..127: rook/bishop masks; 128..255: table offsets;
# 256..383: knight/king; 384..511: white/black pawn attacks.
def slide_offset(index: U32, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, offset) = r
  Sliders.lookup(table, U64.low(offset), index)

def slide_mask(sq: U32, occ: U64, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, mask) = r
  slide_offset(Sliders.pext_index(occ, mask), Array.get(U64, table, U32.add(128, sq)))

def slide(table: Array<U64>, +sq: U32, occ: U64) -> Array<U64> & U64:
  slide_mask(sq, occ, Array.get(U64, table, sq))

def or_result(a: U64, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, b) = r
  (table, U64.or(a, b))

def queen_after(sq: U32, occ: U64, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, rook) = r
  or_result(rook, slide(table, U32.add(64, sq), occ))

def attack(kind: U32, table: Array<U64>, +sq: U32, side: U32, +occ: U64) -> Array<U64> & U64:
  match kind:
    case 0:
      Array.get(U64, table, U32.add(sq, Bool.pick(U32, U32.is_eq(side, 1), 384, 448)))
    case 1:
      Array.get(U64, table, U32.add(256, sq))
    case 2:
      slide(table, U32.add(64, sq), occ)
    case 3:
      slide(table, sq, occ)
    case 4:
      queen_after(sq, occ, slide(table, sq, occ))
    case _:
      Array.get(U64, table, U32.add(320, sq))

# Reverse attack lookup must include pinned enemy pieces as attacks too.
def attacked_bishop(b: Board, by: U32, hits: U64, r: Array<U64> & U64) -> Array<U64> & Bool:
  (table, ba) = r
  +b = b
  hits = U64.or(hits, U64.and(ba, U64.or(get_bishops(b), get_queens(b))))
  (table, Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))

def attacked_rook(+b: Board, +sq: U32, +by: U32, hits: U64, r: Array<U64> & U64) -> Array<U64> & Bool:
  (table, ra) = r
  attacked_bishop(b, by, U64.or(hits, U64.and(ra, U64.or(get_rooks(b), get_queens(b)))),
    attack(2, table, sq, by, occupied(b)))

def attacked_king(+b: Board, +sq: U32, +by: U32, hits: U64, r: Array<U64> & U64) -> Array<U64> & Bool:
  (table, ka) = r
  attacked_rook(b, sq, by, U64.or(hits, U64.and(ka, get_kings(b))), attack(3, table, sq, by, occupied(b)))

def attacked_knight(+b: Board, +sq: U32, +by: U32, hits: U64, r: Array<U64> & U64) -> Array<U64> & Bool:
  (table, na) = r
  attacked_king(b, sq, by, U64.or(hits, U64.and(na, get_knights(b))), attack(5, table, sq, by, occupied(b)))

def attacked_pawn(+b: Board, +sq: U32, +by: U32, r: Array<U64> & U64) -> Array<U64> & Bool:
  (table, pa) = r
  attacked_knight(b, sq, by, U64.and(pa, get_pawns(b)), attack(1, table, sq, by, occupied(b)))

def attacked(table: Array<U64>, +b: Board, +sq: U32, +by: U32) -> Array<U64> & Bool:
  attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))

def in_check(table: Array<U64>, +b: Board, +side: U32) -> Array<U64> & Bool:
  attacked(table, b, U64.ctz(U64.and(get_kings(b), color(b, side))), U32.xor(side, 1))

# Only the original corner rook can preserve each castling right.
def corner_right(sq: U32) -> U32:
  match sq:
    case 0: 2
    case 7: 1
    case 56: 8
    case 63: 4
    case _: 0

def update_piece(bb: U64, remove: U64, target: U64, put: Bool) -> U64:
  U64.or(U64.and_not(bb, remove), Bool.pick(U64, put, target, U64.zero()))

# PRE: m was produced by legal_moves(b). No arbitrary-move application API.
def make_move(+b: Board, m: Ply) -> Board:
  Ply{+src, +dst, +promotion, +flag} = m
  +kind = piece(b, src)
  +castle = U32.is_eq(flag, 2)
  +side = get_turn(b)
  +target = U64.bit(U32.to_nat(dst))
  cap = Bool.pick(U32, U32.is_eq(flag, 1), U32.xor(dst, 8), dst)
  rook_from = Bool.pick(U32, U32.is_eq(U32.and(dst, 7), 6), U32.inc(dst), U32.sub(dst, 2))
  +rook_to = Bool.pick(U64, castle, U64.bit(U32.to_nat(U32.div(U32.add(src, dst), 2))), U64.zero())
  +remove = U64.or(U64.or(U64.bit(U32.to_nat(src)), U64.bit(U32.to_nat(cap))),
    U64.or(target, Bool.pick(U64, castle, U64.bit(U32.to_nat(rook_from)), U64.zero())))
  +put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)
  +added = U64.or(target, rook_to)
  +w = U32.is_eq(side, 1)
  rights_lost = U32.or(U32.or(corner_right(src), corner_right(dst)),
    Bool.pick(U32, U32.is_eq(kind, 5), Bool.pick(U32, w, 3, 12), 0))
  ep = Bool.pick(U32, Bool.and(U32.is_zero(kind), U32.is_eq(U32.xor(src, dst), 16)),
    U32.div(U32.add(src, dst), 2), 64)
  Board{
    update_piece(get_pawns(b), remove, target, U32.is_eq(put, 0)),
    update_piece(get_knights(b), remove, target, U32.is_eq(put, 1)),
    update_piece(get_bishops(b), remove, target, U32.is_eq(put, 2)),
    U64.or(update_piece(get_rooks(b), remove, target, U32.is_eq(put, 3)), rook_to),
    update_piece(get_queens(b), remove, target, U32.is_eq(put, 4)),
    update_piece(get_kings(b), remove, target, U32.is_eq(put, 5)),
    update_piece(get_white(b), remove, added, w),
    update_piece(get_black(b), remove, added, Bool.not(w)),
    U32.xor(side, 1), U32.and(get_rights(b), U32.not(rights_lost)), ep}

def put_move(promote: Bool, +src: U32, +dst: U32, flag: U32, tail: List<&2, Ply>) -> List<&2, Ply>:
  match promote:
    case True{}:
      Con{Ply{src, dst, 1, 0}, Con{Ply{src, dst, 2, 0},
        Con{Ply{src, dst, 3, 0}, Con{Ply{src, dst, 4, 0}, tail}}}}
    case False{}:
      Con{Ply{src, dst, 0, flag}, tail}

def destinations(n: Nat, empty: Bool, +bb: U64, +src: U32, +pawn: Bool,
  +ep_sq: U32, tail: List<&2, Ply>) -> List<&2, Ply>:
  match n:
    case 0n: tail
    case 1n+p:
      match empty:
        case True{}: tail
        case False{}:
          +dst = U64.ctz(bb)
          +rank = U32.div(dst, 8)
          +rest = U64.clear_lsb(bb)
          destinations(p, U64.is_zero(rest), rest, src, pawn, ep_sq,
            put_move(Bool.and(pawn, Bool.or(U32.is_zero(rank), U32.is_eq(rank, 7))), src, dst,
              Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, ep_sq))), tail))

# Input ep metadata may name an unavailable capture, so check rank, victim
# and an empty target before even proposing it. King safety filters x-rays.
def ep_target(+b: Board) -> U64:
  +target = get_ep(b)
  +w = U32.is_eq(get_turn(b), 1)
  expected_rank = Bool.pick(U32, w, 5, 2)
  +victim = U32.xor(target, 8)
  ok = Bool.and(U32.is_lt(target, 64), Bool.and(U32.is_eq(U32.div(target, 8), expected_rank),
    Bool.and(Bool.not(U64.test_bit(occupied(b), U32.to_nat(target))),
      U64.test_bit(U64.and(get_pawns(b), color(b, U32.xor(get_turn(b), 1))), U32.to_nat(victim)))))
  Bool.pick(U64, ok, U64.bit(U32.to_nat(target)), U64.zero())

def pawn_after(advances: U64, capturable: U64, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, pa) = r
  (table, U64.or(advances, U64.and(pa, capturable)))

def pawn_targets(table: Array<U64>, +b: Board, +src: U32) -> Array<U64> & U64:
  +w = U32.is_eq(get_turn(b), 1)
  +step = Bool.pick(U32, w, 8, 4294967288)
  +first = U32.add(src, step)
  +second = U32.add(first, step)
  +occ = occupied(b)
  +empty1 = Bool.and(U32.is_lt(first, 64), Bool.not(U64.test_bit(occ, U32.to_nat(first))))
  start_rank = Bool.pick(U32, w, 1, 6)
  empty2 = Bool.and(empty1, Bool.and(U32.is_eq(U32.div(src, 8), start_rank),
    Bool.not(U64.test_bit(occ, U32.to_nat(second)))))
  advances = U64.or(Bool.pick(U64, empty1, U64.bit(U32.to_nat(first)), U64.zero()),
    Bool.pick(U64, empty2, U64.bit(U32.to_nat(second)), U64.zero()))
  enemies = U64.and_not(color(b, U32.xor(get_turn(b), 1)), get_kings(b))
  pawn_after(advances, U64.or(enemies, ep_target(b)), attack(0, table, src, get_turn(b), occ))

def exclude_targets(exclude: U64, r: Array<U64> & U64) -> Array<U64> & U64:
  (table, targets) = r
  (table, U64.and_not(targets, exclude))

def piece_targets(pawn: Bool, table: Array<U64>, +b: Board, +src: U32) -> Array<U64> & U64:
  match pawn:
    case True{}:
      pawn_targets(table, b, src)
    case False{}:
      exclude_targets(U64.or(color(b, get_turn(b)), get_kings(b)),
        attack(piece(b, src), table, src, get_turn(b), occupied(b)))

def bit_squares(n: Nat, empty: Bool, +bb: U64, acc: List<&2, U32>) -> List<&2, U32>:
  match n:
    case 0n: acc
    case 1n+p:
      match empty:
        case True{}: acc
        case False{}:
          +rest = U64.clear_lsb(bb)
          bit_squares(p, U64.is_zero(rest), rest, Con{U64.ctz(bb), acc})

def scan_after(src: U32, pawn: Bool, ep_sq: U32, acc: List<&2, Ply>, r: Array<U64> & U64) -> Array<U64> & List<&2, Ply>:
  (table, +targets) = r
  (table, destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc))

def scan_step(+b: Board, +src: U32, r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  (table, acc) = r
  +pawn = U64.test_bit(get_pawns(b), U32.to_nat(src))
  scan_after(src, pawn, get_ep(b), acc, piece_targets(pawn, table, b, src))

def scan(xs: List<&2, U32>, +b: Board, r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  match xs:
    case Nil{}: r
    case Con{src, rest}: scan(rest, b, scan_step(b, src, r))

# Transit safety is checked with the old king square vacated.
def castle_finish(m: Ply, start_check: Bool, +acc: List<&2, Ply>, r: Array<U64> & Bool) -> Array<U64> & List<&2, Ply>:
  (table, transit_check) = r
  (table, Bool.pick(List<&2, Ply>, Bool.or(start_check, transit_check), acc, Con{m, acc}))

def castle_start(+b: Board, +src: U32, dst: U32, transit: U32,
  acc: List<&2, Ply>, r: Array<U64> & Bool) -> Array<U64> & List<&2, Ply>:
  (table, start_check) = r
  castle_finish(Ply{src, dst, 0, 2}, start_check, acc,
    in_check(table, make_move(b, Ply{src, transit, 0, 0}), get_turn(b)))

def castle_checked(ok: Bool, table: Array<U64>, +b: Board, src: U32,
  dst: U32, transit: U32, acc: List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  match ok:
    case False{}: (table, acc)
    case True{}:
      castle_start(b, src, dst, transit, acc, in_check(table, b, get_turn(b)))

def castle_side(+b: Board, +king_side: Bool,
  r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  (table, acc) = r
  +home = Bool.pick(U32, U32.is_eq(get_turn(b), 1), 0, 56)
  +src = U32.add(home, 4)
  rook = U32.add(home, Bool.pick(U32, king_side, 7, 0))
  dst = U32.add(home, Bool.pick(U32, king_side, 6, 2))
  transit = U32.add(home, Bool.pick(U32, king_side, 5, 3))
  right = U32.mul(Bool.pick(U32, U32.is_eq(get_turn(b), 1), 1, 4), Bool.pick(U32, king_side, 1, 2))
  between = U64.shln(U64.from_u32(Bool.pick(U32, king_side, 96, 14)), U32.to_nat(home))
  +own = color(b, get_turn(b))
  ok = Bool.and(Bool.not(U32.is_zero(U32.and(get_rights(b), right))),
    Bool.and(U64.test_bit(U64.and(get_kings(b), own), U32.to_nat(src)),
      Bool.and(U64.test_bit(U64.and(get_rooks(b), own), U32.to_nat(rook)), U64.is_zero(U64.and(occupied(b), between)))))
  castle_checked(ok, table, b, src, dst, transit, acc)

def filter_after(m: Ply, +acc: List<&2, Ply>, r: Array<U64> & Bool) -> Array<U64> & List<&2, Ply>:
  (table, check) = r
  (table, Bool.pick(List<&2, Ply>, check, acc, Con{m, acc}))

def filter_step(+b: Board, +m: Ply, r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  (table, acc) = r
  filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))

def filter_legal(xs: List<&2, Ply>, +b: Board, r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  match xs:
    case Nil{}: r
    case Con{m, rest}: filter_legal(rest, b, filter_step(b, m, r))

def filter_start(+b: Board, r: Array<U64> & List<&2, Ply>) -> Array<U64> & List<&2, Ply>:
  (table, moves) = r
  filter_legal(moves, b, (table, Nil{}))

def legal_moves(table: Array<U64>, +b: Board) -> Array<U64> & List<&2, Ply>:
  +own = color(b, get_turn(b))
  filter_start(b, castle_side(b, False{}, castle_side(b, True{},
    scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Nil{})))))

def children(ms: List<&2, Ply>, +b: Board) -> List<&2, Board>:
  match ms:
    case Nil{}: Nil{}
    case Con{m, rest}: Con{make_move(b, m), children(rest, b)}

# Depth is the first decreasing argument; siblings decrease the second one.
# Draw adjudication is intentionally absent from standard perft.
# Bulk-count the last ply; do not construct boards that perft will not visit.
def with_moves(leaf: Bool, f: Array<U64> -> List<&2, Ply> -> (Array<U64> & U64), r: Array<U64> & List<&2, Ply>) -> Array<U64> & U64:
  match leaf:
    case True{}:
      (table, moves) = r
      (table, U64.from_u32(U32.from_nat(List.length(&2, Ply, moves))))
    case False{}:
      (table, moves) = r
      f(table, moves)

def with_count(f: Array<U64> -> U64 -> (Array<U64> & U64), r: Array<U64> & U64) -> Array<U64> & U64:
  (table, count) = r
  f(table, count)

def perft(+depth: Nat, boards: List<&2, Board>, table: Array<U64>, total: U64) -> Array<U64> & U64:
  match depth:
    case 0n:
      (table, U64.add(total, U64.from_u32(U32.from_nat(List.length(&2, Board, boards)))))
    case 1n+p:
      match boards:
        case Nil{}: (table, total)
        case Con{+b, rest}:
          with_count(t => value => perft(depth, rest, t, U64.add(total, value)),
            with_moves(Nat.is_eq(p, 0n), t => moves => perft(p, children(moves, b), t, U64.zero()),
              legal_moves(table, b)))
