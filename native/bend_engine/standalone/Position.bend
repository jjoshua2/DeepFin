import Base
import ../legal_probe/Chess.bend as Chess
import ./Text.bend as Text

# Newest prior position first. No history is delegated to a foreign helper.
type Past is Data:
  Past{board: Chess.Board, halfmove: U32, fullmove: U32, move: Chess.Ply}

type Game is Data:
  Game{board: Chess.Board, halfmove: U32, fullmove: U32, history: List<&2, Past>}

def empty() -> Chess.Board:
  Chess.Board{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), 1, 0, 64}

def board(g: Game) -> Chess.Board:
  Game{b, half, full, hist} = g
  b

def kind(c: U32) -> U32:
  match c:
    case 112: 0
    case 110: 1
    case 98: 2
    case 114: 3
    case 113: 4
    case 107: 5
    case _: 6

def put(+kind: U32, +white: Bool, sq: U32, b: Chess.Board) -> Chess.Board:
  Chess.Board{p, n, bi, r, q, k, w, bl, turn, rights, ep} = b
  +bit = U64.bit(U32.to_nat(sq))
  Chess.Board{U64.or(p, Chess.select_u64(U32.is_eq(kind, 0), bit, U64.zero())),
    U64.or(n, Chess.select_u64(U32.is_eq(kind, 1), bit, U64.zero())),
    U64.or(bi, Chess.select_u64(U32.is_eq(kind, 2), bit, U64.zero())),
    U64.or(r, Chess.select_u64(U32.is_eq(kind, 3), bit, U64.zero())),
    U64.or(q, Chess.select_u64(U32.is_eq(kind, 4), bit, U64.zero())),
    U64.or(k, Chess.select_u64(U32.is_eq(kind, 5), bit, U64.zero())),
    U64.or(w, Chess.select_u64(white, bit, U64.zero())),
    U64.or(bl, Chess.select_u64(white, U64.zero(), bit)), turn, rights, ep}

type Layout is Data:
  Layout{b: Chess.Board, file: U32, rank: U32, valid: Bool}

def digit(is_digit: Bool, +c: U32, l: Layout) -> Layout:
  match is_digit:
    case True{}:
      Layout{b, file, rank, valid} = l
      +next = U32.add(file, U32.sub(c, 48))
      Layout{b, next, rank, Bool.and(valid, U32.is_le(next, 8))}
    case False{}:
      Layout{b, +file, +rank, valid} = l
      +p = kind(U32.or(c, 32))
      Layout{put(p, U32.is_lt(c, 97), U32.add(file, U32.mul(rank, 8)), b), U32.inc(file), rank,
        Bool.and(valid, Bool.and(U32.is_lt(p, 6), U32.is_lt(file, 8)))}

def layout_char(+c: U32, l: Layout) -> Layout:
  match c:
    case 47:
      Layout{b, file, +rank, valid} = l
      Layout{b, 0, U32.sub(rank, 1), Bool.and(valid, Bool.and(U32.is_eq(file, 8), U32.is_gt(rank, 0)))}
    case _:
      digit(Bool.and(U32.is_ge(c, 49), U32.is_le(c, 56)), c, l)

def layout(s: String, l: Layout) -> Layout:
  match s:
    case SNil{}: l
    case SCon{Chr{c}, rest}: layout(rest, layout_char(c, l))

def board_when(ok: Bool, b: Chess.Board) -> Maybe<&2, Chess.Board>:
  match ok:
    case True{}: Some{b}
    case False{}: None{}

def layout_result(l: Layout) -> Maybe<&2, Chess.Board>:
  Layout{b, file, rank, valid} = l
  ok = Bool.and(valid, Bool.and(U32.is_eq(file, 8), U32.is_zero(rank)))
  board_when(ok, b)

def right(c: U32) -> U32:
  match c:
    case 75: 1
    case 81: 2
    case 107: 4
    case 113: 8
    case _: 0

def rights_go(s: String, +acc: U32, valid: Bool) -> Maybe<&2, U32>:
  match s:
    case SNil{}: Text.only(valid, acc)
    case SCon{Chr{c}, rest}:
      +bit = right(c)
      rights_go(rest, U32.or(acc, bit), Bool.and(valid, Bool.and(U32.is_gt(bit, 0), U32.is_zero(U32.and(acc, bit)))))

def rights_dash(dash: Bool, +s: String) -> Maybe<&2, U32>:
  match dash:
    case True{}: Some{0}
    case False{}: rights_go(s, 0, Bool.not(String.is_empty(s)))

def rights(+s: String) -> Maybe<&2, U32>:
  rights_dash(String.eq(s, "-"), s)

def ep_dash(dash: Bool, s: String) -> Maybe<&2, U32>:
  match dash:
    case True{}: Some{64}
    case False{}: Text.square(s)

def ep(+s: String) -> Maybe<&2, U32>:
  ep_dash(String.eq(s, "-"), s)

def side(+s: String) -> Maybe<&2, U32>:
  Text.only(Bool.or(String.eq(s, "w"), String.eq(s, "b")), Bool.to_u32(String.eq(s, "w")))

def metadata(b: Chess.Board, turn: U32, rights: U32, ep: U32) -> Chess.Board:
  Chess.Board{p, n, bi, r, q, k, w, bl, oldturn, oldrights, oldep} = b
  Chess.Board{p, n, bi, r, q, k, w, bl, turn, rights, ep}

def fen(pieces: String, turn: String, castle: String, target: String, half: String, full: String) -> Maybe<&2, Game>:
  do Maybe<&2, Game>:
    b : Chess.Board <- layout_result(layout(pieces, Layout{empty(), 0, 7, True{}}))
    t : U32 <- side(turn)
    r : U32 <- rights(castle)
    e : U32 <- ep(target)
    h : U32 <- Text.number(half)
    h : U32 <- Text.bounded(h)
    f : U32 <- Text.number(full)
    f : U32 <- Text.positive(f)
    f : U32 <- Text.bounded(f)
    return Game{metadata(b, t, r, e), h, f, Nil{}}

def start() -> Game:
  Game{Chess.Board{U64.from_parts(16711680, 65280), U64.from_parts(1107296256, 66),
    U64.from_parts(603979776, 36), U64.from_parts(2164260864, 129),
    U64.from_parts(134217728, 8), U64.from_parts(268435456, 16),
    U64.from_parts(0, 65535), U64.from_parts(4294901760, 0), 1, 15, 64}, 0, 1, Nil{}}

def occupied_by(+b: Chess.Board, +sq: U32, p: U32, side: U32) -> Bool:
  Bool.and(U64.test_bit(Chess.color(b, side), U32.to_nat(sq)), U32.is_eq(Chess.piece(b, sq), p))

def valid_right(+b: Chess.Board, bit: U32, rook: U32, king: U32, +side: U32) -> Bool:
  Bool.or(U32.is_zero(U32.and(Chess.get_rights(b), bit)),
    Bool.and(occupied_by(b, rook, 3, side), occupied_by(b, king, 5, side)))

def valid_ep(+sq: U32, +b: Chess.Board) -> Bool:
  match sq:
    case 64: True{}
    case _:
      +white = U32.is_eq(Chess.get_turn(b), 1)
      origin = Bool.pick(U32, white, U32.add(sq, 8), U32.sub(sq, 8))
      Bool.and(U32.is_eq(U32.div(sq, 8), Bool.pick(U32, white, 5, 2)),
        Bool.and(Bool.not(U64.test_bit(Chess.occupied(b), U32.to_nat(sq))),
          Bool.and(Bool.not(U64.test_bit(Chess.occupied(b), U32.to_nat(origin))),
            occupied_by(b, U32.xor(sq, 8), 0, U32.xor(Chess.get_turn(b), 1)))))

def structural(+b: Chess.Board) -> Bool:
  Bool.and(U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b), Chess.get_white(b))), 1),
    Bool.and(U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b), Chess.get_black(b))), 1),
      Bool.and(U32.is_le(U64.popcount(Chess.get_white(b)), 16),
        Bool.and(U32.is_le(U64.popcount(Chess.get_black(b)), 16),
          Bool.and(U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b), Chess.get_white(b))), 8),
            Bool.and(U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b), Chess.get_black(b))), 8),
              U64.is_zero(U64.and(Chess.get_pawns(b), U64.from_parts(4278190080, 255)))))))))

def valid(+b: Chess.Board) -> Bool:
  Bool.and(structural(b), Bool.and(valid_ep(Chess.get_ep(b), b),
    Bool.and(valid_right(b, 1, 7, 4, 1), Bool.and(valid_right(b, 2, 0, 4, 1),
      Bool.and(valid_right(b, 4, 63, 60, 0), valid_right(b, 8, 56, 60, 0))))))

def promo(p: U32) -> String:
  match p:
    case 1: "n"
    case 2: "b"
    case 3: "r"
    case 4: "q"
    case _: ""

def move_text(m: Chess.Ply) -> String:
  Chess.Ply{src, dst, promotion, flag} = m
  Text.square_text(src) ++ Text.square_text(dst) ++ promo(promotion)

def chosen(equal: Bool, m: Chess.Ply, rest: Maybe<&2, Chess.Ply>) -> Maybe<&2, Chess.Ply>:
  match equal:
    case True{}: Some{m}
    case False{}: rest

def find_move(ms: List<&2, Chess.Ply>, +text: String) -> Maybe<&2, Chess.Ply>:
  match ms:
    case Nil{}: None{}
    case Con{+m, rest}: chosen(String.eq(move_text(m), text), m, find_move(rest, text))

def apply(g: Game, +m: Chess.Ply) -> Game:
  Game{+b, +half, +full, history} = g
  Chess.Ply{src, dst, promotion, flag} = m
  reset = Bool.or(U32.is_zero(Chess.piece(b, src)), U64.test_bit(Chess.occupied(b), U32.to_nat(dst)))
  Game{Chess.make_move(b, m), Bool.pick(U32, reset, 0, U32.inc(half)),
    U32.add(full, Bool.to_u32(U32.is_zero(Chess.get_turn(b)))), Con{Past{b, half, full, m}, history}}
