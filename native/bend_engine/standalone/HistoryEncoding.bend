# DeepFin's 112 base planes, authored in Bend. NOT a full 146/175-plane input.
# Both layouts keep every history frame in the CURRENT board's point of view.
# No Python/C encoder, model callback, or pre-generated plane data.
import Base
import ../legal_probe/Chess.bend as Chess
import ./Position.bend as Pos
import ./Rules.bend as Rules

type Layout is Data:
  Root{}
  RootLegacyMeta{}

# Logical shape is [112,8,8]; the Array capacity is 8192, not a model channel count.
# Exactly the first 7168 contiguous F32 entries are initialized/exposed as planes.
type Encoded is Type:
  Encoded{table: Array<U64>, values: Array<F32>}

def name(layout: Layout) -> String:
  match layout:
    case Root{}: "lc0_root"
    case RootLegacyMeta{}: "lc0_root_legacy_meta"

def parse_meta(matches: Bool) -> Maybe<&2, Layout>:
  match matches:
    case True{}: Some{RootLegacyMeta{}}
    case False{}: None{}

def parse_root(matches: Bool, s: String) -> Maybe<&2, Layout>:
  match matches:
    case True{}: Some{Root{}}
    case False{}: parse_meta(String.eq(s, "lc0_root_legacy_meta"))

def parse(+s: String) -> Maybe<&2, Layout>:
  parse_root(String.eq(s, "lc0_root"), s)

def piece_mask(kind: U32, b: Chess.Board) -> U64:
  match kind:
    case 0: Chess.get_pawns(b)
    case 1: Chess.get_knights(b)
    case 2: Chess.get_bishops(b)
    case 3: Chess.get_rooks(b)
    case 4: Chess.get_queens(b)
    case _: Chess.get_kings(b)

# Square rank flip only for black: file order is never reversed.
def squares(n: Nat, +sq: U32, +offset: U32, +flip: U32, +mask: U64,
  +value: F32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n: a
    case 1n+p:
      v = Bool.pick(F32, U64.test_bit(mask, U32.to_nat(U32.xor(sq, flip))), value, 0.0)
      squares(p, U32.inc(sq), offset, flip, mask, value, Array.set(F32, a, U32.add(offset, sq), v))

def plane(index: U32, flip: U32, mask: U64, value: F32, a: Array<F32>) -> Array<F32>:
  squares(64n, 0, U32.mul(index, 64), flip, mask, value, a)

def fill(index: U32, value: F32, a: Array<F32>) -> Array<F32>:
  plane(index, 0, U64.not(U64.zero()), value, a)

def pieces(n: Nat, +i: U32, +base: U32, +turn: U32, +b: Chess.Board, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n: a
    case 1n+p:
      side = Bool.pick(U32, U32.is_lt(i, 6), turn, U32.xor(turn, 1))
      mask = U64.and(piece_mask(U32.mod(i, 6), b), Chess.color(b, side))
      flip = Bool.pick(U32, U32.is_zero(turn), 56, 0)
      pieces(p, U32.inc(i), base, turn, b, plane(U32.add(base, i), flip, mask, 1.0, a))

def repetition_plane(base: U32, a: Array<F32>, r: Array<U64> & U32) -> Encoded:
  (table, count) = r
  Encoded{table, fill(U32.add(base, 12), Bool.pick(F32, U32.is_gt(count, 1), 1.0, 0.0), a)}

def normalized_frame(base: U32, half: U32, hist: List<&2, Pos.Past>, a: Array<F32>,
  r: Array<U64> & Chess.Board) -> Encoded:
  (table, target) = r
  repetition_plane(base, a, Rules.repetitions(U32.to_nat(U32.min(half, 544)), hist, target, (table, 1)))

def frame(+base: U32, turn: U32, g: Pos.Game, e: Encoded) -> Encoded:
  Pos.Game{+b, half, full, hist} = g
  Encoded{table, a} = e
  normalized_frame(base, half, hist, pieces(12n, 0, base, turn, b, a),
    Rules.normalize_ep(Chess.get_ep(b), b, table))

# Unavailable pre-FEN frames stay zero; never repeat-pad the earliest board.
# Each historical repetition flag uses only the history BEFORE that frame.
def earlier(fuel: Nat, +base: U32, +turn: U32, hist: List<&2, Pos.Past>, e: Encoded) -> Encoded:
  match fuel:
    case 0n: e
    case 1n+p:
      match hist:
        case Nil{}: e
        case Con{Pos.Past{b, half, full, move}, +rest}:
          earlier(p, U32.add(base, 13), turn, rest, frame(base, turn, Pos.Game{b, half, full, rest}, e))

def castling(+rights: U32, +turn: U32, a: Array<F32>) -> Array<F32>:
  +us = Bool.pick(U32, U32.is_zero(turn), 4, 1)
  +them = Bool.pick(U32, U32.is_zero(turn), 1, 4)
  a = fill(104, Bool.pick(F32, U32.is_gt(U32.and(rights, U32.mul(us, 2)), 0), 1.0, 0.0), a)
  a = fill(105, Bool.pick(F32, U32.is_gt(U32.and(rights, us), 0), 1.0, 0.0), a)
  a = fill(106, Bool.pick(F32, U32.is_gt(U32.and(rights, U32.mul(them, 2)), 0), 1.0, 0.0), a)
  fill(107, Bool.pick(F32, U32.is_gt(U32.and(rights, them), 0), 1.0, 0.0), a)

def ep_plane(ep: U32, a: Array<F32>) -> Array<F32>:
  match ep:
    case 64: a
    case _: plane(110, 0, U64.shln(U64.from_parts(16843009, 16843009), U32.to_nat(U32.mod(ep, 8))), 1.0, a)

def scaled(layout: Layout, half: U32, ep: U32, a: Array<F32>) -> Array<F32>:
  match layout:
    case Root{}: fill(109, F32.from_nat(U32.to_nat(half)), a)
    case RootLegacyMeta{}:
      ep_plane(ep, fill(109, F32.div(F32.from_nat(U32.to_nat(U32.min(half, 100))), 100.0), a))

def metadata(layout: Layout, g: Pos.Game, e: Encoded) -> Encoded:
  Pos.Game{+b, half, full, hist} = g
  Encoded{table, a} = e
  +turn = Chess.get_turn(b)
  a = castling(Chess.get_rights(b), turn, a)
  a = fill(108, Bool.pick(F32, U32.is_zero(turn), 1.0, 0.0), a)
  a = scaled(layout, half, Chess.get_ep(b), a)
  Encoded{table, fill(111, 1.0, a)}

def encode(layout: Layout, +g: Pos.Game, table: Array<U64>) -> Encoded:
  Pos.Game{+b, half, full, hist} = g
  +turn = Chess.get_turn(b)
  metadata(layout, g, earlier(7n, 13, turn, hist,
    frame(0, turn, g, Encoded{table, Array.new(F32, 13n, 0.0)})))
