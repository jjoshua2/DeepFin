# DeepFin policy spaces, generated from geometry in Bend (no embedded lookup file).
# Native packed move keys, full AZ-4672 actions, and compact LC0-1858 model slots
# are DIFFERENT spaces. Geometry alone never establishes legality on a board.
import Base
import ../legal_probe/Chess.bend as Chess

# The U32 sentinel is internal invalid data, never a model slot or a move.
type Maps is Type:
  Maps{full_to_compact: Array<U32>, compact_to_full: Array<U32>,
    pair_to_full: Array<U32>, count: U32}

type Space is Data:
  Full{}
  Compact{}

type Entry is Data:
  Entry{move: Chess.Ply, full: U32, compact: U32}

# Deltas use bias eight; unsigned wrap is never used as a valid board coordinate.
def queen_dir(direction: U32) -> U32 & U32:
  match direction:
    case 0: (1, 2)
    case 1: (2, 2)
    case 2: (2, 1)
    case 3: (2, 0)
    case 4: (1, 0)
    case 5: (0, 0)
    case 6: (0, 1)
    case _: (0, 2)

def queen_delta(+distance: U32, r: U32 & U32) -> U32 & U32:
  (x, y) = r
  (U32.sub(U32.add(8, U32.mul(x, distance)), distance),
   U32.sub(U32.add(8, U32.mul(y, distance)), distance))

def knight_delta(i: U32) -> U32 & U32:
  match i:
    case 0: (9, 10)
    case 1: (10, 9)
    case 2: (10, 7)
    case 3: (9, 6)
    case 4: (7, 6)
    case 5: (6, 7)
    case 6: (6, 9)
    case _: (7, 10)

def special_delta(knight: Bool, +p: U32) -> U32 & U32:
  match knight:
    case True{}: knight_delta(U32.sub(p, 56))
    case False{}: (U32.add(7, U32.mod(U32.sub(p, 64), 3)), 9)

def delta_kind(queen: Bool, +p: U32) -> U32 & U32:
  match queen:
    case True{}: queen_delta(U32.inc(U32.mod(p, 7)), queen_dir(U32.div(p, 7)))
    case False{}: special_delta(U32.is_lt(p, 64), p)

def landing_coords(+x: U32, +y: U32) -> U32:
  Bool.pick(U32, Bool.and(Bool.and(U32.is_ge(x, 8), U32.is_lt(x, 16)),
    Bool.and(U32.is_ge(y, 8), U32.is_lt(y, 16))),
    U32.add(U32.sub(x, 8), U32.mul(U32.sub(y, 8), 8)), 64)

def landing(sq: U32, r: U32 & U32) -> U32:
  (dx, dy) = r
  +sq = sq
  landing_coords(U32.add(U32.mod(sq, 8), dx), U32.add(U32.div(sq, 8), dy))

def destination(+full: U32) -> U32:
  +p = U32.mod(full, 73)
  landing(U32.div(full, 73), delta_kind(U32.is_lt(p, 56), p))

def valid_index(+full: U32, dst: U32) -> Bool:
  Bool.and(U32.is_lt(full, 4672), Bool.and(U32.is_lt(dst, 64),
    Bool.or(U32.is_lt(U32.mod(full, 73), 64), U32.is_eq(U32.div(full, 584), 6))))

def set_pair(normal: Bool, index: U32, full: U32, pairs: Array<U32>) -> Array<U32>:
  match normal:
    case False{}: pairs
    case True{}: Array.set(U32, pairs, index, full)

def put(valid: Bool, +i: U32, dst: U32, maps: Maps) -> Maps:
  match valid:
    case False{}: maps
    case True{}:
      Maps{f2c, c2f, pairs, +n} = maps
      Maps{Array.set(U32, f2c, i, n), Array.set(U32, c2f, n, i),
        set_pair(U32.is_lt(U32.mod(i, 73), 64), U32.add(U32.mul(U32.div(i, 73), 64), dst), i, pairs), U32.inc(n)}

def build_step(+i: U32, +dst: U32, maps: Maps) -> Maps:
  put(valid_index(i, dst), i, dst, maps)

def build_go(fuel: Nat, +i: U32, maps: Maps) -> Maps:
  match fuel:
    case 0n: maps
    case 1n+p: build_go(p, U32.inc(i), build_step(i, destination(i), maps))

def build() -> Maps:
  build_go(4672n, 0, Maps{Array.new(U32, 13n, 4294967295),
    Array.new(U32, 11n, 4294967295), Array.new(U32, 12n, 4294967295), 0})

def read_f2c(c2f: Array<U32>, pairs: Array<U32>, n: U32, r: Array<U32> & U32) -> Maps & U32:
  (f2c, value) = r
  (Maps{f2c, c2f, pairs, n}, value)

def read_c2f(f2c: Array<U32>, pairs: Array<U32>, n: U32, r: Array<U32> & U32) -> Maps & U32:
  (c2f, value) = r
  (Maps{f2c, c2f, pairs, n}, value)

def read_pair(f2c: Array<U32>, c2f: Array<U32>, n: U32, r: Array<U32> & U32) -> Maps & U32:
  (pairs, value) = r
  (Maps{f2c, c2f, pairs, n}, value)

# No native Array access until the logical width check passes.
def to_compact_valid(valid: Bool, index: U32, maps: Maps) -> Maps & U32:
  match valid:
    case False{}: (maps, 4294967295)
    case True{}:
      Maps{f2c, c2f, pairs, n} = maps
      read_f2c(c2f, pairs, n, Array.get(U32, f2c, index))

def to_compact(+index: U32, maps: Maps) -> Maps & U32:
  to_compact_valid(U32.is_lt(index, 4672), index, maps)

def to_full_valid(valid: Bool, index: U32, maps: Maps) -> Maps & U32:
  match valid:
    case False{}: (maps, 4294967295)
    case True{}:
      Maps{f2c, c2f, pairs, n} = maps
      read_c2f(f2c, pairs, n, Array.get(U32, c2f, index))

def to_full(+index: U32, maps: Maps) -> Maps & U32:
  to_full_valid(U32.is_lt(index, 1858), index, maps)

def mirror_special(knight: Bool, +p: U32) -> U32:
  match knight:
    case True{}: U32.sub(119, p)
    case False{}:
      +r = U32.sub(p, 64)
      U32.add(64, U32.add(U32.mul(U32.div(r, 3), 3), U32.sub(2, U32.mod(r, 3))))

def mirror_plane(queen: Bool, +p: U32) -> U32:
  match queen:
    case True{}: U32.add(U32.mul(U32.mod(U32.sub(8, U32.div(p, 7)), 8), 7), U32.mod(p, 7))
    case False{}: mirror_special(U32.is_lt(p, 64), p)

def mirror_valid(valid: Bool, +full: U32) -> U32:
  match valid:
    case False{}: 4294967295
    case True{}:
      +p = U32.mod(full, 73)
      U32.add(U32.mul(U32.xor(U32.div(full, 73), 7), 73), mirror_plane(U32.is_lt(p, 56), p))

def mirror_full(+full: U32) -> U32:
  mirror_valid(U32.is_lt(full, 4672), full)

def mirror_resolved(r: Maps & U32) -> Maps & U32:
  (maps, full) = r
  to_compact(mirror_full(full), maps)

def mirror_compact(index: U32, maps: Maps) -> Maps & U32:
  mirror_resolved(to_full(index, maps))

# These APIs encode GENERATED legal moves; field/geometry checks are defensive,
# not a second legal generator. Reverse resolution must use current legal entries.
def promotion_shape(+src: U32, +dst: U32) -> Bool:
  Bool.and(U32.is_eq(U32.div(src, 8), 6), Bool.and(U32.is_eq(U32.div(dst, 8), 7),
    U32.is_le(U32.sub(U32.inc(U32.mod(dst, 8)), U32.mod(src, 8)), 2)))

def move_full(under: Bool, +src: U32, dst: U32, promotion: U32, maps: Maps) -> Maps & U32:
  match under:
    case True{}:
      (maps, U32.add(U32.mul(src, 73), U32.add(64,
        U32.add(U32.mul(U32.sub(promotion, 1), 3), U32.sub(U32.inc(U32.mod(dst, 8)), U32.mod(src, 8))))))
    case False{}:
      Maps{f2c, c2f, pairs, n} = maps
      read_pair(f2c, c2f, n, Array.get(U32, pairs, U32.add(U32.mul(src, 64), dst)))

def entry_compact(move: Chess.Ply, full: U32, r: Maps & U32) -> Maps & Maybe<&2, Entry>:
  (maps, compact) = r
  match compact:
    case 4294967295: (maps, None{})
    case _: (maps, Some{Entry{move, full, compact}})

def entry_full(move: Chess.Ply, r: Maps & U32) -> Maps & Maybe<&2, Entry>:
  (maps, +full) = r
  entry_compact(move, full, to_compact(full, maps))

def encode_shape(valid: Bool, src: U32, dst: U32, +promotion: U32, move: Chess.Ply, maps: Maps) -> Maps & Maybe<&2, Entry>:
  match valid:
    case False{}: (maps, None{})
    case True{}: entry_full(move, move_full(Bool.and(U32.is_gt(promotion, 0), U32.is_lt(promotion, 4)), src, dst, promotion, maps))

def encode_fields(valid: Bool, +turn: U32, +move: Chess.Ply, maps: Maps) -> Maps & Maybe<&2, Entry>:
  match valid:
    case False{}: (maps, None{})
    case True{}:
      Chess.Ply{src, dst, +promotion, flag} = move
      +flip = Bool.pick(U32, U32.is_zero(turn), 56, 0)
      +src = U32.xor(src, flip)
      +dst = U32.xor(dst, flip)
      encode_shape(Bool.or(U32.is_zero(promotion), promotion_shape(src, dst)), src, dst, promotion, move, maps)

def encode_generated(+turn: U32, +move: Chess.Ply, maps: Maps) -> Maps & Maybe<&2, Entry>:
  Chess.Ply{src, dst, promotion, flag} = move
  encode_fields(Bool.and(U32.is_le(turn, 1), Bool.and(U32.is_lt(src, 64),
    Bool.and(U32.is_lt(dst, 64), Bool.and(U32.is_le(promotion, 4), U32.is_le(flag, 2))))), turn, move, maps)

type Collected is Type:
  Collected{maps: Maps, entries: List<&2, Entry>, valid: Bool}

def entries_add(acc: List<&2, Entry>, valid: Bool, r: Maps & Maybe<&2, Entry>) -> Collected:
  (maps, entry) = r
  match entry:
    case None{}: Collected{maps, acc, False{}}
    case Some{entry}: Collected{maps, Con{entry, acc}, valid}

def entries_finish(valid: Bool, acc: List<&2, Entry>, maps: Maps) -> Maps & Maybe<&2, List<&2, Entry>>:
  match valid:
    case False{}: (maps, None{})
    case True{}: (maps, Some{List.reverse(&2, Entry, acc)})

def entries_go(+turn: U32, moves: List<&2, Chess.Ply>, r: Collected) -> Maps & Maybe<&2, List<&2, Entry>>:
  match moves:
    case Nil{}:
      Collected{maps, acc, valid} = r
      entries_finish(valid, acc, maps)
    case Con{move, rest}:
      Collected{maps, acc, valid} = r
      entries_go(turn, rest, entries_add(acc, valid, encode_generated(turn, move, maps)))

def entries(turn: U32, moves: List<&2, Chess.Ply>, maps: Maps) -> Maps & Maybe<&2, List<&2, Entry>>:
  entries_go(turn, moves, Collected{maps, Nil{}, True{}})

# Pure boundary helpers for already generated legal entries. No fallback move.
def full_space(space: Space) -> Bool:
  match space:
    case Full{}: True{}
    case Compact{}: False{}

def entry_id(space: Space, e: Entry) -> U32:
  Entry{move, full, compact} = e
  Bool.pick(U32, full_space(space), full, compact)

def chosen(hit: Bool, entry: Entry, rest: Maybe<&2, Entry>) -> Maybe<&2, Entry>:
  match hit:
    case True{}: Some{entry}
    case False{}: rest

def resolve(+space: Space, +index: U32, entries: List<&2, Entry>) -> Maybe<&2, Entry>:
  match entries:
    case Nil{}: None{}
    case Con{+entry, rest}: chosen(U32.is_eq(index, entry_id(space, entry)), entry, resolve(space, index, rest))
