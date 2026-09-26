# Native Chess.Board adapter for PositionIndex. Valid orthodox boards only.
# Structural identity follows the DAG's pseudo-legal EP rule, NOT repetition.
import Base
import ../legal_probe/Chess.bend as C
import ./PositionIndex.bend as P

def ep_from(+b: C.Board, +ep: U32) -> U32:
  +target = U64.bit(U32.to_nat(ep))
  +not_a = U64.from_parts(4278124286, 4278124286)
  +not_h = U64.from_parts(2139062143, 2139062143)
  attackers = C.select_u64(U32.is_eq(C.get_turn(b), 1),
    U64.or(U64.and(U64.shrn(target, 7n), not_a), U64.and(U64.shrn(target, 9n), not_h)),
    U64.or(U64.and(U64.shln(target, 7n), not_h), U64.and(U64.shln(target, 9n), not_a)))
  available = Bool.not(U64.is_zero(U64.and(attackers, U64.and(C.get_pawns(b), C.color(b, C.get_turn(b))))))
  Bool.pick(U32, available, ep, 64)

def ep_checked(b: C.Board, ep: U32, present: Bool) -> U32:
  match present:
    case False{}: 64
    case True{}: ep_from(b, ep)

def canonical_ep(+b: C.Board) -> U32:
  +ep = C.get_ep(b)
  ep_checked(b, ep, U32.is_lt(ep, 64))

def identity(+b: C.Board) -> P.Identity:
  P.Identity{C.get_pawns(b), C.get_knights(b), C.get_bishops(b), C.get_rooks(b),
    C.get_queens(b), C.get_kings(b), C.get_white(b), C.get_black(b),
    C.get_turn(b), U32.and(C.get_rights(b), 15), canonical_ep(b)}

# Deterministic word-wise mixer over the canonical fields, not CBoard Zobrist.
# No serialized-key compatibility or cryptographic/distribution claim is made.
def mix(hash: U64, word: U64) -> U64:
  U64.mul(U64.xor(hash, word), U64.from_parts(256, 435))

def hash(fields: P.Identity) -> U64:
  P.Identity{p, n, b, r, q, k, w, x, turn, castling, ep} = fields
  h = mix(U64.from_parts(3421674724, 2216829733), p)
  h = mix(h, n)
  h = mix(h, b)
  h = mix(h, r)
  h = mix(h, q)
  h = mix(h, k)
  h = mix(h, w)
  h = mix(h, x)
  h = mix(h, U64.from_u32(turn))
  h = mix(h, U64.from_u32(castling))
  mix(h, U64.from_u32(ep))

# Deriving the key and fields together removes caller hash/identity mismatches.
# Keep the game/history owner separately; no board, clock, or history is mutated.
def intern(index: P.Index, b: C.Board) -> P.Index & P.Outcome:
  +fields = identity(b)
  P.intern(index, hash(fields), fields)

def get(index: P.Index, b: C.Board) -> P.Index & Maybe<&2, U32>:
  +fields = identity(b)
  P.get(index, hash(fields), fields)
