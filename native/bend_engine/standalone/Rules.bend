# Orthodox rule facts over Bend-owned history. No foreign chess/rules callbacks.
# Automatic draws only: optional threefold/50-move claims do not end a line.
import Base
import ../legal_probe/Chess.bend as Chess
import ./Position.bend as Pos

# Codes: 0 ongoing, 1 mate, 2 stalemate, 3 insufficient material,
# 4 seventy-five moves, 5 fivefold repetition. Counts include the current board.
type Facts is Data:
  Facts{reason: U32, repetitions: U32, halfmove: U32}

def name(code: U32) -> String:
  match code:
    case 1: "checkmate"
    case 2: "stalemate"
    case 3: "insufficient_material"
    case 4: "seventyfive_moves"
    case 5: "fivefold_repetition"
    case _: "ongoing"

def has_ep(ms: List<&2, Chess.Ply>) -> Bool:
  match ms:
    case Nil{}: False{}
    case Con{Chess.Ply{src, dst, promo, flag}, rest}:
      Bool.or(U32.is_eq(flag, 1), has_ep(rest))

def canonical(+b: Chess.Board, legal_ep: Bool) -> Chess.Board:
  Pos.metadata(b, Chess.get_turn(b), Chess.get_rights(b),
    Bool.pick(U32, legal_ep, Chess.get_ep(b), 64))

def normalized(b: Chess.Board, r: Array<U64> & List<&2, Chess.Ply>) -> Array<U64> & Chess.Board:
  (table, ms) = r
  (table, canonical(b, has_ep(ms)))

def normalize_ep(ep: U32, +b: Chess.Board, table: Array<U64>) -> Array<U64> & Chess.Board:
  match ep:
    case 64: (table, b)
    case _: normalized(b, Chess.legal_moves(table, b))

# Full structural identity, not a collision-prone hash. FEN clocks are excluded.
def equal(+a: Chess.Board, +b: Chess.Board) -> Bool:
  Bool.and(U64.is_eq(Chess.get_pawns(a), Chess.get_pawns(b)),
    Bool.and(U64.is_eq(Chess.get_knights(a), Chess.get_knights(b)),
      Bool.and(U64.is_eq(Chess.get_bishops(a), Chess.get_bishops(b)),
        Bool.and(U64.is_eq(Chess.get_rooks(a), Chess.get_rooks(b)),
          Bool.and(U64.is_eq(Chess.get_queens(a), Chess.get_queens(b)),
            Bool.and(U64.is_eq(Chess.get_kings(a), Chess.get_kings(b)),
              Bool.and(U64.is_eq(Chess.get_white(a), Chess.get_white(b)),
                Bool.and(U64.is_eq(Chess.get_black(a), Chess.get_black(b)),
                  Bool.and(U32.is_eq(Chess.get_turn(a), Chess.get_turn(b)),
                    Bool.and(U32.is_eq(Chess.get_rights(a), Chess.get_rights(b)),
                      U32.is_eq(Chess.get_ep(a), Chess.get_ep(b))))))))))))

def count_seen(target: Chess.Board, count: U32, r: Array<U64> & Chess.Board) -> Array<U64> & U32:
  (table, old) = r
  (table, U32.add(count, Bool.to_u32(equal(target, old))))

def count_one(+target: Chess.Board, +old: Chess.Board, r: Array<U64> & U32) -> Array<U64> & U32:
  (table, count) = r
  count_seen(target, count, normalize_ep(Chess.get_ep(old), old, table))

# The input protocol bounds played history to 512; a search appends at most 32.
# Only the reversible window can contribute a repetition, even for a FEN root.
def repetitions(fuel: Nat, hist: List<&2, Pos.Past>, +target: Chess.Board,
  r: Array<U64> & U32) -> Array<U64> & U32:
  match fuel:
    case 0n: r
    case 1n+p:
      match hist:
        case Nil{}: r
        case Con{Pos.Past{old, half, full, move}, rest}:
          repetitions(p, rest, target, count_one(target, old, r))

# Conservative dead-position subset: bare kings, a sole bishop/knight, or only
# bishops all on the same color. Two knights and opposite-color bishops remain
# nonterminal. This is intentionally not an exhaustive dead-position solver.
def insufficient(+b: Chess.Board) -> Bool:
  +minor = U64.or(Chess.get_knights(b), Chess.get_bishops(b))
  +bishops = Chess.get_bishops(b)
  +dark = U64.from_parts(2857740885, 2857740885)
  one_color = Bool.or(U64.is_zero(U64.and(bishops, dark)), U64.is_zero(U64.and(bishops, U64.not(dark))))
  Bool.and(U64.is_zero(U64.or(Chess.get_pawns(b), U64.or(Chess.get_rooks(b), Chess.get_queens(b)))),
    Bool.or(U32.is_le(U64.popcount(minor), 1), Bool.and(U64.is_zero(Chess.get_knights(b)), one_color)))

def counted(b: Chess.Board, +half: U32, r: Array<U64> & U32) -> Array<U64> & Facts:
  (table, +count) = r
  reason = Bool.pick(U32, insufficient(b), 3,
    Bool.pick(U32, U32.is_ge(half, 150), 4, Bool.pick(U32, U32.is_ge(count, 5), 5, 0)))
  (table, Facts{reason, count, half})

# Caller supplies the complete legal list. Native search checks no-move terminals
# first; diagnostic inspection uses the same order so mate wins at halfmove 150.
def automatic(g: Pos.Game, legal: List<&2, Chess.Ply>, table: Array<U64>) -> Array<U64> & Facts:
  Pos.Game{+b, +half, full, hist} = g
  target = canonical(b, has_ep(legal))
  counted(b, half, repetitions(U32.to_nat(U32.min(half, 544)), hist, target, (table, 1)))

def no_moves(half: U32, r: Array<U64> & Bool) -> Array<U64> & Facts:
  (table, check) = r
  (table, Facts{Bool.pick(U32, check, 1, 2), 1, half})

def no_moves_game(g: Pos.Game, table: Array<U64>) -> Array<U64> & Facts:
  Pos.Game{+b, half, full, hist} = g
  no_moves(half, Chess.in_check(table, b, Chess.get_turn(b)))

def inspected(g: Pos.Game, r: Array<U64> & List<&2, Chess.Ply>) -> Array<U64> & Facts:
  (table, moves) = r
  match moves:
    case Nil{}: no_moves_game(g, table)
    case Con{m, rest}: automatic(g, Con{m, rest}, table)

def inspect(+g: Pos.Game, table: Array<U64>) -> Array<U64> & Facts:
  inspected(g, Chess.legal_moves(table, Pos.board(g)))
