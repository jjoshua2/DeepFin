# Standard UCI clocks resolved in Bend before entering the existing search loop.
# This module owns no tree, model or native effect. A clock allocation is a
# cooperative ceiling; the bounded search may complete before that ceiling.
import Base
import ./Protocol.bend as Protocol
import ./Text.bend as Text

type Clocks is Data:
  Clocks{white: U32, black: U32, white_inc: U32, black_inc: U32, moves: U32, seen: U32, valid: Bool}

type Reading is Data:
  Reading{clocks: Clocks, pending: U32, rest: List<&2, String>}

def field(+word: String) -> U32:
  Bool.pick(U32, String.eq(word, "wtime"), 1, Bool.pick(U32, String.eq(word, "btime"), 2,
    Bool.pick(U32, String.eq(word, "winc"), 4, Bool.pick(U32, String.eq(word, "binc"), 8,
      Bool.pick(U32, String.eq(word, "movestogo"), 16, 0)))))

def set(which: U32, value: Maybe<&2, U32>, clocks: Clocks) -> Clocks:
  match value:
    case None{}:
      Clocks{w, b, wi, bi, moves, seen, valid} = clocks
      Clocks{w, b, wi, bi, moves, seen, False{}}
    case Some{+v}:
      Clocks{w, b, wi, bi, moves, +seen, valid} = clocks
      +which = which
      range = Bool.pick(Bool, U32.is_eq(which, 16),
        Bool.and(U32.is_gt(v, 0), U32.is_le(v, 1000)), U32.is_le(v, 86400000))
      ok = Bool.and(valid, Bool.and(range, U32.is_zero(U32.and(seen, which))))
      Clocks{Bool.pick(U32, U32.is_eq(which, 1), v, w), Bool.pick(U32, U32.is_eq(which, 2), v, b),
        Bool.pick(U32, U32.is_eq(which, 4), v, wi), Bool.pick(U32, U32.is_eq(which, 8), v, bi),
        Bool.pick(U32, U32.is_eq(which, 16), v, moves), U32.or(seen, which), ok}

def existing(word: String, clocks: Clocks, rest: List<&2, String>, +which: U32) -> Reading:
  numeric = Bool.and(U32.is_gt(which, 0), Bool.and(Bool.not(U32.is_eq(which, 8)), Bool.not(U32.is_eq(which, 32))))
  Reading{clocks, Bool.pick(U32, numeric, 32, 0), Con{word, rest}}

def word_field(+word: String, clocks: Clocks, rest: List<&2, String>, which: U32) -> Reading:
  match which:
    case 0: existing(word, clocks, rest, Protocol.field(word))
    case _: Reading{clocks, which, rest}

def word(+word: String, reading: Reading) -> Reading:
  Reading{clocks, pending, rest} = reading
  match pending:
    case 0: word_field(word, clocks, rest, field(word))
    # Preserve the argument after nodes/depth/movetime/evals verbatim. Otherwise
    # "nodes wtime 1000 4" could wrongly become a valid "nodes 4" after filtering.
    case 32: Reading{clocks, 0, Con{word, rest}}
    case _: Reading{set(pending, Text.number(word), clocks), 0, rest}

def read(xs: List<&2, String>, reading: Reading) -> Reading:
  match xs:
    case Nil{}: reading
    case Con{head, tail}: read(tail, word(head, reading))

# Bounds keep all arithmetic below U32 overflow. At/below the
# configured reserve return immediately without selecting or evaluating a search leaf.
def milliseconds(+remaining: U32, increment: U32, moves: U32, overhead: U32) -> U32:
  available = U32.sub(remaining, U32.min(remaining, overhead))
  target = U32.add(U32.div(remaining, moves), U32.div(U32.mul(increment, 3), 4))
  U32.min(60000, U32.min(available, U32.max(1, target)))

def allocated(ms: U32, ok: Bool, lim: Protocol.Limits) -> Protocol.Limits:
  Protocol.Limits{+nodes, depth, duration, infinite, +seen, valid, evals} = lim
  +ms = ms
  # Internal guard follows measured-limits semantics; explicit nodes still win.
  budget = Bool.pick(U32, U32.is_zero(ms), 0,
    Bool.pick(U32, U32.is_zero(U32.and(seen, 1)), 65536, nodes))
  Protocol.Limits{budget, depth, ms, infinite, U32.or(seen, 64), Bool.and(valid, ok), evals}

def valid_limits(ok: Bool, lim: Protocol.Limits) -> Protocol.Limits:
  Protocol.Limits{nodes, depth, duration, infinite, seen, valid, evals} = lim
  Protocol.Limits{nodes, depth, duration, infinite, seen, Bool.and(ok, valid), evals}

def resolve_active(turn: U32, clocks: Clocks, lim: Protocol.Limits, overhead: U32) -> Protocol.Limits:
  Clocks{w, b, wi, bi, moves, seen, valid} = clocks
  Protocol.Limits{nodes, depth, +duration, +infinite, +limits_seen, limits_valid, evals} = lim
  +white = U32.is_eq(turn, 1)
  present = U32.is_gt(U32.and(seen, Bool.pick(U32, white, 1, 2)), 0)
  ok = Bool.and(valid, Bool.and(present, Bool.not(infinite)))
  # Explicit movetime takes precedence, but supplied clock syntax is checked.
  ms = Bool.pick(U32, U32.is_gt(U32.and(limits_seen, 4), 0), duration,
    milliseconds(Bool.pick(U32, white, w, b), Bool.pick(U32, white, wi, bi), moves, overhead))
  allocated(ms, ok, Protocol.Limits{nodes, depth, duration, infinite, limits_seen, limits_valid, evals})

def resolve(turn: U32, clocks: Clocks, lim: Protocol.Limits, overhead: U32) -> Protocol.Limits:
  Clocks{w, b, wi, bi, moves, seen, valid} = clocks
  match seen:
    case 0: valid_limits(valid, lim)
    case _: resolve_active(turn, Clocks{w, b, wi, bi, moves, seen, valid}, lim, overhead)

def finish(turn: U32, reading: Reading, overhead: U32) -> Protocol.Limits:
  Reading{Clocks{w, b, wi, bi, moves, seen, valid}, pending, rest} = reading
  lim = Protocol.go(List.reverse(&2, String, rest),
    Protocol.GoState{Protocol.Limits{64, 4, 0, False{}, 0, True{}, 0}, 0})
  resolve(turn, Clocks{w, b, wi, bi, moves, seen, Bool.and(valid, U32.is_zero(pending))}, lim, overhead)

def go(xs: List<&2, String>, turn: U32, overhead: U32) -> Protocol.Limits:
  finish(turn, read(xs, Reading{Clocks{0, 0, 0, 0, 30, 0, True{}}, 0, Nil{}}), overhead)
