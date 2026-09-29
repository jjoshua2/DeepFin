import Base
import ./Text.bend as Text
import ./Position.bend as Pos
import ../legal_probe/Chess.bend as Chess

type PositionInput is Data:
  PositionInput{game: Maybe<&2, Pos.Game>, moves: List<&2, String>}

def tail_checked(ok: Bool, game: Maybe<&2, Pos.Game>, rest: List<&2, String>) -> PositionInput:
  match ok:
    case True{}: PositionInput{game, rest}
    case False{}: PositionInput{None{}, Nil{}}

def position_tail(game: Maybe<&2, Pos.Game>, tail: List<&2, String>) -> PositionInput:
  match tail:
    case Nil{}: PositionInput{game, Nil{}}
    case Con{h, rest}: tail_checked(String.eq(h, "moves"), game, rest)

def position_kind(start: Bool, args: List<&2, String>) -> PositionInput:
  match start:
    case True{}: position_tail(Some{Pos.start()}, args)
    case False{}:
      match args:
        case Con{p, Con{t, Con{r, Con{e, Con{h, Con{f, tail}}}}}}:
          position_tail(Pos.fen(p, t, r, e, h, f), tail)
        case _: PositionInput{None{}, Nil{}}

def head_checked(ok: Bool, head: String, tail: List<&2, String>) -> PositionInput:
  match ok:
    case True{}: position_kind(String.eq(head, "startpos"), tail)
    case False{}: PositionInput{None{}, Nil{}}

def position_head(+head: String, tail: List<&2, String>) -> PositionInput:
  head_checked(Bool.or(String.eq(head, "startpos"), String.eq(head, "fen")), head, tail)

def parse_position(args: List<&2, String>) -> PositionInput:
  match args:
    case Con{h, t}: position_head(h, t)
    case Nil{}: PositionInput{None{}, Nil{}}

def game_valid(valid: Bool, g: Pos.Game) -> Maybe<&2, Pos.Game>:
  match valid:
    case True{}: Some{g}
    case False{}: None{}

def king_checked(g: Pos.Game, r: Array<U64> & Bool) -> Array<U64> & Maybe<&2, Pos.Game>:
  (table, illegal) = r
  (table, game_valid(Bool.not(illegal), g))

def valid_board(valid: Bool, +g: Pos.Game, table: Array<U64>) -> Array<U64> & Maybe<&2, Pos.Game>:
  match valid:
    case False{}: (table, None{})
    case True{}: king_checked(g, Chess.in_check(table, Pos.board(g), U32.xor(Chess.get_turn(Pos.board(g)), 1)))

def validate(g: Maybe<&2, Pos.Game>, table: Array<U64>) -> Array<U64> & Maybe<&2, Pos.Game>:
  match g:
    case None{}: (table, None{})
    case Some{+g}: valid_board(Pos.valid(Pos.board(g)), g, table)

def move_apply(g: Pos.Game, m: Maybe<&2, Chess.Ply>) -> Maybe<&2, Pos.Game>:
  match m:
    case None{}: None{}
    case Some{m}: Some{Pos.apply(g, m)}

def move_found(g: Pos.Game, text: String, r: Array<U64> & List<&2, Chess.Ply>) -> Array<U64> & Maybe<&2, Pos.Game>:
  (table, moves) = r
  (table, move_apply(g, Pos.find_move(moves, text)))

def apply_one(text: String, maybe: Maybe<&2, Pos.Game>, table: Array<U64>) -> Array<U64> & Maybe<&2, Pos.Game>:
  match maybe:
    case None{}: (table, None{})
    case Some{+g}: move_found(g, text, Chess.legal_moves(table, Pos.board(g)))

def moves(xs: List<&2, String>, r: Array<U64> & Maybe<&2, Pos.Game>) -> Array<U64> & Maybe<&2, Pos.Game>:
  match xs:
    case Nil{}: r
    case Con{h, t}:
      (table, g) = r
      moves(t, apply_one(h, g, table))

def limited(ok: Bool, xs: List<&2, String>, r: Array<U64> & Maybe<&2, Pos.Game>) -> Array<U64> & Maybe<&2, Pos.Game>:
  match ok:
    case False{}:
      (table, g) = r
      (table, None{})
    case True{}: moves(xs, r)

def position(input: PositionInput, table: Array<U64>) -> Array<U64> & Maybe<&2, Pos.Game>:
  PositionInput{g, +xs} = input
  limited(Nat.is_le(List.length(&2, String, xs), 512n), xs, validate(g, table))

type Limits is Data:
  Limits{nodes: U32, depth: U32, milliseconds: U32, infinite: Bool, seen: U32, valid: Bool, evals: U32}

def field(+s: String) -> U32:
  Bool.pick(U32, String.eq(s, "nodes"), 1, Bool.pick(U32, String.eq(s, "depth"), 2,
    Bool.pick(U32, String.eq(s, "movetime"), 4, Bool.pick(U32, String.eq(s, "infinite"), 8, Bool.pick(U32, String.eq(s, "evals"), 16, Bool.pick(U32, String.eq(s, "profile"), 32, 0))))))

def set_limit(+which: U32, value: Maybe<&2, U32>, lim: Limits) -> Limits:
  match value:
    case None{}:
      Limits{n, d, ms, inf, seen, valid, evals} = lim
      Limits{n, d, ms, inf, seen, False{}, evals}
    case Some{+v}:
      Limits{n, d, ms, inf, +seen, valid, evals} = lim
      max = Bool.pick(U32, U32.is_eq(which, 1), 256, Bool.pick(U32, U32.is_eq(which, 2), 32, Bool.pick(U32, U32.is_eq(which, 16), 65536, 60000)))
      ok = Bool.and(valid, Bool.and(U32.is_gt(which, 0), Bool.and(U32.is_zero(U32.and(seen, which)),
        Bool.and(U32.is_gt(v, 0), U32.is_le(v, max)))))
      Limits{Bool.pick(U32, U32.is_eq(which, 1), v, n), Bool.pick(U32, U32.is_eq(which, 2), v, d),
        Bool.pick(U32, U32.is_eq(which, 4), v, ms), inf, U32.or(seen, which), ok, Bool.pick(U32, U32.is_eq(which, 16), v, evals)}

def unlimited(l: Limits) -> Limits:
  Limits{n, d, ms, inf, +seen, valid, evals} = l
  Limits{n, d, ms, True{}, U32.or(seen, 8), Bool.and(valid, U32.is_zero(U32.and(seen, 12))), evals}

type GoState is Data:
  GoState{lim: Limits, pending: U32}

def go_field(which: U32, lim: Limits) -> GoState:
  match which:
    case 8: GoState{unlimited(lim), 0}
    case 32:
      Limits{n, d, ms, inf, +seen, valid, evals} = lim
      GoState{Limits{n, d, ms, inf, U32.or(seen, 32), Bool.and(valid, U32.is_zero(U32.and(seen, 32))), evals}, 0}
    case 0: GoState{set_limit(0, None{}, lim), 0}
    case _: GoState{lim, which}

def go_word(word: String, state: GoState) -> GoState:
  GoState{lim, pending} = state
  match pending:
    case 0: go_field(field(word), lim)
    case _: GoState{set_limit(pending, Text.number(word), lim), 0}

def measured_limits(lim: Limits) -> Limits:
  Limits{n, d, ms, inf, +seen, valid, evals} = lim
  # Bounded safety guard, NOT the requested NN-row or wall-time budget.
  measured = Bool.and(U32.is_zero(U32.and(seen, 1)), U32.is_gt(U32.and(seen, 20), 0))
  Limits{Bool.pick(U32, measured, 65536, n), d, ms, inf, seen, valid, evals}

def go_finish(s: GoState) -> Limits:
  GoState{lim, pending} = s
  match pending:
    case 0: measured_limits(lim)
    case _: set_limit(pending, None{}, lim)

def go(xs: List<&2, String>, state: GoState) -> Limits:
  match xs:
    case Nil{}: go_finish(state)
    case Con{h, t}: go(t, go_word(h, state))

def command(+s: String) -> U32:
  Bool.pick(U32, String.eq(s, "uci"), 1, Bool.pick(U32, String.eq(s, "isready"), 2,
    Bool.pick(U32, String.eq(s, "position"), 3, Bool.pick(U32, String.eq(s, "go"), 4,
      Bool.pick(U32, String.eq(s, "stop"), 5, Bool.pick(U32, String.eq(s, "quit"), 6,
        Bool.pick(U32, String.eq(s, "ucinewgame"), 7, Bool.pick(U32, String.eq(s, "d"), 8, Bool.pick(U32, String.eq(s, "perft"), 9, Bool.pick(U32, String.eq(s, "rules"), 10, Bool.pick(U32, String.eq(s, "encode_history"), 11, Bool.pick(U32, String.eq(s, "encode_input"), 12, Bool.pick(U32, String.eq(s, "policy"), 13, Bool.pick(U32, String.eq(s, "encode_request"), 14, 0))))))))))))))
