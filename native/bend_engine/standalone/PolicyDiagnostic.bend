# Read-only policy boundary inspection. Maps are generated in Bend, not supplied
# by a Python client. Each diagnostic owns disposable maps; default search is unchanged.
import Base
import ../legal_probe/Chess.bend as Chess
import ../session_probe/Search.bend as Search
import ./Policy.bend as Policy
import ./Position.bend as Pos
import ./Text.bend as Text

type Request is Data:
  Request{kind: U32, index: U32, text: String}

# kind: 0=invalid, 1=tables, 2=legal, 3=full decode, 4=compact decode,
# 5=exact private key, 6=exact legal UCI text.
def bare(head: String) -> Request:
  +head = head
  Request{Bool.pick(U32, String.eq(head, "tables"), 1,
    Bool.pick(U32, String.eq(head, "legal"), 2, 0)), 0, ""}

def numbered(kind: U32, value: Maybe<&2, U32>) -> Request:
  match value:
    case None{}: Request{0, 0, ""}
    case Some{n}: Request{kind, n, ""}

def decode_space(+space: String, number: String) -> Request:
  numbered(Bool.pick(U32, String.eq(space, "az_4672"), 3,
    Bool.pick(U32, String.eq(space, "lc0_1858"), 4, 0)), Text.number(number))

def one_arg_kind(encode: Bool, head: String, arg: String) -> Request:
  match encode:
    case True{}: Request{6, 0, arg}
    case False{}: numbered(Bool.pick(U32, String.eq(head, "key"), 5, 0), Text.number(arg))

def one_arg(+head: String, arg: String) -> Request:
  one_arg_kind(String.eq(head, "encode"), head, arg)

def two_args(decode: Bool, space: String, number: String) -> Request:
  match decode:
    case True{}: decode_space(space, number)
    case False{}: Request{0, 0, ""}

def parse(args: List<&2, String>) -> Request:
  match args:
    case Con{head, Nil{}}: bare(head)
    case Con{head, Con{arg, Nil{}}}: one_arg(head, arg)
    case Con{head, Con{space, Con{number, Nil{}}}}: two_args(String.eq(head, "decode"), space, number)
    case _: Request{0, 0, ""}

def full_row(+index: U32, r: Policy.Maps & U32) -> IO(Policy.Maps):
  (maps, compact) = r
  do IO<Policy.Maps>:
    IO.print("info string policy_full " ++ U32.show(index) ++ " " ++ U32.show(compact)
      ++ " " ++ U32.show(Policy.mirror_full(index)))
    return maps

def full_rows(fuel: Nat, +i: U32, maps: Policy.Maps) -> IO(Policy.Maps):
  match fuel:
    case 0n: IO.pure(Policy.Maps, maps)
    case 1n+p:
      do IO<Policy.Maps>:
        maps : Policy.Maps <- full_row(i, Policy.to_compact(i, maps))
        full_rows(p, U32.inc(i), maps)

def compact_row(index: U32, full: U32, r: Policy.Maps & U32) -> IO(Policy.Maps):
  (maps, mirrored) = r
  do IO<Policy.Maps>:
    IO.print("info string policy_compact " ++ U32.show(index) ++ " " ++ U32.show(full)
      ++ " " ++ U32.show(mirrored))
    return maps

def compact_found(index: U32, r: Policy.Maps & U32) -> IO(Policy.Maps):
  (maps, +full) = r
  compact_row(index, full, Policy.to_compact(Policy.mirror_full(full), maps))

def compact_rows(fuel: Nat, +i: U32, maps: Policy.Maps) -> IO(Policy.Maps):
  match fuel:
    case 0n: IO.pure(Policy.Maps, maps)
    case 1n+p:
      do IO<Policy.Maps>:
        maps : Policy.Maps <- compact_found(i, Policy.to_full(i, maps))
        compact_rows(p, U32.inc(i), maps)

def pair_row(index: U32, r: Array<U32> & U32) -> IO(Array<U32>):
  (pairs, full) = r
  do IO<Array<U32>>:
    IO.print("info string policy_pair " ++ U32.show(index) ++ " " ++ U32.show(full))
    return pairs

def pair_rows(fuel: Nat, +i: U32, pairs: Array<U32>) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string policy_end")
    case 1n+p:
      do IO<Unit>:
        pairs : Array<U32> <- pair_row(i, Array.get(U32, pairs, i))
        pair_rows(p, U32.inc(i), pairs)

def finish_tables(maps: Policy.Maps) -> IO(Unit):
  Policy.Maps{f2c, c2f, pairs, n} = maps
  pair_rows(4096n, 0, pairs)

def tables(maps: Policy.Maps) -> IO(Unit):
  do IO<Unit>:
    IO.print("info string policy_tables az_4672=4672 lc0_1858=1858 invalid=4294967295")
    maps : Policy.Maps <- full_rows(4672n, 0, maps)
    maps : Policy.Maps <- compact_rows(1858n, 0, maps)
    finish_tables(maps)

def entry_text(e: Policy.Entry) -> String:
  Policy.Entry{+move, full, compact} = e
  "info string policy_move " ++ Pos.move_text(move) ++ " " ++ U32.show(Search.key(move))
    ++ " " ++ U32.show(full) ++ " " ++ U32.show(compact)

def emit_entries(entries: List<&2, Policy.Entry>) -> IO(Unit):
  match entries:
    case Nil{}: IO.print("info string policy_end")
    case Con{entry, rest}:
      do IO<Unit>:
        IO.print(entry_text(entry))
        emit_entries(rest)

def key_match(+key: U32, entries: List<&2, Policy.Entry>) -> Maybe<&2, Policy.Entry>:
  match entries:
    case Nil{}: None{}
    case Con{+entry, rest}:
      Policy.Entry{move, full, compact} = entry
      Policy.chosen(U32.is_eq(Search.key(move), key), entry, key_match(key, rest))

def text_match(+text: String, entries: List<&2, Policy.Entry>) -> Maybe<&2, Policy.Entry>:
  match entries:
    case Nil{}: None{}
    case Con{+entry, rest}:
      Policy.Entry{move, full, compact} = entry
      Policy.chosen(String.eq(Pos.move_text(move), text), entry, text_match(text, rest))

def error() -> IO(Unit):
  IO.print("info string policy error: invalid request or no exact legal move; root preserved")

def selected(entry: Maybe<&2, Policy.Entry>) -> IO(Unit):
  match entry:
    case None{}: error()
    case Some{entry}:
      do IO<Unit>:
        IO.print(entry_text(entry))
        IO.print("info string policy_end")

def respond(request: Request, +entries: List<&2, Policy.Entry>) -> IO(Unit):
  Request{kind, index, text} = request
  match kind:
    case 2:
      do IO<Unit>:
        IO.print("info string policy_legal " ++ U32.show(U32.from_nat(List.length(&2, Policy.Entry, entries))))
        emit_entries(entries)
    case 3: selected(Policy.resolve(Policy.Full{}, index, entries))
    case 4: selected(Policy.resolve(Policy.Compact{}, index, entries))
    case 5: selected(key_match(index, entries))
    case 6: selected(text_match(text, entries))
    case _: error()

def mapped(request: Request, r: Policy.Maps & Maybe<&2, List<&2, Policy.Entry>>) -> IO(Unit):
  (maps, entries) = r
  match entries:
    case None{}: IO.die(Unit, 2, "policy mapping invariant failed")
    case Some{entries}: respond(request, entries)

def generated(request: Request, turn: U32, maps: Policy.Maps, r: Array<U64> & List<&2, Chess.Ply>) -> IO(Array<U64>):
  (table, moves) = r
  do IO<Array<U64>>:
    mapped(request, Policy.entries(turn, moves, maps))
    return table

def dispatch(+request: Request, +board: Chess.Board, table: Array<U64>, maps: Policy.Maps) -> IO(Array<U64>):
  Request{kind, index, text} = request
  match kind:
    case 0:
      do IO<Array<U64>>:
        error()
        return table
    case 1:
      do IO<Array<U64>>:
        tables(maps)
        return table
    case _: generated(request, Chess.get_turn(board), maps, Chess.legal_moves(table, board))

def checked(valid: Bool, request: Request, board: Chess.Board, table: Array<U64>, maps: Policy.Maps) -> IO(Array<U64>):
  match valid:
    case False{}: IO.die(Array<U64>, 2, "policy vocabulary count mismatch")
    case True{}: dispatch(request, board, table, maps)

def prepared(request: Request, board: Chess.Board, table: Array<U64>, maps: Policy.Maps) -> IO(Array<U64>):
  Policy.Maps{f2c, c2f, pairs, +n} = maps
  checked(U32.is_eq(n, 1858), request, board, table, Policy.Maps{f2c, c2f, pairs, n})

def run(args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  Pos.Game{board, half, full, history} = root
  prepared(parse(args), board, table, Policy.build())
