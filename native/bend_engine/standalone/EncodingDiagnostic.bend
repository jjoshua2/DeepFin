# Read-only inspection adapter. All plane values originate in HistoryEncoding.
# This diagnostic never claims to supply missing v1/v2 classical feature planes.
import Base
import ./HistoryEncoding.bend as Encoding
import ./Position.bend as Pos
import ./Protocol.bend as Protocol

# Array ownership stays threaded; serialize F32 bits rather than rounded decimals.
def appended(text: String, r: Array<F32> & F32) -> Array<F32> & String:
  (a, v) = r
  (a, text ++ " " ++ U32.show(F32.bits(v)))

def row(fuel: Nat, +i: U32, r: Array<F32> & String) -> Array<F32> & String:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, text) = r
      row(p, U32.inc(i), appended(text, Array.get(F32, a, i)))

def emit_row(index: U32, r: Array<F32> & String) -> IO(Array<F32>):
  (a, text) = r
  do IO<Array<F32>>:
    IO.print("info string history_plane " ++ U32.show(index) ++ text)
    return a

def emit_rows(fuel: Nat, +plane: U32, a: Array<F32>) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string history_encoding_end")
    case 1n+p:
      do IO<Unit>:
        a : Array<F32> <- emit_row(plane, row(64n, U32.mul(plane, 64), (a, "")))
        emit_rows(p, U32.inc(plane), a)

def encoded(layout: Encoding.Layout, e: Encoding.Encoded) -> IO(Array<U64>):
  Encoding.Encoded{table, values} = e
  do IO<Array<U64>>:
    IO.print("info string history_encoding " ++ Encoding.name(layout) ++ " 112 8 8 repfix=1 partial_input")
    emit_rows(112n, 0, values)
    return table

def error(table: Array<U64>) -> IO(Array<U64>):
  do IO<Array<U64>>:
    IO.print("info string encode_history requires lc0_root or lc0_root_legacy_meta and at most 32 legal moves; root preserved")
    return table

def replayed(+layout: Encoding.Layout, r: Array<U64> & Maybe<&2, Pos.Game>) -> IO(Array<U64>):
  (table, game) = r
  match game:
    case None{}: error(table)
    case Some{game}: encoded(layout, Encoding.encode(layout, game, table))

def path(layout: Encoding.Layout, input: Protocol.PositionInput, table: Array<U64>) -> IO(Array<U64>):
  Protocol.PositionInput{game, +moves} = input
  replayed(layout, Protocol.limited(Nat.is_le(List.length(&2, String, moves), 32n), moves, (table, game)))

def selected(layout: Maybe<&2, Encoding.Layout>, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match layout:
    case None{}: error(table)
    case Some{layout}: path(layout, Protocol.position_tail(Some{root}, args), table)

def run(args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match args:
    case Nil{}: error(table)
    case Con{layout, tail}: selected(Encoding.parse(layout), tail, root, table)
