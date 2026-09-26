# Read-only complete-input inspection, never an inference interface or benchmark.
import Base
import ./Position.bend as Pos
import ./Protocol.bend as Protocol
import ./HistoryEncoding.bend as H
import ./EncodingDiagnostic.bend as Rows
import ./ModelInput.bend as Input

def emit_row(index: U32, r: Array<F32> & String) -> IO(Array<F32>):
  (a, text) = r
  do IO<Array<F32>>:
    IO.print("info string input_plane " ++ U32.show(index) ++ text)
    return a

def emit(fuel: Nat, +plane: U32, a: Array<F32>) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string model_input_end")
    case 1n+p:
      do IO<Unit>:
        a : Array<F32> <- emit_row(plane, Rows.row(64n, U32.mul(plane, 64), (a, "")))
        emit(p, U32.inc(plane), a)

def encoded(layout: H.Layout, version: Input.Version, input: Input.Input) -> IO(Array<U64>):
  Input.Input{table, values, +channels} = input
  do IO<Array<U64>>:
    IO.print("info string model_input " ++ H.name(layout) ++ " " ++ Input.name(version)
      ++ " " ++ U32.show(channels) ++ " 8 8 repfix=1")
    emit(U32.to_nat(channels), 0, values)
    return table

def error(table: Array<U64>) -> IO(Array<U64>):
  do IO<Array<U64>>:
    IO.print("info string encode_input requires lc0_root|lc0_root_legacy_meta v1|v2_threats and at most 32 legal moves; root preserved")
    return table

def replayed(+layout: H.Layout, +version: Input.Version, r: Array<U64> & Maybe<&2, Pos.Game>) -> IO(Array<U64>):
  (table, game) = r
  match game:
    case None{}: error(table)
    case Some{game}: encoded(layout, version, Input.encode(layout, version, game, table))

def path(layout: H.Layout, version: Input.Version, input: Protocol.PositionInput, table: Array<U64>) -> IO(Array<U64>):
  Protocol.PositionInput{game, +moves} = input
  replayed(layout, version, Protocol.limited(Nat.is_le(List.length(&2, String, moves), 32n), moves, (table, game)))

def version(v: Maybe<&2, Input.Version>, layout: H.Layout, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match v:
    case None{}: error(table)
    case Some{v}: path(layout, v, Protocol.position_tail(Some{root}, args), table)

def layout(l: Maybe<&2, H.Layout>, v: String, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match l:
    case None{}: error(table)
    case Some{l}: version(Input.parse(v), l, args, root, table)

def arguments(args: List<&2, String>, l: String, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match args:
    case Nil{}: error(table)
    case Con{v, rest}: layout(H.parse(l), v, rest, root, table)

def run(args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match args:
    case Nil{}: error(table)
    case Con{l, rest}: arguments(rest, l, root, table)
