# Read-only composition gate: exact legal policy entries and complete model input
# from one validated Game. This text format is diagnostic, not an inference ABI.
import Base
import ./Position.bend as Pos
import ./Protocol.bend as Protocol
import ./HistoryEncoding.bend as H
import ./ModelInput.bend as Input
import ./ModelInputDiagnostic.bend as InputDiagnostic
import ./Policy.bend as Policy
import ./PolicyDiagnostic.bend as PolicyDiagnostic
import ./EvaluationInput.bend as Evaluation

def prepared(+layout: H.Layout, +version: Input.Version, r: Evaluation.Preparation) -> IO(Array<U64>):
  match r:
    case Evaluation.Invalid{table, maps}: IO.die(Array<U64>, 2, "evaluation input policy invariant failed")
    case Evaluation.Ready{Evaluation.Prepared{maps, input, +entries}}:
      do IO<Array<U64>>:
        IO.print("info string evaluation_request " ++ H.name(layout) ++ " " ++ Input.name(version)
          ++ " lc0_1858=1858 legal=" ++ U32.show(U32.from_nat(List.length(&2, Policy.Entry, entries))))
        PolicyDiagnostic.emit_entries(entries)
        table : Array<U64> <- InputDiagnostic.encoded(layout, version, input)
        IO.print("info string evaluation_request_end")
        return table

def error(table: Array<U64>) -> IO(Array<U64>):
  do IO<Array<U64>>:
    IO.print("info string encode_request requires lc0_root|lc0_root_legacy_meta v1|v2_threats and at most 32 legal moves; root preserved")
    return table

def replayed(+layout: H.Layout, +version: Input.Version,
  r: Array<U64> & Maybe<&2, Pos.Game>) -> IO(Array<U64>):
  (table, game) = r
  match game:
    case None{}: error(table)
    case Some{game}: prepared(layout, version, Evaluation.prepare(layout, version, game, table, Policy.build()))

def path(layout: H.Layout, version: Input.Version, request: Protocol.PositionInput, table: Array<U64>) -> IO(Array<U64>):
  Protocol.PositionInput{game, +moves} = request
  replayed(layout, version, Protocol.limited(Nat.is_le(List.length(&2, String, moves), 32n), moves, (table, game)))

def version(v: Maybe<&2, Input.Version>, layout: H.Layout, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match v:
    case None{}: error(table)
    case Some{v}: path(layout, v, Protocol.position_tail(Some{root}, args), table)

def layout(l: Maybe<&2, H.Layout>, v: String, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match l:
    case None{}: error(table)
    case Some{l}: version(Input.parse(v), l, args, root, table)

def run(args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(Array<U64>):
  match args:
    case Con{l, Con{v, rest}}: layout(H.parse(l), v, rest, root, table)
    case _: error(table)
