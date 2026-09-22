# Complete dense float32 input shapes; physical array capacity is not a shape.
# No policy mapping or model execution is implied by constructing input planes.
import Base
import ../legal_probe/Chess.bend as Chess
import ./Position.bend as Pos
import ./HistoryEncoding.bend as H
import ./ClassicalEncoding.bend as Features

type Version is Data:
  V1{}
  V2Threats{}

def channels(version: Version) -> U32:
  match version:
    case V1{}: 146
    case V2Threats{}: 175

def threats(version: Version) -> Bool:
  match version:
    case V1{}: False{}
    case V2Threats{}: True{}

def name(version: Version) -> String:
  match version:
    case V1{}: "v1"
    case V2Threats{}: "v2_threats"

def parse_v2(ok: Bool) -> Maybe<&2, Version>:
  match ok:
    case True{}: Some{V2Threats{}}
    case False{}: None{}

def parse_v1(ok: Bool, s: String) -> Maybe<&2, Version>:
  match ok:
    case True{}: Some{V1{}}
    case False{}: parse_v2(String.eq(s, "v2_threats"))

def parse(+s: String) -> Maybe<&2, Version>:
  parse_v1(String.eq(s, "v1"), s)

type Input is Type:
  Input{table: Array<U64>, values: Array<F32>, channels: U32}

def copied(i: U32, dest: Array<F32>, src: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (src, value) = src
  (src, Array.set(F32, dest, i, value))

def copy(fuel: Nat, +i: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match fuel:
    case 0n: r
    case 1n+p:
      (src, dest) = r
      copy(p, U32.inc(i), copied(i, dest, Array.get(F32, src, i)))

def result(channels: U32, e: H.Encoded) -> Input:
  H.Encoded{table, values} = e
  Input{table, values, channels}

def features(+version: Version, b: Chess.Board, table: Array<U64>, arrays: Array<F32> & Array<F32>) -> Input:
  (history, out) = arrays
  result(channels(version), Features.encode(threats(version), b, table, out))

def history(version: Version, b: Chess.Board, e: H.Encoded) -> Input:
  H.Encoded{table, src} = e
  features(version, b, table, copy(7168n, 0, (src, Array.new(F32, 14n, 0.0))))

def encode(layout: H.Layout, version: Version, +game: Pos.Game, table: Array<U64>) -> Input:
  history(version, Pos.board(game), H.encode(layout, game, table))

# Native reusable storage is cleared, including unused capacity, before every
# position. No feature is allowed to inherit an unwritten value from a prior leaf.
def cleared(fuel: Nat, +i: U32, out: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: out
    case 1n+p: cleared(p, U32.inc(i), Array.set(F32, out, i, 0.0))

def history_into(version: Version, b: Chess.Board, out: Array<F32>, e: H.Encoded) -> Input:
  H.Encoded{table, src} = e
  features(version, b, table, copy(7168n, 0, (src, out)))

# Caller owns the fixed 2^14-element workspace; Native.run independently checks
# its physical capacity. Diagnostics/reference encoding retain encode above.
def encode_into(layout: H.Layout, version: Version, +game: Pos.Game,
  table: Array<U64>, out: Array<F32>) -> Input:
  history_into(version, Pos.board(game), cleared(16384n, 0, out), H.encode(layout, game, table))
