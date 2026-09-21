# Runtime application work is Bend-owned; the foreign effect executes only a
# fixed, pre-exported CPU F32 model and transports its raw values. No Python.
import Base
import ../session_probe/Search.bend as Search
import ./Position.bend as Pos
import ./Policy.bend as Policy
import ./HistoryEncoding.bend as H
import ./ModelInput.bend as Input
import ./EvaluationInput.bend as Evaluation
import ./LogitReply.bend as Logits

# Zero is the explicitly material-only build. A native build must load and
# validate its bound model before returning profile 1..4; errors are fatal.
def Native.open() -> IO(U32):
  import "./model_call.c"

def Native.run(input: List<F32>) -> IO(List<F32>):
  import "./model_call.c"

def append(xs: List<F32>, r: Array<F32> & F32) -> Array<F32> & List<F32>:
  (a, x) = r
  (a, Con{x, xs})

def flatten(fuel: Nat, +left: U32, r: Array<F32> & List<F32>) -> List<F32>:
  match fuel:
    case 0n:
      (a, xs) = r
      xs
    case 1n+p:
      (a, xs) = r
      flatten(p, U32.sub(left, 1), append(xs, Array.get(F32, a, U32.sub(left, 1))))

# This notice reports the actual selected path and normalized reply, not a
# request supplied by the Python verifier. Full raw model values may be recorded
# by the optional native trace sink outside the engine's normal execution path.
def path(fuel: Nat, hist: List<&2, Pos.Past>) -> String:
  match fuel:
    case 0n: ""
    case 1n+p:
      match hist:
        case Nil{}: ""
        case Con{Pos.Past{b, half, full, m}, rest}: path(p, rest) ++ " " ++ U32.show(Search.key(m))

def history_length(g: Pos.Game) -> Nat:
  Pos.Game{b, half, full, hist} = g
  List.length(&2, Pos.Past, hist)

def notice_path(root: Pos.Game, game: Pos.Game) -> String:
  Pos.Game{b, half, full, +hist} = game
  path(Nat.sub(List.length(&2, Pos.Past, hist), history_length(root)), hist)

def prior_seen(key: U32, text: String, r: Array<F32> & F32) -> Array<F32> & String:
  (a, v) = r
  (a, text ++ " " ++ U32.show(key) ++ " " ++ U32.show(F32.bits(v)))

def priors(entries: List<&2, Policy.Entry>, +i: U32, r: Array<F32> & String) -> Array<F32> & String:
  match entries:
    case Nil{}: r
    case Con{Policy.Entry{m, full, compact}, rest}:
      (a, text) = r
      priors(rest, U32.inc(i), prior_seen(Search.key(m), text, Array.get(F32, a, i)))

def notice_render(root: Pos.Game, game: Pos.Game, table: Array<U64>, +epoch: U32,
  +request: U32, +node: U32, +win: F32, +draw: F32, +loss: F32, +count: U32,
  r: Array<F32> & String) -> IO(Array<U64> & Search.Reply):
  (policy, text) = r
  do IO<Array<U64> & Search.Reply>:
    IO.print("info string native_path " ++ U32.show(epoch) ++ " " ++ U32.show(request) ++ " " ++ U32.show(node) ++ notice_path(root, game))
    IO.print("info string native_reply " ++ U32.show(epoch) ++ " " ++ U32.show(request) ++ " " ++ U32.show(node)
      ++ " " ++ U32.show(F32.bits(win)) ++ " " ++ U32.show(F32.bits(draw)) ++ " " ++ U32.show(F32.bits(loss))
      ++ " " ++ U32.show(count) ++ text)
    return (table, Search.Reply{epoch, request, node, 0, win, draw, loss, count, policy})

def notice(root: Pos.Game, game: Pos.Game, entries: List<&2, Policy.Entry>,
  table: Array<U64>, reply: Search.Reply) -> IO(Array<U64> & Search.Reply):
  Search.Reply{epoch, request, node, status, win, draw, loss, count, policy} = reply
  match status:
    case 0: notice_render(root, game, table, epoch, request, node, win, draw, loss, count, priors(entries, 0, (policy, "")))
    case _: IO.die(Array<U64> & Search.Reply, 2, "native model output or legal policy contract failed; no material fallback")

def prepared(root: Pos.Game, game: Pos.Game, ticket: Search.Ticket, p: Evaluation.Preparation) -> IO(Array<U64> & Search.Reply):
  match p:
    case Evaluation.Invalid{table, maps}: IO.die(Array<U64> & Search.Reply, 2, "native input preparation failed")
    case Evaluation.Ready{Evaluation.Prepared{maps, Input.Input{table, values, +channels}, +entries}}:
      +count = U32.mul(channels, 64)
      do IO<Array<U64> & Search.Reply>:
        output : List<F32> <- Native.run(flatten(U32.to_nat(count), count, (values, Nil{})))
        notice(root, game, entries, table, Logits.convert(ticket, entries, output))

def layout(profile: U32) -> H.Layout:
  match profile:
    case 1: H.Root{}
    case 2: H.RootLegacyMeta{}
    case 3: H.Root{}
    case _: H.RootLegacyMeta{}

def version(profile: U32) -> Input.Version:
  match profile:
    case 1: Input.V1{}
    case 2: Input.V1{}
    case _: Input.V2Threats{}

def evaluate(+profile: U32, root: Pos.Game, +game: Pos.Game, ticket: Search.Ticket,
  table: Array<U64>) -> IO(Array<U64> & Search.Reply):
  prepared(root, game, ticket, Evaluation.prepare(layout(profile), version(profile), game, table, Policy.build()))
