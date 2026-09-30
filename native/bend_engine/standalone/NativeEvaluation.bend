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
import ./NeuralWork.bend as Work

import ./ModelBuffers.bend as Buffers

# Linear, process-lifetime storage, carried through State even across position,
# ucinewgame and stop. It is not part of the mutable search tree.
type Cache is Type:
  Disabled{}
  Ready{maps: Policy.Maps, input: Array<F32>, output: Array<F32>, diagnostics: Bool}

def cache(model: U32, diagnostics: Bool) -> Cache:
  match model:
    case 0: Disabled{}
    case _: Ready{Policy.build(), Array.new(F32, 14n, 0.0), Array.new(F32, 11n, 0.0), diagnostics}

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

def notice_enabled(diagnostics: Bool, root: Pos.Game, game: Pos.Game, entries: List<&2, Policy.Entry>,
  table: Array<U64>, epoch: U32, request: U32, node: U32, win: F32, draw: F32, loss: F32,
  count: U32, policy: Array<F32>) -> IO(Array<U64> & Search.Reply):
  match diagnostics:
    case True{}: notice_render(root, game, table, epoch, request, node, win, draw, loss, count, priors(entries, 0, (policy, "")))
    case False{}: IO.pure(Array<U64> & Search.Reply, (table, Search.Reply{epoch, request, node, 0, win, draw, loss, count, policy}))

def notice(root: Pos.Game, game: Pos.Game, entries: List<&2, Policy.Entry>,
  table: Array<U64>, diagnostics: Bool, reply: Search.Reply) -> IO(Array<U64> & Search.Reply):
  Search.Reply{epoch, request, node, status, win, draw, loss, count, policy} = reply
  match status:
    case 0: notice_enabled(diagnostics, root, game, entries, table, epoch, request, node, win, draw, loss, count, policy)
    case _: IO.die(Array<U64> & Search.Reply, 2, "native model output or legal policy contract failed; no material fallback")

type Result is Type:
  Result{cache: Cache, table: Array<U64>, reply: Search.Reply, encoding_ms: Nat, backend_ms: Nat, post_ms: Nat}

def material(cache: Cache, r: Array<U64> & Search.Reply) -> Result:
  (table, reply) = r
  Result{cache, table, reply, 0n, 0n, 0n}

def timed_result(cache: Cache, started: Nat, +encoded: Nat, +forwarded: Nat, finished: Nat,
  r: Array<U64> & Search.Reply) -> Result:
  (table, reply) = r
  Result{cache, table, reply, Nat.sub(encoded, started), Nat.sub(forwarded, encoded), Nat.sub(finished, forwarded)}

def converted(+profile: Bool, +diagnostics: Bool, started: Nat, encoded: Nat, forwarded: Nat,
  root: Pos.Game, game: Pos.Game, entries: List<&2, Policy.Entry>, maps: Policy.Maps,
  table: Array<U64>, input: Array<F32>, c: Logits.Converted) -> IO(Result):
  Logits.Converted{output, reply} = c
  do IO<Result>:
    r : Array<U64> & Search.Reply <- notice(root, game, entries, table, diagnostics, reply)
    finished : Nat <- Work.clock(profile)
    return timed_result(Ready{maps, input, output, diagnostics}, started, encoded, forwarded, finished, r)

def forwarded(+profile: Bool, diagnostics: Bool, started: Nat, encoded: Nat,
  root: Pos.Game, game: Pos.Game, ticket: Search.Ticket, +entries: List<&2, Policy.Entry>,
  maps: Policy.Maps, table: Array<U64>, b: Buffers.Buffers) -> IO(Result):
  Buffers.Buffers{input, output, size} = b
  do IO<Result>:
    now : Nat <- Work.clock(profile)
    converted(profile, diagnostics, started, encoded, now, root, game, entries, maps, table,
      input, Logits.convert_buffer(ticket, entries, size, Array.size(F32, output)))

def prepared(+profile: Bool, diagnostics: Bool, started: Nat, root: Pos.Game, game: Pos.Game,
  ticket: Search.Ticket, output: Array<F32>, p: Evaluation.Preparation) -> IO(Result):
  match p:
    case Evaluation.Invalid{table, maps}: IO.die(Result, 2, "native input preparation failed")
    case Evaluation.Ready{Evaluation.Prepared{maps, Input.Input{table, values, +channels}, +entries}}:
      count = U32.mul(channels, 64)
      do IO<Result>:
        encoded : Nat <- Work.clock(profile)
        buffers : Buffers.Buffers <- Buffers.Native.run(count, values, output)
        forwarded(profile, diagnostics, started, encoded, root, game, ticket, entries, maps, table, buffers)

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

def evaluate(cache: Cache, +timing: Bool, +profile: U32, root: Pos.Game, +game: Pos.Game, ticket: Search.Ticket,
  table: Array<U64>) -> IO(Result):
  match cache:
    case Disabled{}: IO.die(Result, 2, "native cache is not initialized")
    case Ready{maps, input, output, diagnostics}:
      do IO<Result>:
        now : Nat <- Work.clock(timing)
        prepared(timing, diagnostics, now, root, game, ticket, output,
          Evaluation.prepare_into(layout(profile), version(profile), game, table, maps, Array.size(F32, input)))
