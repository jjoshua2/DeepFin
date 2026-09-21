# A build-bound CPU model supplies raw logits only. All input/policy preparation,
# matching, probability conversion, and Search.Reply construction happen in Bend.
import Base
import ../../session_probe/Search.bend as Search
import ../Position.bend as Pos
import ../Policy.bend as Policy
import ../HistoryEncoding.bend as H
import ../ModelInput.bend as Input
import ../EvaluationInput.bend as Evaluation
import ./Response.bend as Response

def DfModel.open() -> IO(List<&2, U32>):
  import "./native_call.c"

def DfModel.infer(sequence: U32, words: List<&2, U32>) -> IO(List<&2, U32>):
  import "./native_call.c"

def DfModel.close() -> IO(Unit):
  import "./native_call.c"

type Context is Type:
  Context{maps: Policy.Maps, layout: H.Layout, version: Input.Version, next: U32, trace: Bool}

def config_ok(ok: Bool, history: U32, features: U32, trace: U32) -> IO(Context):
  match ok:
    case False{}: IO.die(Context, 2, "unsupported native model configuration")
    case True{}:
      IO.pure(Context, Context{Policy.build(), Bool.pick(H.Layout, U32.is_zero(history), H.Root{}, H.RootLegacyMeta{}),
        Bool.pick(Input.Version, U32.is_zero(features), Input.V1{}, Input.V2Threats{}), 1, U32.is_eq(trace, 1)})

def config(words: List<&2, U32>) -> IO(Context):
  match words:
    case Con{1, Con{+h, Con{+v, Con{ch, Con{+b, Con{1858, Con{1, Con{+trace, Nil{}}}}}}}}}:
      ok = Bool.and(U32.is_le(h, 1), Bool.and(U32.is_le(v, 1), Bool.and(U32.is_le(trace, 1),
        Bool.and(U32.is_eq(ch, Bool.pick(U32, U32.is_zero(v), 146, 175)),
          Bool.or(U32.is_eq(b, 1), Bool.or(U32.is_eq(b, 2), Bool.or(U32.is_eq(b, 4), Bool.or(U32.is_eq(b, 8), U32.is_eq(b, 16)))))))))
      config_ok(ok, h, v, trace)
    case _: IO.die(Context, 2, "native model startup failed; no fallback")

def start() -> IO(Context):
  do IO<Context>:
    words : List<&2, U32> <- DfModel.open()
    config(words)

def serialized(acc: List<&2, U32>, r: Array<F32> & F32) -> Array<F32> & List<&2, U32>:
  (a, v) = r
  (a, Con{F32.bits(v), acc})

def serialize(fuel: Nat, +i: U32, r: Array<F32> & List<&2, U32>) -> List<&2, U32>:
  match fuel:
    case 0n:
      (a, acc) = r
      List.reverse(&2, U32, acc)
    case 1n+p:
      (a, acc) = r
      serialize(p, U32.inc(i), serialized(acc, Array.get(F32, a, i)))

# Trace is optional, read-only, and absent from normal model evaluation.
def trace_one(entry: Policy.Entry, r: Array<F32> & F32) -> IO(Array<F32>):
  Policy.Entry{move, full, compact} = entry
  (a, value) = r
  do IO<Array<F32>>:
    IO.print("info string model_prior " ++ U32.show(Search.key(move)) ++ " " ++ U32.show(full)
      ++ " " ++ U32.show(compact) ++ " " ++ U32.show(F32.bits(value)))
    return a

def trace_priors(entries: List<&2, Policy.Entry>, +i: U32, a: Array<F32>) -> IO(Array<F32>):
  match entries:
    case Nil{}: IO.pure(Array<F32>, a)
    case Con{entry, rest}:
      do IO<Array<F32>>:
        a : Array<F32> <- trace_one(entry, Array.get(F32, a, i))
        trace_priors(rest, U32.inc(i), a)

def trace_reply(enabled: Bool, sequence: U32, entries: List<&2, Policy.Entry>, r: Search.Reply) -> IO(Search.Reply):
  match enabled:
    case False{}: IO.pure(Search.Reply, r)
    case True{}:
      Search.Reply{+epoch, +request, +node, status, +win, +draw, +loss, +count, priors} = r
      do IO<Search.Reply>:
        IO.print("info string model_reply " ++ U32.show(sequence) ++ " " ++ U32.show(epoch)
          ++ " " ++ U32.show(request) ++ " " ++ U32.show(node) ++ " " ++ U32.show(count)
          ++ " " ++ U32.show(F32.bits(win)) ++ " " ++ U32.show(F32.bits(draw)) ++ " " ++ U32.show(F32.bits(loss)))
        priors : Array<F32> <- trace_priors(entries, 0, priors)
        IO.print("info string model_reply_end")
        return Search.Reply{epoch, request, node, status, win, draw, loss, count, priors}

type Outcome is Type:
  Outcome{table: Array<U64>, context: Context, reply: Search.Reply}

def converted(table: Array<U64>, context: Context, entries: List<&2, Policy.Entry>, r: Response.Conversion) -> IO(Outcome):
  Context{maps, layout, version, +seq, +trace} = context
  match r:
    case Response.Rejected{}: IO.die(Outcome, 2, "invalid native model response; no search fallback")
    case Response.Accepted{reply}:
      do IO<Outcome>:
        reply : Search.Reply <- trace_reply(trace, seq, entries, reply)
        return Outcome{table, Context{maps, layout, version, U32.inc(seq), trace}, reply}

def infer(ok: Bool, ticket: Search.Ticket, context: Context, table: Array<U64>,
  values: Array<F32>, count: U32, +entries: List<&2, Policy.Entry>) -> IO(Outcome):
  match ok:
    case False{}: IO.die(Outcome, 2, "native model input/ticket mismatch")
    case True{}:
      Context{maps, layout, version, +seq, trace} = context
      do IO<Outcome>:
        words : List<&2, U32> <- DfModel.infer(seq, serialize(U32.to_nat(count), 0, (values, Nil{})))
        converted(table, Context{maps, layout, version, seq, trace}, entries, Response.convert(ticket, entries, words))

def prepared(+ticket: Search.Ticket, +layout: H.Layout, +version: Input.Version,
  +seq: U32, trace: Bool, p: Evaluation.Preparation) -> IO(Outcome):
  Search.Ticket{epoch, request, node, board, moves, needed} = ticket
  match p:
    case Evaluation.Invalid{table, maps}: IO.die(Outcome, 2, "native legal-policy construction failed")
    case Evaluation.Ready{Evaluation.Prepared{maps, Input.Input{table, values, +channels}, +entries}}:
      ok = Bool.and(U32.is_gt(seq, 0), Bool.and(U32.is_lt(seq, 4294967295),
        Bool.and(U32.is_eq(channels, Input.channels(version)), Response.matching(moves, entries))))
      infer(ok, ticket, Context{maps, layout, version, seq, trace}, table, values, U32.mul(channels, 64), entries)

def evaluate(game: Pos.Game, ticket: Search.Ticket, table: Array<U64>, context: Context) -> IO(Outcome):
  Context{maps, +layout, +version, seq, trace} = context
  prepared(ticket, layout, version, seq, trace, Evaluation.prepare(layout, version, game, table, maps))
