# Opt-in bounded cohort runner: actual independent trees, actual gathered leaves.
# CPU fixed-batch model only in the build script. This is not the UCI product.
import Base
import ./Cohort.bend as C
import ./AsyncRun.bend as AsyncRun
import ../batch_backend/Async.bend as Async
import ./Report.bend as R
import ../batch_backend/Batch.bend as B
import ../standalone/Tables.bend as Tables
import ../standalone/Policy.bend as Policy
import ../standalone/Position.bend as Pos
import ../standalone/Protocol.bend as Protocol
import ../standalone/Text.bend as Text
import ../standalone/NeuralWork.bend as W
import ../session_probe/Search.bend as S

type Metrics is Data:
  Metrics{started: Nat, calls: U32, rows: U32, gather_ms: Nat, backend_ms: Nat, scatter_ms: Nat}

type State is Type:
  State{roots: List<C.Root>, store: C.Store, config: C.Config, metrics: Metrics, histogram: Array<U32>, finished: Bool}

def increment(r: Array<U32> & U32, at: U32) -> Array<U32>:
  (a,n) = r
  Array.set(U32,a,at,U32.inc(n))

def histogram_seen(+key: U32, +text: String, r: Array<U32> & U32) -> Array<U32> & String:
  (a, +n) = r
  +more = "\"" ++ U32.show(key) ++ "\":" ++ U32.show(n)
  part = Bool.pick(String, String.is_empty(text), more, text ++ "," ++ more)
  (a, Bool.pick(String, U32.is_zero(n), text, part))

def histogram(fuel: Nat, +key: U32, r: Array<U32> & String) -> String:
  match fuel:
    case 0n:
      (a,text) = r
      "{" ++ text ++ "}"
    case 1n+p:
      (a,text) = r
      histogram(p,U32.inc(key),histogram_seen(key,text,Array.get(U32,a,key)))

def summary_checked(ok: Bool, now: Nat, m: Metrics, +batch: U32, hist: Array<U32>, total: R.Totals) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"multi-root accepted rows do not reconcile with execution")
    case True{}:
      Metrics{began,+calls,+rows,gather,back,scatter} = m
      R.Totals{completed,+neural,draws,roots} = total
      +elapsed = Nat.sub(now,began)
      +physical = U32.mul(calls,batch)
      physical_hist = Bool.pick(String,U32.is_zero(calls),"{}","{\"" ++ U32.show(batch) ++ "\":" ++ U32.show(calls) ++ "}")
      IO.print("info string cohort_work {\"schema\":\"deepfin.multi-root-work.v1\",\"scope\":\"bounded_cohort\",\"roots\":" ++ U32.show(roots)
        ++ ",\"wall_seconds\":" ++ W.seconds(elapsed) ++ ",\"completed_simulations\":" ++ U32.show(completed)
        ++ ",\"forward_calls\":" ++ U32.show(calls) ++ ",\"dispatched_real_rows\":" ++ U32.show(rows)
        ++ ",\"executed_real_rows\":" ++ U32.show(rows) ++ ",\"accepted_neural_rows\":" ++ U32.show(neural)
        ++ ",\"padded_rows\":" ++ U32.show(U32.sub(physical,rows)) ++ ",\"physical_rows\":" ++ U32.show(physical)
        ++ ",\"failed_forward_rows\":0,\"failed_rows\":0,\"cancelled_rows\":0,\"stale_rows\":0,\"rejected_rows\":0,\"executed_wasted_rows\":0,\"unresolved_rows\":0,\"unconfirmed_forward_rows\":0"
        ++ ",\"useful_eps\":" ++ W.rate(neural,elapsed) ++ ",\"executed_eps\":" ++ W.rate(rows,elapsed)
        ++ ",\"real_batch_histogram\":" ++ histogram(16n,1,(hist,"")) ++ ",\"physical_batch_histogram\":" ++ physical_hist
        ++ ",\"gathering_seconds\":" ++ W.seconds(gather) ++ ",\"backend_and_transport_seconds\":" ++ W.seconds(back)
        ++ ",\"normalization_and_backup_seconds\":" ++ W.seconds(scatter)
        ++ ",\"phase_seconds\":{\"queue_wait\":null,\"h2d\":null,\"gpu\":null,\"d2h\":null},\"clock_resolution_seconds\":0.001,\"warmup_excluded\":false}")

def summary(now: Nat, +m: Metrics, batch: U32, hist: Array<U32>, +total: R.Totals) -> IO(Unit):
  Metrics{began,calls,rows,g,b,s} = m
  R.Totals{n,neural,d,r} = total
  summary_checked(U32.is_eq(rows,neural),now,m,batch,hist,total)

def finish(cfg: C.Config, roots: List<C.Root>, store: C.Store, m: Metrics, hist: Array<U32>) -> IO(Unit):
  C.Store{table,maps,scratch,logits,input,output,B.Capabilities{profile,batch,channels,ilog,olog}} = store
  do IO<Unit>:
    now : Nat <- IO.now()
    total : R.Totals <- R.roots(cfg,roots,R.Totals{0,0,0,0})
    summary(now,m,batch,hist,total)

def raw_checked(table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,cap: B.Capabilities,
  r: Array<F32> & Bool) -> IO(C.Store):
  (output,valid) = r
  match valid:
    case False{}: IO.die(C.Store,2,"multi-root batch contains nonfinite output; no neural backup")
    case True{}: IO.pure(C.Store,C.Store{table,maps,scratch,logits,input,output,cap})

def shape_checked(ok: Bool, rows: U32, table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,output: Array<F32>,cap: B.Capabilities) -> IO(C.Store):
  match ok:
    case False{}: IO.die(C.Store,2,"multi-root backend shape/row mismatch")
    case True{}: raw_checked(table,maps,scratch,logits,input,cap,C.raw_valid(rows,output))

def returned(+rows: U32, table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,+cap: B.Capabilities, result: B.Result) -> IO(C.Store):
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  B.Result{input,output,real,physical,count} = result
  shape_checked(Bool.and(U32.is_eq(real,rows),Bool.and(U32.is_eq(physical,batch),U32.is_eq(count,U32.mul(rows,1861)))),
    rows,table,maps,scratch,logits,input,output,cap)

def dispatch(+rows: U32, store: C.Store) -> IO(C.Store):
  match rows:
    case 0: IO.pure(C.Store,store)
    case _:
      C.Store{table,maps,scratch,logits,input,output,cap} = store
      do IO<C.Store>:
        result : B.Result <- B.Batch.run(rows,input,output)
        returned(rows,table,maps,scratch,logits,cap,result)

def batch_notice(cfg: C.Config, seq: U32, +rows: U32, batch: U32) -> IO(Unit):
  C.Config{sims,depth,budget,diagnostics} = cfg
  R.diagnostic(Bool.and(diagnostics,U32.is_gt(rows,0)),"info string cohort_batch " ++ U32.show(seq) ++ " " ++ U32.show(rows) ++ " " ++ U32.show(batch))

def next_state(cfg: C.Config, m: Metrics, hist: Array<U32>, +row: U32, steps: U32, begun: Nat, +gathered: Nat, +ran: Nat,
  ended: Nat, tail: List<C.Root>, result: C.Store & List<C.Root>) -> State:
  Metrics{started,calls,rows,g,b,s} = m
  (store,roots) = result
  State{List.append(&1,C.Root,tail,roots),store,cfg,
    Metrics{started,U32.add(calls,Bool.to_u32(U32.is_gt(row,0))),U32.add(rows,row),Nat.add(g,Nat.sub(gathered,begun)),
      Nat.add(b,Nat.sub(ran,gathered)),Nat.add(s,Nat.sub(ended,ran))},
    increment(Array.get(U32,hist,row),row),U32.is_zero(steps)}

def gathered(+cfg: C.Config, +m: Metrics, hist: Array<U32>, begun: Nat, +at: Nat, batch: U32, g: C.Gathered) -> IO(State):
  Metrics{started,calls,n,gather,back,scatter} = m
  C.Gathered{tail,tasks,store,+rows,steps} = g
  do IO<State>:
    store : C.Store <- dispatch(rows,store)
    ran : Nat <- IO.now()
    batch_notice(cfg,U32.inc(calls),rows,batch)
    result : C.Store & List<C.Root> <- C.scatter(cfg,List.reverse(&1,C.Task,tasks),(store,Nil{}))
    ended : Nat <- IO.now()
    return next_state(cfg,m,hist,rows,steps,begun,at,ran,ended,tail,result)

def round(+cfg: C.Config, m: Metrics, hist: Array<U32>, roots: List<C.Root>, store: C.Store) -> IO(State):
  C.Store{table,maps,scratch,logits,input,output,+cap} = store
  B.Capabilities{profile,+batch,channels,ilog,olog} = cap
  do IO<State>:
    begun : Nat <- IO.now()
    g : C.Gathered <- C.gather(16n,cfg,batch,C.Gathered{roots,Nil{},C.Store{table,maps,scratch,logits,input,output,cap},0,0})
    at : Nat <- IO.now()
    gathered(cfg,m,hist,begun,at,batch,g)

def loop(fuel: Nat, state: State) -> IO(Unit):
  match fuel:
    case 0n: IO.die(Unit,2,"multi-root progress bound exceeded")
    case 1n+p:
      State{roots,store,cfg,m,hist,finished} = state
      match finished:
        case True{}: finish(cfg,roots,store,m,hist)
        case False{}:
          do IO<Unit>:
            state : State <- round(cfg,m,hist,roots,store)
            loop(p,state)

type Setup is Type:
  Setup{remaining: List<String>, roots: List<C.Root>, table: Array<U64>, count: U32}

def depth(cfg: C.Config) -> U32:
  C.Config{sims,depth,budget,diagnostics} = cfg
  depth

def position_parsed(cfg: C.Config, remaining: List<String>, roots: List<C.Root>, +count: U32, r: Array<U64> & Maybe<&2,Pos.Game>) -> IO(Setup):
  (table,maybe) = r
  match maybe:
    case None{}: IO.die(Setup,2,"invalid multi-root position/history")
    case Some{+game}:
      IO.pure(Setup,Setup{remaining,Con{C.Root{game,S.start(Pos.board(game),4096,U32.inc(count),depth(cfg)),0,0},roots},table,U32.inc(count)})

def position_checked(ok: Bool,cfg: C.Config,text: String,rest: List<String>,roots: List<C.Root>,table: Array<U64>,count: U32) -> IO(Setup):
  match ok:
    case False{}: IO.die(Setup,2,"multi-root position argument is too long")
    case True{}: position_parsed(cfg,rest,roots,count,Protocol.position(Protocol.parse_position(Text.words(text)),table))

def load_one(cfg: C.Config,setup: Setup) -> IO(Setup):
  Setup{remaining,roots,table,count} = setup
  match remaining:
    case Nil{}: IO.pure(Setup,Setup{Nil{},roots,table,count})
    case Con{+text,rest}: position_checked(Nat.is_le(String.length(text),16384n),cfg,text,rest,roots,table,count)

def load(fuel: Nat,+cfg: C.Config,setup: Setup) -> IO(Setup):
  match fuel:
    case 0n: IO.pure(Setup,setup)
    case 1n+p:
      do IO<Setup>:
        setup : Setup <- load_one(cfg,setup)
        load(p,cfg,setup)

def synchronous(cfg: C.Config,roots: List<C.Root>,table: Array<U64>,+cap: B.Capabilities) -> IO(Unit):
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  do IO<Unit>:
    now : Nat <- IO.now()
    loop(4098n,State{roots,C.Store{table,Policy.build(),Array.new(F32,14n,0.0),Array.new(F32,11n,0.0),
      Array.new(F32,U32.to_nat(ilog),0.0),Array.new(F32,U32.to_nat(olog),0.0),cap},cfg,
      Metrics{now,0,0,0n,0n,0n},Array.new(U32,5n,0),False{}})

def selected_mode(enabled: Bool,cfg: C.Config,roots: List<C.Root>,table: Array<U64>,cap: B.Capabilities) -> IO(Unit):
  match enabled:
    case False{}: synchronous(cfg,roots,table,cap)
    case True{}: AsyncRun.launch(cfg,roots,table,cap)

def opened(cfg: C.Config,roots: List<C.Root>,table: Array<U64>,cap: B.Capabilities) -> IO(Unit):
  do IO<Unit>:
    enabled : Bool <- Async.BatchAsync.enabled()
    selected_mode(enabled,cfg,roots,table,cap)

def loaded(cfg: C.Config,setup: Setup) -> IO(Unit):
  Setup{remaining,roots,table,count} = setup
  match remaining count:
    case Nil{} 0: IO.die(Unit,2,"expected one to sixteen roots")
    case Nil{} _:
      do IO<Unit>:
        cap : B.Capabilities <- B.Batch.open()
        opened(cfg,List.reverse(&1,C.Root,roots),table,cap)
    case _ _: IO.die(Unit,2,"at most sixteen roots are supported")

def configured(ok: Bool,+cfg: C.Config,positions: List<String>) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"expected sims 1..256, depth 1..32, evals 0..256, diagnostics 0..1")
    case True{}:
      do IO<Unit>:
        setup : Setup <- load(16n,cfg,Setup{positions,Nil{},Tables.build(),0})
        loaded(cfg,setup)

def numbers(s: Maybe<&2,U32>,d: Maybe<&2,U32>,e: Maybe<&2,U32>,t: Maybe<&2,U32>,positions: List<String>) -> IO(Unit):
  match s d e t:
    case Some{+s} Some{+d} Some{+e} Some{+t}:
      configured(Bool.and(U32.is_gt(s,0),Bool.and(U32.is_le(s,256),Bool.and(U32.is_gt(d,0),Bool.and(U32.is_le(d,32),Bool.and(U32.is_le(e,256),U32.is_le(t,1)))))),
        C.Config{s,d,e,U32.is_eq(t,1)},positions)
    case _ _ _ _: IO.die(Unit,2,"invalid multi-root numeric argument")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case Con{s,Con{d,Con{e,Con{t,positions}}}}: numbers(Text.number(s),Text.number(d),Text.number(e),Text.number(t),positions)
    case _: IO.die(Unit,2,"usage: SIMS DEPTH EVALS DIAGNOSTICS 'startpos [moves ...]' ['fen ...' ...]")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
