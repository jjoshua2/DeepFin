# A single Bend owner services commands between gathers and polls. The worker
# owns only its private tensor snapshot; it cannot cancel or resume a root.
import Base
import ../collections_probe/Queue.bend as Q
import ./Deadlines.bend as D
import ./DeadlineWork.bend as DW
import ./Cohort.bend as C
import ../standalone/Position.bend as Pos
import ./Report.bend as R
import ./Control.bend as Ctrl
import ../batch_backend/Batch.bend as B
import ../batch_backend/Async.bend as A
import ../standalone/Policy.bend as Policy
import ../standalone/NeuralWork.bend as W
import ../session_probe/Search.bend as S

type Counts is Type:
  Counts{sent: Array<U32>, wasted: Array<U32>, calls: U32, rows: U32, discarded: U32, hist: Array<U32>}
type Mode is Type:
  Ready{roots: Q.Queue<&1, C.Root>}
  Waiting{tail: Q.Queue<&1, C.Root>, tasks: List<C.Task>, token: U32, rows: U32}
  Finished{roots: Q.Queue<&1, C.Root>}
type State is Type:
  State{mode: Mode, store: C.Store, flags: Ctrl.Flags, counts: Counts, started: Nat, rounds: U32}

def inc(+at: U32, r: Array<U32> & U32) -> Array<U32>:
  (a,n) = r
  Array.set(U32,a,at,U32.inc(n))

def identity(root: C.Root) -> C.Root & U32:
  C.Root{g,S.Tree{pool,used,cap,n,pending,+ep,req,stop,depth},neural,draws} = root
  (C.Root{g,S.Tree{pool,used,cap,n,pending,ep,req,stop,depth},neural,draws},ep)

def halt_root(root: C.Root) -> C.Root:
  C.Root{g,S.Tree{pool,used,+cap,n,pending,ep,req,stop,depth},neural,draws} = root
  C.Root{g,S.Tree{pool,used,cap,n,S.no_pending(cap),ep,req,2,depth},neural,draws}

def mark_checked(selected: Bool, root: C.Root) -> C.Root:
  match selected:
    case False{}: root
    case True{}: halt_root(root)

def mark_one(mask: U32, r: C.Root & U32) -> C.Root:
  (root,ep) = r
  mark_checked(Ctrl.marked(mask,ep),root)

def mark(+mask: U32, roots: List<C.Root>) -> List<C.Root>:
  match roots:
    case Nil{}: Nil{}
    case Con{root,rest}: Con{mark_one(mask,identity(root)),mark(mask,rest)}

def mark_queue(+mask: U32, q: Q.Queue<&1, C.Root>) -> Q.Queue<&1, C.Root>:
  Q.Queue{front, rear, size} = q
  Q.Queue{mark(mask, front), mark(mask, rear), size}

def admit_cons(task: C.Task, r: List<C.Task> & Array<U32>) -> List<C.Task> & Array<U32>:
  (rest,sent) = r
  (Con{task,rest},sent)

def admitted(tasks: List<C.Task>, sent: Array<U32>) -> List<C.Task> & Array<U32>:
  match tasks:
    case Nil{}: (Nil{},sent)
    case Con{C.Local{root},rest}: admit_cons(C.Local{root},admitted(rest,sent))
    case Con{C.Pending{root,game,+ticket,entries,row},rest}:
      S.Ticket{+ep,req,node,b,m,needs} = ticket
      admit_cons(C.Pending{root,game,ticket,entries,row},admitted(rest,inc(ep,Array.get(U32,sent,ep))))

def retirement_scattered(tail: Q.Queue<&1, C.Root>,sent: Array<U32>,calls: U32,rows: U32,discarded: U32,hist: Array<U32>,began: Nat,rounds: U32,r: DW.Scattered) -> State:
  DW.Scattered{store,roots,flags,wasted,n} = r
  State{Ready{C.requeue(roots,tail)},store,flags,Counts{sent,wasted,calls,rows,U32.add(discarded,n),hist},began,rounds}

def retire_valid(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32) -> IO(State):
  Counts{sent,wasted,calls,rows,discarded,hist} = counts
  do IO<State>:
    r : DW.Scattered <- DW.scatter(cfg,tasks,DW.Scattered{store,Nil{},flags,wasted,0})
    return retirement_scattered(tail,sent,calls,rows,discarded,hist,began,rounds,r)

def retire(ok: Bool,cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32) -> IO(State):
  match ok:
    case False{}: IO.die(State,2,"async cohort nonfinite batch; no neural backup")
    case True{}: retire_valid(cfg,tail,tasks,store,flags,counts,began,rounds)

def raw_returned(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,cap: B.Capabilities,
  flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,r: Array<F32> & Bool) -> IO(State):
  (output,valid) = r
  retire(valid,cfg,tail,tasks,C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds)

def polled(status: U32, cfg: C.Config, tail: Q.Queue<&1, C.Root>, tasks: List<C.Task>, +token: U32, rows: U32,
  table: Array<U64>, maps: Policy.Maps, scratch: Array<F32>, logits: Array<F32>, input: Array<F32>, output: Array<F32>, cap: B.Capabilities,
  flags: Ctrl.Flags, counts: Counts, began: Nat, rounds: U32) -> IO(State):
  match status:
    case 1:
      do IO<State>:
        A.BatchAsync.wait(token)
        return State{Waiting{tail,tasks,token,rows},C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds}
    case 2: raw_returned(cfg,tail,tasks,table,maps,scratch,logits,input,cap,flags,counts,began,rounds,C.raw_valid(rows,output))
    case 4: IO.die(State,2,"async cohort backend failed; no successful summary")
    case _: IO.die(State,2,"async cohort completion token mismatch")

def poll_returned(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,token: U32,rows: U32,
  table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,cap: B.Capabilities,
  flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,p: A.BatchPolled) -> IO(State):
  A.BatchPolled{output,status} = p
  polled(status,cfg,tail,tasks,token,rows,table,maps,scratch,logits,input,output,cap,flags,counts,began,rounds)

def await(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,+token: U32,+rows: U32,
  store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32) -> IO(State):
  C.Store{table,maps,scratch,logits,input,output,cap} = store
  do IO<State>:
    p : A.BatchPolled <- A.BatchAsync.poll(token,rows,output)
    poll_returned(cfg,tail,tasks,token,rows,table,maps,scratch,logits,input,cap,flags,counts,began,rounds,p)

def submitted(tail: Q.Queue<&1, C.Root>, tasks: List<C.Task>, rows: U32, table: Array<U64>,maps: Policy.Maps,
  scratch: Array<F32>,logits: Array<F32>,output: Array<F32>,cap: B.Capabilities,
  flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,p: A.BatchSubmitted) -> State:
  A.BatchSubmitted{input,token} = p
  State{Waiting{tail,tasks,token,rows},C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds}

def finished_mode(done: Bool, roots: Q.Queue<&1, C.Root>) -> Mode:
  match done:
    case False{}: Ready{roots}
    case True{}: Finished{roots}

def local_scattered(tail: Q.Queue<&1, C.Root>,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,steps: U32,r: C.Store & List<C.Root>) -> State:
  (store,roots) = r
  State{finished_mode(U32.is_zero(steps),C.requeue(roots,tail)),store,flags,counts,began,rounds}

def no_rows(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,steps: U32) -> IO(State):
  do IO<State>:
    r : C.Store & List<C.Root> <- C.scatter(cfg,List.reverse(&1,C.Task,tasks),(store,Nil{}))
    return local_scattered(tail,flags,counts,began,rounds,steps,r)

def reserved(diagnostics: Bool,tail: Q.Queue<&1, C.Root>,+rows: U32,table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,output: Array<F32>,+cap: B.Capabilities,
  flags: Ctrl.Flags,wasted: Array<U32>,+calls: U32,n: U32,discarded: U32,hist: Array<U32>,began: Nat,rounds: U32,r: List<C.Task> & Array<U32>) -> IO(State):
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  (tasks,sent) = r
  do IO<State>:
    p : A.BatchSubmitted <- A.BatchAsync.submit(rows,input)
    R.diagnostic(diagnostics,"info string cohort_batch " ++ U32.show(U32.inc(calls)) ++ " " ++ U32.show(rows) ++ " " ++ U32.show(batch))
    return submitted(tail,tasks,rows,table,maps,scratch,logits,output,cap,flags,
      Counts{sent,wasted,U32.inc(calls),U32.add(n,rows),discarded,inc(rows,Array.get(U32,hist,rows))},began,rounds,p)

def submit_tasks(cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,rows: U32) -> IO(State):
  C.Config{sims,depth,limit,diagnostics} = cfg
  C.Store{table,maps,scratch,logits,input,output,cap} = store
  Counts{sent,wasted,calls,n,discarded,hist} = counts
  reserved(diagnostics,tail,rows,table,maps,scratch,logits,input,output,cap,flags,wasted,calls,n,discarded,hist,began,rounds,admitted(List.reverse(&1,C.Task,tasks),sent))

def gathered_rows(rows: U32,cfg: C.Config,tail: Q.Queue<&1, C.Root>,tasks: List<C.Task>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,steps: U32) -> IO(State):
  match rows:
    case 0: no_rows(cfg,tail,tasks,store,flags,counts,began,rounds,steps)
    case _: submit_tasks(cfg,tail,tasks,store,flags,counts,began,rounds,rows)

def admission_ready(ok: Bool,cfg: C.Config,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,g: C.Gathered) -> IO(State):
  match ok:
    case False{}: IO.die(State,2,"cohort deadline admission did not settle")
    case True{}:
      C.Gathered{tail,tasks,store,rows,steps} = g
      gathered_rows(rows,cfg,tail,List.reverse(&1,C.Task,tasks),store,flags,counts,began,rounds,steps)

def admission_returned(cfg: C.Config,counts: Counts,began: Nat,rounds: U32,a: DW.Admission) -> IO(State):
  DW.Admission{g,flags,settled} = a
  admission_ready(settled,cfg,flags,counts,began,rounds,g)

def gathered(+cfg: C.Config,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,g: C.Gathered) -> IO(State):
  C.Gathered{tail,tasks,store,rows,steps} = g
  do IO<State>:
    a : DW.Admission <- DW.prepare(17n,cfg,DW.Admission{C.Gathered{tail,List.reverse(&1,C.Task,tasks),store,rows,steps},flags,False{}})
    admission_returned(cfg,counts,began,rounds,a)

def round_ready(+cfg: C.Config,roots: Q.Queue<&1, C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32) -> IO(State):
  C.Store{table,maps,scratch,logits,input,output,+cap} = store
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  Ctrl.Flags{mask,n,eof,timers} = flags
  do IO<State>:
    g : C.Gathered <- C.gather(16n,cfg,batch,C.Gathered{mark_queue(Ctrl.stopped(flags),roots),Nil{},C.Store{table,maps,scratch,logits,input,output,cap},0,0})
    gathered(cfg,flags,counts,began,U32.inc(rounds),g)

def bounded(rounds: U32,cfg: C.Config,roots: Q.Queue<&1, C.Root>,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat) -> IO(State):
  match rounds:
    case 4098: IO.die(State,2,"async cohort progress bound")
    case _: round_ready(cfg,roots,store,flags,counts,began,rounds)

def step_mode(mode: Mode,cfg: C.Config,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32) -> IO(State):
  match mode:
    case Finished{roots}: IO.pure(State,State{Finished{roots},store,flags,counts,began,rounds})
    case Waiting{tail,tasks,token,rows}: await(cfg,tail,tasks,token,rows,store,flags,counts,began,rounds)
    case Ready{roots}: bounded(rounds,cfg,roots,store,flags,counts,began)

def progress(cfg: C.Config,s: State) -> IO(State):
  State{mode,store,flags,counts,began,rounds} = s
  step_mode(mode,cfg,store,flags,counts,began,rounds)

# Reporting is deliberately separate from the synchronous v1 contract.
type RootTotals is Type:
  RootTotals{sent: Array<U32>, wasted: Array<U32>, total: R.Totals}

def deadline_limits(timers: D.Timers) -> List<&2,D.Limit>:
  D.Timers{limits,expired} = timers
  limits

def deadline_mask(timers: D.Timers) -> U32:
  D.Timers{limits,expired} = timers
  expired

def root_wasted(cfg: C.Config,mask: U32,+timers: D.Timers,began: Nat,root: C.Root,sent: Array<U32>,n: U32,+ep: U32,total: R.Totals,r: Array<U32> & U32) -> IO(RootTotals):
  (wasted,d) = r
  do IO<RootTotals>:
    t : R.Totals <- R.async_root(cfg,root,n,d,Ctrl.marked(mask,ep),D.lookup(ep,deadline_limits(timers)),D.marked(deadline_mask(timers),ep),began)
    return RootTotals{sent,wasted,R.add(total,t)}

def root_sent(cfg: C.Config,mask: U32,timers: D.Timers,began: Nat,root: C.Root,wasted: Array<U32>,+ep: U32,total: R.Totals,r: Array<U32> & U32) -> IO(RootTotals):
  (sent,n) = r
  root_wasted(cfg,mask,timers,began,root,sent,n,ep,total,Array.get(U32,wasted,ep))

def root_keyed(cfg: C.Config,mask: U32,timers: D.Timers,began: Nat,sent: Array<U32>,wasted: Array<U32>,total: R.Totals,r: C.Root & U32) -> IO(RootTotals):
  (root,+ep) = r
  root_sent(cfg,mask,timers,began,root,wasted,ep,total,Array.get(U32,sent,ep))

def report_one(cfg: C.Config,mask: U32,timers: D.Timers,began: Nat,root: C.Root,r: RootTotals) -> IO(RootTotals):
  RootTotals{sent,wasted,total} = r
  root_keyed(cfg,mask,timers,began,sent,wasted,total,identity(root))

def totals(r: RootTotals) -> R.Totals:
  RootTotals{sent,wasted,total} = r
  total

def root_report(+cfg: C.Config,+mask: U32,+timers: D.Timers,+began: Nat,roots: List<C.Root>,r: RootTotals) -> IO(R.Totals):
  match roots:
    case Nil{}: IO.pure(R.Totals,totals(r))
    case Con{root,rest}:
      do IO<R.Totals>:
        r : RootTotals <- report_one(cfg,mask,timers,began,root,r)
        root_report(cfg,mask,timers,began,rest,r)

def bin_seen(+at: U32,+text: String,r: Array<U32> & U32) -> Array<U32> & String:
  (hist,+n) = r
  +part = "\"" ++ U32.show(at) ++ "\":" ++ U32.show(n)
  more = Bool.pick(String,U32.is_zero(n),text,Bool.pick(String,String.is_empty(text),part,text ++ "," ++ part))
  (hist,more)

def bins(fuel: Nat,+at: U32,r: Array<U32> & String) -> String:
  match fuel:
    case 0n:
      (hist,text) = r
      "{" ++ text ++ "}"
    case 1n+p:
      (hist,text) = r
      bins(p,U32.inc(at),bin_seen(at,text,Array.get(U32,hist,at)))

def summary(ok: Bool, began: Nat, now: Nat, expired: U32, +batch: U32, +calls: U32, +rows: U32, +wasted: U32, hist: Array<U32>, total: R.Totals) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"async cohort work does not reconcile")
    case True{}:
      R.Totals{completed,+accepted,draws,roots} = total
      +physical = U32.mul(calls,batch)
      +elapsed = Nat.sub(now,began)
      IO.print("info string cohort_work {\"schema\":\"deepfin.multi-root-async-work.v2\",\"scope\":\"bounded_async_cohort\",\"roots\":" ++ U32.show(roots)
        ++ ",\"deadline_expired_mask\":" ++ U32.show(expired) ++ ",\"wall_seconds\":" ++ W.seconds(elapsed) ++ ",\"completed_simulations\":" ++ U32.show(completed)
        ++ ",\"forward_calls\":" ++ U32.show(calls) ++ ",\"dispatched_real_rows\":" ++ U32.show(rows) ++ ",\"executed_real_rows\":" ++ U32.show(rows)
        ++ ",\"accepted_neural_rows\":" ++ U32.show(accepted) ++ ",\"cancelled_rows\":" ++ U32.show(wasted) ++ ",\"executed_wasted_rows\":" ++ U32.show(wasted)
        ++ ",\"physical_rows\":" ++ U32.show(physical) ++ ",\"padded_rows\":" ++ U32.show(U32.sub(physical,rows))
        ++ ",\"unconfirmed_forward_rows\":0,\"unresolved_rows\":0,\"failed_rows\":0,\"failed_forward_rows\":0,\"stale_rows\":0,\"rejected_rows\":0"
        ++ ",\"useful_eps\":" ++ W.rate(accepted,elapsed) ++ ",\"executed_eps\":" ++ W.rate(rows,elapsed)
        ++ ",\"real_batch_histogram\":" ++ bins(16n,1,(hist,"")) ++ ",\"physical_batch_histogram\":"
        ++ Bool.pick(String,U32.is_zero(calls),"{}","{\"" ++ U32.show(batch) ++ "\":" ++ U32.show(calls) ++ "}")
        ++ ",\"gathering_seconds\":null,\"backend_and_transport_seconds\":null,\"normalization_and_backup_seconds\":null"
        ++ ",\"phase_seconds\":{\"queue_wait\":null,\"h2d\":null,\"gpu\":null,\"d2h\":null},\"clock_resolution_seconds\":0.001,\"warmup_excluded\":false}")

def summary_total(began: Nat,now: Nat,expired: U32,batch: U32,calls: U32,+rows: U32,+discarded: U32,hist: Array<U32>,+total: R.Totals) -> IO(Unit):
  R.Totals{completed,accepted,draws,nroots} = total
  summary(U32.is_eq(rows,U32.add(accepted,discarded)),began,now,expired,batch,calls,rows,discarded,hist,total)

def finish(cfg: C.Config, roots: List<C.Root>, store: C.Store, +flags: Ctrl.Flags, counts: Counts, +began: Nat) -> IO(Unit):
  C.Store{table,maps,scratch,logits,input,output,B.Capabilities{profile,batch,channels,ilog,olog}} = store
  Ctrl.Flags{mask,n,eof,+timers} = flags
  Counts{sent,wasted,calls,+rows,+discarded,hist} = counts
  do IO<Unit>:
    A.BatchAsync.shutdown()
    now : Nat <- IO.now()
    total : R.Totals <- root_report(cfg,Ctrl.stopped(flags),timers,began,roots,RootTotals{sent,wasted,R.Totals{0,0,0,0}})
    summary_total(began,now,deadline_mask(timers),batch,calls,rows,discarded,hist,total)

type Observed is Type:
  Observed{mode: Mode, live: U32}

def observed_ready(r: Q.Queue<&1, C.Root> & U32) -> Observed:
  (roots,live) = r
  Observed{Ready{roots},live}

def observed_tasks(tail: Q.Queue<&1, C.Root>,token: U32,rows: U32,r: List<C.Task> & U32) -> Observed:
  (tasks,live) = r
  Observed{Waiting{tail,tasks,token,rows},live}

def observed_tail(cfg: C.Config,tasks: List<C.Task>,token: U32,rows: U32,r: Q.Queue<&1, C.Root> & U32) -> Observed:
  (tail,live) = r
  observed_tasks(tail,token,rows,DW.scan_tasks(cfg,tasks,live))

def observe(+cfg: C.Config,mode: Mode) -> Observed:
  match mode:
    case Ready{roots}: observed_ready(DW.scan_queue(cfg,roots,0))
    case Waiting{tail,tasks,token,rows}: observed_tail(cfg,tasks,token,rows,DW.scan_queue(cfg,tail,0))
    case Finished{roots}: Observed{Finished{roots},0}

def service(cfg: C.Config,store: C.Store,flags: Ctrl.Flags,counts: Counts,began: Nat,rounds: U32,r: Observed) -> IO(State):
  Observed{mode,live} = r
  do IO<State>:
    flags : Ctrl.Flags <- Ctrl.poll(live,flags)
    progress(cfg,State{mode,store,flags,counts,began,rounds})

def loop(fuel: Nat,+cfg: C.Config,s: State) -> IO(Unit):
  match fuel:
    case 0n:
      do IO<Unit>:
        A.BatchAsync.shutdown()
        IO.die(Unit,2,"async cohort event bound; physical work joined")
    case 1n+p:
      State{mode,store,flags,counts,began,rounds} = s
      match mode:
        case Finished{roots}: finish(cfg,C.queue_list(roots),store,flags,counts,began)
        case _:
          do IO<Unit>:
            s : State <- service(cfg,store,flags,counts,began,rounds,observe(cfg,mode))
            loop(p,cfg,s)

def count_cons(root: C.Root,r: List<C.Root> & U32) -> List<C.Root> & U32:
  (rest,n) = r
  (Con{root,rest},n)

def count(roots: List<C.Root>,n: U32) -> List<C.Root> & U32:
  match roots:
    case Nil{}: (Nil{},n)
    case Con{root,rest}: count_cons(root,count(rest,U32.inc(n)))

def counted(cfg: C.Config,table: Array<U64>,+cap: B.Capabilities,r: List<C.Root> & U32) -> IO(Unit):
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  (roots,n) = r
  do IO<Unit>:
    now : Nat <- IO.now()
    loop(1000000000n,cfg,State{Ready{C.queued(roots)},C.Store{table,Policy.build(),Array.new(F32,14n,0.0),Array.new(F32,11n,0.0),
      Array.new(F32,U32.to_nat(ilog),0.0),Array.new(F32,U32.to_nat(olog),0.0),cap},Ctrl.Flags{0,n,False{},D.Timers{Nil{},0}},
      Counts{Array.new(U32,5n,0),Array.new(U32,5n,0),0,0,0,Array.new(U32,5n,0)},now,0})

def launch(cfg: C.Config,roots: List<C.Root>,table: Array<U64>,cap: B.Capabilities) -> IO(Unit):
  counted(cfg,table,cap,count(roots,0))
