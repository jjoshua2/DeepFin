# A persistent owner around the qualified async search step. Lifecycle commits
# occur only after every row of the current batch has physically retired.
import Base
import ../collections_probe/Queue.bend as Q
import ./AsyncRun.bend as A
import ./Cohort.bend as C
import ./Control.bend as Ctrl
import ./Deadlines.bend as D
import ./DeadlineWork.bend as DW
import ./Report.bend as R
import ./LiveRegistry.bend as G
import ./LiveReport.bend as LR
import ../standalone/Position.bend as Pos
import ../standalone/Protocol.bend as P
import ../standalone/Text.bend as T
import ../standalone/Policy.bend as Policy
import ../standalone/NeuralWork.bend as W
import ../session_probe/Search.bend as S
import ../batch_backend/Batch.bend as B
import ../batch_backend/Async.bend as BA

type Change is Data:
  NoChange{}
  Install{slot: U32, old: U32, generation: U32, game: Pos.Game}
  Remove{slot: U32, generation: U32}
type Meta is Data:
  Meta{book: G.Book, pending: Change, closing: Bool, reported: U32, total: R.Totals, capacity: U32}
type Session is Type:
  Session{engine: A.State, meta: Meta}

def notice(s: Session,+text: String) -> IO(Session):
  do IO<Session>:
    IO.print("info string live_control " ++ text)
    return s

def pending(+change: Change) -> Bool:
  match change:
    case NoChange{}: False{}
    case _: True{}

def stopped_state(+mask: U32,s: Session) -> Session:
  Session{A.State{mode,store,Ctrl.Flags{old,n,eof,timers},counts,began,rounds},meta} = s
  Session{A.State{mode,store,Ctrl.Flags{U32.or(old,mask),n,eof,timers},counts,began,rounds},meta}

# Reporting does not duplicate tree ownership. It returns the same pool.
type Sweep is Type:
  Sweep{roots: List<C.Root>, sent: Array<U32>, wasted: Array<U32>, meta: Meta}

def emit_returned(+slot: U32,sent: Array<U32>,wasted: Array<U32>,roots: List<C.Root>,+meta: Meta,r: LR.Result) -> Sweep:
  Meta{book,change,closing,reported,total,capacity} = meta
  LR.Result{root,t} = r
  Sweep{Con{root,roots},sent,wasted,Meta{book,change,closing,U32.or(reported,D.bit(slot)),R.add(total,t),capacity}}

def emit_waste(+cfg: C.Config,+flags: Ctrl.Flags,+began: Nat,+slot: U32,+gen: U32,root: C.Root,roots: List<C.Root>,sent: Array<U32>,+n: U32,+meta: Meta,r: Array<U32> & U32) -> IO(Sweep):
  Ctrl.Flags{mask,nroots,eof,D.Timers{limits,expired}} = flags
  (wasted,d) = r
  do IO<Sweep>:
    r : LR.Result <- LR.emit(cfg,root,gen,n,d,Ctrl.marked(mask,slot),D.marked(expired,slot),D.lookup(slot,limits),began)
    return emit_returned(slot,sent,wasted,roots,meta,r)

def emit_sent(+cfg: C.Config,+flags: Ctrl.Flags,+began: Nat,+slot: U32,+gen: U32,root: C.Root,roots: List<C.Root>,wasted: Array<U32>,+meta: Meta,r: Array<U32> & U32) -> IO(Sweep):
  (sent,n) = r
  emit_waste(cfg,flags,began,slot,gen,root,roots,sent,n,meta,Array.get(U32,wasted,slot))

def report_checked(+ok: Bool,+cfg: C.Config,+flags: Ctrl.Flags,+began: Nat,+slot: U32,+gen: U32,root: C.Root,s: Sweep) -> IO(Sweep):
  match ok:
    case False{}:
      Sweep{roots,sent,wasted,meta} = s
      IO.pure(Sweep,Sweep{Con{root,roots},sent,wasted,meta})
    case True{}:
      Sweep{roots,sent,wasted,meta} = s
      emit_sent(cfg,flags,began,slot,gen,root,roots,wasted,meta,Array.get(U32,sent,slot))

def report_one(+cfg: C.Config,+flags: Ctrl.Flags,+began: Nat,root: C.Root,s: Sweep) -> IO(Sweep):
  C.Root{game,S.Tree{pool,used,cap,+n,request_node,+slot,req,+stop,depth},+neural,draws} = root
  Sweep{roots,sent,wasted,+meta} = s
  Meta{+book,change,closing,reported,total,capacity} = meta
  report_checked(Bool.and(Bool.not(DW.is_active(cfg,n,stop,neural)),Bool.not(D.marked(reported,slot))),cfg,flags,began,slot,G.generation(slot,book),
    C.Root{game,S.Tree{pool,used,cap,n,request_node,slot,req,stop,depth},neural,draws},Sweep{roots,sent,wasted,meta})

def report_all(+cfg: C.Config,+flags: Ctrl.Flags,+began: Nat,roots: List<C.Root>,s: Sweep) -> IO(Sweep):
  match roots:
    case Nil{}:
      Sweep{roots,sent,wasted,meta} = s
      IO.pure(Sweep,Sweep{List.reverse(&1,C.Root,roots),sent,wasted,meta})
    case Con{root,rest}:
      do IO<Sweep>:
        s : Sweep <- report_one(cfg,flags,began,root,s)
        report_all(cfg,flags,began,rest,s)

# Scan both physical queue lists and return them without flattening/requeueing.
# Tree ownership and FIFO order are preserved even when no result is emitted.
def flush_rear(+finished: Bool,store: C.Store,+flags: Ctrl.Flags,+calls: U32,+rows: U32,+discarded: U32,hist: Array<U32>,+began: Nat,+rounds: U32,front: List<C.Root>,size: Nat,s: Sweep) -> Session:
  Sweep{rear,sent,wasted,meta} = s
  Session{A.State{A.finished_mode(finished,Q.Queue{front,rear,size}),store,flags,A.Counts{sent,wasted,calls,rows,discarded,hist},began,rounds},meta}

def flush_front(+cfg: C.Config,+finished: Bool,store: C.Store,+flags: Ctrl.Flags,+calls: U32,+rows: U32,+discarded: U32,hist: Array<U32>,+began: Nat,+rounds: U32,rear: List<C.Root>,size: Nat,s: Sweep) -> IO(Session):
  Sweep{front,sent,wasted,meta} = s
  do IO<Session>:
    r : Sweep <- report_all(cfg,flags,began,A.mark(Ctrl.stopped(flags),rear),Sweep{Nil{},sent,wasted,meta})
    return flush_rear(finished,store,flags,calls,rows,discarded,hist,began,rounds,front,size,r)

def flush_roots(+cfg: C.Config,+finished: Bool,roots: Q.Queue<&1,C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta) -> IO(Session):
  Q.Queue{front,rear,size} = roots
  A.Counts{sent,wasted,calls,rows,discarded,hist} = counts
  do IO<Session>:
    s : Sweep <- report_all(cfg,flags,began,A.mark(Ctrl.stopped(flags),front),Sweep{Nil{},sent,wasted,meta})
    flush_front(cfg,finished,store,flags,calls,rows,discarded,hist,began,rounds,rear,size,s)

def flush(+cfg: C.Config,s: Session) -> IO(Session):
  Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
  match mode:
    case A.Waiting{tail,tasks,token,rows}: IO.pure(Session,Session{A.State{A.Waiting{tail,tasks,token,rows},store,flags,counts,began,rounds},meta})
    case A.Ready{roots}: flush_roots(cfg,False{},roots,store,flags,counts,began,rounds,meta)
    case A.Finished{roots}: flush_roots(cfg,True{},roots,store,flags,counts,began,rounds,meta)

# Dropping a pool is legal only at the barrier, after it has been reported.
def retain_root(keep: Bool,root: C.Root,rest: List<C.Root>) -> List<C.Root>:
  match keep:
    case False{}: rest
    case True{}: Con{root,rest}

def without_list(+slot: U32,roots: List<C.Root>) -> List<C.Root>:
  match roots:
    case Nil{}: Nil{}
    case Con{C.Root{g,S.Tree{pool,used,cap,n,p,+ep,req,stop,d},neural,draws},rest}:
      retain_root(Bool.not(U32.is_eq(slot,ep)),C.Root{g,S.Tree{pool,used,cap,n,p,ep,req,stop,d},neural,draws},without_list(slot,rest))

# Only lifecycle commits flatten the bounded queue; search/report sweeps retain
# its physical lists and the inherited FIFO gather/requeue implementation.
def without(+slot: U32,roots: Q.Queue<&1,C.Root>) -> Q.Queue<&1,C.Root>:
  C.queued(without_list(slot,C.queue_list(roots)))

def clear_flags(+slot: U32,+f: Ctrl.Flags) -> Ctrl.Flags:
  Ctrl.Flags{mask,n,eof,timers} = f
  Ctrl.Flags{G.clear(mask,slot),n,eof,G.clear_timer(slot,timers)}

def clear_counts(+slot: U32,c: A.Counts) -> A.Counts:
  A.Counts{sent,wasted,calls,rows,discarded,hist} = c
  A.Counts{Array.set(U32,sent,slot,0),Array.set(U32,wasted,slot,0),calls,rows,discarded,hist}

def install_now(+cfg: C.Config,+slot: U32,+gen: U32,+game: Pos.Game,roots: Q.Queue<&1,C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+meta: Meta) -> IO(Session):
  C.Config{sims,depth,budget,diagnostics} = cfg
  Meta{book,change,closing,reported,total,capacity} = meta
  root : C.Root = C.Root{game,S.start(Pos.board(game),capacity,slot,depth),0,0}
  notice(Session{A.State{A.Ready{Q.push(&1,C.Root,without(slot,roots),root)},store,clear_flags(slot,flags),clear_counts(slot,counts),began,0},
    Meta{G.install(slot,gen,book),NoChange{},closing,G.clear(reported,slot),total,capacity}},"admitted " ++ U32.show(slot) ++ " " ++ U32.show(gen))

def remove_now(+slot: U32,+gen: U32,roots: Q.Queue<&1,C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta) -> IO(Session):
  Meta{book,change,closing,reported,total,capacity} = meta
  notice(Session{A.State{A.Ready{without(slot,roots)},store,clear_flags(slot,flags),clear_counts(slot,counts),began,rounds},
    Meta{G.remove(slot,book),NoChange{},closing,G.clear(reported,slot),total,capacity}},"removed " ++ U32.show(slot) ++ " " ++ U32.show(gen))

def commit_roots(+cfg: C.Config,roots: Q.Queue<&1,C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta) -> IO(Session):
  Meta{book,change,closing,reported,total,capacity} = meta
  match change:
    case NoChange{}: IO.pure(Session,Session{A.State{A.Ready{roots},store,flags,counts,began,rounds},meta})
    case Install{slot,old,gen,game}: install_now(cfg,slot,gen,game,roots,store,flags,counts,began,meta)
    case Remove{slot,gen}: remove_now(slot,gen,roots,store,flags,counts,began,rounds,meta)

def commit_finished(+cfg: C.Config,roots: Q.Queue<&1,C.Root>,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta) -> IO(Session):
  Meta{book,change,closing,reported,total,capacity} = meta
  match change:
    case NoChange{}: IO.pure(Session,Session{A.State{A.Finished{roots},store,flags,counts,began,rounds},meta})
    case _: commit_roots(cfg,roots,store,flags,counts,began,rounds,meta)

def commit_mode(mode: A.Mode,+cfg: C.Config,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta) -> IO(Session):
  match mode:
    case A.Waiting{tail,tasks,token,rows}: IO.pure(Session,Session{A.State{A.Waiting{tail,tasks,token,rows},store,flags,counts,began,rounds},meta})
    case A.Finished{roots}: commit_finished(cfg,roots,store,flags,counts,began,rounds,meta)
    case A.Ready{roots}: commit_roots(cfg,roots,store,flags,counts,began,rounds,meta)

def commit(+cfg: C.Config,s: Session) -> IO(Session):
  Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
  commit_mode(mode,cfg,store,flags,counts,began,rounds,meta)

# Commands are versioned in this entrypoint. Legacy unversioned root controls
# are not passed through: a delayed cancel must not hit a replacement generation.
def stage_marked(+slot: U32,+gen: U32,s: Session) -> IO(Session):
  Session{engine,Meta{book,change,closing,reported,total,capacity}} = s
  notice(Session{engine,Meta{book,Remove{slot,gen},closing,reported,total,capacity}},"pending-remove " ++ U32.show(slot) ++ " " ++ U32.show(gen))

def stage_removed(+slot: U32,+gen: U32,s: Session) -> IO(Session):
  stage_marked(slot,gen,stopped_state(D.bit(slot),s))

def reserved(+slot: U32,+old: U32,+game: Pos.Game,engine: A.State,+closing: Bool,+reported: U32,+total: R.Totals,+capacity: U32,r: G.Book & U32) -> IO(Session):
  (book,+gen) = r
  mask = Bool.pick(U32,U32.is_zero(old),0,D.bit(slot))
  notice(stopped_state(mask,Session{engine,Meta{book,Install{slot,old,gen,game},closing,reported,total,capacity}}),
    "pending-install " ++ U32.show(slot) ++ " " ++ U32.show(gen))

def validated(+slot: U32,+old: U32,mode: A.Mode,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,output: Array<F32>,cap: B.Capabilities,
  +flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta,r: Array<U64> & Maybe<&2,Pos.Game>) -> IO(Session):
  Meta{book,change,closing,reported,total,capacity} = meta
  (table,game) = r
  match game:
    case None{}: notice(Session{A.State{mode,C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds},meta},"error invalid-position")
    case Some{game}:
      reserved(slot,old,game,A.State{mode,C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds},closing,reported,total,capacity,G.reserve(book))

def parse_game(+slot: U32,+old: U32,words: List<&2,String>,s: Session) -> IO(Session):
  Session{A.State{mode,C.Store{table,maps,scratch,logits,input,output,cap},flags,counts,began,rounds},meta} = s
  validated(slot,old,mode,maps,scratch,logits,input,output,cap,flags,counts,began,rounds,meta,P.position(P.parse_position(words),table))

def stage_valid(+ok: Bool,+slot: U32,+old: U32,words: List<&2,String>,s: Session) -> IO(Session):
  match ok:
    case False{}: notice(s,"error capacity-or-generation-limit")
    case True{}: parse_game(slot,old,words,s)

def stage_available(+kind: U32,+slot: U32,+gen: U32,words: List<&2,String>,s: Session) -> IO(Session):
  match kind:
    case 4: stage_removed(slot,gen,s)
    case _:
      Session{engine,+meta} = s
      Meta{+book,change,closing,reported,total,capacity} = meta
      stage_valid(Bool.and(U32.is_gt(slot,0),G.room(book)),slot,gen,words,Session{engine,meta})

def stage_busy(+busy: Bool,+kind: U32,+slot: U32,+gen: U32,words: List<&2,String>,s: Session) -> IO(Session):
  match busy:
    case True{}: notice(s,"error lifecycle-busy")
    case False{}: stage_available(kind,slot,gen,words,s)

def stage(+kind: U32,+slot: U32,+gen: U32,words: List<&2,String>,s: Session) -> IO(Session):
  Session{engine,+meta} = s
  Meta{book,change,closing,reported,total,capacity} = meta
  stage_busy(pending(change),kind,slot,gen,words,Session{engine,meta})

def add(words: List<&2,String>,s: Session) -> IO(Session):
  Session{engine,+meta} = s
  Meta{book,change,closing,reported,total,capacity} = meta
  stage(1,G.available(book),0,words,Session{engine,meta})

def control_returned(engine: A.State,+meta: Meta,+flags: Ctrl.Flags) -> Session:
  A.State{mode,store,old,counts,began,rounds} = engine
  Session{A.State{mode,store,flags,counts,began,rounds},meta}

def root_control(+kind: U32,+slot: U32,+gen: U32,+ms: String,+live: U32,+now: Nat,s: Session) -> IO(Session):
  match kind:
    case 2:
      Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
      do IO<Session>:
        IO.print("info string live_target " ++ U32.show(slot) ++ " " ++ U32.show(gen))
        flags : Ctrl.Flags <- Ctrl.cancel(Some{slot},flags)
        return Session{A.State{mode,store,flags,counts,began,rounds},meta}
    case _:
      Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
      do IO<Session>:
        IO.print("info string live_target " ++ U32.show(slot) ++ " " ++ U32.show(gen))
        flags : Ctrl.Flags <- Ctrl.deadline_numbers(live,now,Some{slot},T.number(ms),flags)
        return Session{A.State{mode,store,flags,counts,began,rounds},meta}

def target_live(+ok: Bool,+kind: U32,+slot: U32,+gen: U32,+ms: String,+live: U32,+now: Nat,s: Session) -> IO(Session):
  match ok:
    case False{}: notice(s,"error inactive-root")
    case True{}: root_control(kind,slot,gen,ms,live,now,s)

def targeted(+valid: Bool,+kind: U32,+slot: U32,+gen: U32,words: List<&2,String>,+live: U32,+now: Nat,s: Session) -> IO(Session):
  match valid:
    case False{}: notice(s,"error stale-identity")
    case True{}:
      match kind words:
        case 2 Nil{}: target_live(Ctrl.marked(live,slot),kind,slot,gen,"",live,now,s)
        case 5 Con{ms,Nil{}}: target_live(Ctrl.marked(live,slot),kind,slot,gen,ms,live,now,s)
        case 4 Nil{}: stage(kind,slot,gen,Nil{},s)
        case 3 _: stage(kind,slot,gen,words,s)
        case _ _: notice(s,"error malformed-command")

def target_numbers(+kind: U32,id: Maybe<&2,U32>,gen: Maybe<&2,U32>,words: List<&2,String>,+live: U32,+now: Nat,s: Session) -> IO(Session):
  match id gen:
    case Some{+id} Some{+gen}:
      Session{engine,+meta} = s
      Meta{book,change,closing,reported,total,capacity} = meta
      targeted(G.matches(id,gen,book),kind,id,gen,words,live,now,Session{engine,meta})
    case _ _: notice(s,"error malformed-identity")

def abandon(+change: Change) -> IO(Unit):
  match change:
    case NoChange{}: IO.pure(Unit,Unit{})
    case Install{slot,old,gen,game}: IO.print("info string live_control abandoned " ++ U32.show(slot) ++ " " ++ U32.show(gen))
    case Remove{slot,gen}: IO.print("info string live_control abandoned-remove " ++ U32.show(slot) ++ " " ++ U32.show(gen))

def stop_all(+quit: Bool,s: Session) -> IO(Session):
  Session{engine,Meta{book,change,closing,reported,total,capacity}} = s
  do IO<Session>:
    abandon(change)
    notice(stopped_state(131070,Session{engine,Meta{book,NoChange{},quit,reported,total,capacity}}),Bool.pick(String,quit,"quit","stop"))

def tag(+head: String) -> U32:
  Bool.pick(U32,String.eq(head,"add"),1,Bool.pick(U32,String.eq(head,"cancel"),2,
    Bool.pick(U32,String.eq(head,"replace"),3,Bool.pick(U32,String.eq(head,"remove"),4,Bool.pick(U32,String.eq(head,"deadline"),5,0)))))

def single_code(+code: U32,s: Session) -> IO(Session):
  match code:
    case 1:
      do IO<Session>:
        IO.print("info string live_ready")
        return s
    case 2: stop_all(False{},s)
    case 3: stop_all(True{},s)
    case _: notice(s,"error malformed-command")

def single(+head: String,s: Session) -> IO(Session):
  single_code(Bool.pick(U32,String.eq(head,"isready"),1,Bool.pick(U32,String.eq(head,"stop"),2,Bool.pick(U32,String.eq(head,"quit"),3,0))),s)

def multiple(+kind: U32,words: List<&2,String>,+live: U32,+now: Nat,s: Session) -> IO(Session):
  match kind words:
    case 1 _: add(words,s)
    case 0 _: notice(s,"error unknown-command")
    case _ Con{id,Con{gen,rest}}: target_numbers(kind,T.number(id),T.number(gen),rest,live,now,s)
    case _ _: notice(s,"error malformed-command")

def command(+live: U32,+now: Nat,words: List<&2,String>,s: Session) -> IO(Session):
  match words:
    case Nil{}: IO.pure(Session,s)
    case Con{head,Nil{}}: single(head,s)
    case Con{head,rest}: multiple(tag(head),rest,live,now,s)

def end_input(s: Session) -> IO(Session):
  Session{engine,Meta{book,change,closing,reported,total,capacity}} = s
  IO.pure(Session,Session{engine,Meta{book,change,True{},reported,total,capacity}})

def received(+live: U32,+now: Nat,s: Session,packet: Ctrl.Packet) -> IO(Session):
  Ctrl.Packet{kind,text} = packet
  match kind:
    case 0: IO.pure(Session,s)
    case 1: command(live,now,T.words(text),s)
    case 2: end_input(s)
    case _: notice(s,"error malformed-line")

def poll_open(+closing: Bool,+live: U32,s: Session) -> IO(Session):
  match closing:
    case True{}: IO.pure(Session,s)
    case False{}:
      do IO<Session>:
        packet : Ctrl.Packet <- Ctrl.Wire.poll()
        now : Nat <- IO.now()
        received(live,now,s,packet)

def observed(store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,+began: Nat,+rounds: U32,+meta: Meta,o: A.Observed) -> IO(Session):
  Meta{book,change,closing,reported,total,capacity} = meta
  A.Observed{mode,+live} = o
  do IO<Session>:
    flags : Ctrl.Flags <- Ctrl.tick(live,flags)
    poll_open(closing,live,Session{A.State{mode,store,flags,counts,began,rounds},meta})

def service(+cfg: C.Config,s: Session) -> IO(Session):
  Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
  observed(store,flags,counts,began,rounds,meta,A.observe(cfg,mode))

def done_mode(mode: A.Mode) -> A.Mode & Bool:
  match mode:
    case A.Finished{roots}: (A.Finished{roots},True{})
    case _: (mode,False{})

def finish_checked(+ok: Bool,+began: Nat,+batch: U32,counts: A.Counts,+total: R.Totals) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"live lifetime work does not reconcile")
    case True{}:
      A.Counts{sent,waste,+calls,+rows,+discarded,hist} = counts
      R.Totals{completed,+accepted,draws,roots} = total
      +physical = U32.mul(calls,batch)
      do IO<Unit>:
        BA.BatchAsync.shutdown()
        now : Nat <- IO.now()
        IO.print("info string live_work {\"schema\":\"deepfin.live-cohort-work.v1\",\"reported_generations\":" ++ U32.show(roots)
          ++ ",\"batch_size\":" ++ U32.show(batch) ++ ",\"completed_simulations\":" ++ U32.show(completed) ++ ",\"forward_calls\":" ++ U32.show(calls)
          ++ ",\"dispatched_real_rows\":" ++ U32.show(rows) ++ ",\"executed_real_rows\":" ++ U32.show(rows)
          ++ ",\"accepted_neural_rows\":" ++ U32.show(accepted) ++ ",\"cancelled_rows\":" ++ U32.show(discarded)
          ++ ",\"executed_wasted_rows\":" ++ U32.show(discarded) ++ ",\"physical_rows\":" ++ U32.show(physical)
          ++ ",\"padded_rows\":" ++ U32.show(U32.sub(physical,rows)) ++ ",\"unconfirmed_forward_rows\":0,\"unresolved_rows\":0"
          ++ ",\"wall_seconds\":" ++ W.seconds(Nat.sub(now,began)) ++ ",\"warmup_excluded\":false,\"useful_eps\":null}")

def finish_state(engine: A.State,+meta: Meta) -> IO(Unit):
  A.State{mode,C.Store{table,maps,scratch,logits,input,output,B.Capabilities{profile,batch,channels,ilog,olog}},flags,
    A.Counts{sent,waste,calls,+rows,+discarded,hist},began,rounds} = engine
  Meta{book,change,closing,reported,+total,capacity} = meta
  R.Totals{completed,accepted,draws,roots} = total
  finish_checked(U32.is_eq(rows,U32.add(accepted,discarded)),began,batch,A.Counts{sent,waste,calls,rows,discarded,hist},total)

def finish(s: Session) -> IO(Unit):
  Session{engine,meta} = s
  finish_state(engine,meta)

def step(+cfg: C.Config,s: Session) -> IO(Session):
  Session{engine,meta} = s
  do IO<Session>:
    engine : A.State <- A.progress(cfg,engine)
    s : Session <- flush(cfg,Session{engine,meta})
    commit(cfg,s)

type Iteration is Type:
  Iteration{done: Bool, idle: Bool, session: Session}

def classified(store: C.Store,flags: Ctrl.Flags,counts: A.Counts,began: Nat,rounds: U32,+meta: Meta,r: A.Mode & Bool) -> Iteration:
  Meta{book,change,closing,reported,total,capacity} = meta
  (mode,+idle) = r
  Iteration{Bool.and(closing,Bool.and(idle,Bool.not(pending(change)))),idle,Session{A.State{mode,store,flags,counts,began,rounds},meta}}

def classify(s: Session) -> Iteration:
  Session{A.State{mode,store,flags,counts,began,rounds},meta} = s
  classified(store,flags,counts,began,rounds,meta,done_mode(mode))

def loop(fuel: Nat,+cfg: C.Config,it: Iteration) -> IO(Unit):
  match fuel:
    case 0n:
      do IO<Unit>:
        BA.BatchAsync.shutdown()
        IO.die(Unit,2,"live event bound; physical work joined")
    case 1n+p:
      Iteration{done,idle,s} = it
      match done:
        case True{}: finish(s)
        case False{}:
          do IO<Unit>:
            IO.sleep(Bool.pick(U32,idle,1,0))
            s : Session <- service(cfg,s)
            s : Session <- flush(cfg,s)
            s : Session <- commit(cfg,s)
            s : Session <- step(cfg,s)
            loop(p,cfg,classify(s))

def opened(+cfg: C.Config,+capacity: U32,table: Array<U64>,+cap: B.Capabilities) -> IO(Unit):
  B.Capabilities{profile,batch,channels,ilog,olog} = cap
  do IO<Unit>:
    now : Nat <- IO.now()
    IO.print("info string live_open deepfin.live-cohort.v1")
    loop(1000000000n,cfg,classify(Session{A.State{A.Ready{Q.new(&1,C.Root)},C.Store{table,Policy.build(),Array.new(F32,14n,0.0),Array.new(F32,11n,0.0),
      Array.new(F32,U32.to_nat(ilog),0.0),Array.new(F32,U32.to_nat(olog),0.0),cap},Ctrl.Flags{0,16,False{},D.Timers{Nil{},0}},
      A.Counts{Array.new(U32,5n,0),Array.new(U32,5n,0),0,0,0,Array.new(U32,5n,0)},now,0},
      Meta{G.Book{Nil{},1},NoChange{},False{},0,R.Totals{0,0,0,0},capacity}}))
