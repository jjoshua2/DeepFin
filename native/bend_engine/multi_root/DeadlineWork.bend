# Deadline-aware ownership transformations for the existing cohort algorithm.
# No timers or root identities move into the native worker.
import Base
import ../collections_probe/Queue.bend as Q
import ./Cohort.bend as C
import ./Control.bend as Ctrl
import ./Deadlines.bend as D
import ../standalone/Position.bend as Pos
import ../standalone/Policy.bend as Policy
import ../batch_backend/Batch.bend as B
import ../session_probe/Search.bend as S

def halt(root: C.Root) -> C.Root:
  C.Root{g,S.Tree{pool,used,cap,n,pending,ep,req,stop,depth},neural,draws} = root
  C.Root{g,S.Tree{pool,used,cap,n,4096,ep,req,2,depth},neural,draws}

def is_active(+cfg: C.Config,+n: U32,+stop: U32,+neural: U32) -> Bool:
  C.Config{sims,depth,+budget,diagnostics} = cfg
  Bool.and(U32.is_zero(stop),Bool.and(U32.is_lt(n,sims),Bool.or(U32.is_zero(budget),U32.is_lt(neural,budget))))

def prepend_root(root: C.Root,r: List<C.Root> & U32) -> List<C.Root> & U32:
  (rest,mask) = r
  (Con{root,rest},mask)

def scan_roots(+cfg: C.Config,rs: List<C.Root>,+mask: U32) -> List<C.Root> & U32:
  match rs:
    case Nil{}: (Nil{},mask)
    case Con{C.Root{g,S.Tree{pool,used,cap,+n,pending,+ep,req,+stop,lim},+neural,draws},rest}:
      ok = is_active(cfg,n,stop,neural)
      root : C.Root = C.Root{g,S.Tree{pool,used,cap,n,pending,ep,req,stop,lim},neural,draws}
      prepend_root(root,scan_roots(cfg,rest,U32.or(mask,Bool.pick(U32,ok,D.bit(ep),0))))

# Read masks in both physical lists without reversing or changing the FIFO.
def scan_queue_rear(front: List<C.Root>, size: Nat, r: List<C.Root> & U32) -> Q.Queue<&1, C.Root> & U32:
  (rear, mask) = r
  (Q.Queue{front, rear, size}, mask)

def scan_queue_front(cfg: C.Config, rear: List<C.Root>, size: Nat, r: List<C.Root> & U32) -> Q.Queue<&1, C.Root> & U32:
  (front, mask) = r
  scan_queue_rear(front, size, scan_roots(cfg, rear, mask))

def scan_queue(+cfg: C.Config, q: Q.Queue<&1, C.Root>, mask: U32) -> Q.Queue<&1, C.Root> & U32:
  Q.Queue{front, rear, size} = q
  scan_queue_front(cfg, rear, size, scan_roots(cfg, front, mask))

def prepend_task(task: C.Task,r: List<C.Task> & U32) -> List<C.Task> & U32:
  (rest,mask) = r
  (Con{task,rest},mask)

def scan_tasks(+cfg: C.Config,ts: List<C.Task>,+mask: U32) -> List<C.Task> & U32:
  match ts:
    case Nil{}: (Nil{},mask)
    case Con{C.Local{C.Root{g,S.Tree{pool,used,cap,+n,pending,+ep,req,+stop,lim},+neural,draws}},rest}:
      ok = is_active(cfg,n,stop,neural)
      root : C.Root = C.Root{g,S.Tree{pool,used,cap,n,pending,ep,req,stop,lim},neural,draws}
      prepend_task(C.Local{root},scan_tasks(cfg,rest,U32.or(mask,Bool.pick(U32,ok,D.bit(ep),0))))
    case Con{C.Pending{root,game,+ticket,entries,row},rest}:
      S.Ticket{ep,req,node,b,m,needs} = ticket
      prepend_task(C.Pending{root,game,ticket,entries,row},scan_tasks(cfg,rest,U32.or(mask,D.bit(ep))))

# Pruning BEFORE admission compacts surviving inputs in ascending source order.
# Destination <= source; no admitted row or input snapshot is modified here.
def shifted(+dst: U32,r: Array<F32> & F32) -> Array<F32>:
  (a,x) = r
  Array.set(F32,a,dst,x)

def shift(+fuel: Nat,+src: U32,+dst: U32,a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: shift(p,U32.inc(src),U32.inc(dst),shifted(dst,Array.get(F32,a,src)))

def move_different(+different: Bool,+width: U32,+src: U32,+dst: U32,a: Array<F32>) -> Array<F32>:
  match different:
    case False{}: a
    case True{}: shift(U32.to_nat(width),U32.mul(src,width),U32.mul(dst,width),a)

def move(+width: U32,+src: U32,+dst: U32,a: Array<F32>) -> Array<F32>:
  move_different(Bool.not(U32.is_eq(src,dst)),width,src,dst,a)

type Compact is Type:
  Compact{tasks: List<C.Task>, input: Array<F32>, rows: U32}

def compact_one(+drop: Bool,+width: U32,root: C.Root,game: Pos.Game,ticket: S.Ticket,entries: List<&2,Policy.Entry>,+src: U32,c: Compact) -> Compact:
  match drop:
    case True{}:
      Compact{ts,input,rows} = c
      Compact{Con{C.Local{halt(root)},ts},input,rows}
    case False{}:
      Compact{ts,input,+rows} = c
      Compact{Con{C.Pending{root,game,ticket,entries,rows},ts},move(width,src,rows,input),U32.inc(rows)}

def compact(+mask: U32,+width: U32,ts: List<C.Task>,c: Compact) -> Compact:
  match ts:
    case Nil{}:
      Compact{ts,input,rows} = c
      Compact{List.reverse(&1,C.Task,ts),input,rows}
    case Con{C.Local{root},rest}:
      Compact{ts,input,rows} = c
      compact(mask,width,rest,Compact{Con{C.Local{root},ts},input,rows})
    case Con{C.Pending{root,game,+ticket,entries,src},rest}:
      S.Ticket{ep,req,node,b,m,needs} = ticket
      compact(mask,width,rest,compact_one(Ctrl.marked(mask,ep),width,root,game,ticket,entries,src,c))

type Admission is Type:
  Admission{g: C.Gathered, flags: Ctrl.Flags, settled: Bool}

def checked(+before: U32,g: C.Gathered,+flags: Ctrl.Flags) -> Admission:
  Admission{g,flags,U32.is_eq(before,Ctrl.stopped(flags))}

def compacted(+live: U32,+before: U32,tail: Q.Queue<&1, C.Root>,table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,output: Array<F32>,cap: B.Capabilities,+steps: U32,+flags: Ctrl.Flags,c: Compact) -> IO(Admission):
  Compact{ts,input,rows} = c
  do IO<Admission>:
    flags : Ctrl.Flags <- Ctrl.tick(live,flags)
    return checked(before,C.Gathered{tail,ts,C.Store{table,maps,scratch,logits,input,output,cap},rows,steps},flags)

def scanned_scan_tasks(tail: Q.Queue<&1, C.Root>,table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,output: Array<F32>,+cap: B.Capabilities,+steps: U32,+flags: Ctrl.Flags,r: List<C.Task> & U32) -> IO(Admission):
  (ts,+live) = r
  do IO<Admission>:
    +flags : Ctrl.Flags <- Ctrl.tick(live,flags)
    compacted(live,Ctrl.stopped(flags),tail,table,maps,scratch,logits,output,cap,steps,flags,
      compact(Ctrl.stopped(flags),U32.mul(C.cap_channels(cap),64),ts,Compact{Nil{},input,0}))

def scanned_scan_roots(+cfg: C.Config,ts: List<C.Task>,table: Array<U64>,maps: Policy.Maps,scratch: Array<F32>,logits: Array<F32>,input: Array<F32>,output: Array<F32>,cap: B.Capabilities,+steps: U32,+flags: Ctrl.Flags,r: Q.Queue<&1, C.Root> & U32) -> IO(Admission):
  (tail,live) = r
  scanned_scan_tasks(tail,table,maps,scratch,logits,input,output,cap,steps,flags,scan_tasks(cfg,ts,live))

def prepare_step(+settled: Bool,+cfg: C.Config,g: C.Gathered,+flags: Ctrl.Flags) -> IO(Admission):
  match settled:
    case True{}: IO.pure(Admission,Admission{g,flags,True{}})
    case False{}:
      C.Gathered{tail,ts,C.Store{table,maps,scratch,logits,input,output,cap},rows,steps} = g
      scanned_scan_roots(cfg,ts,table,maps,scratch,logits,input,output,cap,steps,flags,scan_queue(cfg,tail,0))

# Check again after input compaction. Only newly stopped roots require another
# pass; after at most sixteen removals the final observation is stable.
def prepare(+fuel: Nat,+cfg: C.Config,a: Admission) -> IO(Admission):
  match fuel:
    case 0n: IO.pure(Admission,a)
    case 1n+p:
      Admission{g,flags,settled} = a
      do IO<Admission>:
        a : Admission <- prepare_step(settled,cfg,g,flags)
        prepare(p,cfg,a)

# A raw batch has already passed the all-row finiteness gate. Each individual
# result starts normalization/backup only after a fresh deadline observation.
# That one result-processing step is synchronous, not preemptible.
type Scattered is Type:
  Scattered{store: C.Store, roots: List<C.Root>, flags: Ctrl.Flags, wasted: Array<U32>, discarded: U32}

def scatter_returned(flags: Ctrl.Flags,wasted: Array<U32>,discarded: U32,roots: List<C.Root>,r: C.Store & C.Root) -> Scattered:
  (store,root) = r
  Scattered{store,Con{root,roots},flags,wasted,discarded}

def rejected(root: C.Root,store: C.Store,roots: List<C.Root>,flags: Ctrl.Flags,discarded: U32,+ep: U32,r: Array<U32> & U32) -> Scattered:
  (wasted,n) = r
  Scattered{store,Con{halt(root),roots},flags,Array.set(U32,wasted,ep,U32.inc(n)),U32.inc(discarded)}

def apply_checked(drop: Bool,cfg: C.Config,root: C.Root,game: Pos.Game,ticket: S.Ticket,entries: List<&2,Policy.Entry>,row: U32,+ep: U32,s: Scattered) -> IO(Scattered):
  match drop:
    case True{}:
      Scattered{store,roots,flags,wasted,discarded} = s
      IO.pure(Scattered,rejected(root,store,roots,flags,discarded,ep,Array.get(U32,wasted,ep)))
    case False{}:
      Scattered{store,roots,flags,wasted,discarded} = s
      do IO<Scattered>:
        r : C.Store & C.Root <- C.apply_task(cfg,C.Pending{root,game,ticket,entries,row},store)
        return scatter_returned(flags,wasted,discarded,roots,r)

def scatter_one(cfg: C.Config,task: C.Task,s: Scattered) -> IO(Scattered):
  match task:
    case C.Local{root}:
      Scattered{store,roots,flags,wasted,discarded} = s
      IO.pure(Scattered,Scattered{store,Con{root,roots},flags,wasted,discarded})
    case C.Pending{root,game,+ticket,entries,row}:
      S.Ticket{+ep,req,node,b,m,needs} = ticket
      Scattered{store,roots,flags,wasted,discarded} = s
      do IO<Scattered>:
        +flags : Ctrl.Flags <- Ctrl.tick(D.bit(ep),flags)
        apply_checked(Ctrl.marked(Ctrl.stopped(flags),ep),cfg,root,game,ticket,entries,row,ep,
          Scattered{store,roots,flags,wasted,discarded})

def scatter(+cfg: C.Config,ts: List<C.Task>,s: Scattered) -> IO(Scattered):
  match ts:
    case Nil{}:
      Scattered{store,roots,flags,wasted,discarded} = s
      IO.pure(Scattered,Scattered{store,List.reverse(&1,C.Root,roots),flags,wasted,discarded})
    case Con{task,rest}:
      do IO<Scattered>:
        s : Scattered <- scatter_one(cfg,task,s)
        scatter(cfg,rest,s)
