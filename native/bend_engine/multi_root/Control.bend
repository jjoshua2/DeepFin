# Raw I/O stays generic; control parsing and per-root cancellation belong to Bend.
import Base
import ./Deadlines.bend as D
import ../standalone/Text.bend as T

type Packet is Data:
  Packet{kind: U32, text: String}
type Flags is Data:
  Flags{cancelled: U32, roots: U32, eof: Bool, timers: D.Timers}

def Wire.poll() -> IO(Packet):
  import "../standalone/poll.c"

def marked(+mask: U32, id: U32) -> Bool:
  U32.is_gt(U32.and(mask,U32.shln(1,U32.to_nat(id))),0)

def notice(f: Flags, text: String) -> IO(Flags):
  do IO<Flags>:
    IO.print("info string cohort_control " ++ text)
    return f

def tag(+text: String) -> U32:
  Bool.pick(U32,String.eq(text,"isready"),1,
    Bool.pick(U32,String.eq(text,"stop"),2,Bool.pick(U32,String.eq(text,"quit"),3,0)))

def ended(f: Flags) -> IO(Flags):
  Flags{mask,roots,eof,timers} = f
  IO.pure(Flags,Flags{mask,roots,True{},timers})

def expired_notice(ok: Bool,id: U32) -> IO(Unit):
  match ok:
    case True{}: IO.print("info string cohort_control expired " ++ U32.show(id))
    case False{}: IO.pure(Unit,Unit{})

def stopped(f: Flags) -> U32:
  Flags{mask,roots,eof,timers} = f
  D.stopped(mask,timers)

def cancelled(ok: Bool, +id: U32, +mask: U32, roots: U32, eof: Bool, timers: D.Timers) -> IO(Flags):
  match ok:
    case False{}: notice(Flags{mask,roots,eof,timers},"error invalid-root")
    case True{}: notice(Flags{U32.or(mask,U32.shln(1,U32.to_nat(id))),roots,eof,timers},"cancel " ++ U32.show(id))

def single(code: U32, f: Flags) -> IO(Flags):
  match code:
    case 1:
      do IO<Flags>:
        IO.print("info string cohort_ready")
        return f
    case 2:
      Flags{mask,roots,eof,timers} = f
      notice(Flags{131070,roots,eof,timers},"stop")
    case 3:
      Flags{mask,roots,eof,timers} = f
      notice(Flags{131070,roots,True{},timers},"quit")
    case _: notice(f,"error invalid-command")

def installed(ok: Bool,+id: U32,ms: U32,due: Nat,mask: U32,roots: U32,eof: Bool,timers: D.Timers) -> IO(Flags):
  match ok:
    case False{}: notice(Flags{mask,roots,eof,timers},"error deadline-extension")
    case True{}: notice(Flags{mask,roots,eof,D.install(id,due,timers)},"deadline " ++ U32.show(id) ++ " " ++ U32.show(ms))

def expired_notices(fuel: Nat,+id: U32,+new_mask: U32) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        expired_notice(marked(new_mask,id),id)
        expired_notices(p,U32.inc(id),new_mask)

def cancel(n: Maybe<&2,U32>, f: Flags) -> IO(Flags):
  match n:
    case None{}: notice(f,"error invalid-root")
    case Some{+id}:
      Flags{mask,+roots,eof,timers} = f
      cancelled(Bool.and(U32.is_gt(id,0),U32.is_le(id,roots)),id,mask,roots,eof,timers)

def deadline_live(ok: Bool,+id: U32,ms: U32,+due: Nat,f: Flags) -> IO(Flags):
  match ok:
    case False{}: notice(f,"error inactive-root")
    case True{}:
      Flags{mask,roots,eof,+timers} = f
      installed(D.allowed(id,due,timers),id,ms,due,mask,roots,eof,timers)

def advanced(mask: U32,roots: U32,eof: Bool,previous: U32,+timers: D.Timers) -> IO(Flags):
  D.Timers{limits,expired} = timers
  do IO<Flags>:
    expired_notices(16n,1,U32.and(expired,U32.xor(previous,4294967295)))
    return Flags{mask,roots,eof,timers}

def two(ok: Bool, n: String, f: Flags) -> IO(Flags):
  match ok:
    case False{}: notice(f,"error invalid-command")
    case True{}: cancel(T.number(n),f)

def deadline_valid(ok: Bool,+live: U32,now: Nat,+id: U32,+ms: U32,+f: Flags) -> IO(Flags):
  match ok:
    case False{}: notice(f,"error invalid-deadline")
    case True{}:
      Flags{mask,roots,eof,timers} = f
      deadline_live(Bool.and(marked(live,id),Bool.not(marked(D.stopped(mask,timers),id))),id,ms,Nat.add(now,U32.to_nat(ms)),f)

def tick_enabled(enabled: Bool,live: U32,f: Flags) -> IO(Flags):
  match enabled:
    case False{}: IO.pure(Flags,f)
    case True{}:
      Flags{+mask,roots,eof,+timers} = f
      D.Timers{limits,previous} = timers
      do IO<Flags>:
        now : Nat <- IO.now()
        advanced(mask,roots,eof,previous,D.advance(now,U32.and(live,U32.xor(mask,4294967295)),timers))

def deadline_numbers(live: U32,now: Nat,id: Maybe<&2,U32>,ms: Maybe<&2,U32>,+f: Flags) -> IO(Flags):
  match id ms:
    case Some{+id} Some{+ms}:
      Flags{mask,roots,eof,timers} = f
      deadline_valid(Bool.and(U32.is_gt(id,0),Bool.and(U32.is_le(id,roots),U32.is_le(ms,3600000))),live,now,id,ms,f)
    case _ _: notice(f,"error invalid-deadline")

def tick(live: U32,+f: Flags) -> IO(Flags):
  Flags{mask,roots,eof,timers} = f
  tick_enabled(Bool.not(D.empty(timers)),live,f)

def deadline_command(ok: Bool,live: U32,now: Nat,id: String,ms: String,f: Flags) -> IO(Flags):
  match ok:
    case False{}: notice(f,"error invalid-command")
    case True{}: deadline_numbers(live,now,T.number(id),T.number(ms),f)

def command(live: U32,now: Nat,words: List<&2,String>, f: Flags) -> IO(Flags):
  match words:
    case Nil{}: IO.pure(Flags,f)
    case Con{text,Nil{}}: single(tag(text),f)
    case Con{text,Con{n,Nil{}}}: two(String.eq(text,"cancel"),n,f)
    case Con{text,Con{id,Con{ms,Nil{}}}}: deadline_command(String.eq(text,"deadline"),live,now,id,ms,f)
    case _: notice(f,"error invalid-command")

def received(live: U32,now: Nat,f: Flags, packet: Packet) -> IO(Flags):
  Packet{kind,text} = packet
  match kind:
    case 0: IO.pure(Flags,f)
    case 1: command(live,now,T.words(text),f)
    case 2: ended(f)
    case _: notice(f,"error malformed-line")

def poll_wire(+live: U32,f: Flags) -> IO(Flags):
  Flags{mask,roots,+eof,timers} = f
  match eof:
    case True{}: IO.pure(Flags,Flags{mask,roots,eof,timers})
    case False{}:
      do IO<Flags>:
        p : Packet <- Wire.poll()
        now : Nat <- IO.now()
        received(live,now,Flags{mask,roots,eof,timers},p)

# Commands cannot revive stopped/expired roots. Deadline durations are relative
# to command processing; waiting behind another physical batch consumes that time.
def poll(+live: U32,f: Flags) -> IO(Flags):
  do IO<Flags>:
    f : Flags <- tick(live,f)
    f : Flags <- poll_wire(live,f)
    tick(live,f)
