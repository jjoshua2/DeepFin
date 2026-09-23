# Raw I/O stays generic; control parsing and per-root cancellation belong to Bend.
import Base
import ../standalone/Text.bend as T

type Packet is Data:
  Packet{kind: U32, text: String}
def Wire.poll() -> IO(Packet):
  import "../standalone/poll.c"

type Flags is Data:
  Flags{cancelled: U32, roots: U32, eof: Bool}

def marked(+mask: U32, id: U32) -> Bool:
  U32.is_gt(U32.and(mask,U32.shln(1,U32.to_nat(id))),0)

def notice(f: Flags, text: String) -> IO(Flags):
  do IO<Flags>:
    IO.print("info string cohort_control " ++ text)
    return f

def cancelled(ok: Bool, +id: U32, +mask: U32, roots: U32, eof: Bool) -> IO(Flags):
  match ok:
    case False{}: notice(Flags{mask,roots,eof},"error invalid-root")
    case True{}: notice(Flags{U32.or(mask,U32.shln(1,U32.to_nat(id))),roots,eof},"cancel " ++ U32.show(id))

def cancel(n: Maybe<&2,U32>, f: Flags) -> IO(Flags):
  match n:
    case None{}: notice(f,"error invalid-root")
    case Some{+id}:
      Flags{mask,+roots,eof} = f
      cancelled(Bool.and(U32.is_gt(id,0),U32.is_le(id,roots)),id,mask,roots,eof)

def single(code: U32, f: Flags) -> IO(Flags):
  match code:
    case 1:
      do IO<Flags>:
        IO.print("info string cohort_ready")
        return f
    case 2:
      Flags{mask,roots,eof} = f
      notice(Flags{131070,roots,eof},"stop")
    case 3:
      Flags{mask,roots,eof} = f
      notice(Flags{131070,roots,True{}},"quit")
    case _: notice(f,"error invalid-command")

def tag(+text: String) -> U32:
  Bool.pick(U32,String.eq(text,"isready"),1,
    Bool.pick(U32,String.eq(text,"stop"),2,Bool.pick(U32,String.eq(text,"quit"),3,0)))

def two(ok: Bool, n: String, f: Flags) -> IO(Flags):
  match ok:
    case False{}: notice(f,"error invalid-command")
    case True{}: cancel(T.number(n),f)

def command(words: List<&2,String>, f: Flags) -> IO(Flags):
  match words:
    case Nil{}: IO.pure(Flags,f)
    case Con{text,Nil{}}: single(tag(text),f)
    case Con{text,Con{n,Nil{}}}: two(String.eq(text,"cancel"),n,f)
    case _: notice(f,"error invalid-command")

def ended(f: Flags) -> IO(Flags):
  Flags{mask,roots,eof} = f
  IO.pure(Flags,Flags{mask,roots,True{}})

def received(f: Flags, packet: Packet) -> IO(Flags):
  Packet{kind,text} = packet
  match kind:
    case 0: IO.pure(Flags,f)
    case 1: command(T.words(text),f)
    case 2: ended(f)
    case _: notice(f,"error malformed-line")

def poll(f: Flags) -> IO(Flags):
  Flags{mask,roots,+eof} = f
  match eof:
    case True{}: IO.pure(Flags,Flags{mask,roots,eof})
    case False{}:
      do IO<Flags>:
        p : Packet <- Wire.poll()
        received(Flags{mask,roots,eof},p)
