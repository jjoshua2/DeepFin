# Pure, bounded per-root deadline state. Nat timestamps avoid U32 wraparound.
# Duration zero expires immediately; an existing deadline may only be shortened.
import Base

type Limit is Data:
  Limit{root: U32, due: Nat}
type Timers is Data:
  Timers{limits: List<&2,Limit>, expired: U32}

def bit(root: U32) -> U32:
  U32.shln(1,U32.to_nat(root))

def marked(mask: U32,root: U32) -> Bool:
  U32.is_gt(U32.and(mask,bit(root)),0)

def empty(t: Timers) -> Bool:
  Timers{limits,expired} = t
  match limits:
    case Nil{}: True{}
    case _: False{}

def lookup(+root: U32,limits: List<&2,Limit>) -> Maybe<&2,Nat>:
  match limits:
    case Nil{}: None{}
    case Con{Limit{id,due},rest}:
      Bool.pick(Maybe<&2,Nat>,U32.is_eq(root,id),Some{due},lookup(root,rest))

def insert(+root: U32,+due: Nat,limits: List<&2,Limit>) -> List<&2,Limit>:
  match limits:
    case Nil{}: Con{Limit{root,due},Nil{}}
    case Con{+limit,+rest}:
      Limit{id,old} = limit
      Bool.pick(List<&2,Limit>,U32.is_eq(root,id),Con{Limit{root,due},rest},Con{limit,insert(root,due,rest)})

def no_extension(due: Nat,previous: Maybe<&2,Nat>) -> Bool:
  match previous:
    case None{}: True{}
    case Some{old}: Nat.is_le(due,old)

def allowed(+root: U32,+due: Nat,+t: Timers) -> Bool:
  Timers{limits,expired} = t
  Bool.and(Bool.not(marked(expired,root)),no_extension(due,lookup(root,limits)))

def install(root: U32,due: Nat,t: Timers) -> Timers:
  Timers{limits,expired} = t
  Timers{insert(root,due,limits),expired}

def expired_mask(+now: Nat,+live: U32,limits: List<&2,Limit>,+mask: U32) -> U32:
  match limits:
    case Nil{}: mask
    case Con{Limit{+id,due},rest}:
      next = Bool.pick(U32,Bool.and(marked(live,id),Nat.is_ge(now,due)),U32.or(mask,bit(id)),mask)
      expired_mask(now,live,rest,next)

def advance(now: Nat,live: U32,t: Timers) -> Timers:
  Timers{+limits,expired} = t
  Timers{limits,expired_mask(now,live,limits,expired)}

def stopped(manual: U32,t: Timers) -> U32:
  Timers{limits,expired} = t
  U32.or(manual,expired)
