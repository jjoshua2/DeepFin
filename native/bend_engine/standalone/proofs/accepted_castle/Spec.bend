# Proof-only fixed-table observations. These are actual check answers, not
# independent attack geometry and not an alternative production generator.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as O

def answer(r: Array<U64> & Bool) -> Bool:
  (table,x) = r
  x

def mask(r: Array<U64> & U64) -> U64:
  (table,x) = r
  x

def checked(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply) -> Bool:
  answer(Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)))

def initial(+c: O.Cells,+b: Chess.Board) -> Bool:
  answer(Chess.in_check(O.pack(c),b,Chess.get_turn(b)))

def rays(+c: O.Cells,+b: Chess.Board) -> U64:
  +side = Chess.get_turn(b)
  sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side)))
  mask(Chess.attack(4,O.pack(c),sq,side,Chess.occupied(b)))

def retain(reject: Bool,m: Chess.Ply,acc: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match reject:
    case True{}: acc
    case False{}: Con{m,acc}

def full(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,acc: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match xs:
    case Nil{}: acc
    case Con{+m,rest}: full(rest,c,b,retain(checked(c,b,m),m,acc))

def fast_step(required: Bool,+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,acc: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match required:
    case True{}: retain(checked(c,b,m),m,acc)
    case False{}: Con{m,acc}

def fast(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,+sensitive: U64,acc: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match xs:
    case Nil{}: acc
    case Con{+m,rest}: fast(rest,c,b,sensitive,fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc))

def choose(check: Bool,+c: O.Cells,+b: Chess.Board,xs: List<&2,Chess.Ply>,rays: U64) -> List<&2,Chess.Ply>:
  match check:
    case True{}: full(xs,c,b,Nil{})
    case False{}: fast(xs,c,b,U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),Nil{})

def prepared(+c: O.Cells,+b: Chess.Board,xs: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  choose(initial(c,b),c,b,xs,rays(c,b))

def safe(-c: O.Cells,-b: Chess.Board,-m: Chess.Ply) -> Type:
  {checked(c,b,m) == False{} : Bool}

def required_safe(-c: O.Cells,-b: Chess.Board,-sensitive: U64,-m: Chess.Ply) -> Type:
  {Chess.filter_requires(sensitive,m) == True{} : Bool} -> safe(c,b,m)
