# Output provenance: preserved caller tail OR one guarded new move.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castling/Route.bend as R

def moves(r: Array<U64> & List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  (table,ms) = r
  ms

def growth(-ok: Bool,-m: Chess.Ply,-tail: List<&2,Chess.Ply>,-out: Array<U64> & List<&2,Chess.Ply>) -> Type:
  {moves(out) == tail : List<&2,Chess.Ply>} | ({ok == True{} : Bool} & {moves(out) == Con{m,tail} : List<&2,Chess.Ply>})

# Flow.side checks these terms against the imported producer, not source tokens.
def home(b: Chess.Board) -> U32:
  Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),0,56)

def src(b: Chess.Board) -> U32:
  U32.add(home(b),4)

def dst(b: Chess.Board,ks: Bool) -> U32:
  U32.add(home(b),Bool.pick(U32,ks,6,2))

def transit(b: Chess.Board,ks: Bool) -> U32:
  U32.add(home(b),Bool.pick(U32,ks,5,3))

def rook(b: Chess.Board,ks: Bool) -> U32:
  U32.add(home(b),Bool.pick(U32,ks,7,0))

def right(b: Chess.Board,ks: Bool) -> U32:
  U32.mul(Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),1,4),Bool.pick(U32,ks,1,2))

def between(b: Chess.Board,ks: Bool) -> U64:
  U64.shln(U64.from_u32(Bool.pick(U32,ks,96,14)),U32.to_nat(home(b)))

def rights_ok(+b: Chess.Board,ks: Bool) -> Bool:
  Bool.not(U32.is_zero(U32.and(Chess.get_rights(b),right(b,ks))))

def king_ok(+b: Chess.Board) -> Bool:
  U64.test_bit(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),U32.to_nat(src(b)))

def rook_ok(+b: Chess.Board,ks: Bool) -> Bool:
  U64.test_bit(U64.and(Chess.get_rooks(b),Chess.color(b,Chess.get_turn(b))),U32.to_nat(rook(b,ks)))

def path_clear(+b: Chess.Board,ks: Bool) -> Bool:
  U64.is_zero(U64.and(Chess.occupied(b),between(b,ks)))

def guard(+b: Chess.Board,+ks: Bool) -> Bool:
  Bool.and(rights_ok(b,ks),Bool.and(king_ok(b),Bool.and(rook_ok(b,ks),path_clear(b,ks))))

def move(+b: Chess.Board,ks: Bool) -> Chess.Ply:
  Chess.Ply{src(b),dst(b,ks),0,2}

# Proof-level membership compares the entire Ply, including promotion and flag.
def member(-m: Chess.Ply,xs: List<&2,Chess.Ply>) -> Type:
  match xs:
    case Nil{}: Empty
    case Con{h,t}: {m == h : Chess.Ply} | member(m,t)

def choose(white: Bool,ks: Bool) -> R.Route:
  match white ks:
    case True{} True{}: R.WhiteKing{}
    case True{} False{}: R.WhiteQueen{}
    case False{} True{}: R.BlackKing{}
    case False{} False{}: R.BlackQueen{}

def route(b: Chess.Board,ks: Bool) -> R.Route:
  choose(U32.is_eq(Chess.get_turn(b),1),ks)
