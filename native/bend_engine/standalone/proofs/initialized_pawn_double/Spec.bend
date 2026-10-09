import Base
import ../../../legal_probe/Chess.bend as C
import ../attack_geometry/Geometry.bend as G

def kind(w: Bool) -> G.Leaper:
  match w:
    case True{}: G.WhitePawn{}
    case False{}: G.BlackPawn{}

def white(b: C.Board) -> Bool: U32.is_eq(C.get_turn(b),1)
def step(w: Bool) -> U32: Bool.pick(U32,w,8,4294967288)
def first(+s: U32,w: Bool) -> U32: U32.add(s,step(w))
def second(+s: U32,+w: Bool) -> U32: U32.add(first(s,w),step(w))
def start(w: Bool) -> U32: Bool.pick(U32,w,1,6)
def empty1(+b: C.Board,+s: U32) -> Bool:
  Bool.and(U32.is_lt(first(s,white(b)),64),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(first(s,white(b))))))
def rank(+b: C.Board,s: U32) -> Bool: U32.is_eq(U32.div(s,8),start(white(b)))
def empty2(+b: C.Board,+s: U32) -> Bool:
  Bool.and(empty1(b,s),Bool.and(rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(second(s,white(b)))))))
def one(+b: C.Board,+s: U32) -> U64: C.select_u64(empty1(b,s),U64.bit(U32.to_nat(first(s,white(b)))),U64.zero())
def two(+b: C.Board,+s: U32) -> U64: C.select_u64(empty2(b,s),U64.bit(U32.to_nat(second(s,white(b)))),U64.zero())
def capturable(+b: C.Board) -> U64:
  U64.or(U64.and_not(C.color(b,U32.xor(C.get_turn(b),1)),C.get_kings(b)),C.ep_target(b))
def targets(+b: C.Board,+s: U32) -> U64:
  U64.or(U64.or(one(b,s),two(b,s)),U64.and(G.mask(kind(white(b)),U32.to_nat(s)),capturable(b)))
def mask(r: Array<U64> & U64) -> U64:
  (table,bb)=r
  bb
def double(-b: C.Board,-s: U32,-d: U32) -> Type:
  {empty2(b,s) == True{} : Bool} & {d == second(s,white(b)) : U32}
