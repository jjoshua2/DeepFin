# Satisfiable input premises and geometric safe/unsafe witnesses.
# No giant closed initialized array is normalized here.
import Base
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S

def board(+white: Bool,+ks: Bool) -> Chess.Board:
  P.metadata(P.put(5,white,K.src(white),P.put(3,white,
    U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,7,0)),
    P.put(5,Bool.not(white),K.src(Bool.not(white)),P.empty()))),Bool.to_u32(white),15,64)

def consistent(+white: Bool,+ks: Bool) -> {B.valid(board(white,ks)) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def valid_side(+white: Bool,+ks: Bool) -> {Chess.get_turn(board(white,ks)) == Bool.to_u32(white) : U32}:
  {==}

def singleton(+white: Bool,+ks: Bool) -> {K.kings(board(white,ks),white) == U64.bit(U32.to_nat(K.src(white))) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def guard(+white: Bool,+ks: Bool) -> {E.guard(board(white,ks),ks) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

# A rook on g8 attacks g1, not the initial e1 or transit f1.
def destination_attacked() -> Chess.Board:
  P.put(3,False{},62,board(True{},True{}))
def bad_consistent() -> {B.valid(destination_attacked()) == True{} : Bool}: {==}
def bad_singleton() -> {K.kings(destination_attacked(),True{}) == U64.from_u32(16) : U64}: {==}
def bad_guard_true() -> {E.guard(destination_attacked(),True{}) == True{} : Bool}: {==}
# The native reference separately checks start/transit false and destination true.
