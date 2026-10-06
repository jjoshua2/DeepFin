# A public actual-full-Ply occurrence consumer, with no board validity premise.
import Base
import ./Source.bend as Source
import ./Spec.bend as S
import ../u64/Words.bend as Old
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C

def choose(yes: Bool,+b: C.Board,+m: C.Ply,tag: {yes == True{} : Bool},cert: S.gate(yes,b,m)) -> S.facts(b,m):
  match yes:
    case False{}: Old.false_true(S.facts(b,m),tag)
    case True{}: cert

def known(+b: C.Board,+m: C.Ply,tag: {S.flag(m) == 1 : U32},cert: S.property(b,m)) -> S.facts(b,m):
  choose(U32.is_eq(S.flag(m),1),b,m,Equal.cong(U32,Bool,f => U32.is_eq(f,1),S.flag(m),1,tag),cert)

def legal_member(table: Array<U64>,+b: C.Board,+m: C.Ply,
  here: E.member(m,E.moves(C.legal_moves(table,b))),tag: {S.flag(m) == 1 : U32}) -> S.facts(b,m):
  known(b,m,tag,Source.member(table,b,m,here))

# The actual priority decoder selects pawn0 when the producer's pawn bit holds.
def pawn_kind(+b: C.Board,+sq: U32,yes: {S.pawn(b,sq) == True{} : Bool}) -> {C.piece(b,sq) == 0 : U32}:
  %Equal.sym(Bool,S.pawn(b,sq),True{},yes) :
    {Bool.pick(U32,_,0,Bool.pick(U32,U64.test_bit(C.get_knights(b),U32.to_nat(sq)),1,Bool.pick(U32,U64.test_bit(C.get_bishops(b),U32.to_nat(sq)),2,Bool.pick(U32,U64.test_bit(C.get_rooks(b),U32.to_nat(sq)),3,Bool.pick(U32,U64.test_bit(C.get_queens(b),U32.to_nat(sq)),4,5))))) == 0 : U32}
  {==}

def kind_known(+b: C.Board,+m: C.Ply,cert: S.facts(b,m)) -> {C.piece(b,S.src(m)) == 0 : U32}:
  (promotion,(yes,rest))=cert
  pawn_kind(b,S.src(m),yes)

def decoded_kind(table: Array<U64>,+b: C.Board,+m: C.Ply,
  here: E.member(m,E.moves(C.legal_moves(table,b))),tag: {S.flag(m) == 1 : U32}) -> {C.piece(b,S.src(m)) == 0 : U32}:
  kind_known(b,m,legal_member(table,b,m,here,tag))
