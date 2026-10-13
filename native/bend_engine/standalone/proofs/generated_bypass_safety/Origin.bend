import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../castle_chain/Spec.bend as C
import ../castle_chain/Sides.bend as Sides
import ../castle_chain/Lists.bend as L
import ../castle_emission/Spec.bend as E
import ../candidate_bypass_classification/Generated.bend as G
import ../castle_safety/Spec.bend as Sensitive
import ../castle_safety/Mask.bend as Mask
import ../u64/Words.bend as W
import ../../../legal_probe/Chess.bend as Chess

# Preserve actual input-list membership through both guarded castle producers.
def before(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,table: Array<U64>,+xs: List<&2,Chess.Ply>,
  next: @r: Array<U64> & List<&2,Chess.Ply> ->
    C.every(m => E.member(m,xs) | C.castle(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(G.sides(b,(table,xs))):
  Sides.side(m => E.member(m,xs) | C.castle(b,m),
    r => Goal(Chess.castle_side(b,False{},r)),b,True{},(table,xs),
    L.old_tail(b,xs,xs,L.self(xs)),g => Inr{Inl{(g,{==})}},
    r => good => Sides.side(m => E.member(m,xs) | C.castle(b,m),Goal,
      b,False{},r,good,g => Inr{Inr{(g,{==})}},next))

def result(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  C.every(m => E.member(m,E.moves(r)) | C.castle(b,m),E.moves(G.sides(b,r))):
  (table,+xs)=r
  before(r => C.every(m => E.member(m,xs) | C.castle(b,m),E.moves(r)),
    b,table,xs,r => good => good)

def all(+c: O.Cells,+b: Chess.Board) ->
  C.every(m => E.member(m,S.scan(c,b)) | C.castle(b,m),S.candidates(c,b)):
  result(b,C.candidates(O.pack(c),b))

def member(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,S.candidates(c,b))) -> E.member(m,S.scan(c,b)) | C.castle(b,m):
  L.at(x => E.member(x,S.scan(c,b)) | C.castle(b,x),S.candidates(c,b),m,all(c,b),here)

def scan_origin(+c: O.Cells,+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  origin: E.member(m,S.scan(c,b)) | C.castle(b,m),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),m) == False{} : Bool}) ->
  E.member(m,S.scan(c,b)):
  match origin:
    case Inl{scan}: scan
    case Inr{castle}:
      W.false_true(E.member(m,S.scan(c,b)),Equal.trans(Bool,False{},
        Chess.filter_requires(Sensitive.sensitive(b,rays),m),True{},
        Equal.sym(Bool,Chess.filter_requires(Sensitive.sensitive(b,rays),m),False{},bypass),
        Mask.certificate(b,m,rays,castle)))

def scan_bypass(+c: O.Cells,+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  here: E.member(m,S.candidates(c,b)),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),m) == False{} : Bool}) ->
  E.member(m,S.scan(c,b)):
  scan_origin(c,b,rays,m,member(c,b,m,here),bypass)
