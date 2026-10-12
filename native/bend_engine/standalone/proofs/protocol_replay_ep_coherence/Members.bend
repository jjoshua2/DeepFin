# Full-Ply occurrence certificates for the exact initialized legal-list query.
import Base
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../child_ep_coherence/Coherence.bend as EP
import ../initialized_attacks/Build.bend as Init
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def extend(+head: Chess.Ply,+rest: List<&2,Chess.Ply>,xs: List<&2,Chess.Ply>,
  fields: C.every(m => E.member(m,rest),xs)) -> C.every(m => E.member(m,Con{head,rest}),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,tail}:
      (here,later)=fields
      (Inr{here},extend(head,rest,tail,later))
def self(+xs: List<&2,Chess.Ply>) -> C.every(m => E.member(m,xs),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+head,+rest}:
      (Inl{{==}},extend(head,rest,rest,self(rest)))
def ep_all(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  xs: List<&2,Chess.Ply>,+depth: {d == 17n : Nat},+full: {extra == 64n : Nat},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  fields: C.every(m => E.member(m,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),xs)) ->
  C.every(m => {EP.child(b,m) == True{} : Bool},xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,tail}:
      (here,later)=fields
      (EP.total(d,seed,n,0,512,extra,b,m,white,depth,full,canonical,here),
        ep_all(d,seed,n,extra,b,white,tail,depth,full,canonical,later))

# Build.run is definitionally the same producer as Initialized.run with key0/at512.
def legal(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  C.every(m => {EP.child(b,m) == True{} : Bool},E.moves(Chess.legal_moves(O.pack(c),b))):
  %Equal.sym(Array<U64>,O.pack(c),Init.run(d,seed,n,extra),built) :
    C.every(m => {EP.child(b,m) == True{} : Bool},E.moves(Chess.legal_moves(_,b)))
  ep_all(d,seed,n,extra,b,white,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),depth,full,canonical,
    self(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))))
