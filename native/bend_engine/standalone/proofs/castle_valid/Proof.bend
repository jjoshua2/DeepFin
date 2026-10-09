import Base
import ./Spec.bend as S
import ./Ordinary.bend as Ordinary
import ./Castles.bend as Castles
import ./Emission.bend as Emission
import ../king_counts/Actual.bend as K
import ../board/Spec.bend as Board
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../ep_occurrence_tag/Spec.bend as EP
import ../ep_occurrence_tag/Source.bend as Tag
import ../storage/Observe.bend as O
import ../storage/Representation.bend as Storage
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def castle_side(+b: Chess.Board,+m: Chess.Ply,+ks: Bool,parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},
  cert: C.side(b,ks,m)) -> S.white_kingside(b,m):
  (guard,same)=cert
  %Equal.sym(Chess.Ply,m,E.move(b,ks),same) : S.white_kingside(b,_)
  Castles.post(b,ks,parent)
def castle(+b: Chess.Board,+m: Chess.Ply,parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},cert: C.castle(b,m)) -> S.white_kingside(b,m):
  match cert:
    case Inl{k}: castle_side(b,m,True{},parent,k)
    case Inr{q}: castle_side(b,m,False{},parent,q)
def post(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},
  tagged: EP.property(b,m),cert: Emission.certificate(b,m)) -> S.white_kingside(b,m):
  match cert:
    case Inl{ordinary}: Ordinary.post(b,m,rows,coherent,tagged,parent,ordinary)
    case Inr{castling}: castle(b,m,parent,castling)
def both(-b: Chess.Board,+xs: List<&2,Chess.Ply>,tags: C.every(m => EP.property(b,m),xs),
  certs: C.every(m => Emission.certificate(b,m),xs)) -> C.every(m => EP.property(b,m) & Emission.certificate(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      (tag,rest_tags)=tags
      (cert,rest_certs)=certs
      ((tag,cert),both(b,t,rest_tags,rest_certs))
def paired(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool},
  cert: EP.property(b,m) & Emission.certificate(b,m)) -> S.white_kingside(b,m):
  (tagged,generated)=cert
  post(b,m,rows,coherent,parent,tagged,generated)
def image(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(O.pack(c),b)))) -> S.white_kingside(b,m):
  (wk,(wq,(bk,bq)))=parent
  paired(b,m,rows,coherent,wk,
    Lists.at(x => EP.property(b,x) & Emission.certificate(b,x),E.moves(Chess.legal_moves(O.pack(c),b)),m,
      both(b,E.moves(Chess.legal_moves(O.pack(c),b)),
        Tag.legal(r => C.every(x => EP.property(b,x),E.moves(r)),O.pack(c),b,r => good => good),
        Emission.legal(r => C.every(x => Emission.certificate(b,x),E.moves(r)),O.pack(c),b,rows,coherent,r => good => good)),here))
def transfer(-table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) -> S.white_kingside(b,m):
  (+c,same)=cert
  image(c,b,m,rows,coherent,parent,
    %Equal.sym(Array<U64>,O.pack(c),table,same) : E.member(m,E.moves(Chess.legal_moves(_,b)))
    here)
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  turn: S.canonical(b),coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> S.white_kingside(b,m):
  transfer(table,b,m,rows,coherent,parent,here,Storage.reify(table))
