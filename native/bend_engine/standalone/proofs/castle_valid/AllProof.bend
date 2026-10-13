import Base
import ./Right.bend as A
import ./Spec.bend as S
import ./GeneralOrdinary.bend as Ordinary
import ./GeneralCastles.bend as Castles
import ./Emission.bend as Emission
import ../king_counts/Actual.bend as K
import ../color_counts/Actual.bend as Actual
import ../king_counts/Selector.bend as Selector
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

def tag_no(+b: Chess.Board,+m: Chess.Ply,chosen: {U32.is_eq(EP.flag(m),1) == False{} : Bool}) -> EP.property(b,m):
  %Equal.sym(Bool,U32.is_eq(EP.flag(m),1),False{},chosen) : EP.gate(_,b,m)
  Unit{}
def tag_yes(+b: Chess.Board,+m: Chess.Ply,chosen: {U32.is_eq(EP.flag(m),1) == True{} : Bool},facts: EP.facts(b,m)) -> EP.property(b,m):
  %Equal.sym(Bool,U32.is_eq(EP.flag(m),1),True{},chosen) : EP.gate(_,b,m)
  facts
def ordinary_leaves(+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  t0: EP.property(b,m),t1: EP.property(b,m),t2: EP.property(b,m),t3: EP.property(b,m),
  +sb: {Nat.is_lt(U32.to_nat(Actual.source(m)),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(Actual.source(m))) == True{} : Bool},
  +db: {Nat.is_lt(U32.to_nat(Actual.destination(m)),64n) == True{} : Bool},
  +no: {U32.is_eq(Actual.flag(m),2) == False{} : Bool},
  +selector: {Selector.put(b,m) == U64.test_bit(Chess.get_kings(b),U32.to_nat(Actual.source(m))) : Bool},
  +clear: {U64.and(U64.bit(U32.to_nat(Actual.destination(m))),Chess.get_kings(b)) == U64.zero() : U64},
  +capclear: {U64.and(U64.bit(U32.to_nat(K.cap(m))),Chess.get_kings(b)) == U64.zero() : U64}) -> S.parent(Chess.make_move(b,m)):
  (wk,(wq,(bk,bq)))=parent
  (Ordinary.post(A.Wk{},b,m,rows,coherent,t0,wk,((sb,(owned,(db,no))),(selector,(clear,capclear)))),
   (Ordinary.post(A.Wq{},b,m,rows,coherent,t1,wq,((sb,(owned,(db,no))),(selector,(clear,capclear)))),
    (Ordinary.post(A.Bk{},b,m,rows,coherent,t2,bk,((sb,(owned,(db,no))),(selector,(clear,capclear)))),
     Ordinary.post(A.Bq{},b,m,rows,coherent,t3,bq,((sb,(owned,(db,no))),(selector,(clear,capclear)))))))
def ordinary_tags(+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),t0: EP.property(b,m),t1: EP.property(b,m),t2: EP.property(b,m),t3: EP.property(b,m),cert: K.emitted(b,m)) -> S.parent(Chess.make_move(b,m)):
  ((sb,(owned,(db,no))),(selector,(clear,capclear)))=cert
  ordinary_leaves(b,m,rows,coherent,parent,t0,t1,t2,t3,sb,owned,db,no,selector,clear,capclear)
def ordinary_true(+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  +chosen: {U32.is_eq(EP.flag(m),1) == True{} : Bool},
  +promotion: {EP.promotion(m) == 0 : U32},+pawn: {EP.pawn(b,EP.src(m)) == True{} : Bool},
  +same: {EP.dst(m) == Chess.get_ep(b) : U32},+bound: {Nat.is_lt(U32.to_nat(EP.dst(m)),64n) == True{} : Bool},
  cert: K.emitted(b,m)) -> S.parent(Chess.make_move(b,m)):
  ordinary_tags(b,m,rows,coherent,parent,
    tag_yes(b,m,chosen,(promotion,(pawn,(same,bound)))),
    tag_yes(b,m,chosen,(promotion,(pawn,(same,bound)))),
    tag_yes(b,m,chosen,(promotion,(pawn,(same,bound)))),
    tag_yes(b,m,chosen,(promotion,(pawn,(same,bound)))),cert)
def ordinary_when(+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),yes: Bool,
  +chosen: {U32.is_eq(EP.flag(m),1) == yes : Bool},tagged: EP.gate(yes,b,m),cert: K.emitted(b,m)) -> S.parent(Chess.make_move(b,m)):
  match yes:
    case False{}: ordinary_tags(b,m,rows,coherent,parent,tag_no(b,m,chosen),tag_no(b,m,chosen),tag_no(b,m,chosen),tag_no(b,m,chosen),cert)
    case True{}:
      (promotion,(pawn,(same,bound)))=tagged
      ordinary_true(b,m,rows,coherent,parent,chosen,promotion,pawn,same,bound,cert)
def ordinary(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),tagged: EP.property(b,m),cert: K.emitted(b,m)) -> S.parent(Chess.make_move(b,m)):
  ordinary_when(b,m,rows,coherent,parent,U32.is_eq(EP.flag(m),1),{==},tagged,cert)
def castle_side(+b: Chess.Board,+m: Chess.Ply,+ks: Bool,parent: S.parent(b),cert: C.side(b,ks,m)) -> S.parent(Chess.make_move(b,m)):
  (wk,(wq,(bk,bq)))=parent
  (guard,same)=cert
  %Equal.sym(Chess.Ply,m,E.move(b,ks),same) : S.parent(Chess.make_move(b,_))
  (Castles.post(A.Wk{},b,ks,wk),(Castles.post(A.Wq{},b,ks,wq),(Castles.post(A.Bk{},b,ks,bk),Castles.post(A.Bq{},b,ks,bq))))
def castle(+b: Chess.Board,+m: Chess.Ply,parent: S.parent(b),cert: C.castle(b,m)) -> S.parent(Chess.make_move(b,m)):
  match cert:
    case Inl{k}: castle_side(b,m,True{},parent,k)
    case Inr{q}: castle_side(b,m,False{},parent,q)
def post(+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),tagged: EP.property(b,m),cert: Emission.certificate(b,m)) -> S.parent(Chess.make_move(b,m)):
  match cert:
    case Inl{generated}: ordinary(b,m,rows,coherent,parent,tagged,generated)
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
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),cert: EP.property(b,m) & Emission.certificate(b,m)) -> S.parent(Chess.make_move(b,m)):
  (tagged,generated)=cert
  post(b,m,rows,coherent,parent,tagged,generated)
def image(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  +coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(O.pack(c),b)))) -> S.parent(Chess.make_move(b,m)):
  paired(b,m,rows,coherent,parent,
    Lists.at(x => EP.property(b,x) & Emission.certificate(b,x),E.moves(Chess.legal_moves(O.pack(c),b)),m,
      both(b,E.moves(Chess.legal_moves(O.pack(c),b)),
        Tag.legal(r => C.every(x => EP.property(b,x),E.moves(r)),O.pack(c),b,r => good => good),
        Emission.legal(r => C.every(x => Emission.certificate(b,x),E.moves(r)),O.pack(c),b,rows,coherent,r => good => good)),here))
def transfer(-table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),here: E.member(m,E.moves(Chess.legal_moves(table,b))),
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) -> S.parent(Chess.make_move(b,m)):
  (+c,same)=cert
  image(c,b,m,rows,coherent,parent,
    %Equal.sym(Array<U64>,O.pack(c),table,same) : E.member(m,E.moves(Chess.legal_moves(_,b)))
    here)
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  turn: S.canonical(b),coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> S.parent(Chess.make_move(b,m)):
  transfer(table,b,m,rows,coherent,parent,here,Storage.reify(table))
