import Base
import ./Raw.bend as Raw
import ./Enemy.bend as Enemy
import ../promotion/Spec.bend as P
import ../ordinary_king_frame/Frame.bend as F
import ../ordinary_king_frame/Enemy.bend as Old
import ../board/Spec.bend as B
import ../ordinary_slider_preservation/Check.bend as C
import ../ordinary_slider_preservation/Hits.bend as H
import ../ordinary_nonslider_preservation/Prefix.bend as Prefix
import ../ordinary_nonslider_preservation/Compose.bend as Compose
import ../ray/Slider.bend as Slider
import ../occupancy_insertion/Lattice.bend as L
import ../prepared_ray_coverage/Query.bend as Q
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def masked_piece(+mask: U64,+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {U64.and(mask,Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece)) ==
    U64.and(mask,Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece)) : U64}:
  Equal.cong(U64,U64,kind => U64.and(mask,kind),Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece),
    Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),Enemy.kind_equal(b,src,dst,choice,piece,white,canonical))

def prefix_equal(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Prefix.hits(c,Raw.moved(b,src,dst,choice),q,U32.xor(Chess.get_turn(b),1)) ==
    Prefix.hits(c,F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)) : U64}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  +side=Chess.get_turn(b)
  Equal.trans(U64,Prefix.hits(c,post,q,U32.xor(side,1)),Prefix.group(c,post,q,side),
    Prefix.hits(c,plain,q,U32.xor(side,1)),Prefix.shape(c,post,q,side),
    Equal.trans(U64,Prefix.group(c,post,q,side),Prefix.group(c,plain,q,side),Prefix.hits(c,plain,q,U32.xor(side,1)),
      Slider.combine(U64.or(Prefix.pawn(c,post,q,side),Prefix.knight(c,post,q,side)),
        U64.or(Prefix.pawn(c,plain,q,side),Prefix.knight(c,plain,q,side)),Prefix.king(c,post,q,side),Prefix.king(c,plain,q,side),
        Slider.combine(Prefix.pawn(c,post,q,side),Prefix.pawn(c,plain,q,side),Prefix.knight(c,post,q,side),Prefix.knight(c,plain,q,side),
          masked_piece(C.pawn(c,U32.from_nat(q),U32.xor(side,1)),b,src,dst,choice,B.Pawn{},white,canonical),
          masked_piece(C.knight(c,U32.from_nat(q)),b,src,dst,choice,B.Knight{},white,canonical)),
        masked_piece(C.king(c,U32.from_nat(q)),b,src,dst,choice,B.King{},white,canonical)),
      Equal.sym(U64,Prefix.hits(c,plain,q,U32.xor(side,1)),Prefix.group(c,plain,q,side),Prefix.shape(c,plain,q,side))))

def enemies_equal(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+piece: B.Piece,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {C.enemies(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece) == C.enemies(F.moved(b,src,dst),Chess.get_turn(b),piece) : U64}:
  Slider.combine(Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece),
    Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),
    Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),B.Queen{}),
    Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),B.Queen{}),
    Enemy.kind_equal(b,src,dst,choice,piece,white,canonical),Enemy.kind_equal(b,src,dst,choice,B.Queen{},white,canonical))

def part(+b: Chess.Board,+q: Nat,+side: U32,+diagonal: Bool,piece: B.Piece) -> U64:
  U64.and(Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(b),False{}),C.enemies(b,side,piece))

def slider_part(+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+choice: P.Choice,+diagonal: Bool,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {part(Raw.moved(b,src,dst,choice),q,Chess.get_turn(b),diagonal,piece) ==
    part(F.moved(b,src,dst),q,Chess.get_turn(b),diagonal,piece) : U64}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  +side=Chess.get_turn(b)
  Equal.trans(U64,part(post,q,side,diagonal,piece),
    U64.and(Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(plain),False{}),C.enemies(post,side,piece)),
    part(plain,q,side,diagonal,piece),
    Equal.cong(U64,U64,occ => U64.and(Tables.slider(U32.from_nat(q),diagonal,occ,False{}),C.enemies(post,side,piece)),
      Chess.occupied(post),Chess.occupied(plain),Raw.occupancy(b,src,dst,choice)),
    Equal.cong(U64,U64,kinds => U64.and(Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(plain),False{}),kinds),
      C.enemies(post,side,piece),C.enemies(plain,side,piece),enemies_equal(b,src,dst,choice,piece,white,canonical)))

def slider_shape(+b: Chess.Board,+q: Nat,+side: U32) ->
  {H.hits(b,q,U32.xor(side,1)) == U64.or(part(b,q,side,False{},B.Rook{}),part(b,q,side,True{},B.Bishop{})) : U64}:
  +enemy=Chess.color(b,U32.xor(side,1))
  Equal.trans(U64,H.hits(b,q,U32.xor(side,1)),
    U64.or(U64.and(C.rook_raw(b,q),enemy),U64.and(C.bishop_raw(b,q),enemy)),
    U64.or(part(b,q,side,False{},B.Rook{}),part(b,q,side,True{},B.Bishop{})),
    L.and_or(C.rook_raw(b,q),C.bishop_raw(b,q),enemy),
    Slider.combine(U64.and(C.rook_raw(b,q),enemy),part(b,q,side,False{},B.Rook{}),
      U64.and(C.bishop_raw(b,q),enemy),part(b,q,side,True{},B.Bishop{}),
      C.gate(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),Chess.get_rooks(b),Chess.get_queens(b),enemy),
      C.gate(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),Chess.get_bishops(b),Chess.get_queens(b),enemy)))

def sliders_equal(+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {H.hits(Raw.moved(b,src,dst,choice),q,U32.xor(Chess.get_turn(b),1)) ==
    H.hits(F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)) : U64}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  +side=Chess.get_turn(b)
  Equal.trans(U64,H.hits(post,q,U32.xor(side,1)),U64.or(part(post,q,side,False{},B.Rook{}),part(post,q,side,True{},B.Bishop{})),
    H.hits(plain,q,U32.xor(side,1)),slider_shape(post,q,side),
    Equal.trans(U64,U64.or(part(post,q,side,False{},B.Rook{}),part(post,q,side,True{},B.Bishop{})),
      U64.or(part(plain,q,side,False{},B.Rook{}),part(plain,q,side,True{},B.Bishop{})),H.hits(plain,q,U32.xor(side,1)),
      Slider.combine(part(post,q,side,False{},B.Rook{}),part(plain,q,side,False{},B.Rook{}),
        part(post,q,side,True{},B.Bishop{}),part(plain,q,side,True{},B.Bishop{}),
        slider_part(b,q,src,dst,choice,False{},B.Rook{},white,canonical),slider_part(b,q,src,dst,choice,True{},B.Bishop{},white,canonical)),
      Equal.sym(U64,H.hits(plain,q,U32.xor(side,1)),U64.or(part(plain,q,side,False{},B.Rook{}),part(plain,q,side,True{},B.Bishop{})),slider_shape(plain,q,side))))

def hits_equal(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {C.hits(c,Raw.moved(b,src,dst,choice),q,U32.xor(Chess.get_turn(b),1)) ==
    C.hits(c,F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)) : U64}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  +by=U32.xor(Chess.get_turn(b),1)
  Equal.trans(U64,C.hits(c,post,q,by),U64.or(Prefix.hits(c,post,q,by),H.hits(post,q,by)),C.hits(c,plain,q,by),Compose.split(c,post,q,by),
    Equal.trans(U64,U64.or(Prefix.hits(c,post,q,by),H.hits(post,q,by)),U64.or(Prefix.hits(c,plain,q,by),H.hits(plain,q,by)),C.hits(c,plain,q,by),
      Slider.combine(Prefix.hits(c,post,q,by),Prefix.hits(c,plain,q,by),H.hits(post,q,by),H.hits(plain,q,by),
        prefix_equal(c,b,q,src,dst,choice,white,canonical),sliders_equal(b,q,src,dst,choice,white,canonical)),
      Equal.sym(U64,C.hits(c,plain,q,by),U64.or(Prefix.hits(c,plain,q,by),H.hits(plain,q,by)),Compose.split(c,plain,q,by))))

# Transfer an actual promotion-post whole pair by the derived occupancy equality.
def lookup_shadow(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+q: Nat,+by: U32,+kind: U32,+diagonal: Bool,
  actual: {Chess.attack(kind,O.pack(c),U32.from_nat(q),by,Chess.occupied(Raw.moved(b,src,dst,choice))) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(Raw.moved(b,src,dst,choice)),False{})) : Array<U64> & U64}) ->
  {Chess.attack(kind,O.pack(c),U32.from_nat(q),by,Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(F.moved(b,src,dst)),False{})) : Array<U64> & U64}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  Equal.trans(Array<U64> & U64,Chess.attack(kind,O.pack(c),U32.from_nat(q),by,Chess.occupied(plain)),
    Chess.attack(kind,O.pack(c),U32.from_nat(q),by,Chess.occupied(post)),
    (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(plain),False{})),
    Equal.cong(U64,Array<U64> & U64,occ => Chess.attack(kind,O.pack(c),U32.from_nat(q),by,occ),
      Chess.occupied(plain),Chess.occupied(post),Equal.sym(U64,Chess.occupied(post),Chess.occupied(plain),Raw.occupancy(b,src,dst,choice))),
    Equal.trans(Array<U64> & U64,Chess.attack(kind,O.pack(c),U32.from_nat(q),by,Chess.occupied(post)),
      (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(post),False{})),
      (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,Chess.occupied(plain),False{})),actual,
      Equal.cong(U64,Array<U64> & U64,occ => (O.pack(c),Tables.slider(U32.from_nat(q),diagonal,occ,False{})),
        Chess.occupied(post),Chess.occupied(plain),Raw.occupancy(b,src,dst,choice))))

def attacked(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  rP: Q.rook(c,q,U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))),
  dP: Q.bishop(c,q,U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))),
  rO: Q.rook(c,q,U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  dO: Q.bishop(c,q,U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ->
  {Chess.attacked(O.pack(c),Raw.moved(b,src,dst,choice),U32.from_nat(q),U32.xor(Chess.get_turn(b),1)) ==
    Chess.attacked(O.pack(c),F.moved(b,src,dst),U32.from_nat(q),U32.xor(Chess.get_turn(b),1)) : Array<U64> & Bool}:
  +post=Raw.moved(b,src,dst,choice)
  +plain=F.moved(b,src,dst)
  +by=U32.xor(Chess.get_turn(b),1)
  Equal.trans(Array<U64> & Bool,Chess.attacked(O.pack(c),post,U32.from_nat(q),by),
    (O.pack(c),Bool.not(U64.is_zero(C.hits(c,post,q,by)))),Chess.attacked(O.pack(c),plain,U32.from_nat(q),by),
    Compose.shape(c,post,q,by,rP,dP),
    Equal.trans(Array<U64> & Bool,(O.pack(c),Bool.not(U64.is_zero(C.hits(c,post,q,by)))),
      (O.pack(c),Bool.not(U64.is_zero(C.hits(c,plain,q,by)))),Chess.attacked(O.pack(c),plain,U32.from_nat(q),by),
      Equal.cong(U64,Array<U64> & Bool,hits => (O.pack(c),Bool.not(U64.is_zero(hits))),C.hits(c,post,q,by),C.hits(c,plain,q,by),
        hits_equal(c,b,q,src,dst,choice,white,canonical)),
      Equal.sym(Array<U64> & Bool,Chess.attacked(O.pack(c),plain,U32.from_nat(q),by),
        (O.pack(c),Bool.not(U64.is_zero(C.hits(c,plain,q,by)))),Compose.shape(c,plain,q,by,rO,dO))))
