# Bounded reverse-mask bits equal independently evaluated forward edges.
import Base
import ./Spec.bend as S
import ./Keys.bend as Keys
import ../board/Spec.bend as Piece
import ../initialized_attacks/Spec.bend as I
import ../decoder/Bits.bend as Bit
import ../attack_geometry/Geometry.bend as G
import ../attack_reversal/Spec.bend as Member
import ../attack_reversal/Mask.bend as LeaperBits
import ../attack_reversal/Reversal.bend as Leaper
import ../slider_reversal/Spec.bend as Slider
import ../slider_reversal/Reversal.bend as Reverse
import ../slider_reversal/Reflect.bend as Reflect

def reversed(kind: S.AttackKind,+target: Nat,by: Bool,+occ: U64) -> U64:
  match kind:
    case S.Pawn{}: I.mask(Piece.Pawn{},target,Bool.not(by),occ)
    case S.Knight{}: I.mask(Piece.Knight{},target,False{},occ)
    case S.King{}: I.mask(Piece.King{},target,False{},occ)
    case S.Rook{}: I.mask(Piece.Rook{},target,False{},occ)
    case S.Bishop{}: I.mask(Piece.Bishop{},target,False{},occ)

def leaper(+kind: G.Leaper,+src: Nat,+dst: Nat,
  +sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Bit.get64(G.mask(G.reverse(kind),dst),src) == Member.contains(dst,G.targets(kind,src)) : Bool}:
  Equal.trans(Bool,Bit.get64(G.mask(G.reverse(kind),dst),src),U64.test_bit(G.mask(G.reverse(kind),dst),src),Member.contains(dst,G.targets(kind,src)),
    Equal.sym(Bool,U64.test_bit(G.mask(G.reverse(kind),dst),src),Bit.get64(G.mask(G.reverse(kind),dst),src),Bit.test64(G.mask(G.reverse(kind),dst),src,sb)),
    Equal.trans(Bool,U64.test_bit(G.mask(G.reverse(kind),dst),src),Member.contains(src,G.targets(G.reverse(kind),dst)),Member.contains(dst,G.targets(kind,src)),
      LeaperBits.test(G.reverse(kind),dst,src,sb),
      Equal.sym(Bool,Member.contains(dst,G.targets(kind,src)),Member.contains(src,G.targets(G.reverse(kind),dst)),Leaper.bools(kind,src,dst,sb,tb))))

def slider(+kind: Slider.Slider,+src: Nat,+dst: Nat,+occ: U64,+mask: U64,
  +sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool},
  route: {mask == Slider.geometry(kind,dst,occ) : U64}) ->
  {Bit.get64(mask,src) == Slider.answer(kind,Slider.paths(src,dst),occ) : Bool}:
  Equal.trans(Bool,Bit.get64(mask,src),U64.test_bit(mask,src),Slider.answer(kind,Slider.paths(src,dst),occ),
    Equal.sym(Bool,U64.test_bit(mask,src),Bit.get64(mask,src),Bit.test64(mask,src,sb)),
    Equal.trans(Bool,U64.test_bit(mask,src),U64.test_bit(Slider.geometry(kind,dst,occ),src),Slider.answer(kind,Slider.paths(src,dst),occ),
      Equal.cong(U64,Bool,m => U64.test_bit(m,src),mask,Slider.geometry(kind,dst,occ),route),
      Equal.trans(Bool,U64.test_bit(Slider.geometry(kind,dst,occ),src),U64.test_bit(Slider.geometry(kind,src,occ),dst),Slider.answer(kind,Slider.paths(src,dst),occ),
        Reverse.geometry(kind,dst,src,occ,tb,sb),Reflect.geometry(kind,src,dst,occ,tb))))

def edge(+kind: S.AttackKind,+src: Nat,+dst: Nat,+by: Bool,+occ: U64,
  +sb: {Nat.is_lt(src,64n) == True{} : Bool},+tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Bit.get64(reversed(kind,dst,by,occ),src) == S.edge(kind,src,dst,by,occ) : Bool}:
  match kind:
    case S.Pawn{}:
      match by:
        case True{}: leaper(G.WhitePawn{},src,dst,sb,tb)
        case False{}: leaper(G.BlackPawn{},src,dst,sb,tb)
    case S.Knight{}: leaper(G.Knight{},src,dst,sb,tb)
    case S.King{}: leaper(G.King{},src,dst,sb,tb)
    case S.Rook{}: slider(Slider.Rook{},src,dst,occ,reversed(S.Rook{},dst,by,occ),sb,tb,Keys.rook(dst,occ,tb))
    case S.Bishop{}: slider(Slider.Bishop{},src,dst,occ,reversed(S.Bishop{},dst,by,occ),sb,tb,Keys.bishop(dst,occ,tb))
