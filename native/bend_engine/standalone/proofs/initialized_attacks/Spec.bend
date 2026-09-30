# Independent target-centred geometry and per-square witnesses.
# No production attack query, mask generator, table read, or accumulator defines the answer.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Piece
import ../attack_geometry/Geometry.bend as G
import ../ray/Spec.bend as Ray
import ../attack_witness/Spec.bend as W

def mask(kind: Piece.Piece,+sq: Nat,white: Bool,+occ: U64) -> U64:
  match kind:
    case Piece.Pawn{}: G.mask(Bool.pick(G.Leaper,white,G.WhitePawn{},G.BlackPawn{}),sq)
    case Piece.Knight{}: G.mask(G.Knight{},sq)
    case Piece.Bishop{}: Ray.rays(Nat.add(64n,sq),occ)
    case Piece.Rook{}: Ray.rays(sq,occ)
    case Piece.Queen{}: U64.or(Ray.rays(sq,occ),Ray.rays(Nat.add(64n,sq),occ))
    case Piece.King{}: G.mask(G.King{},sq)

def masks(+sq: Nat,by: Bool,+occ: U64) -> W.Masks:
  W.Masks{mask(Piece.Pawn{},sq,Bool.not(by),occ),mask(Piece.Knight{},sq,False{},occ),
    mask(Piece.King{},sq,False{},occ),mask(Piece.Rook{},sq,False{},occ),mask(Piece.Bishop{},sq,False{},occ)}

def attacked(+b: Chess.Board,sq: Nat,+by: Bool) -> Bool:
  W.present(b,Bool.to_u32(by),masks(sq,by,Chess.occupied(b)))
