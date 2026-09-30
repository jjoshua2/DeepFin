# Actual storage layout and mask expressions, separate from independent geometry.
import Base
import ../../Tables.bend as Tables
import ./Geometry.bend as G

def base(kind: G.Leaper) -> U32:
  match kind:
    case G.Knight{}: 256
    case G.King{}: 320
    case G.WhitePawn{}: 384
    case G.BlackPawn{}: 448

def offset(kind: G.Leaper) -> Nat:
  match kind:
    case G.Knight{}: 256n
    case G.King{}: 320n
    case G.WhitePawn{}: 384n
    case G.BlackPawn{}: 448n

def end(kind: G.Leaper) -> U32:
  match kind:
    case G.Knight{}: 320
    case G.King{}: 384
    case G.WhitePawn{}: 448
    case G.BlackPawn{}: 512

def index(kind: G.Leaper,sq: Nat) -> U32:
  U32.add(base(kind),U32.from_nat(sq))

def value(kind: G.Leaper,sq: Nat) -> U64:
  match kind:
    case G.Knight{}: Tables.leaps(8n,U32.from_nat(sq),0,False{},U64.zero())
    case G.King{}: Tables.leaps(8n,U32.from_nat(sq),0,True{},U64.zero())
    case G.WhitePawn{}: Tables.pawns(U32.from_nat(sq),1)
    case G.BlackPawn{}: Tables.pawns(U32.from_nat(sq),4294967295)

def same(a: G.Leaper,b: G.Leaper) -> Bool:
  match a b:
    case G.Knight{} G.Knight{}: True{}
    case G.King{} G.King{}: True{}
    case G.WhitePawn{} G.WhitePawn{}: True{}
    case G.BlackPawn{} G.BlackPawn{}: True{}
    case _ _: False{}

def overlaps(a: G.Leaper,b: G.Leaper,i: Nat,j: Nat) -> Bool:
  Bool.and(same(a,b),Nat.is_eq(i,j))

def block(a: Array<U64>,+sq: Nat) -> Array<U64>:
  a = Array.set(U64,a,index(G.Knight{},sq),value(G.Knight{},sq))
  a = Array.set(U64,a,index(G.King{},sq),value(G.King{},sq))
  a = Array.set(U64,a,index(G.WhitePawn{},sq),value(G.WhitePawn{},sq))
  Array.set(U64,a,index(G.BlackPawn{},sq),value(G.BlackPawn{},sq))
