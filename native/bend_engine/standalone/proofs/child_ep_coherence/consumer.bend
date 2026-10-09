import Base
import ./Coherence.bend as EP
import ../pawn_ep_occurrence/Bridge.bend as Prior
import ../child_metadata/Post.bend as Metadata
import ../child_metadata/Source.bend as Source
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../attack_geometry/Initialized.bend as I
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as C

def joined(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+white: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},good: {B.valid(b) == True{} : Bool},+canonical: {C.get_turn(b) == Bool.to_u32(white) : U32},parent_ep: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  parts: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b))) & E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> Metadata.child(b,m,white) & {EP.child(b,m) == True{} : Bool}:
  (for_metadata,for_ep)=parts
  (Metadata.generated(b,m,white,good,canonical,Source.member(I.run(depth,seed,n,key,at,extra),b,m,for_metadata)),
    EP.total(depth,seed,n,key,at,extra,b,m,white,depth_ok,full,canonical,for_ep))

# Parent partition, canonical turn, and parent EP coherence stay explicit in this
# composed metadata contract. EP.total alone needs canonical turn and initialized
# full-Ply origin; it also handles the harmless flag1 double-step branch.
def legal_member(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+white: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},good: {B.valid(b) == True{} : Bool},canonical: {C.get_turn(b) == Bool.to_u32(white) : U32},parent_ep: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> Metadata.child(b,m,white) & {EP.child(b,m) == True{} : Bool}:
  joined(depth,seed,n,key,at,extra,b,m,white,depth_ok,full,good,canonical,parent_ep,
    Prior.duplicate(E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)),m,here))
