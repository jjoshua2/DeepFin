import Base
import ./Spec.bend as S
import ./Bridge.bend as Bridge
import ../attack_geometry/Initialized.bend as I
import ../initialized_pawn_double/Spec.bend as D
import ../child_metadata/Range.bend as R
import ../child_metadata/Post.bend as Post
import ../child_metadata/consumer.bend as Prior
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as Board
import ../../../legal_probe/Chess.bend as C

def generated_ep_fields(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},+trigger: {Bridge.ep_trigger(b,m) == True{} : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) ->
  D.double(b,R.src(m),S.dst(m)) & {C.get_ep(C.make_move(b,m)) == Bridge.midpoint(m) : U32}:
  (Bridge.initialized(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,here),Bridge.tag(b,m,trigger))

# Partition and canonical turn belong to the inherited child metadata contract.
# They are separate from root validation and from the initialized array premises.
def combine(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+white: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},good: {Board.valid(b) == True{} : Bool},
  canonical: {C.get_turn(b) == Bool.to_u32(white) : U32},
  parts: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b))) & E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) ->
  Post.child(b,m,white) & ({Bridge.ep_trigger(b,m) == True{} : Bool} ->
    (D.double(b,R.src(m),S.dst(m)) & {C.get_ep(C.make_move(b,m)) == Bridge.midpoint(m) : U32})):
  (old,new)=parts
  (Prior.legal_member(I.run(depth,seed,n,key,at,extra),b,m,white,good,canonical,old),
    trigger => generated_ep_fields(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,new))

def legal_member(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+white: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},good: {Board.valid(b) == True{} : Bool},
  canonical: {C.get_turn(b) == Bool.to_u32(white) : U32},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) ->
  Post.child(b,m,white) & ({Bridge.ep_trigger(b,m) == True{} : Bool} ->
    (D.double(b,R.src(m),S.dst(m)) & {C.get_ep(C.make_move(b,m)) == Bridge.midpoint(m) : U32})):
  combine(depth,seed,n,key,at,extra,b,m,white,depth_ok,full,good,canonical,
    Bridge.duplicate(E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)),m,here))
