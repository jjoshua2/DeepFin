import Base
import ./Spec.bend as S
import ./Origin.bend as Origin
import ../storage/Observe.bend as O
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../castle_emission/Spec.bend as E
import ../accepted_castle/Spec.bend as Prepared
import ../accepted_castle/Generated.bend as Split
import ../candidate_bypass_classification/Fields.bend as Fields
import ../candidate_bypass_classification/Scan.bend as Scan
import ../candidate_bypass_classification/Bypass.bend as Bypass
import ../ordinary_nonslider_preservation/consumer.bend as Ordinary
import ../promotion_bypass_preservation/consumer.bend as Promotion
import ../promotion/Spec.bend as P
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess

# Derive bypass safety from actual source membership and exact lookup contracts.

def plain_pairs(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  here: E.member(Chess.Ply{src,dst,0,0},S.scan(c,b)),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  old: S.old(c,b),post: S.post(c,b,Chess.Ply{src,dst,0,0}),+initial: S.initial(c,b)) ->
  Prepared.safe(c,b,Chess.Ply{src,dst,0,0}):
  (old_r,old_d)=old
  (new_r,new_d)=post
  Equal.cong(Array<U64> & Bool,Bool,Prepared.answer,
    Chess.in_check(O.pack(c),Chess.make_move(b,Chess.Ply{src,dst,0,0}),Chess.get_turn(b)),
    (O.pack(c),False{}),
    Ordinary.use_no_check(c,b,src,dst,white,here,good,nonempty,canonical,bypass,
      old_r,old_d,new_r,new_d,initial))

def promotion_pairs(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  here: E.member(P.ply(src,dst,choice),S.scan(c,b)),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),P.ply(src,dst,choice)) == False{} : Bool},
  old: S.old(c,b),post: S.post(c,b,P.ply(src,dst,choice)),+initial: S.initial(c,b)) ->
  Prepared.safe(c,b,P.ply(src,dst,choice)):
  (old_r,old_d)=old
  (new_r,new_d)=post
  Equal.cong(Array<U64> & Bool,Bool,Prepared.answer,
    Chess.in_check(O.pack(c),Chess.make_move(b,P.ply(src,dst,choice)),Chess.get_turn(b)),
    (O.pack(c),False{}),
    Promotion.use_no_check(c,b,src,dst,choice,white,here,good,nonempty,canonical,bypass,
      old_r,old_d,new_r,new_d,initial))

def plain(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  fields: Fields.plain(m),+good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b)) ->
  E.member(m,S.scan(c,b)) -> S.post(c,b,m) ->
  {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool} -> Prepared.safe(c,b,m):
  (+src,(+dst,same))=fields
  %Equal.sym(Chess.Ply,m,Chess.Ply{src,dst,0,0},same) :
    E.member(_,S.scan(c,b)) -> S.post(c,b,_) ->
    {Chess.filter_requires(F.prepared_sensitive(c,b),_) == False{} : Bool} -> Prepared.safe(c,b,_)
  here => post => bypass =>
    plain_pairs(c,b,src,dst,white,here,good,nonempty,canonical,bypass,old,post,initial)

def promoted(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  fields: Fields.promoted(m),+good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b)) ->
  E.member(m,S.scan(c,b)) -> S.post(c,b,m) ->
  {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool} -> Prepared.safe(c,b,m):
  (+src,(+dst,(+choice,same)))=fields
  %Equal.sym(Chess.Ply,m,P.ply(src,dst,choice),same) :
    E.member(_,S.scan(c,b)) -> S.post(c,b,_) ->
    {Chess.filter_requires(F.prepared_sensitive(c,b),_) == False{} : Bool} -> Prepared.safe(c,b,_)
  here => post => bypass =>
    promotion_pairs(c,b,src,dst,choice,white,here,good,nonempty,canonical,bypass,old,post,initial)

def shaped(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  fields: Fields.bypass_shape(m),here: E.member(m,S.scan(c,b)),post: S.post(c,b,m),
  bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool},
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b)) -> Prepared.safe(c,b,m):
  match fields:
    case Inl{ordinary}:
      plain(c,b,m,white,ordinary,good,nonempty,canonical,old,initial)(here)(post)(bypass)
    case Inr{promotion}:
      promoted(c,b,m,white,promotion,good,nonempty,canonical,old,initial)(here)(post)(bypass)

# Membership is duplicated by its structural certificate, not by deduplicating
# values or changing any candidate list. Every call uses the actual full Ply.

def scan_case(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  here: E.member(m,S.scan(c,b)),
  +bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool},
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookup: S.lookup(c,b,m)) -> Prepared.safe(c,b,m):
  Split.split_member(Prepared.safe(c,b,m),S.scan(c,b),m,here,fields_here => rest =>
    Split.split_member(Prepared.safe(c,b,m),S.scan(c,b),m,rest,proof_here => lookup_here =>
      shaped(c,b,m,white,
        Bypass.scanned(m,F.prepared_sensitive(c,b),Scan.member(O.pack(c),b,m,fields_here),bypass),
        proof_here,lookup(lookup_here)(bypass),bypass,
        good,nonempty,canonical,old,initial)))

def use_bypass(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  here: E.member(m,S.candidates(c,b)),
  +bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool},
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookup: S.lookup(c,b,m)) -> Prepared.safe(c,b,m):
  scan_case(c,b,m,white,Origin.scan_bypass(c,b,Prepared.rays(c,b),m,here,bypass),
    bypass,good,nonempty,canonical,old,initial,lookup)

def candidate(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  here: E.member(m,S.candidates(c,b)),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookup: S.lookup(c,b,m)) ->
  F.bypass_safe(c,b,F.prepared_sensitive(c,b),m):
  bypass => use_bypass(c,b,m,white,here,bypass,good,nonempty,canonical,old,initial,lookup)
