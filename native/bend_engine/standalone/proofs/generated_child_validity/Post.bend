# Actual full-Ply representation preservation from generated field certificates.
import Base
import ../board/Spec.bend as B
import ../candidate_bypass_classification/Fields.bend as Fields
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Castle
import ../move_update/consumer.bend as Ordinary
import ../en_passant/consumer.bend as EP
import ../promotion/consumer.bend as Promotion
import ../promotion/Spec.bend as P
import ../../../legal_probe/Chess.bend as Chess

def valid(-b: Chess.Board,-m: Chess.Ply) -> Data:
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}

def plain(+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},fields: Fields.plain(m)) -> valid(b,m):
  (+src,(+dst,same))=fields
  %Equal.sym(Chess.Ply,m,Chess.Ply{src,dst,0,0},same) : valid(b,_)
  Ordinary.consistent_move(b,src,dst,good)

def ep(+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},fields: Fields.ep(m)) -> valid(b,m):
  (+src,(+dst,same))=fields
  %Equal.sym(Chess.Ply,m,Chess.Ply{src,dst,0,1},same) : valid(b,_)
  EP.consistent(b,src,dst,good)

def promoted(+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},fields: Fields.promoted(m)) -> valid(b,m):
  (+src,(+dst,(+choice,same)))=fields
  %Equal.sym(Chess.Ply,m,P.ply(src,dst,choice),same) : valid(b,_)
  Promotion.arbitrary_promotion(b,src,dst,choice,good)

def scanned(+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},fields: Fields.scanned(m)) -> valid(b,m):
  match fields:
    case Inl{ordinary}: plain(b,m,good,ordinary)
    case Inr{other}:
      match other:
        case Inl{capture}: ep(b,m,good,capture)
        case Inr{promotion}: promoted(b,m,good,promotion)

def generated(+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},fields: Fields.generated(b,m)) -> valid(b,m):
  match fields:
    case Inl{scan}: scanned(b,m,good,scan)
    case Inr{castle}: Castle.preserve_certificate(b,m,good,castle)

def occurrences(+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  +good: {B.valid(b) == True{} : Bool},
  fields: C.every(m => Fields.generated(b,m),xs)) -> C.every(m => valid(b,m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,+rest}:
      (head,tail)=fields
      (generated(b,m,good,head),occurrences(b,rest,good,tail))

