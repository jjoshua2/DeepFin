# Forced cases are removed using the actual check selector, not raw flag labels.
import Base
import ./Fields.bend as Fields
import ./Generated.bend as Generated
import ../castle_safety/Spec.bend as Sensitive
import ../castle_safety/Mask.bend as Mask
import ../castle_emission/Spec.bend as E
import ../u64/Words.bend as Words
import ../../../legal_probe/Chess.bend as Chess

def ep_required(+m: Chess.Ply,+sensitive: U64,cert: Fields.ep(m)) ->
  {Chess.filter_requires(sensitive,m) == True{} : Bool}:
  (+src,(+dst,same)) = cert
  %Equal.sym(Chess.Ply,m,Chess.Ply{src,dst,0,1},same) :
    {Chess.filter_requires(sensitive,_) == True{} : Bool}
  {==}

def scanned(+m: Chess.Ply,+sensitive: U64,cert: Fields.scanned(m),
  bypass: {Chess.filter_requires(sensitive,m) == False{} : Bool}) -> Fields.bypass_shape(m):
  match cert:
    case Inl{plain}: Inl{plain}
    case Inr{special}:
      match special:
        case Inl{ep}:
          Words.false_true(Fields.bypass_shape(m),Equal.trans(Bool,False{},Chess.filter_requires(sensitive,m),True{},
            Equal.sym(Bool,Chess.filter_requires(sensitive,m),False{},bypass),ep_required(m,sensitive,ep)))
        case Inr{promoted}: Inr{promoted}

def generated(+b: Chess.Board,+rays: U64,+m: Chess.Ply,cert: Fields.generated(b,m),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),m) == False{} : Bool}) -> Fields.bypass_shape(m):
  match cert:
    case Inl{scan}: scanned(m,Sensitive.sensitive(b,rays),scan,bypass)
    case Inr{castle}:
      Words.false_true(Fields.bypass_shape(m),Equal.trans(Bool,False{},Chess.filter_requires(Sensitive.sensitive(b,rays),m),True{},
        Equal.sym(Bool,Chess.filter_requires(Sensitive.sensitive(b,rays),m),False{},bypass),Mask.certificate(b,m,rays,castle)))

def prefilter(table: Array<U64>,+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  here: E.member(m,E.moves(Generated.produced(table,b))),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),m) == False{} : Bool}) -> Fields.bypass_shape(m):
  generated(b,rays,m,Generated.member_prefilter(table,b,m,here),bypass)

def legal(table: Array<U64>,+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),m) == False{} : Bool}) -> Fields.bypass_shape(m):
  generated(b,rays,m,Generated.member_legal(table,b,m,here),bypass)
