import Base
import ./Actual.bend as A
import ../color_counts/Actual.bend as Sources
import ../color_counts/Emission.bend as Prior
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  zero: {Chess.get_rights(b) == 0 : U32},here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> A.decreases(b,m):
  A.post(b,m,Prior.legal(z => E.member(m,E.moves(z)) -> Sources.emitted(b,m),table,b,zero,
    z => pz => Lists.extract(m => Sources.emitted(b,m),z,m,pz))(here))
