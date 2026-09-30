import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

def finish_bits(+m: Chess.Ply,+start_check: Bool,+tail: List<&2,Chess.Ply>,table: Array<U64>,+transit: Bool) ->
  S.growth(True{},m,tail,Chess.castle_finish(m,start_check,tail,(table,transit))):
  match start_check transit:
    case True{} True{}: Inl{{==}}
    case True{} False{}: Inl{{==}}
    case False{} True{}: Inl{{==}}
    case False{} False{}: Inr{({==},{==})}

def finish(+m: Chess.Ply,+start_check: Bool,+tail: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  S.growth(True{},m,tail,Chess.castle_finish(m,start_check,tail,r)):
  (table,transit) = r
  finish_bits(m,start_check,tail,table,transit)

def start(+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & Bool) ->
  S.growth(True{},Chess.Ply{src,dst,0,2},tail,Chess.castle_start(b,src,dst,transit,tail,r)):
  (table,check) = r
  finish(Chess.Ply{src,dst,0,2},check,tail,
    Chess.in_check(table,Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b)))

def checked(+ok: Bool,table: Array<U64>,+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+tail: List<&2,Chess.Ply>) ->
  S.growth(ok,Chess.Ply{src,dst,0,2},tail,Chess.castle_checked(ok,table,b,src,dst,transit,tail)):
  match ok:
    case False{}: Inl{{==}}
    case True{}: start(b,src,dst,transit,tail,Chess.in_check(table,b,Chess.get_turn(b)))

def side(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>) ->
  S.growth(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail))):
  checked(S.guard(b,ks),table,b,S.src(b),S.dst(b,ks),S.transit(b,ks),tail)

def rejected(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,e: {S.guard(b,ks) == False{} : Bool}) ->
  {Chess.castle_side(b,ks,(table,tail)) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Bool,S.guard(b,ks),False{},e) :
    {Chess.castle_checked(_,table,b,S.src(b),S.dst(b,ks),S.transit(b,ks),tail) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}
  {==}
