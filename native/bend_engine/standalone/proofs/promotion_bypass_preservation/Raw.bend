import Base
import ../promotion/Spec.bend as P
import ../promotion/Actual.bend as Actual
import ../promotion/Encoding.bend as Encoding
import ../move_update/Ordinary.bend as Ordinary
import ../ordinary_king_frame/Frame.bend as F
import ../ray/Slider.bend as Slider
import ../castle_safety/Spec.bend as Sensitive
import ../../../legal_probe/Chess.bend as Chess

def moved(+b: Chess.Board,src: U32,dst: U32,choice: P.Choice) -> Chess.Board:
  Chess.make_move(b,P.ply(src,dst,choice))

# The existing actual promotion producer supplies selection; no shadow candidate.
def unfold(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice) ->
  {moved(b,src,dst,choice) == Actual.dispatch(b,src,dst,P.tag(choice)) : Chess.Board}:
  Equal.trans(Chess.Board,moved(b,src,dst,choice),
    Actual.dispatch(b,src,dst,Bool.pick(U32,U32.is_zero(P.tag(choice)),Chess.piece(b,src),P.tag(choice))),
    Actual.dispatch(b,src,dst,P.tag(choice)),Actual.unfold(b,src,dst,choice),
    Equal.cong(U32,Chess.Board,put => Actual.dispatch(b,src,dst,put),
      Bool.pick(U32,U32.is_zero(P.tag(choice)),Chess.piece(b,src),P.tag(choice)),P.tag(choice),
      Encoding.selected(Chess.piece(b,src),choice)))

def dispatch_color(+b: Chess.Board,+src: U32,+dst: U32,+put: U32,+side: U32) ->
  {Chess.color(Actual.dispatch(b,src,dst,put),side) ==
    Chess.color(Ordinary.dispatch(b,src,dst,Chess.piece(b,src)),side) : U64}:
  match b:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep}: {==}

# Promotion changes a friendly kind; actual color planes and occupancy agree.
def colors(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+side: U32) ->
  {Chess.color(moved(b,src,dst,choice),side) == Chess.color(F.moved(b,src,dst),side) : U64}:
  Equal.trans(U64,Chess.color(moved(b,src,dst,choice),side),
    Chess.color(Actual.dispatch(b,src,dst,P.tag(choice)),side),Chess.color(F.moved(b,src,dst),side),
    Equal.cong(Chess.Board,U64,next => Chess.color(next,side),moved(b,src,dst,choice),
      Actual.dispatch(b,src,dst,P.tag(choice)),unfold(b,src,dst,choice)),
    Equal.trans(U64,Chess.color(Actual.dispatch(b,src,dst,P.tag(choice)),side),
      Chess.color(Ordinary.dispatch(b,src,dst,Chess.piece(b,src)),side),Chess.color(F.moved(b,src,dst),side),
      dispatch_color(b,src,dst,P.tag(choice),side),
      Equal.sym(U64,Chess.color(F.moved(b,src,dst),side),
        Chess.color(Ordinary.dispatch(b,src,dst,Chess.piece(b,src)),side),
        Equal.cong(Chess.Board,U64,next => Chess.color(next,side),F.moved(b,src,dst),
          Ordinary.dispatch(b,src,dst,Chess.piece(b,src)),Ordinary.unfold(b,src,dst)))))

def occupancy(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice) ->
  {Chess.occupied(moved(b,src,dst,choice)) == Chess.occupied(F.moved(b,src,dst)) : U64}:
  Slider.combine(Chess.get_white(moved(b,src,dst,choice)),Chess.get_white(F.moved(b,src,dst)),
    Chess.get_black(moved(b,src,dst,choice)),Chess.get_black(F.moved(b,src,dst)),
    colors(b,src,dst,choice,1),colors(b,src,dst,choice,0))

def nonking(+choice: P.Choice) -> {U32.is_eq(P.tag(choice),5) == False{} : Bool}:
  match choice:
    case P.Knight{}: {==}
    case P.Bishop{}: {==}
    case P.Rook{}: {==}
    case P.Queen{}: {==}

def kings_update(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice) ->
  {Chess.get_kings(moved(b,src,dst,choice)) ==
    Chess.update_piece(Chess.get_kings(b),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),5)) : U64}:
  match b:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,Chess.get_kings,moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),unfold(board,src,dst,choice))

def kings_erased(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice) ->
  {Chess.get_kings(moved(b,src,dst,choice)) == U64.and_not(Chess.get_kings(b),F.removal(src,dst)) : U64}:
  Equal.trans(U64,Chess.get_kings(moved(b,src,dst,choice)),
    Chess.update_piece(Chess.get_kings(b),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),5)),
    U64.and_not(Chess.get_kings(b),F.removal(src,dst)),kings_update(b,src,dst,choice),
    F.no_put(Chess.get_kings(b),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),5),nonking(choice)))

def kings_equal(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,
  not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {Chess.get_kings(moved(b,src,dst,choice)) == Chess.get_kings(F.moved(b,src,dst)) : U64}:
  Equal.trans(U64,Chess.get_kings(moved(b,src,dst,choice)),U64.and_not(Chess.get_kings(b),F.removal(src,dst)),
    Chess.get_kings(F.moved(b,src,dst)),kings_erased(b,src,dst,choice),
    Equal.sym(U64,Chess.get_kings(F.moved(b,src,dst)),U64.and_not(Chess.get_kings(b),F.removal(src,dst)),
      Equal.trans(U64,Chess.get_kings(F.moved(b,src,dst)),
        Chess.update_piece(Chess.get_kings(b),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5)),
        U64.and_not(Chess.get_kings(b),F.removal(src,dst)),F.kings_update(b,src,dst),
        F.no_put(Chess.get_kings(b),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5),not_king))))

def plane_equal(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+side: U32,
  not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {F.plane(moved(b,src,dst,choice),side) == F.plane(F.moved(b,src,dst),side) : U64}:
  Equal.trans(U64,F.plane(moved(b,src,dst,choice),side),
    U64.and(Chess.get_kings(F.moved(b,src,dst)),Chess.color(moved(b,src,dst,choice),side)),
    F.plane(F.moved(b,src,dst),side),
    Equal.cong(U64,U64,kings => U64.and(kings,Chess.color(moved(b,src,dst,choice),side)),
      Chess.get_kings(moved(b,src,dst,choice)),Chess.get_kings(F.moved(b,src,dst)),kings_equal(b,src,dst,choice,not_king)),
    Equal.cong(U64,U64,own => U64.and(Chess.get_kings(F.moved(b,src,dst)),own),
      Chess.color(moved(b,src,dst,choice),side),Chess.color(F.moved(b,src,dst),side),colors(b,src,dst,choice,side)))

def bypass(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,+choice: P.Choice) ->
  {Chess.filter_requires(Sensitive.sensitive(b,rays),P.ply(src,dst,choice)) ==
    Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) : Bool}:
  {==}

# EP cannot bypass for any source/mask/promotion; flag-two alone is not forced.
def ep_requires(+sensitive: U64,+src: U32,+dst: U32,+promotion: U32) ->
  {Chess.filter_requires(sensitive,Chess.Ply{src,dst,promotion,1}) == True{} : Bool}:
  {==}
