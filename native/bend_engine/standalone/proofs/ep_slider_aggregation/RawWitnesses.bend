# Same literal raw board as PR1050, avoiding its unrelated witness import closure.
import Base
import ./Slider.bend as Slider
import ../ep_geometric_hits/Plane.bend as Plane
import ../../../legal_probe/Chess.bend as C

def raw() -> C.Board:
  C.Board{U64.bit(36n),U64.zero(),U64.zero(),U64.or(U64.or(U64.or(U64.bit(36n),U64.bit(35n)),U64.bit(43n)),U64.bit(47n)),U64.zero(),U64.zero(),
    U64.or(U64.bit(36n),U64.bit(43n)),U64.or(U64.or(U64.or(U64.bit(36n),U64.bit(35n)),U64.bit(43n)),U64.bit(47n)),7,0,64}
def raw_kind() -> {C.piece(raw(),36) == 0 : U32}: {==}

# Arbitrary raw rows/turn and fixed query side still work in the low theorem.
def raw_old() -> {Slider.raw(40,raw(),False{},7) == U64.bit(43n) : U64}: {==}
def raw_new() -> {Slider.raw(40,Plane.moved(raw(),36,43),False{},7) == U64.zero() : U64}: {==}
def raw_subset() ->
  {U64.and(Slider.raw(40,raw(),False{},7),Slider.raw(40,Plane.moved(raw(),36,43),False{},7)) == Slider.raw(40,Plane.moved(raw(),36,43),False{},7) : U64}:
  Slider.subset(40,raw(),36,43,False{},7,
    (41n <> 42n <> 43n <> 44n <> 45n <> 46n <> 47n <> Nil{}),Nil{},(48n <> 56n <> Nil{}),(32n <> 24n <> 16n <> 8n <> 0n <> Nil{}),
    raw_kind(),{==},{==},{==},{==},
    {==},{==},{==},{==})
def out_of_range_query() ->
  {U64.and(Slider.raw(72,raw(),True{},7),Slider.raw(72,Plane.moved(raw(),36,43),True{},7)) == Slider.raw(72,Plane.moved(raw(),36,43),True{},7) : U64}:
  Slider.subset(72,raw(),36,43,True{},7,Nil{},Nil{},Nil{},Nil{},raw_kind(),
    {==},{==},{==},{==},{==},{==},{==},{==})

