# One evaluated position owns BOTH complete feature input and legal policy slots.
# No inference/softmax here. Callers retain Maps and the attack table across calls.
# Game must be validated by Position/Protocol or SearchHistory before preparation.
import Base
import ../legal_probe/Chess.bend as Chess
import ./Position.bend as Pos
import ./HistoryEncoding.bend as H
import ./ModelInput.bend as Input
import ./Policy.bend as Policy

type Prepared is Type:
  Prepared{maps: Policy.Maps, input: Input.Input, entries: List<&2, Policy.Entry>}

type Preparation is Type:
  Invalid{table: Array<U64>, maps: Policy.Maps}
  Ready{prepared: Prepared}

def mapped(layout: H.Layout, version: Input.Version, game: Pos.Game,
  table: Array<U64>, r: Policy.Maps & Maybe<&2, List<&2, Policy.Entry>>) -> Preparation:
  (maps, entries) = r
  match entries:
    case None{}: Invalid{table, maps}
    case Some{entries}: Ready{Prepared{maps, Input.encode(layout, version, game, table), entries}}

def generated(layout: H.Layout, version: Input.Version, +game: Pos.Game,
  maps: Policy.Maps, r: Array<U64> & List<&2, Chess.Ply>) -> Preparation:
  (table, moves) = r
  mapped(layout, version, game, table, Policy.entries(Chess.get_turn(Pos.board(game)), moves, maps))

def checked(ok: Bool, layout: H.Layout, version: Input.Version, +game: Pos.Game,
  table: Array<U64>, maps: Policy.Maps) -> Preparation:
  match ok:
    case False{}: Invalid{table, maps}
    case True{}: generated(layout, version, game, maps, Chess.legal_moves(table, Pos.board(game)))

def prepare(layout: H.Layout, version: Input.Version, game: Pos.Game,
  table: Array<U64>, maps: Policy.Maps) -> Preparation:
  Policy.Maps{f2c, c2f, pairs, +count} = maps
  checked(U32.is_eq(count, 1858), layout, version, game, table, Policy.Maps{f2c, c2f, pairs, count})

def mapped_into(layout: H.Layout, version: Input.Version, game: Pos.Game,
  table: Array<U64>, values: Array<F32>, r: Policy.Maps & Maybe<&2, List<&2, Policy.Entry>>) -> Preparation:
  (maps, entries) = r
  match entries:
    case None{}: Invalid{table, maps}
    case Some{entries}: Ready{Prepared{maps, Input.encode_into(layout, version, game, table, values), entries}}

def generated_into(layout: H.Layout, version: Input.Version, +game: Pos.Game,
  maps: Policy.Maps, values: Array<F32>, r: Array<U64> & List<&2, Chess.Ply>) -> Preparation:
  (table, moves) = r
  mapped_into(layout, version, game, table, values, Policy.entries(Chess.get_turn(Pos.board(game)), moves, maps))

def checked_into(ok: Bool, layout: H.Layout, version: Input.Version, +game: Pos.Game,
  table: Array<U64>, maps: Policy.Maps, values: Array<F32>) -> Preparation:
  match ok:
    case False{}: Invalid{table, maps}
    case True{}: generated_into(layout, version, game, maps, values, Chess.legal_moves(table, Pos.board(game)))

def prepare_into(layout: H.Layout, version: Input.Version, game: Pos.Game,
  table: Array<U64>, maps: Policy.Maps, r: Array<F32> & U32) -> Preparation:
  Policy.Maps{f2c, c2f, pairs, +count} = maps
  (values, capacity) = r
  checked_into(Bool.and(U32.is_eq(count, 1858), U32.is_eq(capacity, 16384)), layout,
    version, game, table, Policy.Maps{f2c, c2f, pairs, count}, values)
