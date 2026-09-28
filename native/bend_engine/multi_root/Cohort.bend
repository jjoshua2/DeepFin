# Bounded single-owner multi-root search. One prepared leaf per root per sweep.
# No host scheduler, shared tree, asynchronous return or cached history shortcut.
import Base
import ../legal_probe/Chess.bend as Chess
import ../batch_backend/Batch.bend as B
import ../session_probe/Search.bend as S
import ../standalone/Position.bend as Pos
import ../standalone/Policy.bend as Policy
import ../standalone/EvaluationInput.bend as E
import ../standalone/ModelInput.bend as Input
import ../standalone/NativeEvaluation.bend as N
import ../standalone/LogitReply.bend as L
import ../standalone/SearchHistory.bend as History
import ../standalone/Rules.bend as Rules

# Immutable configuration; all roots use the one bound model and encoding.
type Config is Data:
  Config{sims: U32, depth: U32, eval_limit: U32, diagnostics: Bool}

type Root is Type:
  Root{game: Pos.Game, tree: S.Tree, neural: U32, rule_draws: U32}

type Store is Type:
  Store{table: Array<U64>, maps: Policy.Maps, scratch: Array<F32>, logits: Array<F32>, input: Array<F32>, output: Array<F32>, cap: B.Capabilities}

# The ticket, exact legal entries and owning root travel together. A row number
# is never guessed from a root's position in the queue or a matching board hash.
type Task is Type:
  Local{root: Root}
  Pending{root: Root, game: Pos.Game, ticket: S.Ticket, entries: List<&2, Policy.Entry>, row: U32}

type Gathered is Type:
  Gathered{tail: List<Root>, tasks: List<Task>, store: Store, rows: U32, steps: U32}

def copy_seen(dst: Array<F32>, at: U32, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (src, value) = r
  (src, Array.set(F32, dst, at, value))

def copy(fuel: Nat, +src_at: U32, +dst_at: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match fuel:
    case 0n: r
    case 1n+p:
      (src, dst) = r
      copy(p, U32.inc(src_at), U32.inc(dst_at), copy_seen(dst, dst_at, Array.get(F32, src, src_at)))

def active(cfg: Config, root: Root) -> Root & Bool:
  Config{sims, depth, +budget, diagnostics} = cfg
  Root{game, S.Tree{pool, used, cap, +n, pending, epoch, request, +stop, lim}, +neural, draws} = root
  ok = Bool.and(U32.is_zero(stop), Bool.and(U32.is_lt(n, sims), Bool.or(U32.is_zero(budget), U32.is_lt(neural, budget))))
  (Root{game, S.Tree{pool, used, cap, n, pending, epoch, request, stop, lim}, neural, draws}, ok)

def rule_reply(ticket: S.Ticket) -> S.Reply:
  S.Ticket{epoch, request, node, board, moves, needs} = ticket
  S.Reply{epoch, request, node, 3, 0.0, 1.0, 0.0, 0, Array.new(F32, 8n, 0.0)}

def row_copied(root: Root, game: Pos.Game, ticket: S.Ticket, entries: List<&2, Policy.Entry>,
  tail: List<Root>, tasks: List<Task>, +row: U32, steps: U32, table: Array<U64>, maps: Policy.Maps,
  logits: Array<F32>, output: Array<F32>, cap: B.Capabilities, r: Array<F32> & Array<F32>) -> Gathered:
  (scratch, input) = r
  Gathered{tail, Con{Pending{root, game, ticket, entries, row}, tasks},
    Store{table, maps, scratch, logits, input, output, cap}, U32.inc(row), steps}

def cap_channels(cap: B.Capabilities) -> U32:
  B.Capabilities{profile, batch, channels, ilog, olog} = cap
  channels

def cap_profile(cap: B.Capabilities) -> U32:
  B.Capabilities{profile, batch, channels, ilog, olog} = cap
  profile

def encoded_checked(ok: Bool, root: Root, game: Pos.Game, ticket: S.Ticket,
  entries: List<&2, Policy.Entry>, tail: List<Root>, tasks: List<Task>, +row: U32, steps: U32,
  table: Array<U64>, maps: Policy.Maps, scratch: Array<F32>, logits: Array<F32>, input: Array<F32>,
  output: Array<F32>, cap: B.Capabilities, +width: U32) -> IO(Gathered):
  match ok:
    case False{}: IO.die(Gathered, 2, "multi-root ticket/encoding mismatch")
    case True{}:
      IO.pure(Gathered, row_copied(root, game, ticket, entries, tail, tasks, row, steps, table, maps,
        logits, output, cap, copy(U32.to_nat(width), 0, U32.mul(row, width), (scratch, input))))

def ticket_moves(t: S.Ticket) -> List<&2, Chess.Ply>:
  S.Ticket{epoch, request, node, board, moves, needs} = t
  moves

def encoded(root: Root, game: Pos.Game, +ticket: S.Ticket, tail: List<Root>, tasks: List<Task>, row: U32, steps: U32,
  logits: Array<F32>, input: Array<F32>, output: Array<F32>, +cap: B.Capabilities, prepared: E.Preparation) -> IO(Gathered):
  match prepared:
    case E.Invalid{table, maps}: IO.die(Gathered, 2, "multi-root leaf encoding failed")
    case E.Ready{E.Prepared{maps, Input.Input{table, scratch, +channels}, +entries}}:
      encoded_checked(Bool.and(U32.is_eq(channels, cap_channels(cap)), L.aligned(entries, ticket_moves(ticket))),
        root, game, ticket, entries, tail, tasks, row, steps, table, maps, scratch, logits, input, output, cap, U32.mul(channels, 64))

def rule_checked(ok: Bool, root: Root, tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, store: Store) -> IO(Gathered):
  match ok:
    case False{}: IO.die(Gathered, 2, "multi-root rule reply was not accepted")
    case True{}: IO.pure(Gathered, Gathered{tail, Con{Local{root}, tasks}, store, row, steps})

def rule_applied(root: Pos.Game, neural: U32, draws: U32, tail: List<Root>, tasks: List<Task>,
  row: U32, steps: U32, store: Store, before: U32, tree: S.Tree) -> IO(Gathered):
  S.Tree{pool, used, cap, +n, pending, epoch, request, +stop, lim} = tree
  rule_checked(Bool.and(U32.is_zero(stop), U32.is_eq(n, U32.inc(before))),
    Root{root, S.Tree{pool, used, cap, n, pending, epoch, request, stop, lim}, neural, U32.inc(draws)}, tail, tasks, row, steps, store)

def rule_notice(enabled: Bool, ep: U32, req: U32, node: U32, code: U32) -> IO(Unit):
  match enabled:
    case False{}: IO.pure(Unit, Unit{})
    case True{}: IO.print("info string cohort_rule " ++ U32.show(ep) ++ " " ++ U32.show(req) ++ " " ++ U32.show(node) ++ " " ++ U32.show(code))

def apply_rule(cfg: Config, root: Root, +ticket: S.Ticket, tail: List<Root>, tasks: List<Task>,
  row: U32, steps: U32, store: Store, code: U32) -> IO(Gathered):
  Config{sims, depth, budget, diagnostics} = cfg
  Root{g, S.Tree{pool, used, capacity, +before, pending, epoch, request, stop, lim}, neural, draws} = root
  S.Ticket{ep, req, node, board, moves, needs} = ticket
  do IO<Gathered>:
    rule_notice(diagnostics, ep, req, node, code)
    rule_applied(g, neural, draws, tail, tasks, row, steps, store, before,
      S.resume(S.Tree{pool, used, capacity, before, pending, epoch, request, stop, lim}, ticket, rule_reply(ticket)))

def adjudicated(cfg: Config, root: Root, +game: Pos.Game, ticket: S.Ticket,
  tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, maps: Policy.Maps,
  scratch: Array<F32>, logits: Array<F32>, input: Array<F32>, output: Array<F32>, +cap: B.Capabilities,
  r: Array<U64> & Rules.Facts) -> IO(Gathered):
  (table, Rules.Facts{code, count, half}) = r
  match code:
    case 0:
      encoded(root, game, ticket, tail, tasks, row, steps, logits, input, output, cap,
        E.prepare_into(N.layout(cap_profile(cap)), N.version(cap_profile(cap)), game, table, maps, Array.size(F32, scratch)))
    case _:
      apply_rule(cfg, root, ticket, tail, tasks, row, steps, Store{table, maps, scratch, logits, input, output, cap}, code)

def history_checked(ok: Bool, cfg: Config, root: Pos.Game, neural: U32, draws: U32, +ticket: S.Ticket,
  tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, store: Store, tree: S.Tree, +game: Pos.Game) -> IO(Gathered):
  match ok:
    case False{}: IO.die(Gathered, 2, "multi-root invalid leaf history")
    case True{}:
      Store{table, maps, scratch, logits, input, output, cap} = store
      adjudicated(cfg, Root{root, tree, neural, draws}, game, ticket, tail, tasks, row, steps, maps,
        scratch, logits, input, output, cap, Rules.automatic(game, ticket_moves(ticket), table))

def rebuilt(cfg: Config, root: Pos.Game, neural: U32, draws: U32, ticket: S.Ticket,
  tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, store: Store, rebuilt: History.Rebuilt) -> IO(Gathered):
  History.Rebuilt{tree, game, valid} = rebuilt
  history_checked(valid, cfg, root, neural, draws, ticket, tail, tasks, row, steps, store, tree, game)

def prepared_needs(needs: Bool, cfg: Config, +game: Pos.Game, neural: U32, draws: U32, tail: List<Root>, tasks: List<Task>,
  row: U32, steps: U32, store: Store, tree: S.Tree, ticket: S.Ticket, node: U32, board: Chess.Board) -> IO(Gathered):
  match needs:
    case False{}: IO.pure(Gathered, Gathered{tail, Con{Local{Root{game, tree, neural, draws}}, tasks}, store, row, steps})
    case True{}: rebuilt(cfg, game, neural, draws, ticket, tail, tasks, row, steps, store, History.rebuild(game, node, board, tree))

def prepared(cfg: Config, game: Pos.Game, neural: U32, draws: U32, tail: List<Root>, tasks: List<Task>, row: U32, steps: U32,
  maps: Policy.Maps, scratch: Array<F32>, logits: Array<F32>, input: Array<F32>, output: Array<F32>, cap: B.Capabilities, p: S.Prepared) -> IO(Gathered):
  S.Prepared{table, tree, +ticket} = p
  S.Ticket{epoch, request, node, board, moves, needs} = ticket
  prepared_needs(needs, cfg, game, neural, draws, tail, tasks, row, steps,
    Store{table, maps, scratch, logits, input, output, cap}, tree, ticket, node, board)

def visit_active(enabled: Bool, cfg: Config, root: Root, tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, store: Store) -> IO(Gathered):
  match enabled:
    case False{}: IO.pure(Gathered, Gathered{tail, Con{Local{root}, tasks}, store, row, steps})
    case True{}:
      Root{game, tree, neural, draws} = root
      Store{table, maps, scratch, logits, input, output, cap} = store
      prepared(cfg, game, neural, draws, tail, tasks, row, U32.inc(steps), maps, scratch, logits, input, output, cap, S.prepare(table, tree))

def visit(cfg: Config, tail: List<Root>, tasks: List<Task>, row: U32, steps: U32, store: Store, r: Root & Bool) -> IO(Gathered):
  (root, enabled) = r
  visit_active(enabled, cfg, root, tail, tasks, row, steps, store)

def gather_next(full: Bool, +cfg: Config, g: Gathered) -> IO(Gathered):
  match full:
    case True{}: IO.pure(Gathered, g)
    case False{}:
      Gathered{tail, tasks, store, row, steps} = g
      match tail:
        case Nil{}: IO.pure(Gathered, Gathered{Nil{}, tasks, store, row, steps})
        case Con{root, rest}: visit(cfg, rest, tasks, row, steps, store, active(cfg, root))

def gather(fuel: Nat, +cfg: Config, +batch: U32, g: Gathered) -> IO(Gathered):
  match fuel:
    case 0n: IO.pure(Gathered, g)
    case 1n+p:
      Gathered{tail, tasks, store, +row, steps} = g
      do IO<Gathered>:
        g : Gathered <- gather_next(U32.is_eq(row, batch), cfg, Gathered{tail, tasks, store, row, steps})
        gather(p, cfg, batch, g)

# Validate ALL real raw rows before any neural reply is normalized or backed up.
# Backend delivery does not constitute acceptance. Each unchanged Search.resume
# must increment its own tree's completed count exactly once.
def raw_valid(rows: U32, output: Array<F32>) -> Array<F32> & Bool:
  L.finite_buffer(U32.to_nat(U32.mul(rows, 1861)), 0, (output, True{}))

def accepted_checked(ok: Bool, root: Root) -> IO(Root):
  match ok:
    case False{}: IO.die(Root, 2, "multi-root neural reply was not accepted")
    case True{}: IO.pure(Root, root)

def accepted_root(game: Pos.Game, neural: U32, draws: U32, before: U32, tree: S.Tree) -> IO(Root):
  S.Tree{pool, used, cap, +n, pending, epoch, request, +stop, lim} = tree
  accepted_checked(Bool.and(U32.is_zero(stop), U32.is_eq(n, U32.inc(before))),
    Root{game, S.Tree{pool, used, cap, n, pending, epoch, request, stop, lim}, U32.inc(neural), draws})

def resumed(root: Pos.Game, neural: U32, draws: U32, before: U32, tree: S.Tree, ticket: S.Ticket,
  maps: Policy.Maps, scratch: Array<F32>, logits: Array<F32>, input: Array<F32>, output: Array<F32>, cap: B.Capabilities,
  r: Array<U64> & S.Reply) -> IO(Store & Root):
  (table, reply) = r
  do IO<Store & Root>:
    root : Root <- accepted_root(root, neural, draws, before, S.resume(tree, ticket, reply))
    return (Store{table, maps, scratch, logits, input, output, cap}, root)

def converted(cfg: Config, root: Root, game: Pos.Game, ticket: S.Ticket, entries: List<&2, Policy.Entry>, table: Array<U64>,
  maps: Policy.Maps, scratch: Array<F32>, input: Array<F32>, output: Array<F32>, cap: B.Capabilities, c: L.Converted) -> IO(Store & Root):
  Config{sims, depth, budget, diagnostics} = cfg
  Root{+g, S.Tree{pool, used, capacity, +before, pending, epoch, request, stop, lim}, neural, draws} = root
  L.Converted{logits, reply} = c
  do IO<Store & Root>:
    r : Array<U64> & S.Reply <- N.notice(g, game, entries, table, diagnostics, reply)
    resumed(g, neural, draws, before, S.Tree{pool, used, capacity, before, pending, epoch, request, stop, lim}, ticket,
      maps, scratch, logits, input, output, cap, r)

def row_read(cfg: Config, root: Root, game: Pos.Game, +ticket: S.Ticket, +entries: List<&2, Policy.Entry>,
  table: Array<U64>, maps: Policy.Maps, scratch: Array<F32>, input: Array<F32>, cap: B.Capabilities, r: Array<F32> & Array<F32>) -> IO(Store & Root):
  (output, logits) = r
  converted(cfg, root, game, ticket, entries, table, maps, scratch, input, output, cap,
    L.convert_buffer(ticket, entries, 1861, Array.size(F32, logits)))

def apply_task(cfg: Config, task: Task, store: Store) -> IO(Store & Root):
  match task:
    case Local{root}: IO.pure(Store & Root, (store, root))
    case Pending{root, game, ticket, entries, row}:
      Store{table, maps, scratch, logits, input, output, cap} = store
      row_read(cfg, root, game, ticket, entries, table, maps, scratch, input, cap,
        copy(1861n, U32.mul(row, 1861), 0, (output, logits)))

def scatter_returned(roots: List<Root>, r: Store & Root) -> Store & List<Root>:
  (store, root) = r
  (store, Con{root, roots})

def scatter(+cfg: Config, tasks: List<Task>, r: Store & List<Root>) -> IO(Store & List<Root>):
  match tasks:
    case Nil{}:
      (store, roots) = r
      IO.pure(Store & List<Root>, (store, List.reverse(&1, Root, roots)))
    case Con{task, rest}:
      (store, roots) = r
      do IO<Store & List<Root>>:
        result : Store & Root <- apply_task(cfg, task, store)
        scatter(cfg, rest, scatter_returned(roots, result))
