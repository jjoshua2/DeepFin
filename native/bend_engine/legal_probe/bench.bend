# Opt-in single-thread measurement. No output or fixture work inside the timer.
import Base
import ./Chess.bend as Chess

type Job is Type:
  Job{board: Chess.Board, table: Array<U64>, depth: U32, mode: U32, seed: U32}

def Job.load() -> IO(Job):
  import "./input.c"

def Bench.start() -> IO(Unit):
  import "./bench_clock.c"

def Bench.finish(table: Array<U64>, depth: U32, hi: U32, lo: U32) -> IO(Unit):
  import "./bench_clock.c"

def Bench.warmup(hi: U32, lo: U32) -> IO(Unit):
  import "./bench_clock.c"

def finish(depth: U32, r: Array<U64> & U64) -> IO(Unit):
  (table, +count) = r
  Bench.finish(table, depth, U64.high(count), U64.low(count))

def measured(+b: Chess.Board, +depth: U32, r: Array<U64> & U64) -> IO(Unit):
  (table, +warmup) = r
  do IO<Unit>:
    Bench.warmup(U64.high(warmup), U64.low(warmup))
    Bench.start()
    finish(depth, Chess.perft(U32.to_nat(depth), [b], table, U64.zero()))

def validated(+b: Chess.Board, +depth: U32, r: Array<U64> & Bool) -> IO(Unit):
  (table, bad) = r
  match bad:
    case True{}: IO.die(Unit, 2, "invalid input: nonmoving king is attacked")
    case False{}:
      measured(b, depth, Chess.perft(U32.to_nat(depth), [b], table, U64.zero()))

def run(job: Job) -> IO(Unit):
  Job{+b, table, depth, mode, seed} = job
  validated(b, depth, Chess.in_check(table, b, U32.xor(Chess.get_turn(b), 1)))

def main() -> IO(Unit):
  do IO<Unit>:
    job : Job <- Job.load()
    run(job)
