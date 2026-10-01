# Effective per-session UCI resources. No board/search/model state lives here.
import Base
import ./Text.bend as Text

type Options is Data:
  Options{tree_nodes: U32, overhead_ms: U32}

def defaults() -> Options:
  Options{4096, 50}

def tree_nodes(options: Options) -> U32:
  Options{nodes, overhead} = options
  nodes

def overhead(options: Options) -> U32:
  Options{nodes, ms} = options
  ms

def advertise() -> IO(Unit):
  do IO<Unit>:
    IO.print("option name TreeNodes type spin default 4096 min 4096 max 65536")
    IO.print("option name Move Overhead type spin default 50 min 0 max 5000")

def selected(ok: Bool, options: Options) -> Maybe<&2, Options>:
  match ok:
    case False{}: None{}
    case True{}: Some{options}

def value(which: U32, number: Maybe<&2, U32>, options: Options) -> Maybe<&2, Options>:
  match number:
    case None{}: None{}
    case Some{+n}:
      Options{nodes, overhead} = options
      match which:
        case 1: selected(Bool.and(U32.is_ge(n, 4096), U32.is_le(n, 65536)), Options{n, overhead})
        case 2: selected(U32.is_le(n, 5000), Options{nodes, n})
        case _: None{}

def named(ok: Bool, which: U32, text: String, options: Options) -> Maybe<&2, Options>:
  match ok:
    case False{}: None{}
    case True{}: value(which, Text.number(text), options)

def parse(args: List<&2, String>, options: Options) -> Maybe<&2, Options>:
  match args:
    case Con{name, Con{option, Con{value, Con{number, Nil{}}}}}:
      named(Bool.and(String.eq(name, "name"), Bool.and(String.eq(option, "TreeNodes"), String.eq(value, "value"))),
        1, number, options)
    case Con{name, Con{first, Con{second, Con{value, Con{number, Nil{}}}}}}:
      named(Bool.and(String.eq(name, "name"), Bool.and(String.eq(first, "Move"),
        Bool.and(String.eq(second, "Overhead"), String.eq(value, "value")))), 2, number, options)
    case _: None{}

def show(options: Options) -> String:
  Options{nodes, overhead} = options
  "session_options TreeNodes=" ++ U32.show(nodes) ++ " MoveOverhead=" ++ U32.show(overhead)
