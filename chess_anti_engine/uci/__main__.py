"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``. Model load +
evaluator construction run on a background thread so the ``uci`` handshake
can reply instantly; ``isready`` and later commands block until the
engine is actually ready.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import threading

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import (
    BatchCoalescingDispatcher,
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig

from .engine import Engine, EngineOptions, emit_handshake
from .model_loader import load_model_from_checkpoint
from .protocol import CmdQuit, CmdUci, parse_command
from .search import SearchWorker


def _warmup_evaluator(
    evaluator,
    *,
    n_walkers: int = 1,
    walker_gather: int = 1,
    input_history_encoding: str = "legacy",
    input_extra_features: str | None = None,
    compute_relations: bool = False,
) -> None:
    """Trigger torch.compile + CUDA graph capture for the shapes the UCI
    search will actually hit, so the first `go` doesn't pay compile
    latency (~3-5s per new shape).

    Walker path (n_walkers > 1) hits batches 1..G per walker (where G is
    walker_gather) plus up to N_walkers×G from coalesced wavefronts.
    Gumbel path (n_walkers == 1) hits batch=1 for root eval and batch=128
    for the single-game bucket (gumbel_c._BUCKETS). We warm the endpoints
    for the selected path — intermediate sizes fall through the same
    compiled graph once the endpoints are captured.

    Warmup is skipped silently on failure — the real ``go`` will see the
    same error and surface it there.
    """
    cb = CBoard.from_board(chess.Board())
    encoded = encode_cboard(
        cb,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
    )
    if n_walkers > 1:
  # batch=1 covers the pre-gather single-descent phase (e.g. when
  # budget runs out mid-gather). walker_gather is the per-walker
  # submit; n_walkers*walker_gather is the fully-coalesced wavefront.
        batches = sorted({1, int(walker_gather), int(n_walkers) * int(walker_gather)})
    else:
  # Gumbel path: root eval at 1, bucket flushes at 128.
        batches = [1, 128]
    rel_row = cb.compute_relations() if compute_relations else None
    for batch in batches:
        xs = np.broadcast_to(encoded, (batch, *encoded.shape)).astype(np.float32, copy=True)
        try:
            if rel_row is None:
                evaluator.evaluate_encoded(xs)
            else:
  # Warm WITH relations so compile/cudagraph captures the graph the
  # real search will replay (relations change the traced forward).
                rels = np.broadcast_to(rel_row, (batch, *rel_row.shape)).copy()
                evaluator.evaluate_encoded(xs, relations=rels)
        except Exception:
            break


def _warmup_pucv_evaluator(
    evaluator,
    *,
    gather: int,
    input_history_encoding: str = "legacy",
    input_extra_features: str | None = None,
    compute_relations: bool = False,
) -> None:
    """Warm the exact inplace-async shapes used by MultiGpuPucvPool."""
    cb = CBoard.from_board(chess.Board())
    encoded = encode_cboard(
        cb,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
    )
    rel_row = cb.compute_relations() if compute_relations else None
    batches = sorted({1, max(1, int(gather))})
    for batch in batches:
        for slot in range(min(2, int(getattr(evaluator, "n_slots", 1)))):
            try:
                buf = evaluator.get_input_buffer(batch, slot=slot)
                buf[:] = np.broadcast_to(encoded, (batch, *encoded.shape))
                if rel_row is None:
                    _pol, _wdl, event = evaluator.evaluate_inplace_async(batch, slot=slot)
                else:
                    rels = np.broadcast_to(rel_row, (batch, *rel_row.shape)).copy()
                    _pol, _wdl, event = evaluator.evaluate_inplace_async(
                        batch, slot=slot, relations=rels,
                    )
                if event is not None:
                    event.synchronize()
            except Exception:
                return


def _load_models(checkpoint: str, devices: list[str]):
    """Load one model per device. Cached at startup and reused across
    evaluator rebuilds (e.g., when the UCI ``MaxBatch`` option changes)."""
    if len(devices) > 1:
  # Parallel load — each is hundreds of MB of weight copy + CUDA init.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            return list(pool.map(
                lambda d: load_model_from_checkpoint(checkpoint, device=d),
                devices,
            ))
    return [load_model_from_checkpoint(checkpoint, device=devices[0])]


def _make_evaluator_factory(
    models, devices, coalesce, n_walkers, walker_gather,
    *, compile_mode: str | None = None,
):
    """Return a ``build(max_batch) -> evaluator`` closure. The models are
    captured once at startup; each call constructs fresh evaluator
    wrappers at the new max_batch and warms them at the shapes the
    walker count + gumbel bucket will actually hit.

    ``compile_mode`` (e.g. ``"reduce-overhead"``) wraps the model with
    ``torch.compile`` before warmup. Compile + cudagraph capture must happen
    on the same thread that later replays the graph. In UCI the search itself
    runs on worker threads, so compiled evaluators use the submitter-thread
    dispatcher even at walkers=1; eager walkers=1 can keep the direct inplace
    evaluator path.
    """
    input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
    input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
    use_relations = bool(getattr(models[0], "use_dynamic_relations", False))

    def build(max_batch: int, eval_cache_entries: int = 0):
        if compile_mode:
            import torch
            from typing import cast
            compiled_models = [
                cast("torch.nn.Module", torch.compile(m, mode=compile_mode))
                for m in models
            ]
        else:
            compiled_models = list(models)

        if len(devices) > 1:
            evaluators = [
                DirectGPUEvaluator(m, device=d, max_batch=max_batch, n_slots=2)
                for m, d in zip(compiled_models, devices)
            ]
            evaluator = MultiGPUDispatcher(evaluators)
        else:
            evaluator = DirectGPUEvaluator(
                compiled_models[0], device=devices[0],
                max_batch=max_batch, n_slots=2,
            )
  # Always wrap in ThreadSafeGPUDispatcher so the UCI `Threads`
  # option can bump walker count at runtime without a race. Lock
  # is uncontended at 1 thread — overhead is ~10ns.
            evaluator = ThreadSafeGPUDispatcher(evaluator)
  # The submitter-thread dispatcher is mandatory for compiled evaluators:
  # CUDA graph capture and replay need to stay on the same thread. For eager
  # evaluation it is only a batching optimization, so --no-coalesce can still
  # bypass it when there is no compile mode.
        if compile_mode is not None or (coalesce and n_walkers > 1):
            evaluator = BatchCoalescingDispatcher(evaluator, max_batch=max_batch)
        if eval_cache_entries > 0:
            evaluator = EncodedEvalCache(
                evaluator, max_entries=int(eval_cache_entries),
            )
        _warmup_evaluator(
            evaluator, n_walkers=n_walkers, walker_gather=walker_gather,
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
            compute_relations=use_relations,
        )
        return evaluator
    return build


def _make_multi_gpu_pucv_factory_builder(
    models,
    devices,
    *,
    compile_mode: str | None = None,
):
    """Return ``build(max_batch) -> list[factory]`` for MultiGpuPucvPool.

    Each factory compiles, constructs, and warms its evaluator on the pool
    worker thread that will replay cudagraphs during search.
    """
    input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
    input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
    use_relations = bool(getattr(models[0], "use_dynamic_relations", False))

    def build(max_batch: int, gather: int):
        effective_gather = min(max(1, int(gather)), int(max_batch))
        factories = []
        for model, device in zip(models, devices):
            def make_one(m=model, d=device):
                if compile_mode:
                    import torch
                    from typing import cast
                    compiled = cast(
                        "torch.nn.Module",
                        torch.compile(m, mode=compile_mode),
                    )
                else:
                    compiled = m
                evaluator = DirectGPUEvaluator(
                    compiled, device=d, max_batch=max_batch, n_slots=2,
                )
                _warmup_pucv_evaluator(
                    evaluator,
                    gather=effective_gather,
                    input_history_encoding=input_history_encoding,
                    input_extra_features=input_extra_features,
                    compute_relations=use_relations,
                )
                return evaluator

            factories.append(make_one)
        return factories

    return build


def _build_engine(
    *,
    evaluator,
    primary_device: str,
    chunk_sims: int,
    topk: int,
    c_scale: float = 0.025,  # UCI/high-sim tuned default (was 0.1); see --c-scale help
    c_visit: float = 50.0,
    c_puct: float = 2.5,
    fpu_reduction: float = 1.2,
    n_walkers: int,
    vloss_weight: int,
    walker_gather: int,
    pucv_vloss_mode: int,
    max_batch: int,
    vl_gather: int,
    eval_cache_entries: int,
    use_multi_gpu_pucv: bool,
    input_history_encoding: str = "legacy",
    input_extra_features: str = "v1",
    policy_encoding: str = "az_4672",
    compute_relations: bool = False,
    rebuild_evaluator=None,
    rebuild_multi_gpu_pucv_factories=None,
    options: EngineOptions | None = None,
) -> Engine:
    worker = SearchWorker(
        evaluator,
        device=primary_device,
        gumbel_cfg=GumbelConfig(
            simulations=chunk_sims,
            topk=topk,
            c_scale=c_scale,
            c_visit=c_visit,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            add_noise=False,
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
            policy_encoding=policy_encoding,
            compute_relations=compute_relations,
        ),
        chunk_sims=chunk_sims,
        n_walkers=n_walkers,
        vloss_weight=vloss_weight,
        walker_gather=walker_gather,
        pucv_vloss_mode=pucv_vloss_mode,
        eval_cache_entries=eval_cache_entries,
    )
    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_evaluator,
        rebuild_multi_gpu_pucv_factories=rebuild_multi_gpu_pucv_factories,
        options=options,
    )
    if use_multi_gpu_pucv and rebuild_multi_gpu_pucv_factories is not None:
        effective_gather = min(vl_gather, max_batch)
        factories = rebuild_multi_gpu_pucv_factories(max_batch, effective_gather)
        if len(factories) >= 2:
            worker.install_multi_gpu_pucv(
                factories, gather=effective_gather, as_factories=True,
            )
    return engine


def _pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(prog="chess-anti-engine-uci")
  # --checkpoint is required, but we accept DEEPFIN_CKPT as the default so
  # the `deepfin` console-script entry point (from pyproject.toml) can be
  # launched as a bare executable by chess GUIs that don't pass CLI args.
    p.add_argument("--checkpoint", default=os.environ.get("DEEPFIN_CKPT"),
                   help="path to trainer.pt or checkpoint dir (falls back to $DEEPFIN_CKPT)")
    p.add_argument("--device", default="auto", help="cpu|cuda|cuda:N (default: auto)")
    p.add_argument("--devices", default=None,
                   help="comma-separated device list for multi-GPU (e.g. 'cuda:0,cuda:1'). Overrides --device.")
  # Defaults from the 2026-04-21 bench sweep (bench_uci_engine.py --sweep):
  # chunk=512/topk=32/mb=1024 gave ~7.3x startpos nps vs 32/16/32. Chunk cap
  # of 512 (not the full node budget) keeps `stop` latency under ~400ms on
  # single-game CUDA searches.
    p.add_argument("--chunk-sims", type=int, default=512,
                   help="sims per start_gumbel_sims call (default: 512). Higher = fewer Python-C roundtrips, coarser stop latency.")
    p.add_argument("--topk", type=int, default=32, help="Gumbel root candidates (default: 32)")
    p.add_argument("--c-scale", type=float, default=0.025,
                   help="Gumbel value-transform scale in sigma(q): lower leans on the prior "
                        "policy, higher trusts the search Q more. Default 0.025 — tuned "
                        "2026-06-16 (was 0.1; +270 Elo). q_scale=c_scale*(c_visit+max_visit) "
                        "explodes at high sims, so 0.1 over-trusted the overconfident value head.")
    p.add_argument("--c-visit", type=float, default=50.0, help="Gumbel c_visit constant (default: 50.0)")
    p.add_argument("--c-puct", type=float, default=2.5, help="PUCT exploration constant (default: 2.5)")
    p.add_argument("--fpu-reduction", type=float, default=1.2,
                   help="first-play-urgency reduction for unvisited children (default: 1.2)")
    p.add_argument("--max-batch", type=int, default=1024,
                   help="DirectGPUEvaluator max batch (default: 1024). Must be >= expected leaf count per wavefront.")
    p.add_argument("--eval-cache-entries", type=int, default=0,
                   help="LRU entries for encoded eval caches, including single-thread PUCV (default: 0/off)")
    p.add_argument("--log-level", default="WARNING",
                   help="stderr log level (DEBUG|INFO|WARNING). DEBUG enables per-search gumbel profile with GPU-calls/avg-batch.")
  # --walkers > 1 switches from the Gumbel-chunked path to a PUCT walker
  # pool with virtual loss. Better dispatch-bound throughput, no
  # sequential-halving semantics. walkers=2 is ~10x the baseline on CUDA
  # (bench_uci_engine --walker-sweep, 2026-04-22) — we default to 2 so the
  # ship path gets the win. --walkers 1 opts into the classic Gumbel path.
    p.add_argument("--walkers", type=int, default=2,
                   help="PUCT walker threads (default: 2; 1 = classic Gumbel; >2 = noisy scaling)")
    p.add_argument("--vloss-weight", type=int, default=3,
                   help="virtual-loss weight in walker mode (default: 3, lc0 default)")
  # Per-walker leaf gather: each walker does G descents → one NN batch.
  # Default 1 = current behavior. Increase (4-8) to amplify effective
  # submit batch size without more walker threads. Matches lc0's
  # MinibatchSize semantic (our --minibatch-size UCI option controls
  # the separate Gumbel path's C state machine, not this).
    p.add_argument("--walker-gather", type=int, default=1,
                   help="per-walker leaf gather (default: 1; lc0-style amplification at 4-8)")
    p.add_argument("--vl-gather", type=int, default=512,
                   help="leaf gather for UseVL and UseMultiGpuPUCV (default: 512)")
    p.add_argument("--multi-gpu-pucv", action="store_true",
                   help="with --devices, use shared-tree per-GPU PUCV workers instead of the routing dispatcher")
    p.add_argument("--pucv-pending-mode", choices=["legacy", "virtual-mean"],
                   default="legacy",
                   help="pending accounting for batched PUCV paths (default: legacy)")
  # Coalesce concurrent walker calls into batched submits. On by default
  # when walkers > 1 since batch=1 per walker wastes GPU.
    p.add_argument("--no-coalesce", dest="coalesce", action="store_false",
                   help="disable walker-call coalescing (debug / A-B bench only)")
    p.set_defaults(coalesce=True)
  # torch.compile + cudagraph wins are large for 384-dim/9-layer at small
  # batches (selfplay sees ~3-5x). Selfplay path uses ``reduce-overhead``;
  # we mirror that here. Cache lives at --compile-cache-dir so repeated
  # UCI launches reuse Inductor's FX graph cache + Triton kernels (cold
  # compile is ~30-90s; warm is ~1s).
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="apply torch.compile to the model (default: on)")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="disable torch.compile (eager mode; useful for debugging)")
    p.set_defaults(compile=True)
  # 2026-04-28 sweep on 10-layer ckpt @ chunk=8192 mb=4096 walkers=1:
  # max-autotune = ~52k nps, reduce-overhead = ~44k nps. Cold compile is
  # ~2-3min slower under max-autotune but the FX graph + autotune cache
  # at --compile-cache-dir means second-run cost is ~seconds, not minutes.
    p.add_argument("--compile-mode", default="max-autotune",
                   help="torch.compile mode: max-autotune (default; ~52k nps benchmark) | "
                   "reduce-overhead (~44k nps; faster cold compile) | default (eager-ish)")
    p.add_argument("--compile-cache-dir",
                   default=os.environ.get(
                       "DEEPFIN_COMPILE_CACHE",
                       os.path.expanduser("~/.cache/deepfin/worker_cache"),
                   ),
                   help="shared TorchInductor/Triton cache root (env: DEEPFIN_COMPILE_CACHE; "
                   "default: ~/.cache/deepfin/worker_cache)")
    args = p.parse_args()

    if not args.checkpoint:
        p.error(
            "--checkpoint is required (or set DEEPFIN_CKPT in the environment). "
            "GUI launchers typically set DEEPFIN_CKPT in the shell profile."
        )

  # Logs must go to stderr — stdout is reserved for UCI protocol.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

  # Configure shared inductor + triton cache before any torch.compile
  # call. Mirrors selfplay (worker._configure_shared_compile_cache) so a
  # `deepfin` UCI invocation reuses the same compiled kernels as training.
    if args.compile:
        from chess_anti_engine.worker import _configure_shared_compile_cache
        from pathlib import Path
        _configure_shared_compile_cache(
            cache_dir=Path(args.compile_cache_dir).expanduser(),
        )

  # UCI assumes line-buffered I/O. When a GUI pipes stdout, Python defaults
  # to block-buffered, which swallows our responses until the buffer fills.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

  # --devices wins over --device when set (explicit multi-GPU list).
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [_pick_device(args.device)]
    use_multi_gpu_pucv = bool(args.multi_gpu_pucv and len(devices) > 1)
    startup_options = EngineOptions(
        threads=max(1, int(args.walkers)),
        leaf_gather=max(1, int(args.walker_gather)),
        use_multi_gpu_pucv=use_multi_gpu_pucv,
        pucv_pending_mode=str(args.pucv_pending_mode),
        vl_gather=max(32, int(args.vl_gather)),
        max_batch=max(64, int(args.max_batch)),
        eval_cache_entries=max(0, int(args.eval_cache_entries)),
    )

  # Background-build so `uci` can be answered before model load finishes.
  # Any command other than uci/quit blocks on `engine_ready` below, which
  # gives us correct `readyok` semantics (readyok only fires once the
  # engine truly exists) for free.
    engine_ref: list[Engine | None] = [None]
    engine_error: list[BaseException | None] = [None]
    engine_ready = threading.Event()

    def _build() -> None:
        try:
            n_walkers = max(1, int(args.walkers))
            walker_gather = max(1, int(args.walker_gather))
            models = _load_models(args.checkpoint, devices)
            input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
            input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
            policy_encoding = str(getattr(models[0], "policy_encoding", "az_4672"))
            use_dynamic_relations = bool(getattr(models[0], "use_dynamic_relations", False))
            if use_dynamic_relations:
                print(
                    "info string model uses dynamic relations; transporting "
                    "relation matrices through search",
                    flush=True,
                )
            compile_mode = str(args.compile_mode) if args.compile else None
            build_eval = _make_evaluator_factory(
                models, devices, coalesce=bool(args.coalesce),
                n_walkers=n_walkers, walker_gather=walker_gather,
                compile_mode=compile_mode,
            )
            build_pucv_factories = (
                _make_multi_gpu_pucv_factory_builder(
                    models,
                    devices,
                    compile_mode=compile_mode,
                )
                if len(devices) > 1
                else None
            )
  # Initial build: warms the evaluator too (see factory body).
            evaluator = build_eval(args.max_batch, startup_options.eval_cache_entries)
            engine_ref[0] = _build_engine(
                evaluator=evaluator, primary_device=devices[0],
                chunk_sims=args.chunk_sims, topk=args.topk,
                c_scale=args.c_scale, c_visit=args.c_visit,
                c_puct=args.c_puct, fpu_reduction=args.fpu_reduction,
                n_walkers=n_walkers, vloss_weight=int(args.vloss_weight),
                walker_gather=walker_gather,
                pucv_vloss_mode=1 if args.pucv_pending_mode == "virtual-mean" else 0,
                max_batch=startup_options.max_batch,
                vl_gather=startup_options.vl_gather,
                eval_cache_entries=startup_options.eval_cache_entries,
                use_multi_gpu_pucv=use_multi_gpu_pucv,
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
                policy_encoding=policy_encoding,
                compute_relations=use_dynamic_relations,
                rebuild_evaluator=build_eval,
                rebuild_multi_gpu_pucv_factories=build_pucv_factories,
                options=startup_options,
            )
        except BaseException as exc:  # pragma: no cover — surfaced via readyok
            engine_error[0] = exc
        finally:
            engine_ready.set()

    threading.Thread(target=_build, daemon=True, name="deepfin-build").start()

    try:
        for raw in sys.stdin:
            cmd = parse_command(raw)
            if isinstance(cmd, CmdUci):
                emit_handshake(startup_options)
                continue
            if isinstance(cmd, CmdQuit):
                break
            if not engine_ready.is_set():
                engine_ready.wait()
            if engine_error[0] is not None:
                print(f"info string engine load failed: {engine_error[0]!r}", flush=True)
                raise engine_error[0]
            engine = engine_ref[0]
            assert engine is not None
            engine.dispatch(cmd)
            if engine.quit_requested:
                break
    finally:
  # Close whatever evaluator is CURRENT at shutdown time, not a
  # one-shot snapshot — ``MaxBatch`` setoption rebuilds the evaluator
  # via ``SearchWorker.set_evaluator``, which already closes the old
  # one; ``engine.close()`` then closes the live one. Guarantees the
  # non-daemon submitter thread joins before Python's interpreter
  # shutdown starts tearing down PyTorch's CUDA context.
        eng = engine_ref[0]
        if eng is not None:
            try:
                eng.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
