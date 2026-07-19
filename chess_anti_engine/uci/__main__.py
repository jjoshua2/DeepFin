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
import io
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
    CUDAOwnerDispatcher,
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
)
from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS, GumbelConfig

from .engine import Engine, EngineOptions, _emit_info_string, _println, emit_handshake
from .model_loader import load_model_from_checkpoint
from .protocol import CmdQuit, CmdUci, parse_command
from .search import SearchWorker
import contextlib


def _warmup_evaluator(
    evaluator,
    *,
    n_walkers: int = 1,
    walker_gather: int = 1,
    n_devices: int = 1,
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

    ``n_devices``: each shape is issued this many times so a routing chain
    containing ``MultiGPUDispatcher`` warms EVERY device, not just the
    first few. Sequential warmup calls always tie on in-flight count (0),
    and the dispatcher's round-robin tiebreaker cycles ties, so N calls
    visit all N devices deterministically. A cold device would otherwise
    pay its compile/capture stall on a mid-game root eval, on the clock
    and outside the chunk clock-guard.

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
            for _ in range(max(1, int(n_devices))):
                if rel_row is None:
                    evaluator.evaluate_encoded(xs)
                else:
  # Warm WITH relations so compile/cudagraph captures the graph the
  # real search will replay (relations change the traced forward).
                    rels = np.broadcast_to(rel_row, (batch, *rel_row.shape)).copy()
                    evaluator.evaluate_encoded(xs, relations=rels)
            # Compact UCI search uses a distinct compiled forward/gather graph.
            # Warm it before readyok as well; otherwise the first timed move
            # pays max-autotune compilation on the clock even though the dense
            # endpoint above was already warm.
            if (
                rel_row is None
                and bool(getattr(evaluator, "supports_legal_bf16", False))
                and hasattr(evaluator, "evaluate_legal_bf16")
            ):
                legal_one = cb.legal_move_indices().astype(np.int32, copy=False)
                legal_counts = np.full((batch,), legal_one.size, dtype=np.int32)
                legal_flat = np.tile(legal_one, batch)
                for _ in range(max(1, int(n_devices))):
                    evaluator.evaluate_legal_bf16(xs, legal_flat, legal_counts)
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
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"

    def build(
        max_batch: int,
        eval_cache_entries: int = 0,
        *,
        primary_only: bool = False,
    ):
        """Build a primary (root / walker) evaluator stack.

        ``primary_only=True`` forces a single-device stack on ``devices[0]``
        even when multiple devices were loaded. Used when the multi-GPU PUCV
        pool owns search concurrency — root eval only needs one device, and
        building MultiGPUDispatcher *plus* pool factories was ≈2× per-device
        evaluator/buffer/cudagraph memory after auto-enable made the pool
        the multi-device default.
        """
        active_models = models
        active_devices = devices
        if primary_only and len(devices) > 1:
            active_models = models[:1]
            active_devices = devices[:1]

        if compile_mode:
            import torch
            from typing import cast
            compiled_models = [
                cast("torch.nn.Module", torch.compile(m, mode=compile_mode))
                for m in active_models
            ]
        else:
            compiled_models = list(active_models)

        if len(active_devices) > 1:
            evaluators = [
                DirectGPUEvaluator(
                    m, device=d, max_batch=max_batch, n_slots=2,
                    input_bf16=compact_bf16, legal_bf16=compact_bf16,
                )
                for m, d in zip(compiled_models, active_devices)
            ]
            evaluator = MultiGPUDispatcher(evaluators)
        else:
            evaluator = DirectGPUEvaluator(
                compiled_models[0], device=active_devices[0],
                max_batch=max_batch, n_slots=2, input_bf16=compact_bf16, legal_bf16=compact_bf16,
            )
  # Always wrap in ThreadSafeGPUDispatcher so the UCI `Threads`
  # option can bump walker count at runtime without a race. Lock
  # is uncontended at 1 thread — overhead is ~10ns.
            evaluator = ThreadSafeGPUDispatcher(evaluator)
  # A persistent owner thread is mandatory for compiled evaluators: CUDA graph
  # capture and replay need to stay on the same thread. Single-walker search
  # forwards the direct evaluator's pinned in-place API; concurrent walkers use
  # the heavier coalescer so their requests can still become one GPU batch.
  # For eager evaluation coalescing is only a batching optimization, so
  # --no-coalesce can bypass it.
        if compile_mode is not None or (coalesce and n_walkers > 1):
            if compile_mode is not None and n_walkers == 1:
                evaluator = CUDAOwnerDispatcher(evaluator)
            else:
                evaluator = BatchCoalescingDispatcher(evaluator, max_batch=max_batch)
  # Warm BELOW the eval cache: EncodedEvalCache dedupes identical rows and
  # serves full-hit batches without touching the inner evaluator, so warming
  # through it would compile/capture only the first call's shape on one
  # device and leave every other shape/device cold. Warming the pre-cache
  # chain still runs on the persistent owner/submitter thread (the thread that
  # replays cudagraphs in real searches).
        warm_target = evaluator
        if eval_cache_entries > 0:
            evaluator = EncodedEvalCache(
                evaluator, max_entries=int(eval_cache_entries),
            )
        _warmup_evaluator(
            warm_target, n_walkers=n_walkers, walker_gather=walker_gather,
            n_devices=len(active_devices),
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
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"

    def build(max_batch: int, gather: int):
        effective_gather = min(max(1, int(gather)), int(max_batch))
        factories = []
        for model, device in zip(models, devices):
            def make_one(m=model, d=device):
  # Bind this pool worker thread to its device BEFORE compile /
  # construction / warmup: threads inherit current-device cuda:0, and
  # cudagraph capture on devices 1..N-1 must not rely on stream
  # contexts alone (same pattern as inference_threaded /
  # train.async_eval; project memory records a cudagraph-TLS crash
  # from exactly this). cpu devices (tests) skip it.
                import torch
                dev = torch.device(d)
                if dev.type == "cuda":
  # bare "cuda" yields index None; `or 0` collapses None/0 to the default device.
                    torch.cuda.set_device(dev.index or 0)
                if compile_mode:
                    from typing import cast
                    compiled = cast(
                        "torch.nn.Module",
                        torch.compile(m, mode=compile_mode),
                    )
                else:
                    compiled = m
                evaluator = DirectGPUEvaluator(
                    compiled, device=d, max_batch=max_batch, n_slots=2,
                    input_bf16=compact_bf16, legal_bf16=compact_bf16,
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
  # Gumbel root/descent split (legacy defaults; see main()'s --c-visit-root etc.)
    c_visit_root: float = -1.0,
    q_visit_floor: float = -1.0,
    q_visit_exp: float = 1.0,
    q_global_scale: bool = False,
  # Root-ONLY LOG value-transform (UCI default; descent keeps linear q_visit_exp).
  # Linear root q_scale=c_scale*(c_visit_root+max_visit) EXPLODES with max_visit
  # (~25000 at 1M nodes) -> the root saturates sigma(q) and blindly trusts the
  # overconfident value head, so deeper search REVERSES (audit: +3.8cp at 32k).
  # Log q_scale=c_scale_root*log1p(c_visit_root+max_visit) stays ~100 from 256 to
  # millions of nodes -> sim-invariant scaling (the TCEC regret curve keeps
  # dropping). Verified no-op/improvement 256->16k, fixes the 32k reversal.
    c_scale_root: float = 7.0,
    q_visit_exp_root: float = -1.0,
    n_walkers: int,
    vloss_weight: int,
    walker_gather: int,
    pucv_vloss_mode: int,
    max_batch: int,
    vl_gather: int,
    eval_cache_entries: int,
    use_multi_gpu_pucv: bool,
    use_root_parallel_gumbel: bool = False,
    devices: tuple[str, ...] | None = None,
    input_history_encoding: str = "legacy",
    input_extra_features: str = "v1",
    policy_encoding: str = "lc0_1858",
    compute_relations: bool = False,
    rebuild_evaluator=None,
    rebuild_multi_gpu_pucv_factories=None,
    options: EngineOptions | None = None,
    gil_profile: bool = False,
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
            c_visit_root=c_visit_root,
            q_visit_floor=q_visit_floor,
            q_visit_exp=q_visit_exp,
            q_global_scale=q_global_scale,
            c_scale_root=c_scale_root,
            q_visit_exp_root=q_visit_exp_root,
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
        search_devices=devices,
        options=options,
        gil_profile=gil_profile,
    )
  # The two multi-GPU modes are mutually exclusive; the gumbel branch reuses
  # the same per-device evaluator factories as the pucv pool (identical
  # evaluator interface + warmup; only the search orchestration differs).
    if use_root_parallel_gumbel and rebuild_multi_gpu_pucv_factories is not None:
        effective_gather = min(vl_gather, max_batch)
        factories = rebuild_multi_gpu_pucv_factories(max_batch, effective_gather)
        if len(factories) >= 2:
            worker.install_root_parallel_gumbel(
                factories, gather=effective_gather, as_factories=True,
                devices=list(devices) if devices else None,
                info_string_cb=_emit_info_string,
            )
    elif use_multi_gpu_pucv and rebuild_multi_gpu_pucv_factories is not None:
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


def _resolve_use_multi_gpu_pucv(flag: bool | None, n_devices: int) -> bool:
    """Decide whether the multi-GPU PUCV pool is active at startup.

    With one device the pool cannot install (needs >= 2 factories) — always
    False, keeping single-device behavior identical. With multiple devices
    the pool is the only path that runs the GPUs concurrently: the routing
    chain (BatchCoalescingDispatcher over MultiGPUDispatcher) has ONE
    submitter thread that calls the inner evaluator synchronously, so
    exactly one GPU is busy at a time — worse than a single device once
    cache behavior is counted. Hence:

    - ``flag`` unset (None): AUTO-ENABLE the pool and say so on stderr.
    - ``flag`` explicitly False: honor it, but warn LOUDLY — the serialized
      routing path is a debugging configuration, not a performance one.
    """
    if n_devices <= 1:
        return False
    if flag is None:
        print(
            f"info: {n_devices} devices and UseMultiGpuPUCV not set — "
            "auto-enabling the multi-GPU PUCV pool (without it the routing "
            "dispatcher serializes GPU work; pass --no-multi-gpu-pucv to "
            "force that debug path)",
            file=sys.stderr, flush=True,
        )
        return True
    if not flag:
        print(
            "WARNING: --no-multi-gpu-pucv with multiple devices — the "
            "routing dispatcher's single submitter runs the GPUs ONE CALL "
            "AT A TIME (serialized, at or below single-GPU throughput). "
            "This configuration is for debugging only.",
            file=sys.stderr, flush=True,
        )
        return False
    return True


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
    p.add_argument("--chunk-sims", type=int, default=2048,
                   help="sims per start_gumbel_sims call (default: 2048). Higher = fuller GPU batches "
                        "(more NPS) at the cost of coarser stop/abort latency; 2048 favours throughput "
                        "for long time controls while leaving ample chunks per move for the abort.")
    p.add_argument("--topk", type=int, default=PLAY_SEARCH_DEFAULTS["topk"], help="Gumbel root candidates (default: 32)")
    p.add_argument("--c-scale", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale"],
                   help="Gumbel value-transform scale in sigma(q): lower leans on the prior "
                        "policy, higher trusts the search Q more. Default 0.025 — tuned "
                        "2026-06-16 (was 0.1; +270 Elo). q_scale=c_scale*(c_visit+max_visit) "
                        "explodes at high sims, so 0.1 over-trusted the overconfident value head.")
    p.add_argument("--c-visit", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit"], help="Gumbel c_visit constant (default: 50.0)")
    p.add_argument("--c-puct", type=float, default=PLAY_SEARCH_DEFAULTS["c_puct"], help="PUCT exploration constant (default: 2.5)")
    p.add_argument("--fpu-reduction", type=float, default=PLAY_SEARCH_DEFAULTS["fpu_reduction"],
                   help="first-play-urgency reduction for unvisited children (default: 1.2)")
  # Gumbel root/descent split params (merged dormant in #68; C path only —
  # run_gumbel_root_many_c). All default to legacy: a GumbelConfig built with
  # the defaults below reproduces the pre-#68 single-floor/linear behavior.
    p.add_argument("--c-visit-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit_root"],
                   help="Gumbel root-halving c_visit override (value-transform floor at the "
                        "root sequential-halving site only; descent keeps --c-visit). A large "
                        "root floor (900, default) makes one c_scale fit every sim count's root "
                        "q_scale — the root/descent SPLIT that fixes search scaling (avoidable "
                        "regret +33%% vs legacy on the audit set; value scales coherently with "
                        "depth). <0 = use --c-visit at both sites (legacy, no split).")
    p.add_argument("--q-visit-floor", type=float, default=-1.0,
                   help="Gumbel decoupled (additive) value-transform floor: when >=0, "
                        "q_scale=q_visit_floor+c_scale*max_visit^exp instead of the legacy "
                        "c_scale*(c_visit+max_visit^exp) whose floor scales WITH c_scale. "
                        "Lets c_scale rise (for sublinear --q-visit-exp) without inflating the "
                        "early-round floor. <0 (default) = legacy coupled floor.")
    p.add_argument("--q-visit-exp", type=float, default=1.0,
                   help="Gumbel exponent on max_visit in q_scale=c_scale*(c_visit+max_visit^exp). "
                        "1.0 = standard linear Gumbel (default). <1 makes search-Q trust grow "
                        "sublinearly with depth so the optimal c_scale is less sim-count-dependent.")
    p.add_argument("--c-scale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                   help="ROOT-ONLY c_scale (descent keeps --c-scale). Pairs with "
                        "--q-visit-exp-root<0 for a LOG root q_scale=c_scale_root*log1p(c_visit_root"
                        "+max_visit), which needs a large c_scale (~7) vs the tiny descent (~0.025). "
                        "<0 = use --c-scale at the root too (legacy linear).")
    p.add_argument("--q-visit-exp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                   help="ROOT-ONLY value-transform exponent (descent keeps --q-visit-exp). DEFAULT "
                        "-1 = LOG slow-growth: linear root q_scale explodes ~25000 at 1M nodes and "
                        "saturates sigma(q) (deeper search reverses, +3.8cp@32k audit); log stays "
                        "~100 from 256 to millions of nodes (sim-invariant). >=90 = use --q-visit-exp.")
    p.add_argument("--q-global-scale", action="store_true",
                   help="Gumbel descent scales the value-transform by the ROOT max child-visit "
                        "(global/search-size) instead of the local node's max_visit, so q_scale is "
                        "uniform across the tree. Pairs with --q-visit-exp<1 for sim-invariance. "
                        "Default off = legacy local behavior.")
    p.add_argument("--max-batch", type=int, default=1024,
                   help="DirectGPUEvaluator max batch (default: 1024). Must be >= expected leaf count per wavefront.")
    p.add_argument("--eval-cache-entries", type=int, default=0,
                   help="LRU entries for encoded eval caches, including single-thread PUCV (default: 0/off)")
    p.add_argument("--abort-factor", type=float, default=1.0,
                   help="visit-margin early-abort aggressiveness (Lc0 smart-pruning factor): "
                        "1.0 = ~provable bound (default); < 1.0 aborts earlier to bank time")
    p.add_argument("--time-budget-scale", type=float, default=1.0,
                   help="per-move time-budget multiplier (default: 1.0). > 1.0 schedules more "
                        "time per move, relying on --abort-factor to bank it back on easy moves; "
                        "still capped at 50%% of remaining time so it cannot flag.")
    p.add_argument("--optimum-fraction", type=float, default=0.7,
                   help="soft-target fraction of the clock budget at which a settled move banks "
                        "the rest (default: 0.7). 0 turns the visit-margin abort fully off "
                        "(spend the whole deadline — the pre-time-management baseline); clamped to (0, 1].")
    p.add_argument("--moves-horizon", type=int, default=50,
                   help="upper cap on the pieces-driven moves estimate (default: 50; the material "
                        "estimate tops out near 30 so this rarely binds). Time allocation is driven by "
                        "pieces on the board, not move number; ignored when the GUI sends movestogo.")
    p.add_argument("--log-level", default="WARNING",
                   help="stderr log level (DEBUG|INFO|WARNING). DEBUG enables per-search gumbel profile with GPU-calls/avg-batch.")
    p.add_argument(
        "--gil-profile", action="store_true",
        help="emit per-search delayed-GIL-reacquisition telemetry as UCI info strings",
    )
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
  # Tri-state: None = unset (auto-enable with >1 devices), True/False =
  # explicit. See _resolve_use_multi_gpu_pucv for why unset defaults ON.
    p.add_argument("--multi-gpu-pucv", dest="multi_gpu_pucv",
                   action="store_true", default=None,
                   help="with --devices, use shared-tree per-GPU PUCV workers instead of the "
                        "routing dispatcher (default: auto-enabled when --devices lists more "
                        "than one device)")
    p.add_argument("--no-multi-gpu-pucv", dest="multi_gpu_pucv",
                   action="store_false",
                   help="force the routing dispatcher even with multiple --devices "
                        "(DEBUG ONLY: its single submitter serializes GPU work)")
    p.add_argument("--search-parallel", choices=["pucv", "gumbel"], default="pucv",
                   help="multi-GPU search mode (requires --devices with >= 2 entries): "
                        "pucv = shared-tree PUCT+virtual-loss pool (default, throughput "
                        "baseline); gumbel = root-parallel Gumbel over evaluator groups "
                        "(quality-preserving — classic sequential-halving root decisions, "
                        "per-candidate subtrees parallelized across devices). With a single "
                        "device both values leave the classic search paths untouched.")
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
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

  # --devices wins over --device when set (explicit multi-GPU list).
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [_pick_device(args.device)]
  # gumbel mode requires >= 2 devices; with one device the new module is
  # never constructed and dispatch falls through to the classic paths.
    use_root_parallel_gumbel = bool(
        args.search_parallel == "gumbel" and len(devices) > 1,
    )
  # The two multi-GPU modes are mutually exclusive; gumbel wins when selected.
  # Otherwise keep #138's tri-state auto-enable for multi-GPU PUCV.
    if use_root_parallel_gumbel:
        use_multi_gpu_pucv = False
    else:
        use_multi_gpu_pucv = _resolve_use_multi_gpu_pucv(
            args.multi_gpu_pucv, len(devices),
        )
    startup_options = EngineOptions(
        threads=max(1, int(args.walkers)),
        leaf_gather=max(1, int(args.walker_gather)),
        use_multi_gpu_pucv=use_multi_gpu_pucv,
        pucv_pending_mode=str(args.pucv_pending_mode),
        search_parallel="gumbel" if use_root_parallel_gumbel else "pucv",
        vl_gather=max(32, int(args.vl_gather)),
        max_batch=max(64, int(args.max_batch)),
        eval_cache_entries=max(0, int(args.eval_cache_entries)),
  # Clamp the time knobs (like the spin options above): a non-positive
  # time_budget_scale would collapse every move to the 20ms floor, and a
  # negative abort_factor would abort the instant the move leads. 0.0 is the
  # maximally-aggressive-but-valid abort_factor (stop on any positive lead).
        abort_factor=max(0.0, float(args.abort_factor)),
        time_budget_scale=max(0.1, float(args.time_budget_scale)),
  # <= 0 is the OFF sentinel (preserved through the clamp); positive values are
  # clamped to (0, 1] inside limits_from_go so an out-of-range soft target can't
  # exceed the hard deadline.
        optimum_fraction=min(1.0, float(args.optimum_fraction)),
        moves_horizon=max(1, int(args.moves_horizon)),
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
  # c_scale / c_visit shape the Gumbel sigma(q) value transform; the PUCT
  # walker pool (n_walkers > 1, the --walkers/Threads default) selects on
  # raw Q and never reads them, so the tuned c_scale=0.025 win only lands on
  # the classic Gumbel path. Warn so the inert tuning isn't silent. The
  # multi-GPU pucv pool is plain PUCT too and overrides both other paths,
  # so it gets the same warning.
            if use_multi_gpu_pucv:
                _println(
                    "info string c-scale/c-visit are Gumbel-only and ignored "
                    "while the multi-GPU PUCV pool is active (plain PUCT); "
                    "run a single device with Threads/--walkers 1 for the "
                    "c_scale-tuned classic Gumbel path"
                )
            elif n_walkers > 1:
                _println(
                    "info string c-scale/c-visit are Gumbel-only and ignored at "
                    f"walkers={n_walkers} (PUCT walker pool); set Threads/--walkers 1 "
                    "for the c_scale-tuned classic Gumbel path"
                )
            models = _load_models(args.checkpoint, devices)
            input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
            input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
            policy_encoding = str(getattr(models[0], "policy_encoding", "lc0_1858"))
            use_dynamic_relations = bool(getattr(models[0], "use_dynamic_relations", False))
            if use_dynamic_relations:
                _println(
                    "info string model uses dynamic relations; transporting "
                    "relation matrices through search"
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
  # Primary stack: when the multi-GPU PUCV pool owns search, only device 0
  # is needed for root eval (and for the walker fallback after opt-out until
  # rebuild expands it). Avoid building MultiGPUDispatcher + pool factories
  # (≈2× per-device memory). rebuild_evaluator closes over startup_options so
  # MaxBatch / UseMultiGpuPUCV toggles keep the sizing correct.
            def rebuild_evaluator(
                max_batch: int, eval_cache_entries: int = 0,
            ):
                return build_eval(
                    max_batch,
                    eval_cache_entries,
                    primary_only=bool(startup_options.use_multi_gpu_pucv),
                )

  # Initial build: warms the evaluator too (see factory body).
            evaluator = rebuild_evaluator(
                args.max_batch, startup_options.eval_cache_entries,
            )
            engine_ref[0] = _build_engine(
                evaluator=evaluator, primary_device=devices[0],
                chunk_sims=args.chunk_sims, topk=args.topk,
                c_scale=args.c_scale, c_visit=args.c_visit,
                c_puct=args.c_puct, fpu_reduction=args.fpu_reduction,
                c_visit_root=args.c_visit_root, q_visit_floor=args.q_visit_floor,
                q_visit_exp=args.q_visit_exp, q_global_scale=bool(args.q_global_scale),
                c_scale_root=args.c_scale_root, q_visit_exp_root=args.q_visit_exp_root,
                n_walkers=n_walkers, vloss_weight=int(args.vloss_weight),
                walker_gather=walker_gather,
                pucv_vloss_mode=1 if args.pucv_pending_mode == "virtual-mean" else 0,
                max_batch=startup_options.max_batch,
                vl_gather=startup_options.vl_gather,
                eval_cache_entries=startup_options.eval_cache_entries,
                use_multi_gpu_pucv=use_multi_gpu_pucv,
                use_root_parallel_gumbel=use_root_parallel_gumbel,
                devices=tuple(devices),
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
                policy_encoding=policy_encoding,
                compute_relations=use_dynamic_relations,
                rebuild_evaluator=rebuild_evaluator,
                rebuild_multi_gpu_pucv_factories=build_pucv_factories,
                options=startup_options,
                gil_profile=bool(args.gil_profile),
            )
  # Warm the SEARCH path (not just the evaluator) before reporting ready, so
  # the first real `go` doesn't pay cudagraph capture mid-move and forfeit on
  # the clock. readyok gates on engine_ready below, so a GUI that waits for
  # readyok (the standard handshake) never starts the clock until this returns.
            engine_ref[0].warmup_search()
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
                _println(f"info string engine load failed: {engine_error[0]!r}")
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
            with contextlib.suppress(Exception):
                eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
